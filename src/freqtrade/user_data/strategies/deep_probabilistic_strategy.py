# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# isort: skip_file
# --- Do not remove these libs ---
import numpy as np  # noqa
import pandas as pd  # noqa
from freqtrade.strategy import IntParameter
from pandas import DataFrame

from freqtrade.strategy.interface import IStrategy

# --------------------------------
# Add your lib to import here
from tqdm import tqdm

import data_processing.marshal_features as mf
from utils import config


class DeepProbabilisticStrategy(IStrategy):
    INTERFACE_VERSION = 2

    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = False

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count = config.FREQTRADE_MAX_CONTEXT

    # Optional order type mapping.
    order_types = {
        'buy': 'limit',
        'sell': 'limit',
        'stoploss': 'market',
        'stoploss_on_exchange': True
    }

    # Optional order time in force.
    order_time_in_force = {
        'buy': 'gtc',
        'sell': 'gtc'
    }

    #   485/500:     27 trades. 20/0/7 Wins/Draws/Losses. Avg profit   1.70%. Median profit   1.84%. Total profit  451.44664289 USDT (   9.03Σ%). Avg duration 0:01:00 min. Objective: 0.84715

    # Buy hyperspace params:
    buy_params = {
        "buy_pred_close_diff_1": 3,  # value loaded from strategy
        "buy_pred_close_diff_2": 1,  # value loaded from strategy
        "buy_pred_close_diff_3": 2,  # value loaded from strategy
    }

    # Sell hyperspace params:
    sell_params = {
        "sell_pred_close_diff_1": -1,  # value loaded from strategy
        "sell_pred_close_diff_2": 0,  # value loaded from strategy
        "sell_pred_close_diff_3": -1,  # value loaded from strategy
    }

    # ROI table:
    minimal_roi = {
        "0": 0.085,
        "8": 0.019,
        "20": 0.003,
        "36": 0
    }

    # Stoploss:
    stoploss = -0.034

    # Trailing stop:
    trailing_stop = True
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.022
    trailing_only_offset_is_reached = True

    # Define the hyperopt parameter spaces, defaults are overwritten by buy_params and sell_params above
    buy_pred_close_diff_1 = IntParameter(0, 6, default=1)
    buy_pred_close_diff_2 = IntParameter(0, 6, default=1)
    buy_pred_close_diff_3 = IntParameter(0, 6, default=1)
    sell_pred_close_diff_1 = IntParameter(-5, 1, default=-4)
    sell_pred_close_diff_2 = IntParameter(-5, 1, default=-4)
    sell_pred_close_diff_3 = IntParameter(-5, 1, default=-4)

    def informative_pairs(self):
        return []

    def run_inference(self, df: DataFrame, is_backtest_mode: bool):
        """
        Make probabilistic predictions on the data. This will return `mean` and `quantiles`, which we will add to the
        dataframe for the populate_buy/sell_trend functions to play off of
        """
        import requests
        import json

        predictor_url = 'http://localhost:8080/invocations'

        def get_predictions(raw_df):
            if isinstance(raw_df, pd.Series):
                if not is_backtest_mode:
                    raise ValueError('A series was passed instead of a dataframe whilst in dry/live mode')

                rolled_df = df.loc[raw_df.index]
            else:
                rolled_df = raw_df

            payload = rolled_df.to_json(orient='split')

            response = requests.post(predictor_url, data=payload)
            if response.ok:
                prediction = json.loads(response.text)
                # print(prediction)
                mean_predictions = prediction[0]['mean']
                # print(mean_predictions[0])

                # Not exactly how pandas rolling + apply is supposed to be used, but it allows us to save multiple
                # values
                df.loc[
                    raw_df.index.max(),
                    ['pred_close_1', 'pred_close_2', 'pred_close_3', 'pred_close_4', 'pred_close_5']
                ] = mean_predictions
                return 0
            else:
                raise ValueError(f"ERROR: [{response.status_code}] from [{predictor_url}]. REASON: [{response.reason}]")

        # Only during back-test mode does the full dataframe of history is passed by freqtrade to this function,
        # while during dry/live mode only 499 (config.FREQTRADE_MAX_CONTEXT) values are passed in.
        # https://www.freqtrade.io/en/stable/strategy-customization/#anatomy-of-a-strategy
        if is_backtest_mode:
            print('Running in back test mode')
            total_num_iterations = len(df) - config.FREQTRADE_MAX_CONTEXT + 1
            print(f"Total number of iterations: [{total_num_iterations}]")

            tqdm.pandas()  # to support the `progress_apply` below

            # `rolling()` needs to operate on one column to prevent repeating for all columns of the dataframe. It
            # doesn't matter what column we use since we really only care of about the index (via `raw=False`)
            df['close'] \
                .rolling(config.FREQTRADE_MAX_CONTEXT) \
                .progress_apply(get_predictions, raw=False)
        else:
            print('Running in dry/live mode')
            get_predictions(df)

        return df

    def calc_pred_close_weighted(self, df: DataFrame) -> DataFrame:
        print("Calculating weighted close from predictions")

        # We take a weighted average of all close price predictions for the next time frame (t+1). Given current time
        # t0, we find the weighted average of the next expected close price (close of t+1) by valuing t0's t+1
        # prediction more than t-1's t+2 prediction. This continues until t-4's t+5 prediction, which is weighted the
        # least. The idea here is that there is a higher chance of deviation from current market conditions for
        # predictions further out. Thus it stands to reason they are least likely to remain accurate and while the
        # information still may be valuable, it should not be considered as strongly as more current predictions.
        weights = [1, 0.8, 0.6, 0.4, 0.2]
        # weights = [1, 0.4, 0.3, 0.2, 0.1]

        df["pred_close_weighted_1"] = \
            (
                    (
                            df['pred_close_1'] * weights[0] +
                            df['pred_close_2'].shift(1) * weights[1] +
                            df['pred_close_3'].shift(2) * weights[2] +
                            df['pred_close_4'].shift(3) * weights[3] +
                            df['pred_close_5'].shift(4) * weights[4]
                    )
                    / sum(weights)
            )

        df["pred_close_weighted_2"] = \
            (
                    (
                            df['pred_close_2'] * weights[0] +
                            df['pred_close_3'].shift(1) * weights[1] +
                            df['pred_close_4'].shift(2) * weights[2] +
                            df['pred_close_5'].shift(3) * weights[3]
                    )
                    / sum(weights[:-1])
            )

        df["pred_close_weighted_3"] = \
            (
                    (
                            df['pred_close_3'] * weights[0] +
                            df['pred_close_4'].shift(1) * weights[1] +
                            df['pred_close_5'].shift(2) * weights[2]
                    )
                    / sum(weights[:-2])
            )

        return df

    def calc_percent_diff(self, df: DataFrame) -> DataFrame:
        print("Calculating percent difference")

        df["pred_close_diff_1"] = (
                (df["pred_close_weighted_1"] - df["close"]) / df["pred_close_weighted_1"] * 100
        )
        df["pred_close_diff_2"] = (
                (df["pred_close_weighted_2"] - df["close"]) / df["pred_close_weighted_2"] * 100
        )
        df["pred_close_diff_3"] = (
                (df["pred_close_weighted_3"] - df["close"]) / df["pred_close_weighted_3"] * 100
        )

        return df

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        is_backtest_mode = len(dataframe) > config.FREQTRADE_MAX_CONTEXT

        if self.config.get('return_cached_dataframe', False):
            print(f"Returning cache dataframe from [{config.CACHED_PRED_CSV_0}]")

            cached_dataframe = pd.concat([
                pd.read_csv(filepath_or_buffer=config.CACHED_PRED_CSV_0, header=0, index_col=0,
                            parse_dates=['date.1']),
                pd.read_csv(filepath_or_buffer=config.CACHED_PRED_CSV_1, header=0, index_col=0,
                            parse_dates=['date.1'])
            ])\
                .reset_index()\
                .drop_duplicates(subset='date', keep='last')\
                .set_index('date').sort_index()

            first_index = str(dataframe['date'].iloc[0])
            last_index = str(dataframe['date'].iloc[-1])
            print(f"First index of provided df is [{first_index}], the last index is [{last_index}]")
            print(f"cached dataframe head [{cached_dataframe.head(10)}]")

            if self.config.get('truncate_cached_dataframe', True):
                print(f"Truncating cached dataframe to fit the provided dataframe's timeframe")
                cached_dataframe = cached_dataframe[first_index:last_index]

            cached_dataframe.rename(columns={'date.1': 'date'}, inplace=True)
            return cached_dataframe

        if self.config.get('marshal_candle_metadata', True):
            print("Running [marshal_candle_metadata]")
            dataframe = mf.marshal_candle_metadata(dataframe)

        if self.config.get('run_inference', True):
            print("Running [run_inference]")
            dataframe = self.run_inference(dataframe, is_backtest_mode)
            dataframe = self.calc_pred_close_weighted(dataframe)
            dataframe = self.calc_percent_diff(dataframe)

        print("Finished populating indicators")
        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if self.config.get('run_inference', True):
            print("Running [populate_buy_trend] with predictions")
            pred_close_diffs = \
                [self.buy_pred_close_diff_1.value, self.buy_pred_close_diff_2.value, self.buy_pred_close_diff_3.value]
            for idx, pred_close_diff in enumerate(pred_close_diffs):
                print(f"buy_pred_close_diff_{idx + 1} is set to [{pred_close_diff}]")

            dataframe.loc[
                (
                        (dataframe['pred_close_diff_1'] > pred_close_diffs[0]) &
                        (dataframe['pred_close_diff_2'] > pred_close_diffs[1]) &
                        (dataframe['pred_close_diff_3'] > pred_close_diffs[2]) &
                        (dataframe['volume'] > 0)  # Make sure Volume is not 0
                ),
                'buy'] = 1
        else:
            print("Running [populate_buy_trend] without predictions")
            dataframe.loc[
                (
                        (dataframe['close'] > dataframe['close'].shift()) &
                        (dataframe['volume'] > 0)  # Make sure Volume is not 0
                ),
                'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if self.config.get('run_inference', True):
            print("Running [populate_sell_trend] with predictions")
            pred_close_diffs = \
                [self.sell_pred_close_diff_1.value, self.sell_pred_close_diff_2.value, self.sell_pred_close_diff_3.value]
            for idx, pred_close_diff in enumerate(pred_close_diffs):
                print(f"sell_pred_close_diff_{idx + 1} is set to [{pred_close_diff}]")

            dataframe.loc[
                (
                        (dataframe['pred_close_diff_1'] < pred_close_diffs[0]) &
                        (dataframe['pred_close_diff_2'] < pred_close_diffs[1]) &
                        (dataframe['pred_close_diff_3'] < pred_close_diffs[2]) &
                        (dataframe['volume'] > 0)  # Make sure Volume is not 0
                ),
                'sell'] = 1
        else:
            print("Running [populate_sell_trend] without predictions")
            dataframe.loc[
                (
                        (dataframe['close'] < dataframe['close'].shift()) &
                        (dataframe['volume'] > 0)  # Make sure Volume is not 0
                ),
                'sell'] = 1

        return dataframe
