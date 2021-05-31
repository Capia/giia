# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# isort: skip_file
# --- Do not remove these libs ---
import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame

from freqtrade.strategy.interface import IStrategy

# --------------------------------
# Add your lib to import here
from tqdm import tqdm

import data_processing.marshal_features as mf
from utils import config


class DeepProbabilisticStrategy(IStrategy):
    INTERFACE_VERSION = 2

    # Minimal ROI designed for the strategy.
    # This attribute will be overridden if the config file contains "minimal_roi".
    minimal_roi = {
        "60": 0.01,
        "30": 0.02,
        "0": 0.04
    }

    # Optimal stoploss designed for the strategy.
    # This attribute will be overridden if the config file contains "stoploss".
    stoploss = -0.10

    # Trailing stoploss
    trailing_stop = False
    # trailing_only_offset_is_reached = False
    # trailing_stop_positive = 0.01
    # trailing_stop_positive_offset = 0.0  # Disabled / not configured

    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = False

    # These values can be overridden in the "ask_strategy" section in the config.
    use_sell_signal = True
    sell_profit_only = False
    ignore_roi_if_buy_signal = False

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
            tqdm.pandas()  # to support the `progress_apply` below

            # `rolling()` needs to operate on one column to prevent repeating for all columns of the dataframe. It
            # doesn't matter what column we use since we really only care of about the index (via `raw=False`)
            df['close'] \
                .rolling(config.FREQTRADE_MAX_CONTEXT) \
                .progress_apply(get_predictions, raw=False)
        else:
            print('Running in dry/live mode')
            get_predictions(df)

        df = self.calc_pred_close_weighted(df)
        df = self.calc_percent_diff(df)
        return df

    def calc_pred_close_weighted(self, df: DataFrame) -> DataFrame:
        # We take a weighted average of all close price predictions for the next time frame (t+1). Given current time
        # t0, we find the weighted average of the next expected close price (close of t+1) by valuing t0's t+1
        # prediction more than t-1's t+2 prediction. This continues until t-4's t+5 prediction, which is weighted the
        # least. The idea here is that there is a higher chance of deviation from current market conditions for
        # predictions further out. Thus it stands to reason they are least likely to remain accurate and while the
        # information still may be valuable, it should not be considered as strongly as more current predictions.
        weights = [1, 0.8, 0.6, 0.4, 0.2]

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

        return df

    def calc_percent_diff(self, df: DataFrame) -> DataFrame:
        df["pred_close_diff"] = (
                (df["close"] - df["pred_close_weighted_1"]) / df["close"] * 100
        )

        return df

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        is_backtest_mode = len(dataframe) > config.FREQTRADE_MAX_CONTEXT

        if metadata.get('marshal_candle_metadata', True):
            dataframe = mf.marshal_candle_metadata(dataframe)

        if metadata.get('run_inference', True):
            dataframe = self.run_inference(dataframe, is_backtest_mode)

        print("Finished populating indicators")
        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if metadata.get('run_inference', True):
            dataframe.loc[
                (
                        (dataframe['pred_close_diff'] > 1) &
                        (dataframe['volume'] > 0)  # Make sure Volume is not 0
                ),
                'buy'] = 1
        else:
            # A basic example of setting the buy signal without the predictions
            dataframe.loc[
                (
                        (dataframe['close'] > dataframe['close'].shift()) &
                        (dataframe['volume'] > 0)  # Make sure Volume is not 0
                ),
                'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if metadata.get('run_inference', True):
            dataframe.loc[
                (
                        (dataframe['pred_close_diff'] < -1) &
                        (dataframe['volume'] > 0)  # Make sure Volume is not 0
                ),
                'sell'] = 1
        else:
            # A basic example of setting the sell signal without the predictions
            dataframe.loc[
                (
                        (dataframe['close'] < dataframe['close'].shift()) &
                        (dataframe['volume'] > 0)  # Make sure Volume is not 0
                ),
                'sell'] = 1

        return dataframe
