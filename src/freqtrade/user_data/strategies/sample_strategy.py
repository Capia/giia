# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# isort: skip_file
# --- Do not remove these libs ---
import json

import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame

from freqtrade.strategy.interface import IStrategy

# --------------------------------
# Add your lib to import here
import requests

import data_processing.marshal_features as mf
from utils import config


class SampleStrategy(IStrategy):
    """
    This is a sample strategy to inspire you.
    More information in https://www.freqtrade.io/en/latest/strategy-customization/

    You can:
        :return: a Dataframe with all mandatory indicators for the strategies
    - Rename the class name (Do not forget to update class_name)
    - Add any methods you want to build your strategy
    - Add any lib you need to build your strategy

    You must keep:
    - the lib in the section "Do not remove these libs"
    - the prototype for the methods: minimal_roi, stoploss, populate_indicators, populate_buy_trend,
    populate_sell_trend, hyperopt_space, buy_strategy_generator
    """
    # Strategy interface version - allow new iterations of the strategy interface.
    # Check the documentation or the Sample strategy to get the latest version.
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
        'stoploss_on_exchange': False
    }

    # Optional order time in force.
    order_time_in_force = {
        'buy': 'gtc',
        'sell': 'gtc'
    }

    plot_config = {
        'main_plot': {
            'tema': {},
            'sar': {'color': 'white'},
        },
        'subplots': {
            "MACD": {
                'macd': {'color': 'blue'},
                'macdsignal': {'color': 'orange'},
            },
            "RSI": {
                'rsi': {'color': 'red'},
            }
        }
    }

    def informative_pairs(self):
        """
        Define additional, informative pair/interval combinations to be cached from the exchange.
        These pair/interval combinations are non-tradeable, unless they are part
        of the whitelist as well.
        For more information, please consult the documentation
        :return: List of tuples in the format (pair, interval)
            Sample: return [("ETH/USDT", "5m"),
                            ("BTC/USDT", "15m"),
                            ]
        """
        return []

    def get_inference(self, df: DataFrame, is_backtest_mode: bool):
        import requests
        import json

        predictor_url = "http://localhost:8080/invocations"

        def get_predictions(raw_df):
            if isinstance(raw_df, pd.Series):
                print(f"Have a series, converting to dataframe. (THIS SHOULD ONLY OCCUR IN DRY RUN MODE)")
                rolled_df = df.loc[raw_df.index]
            else:
                rolled_df = raw_df

            payload = rolled_df.to_json(orient='split')

            response = requests.post(predictor_url, data=payload)
            if response.ok:
                prediction = json.loads(response.text)
                # print(prediction)
                mean_prediction = prediction[0]['mean']
                # print(mean_prediction[0])

                # Not exactly how pandas rolling + apply is supposed to be used, but it allows us to save multiple
                # values
                df.loc[
                    raw_df.index.max(),
                    ['mean_close_1', 'mean_close_2', 'mean_close_3', 'mean_close_4', 'mean_close_5']
                ] = mean_prediction
                return 0
            else:
                print('ERROR')
                print(response.status_code, response.reason)
                return 1

        # Only during back-test mode does the full dataframe of history is passed by freqtrade to this function,
        # while during dry/live mode only 499 (config.FREQTRADE_MAX_CONTEXT) values are passed in.
        # https://www.freqtrade.io/en/stable/strategy-customization/#anatomy-of-a-strategy
        if is_backtest_mode:
            print(f"Running in back test mode")

            # `rolling()` needs to operate on one column to prevent repeating for all columns of the dataframe. It
            # doesn't matter what column we use since we really only care of about the index (via `raw=False`)
            df['close']\
                .rolling(config.FREQTRADE_MAX_CONTEXT)\
                .apply(get_predictions, raw=False)
        else:
            get_predictions(df)

        return df

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Adds several different TA indicators to the given DataFrame

        Performance Note: For the best performance be frugal on the number of indicators
        you are using. Let uncomment only the indicator you are using in your strategies
        or your hyperopt configuration, otherwise you will waste your memory and CPU usage.
        :param dataframe: Dataframe with data from the exchange
        :param metadata: Additional information, like the currently traded pair
        :return: a Dataframe with all mandatory indicators for the strategies
        """

        is_backtest_mode = len(dataframe) > config.FREQTRADE_MAX_CONTEXT

        if not metadata.get('already_marshalled', False):
            dataframe = mf.marshal_candles(dataframe)

        # Make probabilistic predictions on the data. This will return `mean` and `quantiles`, which we will add to the
        # dataframe for the populate_buy/sell_trend functions to play off of
        # mean_predictions = []
        # print(dataframe)
        dataframe = self.get_inference(dataframe, is_backtest_mode)

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the buy signal for the given dataframe
        :param dataframe: DataFrame populated with indicators
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with buy column
        """
        # latest_close = dataframe.tail(1)['close']
        # prediction = metadata['prediction']
        # trade = False
        # for mean_predicted_close in prediction[0]['mean']:
        #     percent_difference = (abs(mean_predicted_close - latest_close) / latest_close) * 100.0
        #     if percent_difference > 4:
        #         trade = True
        #
        # if trade:
        #     dataframe.tail(1)['buy'] = 1
        dataframe.loc[
            (
                    (dataframe['mean_close_1'] > dataframe['close']) &
                    (dataframe['volume'] > 0)  # Make sure Volume is not 0
            ),
            'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the sell signal for the given dataframe
        :param dataframe: DataFrame populated with indicators
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with buy column
        """
        # latest_close = dataframe.tail(1)['close']
        # prediction = metadata['prediction']
        # trade = False
        # for mean_predicted_close in prediction[0]['mean']:
        #     percent_difference = (abs(mean_predicted_close - latest_close) / latest_close) * 100.0
        #     if percent_difference > 4:
        #         trade = True
        #
        # if trade:
        #     dataframe.tail(1)['sell'] = 1

        dataframe.loc[
            (
                (dataframe['mean_close_1'] < dataframe['close']) &
                (dataframe['volume'] > 0)  # Make sure Volume is not 0
            ),
            'sell'] = 1

        return dataframe
