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
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib

import data_processing.marshal_features as mf
import requests

# This class is a sample. Feel free to customize it.
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

    # Optimal ticker interval for the strategy.
    timeframe = '5m'

    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = False

    # These values can be overridden in the "ask_strategy" section in the config.
    use_sell_signal = True
    sell_profit_only = False
    ignore_roi_if_buy_signal = False

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 30

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

        # Essentially replicate what mf.marshal_candles is doing. This is so we have all the same values for inference
        # as the ones used for training
        # dataframe = dataframe.round(2)
        # dataframe = mf.add_technical_indicator_features(dataframe)
        dataframe = mf.marshal_candles(dataframe)

        print(f"Length of dataframe [{len(dataframe)}]")

        # predictor_url = "http://localhost:8080/endpoints/giia/invocation"
        # payload = dataframe.to_json()
        # # payload = json.dumps(dataframe)
        #
        # response = requests.post(predictor_url, data=payload)
        # print(response.status_code, response.reason)

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the buy signal for the given dataframe
        :param dataframe: DataFrame populated with indicators
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with buy column
        """
        # dataframe.loc[
        #     (
        #         (qtpylib.crossed_above(dataframe['rsi'], 30)) &  # Signal: RSI crosses above 30
        #         (dataframe['tema'] <= dataframe['bb_middleband']) &  # Guard: tema below BB middle
        #         (dataframe['tema'] > dataframe['tema'].shift(1)) &  # Guard: tema is raising
        #         (dataframe['volume'] > 0)  # Make sure Volume is not 0
        #     ),
        #     'buy'] = 1

        dataframe.loc[
            (
                # (dataframe['pattern_detected'] > 0) &
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
        # dataframe.loc[
        #     (
        #         (qtpylib.crossed_above(dataframe['rsi'], 70)) &  # Signal: RSI crosses above 70
        #         (dataframe['tema'] > dataframe['bb_middleband']) &  # Guard: tema above BB middle
        #         (dataframe['tema'] < dataframe['tema'].shift(1)) &  # Guard: tema is falling
        #         (dataframe['volume'] > 0)  # Make sure Volume is not 0
        #     ),
        #     'sell'] = 1

        dataframe.loc[
            (
                    # (dataframe['pattern_detected'] <= 0) &
                    (dataframe['volume'] > 0)  # Make sure Volume is not 0
            ),
            'sell'] = 1

        return dataframe
