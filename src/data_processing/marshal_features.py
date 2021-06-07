from math import sqrt

import numpy as np
import pandas as pd
import talib
import talib.abstract as ta
from pandas import DataFrame, concat

from data_processing.candle_rankings import candle_rankings_2

# NATURAL_VOLUME_BREAKS = [0.0, 125.416263, 298.632517, 608.76506, 1197.486795, 2445.284399, 8277.1723]
NATURAL_VOLUME_BREAKS = [0.0, 604.22808, 1263.66949, 2312.48193, 4048.12917, 7249.01953, 35632.59882]
# NATURAL_VOLUME_BREAKS = [0.0, 69.912673, 148.257113, 252.325899, 402.375676, 621.793142, 946.410967, 1432.262112,
#                          2189.231502, 3517.888325, 8277.1723]


def marshal_candle_metadata(df: DataFrame, drop_date_column=False) -> DataFrame:
    # This should be first as all subsequent feature engineering should be based on the round number
    # df = df.round(2)

    df['log_return_close'] = np.log(df['close']).diff()

    # These features are easier to manipulate with an integer index, so we add them before setting the time-series index
    df = add_technical_indicator_features(df)
    # Some of the indicators have a warm up period where the first n values are NaN. These need to be removed. The
    # longest warm up period for the given indicators is 33
    df = df.iloc[33:]

    # Index by datetime
    df = df.set_index('date', drop=drop_date_column)

    # Then remove UTC timezone since GluonTS does not work with it
    df.index = df.index.tz_localize(None)

    return df


def add_technical_indicator_features(df: DataFrame) -> DataFrame:
    # NOTE: Appending to Dataframes is slow, thus these should all return separate dataframes to then append together
    # only one time
    dfs = [
        df,
        get_momentum_indicators(df),
        get_overlap_studies(df),
        get_pattern_recognition(df),
        get_volume_bin(df)
    ]

    return concat(dfs, axis=1)


def get_momentum_indicators(df: DataFrame) -> DataFrame:
    momentum_indicator_data = {}

    _transpose_and_add(momentum_indicator_data, "mfi", ta.MFI(df))
    _transpose_and_add(momentum_indicator_data, "roc", ta.ROC(df))
    _transpose_and_add(momentum_indicator_data, "adx", ta.ADX(df))
    _transpose_and_add(momentum_indicator_data, "rsi", ta.RSI(df))

    stoch = ta.STOCH(df)
    _transpose_and_add(momentum_indicator_data, "slowd", stoch['slowd'])
    _transpose_and_add(momentum_indicator_data, "slowk", stoch['slowk'])

    macd = ta.MACD(df)
    _transpose_and_add(momentum_indicator_data, "macd", macd['macd'])
    _transpose_and_add(momentum_indicator_data, "macdsignal", macd['macdsignal'])
    _transpose_and_add(momentum_indicator_data, "macdhist", macd['macdhist'])

    momentum_indicator_df = DataFrame(momentum_indicator_data)

    _verify_df_length(df, momentum_indicator_df, "momentum_indicator")
    return momentum_indicator_df


def get_overlap_studies(df: DataFrame) -> DataFrame:
    overlap_studies_data = {}

    # Bollinger Bands
    # bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(df), window=20, stds=2)
    # _transpose_and_add(overlap_studies_data, "bb_lowerband", bollinger['lower'])
    # _transpose_and_add(overlap_studies_data, "bb_middleband", bollinger['mid'])
    # _transpose_and_add(overlap_studies_data, "bb_upperband", bollinger['upper'])

    # Parabolic SAR
    # dataframe['sar'] = ta.SAR(dataframe)

    timeperiod = 5

    # TEMA - Triple Exponential Moving Average
    # _transpose_and_add(overlap_studies_data, "tema", ta.TEMA(df, timeperiod=timeperiod))
    # _transpose_and_add(overlap_studies_data, "ema", ta.SMA(df, timeperiod=timeperiod))

    # HMA = WMA(2 * WMA(n / 2) − WMA(n)), sqrt(n))
    # hma = wma(2 * wma(data, period // 2) - wma(data, period), sqrt(period))

    hma = ta.WMA(2 * ta.WMA(df, timeperiod=timeperiod // 2) - ta.WMA(df, timeperiod=timeperiod), timeperiod=int(sqrt(timeperiod)))
    _transpose_and_add(overlap_studies_data, "hma", hma)

    # Cycle Indicator
    # ------------------------------------
    # Hilbert Transform Indicator - SineWave
    # hilbert = ta.HT_SINE(dataframe)
    # dataframe['htsine'] = hilbert['sine']
    # dataframe['htleadsine'] = hilbert['leadsine']

    overlap_studies_df = DataFrame(overlap_studies_data)

    _verify_df_length(df, overlap_studies_df, "overlap_studies")
    return overlap_studies_df


def get_pattern_recognition(df: DataFrame) -> DataFrame:
    pattern_names = talib.get_function_groups()['Pattern Recognition']
    pattern_data = [None] * len(pattern_names)

    for idx, pattern in enumerate(pattern_names):
        # below is same as;
        # talib.CDL3LINESTRIKE(op, hi, lo, cl)
        pattern_data[idx] = getattr(talib, pattern)(df["open"], df["high"], df["low"], df["close"])

    pattern_data = np.array(pattern_data).T.tolist()
    pattern_df = _get_pattern_sparse2dense(pattern_data, pattern_names)
    # pattern_df = self._get_pattern_one_hot(pattern_data, pattern_names)

    _verify_df_length(df, pattern_df, "pattern_recognition")
    return pattern_df


def _get_pattern_one_hot(pattern_data: list, candle_names: list):
    patterns = []
    no_pattern_metadata = candle_rankings_2.get("NO_PATTERN")

    pattern_embedding = np.array(range(0, len(candle_rankings_2)))
    one_hot_encoding = np.zeros((pattern_embedding.size, np.max(pattern_embedding) + 1))
    one_hot_encoding[np.arange(pattern_embedding.size), pattern_embedding] = 1

    for r_index, row in enumerate(pattern_data):
        best_rank = no_pattern_metadata["rank"]
        best_pattern = one_hot_encoding[no_pattern_metadata["encoding"] - 1]

        for c_index, col in enumerate(row):
            if col != 0:
                pattern = candle_names[c_index]
                if col > 0:
                    pattern = pattern + '_Bull'
                else:
                    pattern = pattern + '_Bear'

                pattern_metadata = candle_rankings_2.get(pattern, no_pattern_metadata)
                if pattern_metadata["rank"] < best_rank:
                    best_rank = pattern_metadata["rank"]
                    best_pattern = one_hot_encoding[pattern_metadata["encoding"] - 1]

        patterns.append(best_pattern)

    pattern_df = DataFrame(patterns, columns=pattern_embedding)

    return pattern_df


def _get_pattern_sparse2dense(pattern_data: list, candle_names: list):
    patterns = []
    no_pattern_metadata = candle_rankings_2.get("NO_PATTERN")

    for row in pattern_data:
        pattern_count = 0
        best_rank = no_pattern_metadata["rank"]
        best_pattern = no_pattern_metadata["encoding"]

        for c_index, col in enumerate(row):
            if col != 0:
                pattern_count += 1
                pattern = candle_names[c_index]
                if col > 0:
                    pattern = pattern + '_Bull'
                else:
                    pattern = pattern + '_Bear'

                pattern_metadata = candle_rankings_2.get(pattern, no_pattern_metadata)
                if pattern_metadata["rank"] < best_rank:
                    best_rank = pattern_metadata["rank"]
                    best_pattern = pattern_metadata["encoding"]

        patterns.append([pattern_count, best_pattern])

    pattern_df = DataFrame(patterns, columns=["pattern_count", "pattern_detected"])

    return pattern_df


def get_volume_bin(df: DataFrame, use_natural_breaks=True) -> DataFrame:
    def natural_breaks(df, num_breaks=10):
        # This takes some time, so I ran it once and save the classes to a constant
        # import jenkspy
        # NATURAL_VOLUME_BREAKS = jenkspy.jenks_breaks(df["volume"], nb_class=num_breaks)
        # print(NATURAL_VOLUME_BREAKS)
        num_breaks = len(NATURAL_VOLUME_BREAKS) - 1

        volume_bin_data = pd.cut(
            df['volume'],
            bins=NATURAL_VOLUME_BREAKS,
            labels=[x for x in range(num_breaks)],
            include_lowest=True
        )
        return volume_bin_data

    def quantile_breaks(df, num_breaks=10):
        volume_bin_data, breaks = pd.qcut(
            df['volume'],
            q=num_breaks,
            labels=[str(f"bucket_{x}") for x in range(num_breaks)],
            retbins=True
        )
        return volume_bin_data

    volume_bin_df = natural_breaks(df) if use_natural_breaks else quantile_breaks(df)
    volume_bin_df = volume_bin_df.rename("volume_bin").to_frame()

    _verify_df_length(df, volume_bin_df, "volume_bin")
    return volume_bin_df


def _transpose_and_add(data_map: dict, key: str, df: DataFrame):
    data_map[key] = np.array(df).T.tolist()


def _verify_df_length(original_df: DataFrame, feature_df: DataFrame, feature_name: str):
    print(f"Number of {feature_name}: {len(feature_df.columns)}")

    assert len(original_df) == len(feature_df), \
        f"The original dataframe and the {feature_name} dataframe are different lengths. original_df " \
        f"[{len(original_df)}] != feature_df [{len(feature_df)}]. They cannot be combined"

    # At this point we know the dataframes are the same length. Thus we can now overwrite the integer based index of
    # the feature dataframe with the index of the original df, ensuring they line up 1-to-1
    feature_df.set_index(original_df.index, inplace=True)
