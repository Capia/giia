import numpy as np
import pandas as pd
import talib
import talib.abstract as ta
from pandas import DataFrame, concat

from data_processing.candle_rankings import candle_rankings_2

NATURAL_VOLUME_BREAKS = [0.0, 125.416263, 298.632517, 608.76506, 1197.486795, 2445.284399, 8277.1723]
# NATURAL_VOLUME_BREAKS = [0.0, 69.912673, 148.257113, 252.325899, 402.375676, 621.793142, 946.410967, 1432.262112,
#                          2189.231502, 3517.888325, 8277.1723]


def marshal_candles(df: DataFrame) -> DataFrame:
    # This should be first as all subsequent feature engineering should be based on the round number
    df = df.round(2)

    # These features are easier to manipulate with an integer index, so we add them before setting the time-series index
    df = add_technical_indicator_features(df)

    # Index by datetime
    df = df.set_index('date')

    # Then remove UTC timezone since GluonTS does not work with it
    df.index = df.index.tz_localize(None)

    # Shift features down one timeframe and pad
    # df['open'] = df['open'].shift(1)
    # df['high'] = df['high'].shift(1)
    # df['low'] = df['low'].shift(1)
    # df['volume'] = df['volume'].shift(1)
    # df = df[1:]

    return df


def add_technical_indicator_features(df: DataFrame) -> DataFrame:
    # NOTE: Appending to Dataframes is slow, thus these should all return separate dataframes to then append together
    # only one time
    # momentum_indicators = get_momentum_indicators(df)
    pattern_recognition = get_pattern_recognition(df)
    volume_bin = get_volume_bin(df)

    return concat([df, pattern_recognition, volume_bin], axis=1)


def get_momentum_indicators(df: DataFrame) -> DataFrame:
    momentum_indicator_names = talib.get_function_groups()['Momentum Indicators']
    momentum_indicator_data = []
    print(f"Number of patterns: {len(momentum_indicator_names)}")

    for indicator in momentum_indicator_names:
        momentum_indicator_data.append(getattr(ta, indicator)(df))

    print(momentum_indicator_data[0])
    print(momentum_indicator_data)
    momentum_indicator_data = np.array(momentum_indicator_data).T.tolist()
    momentum_indicator_df = DataFrame(momentum_indicator_data,
                                      columns=["momentum_indicator_count", "momentum_indicator_detected"])

    _verify_df_length(df, momentum_indicator_df, "momentum_indicator")
    return momentum_indicator_df


def get_pattern_recognition(df: DataFrame) -> DataFrame:
    pattern_names = talib.get_function_groups()['Pattern Recognition']
    pattern_data = [None] * len(pattern_names)
    print(f"Number of patterns: {len(pattern_names)}")

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
    print("ONE-HOT CONVERSION COMPLETE")

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
    print("PATTERN SPARSE TO DENSE CONVERSION COMPLETE")

    return pattern_df


def get_volume_bin(df: DataFrame, use_natural_breaks=True) -> DataFrame:
    def natural_breaks(df, num_breaks=10):
        # This takes some time, so I ran it once and save the classes to a constant
        # import jenkspy
        # NATURAL_VOLUME_BREAKS = jenkspy.jenks_breaks(df["volume"], nb_class=num_breaks)
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
    volume_bin_df = volume_bin_df.rename("volume_bin")

    _verify_df_length(df, volume_bin_df, "volume_bin")
    return volume_bin_df


def _verify_df_length(original_df: DataFrame, feature_df: DataFrame, feature_name: str):
    assert len(original_df) == len(feature_df), \
        f"The original dataframe and the {feature_name} dataframe are different lengths. original_df " \
        f"[{len(original_df)}] != feature_df [{len(feature_df)}]. They cannot be combined"
