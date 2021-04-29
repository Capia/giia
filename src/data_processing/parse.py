import numpy as np
import pandas as pd
from pathlib import Path

from gluonts.dataset.common import ListDataset, TrainDatasets, CategoricalFeatureInfo, MetaData
from gluonts.dataset.field_names import FieldName
from gluonts.dataset.multivariate_grouper import MultivariateGrouper
from pandas import DataFrame, concat
import talib
from itertools import compress

from data_processing.candle_rankings import candle_rankings, candle_rankings_2
from utils.logger_util import LoggerUtil
from utils import config
from freqtrade.data.history import load_pair_history
from freqtrade.configuration import Configuration


class Parse:
    logger = None

    def __init__(self, logger: LoggerUtil):
        self.logger = logger

    def split_train_test_dataset(self, dataset_dir_path: Path):
        # Copy dataset channels to their respective file
        dataset_dir_path.mkdir(parents=True, exist_ok=True)

        # First prime the user_data_dir key. This will take priority when merged with config.json
        freqtrade_config = Configuration({"user_data_dir": config.FREQTRADE_USER_DATA_DIR})
        freqtrade_config = freqtrade_config.load_from_files([str(config.FREQTRADE_USER_DATA_DIR / "config.json")])

        candles = load_pair_history(
            datadir=config.FREQTRADE_USER_DATA_DIR / "data" / "binance",
            timeframe=freqtrade_config["timeframe"],
            pair=config.CRYPTO_PAIR)

        if candles.empty:
            raise ValueError('The candle dataframe is empty. Ensure that you are loading a dataset that has been '
                             'downloaded to the configured location')

        df = self._marshal_candles(candles)

        # self.logger.log("First sample:")
        # self.logger.log(df.head(1), newline=True)
        # self.logger.log("Last sample:")
        # self.logger.log(df.tail(1), newline=True)
        self.logger.log(f"Number of columns: {len(df.columns)}")
        self.logger.log(df, newline=True)

        # Configure fractions to split dataset between training and testing (validation can be added easily)
        fractions = np.array([0.7, 0.3])

        # Split dataset between training and testing
        train_df, test_df = np.array_split(
            df, (fractions[:-1].cumsum() * len(df)).astype(int))

        # Save df to file to make it easy to visualize/debug. Test df is used as it is smaller and more portable
        test_df.to_csv("test.csv")

        train_dataset = self.df_to_multi_feature_dataset(train_df)
        test_dataset = self.df_to_multi_feature_dataset(test_df)

        datasets = self.build_train_datasets(train_df, train_dataset, test_df, test_dataset)

        datasets.save(str(dataset_dir_path))
        self.logger.log(f"Parsed train and test datasets can be found in [{dataset_dir_path}]", 'debug')

    def _marshal_candles(self, candles: DataFrame) -> DataFrame:
        # These features are easier to manipulate with an integer index, so we run this first
        # df = self._add_indicators(candles)
        df = self._add_pattern_recognition(candles)
        df = self._bin_volume(df)

        # Index by datetime
        df = df.set_index('date')

        # Then remove UTC timezone since GluonTS does not work with it
        df.index = df.index.tz_localize(None)

        # Shift features down one timeframe and pad. This will make the model predict the next target value based
        # on candles from a time frame's previous time frame.
        # df['open'] = df['open'].shift(1)
        # df['high'] = df['high'].shift(1)
        # df['low'] = df['low'].shift(1)
        # df['volume'] = df['volume'].shift(1)
        # df = df[1:]

        return df

    def df_to_covariate_dataset(self, df):
        return ListDataset(
            [
                {
                    FieldName.START: df.index[0],
                    FieldName.TARGET: df[column_name][:].values,
                    FieldName.ITEM_ID: column_name,
                    FieldName.FEAT_STATIC_CAT: np.array([idx]),
                } for idx, column_name in enumerate(df.columns)
            ],
            freq=config.DATASET_FREQ
        )

    def df_to_multi_feature_dataset(self, df):
        # dynamic_cat_features = ["candlestick_pattern"]
        # dynamic_real_features = ["open", "high", "low", "volume", "candlestick_pattern"]
        dynamic_real_features_blacklist = ["close", "volume"]

        return ListDataset(
            [
                {
                    FieldName.START: df.index[0],
                    FieldName.TARGET: df["close"][:].values,
                    FieldName.ITEM_ID: "close",
                    FieldName.FEAT_DYNAMIC_REAL: [
                        df[column_name][:].values for column_name in df.columns if
                        column_name not in dynamic_real_features_blacklist
                    ],
                    # FieldName.FEAT_DYNAMIC_REAL: [
                    #     df[column_name][:].values for column_name in df.columns if column_name in dynamic_real_features
                    # ],
                    # FieldName.FEAT_DYNAMIC_CAT: [
                    #     df[column_name][:].values for column_name in df.columns if column_name in dynamic_cat_features
                    # ],
                }
            ],
            freq=config.DATASET_FREQ
        )

    def df_to_multivariate_target_dataset(self, df):
        return ListDataset(
            [
                {
                    FieldName.START: df.index[0],
                    FieldName.TARGET: [df[column_name][:].values for column_name in df.columns],
                }
            ],
            freq=config.DATASET_FREQ,
            one_dim_target=False
        )

    def build_train_datasets(self, train_df, train_dataset, test_df, test_dataset):
        return TrainDatasets(
            metadata=MetaData(
                freq=config.DATASET_FREQ,
                # target={'name': 'close'},
                feat_static_cat=[
                    CategoricalFeatureInfo(name="num_series", cardinality=len(train_df.columns)),

                    # Not features actually used by the network. Just storing the metadata so it doesn't have to
                    # be calculated later with an iterator
                    CategoricalFeatureInfo(name="ts_train_length", cardinality=len(train_df)),
                    CategoricalFeatureInfo(name="ts_test_length", cardinality=len(test_df)),
                ],

                # Purposely leave out prediction_length as it will couple the hyper parameter to the dataset
            ),
            train=train_dataset,
            test=test_dataset
        )

    def _add_indicators(self, df):
        return concat([df], axis=1)

    def _add_pattern_recognition(self, df):
        pattern_names = talib.get_function_groups()['Pattern Recognition']
        pattern_data = []
        self.logger.log(f"Number of patterns: {len(pattern_names)}")

        # create columns for each pattern
        for candle_name in pattern_names:
            # below is same as;
            # talib.CDL3LINESTRIKE(op, hi, lo, cl)
            pattern_data.append(getattr(talib, candle_name)(df["open"], df["high"], df["low"], df["close"]))

        pattern_data = np.array(pattern_data).T.tolist()
        pattern_df = self._get_pattern_sparse2dense(pattern_data, pattern_names)
        # pattern_df = self._get_pattern_one_hot(pattern_data, pattern_names)

        assert len(df) == len(pattern_df), "The original dataframe and the new indicator dataframe are different " \
                                          f"lengths. df [{len(df)}] vs candle_df [{len(pattern_df)}]. " \
                                          f"They cannot be combined"
        return concat([df, pattern_df], axis=1)

    def _get_pattern_one_hot(self, candle_data_t, candle_names):
        candlestick_pattern = []
        no_pattern_metadata = candle_rankings_2.get("NO_PATTERN")

        pattern_embedding = np.array(range(0, len(candle_rankings_2)))
        one_hot_encoding = np.zeros((pattern_embedding.size, np.max(pattern_embedding) + 1))
        one_hot_encoding[np.arange(pattern_embedding.size), pattern_embedding] = 1

        for r_index, row in enumerate(candle_data_t):
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

            candlestick_pattern.append(best_pattern)

        candle_df = DataFrame(candlestick_pattern, columns=pattern_embedding)

        print("--------------")
        print(candle_df.head(1))
        print("--------------")
        print("ONE-HOT CONVERSION COMPLETE")

        return candle_df

    def _get_pattern_sparse2dense(self, candle_data_t, candle_names):
        patterns = []
        no_pattern_metadata = candle_rankings_2.get("NO_PATTERN")

        for r_index, row in enumerate(candle_data_t):
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
        self.logger.log("PATTERN SPARSE TO DENSE CONVERSION COMPLETE")

        return pattern_df

    def _bin_volume(self, df):
        jenks_breaks = [0.0, 125.416263, 298.632517, 608.76506, 1197.486795, 2445.284399, 8277.1723]

        df['volume_bin'] = pd.qcut(
            df['volume'], q=4, labels=[1, 2, 3, 4])

        # df['volume_bin'] = pd.cut(df['volume'],
        #                           include_lowest=True,
        #                           bins=jenks_breaks,
        #                           labels=[1, 2, 3, 4, 5, 6]
        #                           # labels=['bucket_1', 'bucket_2', 'bucket_3', 'bucket_4', 'bucket_5', 'bucket_6']
        #                           )

        return df
