import numpy as np
from pathlib import Path

from gluonts.dataset.common import ListDataset, TrainDatasets, CategoricalFeatureInfo, MetaData
from gluonts.dataset.field_names import FieldName
from gluonts.dataset.multivariate_grouper import MultivariateGrouper
from pandas import DataFrame, concat
import talib
from itertools import compress

from data_processing.candle_rankings import candle_rankings
from utils.logger_util import LoggerUtil
from utils import config
from freqtrade.data.history import load_pair_history
from freqtrade.configuration import Configuration


class Parse:
    logger = None

    def __init__(self, logger: LoggerUtil):
        self.logger = logger

    def split_train_test_dataset(self, dataset_dir_path: Path):
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

        # Copy dataset channels to their respective file
        dataset_dir_path.mkdir(parents=True, exist_ok=True)

        train_dataset = self.df_to_multi_feature_dataset(train_df)
        test_dataset = self.df_to_multi_feature_dataset(test_df)

        datasets = self.build_train_datasets(train_df, train_dataset, test_df, test_dataset)

        datasets.save(str(dataset_dir_path))
        self.logger.log(f"Parsed train and test datasets can be found in [{dataset_dir_path}]", 'debug')

    def _marshal_candles(self, candles: DataFrame) -> DataFrame:
        # These features are easier to manipulate with an integer index, so we run this first
        df = self._add_indicators(candles)
        # df = df.replace(np.nan, 0)

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
        dynamic_real_features_blacklist = ["close"]

        return ListDataset(
            [
                {
                    FieldName.START: df.index[0],
                    FieldName.TARGET: df["close"][:].values,
                    FieldName.FEAT_DYNAMIC_REAL: [
                        df[column_name][:].values for column_name in df.columns if column_name not in dynamic_real_features_blacklist
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
        candle_names = talib.get_function_groups()['Pattern Recognition']
        candle_data = []
        self.logger.log(f"Number of indicators: {len(candle_names)}")

        # create columns for each pattern
        for candle_name in candle_names:
            # below is same as;
            # candle_date["CDL3LINESTRIKE"] = talib.CDL3LINESTRIKE(op, hi, lo, cl)
            candle_data.append(getattr(talib, candle_name)(df["open"], df["high"], df["low"], df["close"]))

        candle_data_t = np.array(candle_data).T.tolist()
        candle_df = DataFrame(candle_data_t, columns=candle_names)

        # candle_df['candlestick_pattern'] = np.nan
        # candle_df['candlestick_match_count'] = np.nan
        # for index, row in candle_df.iterrows():
        #     # no pattern found
        #     if len(row[candle_names]) - sum(row[candle_names] == 0) == 0:
        #         candle_df.loc[index, 'candlestick_pattern'] = "NO_PATTERN"
        #         candle_df.loc[index, 'candlestick_match_count'] = 0
        #     # single pattern found
        #     elif len(row[candle_names]) - sum(row[candle_names] == 0) == 1:
        #         # bull pattern 100 or 200
        #         if any(row[candle_names].values > 0):
        #             pattern = list(compress(row[candle_names].keys(), row[candle_names].values != 0))[0] + '_Bull'
        #             candle_df.loc[index, 'candlestick_pattern'] = pattern
        #             candle_df.loc[index, 'candlestick_match_count'] = 1
        #         # bear pattern -100 or -200
        #         else:
        #             pattern = list(compress(row[candle_names].keys(), row[candle_names].values != 0))[0] + '_Bear'
        #             candle_df.loc[index, 'candlestick_pattern'] = pattern
        #             candle_df.loc[index, 'candlestick_match_count'] = 1
        #     # multiple patterns matched -- select best performance
        #     else:
        #         # filter out pattern names from bool list of values
        #         patterns = list(compress(row[candle_names].keys(), row[candle_names].values != 0))
        #         container = []
        #         for pattern in patterns:
        #             if row[pattern] > 0:
        #                 container.append(pattern + '_Bull')
        #             else:
        #                 container.append(pattern + '_Bear')
        #         rank_list = [candle_rankings.get(p, 999) for p in container]
        #         if len(rank_list) == len(container):
        #             rank_index_best = rank_list.index(min(rank_list))
        #             candle_df.loc[index, 'candlestick_pattern'] = container[rank_index_best]
        #             candle_df.loc[index, 'candlestick_match_count'] = len(container)
        #
        # # clean up candle columns
        # candle_df.drop(candle_names, axis=1, inplace=True)
        # print("--------------")
        # print(candle_df.head(1))
        # print("--------------")

        assert len(df) == len(candle_df), "The original dataframe and the new indicator dataframe are different " \
                                          f"lengths. df [{len(df)}] vs candle_df [{len(candle_df)}]. " \
                                          f"They cannot be combined"
        return concat([df, candle_df], axis=1)
