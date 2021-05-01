import numpy as np
from pathlib import Path

from gluonts.dataset.common import ListDataset, TrainDatasets, CategoricalFeatureInfo, MetaData
from gluonts.dataset.field_names import FieldName

import data_processing.marshal_features as mf
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

        df = mf.marshal_candles(candles)

        self.logger.log("First sample:")
        self.logger.log(df.head(1), newline=True)
        self.logger.log("Last sample:")
        self.logger.log(df.tail(1), newline=True)
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
        feature_columns = []

        for column_name in df.columns:
            if column_name not in dynamic_real_features_blacklist:
                feature_columns.append(column_name)

        print(feature_columns)

        return ListDataset(
            [
                {
                    FieldName.START: df.index[0],
                    FieldName.TARGET: df["close"][:].values,
                    FieldName.ITEM_ID: "close",
                    FieldName.FEAT_DYNAMIC_REAL: [
                        df[column_name][:].values for column_name in feature_columns
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
