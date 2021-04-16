import numpy as np
from pathlib import Path

from gluonts.dataset.common import ListDataset, TrainDatasets, CategoricalFeatureInfo, MetaData
from gluonts.dataset.field_names import FieldName
from gluonts.dataset.multivariate_grouper import MultivariateGrouper
from pandas import DataFrame

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

        self.logger.log("First sample:")
        self.logger.log(df.head(1), newline=True)
        self.logger.log("Last sample:")
        self.logger.log(df.tail(1), newline=True)

        # Configure fractions to split dataset between training and testing (validation can be added easily)
        fractions = np.array([0.7, 0.3])

        # Split dataset between training and testing
        train_df, test_df = np.array_split(
            df, (fractions[:-1].cumsum() * len(df)).astype(int))

        # Copy dataset channels to their respective file
        dataset_dir_path.mkdir(parents=True, exist_ok=True)

        train_dataset = self.df_to_multivariate_dataset(train_df)
        test_dataset = self.df_to_multivariate_dataset(test_df)

        datasets = TrainDatasets(
            metadata=MetaData(
                freq=config.DATASET_FREQ,
                # target={'name': 'close'},
                feat_static_cat=[
                    CategoricalFeatureInfo(name="num_series", cardinality=len(df.columns)),

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

        # print(MultivariateGrouper.to_ts(datasets.train))
        # print(MultivariateGrouper.to_ts(datasets.train).tail(1))
        # print(MultivariateGrouper.to_ts(datasets.train).describe())

        datasets.save(str(dataset_dir_path))
        self.logger.log(f"Parsed train and test datasets can be found in [{dataset_dir_path}]", 'debug')

    def _marshal_candles(self, candles: DataFrame) -> DataFrame:
        # Index by datetime
        df = candles.set_index('date')

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

    def df_to_dataset(self, df):
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

    def df_to_multivariate_dataset(self, df):
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
