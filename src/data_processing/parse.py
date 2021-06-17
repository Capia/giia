import numpy as np
from pathlib import Path

from freqtrade.data.history import load_pair_history
from freqtrade.configuration import Configuration

import data_processing.gluonts_helper as gh
import data_processing.marshal_features as mf
from utils.logger_util import LoggerUtil
from utils import config


class Parse:
    logger = None

    def __init__(self, logger: LoggerUtil):
        self.logger = logger

    def create_train_test_dataset(self, dataset_dir_path: Path, filedataset_based=True, one_dim_target=True,
                                  starting_date_truncate=None):
        # Copy dataset channels to their respective file
        dataset_dir_path.mkdir(parents=True, exist_ok=True)

        # First prime the user_data_dir key. This will take priority when merged with config.json
        freqtrade_config = Configuration({"user_data_dir": config.FREQTRADE_USER_DATA_DIR})
        freqtrade_config = freqtrade_config.load_from_files([str(config.FREQTRADE_USER_DATA_DIR / "config.json")])

        data_dir = config.FREQTRADE_USER_DATA_DIR / "data" / "binance"
        candles = load_pair_history(
            datadir=data_dir,
            timeframe=freqtrade_config["timeframe"],
            pair=config.CRYPTO_PAIR)

        if candles.empty:
            raise ValueError("The candle dataframe is empty. Ensure that you are loading a dataset that has been "
                             f"downloaded to [{data_dir}]")

        df = mf.marshal_candle_metadata(candles, drop_date_column=True)
        if starting_date_truncate:
            df = df[starting_date_truncate:]

        self.logger.log("First sample:")
        self.logger.log(df.head(1), newline=True)
        self.logger.log("Last sample:")
        self.logger.log(df.tail(1), newline=True)
        self.logger.log(f"Number of rows: {len(df)}")
        self.logger.log(f"Number of raw columns: {len(df.columns)}")

        # Configure fractions to split dataset between training and testing (validation can be added easily)
        fractions = np.array([0.7, 0.3])

        # Split dataset between training and testing
        train_df, test_df = np.array_split(
            df, (fractions[:-1].cumsum() * len(df)).astype(int))
        self.logger.log(f"Train dataset starts at: {train_df.iloc[0]}")
        self.logger.log(f"Test dataset starts at: {test_df.iloc[0]}")

        if filedataset_based:
            self.create_train_test_filedataset(dataset_dir_path, train_df, test_df, one_dim_target)
        else:
            self.create_train_test_csv(dataset_dir_path, train_df, test_df)

    def create_train_test_filedataset(self, dataset_dir_path, train_df, test_df, one_dim_target):
        if one_dim_target:
            self.logger.log("Building a univariate FileDataset")

            feature_columns = []
            train_dataset = gh.df_to_univariate_dataset(train_df)
            test_dataset = gh.df_to_univariate_dataset(test_df)
        else:
            self.logger.log("Building a multivariate FileDataset")

            feature_columns = gh.get_feature_columns(train_df, exclude_close=False)
            self.logger.log(f"Number of feature columns: {len(feature_columns)}")

            train_dataset = gh.df_to_multivariate_target_dataset(train_df, feature_columns)
            test_dataset = gh.df_to_multivariate_target_dataset(test_df, feature_columns)

        datasets = gh.build_train_datasets(train_df, train_dataset, test_df, test_dataset, feature_columns)

        datasets.save(str(dataset_dir_path))
        self.logger.log(f"Parsed train and test datasets can be found in [{dataset_dir_path}]", 'debug')

    def create_train_test_csv(self, dataset_dir_path, train_df, test_df):
        train_df.to_csv(dataset_dir_path / config.TRAIN_CSV_FILENAME)
        test_df.to_csv(dataset_dir_path / config.TEST_CSV_FILENAME)
        self.logger.log(f"Parsed train and test datasets can be found in [{dataset_dir_path}]", 'debug')
