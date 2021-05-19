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
        self.logger.log(f"Number of raw columns: {len(df.columns)}")

        # Configure fractions to split dataset between training and testing (validation can be added easily)
        fractions = np.array([0.7, 0.3])

        # Split dataset between training and testing
        train_df, test_df = np.array_split(
            df, (fractions[:-1].cumsum() * len(df)).astype(int))

        # Save df to file to make it easy to visualize/debug. Test df is used as it is smaller and more portable
        test_df.to_csv("test.csv")

        feature_columns = gh.get_feature_columns(df)
        self.logger.log(f"Number of feature columns: {len(feature_columns)}")

        train_dataset = gh.df_to_multi_feature_dataset(train_df, feature_columns)
        test_dataset = gh.df_to_multi_feature_dataset(test_df, feature_columns)

        datasets = gh.build_train_datasets(train_df, train_dataset, test_df, test_dataset, feature_columns)

        datasets.save(str(dataset_dir_path))
        self.logger.log(f"Parsed train and test datasets can be found in [{dataset_dir_path}]", 'debug')
