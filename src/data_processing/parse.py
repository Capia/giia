import numpy as np
from pathlib import Path
from pandas import DataFrame

from utils.logger_util import LoggerUtil
from freqtrade.data.history import load_pair_history
from freqtrade.configuration import Configuration

import config.const as conf


class Parse:
    logger = None

    def __init__(self, logger: LoggerUtil):
        self.logger = logger

    def split_train_test_dataset(self, dataset_dir_path: Path):
        src_dataset_dir = Path(conf.FREQTRADE_USER_DATA_DIR) / "data" / "binance"
        config_file = Path(conf.FREQTRADE_USER_DATA_DIR) / "config.json"

        config = Configuration.from_files([config_file])
        candles = load_pair_history(
            datadir=src_dataset_dir,
            timeframe=config["timeframe"],
            pair="ETH/BTC")

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
        train, test = np.array_split(
            df, (fractions[:-1].cumsum() * len(df)).astype(int))

        # Copy dataset channels to their respective file
        dataset_dir_path.mkdir(parents=True, exist_ok=True)
        train.to_csv(dataset_dir_path / conf.TRAIN_DATASET_FILENAME)
        test.to_csv(dataset_dir_path / conf.TEST_DATASET_FILENAME)
        self.logger.log(f"Parsed train and test datasets can be found in [{dataset_dir_path}]", 'debug')

    def _marshal_candles(self, candles: DataFrame):
        # Index by datetime
        df = candles.set_index('date')

        # Then remove UTC timezone since GluonTS does not work with it
        df.index = df.index.tz_localize(None)
        return df
