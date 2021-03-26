import numpy as np
from pathlib import Path
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
        train.to_csv(dataset_dir_path / config.TRAIN_DATASET_FILENAME)
        test.to_csv(dataset_dir_path / config.TEST_DATASET_FILENAME)
        self.logger.log(f"Parsed train and test datasets can be found in [{dataset_dir_path}]", 'debug')

    def _marshal_candles(self, candles: DataFrame):
        # Index by datetime
        df = candles.set_index('date')

        # Then remove UTC timezone since GluonTS does not work with it
        df.index = df.index.tz_localize(None)
        return df
