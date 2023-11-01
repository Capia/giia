from pathlib import Path

from freqtrade.configuration import Configuration
from freqtrade.data.history import load_pair_history
from gluonts.dataset.arrow import ArrowWriter
from gluonts.dataset.jsonl import JsonLinesWriter
from gluonts.dataset.pandas import PandasDataset
from gluonts.dataset.stat import calculate_dataset_statistics

import data_processing.gluonts_helper as gh
import data_processing.marshal_features as mf
from utils import config
from utils.logger_util import LoggerUtil


class Parse:
    logger = None

    def __init__(self, logger):
        self.logger = logger

    def get_df(self, starting_date_truncate):
        # First prime the user_data_dir key. This will take priority when merged with config.json
        freqtrade_config = Configuration({
            "user_data_dir": config.FREQTRADE_USER_DATA_DIR,
            "config": [str(config.FREQTRADE_USER_DATA_DIR / "config.json")]
        })
        # freqtrade_config = freqtrade_config.from_files([str(config.FREQTRADE_USER_DATA_DIR / "config.json")])
        data_dir = config.FREQTRADE_USER_DATA_DIR / "data" / "kraken"
        candles = load_pair_history(
            datadir=data_dir,
            timeframe=freqtrade_config.get_config()["timeframe"],
            pair=config.CRYPTO_PAIR)
        if candles.empty:
            raise ValueError("The candle dataframe is empty. Ensure that you are loading a dataset that has been "
                             f"downloaded to [{data_dir}]")

        df = mf.marshal_candle_metadata(candles, starting_date_truncate, drop_date_column=True)

        self.logger.log("First sample:")
        self.logger.log(df.head(1), newline=True)
        self.logger.log("Last sample:")
        self.logger.log(df.tail(1), newline=True)
        self.logger.log(f"Number of raw columns: {len(df.columns)}")
        self.logger.log(f"Number of rows: {len(df)}")

        return df

    def save_df_to_csv(self, df, dataset_dir_path):
        df.to_csv(dataset_dir_path / config.TRAIN_CSV_FILENAME)
        self.logger.log(f"Parsed csv dataframe can be found in [{dataset_dir_path}]", 'debug')
