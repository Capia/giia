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

    def __init__(self, logger: LoggerUtil):
        self.logger = logger

    def create_train_test_dataset(self, dataset_dir_path: Path, filedataset_based=True, one_dim_target=True,
                                  starting_date_truncate=None):
        df = self.get_df(starting_date_truncate)

        # train = PandasDataset([ts.iloc[:-config.HYPER_PARAMETERS['prediction_length'], :] for ts in df])
        # train = PandasDataset(df.iloc[:-config.HYPER_PARAMETERS['prediction_length'], :])
        train = PandasDataset(
            # df,
            df.iloc[:-config.HYPER_PARAMETERS['prediction_length'], :],
            target="roc",
            freq=config.DATASET_FREQ,
            assume_sorted=True
        )
        test = PandasDataset(
            df,
            target="roc",
            freq=config.DATASET_FREQ,
            assume_sorted=True
        )

        print(f"Train dataset stats: {calculate_dataset_statistics(train)}")
        print(f"Test dataset stats: {calculate_dataset_statistics(test)}")

        # Split dataset between training and testing
        # train_df, test_df = np.array_split(
        #     df, (fractions[:-1].cumsum() * len(df)).astype(int))

        # if filedataset_based:
        self.create_train_test_filedataset(dataset_dir_path, train, test)
        # else:
        #     self.create_train_test_csv(dataset_dir_path, train, test)

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

        df = mf.marshal_candle_metadata(candles, drop_date_column=True)
        if starting_date_truncate:
            df = df[starting_date_truncate:]

        self.logger.log("First sample:")
        self.logger.log(df.head(1), newline=True)
        self.logger.log("Last sample:")
        self.logger.log(df.tail(1), newline=True)
        self.logger.log(f"Number of raw columns: {len(df.columns)}")
        self.logger.log(f"Number of rows: {len(df)}")

        return df

    def create_train_test_filedataset(self, dataset_dir_path, train_dataset, test_dataset):
        # if one_dim_target:
        #     self.logger.log("Building a univariate FileDataset")
        #
        # train_dataset = gh.df_to_univariate_dataset(train_df)
        #     test_dataset = gh.df_to_univariate_dataset(test_df)
        # else:
        #     self.logger.log("Building a multivariate FileDataset")
        #
        #     feature_columns = gh.get_feature_columns(train_df, exclude_close=False)
        #     self.logger.log(f"Number of feature columns: {len(feature_columns)}")
        #
        #     train_dataset = gh.df_to_multivariate_target_dataset(train_df, feature_columns)
        #     test_dataset = gh.df_to_multivariate_target_dataset(test_df, feature_columns)

        datasets = gh.build_train_datasets(train_dataset, test_dataset)
        datasets.save(str(dataset_dir_path), ArrowWriter(), overwrite=True)
        self.logger.log(f"Parsed train and test datasets can be found in [{dataset_dir_path}]", 'debug')

    def create_train_test_csv(self, dataset_dir_path, train_df, test_df):
        train_df.to_csv(dataset_dir_path / config.TRAIN_CSV_FILENAME)
        test_df.to_csv(dataset_dir_path / config.TEST_CSV_FILENAME)
        self.logger.log(f"Parsed train and test datasets can be found in [{dataset_dir_path}]", 'debug')
