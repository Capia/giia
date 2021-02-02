from pathlib import Path

import pandas as pd
import numpy as np
from utils.logger_util import LoggerUtil


# Dataset retrieved from:
#   https://finance.yahoo.com/quote/%5EGSPC/history?period1=788936400&period2=1564545600&interval=1mo&filter=history&frequency=1mo
class Parse:
    logger = None
    TRAIN_DATASET_FILENAME = "train.csv"
    TEST_DATASET_FILENAME = "test.csv"

    def __init__(self, logger: LoggerUtil):
        self.logger = logger

    def split_train_test_dataset(self, dataset: str, dataset_dir_path: Path):
        df = pd.read_csv(dataset, header=0, index_col=0)
        self.logger.log("First sample:")
        self.logger.log(df.head(1))
        self.logger.log("\nLast sample:")
        self.logger.log(df.tail(1))

        # Configure fractions to split dataset between training and testing (validation can be added easily)
        fractions = np.array([0.7, 0.3])

        # Split dataset between training and testing
        train, test = np.array_split(
            df, (fractions[:-1].cumsum() * len(df)).astype(int))

        # Copy dataset channels to their respective file
        dataset_dir_path.mkdir(parents=True, exist_ok=True)
        train.to_csv(dataset_dir_path / self.TRAIN_DATASET_FILENAME)
        test.to_csv(dataset_dir_path / self.TEST_DATASET_FILENAME)
        self.logger.log(f"Parsed train and test datasets can be found in [{dataset_dir_path}]", 'debug')
