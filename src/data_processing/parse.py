import os

import pandas as pd
import numpy as np
import ntpath
from utils.logging import LoggerUtil


# Dataset retrieved from:
#   https://finance.yahoo.com/quote/%5EGSPC/history?period1=788936400&period2=1564545600&interval=1mo&filter=history&frequency=1mo
class Parse:
    logger = None
    train_dataset_base_path = "datasets/train"
    test_dataset_base_path = "datasets/test"

    def __init__(self, logger: LoggerUtil):
        self.logger = logger

    def split_train_test_dataset(self, dataset: str):
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

        # Setup dynamic filenames
        dataset_filename = ntpath.basename(dataset)
        train_dataset_path = f"{self.train_dataset_base_path}-{dataset_filename}.csv"
        train_dataset_path = f"{self.test_dataset_base_path}-{dataset_filename}.csv"

        self.logger.log(train_dataset_path, 'debug')
        train.to_csv(train_dataset_path)
        test.to_csv(test_dataset_path)

        return train_dataset_path, test_dataset_path
