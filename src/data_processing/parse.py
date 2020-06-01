import os

import pandas as pd
import numpy as np
from utils.logging import LoggerUtil


# Dataset retrieved from:
#   https://finance.yahoo.com/quote/%5EGSPC/history?period1=788936400&period2=1564545600&interval=1mo&filter=history&frequency=1mo
class Parse:
    logger = None
    train_dataset_path = "datasets/train.csv"
    test_dataset_path = "datasets/test.csv"

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

        self.logger.log(self.train_dataset_path, 'debug')
        train.to_csv(self.train_dataset_path)
        test.to_csv(self.test_dataset_path)

        return self.train_dataset_path, self.test_dataset_path
