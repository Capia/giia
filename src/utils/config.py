# This file should be as basic as possible, with minimal dependencies. All paths should be relative to `./src`
from pathlib import Path

MODEL_NAME = "giia"
MODEL_VERSION = "0.5.3"
MODEL_ID = f"{MODEL_NAME}-{MODEL_VERSION}"

SM_ROLE = 'arn:aws:iam::941048668662:role/service-role/AmazonSageMaker-ExecutionRole-20191206T145896'

TRAIN_DATASET_FILENAME = "train.csv"
TEST_DATASET_FILENAME = "test.csv"

FREQTRADE_USER_DATA_DIR = Path("freqtrade") / "user_data"

CRYPTO_PAIR = "BTC/USDT"

# If you permanently update these values, then you should also update the MODEL_VERSION
_REAL_HYPER_PARAMETERS = {
    'epochs': 30,
    'prediction_length': 15,
    'num_layers': 6,
    'dropout_rate': 0.184484
}

_QUICK_ITERATION_HYPER_PARAMETERS = {
    'epochs': 1,
    'prediction_length': 1,
    'num_layers': 1,
    'dropout_rate': 0.001
}

HYPER_PARAMETERS = _QUICK_ITERATION_HYPER_PARAMETERS
