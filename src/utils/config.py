# This file should be as basic as possible, with minimal dependencies. All paths should be relative to `./src`
from pathlib import Path

MODEL_NAME = "giia"
MODEL_VERSION = "0.5.2"
MODEL_ID = f"{MODEL_NAME}-{MODEL_VERSION}"

SM_ROLE = 'arn:aws:iam::941048668662:role/service-role/AmazonSageMaker-ExecutionRole-20191206T145896'

TRAIN_DATASET_FILENAME = "train.csv"
TEST_DATASET_FILENAME = "test.csv"

FREQTRADE_USER_DATA_DIR = Path("freqtrade") / "user_data"

CRYPTO_PAIR = "BTC/USDT"

# If you permenatly update these values, then you should also update the MODEL_VERSION
HP_EPOCHS = 20
HP_PREDICTION_LENGTH = 16
HP_NUM_LAYERS = 4
HP_DROPOUT_RATE = 0.209371

