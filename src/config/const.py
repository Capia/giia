# This file should be as basic as possible, with little to no dependencies

MODEL_NAME = "giia"
MODEL_VERSION = "0.3.3"
MODEL_ID = f"{MODEL_NAME}-{MODEL_VERSION}"

DATASET = "../freqtrade/user_data/data/binance/ETH_BTC-5m.json"
SRC_DATASET_DIR = "../freqtrade/user_data/data/binance"
SM_ROLE = 'arn:aws:iam::941048668662:role/service-role/AmazonSageMaker-ExecutionRole-20191206T145896'

TRAIN_DATASET_FILENAME = "train.csv"
TEST_DATASET_FILENAME = "test.csv"

