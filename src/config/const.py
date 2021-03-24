# This file should be as basic as possible, with minimal dependencies. All paths should be strings where path logic is
# handled by the caller

MODEL_NAME = "giia"
MODEL_VERSION = "0.3.3"
MODEL_ID = f"{MODEL_NAME}-{MODEL_VERSION}"

FREQTRADE_USER_DATA_DIR = "freqtrade/user_data"
SM_ROLE = 'arn:aws:iam::941048668662:role/service-role/AmazonSageMaker-ExecutionRole-20191206T145896'

TRAIN_DATASET_FILENAME = "train.csv"
TEST_DATASET_FILENAME = "test.csv"

