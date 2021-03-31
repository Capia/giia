# This file should be as basic as possible, with minimal dependencies. All paths should be relative to `./src`
from pathlib import Path

MODEL_NAME = "giia"
MODEL_VERSION = "0.5.4"
MODEL_ID = f"{MODEL_NAME}-{MODEL_VERSION}"

SM_ROLE = 'arn:aws:iam::941048668662:role/service-role/AmazonSageMaker-ExecutionRole-20191206T145896'

TRAIN_DATASET_FILENAME = "train.csv"
TEST_DATASET_FILENAME = "test.csv"
DATASET_FREQ = "5min"

FREQTRADE_USER_DATA_DIR = Path("freqtrade") / "user_data"

CRYPTO_PAIR = "BTC/USDT"

# If you permanently update these values, then you should also update the MODEL_VERSION
# https://docs.aws.amazon.com/sagemaker/latest/dg/deepar_hyperparameters.html
_PROD_HYPER_PARAMETERS = {
    'epochs': 10,
    'batch_size': 1440,
    'context_length': 24,
    'prediction_length': 12,
    'num_layers': 6,
    'num_cells': 40,
    'dropout_rate': 0.184484,
    'learning_rate': 0.001
}

# Use these hyper parameters when developing and testing new features. The model will be less accurate, but these HPs
# can be used to get a general idea of how well the model may perform with PROD HPs, without the longer wait time
_MODERATE_HYPER_PARAMETERS = {
    'epochs': 10,
    'batch_size': 144,
    'num_batches_per_epoch': 50,
    'context_length': 24,
    'prediction_length': 24,
    'num_layers': 4,
    'num_cells': 40,
    'dropout_rate': 0.18,
    'learning_rate': 0.001
}

# Use these hyper parameters when developing and need to quickly iterate. The model will not be accurate, but these HPs
# can be used to make sure the model compiles and runs, and to get a decent idea of performance
_SIMPLE_HYPER_PARAMETERS = {
    'epochs': 1,
    'batch_size': 10,
    'num_batches_per_epoch': 10,
    'context_length': 12,
    'prediction_length': 12,
    'num_layers': 1,
    'num_cells': 20,
    'dropout_rate': 0.01,
    'learning_rate': 0.001
}

# DO NOT COMMIT ANY CHANGES TO THIS CONFIG `HYPER_PARAMETERS = _PROD_HYPER_PARAMETERS`. You can change it for testing,
# just do not commit it
HYPER_PARAMETERS = _PROD_HYPER_PARAMETERS
