# This file should be as basic as possible, with minimal dependencies. All paths should be relative to `./src`
from pathlib import Path

MODEL_NAME = "giia"
MODEL_VERSION = "0.7.6"
MODEL_ID = f"{MODEL_NAME}-{MODEL_VERSION}"

SM_ROLE = 'arn:aws:iam::941048668662:role/service-role/AmazonSageMaker-ExecutionRole-20191206T145896'

METADATA_DATASET_FILENAME = "metadata/metadata.json"
TRAIN_DATASET_FILENAME = "train/data.json"
TEST_DATASET_FILENAME = "test/data.json"
TRAIN_CSV_FILENAME = "train.csv"
TEST_CSV_FILENAME = "test.csv"

DATASET_FREQ = "1min"

FREQTRADE_USER_DATA_DIR = Path("freqtrade") / "user_data"
# Cannot be longer than 499 because freqtrader doesn't have dataframes available to the strategy beyond that.
# https://github.com/freqtrade/freqtrade-strategies/issues/79
FREQTRADE_MAX_CONTEXT = 499

CRYPTO_PAIR = "ETH/USDT"

# If you permanently update these values, then you should also update the MODEL_VERSION
# https://docs.aws.amazon.com/sagemaker/latest/dg/deepar_hyperparameters.html
_PROD_HYPER_PARAMETERS = {
    'epochs': 4,
    'batch_size': 256,
    'num_batches_per_epoch': 100,
    'prediction_length': 5,

    # This cannot be longer than `FREQTRADE_MAX_CONTEXT`
    # Also, this significantly increases memory usage. Beware
    'context_length': 60,

    'num_layers': 8,
    'num_cells': 200,

    'skip_size': 18,
    'ar_window': 18,
    'channels': 180,
    'rnn_num_layers': 180,
    'skip_rnn_num_layers': 18,
    'kernel_size': 18,

    'dropout_rate': 0.1525,
    'learning_rate': 0.001
}

# Use these hyper parameters when developing, testing new features, and tuning. The model will be less accurate, but
# these HPs can be used to get a general idea of how well the model may perform with PROD HPs, without the longer
# wait time
_MODERATE_HYPER_PARAMETERS = {
    'epochs': 2,
    'batch_size': 64,
    'num_batches_per_epoch': 100,
    'prediction_length': 5,
    'context_length': 60,
    'num_layers': 3,
    'num_cells': 65,

    'skip_size': 4,
    'ar_window': 8,
    'channels': 40,
    'rnn_num_layers': 40,
    'skip_rnn_num_layers': 4,
    'kernel_size': 4,

    'dropout_rate': 0.1525,
    'learning_rate': 0.001
}

# Use these hyper parameters when developing and need to quickly iterate. The model will not be accurate, but these HPs
# can be used to make sure the model compiles and runs, and to get a decent idea of performance
_SIMPLE_HYPER_PARAMETERS = {
    'epochs': 1,
    'batch_size': 16,
    'num_batches_per_epoch': 10,
    'prediction_length': 5,
    'context_length': 60,

    'num_layers': 1,
    'num_cells': 20,

    'skip_size': 2,
    'ar_window': 4,
    'channels': 20,
    'rnn_num_layers': 20,
    'skip_rnn_num_layers': 2,
    'kernel_size': 2,

    'dropout_rate': 0.01,
    'learning_rate': 0.001
}

# DO NOT COMMIT ANY CHANGES TO THIS CONFIG `HYPER_PARAMETERS = _PROD_HYPER_PARAMETERS`. You can change it for testing,
# just do not commit it
HYPER_PARAMETERS = _MODERATE_HYPER_PARAMETERS
