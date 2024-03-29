# This file should be as basic as possible, with minimal dependencies. All paths should be relative to `./src`
from pathlib import Path

MODEL_NAME = "giia"
MODEL_TYPE = "tft"
MODEL_VERSION = "1.2.0"
MODEL_ID = f"{MODEL_NAME}-{MODEL_TYPE}-{MODEL_VERSION}"

SM_ROLE = 'arn:aws:iam::941048668662:role/service-role/AmazonSageMaker-ExecutionRole-20191206T145896'

METADATA_DATASET_FILENAME = "metadata.json"
TRAIN_DATASET_FILENAME = "train/data.feather"
TEST_DATASET_FILENAME = "test/data.feather"
TRAIN_CSV_FILENAME = "train.csv"
TEST_CSV_FILENAME = "test.csv"
CACHED_PRED_CSV_0 = "../out/pred_cache_0.csv"
CACHED_PRED_CSV_1 = "../out/pred_cache_1.csv"

DATASET_FREQ = "1min"

FREQTRADE_USER_DATA_DIR = Path("freqtrade") / "user_data"
# Cannot be longer than 499 because freqtrader doesn't have dataframes available to the strategy beyond that.
# https://github.com/freqtrade/freqtrade-strategies/issues/79
FREQTRADE_MAX_CONTEXT = 499

CRYPTO_PAIR = "ETH/USDT"

# If you permanently update these values, then you should also update the MODEL_VERSION
# https://docs.aws.amazon.com/sagemaker/latest/dg/deepar_hyperparameters.html
_PROD_HYPER_PARAMETERS = {
    'epochs': 3,
    'batch_size': 64,
    'num_batches_per_epoch': 500,
    'prediction_length': 5,

    # This cannot be longer than `FREQTRADE_MAX_CONTEXT`
    # Also, this significantly increases memory usage. Beware
    'context_length': 60,

    'num_heads': 32,
    'hidden_dim': 256,
    'variable_dim': 256,

    'model_dim': 2048,
    'num_layers': 8,
    'num_cells': 256,

    'n_hidden_layer': 10,
    'n_neurons_per_layer': 512,
    'distr_output': "StudentTOutput",

    'skip_size': 32,
    'ar_window': 32,
    'channels': 128,
    'rnn_num_layers': 128,
    'skip_rnn_num_layers': 4,
    'kernel_size': 6,

    'dropout_rate': 0.1525,
    'learning_rate': 0.001
}

# Use these hyper parameters when developing, testing new features, and tuning. The model will be less accurate, but
# these HPs can be used to get a general idea of how well the model may perform with PROD HPs, without the longer
# wait time
_MODERATE_HYPER_PARAMETERS = {
    'epochs': 5,
    'batch_size': 32,
    'num_batches_per_epoch': 200,
    'prediction_length': 5,
    'context_length': 30,

    'num_heads': 4,
    'hidden_dim': 32,
    'variable_dim': 32,

    'model_dim': 256,
    'num_layers': 6,
    'num_cells': 192,

    'n_hidden_layer': 8,
    'n_neurons_per_layer': 512,
    'distr_output': "StudentTOutput",

    'skip_size': 4,
    'ar_window': 8,
    'channels': 64,
    'rnn_num_layers': 64,
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
    'context_length': 10,

    'num_heads': 2,
    'hidden_dim': 16,
    'variable_dim': 16,

    'model_dim': 16,
    'num_layers': 1,
    'num_cells': 20,

    'n_hidden_layer': 2,
    'n_neurons_per_layer': 10,
    'distr_output': "StudentTOutput",

    'skip_size': 2,
    'ar_window': 4,
    'channels': 20,
    'rnn_num_layers': 20,
    'skip_rnn_num_layers': 2,
    'kernel_size': 2,

    'dropout_rate': 0.01,
    'learning_rate': 0.001
}

# HYPER_PARAMETERS = _PROD_HYPER_PARAMETERS
# HYPER_PARAMETERS = _MODERATE_HYPER_PARAMETERS
HYPER_PARAMETERS = _SIMPLE_HYPER_PARAMETERS
