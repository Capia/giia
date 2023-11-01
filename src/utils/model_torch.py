import json
import os
import subprocess
from pathlib import Path
from typing import Optional

import gluonts
import pandas as pd
import torch
from tap import Tap

from utils import config


class ModelHyperParameters(Tap):
    model_type: str = config.MODEL_TYPE
    epochs: int = config.HYPER_PARAMETERS["epochs"]
    batch_size: int = config.HYPER_PARAMETERS["batch_size"]
    num_batches_per_epoch: int = config.HYPER_PARAMETERS["num_batches_per_epoch"]
    prediction_length: int = config.HYPER_PARAMETERS["prediction_length"]
    context_length: int = config.HYPER_PARAMETERS["context_length"]
    num_heads: int = config.HYPER_PARAMETERS["num_heads"]
    hidden_dim: int = config.HYPER_PARAMETERS["hidden_dim"]
    variable_dim: int = config.HYPER_PARAMETERS["variable_dim"]
    model_dim: int = config.HYPER_PARAMETERS["model_dim"]
    num_layers: int = config.HYPER_PARAMETERS["num_layers"]
    num_cells: int = config.HYPER_PARAMETERS["num_cells"]
    n_hidden_layer: int = config.HYPER_PARAMETERS["n_hidden_layer"]
    n_neurons_per_layer: int = config.HYPER_PARAMETERS["n_neurons_per_layer"]
    distr_output: str = config.HYPER_PARAMETERS["distr_output"]
    skip_size: int = config.HYPER_PARAMETERS["skip_size"]
    ar_window: int = config.HYPER_PARAMETERS["ar_window"]
    channels: int = config.HYPER_PARAMETERS["channels"]
    rnn_num_layers: int = config.HYPER_PARAMETERS["rnn_num_layers"]
    skip_rnn_num_layers: int = config.HYPER_PARAMETERS["skip_rnn_num_layers"]
    kernel_size: int = config.HYPER_PARAMETERS["kernel_size"]
    dropout_rate: float = config.HYPER_PARAMETERS["dropout_rate"]
    learning_rate: float = config.HYPER_PARAMETERS["learning_rate"]
    dataset_dir: Optional[str] = None
    model_dir: Optional[str] = None

    def process_args(self):
        if not self.dataset_dir:
            self.dataset_dir = os.environ['SM_CHANNEL_DATASET']
        if not self.model_dir:
            self.model_dir = os.environ['SM_MODEL_DIR']


class ModelTorchBase:
    """
    Wrapper class with useful methods for all torch models. This class should not be instantiated directly. Instead,
    use one of the subclasses in `src/ml/models/`. They all inherit from this class and override the `train` method.
    """

    model_hp = None

    def __init__(self, model_hp: ModelHyperParameters):
        self.model_hp = model_hp

    def train(self):
        raise NotImplementedError

    def _get_model_id(self):
        return config.MODEL_ID

    def _describe_env(self):
        print(f"Using the follow arguments: [{self.model_hp}]")

        print(f"The model id is [{config.MODEL_ID}]")
        print(f"The PyTorch version is [{torch.__version__}]")
        print(f"The GluonTS version is [{gluonts.__version__}]")
        print(f"The GPU count is [{torch.cuda.device_count()}]")

        device = 'gpu' if torch.cuda.is_available() else 'cpu'
        if torch.cuda.is_available():
            print("Using GPU context")
            print("nvidia-smi")
            subprocess.call(['nvidia-smi'])
            print("nvcc --version")
            subprocess.call(['nvcc', '--version'])
        else:
            print("Using CPU context")

        return device

    def _get_hidden_dimensions(self):
        n_hidden_layer = self.model_hp.n_hidden_layer
        n_neurons_per_layer = self.model_hp.n_neurons_per_layer
        num_hidden_dimensions = [n_neurons_per_layer] * n_hidden_layer

        print(f"num_hidden_dimensions=[{num_hidden_dimensions}]")
        return num_hidden_dimensions

    @staticmethod
    def _print_metrics(agg_metrics, item_metrics, metadata):
        for key in list(agg_metrics.keys()):
            if key[0].isdigit():
                del agg_metrics[key]
        print("Aggregated performance")
        print(json.dumps(agg_metrics, indent=4))

    @staticmethod
    def _get_df_from_dataset_file(dataset_path: Path):
        df = pd.read_csv(filepath_or_buffer=dataset_path / config.TRAIN_CSV_FILENAME, header=0, index_col=0)

        print(f"First {dataset_path} sample:")
        print(df.head(1))
        print(f"\nLast {dataset_path} sample:")
        print(df.tail(1))
        print(df.describe())

        return df

