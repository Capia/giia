import json
from typing import Optional

import gluonts
import mxnet
from gluonts.mx import GaussianOutput
from gluonts.mx.distribution import NegativeBinomialOutput, PoissonOutput, StudentTOutput
from mxnet.runtime import feature_list
from tap import Tap

from utils import config


class ModelHyperParameters(Tap):
    model_type: str = config.MODEL_TYPE
    epochs: int = config.HYPER_PARAMETERS["epochs"]
    batch_size: int = config.HYPER_PARAMETERS["batch_size"]
    num_batches_per_epoch: int = config.HYPER_PARAMETERS["num_batches_per_epoch"]
    prediction_length: int = config.HYPER_PARAMETERS["prediction_length"]
    context_length: int = config.HYPER_PARAMETERS["context_length"]
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
    dataset_dir: Optional[str]
    model_dir: Optional[str]


class ModelBase:
    """
    Wrapper class with useful methods for all models. This class should not be instantiated directly. Instead, use one
    of the subclasses in `src/models/`. They all inherit from this class and override the `train` method.
    """

    model_hp = None

    def __init__(self, model_hp: ModelHyperParameters):
        self.model_hp = model_hp

    def train(self):
        raise NotImplementedError

    def _describe_model(self):
        print(f"Using the follow arguments: [{self.model_hp}]")

        print(f"The model id is [{config.MODEL_ID}]")
        print(f"The MXNet version is [{mxnet.__version__}]")
        print(f"The GluonTS version is [{gluonts.__version__}]")
        print(f"The GPU count is [{mxnet.context.num_gpus()}]")
        print(f"{feature_list()}")

    def _get_distr_output(self):
        # TODO: Determine the correct distribution method. This article goes over some of the key differences
        # https://www.investopedia.com/articles/06/probabilitydistribution.asp

        if self.model_hp.distr_output == "NegativeBinomialOutput":
            distr_output = NegativeBinomialOutput()
        elif self.model_hp.distr_output == "PoissonOutput":
            distr_output = PoissonOutput()
        elif self.model_hp.distr_output == "GaussianOutput":
            distr_output = GaussianOutput()
        elif self.model_hp.distr_output == "StudentTOutput":
            distr_output = StudentTOutput()
        else:
            raise ValueError(f"[{self.model_hp.distr_output}] is not a valid choice")

        print(f"Using distr_output [{type(distr_output).__name__}]")
        return distr_output

    def _get_hidden_dimensions(self):
        n_hidden_layer = self.model_hp.n_hidden_layer
        n_neurons_per_layer = self.model_hp.n_neurons_per_layer
        num_hidden_dimensions = [n_neurons_per_layer] * n_hidden_layer

        print(f"num_hidden_dimensions=[{num_hidden_dimensions}]")
        return num_hidden_dimensions

    @staticmethod
    def _get_ctx():
        if mxnet.context.num_gpus():
            ctx = mxnet.gpu()
            print("Using GPU context")
        else:
            ctx = mxnet.cpu()
            print("Using CPU context")
        return ctx

    @staticmethod
    def _print_metrics(agg_metrics, item_metrics, metadata):
        for key in list(agg_metrics.keys()):
            if key[0].isdigit():
                del agg_metrics[key]
        print("Aggregated performance")
        print(json.dumps(agg_metrics, indent=4))
