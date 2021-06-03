#
# NOTE: This file must stay at the root of the `./src` directory due to sagemaker-local stripping the path from the
#  entry_point. Follow this issue for new developments https://github.com/aws/sagemaker-python-sdk/issues/1597
#
import os

import mxnet
import gluonts
import json
import argparse

import pandas as pd
import numpy as np

from typing import List, Tuple, Union
from pathlib import Path

from gluonts.dataset.stat import calculate_dataset_statistics
from gluonts.model.deepar import DeepAREstimator
from gluonts.evaluation.backtest import backtest_metrics, make_evaluation_predictions
from gluonts.evaluation import Evaluator
from gluonts.model.predictor import Predictor
from gluonts.model.forecast import Config, Forecast
from gluonts.dataset.common import load_datasets
from gluonts.mx.distribution import PoissonOutput, NegativeBinomialOutput, StudentTOutput
from gluonts.mx.trainer import Trainer
from mxnet.runtime import feature_list

from utils import config


# Creates a training and testing ListDataset, a DeepAR estimator, and performs the training. It also performs
# evaluation and prints performance metrics
def train(model_args):
    _describe_model(model_args)

    dataset_dir_path = Path(model_args.dataset_dir)
    datasets = load_datasets(
        metadata=(dataset_dir_path / config.METADATA_DATASET_FILENAME).parent,
        train=(dataset_dir_path / config.TRAIN_DATASET_FILENAME).parent,
        test=(dataset_dir_path / config.TEST_DATASET_FILENAME).parent,
    )

    print(f"Train dataset stats: {calculate_dataset_statistics(datasets.train)}")
    print(f"Test dataset stats: {calculate_dataset_statistics(datasets.test)}")

    # Get precomputed train length to prevent iterating through a large dataset in memory
    train_dataset_length = int(next(feat.cardinality
                                    for feat in datasets.metadata.feat_static_cat if feat.name == "ts_train_length"))

    if not model_args.num_batches_per_epoch:
        model_args.num_batches_per_epoch = train_dataset_length // model_args.batch_size
        print(f"Defaulting num_batches_per_epoch to: [{model_args.num_batches_per_epoch}] "
              f"= (length of train dataset [{train_dataset_length}]) / (batch size [{model_args.batch_size}])")

    ctx = _get_ctx()
    distr_output = _get_distr_output()

    estimator = DeepAREstimator(
        freq=config.DATASET_FREQ,
        # batch_size=model_args.batch_size,
        context_length=model_args.context_length,
        prediction_length=model_args.prediction_length,
        dropout_rate=model_args.dropout_rate,
        num_layers=model_args.num_layers,
        num_cells=model_args.num_cells,
        distr_output=distr_output,

        # dropoutcell_type='VariationalDropoutCell',
        use_feat_dynamic_real=True,

        trainer=Trainer(
            ctx=ctx,
            epochs=model_args.epochs,
            batch_size=model_args.batch_size,
            num_batches_per_epoch=model_args.num_batches_per_epoch,
            learning_rate=model_args.learning_rate
        )
    )

    # Train the model
    predictor = estimator.train(training_data=datasets.train)

    # Evaluate trained model on test data. This will serialize each of the agg_metrics into a well formatted log.
    # We use this to capture the metrics needed for hyperparameter tuning
    agg_metrics, item_metrics = backtest_metrics(
        test_dataset=datasets.test,
        predictor=predictor,
        evaluator=Evaluator(
            quantiles=[0.1, 0.5, 0.9],
            # seasonality=5
        ),
        num_samples=100,  # number of samples used in probabilistic evaluation
    )

    # Save the model
    predictor.serialize(Path(model_args.model_dir))

    return predictor


def _get_ctx():
    if mxnet.context.num_gpus():
        ctx = mxnet.gpu()
        print("Using GPU context")
    else:
        ctx = mxnet.cpu()
        print("Using CPU context")
    return ctx


def _get_distr_output():
    # Use betaoutput with normalized delta values such open, close, etc (0 through 1)
    # distr_output=NegativeBinomialOutput(),
    # distr_output=PoissonOutput(),
    # distr_output=LogitNormalOutput(),
    distr_output = StudentTOutput()

    print(f"Using distr_output [{type(distr_output).__name__}]")
    return distr_output


# Used for inference. Once the model is trained, we can deploy it and this function will load the trained model. No-op
# implementation as default will properly handle decompressing and deserializing the model
def model_fn(model_dir):
    model_dir_path = Path(model_dir) / "model"
    print(f"Model dir [{str(model_dir_path)}]")

    predictor = Predictor.deserialize(model_dir_path)
    print(f"Predictor metadata [{predictor.__dict__}]")

    return predictor


# Used for inference. This is the entry point for sending a request to receive a prediction
# https://sagemaker.readthedocs.io/en/stable/frameworks/mxnet/using_mxnet.html#serve-an-mxnet-model
def transform_fn(model, request_body, content_type, accept_type):
    input_df = _input_fn(request_body, content_type)
    forecast = _predict_fn(input_df, model)
    json_output = _output_fn(forecast, accept_type)
    return json_output


def _input_fn(request_body: Union[str, bytes], request_content_type: str = "application/json") -> pd.DataFrame:
    # byte array of json -> JSON object -> str in JSON format
    request_json = json.dumps(json.loads(request_body))
    df = pd.read_json(request_json, orient='split')

    # Clean dataframe
    df = df.drop(['sell', 'buy'], axis=1, errors='ignore')
    df = df.drop(df.filter(regex='pred_close_').columns, axis=1, errors='ignore')

    # Index by datetime
    df = df.set_index('date')

    # Then remove UTC timezone since GluonTS does not work with it
    df.index = df.index.tz_localize(None)

    return df


def _predict_fn(input_df: pd.DataFrame, model: Predictor, num_samples=100) -> List[Forecast]:
    import data_processing.gluonts_helper as gh
    feature_columns = gh.get_feature_columns(input_df)
    print(f"Number of feature columns: {len(feature_columns)}")

    dataset = gh.df_to_multi_feature_dataset(input_df, feature_columns, freq=model.freq)
    print(f"Dataset stats: {calculate_dataset_statistics(dataset)}")

    print(f"Starting prediction...")
    forecast_it, ts_it = make_evaluation_predictions(
        dataset=dataset,
        predictor=model,
        num_samples=num_samples
    )
    print(f"Finished prediction")
    return list(forecast_it)


# Because of transform_fn(), we cannot use output_fn() as function name.
# Hence, we prefix our helper function with an underscore.
def _output_fn(
        forecasts: List[Forecast],
        content_type: str = "application/json",
        config: Config = Config(quantiles=["0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9"]),
) -> Union[str, Tuple[str, str]]:
    # jsonify_floats is taken from gluonts/shell/serve/util.py
    def jsonify_floats(json_object):
        """Traverse through the JSON object and converts non JSON-spec compliant floats(nan, -inf, inf) to string.

        Parameters
        ----------
        json_object
            JSON object
        """
        if isinstance(json_object, dict):
            return {k: jsonify_floats(v) for k, v in json_object.items()}
        elif isinstance(json_object, list):
            return [jsonify_floats(item) for item in json_object]
        elif isinstance(json_object, float):
            if np.isnan(json_object):
                return "NaN"
            elif np.isposinf(json_object):
                return "Infinity"
            elif np.isneginf(json_object):
                return "-Infinity"
            return json_object
        return json_object

    json_forecasts = json.dumps([forecast.as_json_dict(config) for forecast in forecasts])
    return json_forecasts, content_type


def _describe_model(model_args):
    print(f"Using the follow arguments: [{model_args}]")

    print(f"The model id is [{config.MODEL_ID}]")
    print(f"The MXNet version is [{mxnet.__version__}]")
    print(f"The GluonTS version is [{gluonts.__version__}]")
    print(f"The GPU count is [{mxnet.context.num_gpus()}]")
    print(f"{feature_list()}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=config.HYPER_PARAMETERS["epochs"])
    parser.add_argument('--batch_size', type=int, default=config.HYPER_PARAMETERS["batch_size"])
    parser.add_argument('--prediction_length', type=int, default=config.HYPER_PARAMETERS["prediction_length"])
    parser.add_argument('--context_length', type=int, default=config.HYPER_PARAMETERS["context_length"])
    parser.add_argument('--num_layers', type=int, default=config.HYPER_PARAMETERS["num_layers"])
    parser.add_argument('--num_cells', type=int, default=config.HYPER_PARAMETERS["num_cells"])
    parser.add_argument('--dropout_rate', type=float, default=config.HYPER_PARAMETERS["dropout_rate"])
    parser.add_argument('--learning_rate', type=float, default=config.HYPER_PARAMETERS["learning_rate"])
    parser.add_argument('--num_batches_per_epoch', type=float,
                        default=config.HYPER_PARAMETERS["num_batches_per_epoch"]
                        if "num_batches_per_epoch" in config.HYPER_PARAMETERS else None)

    # For CLI use, otherwise ignore as the defaults will handle it
    parser.add_argument('--dataset_dir', type=str)
    parser.add_argument('--model_dir', type=str)

    return parser.parse_args()


if __name__ == '__main__':
    """
    To quickly iterate, you can run this via cli with `python3 -m deepar --dataset_dir ../out/datasets --model_dir ../out/local_cli/model`.
    This assumes that you have a valid dataset, which can be created via the train notebook
    """
    args = parse_args()
    if not args.dataset_dir:
        args.dataset_dir = os.environ['SM_CHANNEL_DATASET']
    if not args.model_dir:
        args.model_dir = os.environ['SM_MODEL_DIR']

    model_output_dir_path = Path(args.model_dir)
    model_output_dir_path.mkdir(parents=True, exist_ok=True)

    train(args)
