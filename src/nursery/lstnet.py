#
# NOTE: This file must stay at the root of the `./src` directory due to sagemaker-local stripping the path from the
#  entry_point. Follow this issue for new developments https://github.com/aws/sagemaker-python-sdk/issues/1597
#

import os
import io

import pandas as pd
import argparse
import json
import numpy as np

from typing import List, Tuple, Union
from pathlib import Path

from gluonts.dataset.field_names import FieldName
from gluonts.dataset.multivariate_grouper import MultivariateGrouper
from gluonts.dataset.split import OffsetSplitter
from gluonts.dataset.stat import calculate_dataset_statistics
from gluonts.evaluation.backtest import backtest_metrics, make_evaluation_predictions
from gluonts.evaluation import Evaluator, MultivariateEvaluator
from gluonts.model.deepvar import DeepVAREstimator
from gluonts.model.gpvar import GPVAREstimator
from gluonts.model.lstnet import LSTNetEstimator
from gluonts.model.predictor import Predictor
from gluonts.model.forecast import Config, Forecast
from gluonts.dataset.common import DataEntry, ListDataset
from gluonts.mx.distribution import LogitNormalOutput, PoissonOutput
from gluonts.mx.trainer import Trainer
import mxnet as mx
from mxnet.runtime import feature_list

from utils import config


# Creates a training and testing ListDataset, a DeepAR estimator, and performs the training. It also performs
# evaluation and prints performance metrics
def train(model_args):
    _describe_model(model_args)

    dataset_dir_path = Path(model_args.dataset_dir)
    train_dataset_path = dataset_dir_path / config.TRAIN_DATASET_FILENAME
    test_dataset_filename = dataset_dir_path / config.TEST_DATASET_FILENAME

    # Create train dataset
    train_df = _get_df_from_dataset_file(train_dataset_path)

    # train_dataset, validation_dataset = _vertical_split(
    train_dataset = _vertical_split(
        train_df, model_args.prediction_length)
    # train_statistics = calculate_dataset_statistics(train_dataset)
    # validation_statistics = calculate_dataset_statistics(validation_dataset)
    # print(f"Train dataset stats: {train_statistics}")
    # print(f"Validation dataset stats: {validation_statistics}")

    if not model_args.num_batches_per_epoch:
        model_args.num_batches_per_epoch = len(train_df) // model_args.batch_size
        print(f"Defaulting num_batches_per_epoch to: [{model_args.num_batches_per_epoch}] "
              f"= (length of train dataset [{len(train_df)}]) / (batch size [{model_args.batch_size}])")

    estimator = LSTNetEstimator(
        freq=config.DATASET_FREQ,
        context_length=model_args.context_length,
        prediction_length=model_args.prediction_length,
        # target_dim=4,
        # dropout_rate=model_args.dropout_rate,
        # num_layers=model_args.num_layers,
        # num_cells=model_args.num_cells,
        batch_size=model_args.batch_size,

        num_series=1,
        skip_size=9,
        ar_window=18,
        channels=90,
        rnn_num_layers=90, skip_rnn_num_layers=9,

        # TODO: Determine the correct distribution method. This article goes over some of the key differences
        # https://www.investopedia.com/articles/06/probabilitydistribution.asp
        # distr_output=PoissonOutput(),

        trainer=Trainer(
            epochs=model_args.epochs,
            batch_size=model_args.batch_size,
            num_batches_per_epoch=model_args.num_batches_per_epoch,
            # learning_rate=model_args.learning_rate
        )
    )

    # Train the model
    # TODO: 4 workers for number of vCores, though this should be configurable.
    predictor = estimator.train(
        training_data=train_dataset,
        # validation_data=validation_dataset
    )

    # Create test dataset
    test_df = _get_df_from_dataset_file(test_dataset_filename)

    # test_dataset = ListDataset(
    #     [{
    #         FieldName.START: test_df.index[0],
    #         FieldName.TARGET: [test_df['close'][:], test_df['open'][:], test_df['high'][:], test_df['low'][:]],
    #         FieldName.ITEM_ID: "BTC/USDT",
    #     }],
    #     freq=config.DATASET_FREQ,
    #     one_dim_target=False
    # )

    test_dataset = ListDataset(
        [{
            FieldName.START: test_df.index[0],
            FieldName.TARGET: test_df['close'][:],
            FieldName.FEAT_DYNAMIC_REAL: [test_df['open'][:], test_df['high'][:], test_df['low'][:], test_df['volume'][:]],
            FieldName.ITEM_ID: "BTC/USDT",
        }],
        freq=config.DATASET_FREQ
    )
    grouper_test = MultivariateGrouper(max_target_dim=1)
    test_dataset = grouper_test(test_dataset)

    # Evaluate trained model on test data. This will serialize each of the agg_metrics into a well formatted log.
    # We use this to capture the metrics needed for hyperparameter tuning
    agg_metrics, item_metrics = backtest_metrics(
        test_dataset=test_dataset,
        predictor=predictor,
        evaluator=Evaluator(quantiles=[0.1, 0.5, 0.9]),
        num_samples=100,  # number of samples used in probabilistic evaluation
    )

    # forecast_it, ts_it = make_evaluation_predictions(
    #     test_dataset, predictor=predictor, num_samples=100
    # )
    #
    # evaluator=Evaluator(quantiles=[0.1, 0.5, 0.9])
    # agg_metrics, item_metrics = evaluator(
    #     ts_it, forecast_it, num_series=12
    # )

    # Save the model
    predictor.serialize(Path(model_args.model_dir))

    return predictor


# Used for inference. Once the model is trained, we can deploy it and this function will load the trained model.
def model_fn(model_dir):
    path = Path(model_dir)
    predictor = Predictor.deserialize(path)

    return predictor


# Used for inference. If we send requests to the endpoint, the data will by default be encoded as json string. We decode
# the data from json into a Pandas data frame. We then create the ListDataset and perform inference. The forecasts will
# be sent back as a json object.
def transform_fn(
        model: Predictor,
        request_body: Union[str, bytes],
        content_type: str = "application/json",
        accept_type: str = "application/json",
        num_samples: int = 1000,
) -> Union[bytes, Tuple[bytes, str]]:
    deser_input: List[DataEntry] = _input_fn(request_body, content_type)
    fcast: List[Forecast] = _predict_fn(deser_input, model, num_samples=num_samples)
    ser_output: Union[bytes, Tuple[bytes, str]] = _output_fn(fcast, accept_type)
    return ser_output


# Because of transform_fn(), we cannot use input_fn() as function name
# Hence, we prefix our helper function with an underscore.
def _input_fn(request_body: Union[str, bytes], request_content_type: str = "application/json") -> List[DataEntry]:
    """Deserialize JSON-lines into Python objects.

    Args:
        request_body (str): Incoming payload.
        request_content_type (str, optional): Ignored. Defaults to "".

    Returns:
        List[DataEntry]: List of GluonTS timeseries.
    """
    if isinstance(request_body, bytes):
        request_body = request_body.decode("utf-8")
    return [json.loads(line) for line in io.StringIO(request_body)]


# Because of transform_fn(), we cannot use predict_fn() as function name.
# Hence, we prefix our helper function with an underscore.
def _predict_fn(input_object: List[DataEntry], model: Predictor, num_samples=1000) -> List[Forecast]:
    """Take the deserialized JSON-lines, then perform inference against the loaded model.

    Args:
        input_object (List[DataEntry]): List of GluonTS timeseries.
        model (Predictor): A GluonTS predictor.
        num_samples (int, optional): Number of forecast paths for each timeseries. Defaults to 1000.

    Returns:
        List[Forecast]: List of forecast results.
    """
    # Create ListDataset here, because we need to match their freq with model's freq.
    X = ListDataset(input_object, freq=model.freq)

    it = model.predict(X, num_samples=num_samples)
    return list(it)


# Because of transform_fn(), we cannot use output_fn() as function name.
# Hence, we prefix our helper function with an underscore.
def _output_fn(
        forecasts: List[Forecast],
        content_type: str = "application/json",
        config: Config = Config(quantiles=["0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9"]),
) -> Union[bytes, Tuple[bytes, str]]:
    """Take the prediction result and serializes it according to the response content type.

    Args:
        forecasts (List[Forecast]): List of forecast results.
        content_type (str, optional): Ignored. Defaults to "".

    Returns:
        List[str]: List of JSON-lines, each denotes forecast results in quantiles.
    """

    # jsonify_floats is taken from gluonts/shell/serve/util.py
    #
    # The module depends on flask, and we may not want to import when testing in our own dev env.
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

    str_results = "\n".join((json.dumps(jsonify_floats(forecast.as_json_dict(config))) for forecast in forecasts))
    bytes_results = str.encode(str_results)
    return bytes_results, content_type


def _describe_model(model_args):
    print(f"Using the follow arguments: [{model_args}]")

    print(f"MXNet version [{mx.__version__}]")
    print(f"Number of GPUs available [{mx.context.num_gpus()}]")
    print(f"{feature_list()}")


def _get_df_from_dataset_file(dataset_path: Path):
    df = pd.read_csv(filepath_or_buffer=dataset_path, header=0, index_col=0)

    print(f"First {dataset_path} sample:")
    print(df.head(1))
    print(f"\nLast {dataset_path} sample:")
    print(df.tail(1))
    print(df.describe())

    return df


def _vertical_split(df, offset_from_end):
    """
    Split a dataset time-wise in a train and validation dataset.
    """
    # # This works, but we don't want to predict multiple targets
    # dataset = ListDataset(
    #     [{
    #         FieldName.START: df.index[0],
    #         FieldName.TARGET: [df['close'][:-offset_from_end], df['open'][:-offset_from_end], df['high'][:-offset_from_end], df['low'][:-offset_from_end]],
    #         FieldName.ITEM_ID: "BTC/USDT",
    #     }],
    #     freq=config.DATASET_FREQ,
    #     one_dim_target=False
    # )

    dataset = ListDataset(
        [{
            FieldName.START: df.index[0],
            FieldName.TARGET: df['close'][:],
            FieldName.FEAT_DYNAMIC_REAL: [df['open'][:], df['high'][:], df['low'][:], df['volume'][:]],
            FieldName.ITEM_ID: "BTC/USDT",
        }],
        freq=config.DATASET_FREQ
    )

    # print([
    #     {
    #         FieldName.START: df.index[0],
    #         FieldName.TARGET: df['close'][:-offset_from_end],
    #     },
    #     {
    #         FieldName.START: df.index[0],
    #         FieldName.TARGET: df['open'][:-offset_from_end],
    #     },
    #     {
    #         FieldName.START: df.index[0],
    #         FieldName.TARGET: df['high'][:-offset_from_end],
    #     },
    #     {
    #         FieldName.START: df.index[0],
    #         FieldName.TARGET: df['low'][:-offset_from_end],
    #     },
    # ])
    # dataset = ListDataset(
    #     [
    #         {
    #             FieldName.START: df.index[0],
    #             FieldName.TARGET: df['close'][:-offset_from_end],
    #         },
    #         {
    #             FieldName.START: df.index[0],
    #             FieldName.TARGET: df['open'][:-offset_from_end],
    #         },
    #         {
    #             FieldName.START: df.index[0],
    #             FieldName.TARGET: df['high'][:-offset_from_end],
    #         },
    #         {
    #             FieldName.START: df.index[0],
    #             FieldName.TARGET: df['low'][:-offset_from_end],
    #         },
    #     ],
    #     freq=config.DATASET_FREQ,
    #     # one_dim_target=False
    # )

    # dataset_length = len(next(iter(dataset))["target"])
    #
    # split_offset = dataset_length - offset_from_end
    #
    # splitter = OffsetSplitter(
    #     prediction_length=offset_from_end,
    #     split_offset=split_offset,
    #     max_history=offset_from_end)
    #
    # (_, train_dataset), (_, validation_dataset) = splitter.split(dataset)
    # return train_dataset, validation_dataset

    from gluonts.dataset.multivariate_grouper import MultivariateGrouper
    grouper_train = MultivariateGrouper(max_target_dim=1)

    dataset = grouper_train(dataset)
    return dataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=config.HYPER_PARAMETERS["epochs"])
    parser.add_argument('--batch_size', type=int, default=config.HYPER_PARAMETERS["batch_size"])
    parser.add_argument('--context_length', type=int, default=config.HYPER_PARAMETERS["context_length"])
    parser.add_argument('--prediction_length', type=int, default=config.HYPER_PARAMETERS["prediction_length"])
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
