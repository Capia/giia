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
from gluonts.model.deepar import DeepAREstimator
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.evaluation import Evaluator
from gluonts.model.predictor import Predictor
from gluonts.model.forecast import Config, Forecast
from gluonts.dataset.common import DataEntry, ListDataset
from gluonts.mx.trainer import Trainer

from data_processing.parse import Parse


# Creates a training and testing ListDataset, a DeepAR estimator, and performs the training. It also performs
# evaluation and prints the MSE metric. This is necessary for the hyperparameter tuning later on.

# TODO: Should use https://gist.github.com/ehsanmok/b2c8fa6dbeea55860049414a16ddb3ff#file-lstnet-py-L41
def train(epochs, prediction_length, num_layers, dropout_rate):
    dataset_dir_path = Path(os.environ['SM_CHANNEL_DATASET'])
    train_dataset_path = dataset_dir_path / Parse.TRAIN_DATASET_FILENAME
    test_dataset_filename = dataset_dir_path / Parse.TEST_DATASET_FILENAME

    # Create train dataset
    df = pd.read_csv(filepath_or_buffer=train_dataset_path, header=0, index_col=0)
    describe_df(df, train_dataset_path)

    training_data = ListDataset(
        [{"start": df.index[0], "target": df['Adj Close'][:]}],
        freq="1d"
    )

    # Define DeepAR estimator
    estimator = DeepAREstimator(
        freq="1d",
        prediction_length=prediction_length,
        dropout_rate=dropout_rate,
        num_layers=num_layers,
        trainer=Trainer(epochs=epochs)
    )

    # Train the model
    predictor = estimator.train(training_data=training_data)

    # Create test dataset
    df = pd.read_csv(filepath_or_buffer=test_dataset_filename, header=0, index_col=0)
    describe_df(df, test_dataset_filename)

    test_data = ListDataset(
        [{"start": df.index[0], "target": df['Adj Close'][:]}],
        freq="1d"
    )

    # Evaluate trained model on test data
    forecast_it, ts_it = make_evaluation_predictions(
        dataset=test_data,  # test dataset
        predictor=predictor,  # predictor
        num_samples=100,  # number of sample paths we want for evaluation
    )

    forecasts = list(forecast_it)
    tss = list(ts_it)

    evaluator = Evaluator(quantiles=[0.1, 0.5, 0.9])
    agg_metrics, item_metrics = evaluator(iter(tss), iter(forecasts), num_series=len(test_data))

    # Required for hyperparameter training
    print("MSE:", agg_metrics["MSE"])

    # Save the model
    predictor.serialize(Path(os.environ['SM_MODEL_DIR']))

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


def describe_df(df, dataset_channel_file: str):
    print(f"First {dataset_channel_file} sample:")
    print(df.head(1))
    print(f"\nLast {dataset_channel_file} sample:")
    print(df.tail(1))
    print(df.describe())


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--prediction_length', type=int, default=12)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--dropout_rate', type=float, default=0.1)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train(args.epochs, args.prediction_length, args.num_layers, args.dropout_rate)
