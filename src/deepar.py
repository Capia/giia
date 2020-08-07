#
# NOTE: This file must stay at the root of the `./src` directory due to sagemaker local stripping the path from the
#  entry_point. Follow this issue for new developments https://github.com/aws/sagemaker-python-sdk/issues/1597
#

import os

import pandas as pd
import pathlib
import argparse
import json
from gluonts.model.deepar import DeepAREstimator
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.evaluation import Evaluator
from gluonts.model.predictor import Predictor
from gluonts.dataset.common import ListDataset
from gluonts.trainer import Trainer


# Creates a training and testing ListDataset, a DeepAR estimator, and performs the training. It also performs
# evaluation and prints the MSE metric. This is necessary for the hyperparameter tuning later on.

# TODO: Should use https://gist.github.com/ehsanmok/b2c8fa6dbeea55860049414a16ddb3ff#file-lstnet-py-L41
def train(epochs, prediction_length, num_layers, dropout_rate):
    # Create train dataset
    df = pd.read_csv(filepath_or_buffer=os.environ['SM_CHANNEL_TRAIN'] + "/train.csv", header=0, index_col=0)
    print("First train sample:")
    print(df.head(1))
    print("\nLast train sample:")
    print(df.tail(1))

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
        trainer=Trainer(epochs=epochs),
        # distr_output=MultivariateGaussianOutput(dim=2)
    )

    # Train the model
    predictor = estimator.train(training_data=training_data)

    # Create test dataset
    df = pd.read_csv(filepath_or_buffer=os.environ['SM_CHANNEL_TEST'] + "/test.csv", header=0, index_col=0)
    df.describe()

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
    predictor.serialize(pathlib.Path(os.environ['SM_MODEL_DIR']))

    return predictor


# Used for inference. Once the model is trained, we can deploy it and this function will load the trained model.
def model_fn(model_dir):
    path = pathlib.Path(model_dir)
    predictor = Predictor.deserialize(path)

    return predictor


# Used for inference. If we send requests to the endpoint, the data will by default be encoded as json string. We decode
# the data from json into a Pandas data frame. We then create the ListDataset and perform inference. The forecasts will
# be sent back as a json object.
def transform_fn(model, data, content_type, output_content_type):
    data = json.loads(data)
    df = pd.DataFrame(data)

    test_data = ListDataset([{"start": df.index[0],
                              "target": df.value[:]}],
                            freq="5min")

    forecast_it, ts_it = make_evaluation_predictions(test_data, model, num_samples=100)
    agg_metrics, item_metrics = Evaluator()(ts_it, forecast_it, num_series=len(test_data))
    response_body = json.dumps(agg_metrics)
    response_body = json.dumps({'predictions': list(forecast_it)[0].samples.tolist()[0]})
    return response_body, output_content_type


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
