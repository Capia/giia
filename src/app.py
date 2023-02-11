# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.
import json
import logging
import multiprocessing as mp
import os
import signal
import time
import traceback
from queue import Empty as QueueEmpty
from typing import Callable, Iterable, List, NamedTuple, Set, Tuple
from typing_extensions import Literal

from gluonts.dataset.common import ListDataset
from gluonts.dataset.jsonl import encode_json
from gluonts.model.forecast import Forecast, Quantile
from gluonts.shell.util import forecaster_type_by_name

OutputType = Literal["mean", "samples", "quantiles"]


class ForecastConfig(BaseModel):
    num_samples: int = Field(100, alias="num_eval_samples")
    output_types: Set[OutputType] = {"quantiles", "mean"}
    # FIXME: validate list elements
    quantiles: List[str] = ["0.1", "0.5", "0.9"]

    class Config:
        allow_population_by_field_name = True
        # store additional fields
        extra = "allow"

    def as_json_dict(self, forecast: Forecast) -> dict:
        result = {}

        if "mean" in self.output_types:
            result["mean"] = forecast.mean.tolist()

        if "quantiles" in self.output_types:
            quantiles = map(Quantile.parse, self.quantiles)

            result["quantiles"] = {
                quantile.name: forecast.quantile(quantile.value).tolist()
                for quantile in quantiles
            }

        if "samples" in self.output_types:
            samples = getattr(forecast, "samples", None)
            if samples is not None:
                samples = samples.tolist()

            result["samples"] = samples

            valid_length = getattr(forecast, "valid_length", None)
            if valid_length is not None:
                result["valid_length"] = valid_length.tolist()

        return result


class InferenceRequest(BaseModel):
    instances: list
    configuration: ForecastConfig


class ThrougputIter:
    def __init__(self, iterable: Iterable) -> None:
        self.iter = iter(iterable)
        self.timings: List[float] = []

    def __iter__(self):
        try:
            while True:
                start = time.time()
                element = next(self.iter)
                self.timings.append(time.time() - start)
                yield element
        except StopIteration:
            return None


def log_throughput(instances, timings):
    item_lengths = [len(item["target"]) for item in instances]

    if timings:
        total_time = sum(timings)
        avg_time = total_time / len(timings)
        print(
            "Inference took "
            f"{total_time:.2f}s for {len(timings)} items, "
            f"{avg_time:.2f}s on average."
        )
        for idx, (duration, input_length) in enumerate(
                zip(timings, item_lengths), start=1
        ):
            print(
                f"\t{idx} took -> {duration:.2f}s"
                f" (len(target)=={input_length})."
            )
    else:
        print(
            "No items were provided for inference. No throughput to log."
        )


def handle_predictions(predictor, instances, configuration):
    # create the forecasts
    forecasts = ThrougputIter(
        predictor.predict(
            ListDataset(instances, configuration.freq),
            num_samples=configuration.num_samples,
        )
    )

    predictions = [
        configuration.as_json_dict(forecast) for forecast in forecasts
    ]

    log_throughput(instances, forecasts.timings)
    return predictions


def inference_invocations(predictor_factory) -> Callable[[], Response]:
    def invocations() -> Response:
        predictor = predictor_factory(request.json)
        req = InferenceRequest.parse_obj(request.json)

        predictions = handle_predictions(
            predictor, req.instances, req.configuration
        )
        return jsonify(predictions=encode_json(predictions))

    return invocations


def make_predictions(predictor, dataset, configuration):
    DEBUG = configuration.dict().get("DEBUG")

    # we have to take this as the initial start-time since the first
    # forecast is produced before the loop in predictor.predict
    start = time.time()

    predictions = []

    forecast_iter = predictor.predict(
        dataset,
        num_samples=configuration.num_samples,
    )

    for forecast in forecast_iter:
        end = time.time()
        prediction = configuration.as_json_dict(forecast)

        if DEBUG:
            prediction["debug"] = {"timing": end - start}

        predictions.append(prediction)

        start = time.time()

    return predictions


class ScoredInstanceStat(NamedTuple):
    amount: int
    duration: float


def batch_inference_invocations(
        predictor_factory, configuration, settings
) -> Callable[[], Response]:
    predictor = predictor_factory({"configuration": configuration.dict()})

    scored_instances: List[ScoredInstanceStat] = []
    last_scored = [time.time()]

    def log_scored(when):
        N = 60
        diff = when - last_scored[0]
        if diff > N:
            scored_amount = sum(info.amount for info in scored_instances)
            time_used = sum(info.duration for info in scored_instances)

            print(
                f"Worker pid={os.getpid()}: scored {scored_amount} using on "
                f"avg {round(time_used / scored_amount, 1)} s/ts over the "
                f"last {round(diff)} seconds."
            )
            scored_instances.clear()
            last_scored[0] = time.time()

    def invocations() -> Response:
        request_data = request.data.decode("utf8").strip()

        # request_data can be empty, but .split() will produce a non-empty
        # list, which then means we try to decode an empty string, which
        # causes an error: `''.split() == ['']`
        if request_data:
            instances = list(map(json.loads, request_data.split("\n")))
        else:
            instances = []

        dataset = ListDataset(instances, configuration.freq)

        start_time = time.time()

        if settings.gluonts_batch_timeout > 0:
            predictions = with_timeout(
                make_predictions,
                args=(predictor, dataset, configuration),
                timeout=settings.gluonts_batch_timeout,
            )

            # predictions are None, when predictor timed out
            if predictions is None:
                print(f"predictor timed out for: {request_data}")
                FallbackPredictor = forecaster_type_by_name(
                    settings.gluonts_batch_fallback_predictor
                )
                fallback_predictor = FallbackPredictor(
                    prediction_length=predictor.prediction_length,
                )

                predictions = make_predictions(
                    fallback_predictor, dataset, configuration
                )
        else:
            predictions = make_predictions(predictor, dataset, configuration)

        end_time = time.time()

        scored_instances.append(
            ScoredInstanceStat(
                amount=len(predictions), duration=end_time - start_time
            )
        )

        log_scored(when=end_time)

        for forward_field in settings.gluonts_forward_fields:
            for input_item, prediction in zip(dataset, predictions):
                prediction[forward_field] = input_item.get(forward_field)

        lines = list(map(json.dumps, map(encode_json, predictions)))
        return Response("\n".join(lines), mimetype="application/jsonlines")

    def invocations_error_wrapper() -> Response:
        try:
            return invocations()
        except Exception:
            return Response(
                json.dumps({"error": traceback.format_exc()}),
                mimetype="application/jsonlines",
            )

    if settings.gluonts_batch_suppress_errors:
        return invocations_error_wrapper
    else:
        return invocations


# handler function to support lambda
def handler(event, context):
    model_dir = os.environ['SM_MODEL_DIR']
    predictor = model_fn(model_dir)

    body = json.loads(event['body'])

    if batch_transform_config is not None:
        invocations_fn = batch_inference_invocations(
            predictor_factory, batch_transform_config, settings
        )
    else:
        invocations_fn = inference_invocations(predictor_factory)
    return transform_fn(predictor, body, event['Content-Type'], event['Accept-Type'])


# Used for inference. Once the model is trained, we can deploy it and this function will load the trained model. No-op
# implementation as default will properly handle decompressing and deserializing the model
def model_fn(model_dir):
    model_dir_path = Path(model_dir) / "model"
    print(f"Model dir [{str(model_dir_path)}]")

    predictor = Predictor.deserialize(model_dir_path)
    print(f"Predictor metadata [{predictor.__dict__}]")

    return predictor


# Used for inference. This is the entry point for sending a request to receive a prediction
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
    dataset = gh.df_to_univariate_dataset(input_df, freq=model.freq)
    print(f"Dataset stats: {calculate_dataset_statistics(dataset)}")

    print(f"Starting prediction...")
    forecast_it = model.predict(dataset, num_samples=num_samples)
    print(f"Finished prediction")

    return list(forecast_it)


# Because of transform_fn(), we cannot use output_fn() as function name.
# Hence, we prefix our helper function with an underscore.
def _output_fn(
        forecasts: List[Forecast],
        content_type: str = "application/json",
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

    json_forecasts = json.dumps([ForecastConfig.as_json_dict(forecast) for forecast in forecasts])
    return json_forecasts, content_type
