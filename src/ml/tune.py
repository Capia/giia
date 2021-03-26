import boto3
import sagemaker
from sagemaker.estimator import EstimatorBase
from sagemaker.tuner import HyperparameterTuner, ContinuousParameter, IntegerParameter
import pandas as pd

from utils.utils import Utils
from utils.logger_util import LoggerUtil
from utils import config


class Tune:
    utils = None
    logger = None
    tuning_job_name = None
    is_minimize = None
    objective_name = None

    def __init__(self, utils: Utils, logger: LoggerUtil):
        self.utils = utils
        self.logger = logger

    def create_tuner(self, estimator: EstimatorBase):
        # Hyperparamters are dynamically computed based on current values. Max and min are based on typical values
        # mentioned here https://docs.aws.amazon.com/sagemaker/latest/dg/deepar_hyperparameters.html
        tuner = HyperparameterTuner(
            estimator=estimator,
            objective_metric_name='loss',
            hyperparameter_ranges={
                'epochs': self._get_range_for_hyperparameter(
                    'epochs', hp_allowed_max=1000),
                'prediction_length': self._get_range_for_hyperparameter(
                    'prediction_length'),
                'num_layers': self._get_range_for_hyperparameter(
                    'num_layers', hp_range=5, hp_allowed_max=10),
                'dropout_rate': self._get_range_for_hyperparameter(
                    'dropout_rate', hp_allowed_min=0.001, hp_allowed_max=0.2),
            },
            metric_definitions=[{'Name': 'loss', 'Regex': "MSE: ([0-9\\.]+)"}],
            max_jobs=15,
            max_parallel_jobs=3,
            objective_type='Minimize')
        return tuner

    def fit_tuner(self, tuner: HyperparameterTuner, dataset_dir_uri: str):
        tuner.fit({"dataset": dataset_dir_uri})
        self.tuning_job_name = tuner.latest_tuning_job.job_name
        self.logger.log("Tuning job name: " + self.tuning_job_name)

    def get_tune_job_update(self):
        sage_client = boto3.Session().client('sagemaker')

        # Run this cell to check current status of hyperparameter tuning job
        tuning_job_result = sage_client.describe_hyper_parameter_tuning_job(HyperParameterTuningJobName=self.tuning_job_name)

        status = tuning_job_result['HyperParameterTuningJobStatus']
        if status != 'Completed':
            self.logger.log('The tuning job has not been completed. Please run again to determine if the job completed '
                            'or visit the AWS SageMaker console to determine the job\'s progress')
            return

        job_count = tuning_job_result['TrainingJobStatusCounters']['Completed']
        self.logger.log("%d training jobs have completed" % job_count)

        self.is_minimize = (tuning_job_result['HyperParameterTuningJobConfig']['HyperParameterTuningJobObjective']['Type'] != 'Maximize')
        self.objective_name = tuning_job_result['HyperParameterTuningJobConfig']['HyperParameterTuningJobObjective']['MetricName']

    def report_job_analytics(self):
        job_analytics = sagemaker.HyperparameterTuningJobAnalytics(self.tuning_job_name)

        full_df = job_analytics.dataframe()

        if len(full_df) > 0:
            df = full_df[full_df['FinalObjectiveValue'] > -float('inf')]
            if len(df) > 0:
                df = df.sort_values('FinalObjectiveValue', ascending=self.is_minimize)
                self.logger.log("Number of training jobs with valid objective: %d" % len(df))
                self.logger.log({"lowest":min(df['FinalObjectiveValue']),"highest": max(df['FinalObjectiveValue'])})
                pd.set_option('display.max_colwidth', -1)  # Don't truncate TrainingJobName
            else:
                self.logger.log("No training jobs have reported valid results yet.")

        return df

    def _get_range_for_hyperparameter(self, hp_key, hp_range=None, hp_allowed_min=None, hp_allowed_max=None):
        """Best attempt to dynamically find a hyperparameter's range based on current tuned value"""
        hp = config.HYPER_PARAMETERS[hp_key]

        # First, set defaults if they have not been provided
        if hp_range is None:
            if self.utils.is_integer_num(hp):
                hp_range = hp // 2
            if isinstance(hp, float):
                hp_range = hp / 2

        if hp_allowed_min is None:
            if self.utils.is_integer_num(hp):
                hp_allowed_min = max(1, hp // 2)
            if isinstance(hp, float):
                hp_allowed_min = max(0.0001, hp / 2)

        if hp_allowed_max is None:
            hp_allowed_max = hp * 2

        # Then determine an appropriate range
        hp_min = max(hp_allowed_min, hp - hp_range)
        hp_max = min(hp_allowed_max, hp + hp_range)
        self.logger.log(f"Hyperparameter [{hp_key}] default value is [{hp}]. The min range will be set to [{hp_min}] "
                        f"and the max range will be set to [{hp_max}]")

        # Lastly, return the appropriate parameter range object
        if self.utils.is_integer_num(hp):
            assert self.utils.is_integer_num(hp_range)
            assert self.utils.is_integer_num(hp_allowed_min)
            assert self.utils.is_integer_num(hp_allowed_max)
            return IntegerParameter(hp_min, hp_max)
        if isinstance(hp, float):
            return ContinuousParameter(hp_min, hp_max)
