import boto3
import sagemaker
from sagemaker.estimator import EstimatorBase
from sagemaker.tuner import HyperparameterTuner, ContinuousParameter, IntegerParameter
import pandas as pd

from utils.logging import LoggerUtil


class Tune:
    logger = None
    tuning_job_name = None
    is_minimize = None
    objective_name = None

    def __init__(self, logger: LoggerUtil):
        self.logger = logger

    def create_tuner(self, estimator: EstimatorBase):
        tuner = HyperparameterTuner(
            estimator=estimator,
            objective_metric_name='loss',
            hyperparameter_ranges={
                'epochs': IntegerParameter(5, 20),
                'prediction_length': IntegerParameter(5, 20),
                'num_layers': IntegerParameter(1, 5),
                'dropout_rate': ContinuousParameter(0, 0.5)},
            metric_definitions=[{'Name': 'loss', 'Regex': "MSE: ([0-9\\.]+)"}],
            max_jobs=10,
            max_parallel_jobs=5,
            objective_type='Minimize')
        return tuner

    def fit_tuner(self, tuner: HyperparameterTuner, s3_train_data_path: str, s3_test_data_path: str):
        tuner.fit({'train': s3_train_data_path, "test": s3_test_data_path})
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
