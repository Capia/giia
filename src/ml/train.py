import os

from sagemaker import Session
from sagemaker.estimator import EstimatorBase
from sagemaker.mxnet import MXNet

from utils.logger_util import LoggerUtil


class Train:
    logger = None
    training_job_name = None

    def __init__(self, logger: LoggerUtil):
        self.logger = logger

    def create_model(self, role, instance_type: str, model_output_uri: str, sagemaker_session: Session):
        # TODO: Learning rate?
        # Hyperparamters determined by tuning job
        # https://console.aws.amazon.com/sagemaker/home?region=us-east-1#/hyper-tuning-jobs/mxnet-training-200414-0823?region=us-east-1&tab=bestTrainingJob
        estimator = MXNet(
            entry_point='deepar.py',
            source_dir=os.getcwd(),
            output_path=model_output_uri,
            role=role,
            train_instance_type=instance_type,
            train_instance_count=1,
            framework_version='1.6.0',  # Should be the same X.X.X version found in requirements.txt
            py_version='py3',
            sagemaker_session=sagemaker_session,
            hyperparameters={
                'epochs': 6,
                'prediction_length': 13,
                'num_layers': 4,
                'dropout_rate': 0.02,
            })
        return estimator

    def fit_model(self, estimator: EstimatorBase, dataset_dir_uri: str):
        estimator.fit({"dataset": dataset_dir_uri})
        self.training_job_name = estimator.latest_training_job.job_name
        self.logger.log("Training job name: " + self.training_job_name)
