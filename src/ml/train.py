import os
from pathlib import Path

from sagemaker import Session
from sagemaker.estimator import EstimatorBase
from sagemaker.mxnet import MXNet
from urllib.parse import urlparse, unquote

from utils.logger_util import LoggerUtil


class Train:
    logger = None
    training_job_name = None
    model_dir_path = None

    def __init__(self, logger: LoggerUtil):
        self.logger = logger

    def create_model(self, role, instance_type: str, sagemaker_session: Session):
        # TODO: Learning rate?
        # Hyperparamters determined by tuning job
        # https://console.aws.amazon.com/sagemaker/home?region=us-east-1#/hyper-tuning-jobs/mxnet-training-200414-0823?region=us-east-1&tab=bestTrainingJob
        estimator = MXNet(
            entry_point='deepar.py',
            source_dir=os.getcwd(),
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
                'dropout_rate': 0.02
            })
        return estimator

    def fit_model(self, estimator: EstimatorBase, dataset_dir_uri: str):
        estimator.fit({"dataset": dataset_dir_uri})
        self.logger.log("Training job name: " + estimator.latest_training_job.job_name)
        self.model_dir_path = Path(unquote(urlparse(estimator.model_data).path)).parent.parent
        self.logger.log("Model is save in: " + str(self.model_dir_path))
