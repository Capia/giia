import os

from sagemaker import Session
from sagemaker.estimator import EstimatorBase
from sagemaker.mxnet import MXNet

from utils.logging import LoggerUtil


class Train:
    logger = None

    def __init__(self, logger: LoggerUtil):
        self.logger = logger

    def create_model(self, role, instance_type: str, sagemaker_session: Session):
        # TODO: Learning rate?
        # Hyperparamters determined by tuning job
        # https://console.aws.amazon.com/sagemaker/home?region=us-east-1#/hyper-tuning-jobs/mxnet-training-200414-0823?region=us-east-1&tab=bestTrainingJob
        estimator = MXNet(
            entry_point='ml/train.py',
            # This was only tested locally, determine if this works when running in AWS
            source_dir=os.getcwd(),
            role=role,
            train_instance_type=instance_type,
            train_instance_count=1,
            framework_version='1.6.0', py_version='py3',
            sagemaker_session=sagemaker_session,
            hyperparameters={
                'epochs': 6,
                'prediction_length': 13,
                'num_layers': 4,
                'dropout_rate': 0.02,
            })
        return estimator

    def fit_model(self, estimator: EstimatorBase, train_dataset_path: str, test_dataset_path: str):
        estimator.fit({"train": train_dataset_path, "test": test_dataset_path})
