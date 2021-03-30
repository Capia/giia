import os
from pathlib import Path

from sagemaker import Session
from sagemaker.estimator import EstimatorBase
from sagemaker.mxnet import MXNet
from urllib.parse import urlparse, unquote

from utils.logger_util import LoggerUtil
from utils import config


class Train:
    logger = None
    training_job_name = None
    model_data_path = None

    def __init__(self, logger: LoggerUtil):
        self.logger = logger

    def create_model(self, role, instance_type: str, sagemaker_session: Session, kwargs):
        estimator = MXNet(
            entry_point='deepar.py',
            source_dir=os.getcwd(),
            role=role,
            train_instance_type=instance_type,
            train_instance_count=1,
            framework_version='1.7.0',  # Should be the same mxnet X.X.X version found in requirements.txt
            py_version='py3',
            sagemaker_session=sagemaker_session,

            # TODO
            # enable_sagemaker_metrics=True,

            # TODO: learning_rate, hidden_channels
            hyperparameters=self._get_hyperparameters(),
            **kwargs
        )

        return estimator

    def fit_model(self, estimator: EstimatorBase, dataset_dir_uri: str):
        estimator.fit({"dataset": dataset_dir_uri})
        self.logger.log("Training job name: " + estimator.latest_training_job.job_name)

        self.model_data_path = Path(unquote(urlparse(estimator.model_data).path))
        self.logger.log("Model is saved in: " + str(self.model_data_path))

    def _get_hyperparameters(self) -> dict:
        hp = {
            'epochs': config.HYPER_PARAMETERS['epochs'],
            'batch_size': config.HYPER_PARAMETERS['batch_size'],
            'context_length': config.HYPER_PARAMETERS['context_length'],
            'prediction_length': config.HYPER_PARAMETERS['prediction_length'],
            'num_layers': config.HYPER_PARAMETERS['num_layers'],
            'dropout_rate': config.HYPER_PARAMETERS['dropout_rate']
        }

        if config.HYPER_PARAMETERS['num_batches_per_epoch']:
            hp['num_batches_per_epoch'] = config.HYPER_PARAMETERS['num_batches_per_epoch']

        return hp
