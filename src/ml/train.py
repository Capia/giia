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
    model_dir_path = None

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

            # TODO: Learning rate?
            hyperparameters={
                'epochs': config.HP_EPOCHS,
                'prediction_length': config.HP_PREDICTION_LENGTH,
                'num_layers': config.HP_NUM_LAYERS,
                'dropout_rate': config.HP_DROPOUT_RATE
            },
            kwargs=kwargs
        )
        return estimator

    def fit_model(self, estimator: EstimatorBase, dataset_dir_uri: str):
        estimator.fit({"dataset": dataset_dir_uri})
        self.logger.log("Training job name: " + estimator.latest_training_job.job_name)

        model_data_path = Path(unquote(urlparse(estimator.model_data).path))
        self.logger.log("Full model data path: " + str(model_data_path))

        self.model_dir_path = model_data_path.parent.parent
        self.logger.log("Model is save in: " + str(self.model_dir_path))
