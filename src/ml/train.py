import os
from pathlib import Path

from sagemaker import Session
from sagemaker.estimator import EstimatorBase
from sagemaker.mxnet import MXNet
from sagemaker.debugger import Rule, CollectionConfig, rule_configs
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
            entry_point='lstnet_multi_variate.py',
            source_dir=os.getcwd(),
            role=role,
            instance_type=instance_type,
            instance_count=1,
            framework_version='1.8.0',  # Should be the same mxnet X.X.X version found in requirements.txt
            py_version='py37',
            sagemaker_session=sagemaker_session,
            enable_sagemaker_metrics=True,
            metric_definitions=self._get_metric_definitions(),
            # rules=self._get_debugging_rules(),
            hyperparameters=self._get_hyperparameters(),
            # container_log_level=logging.DEBUG,
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

            'skip_size': config.HYPER_PARAMETERS['skip_size'],
            'ar_window': config.HYPER_PARAMETERS['ar_window'],
            'channels': config.HYPER_PARAMETERS['channels'],
            'rnn_num_layers': config.HYPER_PARAMETERS['rnn_num_layers'],
            'skip_rnn_num_layers': config.HYPER_PARAMETERS['skip_rnn_num_layers'],
            'kernel_size': config.HYPER_PARAMETERS['kernel_size'],

            'dropout_rate': config.HYPER_PARAMETERS['dropout_rate'],
            'learning_rate': config.HYPER_PARAMETERS['learning_rate'],
        }

        if "num_batches_per_epoch" in config.HYPER_PARAMETERS and config.HYPER_PARAMETERS['num_batches_per_epoch']:
            hp['num_batches_per_epoch'] = config.HYPER_PARAMETERS['num_batches_per_epoch']

        return hp

    def _get_debugging_rules(self):
        exploding_tensor_rule = Rule.sagemaker(
            base_config=rule_configs.exploding_tensor(),
            rule_parameters={"collection_names": "weights,losses"},
            collections_to_save=[
                CollectionConfig("weights"),
                CollectionConfig("losses")
            ]
        )

        vanishing_gradient_rule = Rule.sagemaker(
            base_config=rule_configs.vanishing_gradient()
        )

        return [exploding_tensor_rule, vanishing_gradient_rule]

    def _get_metric_definitions(self):
        return [
            {"Name": "train:loss", "Regex": r"Epoch\[\d+\] Evaluation metric 'epoch_loss'=(\S+)"},
            {"Name": "train:learning_rate", "Regex": r"Epoch\[\d+\] Learning rate is (\S+)"},
            {"Name": "test:abs_error", "Regex": r"gluonts\[metric-abs_error\]: (\S+)"},
            {"Name": "test:rmse", "Regex": r"gluonts\[metric-RMSE\]: (\S+)"},
            {"Name": "test:mase", "Regex": r"gluonts\[metric-MASE\]: (\S+)"},
            {"Name": "test:mape", "Regex": r"gluonts\[metric-MAPE\]: (\S+)"},
            {"Name": "test:smape", "Regex": r"gluonts\[metric-sMAPE\]: (\S+)"},
            {"Name": "test:wmape", "Regex": r"gluonts\[metric-wMAPE\]: (\S+)"},
        ]
