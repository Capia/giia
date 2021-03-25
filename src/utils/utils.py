import mxnet

from utils.logger_util import LoggerUtil
from utils import config


class Utils:
    logger = None

    def __init__(self, logger: LoggerUtil):
        self.logger = logger

    def describe_env(self):
        self.logger.log(f"The model id is [{config.MODEL_ID}]")
        self.logger.log(f"The MXNet version is [{mxnet.__version__}]")
        self.logger.log(f"The GPU count is [{mxnet.context.num_gpus()}]")
