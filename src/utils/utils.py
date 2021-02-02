import os

import mxnet

from utils.logger_util import LoggerUtil


class Utils:
    logger = None

    def __init__(self, logger: LoggerUtil):
        self.logger = logger

    def describe_env(self):
        self.logger.log(f"Current working directory [{os.getcwd()}]")
        self.logger.log(f"The MXNet version is [{mxnet.__version__}]")
        self.logger.log(f"The GPU count is [{mxnet.context.num_gpus()}]")
