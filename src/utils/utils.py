import mxnet
from utils.logging import LoggerUtil


class Utils:
    logger = None

    def __init__(self, logger: LoggerUtil):
        self.logger = logger

    def describe_env(self):
        self.logger.log(mxnet.__version__)
        gpu_count = mxnet.context.num_gpus()
        self.logger.log(f"The GPU count is [{gpu_count}]")
