import gluonts
import mxnet
import sagemaker

from utils.logger_util import LoggerUtil
from utils import config


class Utils:
    logger = None

    def __init__(self, logger: LoggerUtil):
        self.logger = logger

    def describe_env(self):
        self.logger.log(f"The model id is [{config.MODEL_ID}]")
        self.logger.log(f"The MXNet version is [{mxnet.__version__}]")
        self.logger.log(f"The GluonTS version is [{gluonts.__version__}]")
        self.logger.log(f"The SageMaker version is [{sagemaker.__version__}]")
        self.logger.log(f"The GPU count is [{mxnet.context.num_gpus()}]")

    def is_integer_num(self, x):
        if isinstance(x, int):
            return True
        if isinstance(x, float):
            return x.is_integer()
        return False
