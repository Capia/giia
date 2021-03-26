# This is a logging tool that runs as a background threaded process. This is because when we close our Jupyter
# notebook but leave it running to train models, anything printed is not saved. So instead of printing any debug
# info, we log() it instead, and it will go to a log file. This is useful when running training over a weekend for
# example.

import logging
import threading
import datetime
from pathlib import Path


class LoggerUtil:
    model_name = None
    log_dir = None
    logger = None
    threaded_logger = None

    def __init__(self, model_name: str, log_dir: Path):
        self.model_name = model_name
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Initialize static variables
        if not LoggerUtil.logger:
            LoggerUtil.logger = logging.getLogger()
        if not LoggerUtil.threaded_logger:
            LoggerUtil.threaded_logger = LoggerUtil.__start_threaded_logger(self)

    def __start_threaded_logger(self):
        threaded_logger = threading.Thread(target=LoggerUtil.__setup_file_logger(self))
        threaded_logger.start()
        threaded_logger.join()
        self.log("Background logger started")
        return threaded_logger

    def __setup_file_logger(self):
        log_file = self.log_dir / f"{self.model_name}-{str(datetime.datetime.now())}.log"
        hdlr = logging.FileHandler(log_file)
        formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
        hdlr.setFormatter(formatter)
        self.logger.addHandler(hdlr)
        self.logger.setLevel(logging.INFO)

    def log(self, message, log_level='info', newline=False):
        if newline:
            message = f"\n{message}"

        # Outputs to Jupyter console
        print('{} {}'.format(datetime.datetime.now(), message))

        # Outputs to file
        if log_level == 'info':
            self.logger.info(message)
        elif log_level == 'warning':
            self.logger.warning(message)
        elif log_level == 'error':
            self.logger.error(message)
        elif log_level == 'critical':
            self.logger.critical(message)
