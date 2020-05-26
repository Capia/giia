# This is a logging tool that runs as a background threaded process. This is because when we close our Jupyter
# notebook but leave it running to train models, anything printed is not saved. So instead of printing any debug
# info, we log() it instead, and it will go to a log file. This is useful when running training over a weekend for
# example.

import logging
import threading
import datetime


class LoggerUtil:
    logger = None
    threaded_logging = None
    model_name = None

    def __init__(self, model_name: str):
        # Initialize static variables
        if not LoggerUtil.logger:
            LoggerUtil.logger = logging.getLogger()
        if not LoggerUtil.threaded_logging:
            LoggerUtil.threaded_logging = LoggerUtil.__start_threaded_logging(self)

        self.model_name = model_name

    def __start_threaded_logging(self):
        threaded_logging = threading.Thread(target=self.__setup_file_logger)
        threaded_logging.start()
        threaded_logging.join()
        self.log("Background logger started")
        return threaded_logging

    def __setup_file_logger(self):
        log_file = "{}-{}.log".format(self.model_name, str(datetime.datetime.now()))
        hdlr = logging.FileHandler(log_file)
        formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
        hdlr.setFormatter(formatter)
        self.logger.addHandler(hdlr)
        self.logger.setLevel(logging.INFO)

    def log(self, message, type='info'):
        # outputs to Jupyter console
        print('{} {}'.format(datetime.datetime.now(), message))
        # outputs to file
        if type == 'info':
            self.logger.info(message)
        elif type == 'warning':
            self.logger.warning(message)
        elif type == 'error':
            self.logger.error(message)
        elif type == 'critical':
            self.logger.critical(message)
