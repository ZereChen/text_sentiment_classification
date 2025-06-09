import logging
import os
import time
from logging.handlers import RotatingFileHandler

LOG_DIR = "logs"
LOG_NAME = "app-" + str(time.time()) + ".log"
os.makedirs(LOG_DIR, exist_ok=True)


class LoggerManager:
    _logger = None

    @staticmethod
    def get_logger(name: str = None):
        if LoggerManager._logger is not None:
            return LoggerManager._logger

        # 创建 logger 对象
        if name is None:
            name = ""
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)

        # 防止重复添加 handler（多次调用时避免重复）
        if not logger.handlers:
            # 创建文件 handler，使用日志轮转机制
            file_handler = RotatingFileHandler(
                os.path.join(LOG_DIR, LOG_NAME),
                maxBytes=1024 * 1024 * 50,  # 每个日志文件最大 50MB
                backupCount=5,  # 最多保留 5 个备份文件
                encoding="utf-8"
            )
            file_handler.setLevel(logging.DEBUG)

            # 创建控制台 handler（可选）
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)

            # 定义日志格式
            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)

            # 添加 handler 到 logger
            logger.addHandler(file_handler)
            logger.addHandler(console_handler)

        LoggerManager._logger = logger
        return logger
