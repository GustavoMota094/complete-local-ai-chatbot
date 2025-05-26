import logging
import sys
from chatbot_service.core.configuration.config import settings

log_setup_done = False

def setup_logging():
    """
    Configures the root logger for the application.

    Sets the logging level, format, and handlers (e.g., console).
    This function should be called once, typically at application startup (e.g., in main.py).
    """
    global log_setup_done
    if log_setup_done:
        logging.debug("Logging setup already performed. Skipping.")
        return

    log_level_name = settings.log_level.upper()
    log_level = getattr(logging, log_level_name, logging.INFO)
    if not isinstance(log_level, int):
        log_level = logging.INFO
        logging.warning(f"Invalid LOG_LEVEL '{settings.log_level}'. Defaulting to INFO.")

    # Define standard log format and date format
    log_format = "%(asctime)s - %(levelname)s - [%(name)s] - %(module)s.%(funcName)s:%(lineno)d - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    formatter = logging.Formatter(log_format, datefmt=date_format)

    # Configure the root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    # --- Console Handler ---
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    logging.info(f"Console logging configured with level {logging.getLevelName(log_level)}")

    # --- Optional: File Handler ---
    # try:
    #     log_file_path = getattr(settings, 'log_file_path', 'chatbot_service.log') # Default filename
    #     # Use RotatingFileHandler for production to manage log file sizes
    #     # from logging.handlers import RotatingFileHandler
    #     # file_handler = RotatingFileHandler(log_file_path, maxBytes=10*1024*1024, backupCount=5, encoding='utf-8') # 10MB per file, 5 backups
    #     file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
    #     file_handler.setFormatter(formatter)
    #     root_logger.addHandler(file_handler)
    #     logging.info(f"File logging configured to '{log_file_path}' with level {logging.getLevelName(log_level)}")
    # except Exception as e:
    #     logging.error(f"Failed to configure file logging to {log_file_path}: {e}", exc_info=True)


    # --- Optional: Reduce verbosity of noisy libraries ---
    # Example: Set specific library loggers to a higher level (e.g., WARNING)
    # logging.getLogger("urllib3").setLevel(logging.WARNING)
    # logging.getLogger("requests").setLevel(logging.WARNING)
    # logging.getLogger("chromadb").setLevel(logging.WARNING) # If ChromaDB is too verbose

    log_setup_done = True
    logging.info(f"Root logger setup complete. Application log level set to {logging.getLevelName(log_level)}.")