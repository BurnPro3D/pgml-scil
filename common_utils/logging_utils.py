import os
import sys
import logging

_format = "[%(asctime)s][%(name)s][%(levelname)s] - %(message)s"

def configure_logger(run_timestamp, log_level=logging.INFO, log_dir=None):
    if not log_dir:
        log_dir = f"../logs/run_{run_timestamp}"
        os.makedirs(log_dir, exist_ok=True)
    log_filepath = os.path.join(log_dir, "training.log")

    # logging.basicConfig(
    #     format=_format, 
    #     level=log_level, 
    #     datefmt="%Y-%m-%d %H:%M:%S",  # Up to seconds
    #     handlers=[
    #         logging.FileHandler(log_filepath, mode="w"),
    #         logging.StreamHandler()  # Still prints to terminal
    #     ]
    # )
    # Get the root logger instead of using basicConfig
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Clear any existing handlers to ensure a clean setup
    # This is the most important step.
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    # Create a formatter
    log_format = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Create and add the file handler
    file_handler = logging.FileHandler(log_filepath, mode="w")
    file_handler.setFormatter(log_format)
    root_logger.addHandler(file_handler)

    # Create and add the stream handler (for console output)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(log_format)
    root_logger.addHandler(stream_handler)

    return log_filepath

def log_to_file(logger_name=None, log_level=logging.INFO, log_filename='tensorflow.log'):
    if not os.path.exists(os.path.dirname(log_filename)):
        os.makedirs(os.path.dirname(log_filename))

    log = logging.getLogger(logger_name) if logger_name else logging.getLogger()
    fh = logging.FileHandler(log_filename)
    fh.setLevel(log_level)
    fh.setFormatter(logging.Formatter(_format))
    log.addHandler(fh)


def log_versions():
    import torch
    import subprocess

    logging.info('--------------- Versions ---------------')
    logging.info('git branch: ' + str(subprocess.check_output(['git', 'branch']).strip()))
    logging.info('git hash: ' + str(subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip()))
    logging.info('Torch: ' + str(torch.__version__))
    logging.info('----------------------------------------')
