import logging
import os
import sys
import socket
from pathlib import Path
from utils.constants import LIGHTNING_LOGGER

def init_logger(out_path, path1, path2):
  logger = logging.getLogger(LIGHTNING_LOGGER)
  # print(f"Handlers: {logger.handlers}")
  for handler in logger.handlers:
    logger.removeHandler(handler)
    # print("Removed handler")
  logger.setLevel(logging.INFO)

  logfile_dir = Path(out_path) / path1 / path2
  logfile_path = logfile_dir / "logs.txt"
  os.makedirs(logfile_dir, exist_ok=True)
  with open(logfile_path, 'a') as fp:  # touch
    pass

  host_name = socket.gethostname()
  f_string = '[' + host_name + \
      '-%(process)d] [%(asctime)s]-[%(filename)s:%(lineno)s]-%(message)s'
  formatter_fh = logging.Formatter(f_string)
  fh = logging.FileHandler(logfile_path, mode='a')
  fh.setFormatter(formatter_fh)
  logger.addHandler(fh)
  formatter_sh = logging.Formatter(f_string)
  sh = logging.StreamHandler(sys.stdout)
  sh.setFormatter(formatter_sh)
  logger.addHandler(sh)
  return logger

def get_logger():
  return logging.getLogger(LIGHTNING_LOGGER)
