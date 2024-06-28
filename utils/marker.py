import os
from pathlib import Path

def place_marker(path):
  Path(path).touch()

def has_marker(path):
  return os.path.exists(path)

def remove_marker(path):
  if os.path.exists(path):
    os.remove(path)
