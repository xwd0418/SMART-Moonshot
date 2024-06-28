import yaml
import pathlib
import os

def load_single_config(path):
  if not os.path.exists(path):
    raise Exception(f"Config {path} does not exist")
  with open(path, "r") as f:
    obj = yaml.safe_load(f)
  if "manifest" not in obj or "args" not in obj:
    raise Exception(f"Config {path} missing manifest or args key")
  if obj["manifest"]["type"] != "single":
    raise Exception(f"Config {path} is not single")
  return obj["args"]
