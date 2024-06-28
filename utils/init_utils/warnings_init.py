import warnings

def clear_warnings():
  warnings.filterwarnings("ignore", ".*Found keys.*")
