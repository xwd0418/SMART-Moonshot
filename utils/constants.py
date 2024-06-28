from datetime import datetime
from pytz import timezone

LIGHTNING_LOGGER = "pytorch_lightning"


# when logging hparam locals, ignore these keys
MODEL_LOGGING_IGNORE = ["kwargs", "__class__", "self"]

# ignore when generating hyperparmaeter strings
ALWAYS_EXCLUDE = ["modelname", "debug", "expname", "foldername", "datasrc", "patience", "ds", "metric", "validate"] + \
    ["hsqc_weights", "ms_weights", "freeze", "load_all_weights"]


# things to exclude when passing var args to model
EXCLUDE_FROM_MODEL_ARGS = [ "load_override", "expname",
                           "foldername", "datasrc",
                           "ds", "freeze", "load_all_weights",
                           "metric", "metricmode", "patience"]



# GUESS
GROUPS = [
    {"lr", "momentum"},  # Learning rate related parameters
    {"batch_size", "epochs"},  # Training size and duration parameters
    # ... other groups
]

def get_curr_time():
  pst = timezone("PST8PDT")
  california_time = datetime.now(pst)
  return california_time.strftime("%m_%d_%Y_%H:%M")
