from argparse import ArgumentParser

from models.ranked_transformer import HsqcRankedTransformer
from models.identity_module import IdentityModule

def apply_args(parser, model_type):
  if model_type == "hsqc_transformer" or model_type == "ms_transformer":
    HsqcRankedTransformer.add_model_specific_args(parser)
  else:
    raise (f"No model for model type {model_type}.")

def training_args(parser):
  # admin stuff
  parser.add_argument("--config", type=str, default=None,
                      help="config file, or empty to use cli args")
  parser.add_argument("--modelname", type=str)
  parser.add_argument("--force_start", type=bool, default=False)
  parser.add_argument("--actually_run", type=bool, default=True)
  parser.add_argument("--load_path", type=str, default=None)

  # logging args
  parser.add_argument("--foldername", type=str, default=f"lightning_logs")
  parser.add_argument("--expname", type=str, default=f"experiment")

  # training args
  parser.add_argument("--epochs", type=int, default=120)
  parser.add_argument("--bs", type=int, default=64)

  # for early stopping/model saving
  parser.add_argument("--metric", type=str, default="val/mean_ce_loss")
  parser.add_argument("--metricmode", type=str, default="max")
  parser.add_argument("--patience", type=int, default=30)

  # data args
  parser.add_argument("--module_type", type=str, default="gim")
  parser.add_argument("--data_len", type=int, default=None)
  parser.add_argument("--batch_size", type=int, default=32)
