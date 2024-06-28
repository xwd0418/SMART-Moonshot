import logging
import os
from argparse import ArgumentParser
from collections import namedtuple

from datasets.generic_index_dataset import GenericIndexedModule
from datasets.identity_dataset import IdentityModule
from datasets.dataset_utils import pad, pad_and_mask, tokenise_and_mask, tokenise_and_mask_encoder

from pysmilesutils.augment import SMILESAugmenter
from models.chemformer.utils import REGEX
from models.chemformer.tokeniser import MolEncTokeniser
from utils.constants import LIGHTNING_LOGGER

Context = namedtuple("Context", ["token_file", "smiles_augment"])

def apply_args(parser: ArgumentParser):
  parser.add_argument("--feats", action="store",
                      type=str, nargs="*", default=["HSQC", "R2-6144FP"])
  parser.add_argument("--feats_handlers", action="store",
                      type=str, nargs="*", default=["pad", "None"])
  parser.add_argument("--ds_path", type=str, default="tempdata/SMILES_dataset")
  parser.add_argument("--token_file", type=str,
                      default="tempdata/chemformer/bart_vocab.txt")
  parser.add_argument("--smiles_augment", type=bool, default=True)
  parser.add_argument("--num_workers", type=int, default=4)
def map_to_handler(k, ctx: Context):
  if k == "pad":
    return pad
  if k == "pad_and_mask":
    return pad_and_mask
  if k == "None":
    return None
  if k == "tokenise":
    tokeniser = MolEncTokeniser.from_vocab_file(
        ctx.token_file, REGEX, 272
    )
    if ctx.smiles_augment == True:
      aug = SMILESAugmenter()  # random enumeration
    else:
      aug = (lambda x: x)

    def tokenise_fn(smiles):
      return tokenise_and_mask(aug(smiles), tokeniser)
    return tokenise_fn

  if k == "tokenise_and_mask_encoder":
    tokeniser = MolEncTokeniser.from_vocab_file(
        ctx.token_file, REGEX, 272
    )
    if ctx.smiles_augment == True:
      aug = SMILESAugmenter()  # random enumeration
    else:
      aug = (lambda x: x)  # random enumeration

    def tokenise_fn2(smiles):
      return tokenise_and_mask_encoder(aug(smiles), tokeniser)
    return tokenise_fn2

def data_mux(args, module_type,
             features=None, feature_handlers=None, ds_path=None,
             batch_size=32, len_override=None,
             num_workers=4):
  """
      constructs data module based on model_type, and also outputs dimensions of dummy data
      (for graph visualization)
      parser: the argument parser
      len_override: to set a custom size of the dataset
  """
  logger = logging.getLogger(LIGHTNING_LOGGER)

  if module_type == "gim":
    if "tokenise" in feature_handlers or "tokenise_and_mask_encoder" in feature_handlers:
      assert (args.get("token_file"))
    context = Context(args.get("token_file"), args.get("smiles_augment"))
    feature_handlers = [map_to_handler(f, context) for f in feature_handlers]
    gim = GenericIndexedModule(ds_path, features, feature_handlers,
                               batch_size=batch_size, len_override=len_override,
                               num_workers=num_workers)
    logger.info(f"[data_init] Attempting to use {num_workers} cpus")
    return gim
  elif module_type == "identity":
    return IdentityModule()
