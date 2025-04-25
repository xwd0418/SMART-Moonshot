import pathlib
import yaml

DATASET_root_path = pathlib.Path("/workspace/")
curr_exp_folder_name = "initial_experiments"

import logging, os, sys, torch
import random, pickle
import numpy as np
import pytorch_lightning as pl
import pytorch_lightning.callbacks as cb

from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.model_summary import summarize
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from moonshot_models.SMART_BART import SmartBart

from moonshot_dataset.NMR_selfie_dataset import NMR_Selfie_DataModule, SELFIES_MAX_LEN
import argparse
from argparse import ArgumentParser



import warnings
warnings.filterwarnings("ignore", category=FutureWarning, message="You are using `torch.load` with `weights_only=False`")
warnings.filterwarnings("ignore", category=UserWarning, message="The PyTorch API of nested tensors is in prototype stage and will change in the near future.")



def data_mux(args):
    """
        constructs data module based on model_type, and also outputs dimensions of dummy data
        (for graph visualization)
    """
    batch_size = args['bs']
        
    if args['optional_inputs']:
        return OptionalInputDataModule(  input_src=["HSQC", "oneD_NMR"], batch_size=batch_size, p_args=args)
    # OneD datamodule is somewhat buggy, so we will not use hsqc_folder_dataset.py 
    if args['only_oneD_NMR']:
        # here the choice is still SMILES_dataset, but infact, OneDDataset use both datasets
        return OneDDataModule(  batch_size=batch_size, p_args=args) 
   

    if args['use_oneD_NMR_no_solvent']:
        return NMR_Selfie_DataModule(input_src=["HSQC", "oneD_NMR"], batch_size=batch_size, p_args=args)
    else:
        return NMR_Selfie_DataModule(input_src=["HSQC"], batch_size=batch_size, p_args=args)



def init_logger(out_path, path1, path2):
    logger = logging.getLogger("lightning")
    logger.setLevel(logging.DEBUG)
    file_path = os.path.join(out_path, path1, path2, "logs.txt")
    os.makedirs(os.path.join(out_path, path1, path2), exist_ok=True)
    with open(file_path, 'w') as fp: # touch
        pass
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh = logging.FileHandler(file_path)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(logging.StreamHandler(sys.stdout))
            
    return logger

def seed_everything(seed):
    """
    Set the random seed for reproducibility.
    """
    pl.seed_everything(seed,  workers=True)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    np.random.seed(seed)
    random.seed(seed)
    # torch.use_deterministic_algorithms(True)

def add_parser_arguments( parser):
    def str2bool(v):    
        # specifically used for arg-paser with boolean values
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')    
        
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--foldername", type=str, default=f"test_run")
    parser.add_argument("--expname", type=str, default=f"experiment")
    parser.add_argument("--bs", type=int, default=8)
    parser.add_argument("--accumulate_grad_batches_num", type=int, default=4)
    parser.add_argument("--use_small_model", type=lambda x:bool(str2bool(x)), default=False, help="use small models")
        
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--num_workers", type=int, default=4)
    # for early stopping/model saving
    parser.add_argument("--metric", type=str, default="val/mean_cosine_similarity") 
    parser.add_argument("--metricmode", type=str, default="max")

    parser.add_argument("--freeze", type=lambda x:bool(str2bool(x)), default=False)
    parser.add_argument("--validate", type=lambda x:bool(str2bool(x)), default=False)
    parser.add_argument("--test", type=lambda x:bool(str2bool(x)), default=False)
    
    parser.add_argument("--debug", type=lambda x:bool(str2bool(x)), default=False)
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to the checkpoint file to resume training")
    parser.add_argument("--delete_checkpoint", type=lambda x:bool(str2bool(x)), default=False, help="Delete the checkpoint file after training")

    # different versions of input/output
    
    parser.add_argument("--use_oneD_NMR_no_solvent",  type=lambda x:bool(str2bool(x)), default=True, help="use 1D NMR data")
    parser.add_argument("--use_MW",  type=lambda x:bool(str2bool(x)), default=True, help="using mass spectra")
    parser.add_argument("--use_Jaccard",  type=lambda x:bool(str2bool(x)), default=False, help="using Jaccard similarity instead of cosine similarity")
    parser.add_argument("--jittering",  type=float, default=0, help="a data augmentation technique that jitters the peaks. Choose 'normal' or 'uniform' to choose jittering distribution" )
    
    # optional 2D input
    parser.add_argument("--optional_inputs",  type=lambda x:bool(str2bool(x)), default=False, help="use optional 2D input, inference will contain different input versions")
    parser.add_argument("--optional_MW",  type=lambda x:bool(str2bool(x)), default=False, help="also make molecular weight as optional input")
    parser.add_argument("--combine_oneD_only_dataset",  type=lambda x:bool(str2bool(x)), default=False, help="use molecules with only 1D input")
    parser.add_argument("--only_oneD_NMR",  type=lambda x:bool(str2bool(x)), default=False, help="only use oneD NMR, C or H or both. By default is both")
    parser.add_argument("--only_C_NMR",  type=lambda x:bool(str2bool(x)), default=False, help="only use oneD C_NMR. Need to use together with only_oneD_NMR")
    parser.add_argument("--only_H_NMR",  type=lambda x:bool(str2bool(x)), default=False, help="only use oneD H_NMR. Need to use together with only_oneD_NMR")
    parser.add_argument("--separate_classifier",  type=lambda x:bool(str2bool(x)), default=False, help="use separate classifier for various 2D/1D input")
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--train_on_all_info_set", type=lambda x:bool(str2bool(x)), default=False)

    
if __name__ == '__main__':
    torch.set_float32_matmul_precision('medium')

    # dependencies: hyun_fp_data, hyun_pair_ranking_set_07_22
    parser = ArgumentParser(add_help=True)
    add_parser_arguments(parser)
    args = vars(parser.parse_known_args()[0])

    SmartBart.add_model_specific_args(parser, use_small_model = args["use_small_model"])
    
    args = vars(parser.parse_known_args()[0])

    seed_everything(seed=args["random_seed"])   
    
    # general args
    # apply_args(parser, args["modelname"])

    # Model args
    if args['foldername'] == "debug" or args['debug'] is True:
        args['debug'] = True
        args["epochs"] = 1

    # Tensorboard setup
    
    out_path       =       DATASET_root_path / f"Moonshot/{curr_exp_folder_name}"
    # out_path =            f"/root/gurusmart/MorganFP_prediction/reproduce_previous_works/{curr_exp_folder_name}"
    out_path_final =      f"/root/gurusmart/Moonshot/{curr_exp_folder_name}"
    os.makedirs(out_path_final, exist_ok=True)
    os.makedirs(out_path, exist_ok=True)
    path1 = args["foldername"]
    path2 = args["expname"]

    # Logger setup
    my_logger = init_logger(out_path, path1, path2)
    
    my_logger.info(f'[Main] Output Path: {out_path}/{path1}/{path2}')
    try:
        my_logger.info(f'[Main] using GPU : {torch.cuda.get_device_name()}')
    except:
        my_logger.info(f'[Main] using GPU: unknown type')
    

    # Trainer, callbacks
    tbl = TensorBoardLogger(save_dir=out_path, name=path1, version=path2)
    metric, metricmode, patience = args["metric"], args["metricmode"], args["patience"]
    if args['optional_inputs']:
        checkpoint_callback = cb.ModelCheckpoint(monitor=f'{args["metric"].replace("/", "_")}/only_hsqc', mode=metricmode, save_top_k = 1, save_last=False)

    else:
        checkpoint_callback = cb.ModelCheckpoint(monitor=metric, mode=metricmode, save_last=False, save_top_k = 1)
        
    early_stop_metric = f'{args["metric"].replace("/", "_")}/only_hsqc' if args['optional_inputs'] else args["metric"]
    early_stopping = EarlyStopping(monitor=early_stop_metric, mode=metricmode, patience=patience)
    lr_monitor = cb.LearningRateMonitor(logging_interval="step")
    trainer = pl.Trainer(
                         max_epochs=args["epochs"],
                         accelerator="auto",
                         logger=tbl, 
                         callbacks=[early_stopping, lr_monitor, checkpoint_callback],
                        #  strategy="fsdp" if torch.cuda.device_count() > 1 else "auto",
                         accumulate_grad_batches=args["accumulate_grad_batches_num"],
                        )
    
    # Model and Data setup
    data_module = data_mux(args)
    if args['optional_inputs']:
        model_class = OptionalInputRankedTransformer
    else:
        model_class = SmartBart
    model = model_class(
        selfie_symbol_to_idx = data_module.symbol_to_idx,
        selfie_max_len = SELFIES_MAX_LEN,
        p_args = args,
    )
    
    if trainer.global_rank == 0:
        my_logger.info(f"[Main] Model Summary: {summarize(model)}")
    
    
    if args["validate"]:
        my_logger.info("[Main] Just performing validation step")
        trainer.validate(model, data_module, )
    else:
            # training
            my_logger.info("[Main] Begin Training!")
            try :
                trainer.fit(model, data_module,ckpt_path=args["checkpoint_path"])

                # Ensure all processes synchronize before switching to test mode
                trainer.strategy.barrier()

                # Now, only rank 0 will proceed to test
                if trainer.global_rank == 0:
                    
                    # testing
                    model.logger_should_sync_dist = False
                    
                    # my_logger.info(f"[Main] my process rank: {os.getpid()}")
                    trainer = pl.Trainer( devices = 1, accumulate_grad_batches=args["accumulate_grad_batches_num"])
                    my_logger.info(f"[Main] Validation metric {checkpoint_callback.monitor}, best score: {checkpoint_callback.best_model_score.item()}")
                    my_logger.info(f"[Main] Testing path {checkpoint_callback.best_model_path}!")
                    all_test_results = [{}]
                    test_result = trainer.test(model, data_module,ckpt_path=checkpoint_callback.best_model_path)
                    test_result[0]['best_epoch'] = checkpoint_callback.best_model_path.split("/")[-1].split("-")[0]
                    if not args['optional_inputs']:
                    
                        NP_classwise_accu = {k.split("/")[-1]:v for k,v in test_result[0].items() if "rank_1_of_NP_class" in k}
                        img_path = pathlib.Path(checkpoint_callback.best_model_path).parents[1] / f"NP_class_accu.png"
                        all_test_results = test_result
                    else:
                        # loader_all_inputs, loader_HSQC_H_NMR, loader_HSQC_C_NMR, loader_only_hsqc, loader_only_1d, loader_only_H_NMR, loader_only_C_NMR
                        for loader_idx, NMR_type in enumerate(["all_inputs", "HSQC_H_NMR", "HSQC_C_NMR", "only_hsqc", "only_1d", "only_H_NMR", "only_C_NMR"]):
                            model.only_test_this_loader(loader_idx=loader_idx)
                            # test/rank_1_of_NP_class/Sesterterpenoids/HSQC_C_NMR
                            NP_classwise_accu = {k.split("/")[-2]:v for k,v in test_result[0].items() if "rank_1_of_NP_class" in k and NMR_type in k}
                            img_path = pathlib.Path(checkpoint_callback.best_model_path).parents[1] / f"NP_class_accu_{NMR_type}.png"
                        all_test_results = test_result
        
        
                    with open(f"{out_path}/{path1}/{path2}/test_result.pkl", "wb") as f:
                        pickle.dump(all_test_results, f)
                        
                    
        
                    my_logger.info("[Main] Done!")

                    for key, value in all_test_results[0].items():
                        my_logger.info(f"{key}: {value}")
                    if args['delete_checkpoint']:
                        os.remove(checkpoint_callback.best_model_path)
                        my_logger.info(f"[Main] Deleted checkpoint {checkpoint_callback.best_model_path}")
            except Exception as e:
                if trainer.global_rank == 0:
                    my_logger.error(f"[Main] Exception during training: \n{e}")
            finally:
                if trainer.global_rank == 0:
                    os.system(f"cp -r {out_path}/* {out_path_final}/ ")
                    my_logger.info(f"[Main] Copied all content from {out_path} to {out_path_final}")
                    logging.shutdown()



    
