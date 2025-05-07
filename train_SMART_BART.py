import pathlib
import yaml

DATASET_root_path = pathlib.Path("/workspace/")
curr_exp_folder_name = "random_smiles_variant"

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
from moonshot_utils.arg_config import str2bool  

from Spectre.utils.interpret_NMR_input_config_args import parse_nmr_input_types
# from Spectre.utils import interpret_NMR_input_config_args


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
    
        
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--foldername", type=str, default=f"test_run")
    parser.add_argument("--expname", type=str, default=f"experiment")
    parser.add_argument("--bs", type=int, default=8)
    parser.add_argument("--accumulate_grad_batches_num", type=int, default=4)
    parser.add_argument("--use_small_model", type=lambda x:bool(str2bool(x)), default=False, help="use small models")
        
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--num_workers", type=int, default=4)
    # for early stopping/model saving
    parser.add_argument("--metric", type=str, default="val/loss") #cosine_similarity
    parser.add_argument("--metricmode", type=str, default="min")

    parser.add_argument("--freeze", type=lambda x:bool(str2bool(x)), default=False)
    parser.add_argument("--train", type=lambda x:bool(str2bool(x)), default=True)
    parser.add_argument("--validate", type=lambda x:bool(str2bool(x)), default=False)
    parser.add_argument("--test", type=lambda x:bool(str2bool(x)), default=False)
    parser.add_argument("--predict", type=lambda x:bool(str2bool(x)), default=False)
    
    
    parser.add_argument("--debug", type=lambda x:bool(str2bool(x)), default=False)
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to the checkpoint file to resume training")
    parser.add_argument("--delete_checkpoint", type=lambda x:bool(str2bool(x)), default=False, help="Delete the checkpoint file after training (to save disk space)")

    # different versions of input/output
    
    parser.add_argument("--use_Jaccard",  type=lambda x:bool(str2bool(x)), default=False, help="using Jaccard similarity instead of cosine similarity")
    parser.add_argument("--jittering",  type=float, default=1.0, help="a data augmentation technique that jitters the peaks. Choose 'normal' or 'uniform' to choose jittering distribution" )

    # control inputs
    parser.add_argument("--optional_inputs",  type=lambda x:bool(str2bool(x)), default=False, help="use optional 2D input, inference will contain different input versions")
    parser.add_argument("--optional_MW",  type=lambda x:bool(str2bool(x)), default=False, help="also make molecular weight as optional input")
    parser.add_argument("--use_HSQC", type=lambda x:bool(str2bool(x)), default=True, help="also make molecular weight as optional input")
    parser.add_argument("--use_H_NMR",  type=lambda x:bool(str2bool(x)), default=True, help="using 1D NMR")
    parser.add_argument("--use_C_NMR",  type=lambda x:bool(str2bool(x)), default=True, help="using 1D NMR")
    parser.add_argument("--use_MW",  type=lambda x:bool(str2bool(x)), default=True, help="using mass spectra")
    parser.add_argument("--train_on_all_info_set", type=lambda x:bool(str2bool(x)), default=False, help="train on subset of training set, where every mol has 3 types of NMR")
    # ## the args above will be used to assign values to the following args
    # parser.add_argument("--use_oneD_NMR_no_solvent",  type=lambda x:bool(str2bool(x)), default=True, help="use 1D NMR data, but not using solvent data")
    # parser.add_argument("--combine_oneD_only_dataset",  type=lambda x:bool(str2bool(x)), default=False, help="will use /workspace/OneD_Only_Dataset/")
    # parser.add_argument("--only_oneD_NMR",  type=lambda x:bool(str2bool(x)), default=False, help="only use oneD NMR, C or H or both. By default is both")
    # parser.add_argument("--only_C_NMR",  type=lambda x:bool(str2bool(x)), default=False, help="only use oneD C_NMR. Need to use together with only_oneD_NMR")
    # parser.add_argument("--only_H_NMR",  type=lambda x:bool(str2bool(x)), default=False, help="only use oneD H_NMR. Need to use together with only_oneD_NMR")

    parser.add_argument("--random_seed", type=int, default=42)

    
def main():
    torch.set_float32_matmul_precision('medium')
    # dependencies: hyun_fp_data, hyun_pair_ranking_set_07_22
    parser = ArgumentParser(add_help=True)
    add_parser_arguments(parser)
    args = vars(parser.parse_known_args()[0])
    
    ckpt_path = args['checkpoint_path']
    if ckpt_path is not None:
        with open( pathlib.Path(args['checkpoint_path']).parent.parent / "hparams.yaml", "r") as f:
            previous_args = yaml.safe_load(f)
            for k in previous_args.keys():
                if k not in args.keys():
                    args[k] = previous_args[k]

    else:
        SmartBart.add_model_specific_args(parser, use_small_model = args["use_small_model"])
        args = vars(parser.parse_known_args()[0])
    args = parse_nmr_input_types(args)


    seed_everything(seed=args["random_seed"])   
    
    # general args
    # apply_args(parser, args["modelname"])

    # Model args
    if args['foldername'] == "debug" or args['debug'] is True:
        args['debug'] = True
        args["epochs"] = 10

    # Tensorboard setup
    
    out_path       =       DATASET_root_path / f"Moonshot/{curr_exp_folder_name}"
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
    
    
    # Model and Data setup
 
    data_module = data_mux(args)
    if args['optional_inputs']:
        model_class = OptionalInputRankedTransformer
    else:
        model_class = SmartBart
    
   
    
    if args["validate"] or args["test"] or args["predict"]:
        assert os.path.exists(args["checkpoint_path"]), f"Checkpoint path {args['checkpoint_path']} does not exist!"
        model = model_class.load_from_checkpoint(checkpoint_path=args["checkpoint_path"], 
                                                 selfie_symbol_to_idx = data_module.symbol_to_idx,
                                                selfie_max_len = SELFIES_MAX_LEN,
                                                p_args = args,
        )
        trainer = pl.Trainer(logger=tbl,  use_distributed_sampler=False)
        
        if args["validate"]:
            my_logger.info("[Main] Just performing validation step")
            model.validate_with_generation = True
            trainer.validate(model, data_module)
                
        if args["test"]:
            
            my_logger.info("[Main] Just performing test step")
            print(f"Checkpoint path: {args['checkpoint_path']}")
            trainer.test(model, data_module)
            
        if args["predict"]:
            
            my_logger.info("[Main] Just performing prediction step")
            prediction = trainer.predict(model, data_module)
            if trainer.global_rank == 0:
                my_logger.info(f"[Main] Prediction result: {prediction}")
        
    else: # training
        model = model_class(
            selfie_symbol_to_idx = data_module.symbol_to_idx,
            selfie_max_len = SELFIES_MAX_LEN,
            p_args = args,
        )
        my_logger.info(f"[Main] Model Summary: {summarize(model)}")
        
        my_logger.info("[Main] Begin Training!")

        trainer = pl.Trainer(
                         max_epochs=args["epochs"],
                         accelerator="auto",
                         logger=tbl, 
                         callbacks=[early_stopping, lr_monitor, checkpoint_callback],
                        #  strategy="fsdp" if torch.cuda.device_count() > 1 else "auto",
                         accumulate_grad_batches=args["accumulate_grad_batches_num"],
                        #  gradient_clip_val=1.0,
        )
        try :
            trainer.fit(model, data_module,ckpt_path=args["checkpoint_path"])

            if trainer.global_rank == 0:
                os.system(f"cp -r {out_path}/* {out_path_final}/ ")
                my_logger.info(f"[Main] Copied all content from {out_path} to {out_path_final}")
                
            if args['debug']:
                my_logger.info(f"[Main] Debug mode, not running test/predict")
                return
            # Ensure all processes synchronize before switching to test mode
            trainer.strategy.barrier()               
               
            # --- CLEANUP: release GPU memory used during training ---
            model.to('cpu')       # move model off GPU
            del model             # delete model
            del trainer           # delete trainer
            import gc
            gc.collect()
            torch.cuda.empty_cache()
                
            # my_logger.info(f"[Main] my process rank: {os.getpid()}")
            trainer = pl.Trainer(logger=False, use_distributed_sampler=False) # ensure accurate test results
            model = model_class.load_from_checkpoint(checkpoint_path=checkpoint_callback.best_model_path, 
                                                 selfie_symbol_to_idx = data_module.symbol_to_idx,
                                                selfie_max_len = SELFIES_MAX_LEN,
                                                p_args = args,
            )
            model.validate_with_generation = True
            
            trainer.strategy.barrier()  
            predict_results = trainer.predict(model, data_module,)
            if trainer.global_rank == 0:
                my_logger.info(f"[Main] Prediction path {checkpoint_callback.best_model_path}!")
                my_logger.info(f"[Main] Prediction result: {predict_results}")
            trainer.strategy.barrier()  
            val_result = trainer.validate(model, data_module) 
            trainer.strategy.barrier()  
            # test_result = trainer.test(model, data_module,)
            test_result = [{}] # not testing to save some time
            trainer.strategy.barrier()  
            
            if trainer.global_rank == 0:
                my_logger.info(f"[Main] Validation metric {checkpoint_callback.monitor}, best score: {checkpoint_callback.best_model_score.item()}")
                my_logger.info(f"[Main] Testing path {checkpoint_callback.best_model_path}!")
                all_test_results = [{}]
                all_val_results = [{}]
                
                
                
                # save the test and validate results
                test_result[0]['best_epoch'] = checkpoint_callback.best_model_path.split("/")[-1].split("-")[0]
                if not args['optional_inputs']:
                    all_test_results = test_result
                    all_val_results = val_result
                else:
                    # loader_all_inputs, loader_HSQC_H_NMR, loader_HSQC_C_NMR, loader_only_hsqc, loader_only_1d, loader_only_H_NMR, loader_only_C_NMR
                    for loader_idx, NMR_type in enumerate(["all_inputs", "HSQC_H_NMR", "HSQC_C_NMR", "only_hsqc", "only_1d", "only_H_NMR", "only_C_NMR"]):
                        model.only_test_this_loader(loader_idx=loader_idx)
                    
                    all_test_results = test_result
                    all_val_results = val_result

                with open(f"{out_path}/{path1}/{path2}/test_result.pkl", "wb") as f:
                    pickle.dump(all_test_results, f)
                with open(f"{out_path}/{path1}/{path2}/val_result.pkl", "wb") as f:
                    pickle.dump(all_val_results, f)
                with open(f"{out_path}/{path1}/{path2}/predict_result.pkl", "wb") as f:
                    pickle.dump(predict_results, f)
                
            

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
                my_logger.info(f"[Main] Copied all test/val/predict pkl files from {out_path} to {out_path_final}")
                logging.shutdown()



if __name__ == '__main__':
    main()