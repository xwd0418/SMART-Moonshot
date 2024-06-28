import logging, os, sys, torch
import random, pickle
import numpy as np
import pytorch_lightning as pl
import pytorch_lightning.callbacks as cb

from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.utilities.model_summary import summarize
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import torch.distributed as dist

from argparse import ArgumentParser
import argparse
from functools import reduce

from dataset.SmilesDataset import SmilesDataModule
from dataset.SmilesTokenizer import SMILESTokenizerBuilder

from models.PretrainingDecoder import SmilePretrainingDecoder
        
def main():
     
    print("Hello World")

    # dependencies: hyun_fp_data, hyun_pair_ranking_set_07_22
    parser = ArgumentParser(add_help=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--expname", type=str, default=f"experiment")
    parser.add_argument("--foldername", type=str, default=f"debug")
    parser.add_argument("--bs", type=int, default=64, help="batch size")
    parser.add_argument("--datasrc", type=str, default="shorter_than_300", help="shorter_than_300 or all")

    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--num_workers", type=int, default=16)
    # for early stopping/model saving
    # parser.add_argument("--metric", type=str, default="val/mean_rank_1")
    parser.add_argument("--metric", type=str, default="val/ce_loss")
    parser.add_argument("--metricmode", type=str, default="min")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to the checkpoint file to resume training")
    parser.add_argument("--random_seed", type=int, default=42)
    
    #model specific
    parser.add_argument(f"--lr", type=float, default=1e-5)
    parser.add_argument(f"--dim_model", type=int, default=384)
    parser.add_argument(f"--num_heads", type=int, default=8)
    parser.add_argument(f"--num_layers", type=int, default=8)
    parser.add_argument(f"--ff_dim", type=int, default=512)
    parser.add_argument(f"--embedding_dropout", type=float, default=0.0)
    parser.add_argument(f"--transformer_dropout", type=float, default=0.1)
    parser.add_argument(f"--weight_decay", type=float, default=0.0)
    #optimizers
    parser.add_argument(f"--noam_factor", type=float, default=1.0)    
    parser.add_argument(f"--warm_up_steps", type=int, default=4000)
    parser.add_argument(f"--scheduler", type=str, default="attention")
    # only validate or test
    parser.add_argument("--validate", type=str2bool, default=False)
    parser.add_argument("--test", type=str2bool, default=False)

    args = vars(parser.parse_known_args()[0])
    
        
    
    seed_everything(seed=args["random_seed"])   


    # Model args
    args_with_model = vars(parser.parse_known_args()[0])
    li_args = list(args_with_model.items())

    # Tensorboard setup
    curr_exp_folder_name = "pretrain_smiles_exp"
    out_path       =      f"/root/Moonshot/SMART-Moonshot/{curr_exp_folder_name}"
    out_path_final =      f"/root/Moonshot/SMART-Moonshot/{curr_exp_folder_name}"
    os.makedirs(out_path_final, exist_ok=True)
    
    path1 = args["foldername"]
    path2 = args["expname"]

    # Logger setup
    my_logger = init_logger(out_path, path1, path2)
    
    my_logger.info(f'[Main] Output Path: {out_path}/{path1}/{path2}')
    my_logger.info(f'[Main] using GPU : {torch.cuda.get_device_name()}')
    
    # Model and Data setup
    tokenizer = SMILESTokenizerBuilder().load_tokenizer()
    tokenizer.pad_token = "[PAD]"
    tokenizer.unk_token = "[UNK]"
    tokenizer.mask_token = "[MASK]"
    tokenizer.cls_token = "[CLS]"
    tokenizer.sep_token = "[SEP]"
    
    model =  SmilePretrainingDecoder(parser_args=args, tokenizer=tokenizer)
    
    my_logger.info(f"[Main] Model Summary: {summarize(model)}")
  
    data_module = SmilesDataModule(tokenizer=tokenizer, parser_args=args)

    # Trainer, callbacks
    metric, metricmode, patience = args["metric"], args["metricmode"], args["patience"]
   
    tbl = TensorBoardLogger(save_dir=out_path, name=path1, version=path2)
    
    checkpoint_callback = cb.ModelCheckpoint(monitor=metric, mode=metricmode, save_last=True, save_top_k = 1)
  
    early_stopping = EarlyStopping(monitor=metric, mode=metricmode, patience=patience)
    lr_monitor = cb.LearningRateMonitor(logging_interval="step")
    trainer = pl.Trainer(
                         max_epochs=args["epochs"],
                         accelerator="gpu",
                         logger=tbl, 
                         callbacks=[checkpoint_callback, early_stopping, lr_monitor],
                        )
    if args["validate"]:
        my_logger.info("[Main] Just performing validation step")
        trainer.validate(model, data_module)
    elif args['test']:
        my_logger.info("[Main] Just performing test step")
        model.change_ranker_for_testing(test_ranking_set_path = "/workspace/ranking_sets_cleaned_by_inchi/SMILES_R0_to_R4_reduced_FP_ranking_sets_only_all_info_molecules/test/rankingset.pt")
        # model.change_ranker_for_testing()
        test_result = trainer.test(model, data_module,ckpt_path=args["checkpoint_path"])
        my_logger.info(f"[Main] test result: {test_result}")
        ## !!! ATTENTION !!!
        ## also need to files to be loaded in hsqc_folder_dataset.py
        
    else:
        test_result = None
        try:
            my_logger.info("[Main] Begin Training!")
            trainer.fit(model, data_module,ckpt_path=args["checkpoint_path"])
            # if dist.is_initialized():
            #     my_logger.info("[Main] Begin Testing:")
            #     rank = dist.get_rank()
            #     if rank == 0: # To only run the test once
            #         model.change_ranker_for_testing()
            #         # testlogger = CSVLogger(save_dir=out_path, name=path1, version=path2)

            #         test_trainer = pl.Trainer(accelerator="gpu", logger=tbl, devices=1,)
            #         test_trainer.test(model, data_module,ckpt_path=checkpoint_callback.best_model_path )
            #         # test_trainer.test(model, data_module,ckpt_path=checkpoint_callback.last_model_path )
            my_logger.info(f"[Main] Testing path {checkpoint_callback.best_model_path}!")
            test_result = trainer.test(model, data_module, ckpt_path=checkpoint_callback.best_model_path)
            # save test result as pickle
            with open(f"{out_path}/{path1}/{path2}/test_result.pkl", "wb") as f:
                pickle.dump(test_result, f)
            
        except Exception as e:
            my_logger.error(f"[Main] Error: {e}")
            raise(e)
        finally: #Finally move all content from out_path to out_path_final
            my_logger.info("[Main] Done!")
            my_logger.info(f"[Main] test result: {test_result}")
            # os.system(f"cp -r {out_path}/* {out_path_final}/ && rm -rf {out_path}/*")

        
#utils 

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

if __name__ == '__main__':
    torch.set_float32_matmul_precision('medium')
    main()
