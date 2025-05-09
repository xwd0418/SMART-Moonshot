import logging
import pickle, random
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS
import torch, os, pytorch_lightning as pl, glob
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence


class SmilesDataset(Dataset):
    
    def __init__(self, split, tokenizer, parser_args=None):
        self.split = split
        if  parser_args['datasrc'] == "shorter_than_300":
            file_path = f"/root/Moonshot/SMART-Moonshot/dataset/{split}_smiles_shorter_than_300.pkl"
            self.max_seq_len = 300+5
        elif parser_args['datasrc'] == "all":
            file_path = f"/root/Moonshot/SMART-Moonshot/dataset/{split}_smiles.pkl"
            self.max_seq_len = 1175+5
        else:
            raise ValueError("Invalid datasrc")
        with open(file_path, 'rb') as f:
            self.smiles = pickle.load(f)
            
        self.tokenizer = tokenizer
        self.parser_args = parser_args
        
    def __len__(self):
        return len(self.smiles)
    
    def __getitem__(self, idx):
        smiles = self.smiles[idx]
        encoded_token = self.tokenizer(smiles, return_tensors='pt', padding='max_length', max_length=self.max_seq_len, truncation=True)
        input_ids = encoded_token['input_ids']
        attention_mask = encoded_token['attention_mask'].bool()
        return{
            "input_ids": input_ids.squeeze(0),
            "attention_mask": attention_mask.squeeze(0),
            "smiles": smiles
        }
                

class SmilesDataModule(pl.LightningDataModule):
    def __init__(self, tokenizer, parser_args):
        super().__init__()
        self.batch_size = parser_args['bs']
        self.parser_args = parser_args
        
        self.tokenizer = tokenizer
    def setup(self, stage):
        if stage == "fit" or stage == "validate" or stage is None:
            self.train = SmilesDataset(split="train", tokenizer=self.tokenizer, parser_args=self.parser_args)
            self.val   = SmilesDataset(split="val",   tokenizer=self.tokenizer, parser_args=self.parser_args)
        if stage == "test":
            self.test = SmilesDataset(split="test", tokenizer=self.tokenizer, parser_args=self.parser_args)
        if stage == "predict":
            raise NotImplementedError("Predict setup not implemented")
        
    def train_dataloader(self) :
        return DataLoader(self.train, shuffle=True, batch_size=self.batch_size, 
                          num_workers=self.parser_args['num_workers'], pin_memory=True, persistent_workers=True)
    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, 
                          num_workers=self.parser_args['num_workers'], pin_memory=True, persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, 
                          num_workers=self.parser_args['num_workers'], pin_memory=True, persistent_workers=True)
