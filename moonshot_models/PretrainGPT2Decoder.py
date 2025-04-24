import logging
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
# from torcheval.metrics import Perplexity
import pytorch_lightning as pl
import torch, pickle
import math, numpy as np
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import torch.distributed as dist
from utils.lr_scheduler import NoamOpt
from models.BaseModel import BaseModel
from transformers import PreTrainedTokenizerFast

DATASET_DIR = '/root/Moonshot/SMART-Moonshot/dataset'

class SmilePretrainingDecoderGPT2(pl.LightningModule):
    """
      Only parameters, no sampling
    """

    def __init__(self,
                 parser_args,
                 tokenizer,
                 *args,
                 **kwargs):
        super().__init__()

        self.save_hyperparameters({**parser_args}, logger=True)

        # attributes
        self.parser_args = parser_args
        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.vocab_size
        print("vocab size is ", self.vocab_size)
        self.pad_token_idx = tokenizer.pad_token_id
        self.lr = parser_args['lr']
        self.weight_decay = parser_args['weight_decay']
        self.dim_model = parser_args['dim_model']
        self.num_decoder_layers = parser_args['num_decoder_layers']
        self.num_heads = parser_args['num_heads']
        # self.ff_dim = parser_args['ff_dim']
        self.scheduler = parser_args['scheduler']
        self.noam_factor = parser_args['noam_factor']
        self.warm_up_steps = parser_args['warm_up_steps']
      
        # self.activation = parser_args['activation']
        self.transformer_dropout = parser_args['transformer_dropout']
        self.dropout = nn.Dropout(parser_args['embedding_dropout'])
        if  parser_args['datasrc'] == "shorter_than_300":
            self.max_seq_len = 300+5
        elif parser_args['datasrc'] == "all":
            self.max_seq_len = 1175+5
        else:
            raise ValueError("Invalid datasrc")
     
        custom_config = GPT2Config(
            n_embd=self.dim_model,        # Embedding size
            n_layer=self.num_decoder_layers,         # Number of attention layers
            n_head=self.num_heads,           # Number of attention heads
            vocal_size=self.vocab_size,     # Vocabulary size
        )

        # Initialize model with custom configuration
        self.model = GPT2LMHeadModel(custom_config)
     
        self.loss_fn = nn.CrossEntropyLoss(reduction="none", ignore_index=self.pad_token_idx)
        self.log_softmax = nn.LogSoftmax(dim=2)
        # self.perplexity_metric = Perplexity(ignore_index=1)
        
        # logging
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []


    
    def forward(self, input_ids, attention_mask, labels=None):
        return self.model(input_ids, attention_mask=attention_mask, labels=labels)

    def training_step(self, batch, batch_idx):
        inputs = batch['input_ids']
        attention_mask = batch['attention_mask']
        # labels = batch['smiles']
        outputs = self.forward(inputs, attention_mask, labels=inputs)
        loss = outputs.loss
        accuracy = self._predicted_accuracy(batch, outputs) # Get predictions
        perplexity=self._calc_perplexity(batch, outputs) # Calculate perplexity

        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train/accuracy', accuracy, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train/perplexity', perplexity, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        metrics = {
            "loss": loss.detach().item(),
            "accuracy": accuracy.detach().item(),
            "perplexity": perplexity,
            }
        self.training_step_outputs.append(metrics)
        return loss

    def validate_or_test_step(self, batch, batch_idx, mode="val"):
        inputs = batch['input_ids']
        attention_mask = batch['attention_mask']
        # labels = batch['smiles']
        outputs = self.forward(inputs, attention_mask, inputs)
        loss = outputs.loss
        accuracy = self._predicted_accuracy(batch, outputs) # Get predictions
        perplexity=self._calc_perplexity(batch, outputs) # Calculate perplexity
        
        # may change to non-teacher forcing ..
        # edit_distance = self._calc_edit_distance(batch)
        edit_distance = self._calc_edit_distance_half_given(batch)
        self.log(f"{mode}/loss", loss)
        metrics = {
            "loss": loss.detach().item(),
            "accuracy": accuracy.detach().item(),
            "perplexity": perplexity,
            "edit_distance": edit_distance.detach().item()
            }
        
        if mode == "val":
            self.validation_step_outputs.append(metrics)
        elif mode == "test":
            self.test_step_outputs.append(metrics)
        return metrics
    
    
    # for inference
    def generate_text(self, prompt, max_length=50):
        # Tokenize the input prompt
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs.input_ids.to(self.device)
        attention_mask = inputs.attention_mask.to(self.device)

        # Generate text
        generated_ids = self.model.generate(input_ids, attention_mask=attention_mask, max_length=max_length)

        # Decode the generated tokens to text
        generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        return generated_text
    