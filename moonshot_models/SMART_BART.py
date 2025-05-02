import sys, os, pickle
import pathlib

# Add the current directory to sys.path
sys.path.append(pathlib.Path(__file__).parent.parent.absolute().__str__())
import pytorch_lightning as pl
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

import selfies
from rdkit.Chem import rdFingerprintGenerator
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

from Spectre.models.encoders.encoder_factory import build_encoder
from torch.optim.lr_scheduler import LambdaLR
from Spectre.utils.lr_scheduler import NoamOpt
from torch.optim import AdamW

from concurrent.futures import ThreadPoolExecutor
import psutil, os


# ------------------------ Model ------------------------
class SmartBart(pl.LightningModule):
    def __init__(self,
                #  vocab_size=50, decoder_embed_dim=784, num_layers=2, nhead=4, lr=1e-3
                selfie_symbol_to_idx,
                selfie_max_len,
                p_args,
                 ):
        super().__init__()
        # self.save_hyperparameters()
        self.p_args = p_args
        self.selfie_symbol_to_idx = selfie_symbol_to_idx
        to_save = {**p_args}
        self.save_hyperparameters(to_save, logger=True)
        self.logger_should_sync_dist = torch.cuda.device_count() > 1
        
        tokenizer_path = pathlib.Path(__file__).parent.parent / 'moonshot_dataset' / 'data' / 'selfies_tokenizer'
        # load idx_to_symbol 
        with open(tokenizer_path / 'idx_to_symbol.pkl', 'rb') as f:
            self.idx_to_symbol = pickle.load(f)
        
        # logging 
        # self.validation_step_outputs = []
        # self.training_step_outputs = []
        # self.test_step_outputs = []
        self.predict_step_outputs = {} # map a molecule to its predicted SMILES 
        
        self.validate_with_generation = False
          
        # NMR encoder config
        if p_args['load_encoder']:
            
            from Spectre.inference.inference_utils import choose_model
            spectre_model = choose_model("optional", load_for_moonshot=True)
            self.NMR_peak_encoder = spectre_model.enc
            self.transformer_encoder = spectre_model.transformer_encoder
            self.NMR_type_embedding = spectre_model.NMR_type_embedding
            self.latent = spectre_model.latent
            
            for param in self.NMR_peak_encoder.parameters():
                param.requires_grad = False
            # repeat for others...
            for param in self.transformer_encoder.parameters():
                param.requires_grad = False
            for param in self.NMR_type_embedding.parameters():
                param.requires_grad = False
            self.latent.requires_grad = False
            self.encoder_frozen_until_epoch = p_args['encoder_frozen_until_epoch']

        else:
            self.NMR_peak_encoder = build_encoder(
                coord_enc = "sce", 
                dim_model = p_args['encoder_embed_dim'], 
                dim_coords = p_args['dim_coords'],
                wavelength_bounds = p_args['wavelength_bounds'],
                gce_resolution = p_args['gce_resolution'],
                use_peak_values = False)
            # self.transformer_encoder = spectre_model.transformer_encoder
            encoder_layer = torch.nn.TransformerEncoderLayer(
                d_model=p_args['encoder_embed_dim'],
                nhead=p_args['num_encoder_heads'],
                dim_feedforward=p_args['encoder_embed_dim'] * 4,
                batch_first=True,
                # norm_first=True,
                dropout=p_args['transformer_dropout'],
            )
            self.transformer_encoder = torch.nn.TransformerEncoder(
                encoder_layer,
                num_layers=p_args['num_encoder_layers'],
            )
            self.NMR_type_embedding = nn.Embedding(4, p_args['encoder_embed_dim'])
            # HSQC, C NMR, H NMR, MW
            self.latent = torch.nn.Parameter(torch.randn(1, 1, p_args['encoder_embed_dim'])) # the <cls> token
        
        
        # SEFILES decoder config
        self.vocab_size = len(selfie_symbol_to_idx)
        self.max_len = selfie_max_len
        
        sz = self.max_len + 1
        decoder_mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        decoder_mask = ~decoder_mask.bool() #decoder_mask.float().masked_fill(decoder_mask == 0, float('-inf')).masked_fill(decoder_mask == 1, float(0.0))
        self.register_buffer("decoder_mask", decoder_mask)
        
        if p_args['load_decoder']:
            decoder_model = None
            self.dec_start_token = decoder_model.dec_start_token
            self.decoder_embedding = decoder_model.decoder_embedding
            self.pos_embedding = decoder_model.pos_embedding
            self.decoder = decoder_model.decoder
            self.lm_output_layer = decoder_model.lm_output_layer
        else:
            self.dec_start_token = nn.Parameter(0.02*torch.randn(1, p_args['decoder_embed_dim']))
            self.decoder_embedding = nn.Embedding(self.vocab_size,  p_args['decoder_embed_dim'])
            self.pos_embedding = nn.Parameter(0.02*torch.randn(1, self.max_len,  p_args['decoder_embed_dim']))
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=p_args['decoder_embed_dim'], 
                nhead=p_args['num_decoder_heads'], 
                dim_feedforward=p_args['decoder_embed_dim'] * 4,
                batch_first=True,
                # norm_first=True,
                dropout=p_args['transformer_dropout'],
            )
            self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=p_args['num_decoder_layers'])
            self.lm_output_layer = nn.Linear(p_args['decoder_embed_dim'], self.vocab_size)
            
        self.selfie_padding_idx = selfie_symbol_to_idx['[PAD]']
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=self.selfie_padding_idx)
        self.automatic_optimization = not self.p_args['use_separate_optimizers']
        self.unfrozen_step = None
        
    # def forward(self, src, tgt):
    def encode(self, NMR, NMR_type_indicator, mask=None):
        """
        Returns
        -------
        latent : torch.Tensor of shape (n_spectra, n_peaks + 1, dim_model)
            The latent representations for the spectrum and each of its
            peaks.
        mem_mask : torch.Tensor
            The memory mask specifying which elements were padding in X.
        """
        if mask is None:
            zeros = ~NMR.sum(dim=2).bool()
            mask = [
                torch.tensor([[False]] * NMR.shape[0]).type_as(zeros),
                zeros,
            ]
            mask = torch.cat(mask, dim=1)
            mask = mask.to(self.device)

        points = self.NMR_peak_encoder(NMR) # something like positional encoding , but encoding cooridinates
        NMR_type_embedding = self.NMR_type_embedding(NMR_type_indicator)
        points += NMR_type_embedding
        latent = self.latent.expand(points.shape[0], -1, -1) # make batch_size copies of latent
        points = torch.cat([latent, points], dim=1)
      
        out = self.transformer_encoder(points, src_key_padding_mask=mask)
        return out, mask
    
    def decode(self, memory, true_targets=None,  encoder_src_key_padding_mask=None):
        """
        memory: output of the encoder
        true_targets: indices of the target selfie tokens
        """

        if true_targets is not None:
            tgt = self.decoder_embedding(true_targets) + self.pos_embedding[:, :true_targets.size(1)]
            tgt = torch.cat([self.dec_start_token.expand(memory.size(0), -1,-1), tgt], dim=1)
            # print("tgt.shape", tgt.shape)
            # print("memory.shape", memory.shape)
            tgt_key_padding_mask = true_targets == self.selfie_padding_idx
            start_token_mask = torch.zeros(tgt_key_padding_mask.size(0), 1, 
                                           dtype=torch.bool, device=tgt_key_padding_mask.device)
            tgt_key_padding_mask = torch.cat([start_token_mask, tgt_key_padding_mask], dim=1)  # [B, T+1]
            
            out = self.decoder(
                            tgt, 
                            memory, 
                            tgt_mask=self.decoder_mask[:-1,:-1], 
                            tgt_key_padding_mask = tgt_key_padding_mask,
                            memory_key_padding_mask=encoder_src_key_padding_mask
                )
            logits = self.lm_output_layer(out)
            logits[:,:,0] = float('-inf')
            return logits
        else:
            tgt = self.dec_start_token.expand(memory.size(0), 1,-1)
            dec_output_logits = []
            for i in range(self.max_len):
                out = self.decoder(
                    tgt, 
                    memory, 
                    tgt_mask=self.decoder_mask[:i+1, :i+1], 
                    tgt_key_padding_mask = None,
                    memory_key_padding_mask=encoder_src_key_padding_mask
                )
                dec_logits = self.lm_output_layer(out[:,-1,:])
                dec_logits[:,0] = float('-inf')
                
                dec_output_logits.append(dec_logits)
                next_embedded_token = self.decoder_embedding(torch.argmax(dec_logits, dim=1)).unsqueeze(1) + self.pos_embedding[:, i, :]
                
                tgt = torch.cat([tgt, next_embedded_token], dim=1)
                
            dec_output_logits = torch.stack(dec_output_logits, dim=1)
            return dec_output_logits
        
    def forward(self, NMR, NMR_type_indicator, tgt_selfie=None, return_representations=False):        
        encoder_memory, src_mask = self.encode(NMR, NMR_type_indicator) 
        logits = self.decode(encoder_memory, tgt_selfie, src_mask)

        return logits

    def training_step(self, batch, batch_idx):
        NMR, NMR_type_indicator, tgt = batch
        tgt_input = tgt[:, :-1]
        tgt_output = tgt
        logits = self(NMR, NMR_type_indicator, tgt_input)
        # print("logits.shape", logits.shape)
        # print("tgt_output.shape", tgt_output.shape)
        loss = self.loss_fn(logits.reshape(-1, logits.size(-1)), tgt_output.reshape(-1))
        
        # if self.global_rank == 0:
        #     optimizers = self.optimizers()
        #     lr_encoder = optimizers.param_groups[0]['lr']
        #     lr_decoder = optimizers.param_groups[1]['lr']
        #     print(f"[Epoch {self.current_epoch}] LR: {lr_encoder:.2e}, {lr_decoder:.2e}")
        
        # Manual optimization when using separate optimizers
        if self.p_args['use_separate_optimizers']:
            opt_encoder, opt_decoder = self.optimizers()
            # Zero grads
            opt_encoder.zero_grad()
            opt_decoder.zero_grad()
            # Backward pass
            self.manual_backward(loss)
            # Step optimizers
            opt_encoder.step()
            opt_decoder.step()
            # Step schedulers if necessary
            scheds = self.lr_schedulers()
            if isinstance(scheds, list) and len(scheds) == 2:
                scheds[0].step()
                scheds[1].step()
                print("stepping both schedulers")
        
        
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        # print("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        NMR, NMR_type_indicator, tgt, truth_smiles, MFPs= batch
        tgt_input = tgt[:, :-1]
        tgt_output = tgt
        if self.validate_with_generation:
            logits = self(NMR, NMR_type_indicator)
        else: # teacher forcing
            logits = self(NMR, NMR_type_indicator, tgt_input)
            
        loss = self.loss_fn(logits.reshape(-1, logits.size(-1)), tgt_output.reshape(-1))

        pred_ids = logits.argmax(dim=-1)  # [B, T]
        predicted_smiles, predicted_mfps = self.decode_smiles_and_mfp_multithreaded(pred_ids)
        count_valid_smiles = torch.sum(predicted_mfps.sum(dim=1) != 0)
        valid_smiles_rate = count_valid_smiles / len(predicted_mfps)
        # print("valid_smiles_rate", valid_smiles_rate)
        cos_sim = F.cosine_similarity(MFPs, predicted_mfps, dim=1)
        
        metrics = {
            "cosine_similarity": cos_sim.mean(),
            "valid_smiles_rate": valid_smiles_rate,
            "loss": loss,
        }
        for k,v in metrics.items():
            self.log(f"val/{k}", v, on_step=False, on_epoch=True,
                     prog_bar = k in ["cosine_similarity", "valid_smiles_rate"],
                     batch_size = NMR.shape[0])
        # return metrics
    
    def test_step(self, batch, batch_idx):
        NMR, NMR_type_indicator, tgt, truth_smiles, MFPs = batch
        # tgt_input = tgt[:, :-1]
        tgt_output = tgt
        logits = self(NMR, NMR_type_indicator)
        loss = self.loss_fn(logits.reshape(-1, logits.size(-1)), tgt_output.reshape(-1))
        
        pred_ids = logits.argmax(dim=-1)  # [B, T]

        predicted_smiles, predicted_mfps = self.decode_smiles_and_mfp_multithreaded(pred_ids)
        count_valid_smiles = torch.sum(predicted_mfps.sum(dim=1) != 0)
        valid_smiles_rate = count_valid_smiles / len(predicted_mfps)
        cos_sim = F.cosine_similarity(MFPs, predicted_mfps, dim=1)
        
        metrics = {
            "cosine_similarity": cos_sim.mean(),
            "valid_smiles_rate": valid_smiles_rate,
            "loss": loss,
        }
        for k,v in metrics.items():
            self.log(f"test/{k}", v, on_step=False, 
                     on_epoch=True, prog_bar = False,
                     batch_size = NMR.shape[0])
       
    def predict_step(self,  batch, batch_idx):
        NMR, NMR_type_indicator, tgt, truth_smiles, MFPs = batch
        # tgt_input = tgt[:, :-1]
        tgt_output = tgt
        logits = self(NMR, NMR_type_indicator)
        
        pred_ids = logits.argmax(dim=-1)  # [B, T]

        predicted_smiles, predicted_mfps = self.decode_smiles_and_mfp_multithreaded(pred_ids)
        count_valid_smiles = torch.sum(predicted_mfps.sum(dim=1) != 0)
        valid_smiles_rate = count_valid_smiles / len(predicted_mfps)
        cos_sim = F.cosine_similarity(MFPs, predicted_mfps, dim=1)
        
        metrics = {
            "truth_smiles": truth_smiles,
            "predicted_smiles": predicted_smiles,
            "cosine_similarity": cos_sim.mean(),
            "valid_smiles_rate": valid_smiles_rate,
        }
        if type(self.predict_step_outputs)==list: # adapt for child class: optional_input_ranked_transformer
            self.predict_step_outputs.append(metrics)
        return metrics
        
    

    def get_selfies(self, seq):
        selfie = ''
        for x in seq:
            x = x.item()
            if x == 1:
                continue
            if x == 2:
                break
            selfie += self.idx_to_symbol[x]

        return selfie
    
    def get_smiles(self, seq):
        selfie = self.get_selfies(seq)
        smiles = selfies.decoder(selfie)
        
        return smiles
        
    def gen_mfp(self, smiles):
        try:
            MFP_generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
            mol = Chem.MolFromSmiles(smiles)
            Chem.SanitizeMol(mol)
            fp = MFP_generator.GetFingerprint(mol)
            return torch.tensor(fp).float()
        except:
            
            # print("smiles is ", smiles)
            # print("Error in gen_mfp")
            return torch.zeros(2048).float()
    
    def decode_smiles_and_mfp_multithreaded(self, pred_ids, max_workers=8):
        """ 
        Returns 
            predicted_smiles: a list of SMILES strings 
            predicted_mfps: a tensor of Morgan fingerprints (batch , 2048)
        """
        def decode_and_fingerprint(seq):
            smiles = self.get_smiles(seq)
            mfp = self.gen_mfp(smiles)
            return smiles, mfp

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(decode_and_fingerprint, pred_ids))

        predicted_smiles = [r[0] for r in results]
        predicted_mfps = torch.stack([r[1] for r in results])

        return predicted_smiles, predicted_mfps.to(self.device)




        
    def on_train_epoch_start(self):
        if self.p_args["load_encoder"] and self.current_epoch == self.encoder_frozen_until_epoch:
            print(f"Unfreezing encoder at epoch {self.current_epoch}")
            for param in self.NMR_peak_encoder.parameters():
                param.requires_grad = True
            for param in self.transformer_encoder.parameters():
                param.requires_grad = True
            for param in self.NMR_type_embedding.parameters():
                param.requires_grad = True
            self.latent.requires_grad = True
            self.unfrozen_step = self.global_step


            

    
    # def configure_optimizers(self):
    #     # Initialize optimizer with a dummy learning rate (1.0)
    #     # since LambdaLR will multiply this value
    #     optim = torch.optim.AdamW(self.parameters(), lr=1.0,
    #                             weight_decay=self.p_args['weight_decay'],
    #                             betas=(0.9, 0.98), eps=1e-9)
        
    #     model_size = (self.p_args["encoder_embed_dim"] + self.p_args["decoder_embed_dim"]) // 2
        
    #     # Lambda function to calculate the absolute learning rate,
    #     # not just a multiplier
    #     def noam_lambda(step):
    #         # Add 1 to step to match original implementation
    #         step += 1
    #         return self.p_args['noam_factor'] * (model_size ** -0.5) * min(
    #             step ** -0.5, step * self.p_args['warm_up_steps'] ** -1.5
    #         )
        
    #     scheduler = LambdaLR(optim, lr_lambda=noam_lambda)
        
    #     return {
    #         "optimizer": optim,
    #         "lr_scheduler": {
    #             "scheduler": scheduler,
    #             "interval": "step",
    #             "frequency": 1,
    #         }
    #     }
        
    def configure_optimizers(self):
        # Group encoder vs decoder parameters
        encoder_params, decoder_params = [], []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if any(k in name for k in ['NMR_peak_encoder', 'transformer_encoder', 'NMR_type_embedding', 'latent']):
                encoder_params.append(param)
            else:
                decoder_params.append(param)
        print(f"encoder_params: {len(encoder_params)}, decoder_params: {len(decoder_params)}")
        
        # Define two learning rates
        # lr_encoder = self.p_args['lr_finetuning'] 
        # lr_decoder = self.p_args['lr']

        # Construct optimizer
        optimizer = AdamW([
            {"params": encoder_params, "lr": 1},
            {"params": decoder_params, "lr": 1}
        ], betas=(0.9, 0.98), eps=1e-9, weight_decay=self.p_args['weight_decay'])


        model_size = (self.p_args["encoder_embed_dim"] + self.p_args["decoder_embed_dim"]) // 2
        
    
        # Define Noam-style schedulers
        # def noam_lambda(model_size, warmup):
            # def lr_fn(step):
            #     step = max(step, 1)
            #     return (model_size ** -0.5) * min(step ** -0.5, step * warmup ** -1.5) * self.p_args['noam_factor']
            # return lr_fn
        
        # Lambda function to calculate the absolute learning rate,
        # not just a multiplier
        def noam_lambda(step):
            step += 1
            return self.p_args['noam_factor'] * (model_size ** -0.5) * min(
                step ** -0.5, step * self.p_args['warm_up_steps'] ** -1.5
            )
        def noam_since(step):
            if self.unfrozen_step is None:
                return 0 # not yet unfrozen)
            step -= self.unfrozen_step
            # print(f"step: {step}")
            return self.p_args['noam_factor'] * (model_size ** -0.5) * min(
                step ** -0.5, step * self.p_args['warm_up_steps'] ** -1.5
            )

        scheduler = LambdaLR(
            optimizer,
            lr_lambda=[
                noam_since,
                noam_lambda,
            ]
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
                "name": "noam"
            }
        }
    
    
    # def configure_optimizers(self):
    #     if self.p_args['use_separate_optimizers']:
    #         return self.configure_separate_optimizers()
    #     else:
    #         return self.configure_single_optimizer()
        
    # def configure_single_optimizer(self):
    #     optim = torch.optim.AdamW(self.parameters(), lr=self.p_args['lr'], 
    #                                  weight_decay = self.p_args['weight_decay'], 
    #                                  betas=(0.9, 0.98), eps=1e-9
    #                             ) 
    #     model_size = (self.p_args["encoder_embed_dim"] + self.p_args["decoder_embed_dim"]) // 2
    #     scheduler = NoamOpt(model_size, self.p_args['warm_up_steps'], optim, self.p_args['noam_factor'])
        
    #     return {
    #         "optimizer": optim,
    #         "lr_scheduler": {
    #             "scheduler": scheduler,
    #             "interval": "step",
    #             "frequency": 1,
    #         }
    #     }
        
    # def configure_separate_optimizers(self):
    #     # Separate encoder and decoder parameters
    #     encoder_params, decoder_params = [], []
    #     for name, param in self.named_parameters():
    #         if not param.requires_grad:
    #             continue
    #         if any(k in name for k in ['NMR_peak_encoder', 'transformer_encoder', 'NMR_type_embedding', 'latent']):
    #             encoder_params.append(param)
    #         else:
    #             decoder_params.append(param)
    #     print(f"encoder_params: {len(encoder_params)}, decoder_params: {len(decoder_params)}")

    #     # Learning rates
    #     lr_encoder = self.p_args['lr_finetuning']
    #     lr_decoder = self.p_args['lr']

    #     # Build separate optimizers
    #     encoder_optim = AdamW(encoder_params, lr=lr_encoder, betas=(0.9, 0.98), eps=1e-9, weight_decay=self.p_args['weight_decay'])
    #     decoder_optim = AdamW(decoder_params, lr=lr_decoder, betas=(0.9, 0.98), eps=1e-9, weight_decay=self.p_args['weight_decay'])

    #     # Model sizes
    #     encoder_dim = self.p_args['encoder_embed_dim']
    #     decoder_dim = self.p_args['decoder_embed_dim']
    #     warmup_steps = self.p_args['warm_up_steps']
    #     factor = self.p_args['noam_factor']

    #     # Wrap with NoamOpt
    #     encoder_scheduler = NoamOpt(encoder_dim, warmup_steps, encoder_optim, factor)
    #     decoder_scheduler = NoamOpt(decoder_dim, warmup_steps, decoder_optim, factor)

    #     return [
    #         {
    #             "optimizer": encoder_optim,
    #             "lr_scheduler": {
    #                 "scheduler": encoder_scheduler,
    #                 "interval": "step",
    #                 "frequency": 1,
    #                 "name": "encoder_noam"
    #             }
    #         },
    #         {
    #             "optimizer": decoder_optim,
    #             "lr_scheduler": {
    #                 "scheduler": decoder_scheduler,
    #                 "interval": "step",
    #                 "frequency": 1,
    #                 "name": "decoder_noam"
    #             }
    #         }
    #     ]
        
    def log(self, name, value, *args, **kwargs):
        # Set 'sync_dist' to True by default
        if kwargs.get('sync_dist') is None:
            kwargs['sync_dist'] = self.logger_should_sync_dist
        # if name == "test/mean_rank_1":
        #     print(kwargs,"\n\n")
        super().log(name, value, *args, **kwargs)
    
    @staticmethod
    def add_model_specific_args(parent_parser, use_small_model = False, model_name=""):
        from moonshot_utils.arg_config import str2bool  
        
        model_name = model_name if len(model_name) == 0 else f"{model_name}_"
        parser = parent_parser.add_argument_group(model_name)
        parser.add_argument(f"--{model_name}lr", type=float, default=1e-5)
        parser.add_argument(f"--{model_name}lr_finetuning", type=float, default=2e-6)
        parser.add_argument(f"--{model_name}noam_factor", type=float, default=1.0)
        
        if use_small_model:
            parser.add_argument(f"--{model_name}dim_coords", metavar='N',
                                type=int, default=[180, 180, 24 ],
                                nargs="+", action="store")
            parser.add_argument(f"--{model_name}num_encoder_heads", type=int, default=8)
            parser.add_argument(f"--{model_name}num_decoder_heads", type=int, default=8)
            parser.add_argument(f"--{model_name}encoder_embed_dim", type=int, default=384)
            parser.add_argument(f"--{model_name}decoder_embed_dim", type=int, default=384)
            parser.add_argument(f"--{model_name}num_encoder_layers", type=int, default=4)
            parser.add_argument(f"--{model_name}num_decoder_layers", type=int, default=4)
            parser.add_argument(f"--{model_name}warm_up_steps", type=int, default=4000)
        else:
            ### large model
            parser.add_argument(f"--{model_name}dim_coords", metavar='N',
                                type=int, default=[365, 365, 54 ],
                                nargs="+", action="store")
            parser.add_argument(f"--{model_name}num_encoder_heads", type=int, default=8)
            parser.add_argument(f"--{model_name}num_decoder_heads", type=int, default=8)
            parser.add_argument(f"--{model_name}encoder_embed_dim", type=int, default=784)
            parser.add_argument(f"--{model_name}decoder_embed_dim", type=int, default=784)
            parser.add_argument(f"--{model_name}num_encoder_layers", type=int, default=16)
            parser.add_argument(f"--{model_name}num_decoder_layers", type=int, default=16)
            parser.add_argument(f"--{model_name}warm_up_steps", type=int, default=8000)
            
            
        parser.add_argument(f"--{model_name}wavelength_bounds",
                            type=float, default=[[0.01, 400.0], [0.01, 20.0]], nargs='+', action='append')
        parser.add_argument(f"--{model_name}transformer_dropout", type=float, default=0.1)
  
        parser.add_argument(f"--{model_name}weight_decay", type=float, default=0.0)
        parser.add_argument(f"--{model_name}scheduler", type=str, default="attention")
        parser.add_argument(f"--{model_name}coord_enc", type=str, default="sce")
        parser.add_argument(f"--{model_name}gce_resolution", type=float, default=1)
        
        parser.add_argument(f"--{model_name}load_encoder", type=lambda x:bool(str2bool(x)), default=True)
        parser.add_argument(f"--{model_name}encoder_frozen_until_epoch", type=int, default=5)
        parser.add_argument(f"--{model_name}load_decoder", type=lambda x:bool(str2bool(x)), default=False)
        
        parser.add_argument(f"--{model_name}use_separate_optimizers", type=lambda x:bool(str2bool(x)), default=False)
        # parser.add_argument(f"--{model_name}freeze_weights", type=lambda x:bool(str2bool(x)), default=False)
        return parent_parser
    