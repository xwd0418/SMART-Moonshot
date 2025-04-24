import sys, os, pickle
import pathlib

# Add the current directory to sys.path
sys.path.append(pathlib.Path(__file__).parent.parent.absolute().__str__())
import pytorch_lightning as pl
import torch
from torch import nn
import selfies
from rdkit.Chem import rdFingerprintGenerator
from rdkit import Chem

from Spectre.models.encoders.encoder_factory import build_encoder
from Spectre.utils.lr_scheduler import NoamOpt
import torch.nn.functional as F


# ------------------------ Model ------------------------
class SmartBart(pl.LightningModule):
    def __init__(self,
                #  vocab_size=50, decoder_embed_dim=784, num_layers=2, nhead=4, lr=1e-3
                selfie_symbol_to_idx,
                selfie_max_len,
                p_args,
                 ):
        super().__init__()
        self.save_hyperparameters()
        self.p_args = p_args
        self.selfie_symbol_to_idx = selfie_symbol_to_idx
        to_save = {**p_args}
        self.save_hyperparameters(to_save, logger=True)
        
        tokenizer_path = pathlib.Path(__file__).parent.parent / 'moonshot_dataset' / 'data' / 'selfies_tokenizer'
        # load idx_to_symbol 
        with open(tokenizer_path / 'idx_to_symbol.pkl', 'rb') as f:
            self.idx_to_symbol = pickle.load(f)
            
        self.MFP_generator = rdFingerprintGenerator.GetMorganGenerator()
        # using default radius=3, fp_size=2048,
        # maybe use multithread in the future
        
        # NMR encoder config
        if p_args['load_encoder']:
            spectre_model = None
            self.NMR_peak_encoder = spectre_model.enc
            self.transformer_encoder = spectre_model.transformer_encoder
            self.NMR_type_embedding = spectre_model.NMR_type_embedding
            self.latent = spectre_model.latent
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
                norm_first=True,
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
                norm_first=True,
                dropout=p_args['transformer_dropout'],
            )
            self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=p_args['num_decoder_layers'])
            self.lm_output_layer = nn.Linear(p_args['decoder_embed_dim'], self.vocab_size)
            
        self.selfie_padding_idx = selfie_symbol_to_idx['[PAD]']
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=self.selfie_padding_idx)

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
        # print("encoder_memory.shape", encoder_memory.shape)
        # print("memory.shape", memory.shape)
        # print("tgt_emb.shape", tgt_emb.shape)
        # print("src_mask.shape", src_mask.shape)
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
        self.log("train/loss", loss)
        # print("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        NMR, NMR_type_indicator, tgt, smiles = batch
        tgt_input = tgt[:, :-1]
        tgt_output = tgt
        logits = self(NMR, NMR_type_indicator, tgt_input)
        loss = self.loss_fn(logits.reshape(-1, logits.size(-1)), tgt_output.reshape(-1))
        self.log("val/loss", loss)
        # print("val_loss", loss)
        pred_ids = logits.argmax(dim=-1)  # [B, T]

        # Decode SELFIES
        predicted_smiles = [self.get_smiles(seq) for seq in pred_ids]
        print("Example decoded SELFIES:")
        for i,s in enumerate(predicted_smiles[:5]):
            print(s)
            cos_sim = F.cosine_similarity(self.gen_mfp(smiles[i]), self.gen_mfp(s), dim=0)
            print("cosine similarity", cos_sim)
        return loss
    
    def test_step(self, batch, batch_idx):
        NMR, NMR_type_indicator, tgt, smiles = batch
        tgt_input = tgt[:, :-1]
        tgt_output = tgt
        logits = self(NMR, NMR_type_indicator)
        loss = self.loss_fn(logits.reshape(-1, logits.size(-1)), tgt_output.reshape(-1))
        self.log("test/loss", loss)
        # print("test_loss", loss)
        return loss

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
        return selfies.decoder(selfie)
        
    def gen_mfp(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        fp = self.MFP_generator.GetFingerprint(mol)
        return torch.tensor(fp).float()

    def configure_optimizers(self):
        optim = torch.optim.AdamW(self.parameters(), lr=self.p_args['lr'], 
                                     weight_decay = self.p_args['weight_decay'], 
                                     betas=(0.9, 0.98), eps=1e-9
                                ) 
        model_size = (self.p_args["encoder_embed_dim"] + self.p_args["decoder_embed_dim"]) // 2
        scheduler = NoamOpt(model_size, self.p_args['warm_up_steps'], optim, self.p_args['noam_factor'])
        
        return {
            "optimizer": optim,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            }
        }

    
    @staticmethod
    def add_model_specific_args(parent_parser, model_name=""):
        model_name = model_name if len(model_name) == 0 else f"{model_name}_"
        parser = parent_parser.add_argument_group(model_name)
        parser.add_argument(f"--{model_name}lr", type=float, default=1e-5)
        parser.add_argument(f"--{model_name}noam_factor", type=float, default=1.0)
        
        # ### large model
        # parser.add_argument(f"--{model_name}dim_coords", metavar='N',
        #                     type=int, default=[365, 365, 54 ],
        #                     nargs="+", action="store")
        # parser.add_argument(f"--{model_name}num_encoder_heads", type=int, default=8)
        # parser.add_argument(f"--{model_name}num_decoder_heads", type=int, default=8)
        # parser.add_argument(f"--{model_name}encoder_embed_dim", type=int, default=784)
        # parser.add_argument(f"--{model_name}decoder_embed_dim", type=int, default=784)
        # parser.add_argument(f"--{model_name}num_encoder_layers", type=int, default=16)
        # parser.add_argument(f"--{model_name}num_decoder_layers", type=int, default=16)
        # parser.add_argument(f"--{model_name}warm_up_steps", type=int, default=8000)
        
        ### small model
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
        
        
        
        parser.add_argument(f"--{model_name}wavelength_bounds",
                            type=float, default=[[0.01, 400.0], [0.01, 20.0]], nargs='+', action='append')
        parser.add_argument(f"--{model_name}transformer_dropout", type=float, default=0.1)
  
        parser.add_argument(f"--{model_name}weight_decay", type=float, default=0.0)
        parser.add_argument(f"--{model_name}scheduler", type=str, default="attention")
        parser.add_argument(f"--{model_name}coord_enc", type=str, default="sce")
        parser.add_argument(f"--{model_name}gce_resolution", type=float, default=1)
        
        parser.add_argument(f"--{model_name}load_encoder", type=bool, default=False)
        parser.add_argument(f"--{model_name}load_decoder", type=bool, default=False)
        # parser.add_argument(f"--{model_name}freeze_weights", type=bool, default=False)
        return parent_parser
    