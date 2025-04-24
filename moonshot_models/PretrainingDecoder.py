import logging
import pytorch_lightning as pl
import torch, pickle
import math, numpy as np
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import torch.distributed as dist
from utils.lr_scheduler import NoamOpt
from utils.transformer_utils import generate_square_subsequent_mask
from torchaudio.functional import edit_distance 


class SmilePretrainingDecoder(pl.LightningModule):
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
        self.num_layers = parser_args['num_layers']
        self.num_heads = parser_args['num_heads']
        self.ff_dim = parser_args['ff_dim']
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
        
        self.register_buffer("empty_memory", torch.zeros((parser_args['bs'], self.max_seq_len, self.dim_model)).float())

        # TransformerDecoderModel
        self.emb = nn.Embedding(self.vocab_size, self.dim_model, padding_idx=self.pad_token_idx)
        dec_norm = nn.LayerNorm(self.dim_model)
        # dec_layer = PreNormDecoderLayer(
        #     dim_model, num_heads, ff_dim, dropout, activation)
        dec_layer = nn.TransformerDecoderLayer(self.dim_model, self.num_heads, self.ff_dim, batch_first=True, dropout=self.transformer_dropout)
        self.decoder = nn.TransformerDecoder(dec_layer, self.num_layers, norm=dec_norm)
        self.token_fc = nn.Linear(self.dim_model, self.vocab_size)
        self.loss_fn = nn.CrossEntropyLoss(reduction="none", ignore_index=self.pad_token_idx)
        self.log_softmax = nn.LogSoftmax(dim=2)
        self.register_buffer("pos_emb", self._positional_embs())
        
        # logging
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

    # Ripped from chemformer
    def _construct_input(self, token_ids):
        """
          Expects tokens in (seq_len, b_s) format

        Returns:
          (seq_len, b_s, dim_model) embedding (with dropout applied)
        """
        seq_len, _ = tuple(token_ids.size())
        token_embs = self.emb(token_ids)
        # token_embs.shape is ???

        # Scaling the embeddings like this is done in other transformer libraries
        token_embs = token_embs * math.sqrt(self.dim_model)

        # generateb by copilot :(seq_len, dim_model) -> (seq_len, 1, dim_model) -> (seq_len, b_s, dim_model)
        # self.pos_emb.shape() is (max_seq_len, dim_model)???
        positional_embs = self.pos_emb[:seq_len,:].unsqueeze(0).transpose(0, 1)
        embs = token_embs + positional_embs
        embs = self.dropout(embs)
        return embs

    # Ripped from chemformer
    def _positional_embs(self):
        """ Produces a tensor of positional embeddings for the model

        Returns a tensor of shape (self.max_seq_len, self.dim_model) filled with positional embeddings,
        which are created from sine and cosine waves of varying wavelength
        """

        encs = torch.tensor(
            [dim / self.dim_model for dim in range(0, self.dim_model, 2)])
        encs = 10000 ** encs
        encs = [(torch.sin(pos / encs), torch.cos(pos / encs))
                for pos in range(self.max_seq_len)]
        encs = [torch.stack(enc, dim=1).flatten()[:self.dim_model]
                for enc in encs]
        encs = torch.stack(encs)
        return encs

    def sample(self, hsqc):
        """
          hsqc: (bs, seq_len)
        """
        bs, _, _ = hsqc.size()
        pad_token_idx = 0
        begin_token_idx = 2
        end_token_idx = 3
        with torch.no_grad():
            # (bs, seq_len)
            tokens = torch.ones((bs, self.max_seq_len), dtype=torch.int64).to(
                self.device) * pad_token_idx
            tokens[:, 0] = begin_token_idx
            # (bs, seq_len)
            pad_mask = torch.zeros((bs, self.max_seq_len),
                                   dtype=torch.bool).to(self.device)

            memory, encoder_mask = self.encode(hsqc)

            for i in range(1, self.max_seq_len):
                decoder_inputs = tokens[:, :i]
                decoder_mask = pad_mask[:, :i]
                # (seq_len, bs, vocab_size) for token_output
                model_output = self.decode(
                    memory, encoder_mask, decoder_inputs, decoder_mask)["token_output"]
                best_ids = torch.argmax(
                    model_output[[-1], :, :], dim=2).squeeze(0).long()  # (bs)

                tokens[:, i] = best_ids
                pad_mask[:, i] = (best_ids == end_token_idx) | (
                    best_ids == pad_token_idx)

                if torch.all(pad_mask):
                    break
            # (bs, seq_len)
            my_tokens = tokens.transpose(0, 1).tolist()
            str_tokens = self.tokeniser.convert_ids_to_tokens(my_tokens)
            mol_strs = self.tokeniser.detokenise(str_tokens)
            return mol_strs

    def sample_rdm(self, hsqc, temperature=1.0, gen_len=None):
        """
          hsqc: (bs, seq_len)
        """
        bs, _, _ = hsqc.size()
        pad_token_idx = 0
        begin_token_idx = 2
        end_token_idx = 3

        max_len = self.max_seq_len
        if gen_len:
            max_len = gen_len

        with torch.no_grad():
            # (bs, seq_len)
            tokens = torch.ones((bs, gen_len), dtype=torch.int64).to(
                self.device) * pad_token_idx
            tokens[:, 0] = begin_token_idx
            # (bs, seq_len)
            pad_mask = torch.zeros((bs, gen_len),
                                   dtype=torch.bool).to(self.device)
            print(f"Max seq len: {gen_len}")
            memory, encoder_mask = self.encode(hsqc)

            for i in tqdm.tqdm(range(1, gen_len)):
                decoder_inputs = tokens[:, :i]
                decoder_mask = pad_mask[:, :i]
                # (seq_len, bs, vocab_size) for token_output
                model_output = self.decode(
                    memory, encoder_mask, decoder_inputs, decoder_mask)["token_output"]
                # (seq_len, bs, vocab_size)
                probability_output = F.softmax(
                    model_output / temperature, dim=2)
                sampled_ids = torch.multinomial(
                    probability_output[-1, :, :], num_samples=1).flatten()
                tokens[:, i] = sampled_ids
                pad_mask[:, i] = (sampled_ids == end_token_idx) | (
                    sampled_ids == pad_token_idx)

                if torch.all(pad_mask):
                    break

            # (bs, seq_len)
            my_tokens = tokens.tolist()
            str_tokens = self.tokeniser.convert_ids_to_tokens(my_tokens)
            mol_strs = self.tokeniser.detokenise(str_tokens)
            return {
                "mol_strs": mol_strs,
                "token_ids": my_tokens,
                "tokens": str_tokens,
            }

    # def decode(self, memory, encoder_mask, decoder_inputs, decoder_mask):
    #     """

    #     Args:
    #         memory: (b_s, seq_len, dim_model)
    #         encoder_padding_mask : (b_s, seq_len)
    #         decoder_inputs: (b_s, seq_len)
    #         decoder_mask: (b_s, seq_len)

    #     Returns:
    #         {
    #           model_output: (s_l, b_s, dim_model)
    #           token_output: (s_l, b_s, vocab_size)
    #         }
    #     """
    #     _, s_l = decoder_mask.size()

    #     # (s_l, s_l)
    #     tgt_mask = generate_square_subsequent_mask(s_l).to(self.device)
    #     # (b_s, s_l, dim)
    #     decoder_embs = self._construct_input(decoder_inputs)

    #     # embs, memory need seq_len, batch_size convention
    #     # (s_l, b_s, dim), (s_l, b_s, dim)
    #     decoder_embs, memory = decoder_embs.transpose(
    #         0, 1), memory.transpose(0, 1)

    #     model_output = self.decoder(
    #         decoder_embs,
    #         memory,
    #         tgt_mask=tgt_mask,  # prevent cheating mask
    #         tgt_key_padding_mask=decoder_mask,  # padding mask
    #         memory_key_padding_mask=encoder_mask  # padding mask
    #     )

    #     token_output = self.token_fc(model_output)

    #     return {
    #         "model_output": model_output,
    #         "token_output": token_output,
    #     }

    def forward(self, batch):
        # I use (batch_size, seq_len convention)
        # see datasets/dataset_utils.py:tokenise_and_mask
        collated_smiles = batch

        # decode
        decoder_inputs = collated_smiles["input_ids"]
        decoder_mask = collated_smiles["attention_mask"]
        assert(decoder_inputs.size() == decoder_mask.size())
        # decoder_inputs.size() is (64,305)
        
        decoder_embs = self._construct_input(decoder_inputs)
        # decoder_embs.size() is (64,305,384)
        
        tgt_mask = generate_square_subsequent_mask(decoder_inputs.size(1), device=self.device)
        # tgt_mask.size() is (305,305)
        
        model_output = self.decoder(
            decoder_embs,
            memory=self.empty_memory,
            tgt_mask=tgt_mask,  # prevent cheating mask
            tgt_key_padding_mask=decoder_mask,  # padding mask
        )
        assert(False)
        # use memory=x and use casaual mask

        # model_output.size() is ???

        token_output = self.token_fc(model_output)
        # token_output.size() is ???
        return {
            "model_output": model_output,
            "token_output": token_output,
        }
    
    def _calc_loss(self, batch_input, model_output):
        """ Calculate the loss for the model

        Args:
            batch_input (dict): Input given to model,
            model_output (dict): Output from model

        Returns:
            loss (singleton tensor),
        """

        target = batch_input["input_ids"]  # (b_s, s_l)
        target_mask = batch_input["attention_mask"]  # (b_s, s_l)
        token_output = model_output["token_output"]  # (s_l, b_s, vocab_size)

        assert (target.size()[0] == token_output.size()[1])

        batch_size, seq_len = tuple(target.size())

        token_pred = token_output.reshape((seq_len * batch_size, -1)).float()
        # token_pred.shape is ???
        loss = self.loss_fn(token_pred, target.reshape(-1)).reshape((seq_len, batch_size))

        inv_target_mask = ~(target_mask > 0)
        num_tokens = inv_target_mask.sum()
        loss = loss.sum() / num_tokens

        return loss

    def _calc_perplexity(self, batch_input, model_output):
        target_ids = batch_input["input_ids"]  # bs, seq_len
        target_mask = batch_input["attention_mask"]  # bs, seq_len
        vocab_dist_output = model_output["token_output"]  # seq_len, bs

        inv_target_mask = ~(target_mask > 0)

        # choose probabilities of token indices
        # logits = log_probabilities
        log_probs = vocab_dist_output.transpose(
            0, 1).gather(2, target_ids.unsqueeze(2)).squeeze(2)
        log_probs = log_probs * inv_target_mask
        log_probs = log_probs.sum(dim=1)

        seq_lengths = inv_target_mask.sum(dim=1)
        exp = - (1 / seq_lengths)
        perp = torch.pow(log_probs.exp(), exp)
        return perp.mean()

    def _calc_my_perplexity(self, batch_input, model_output):
        target_ids = batch_input["input_ids"]  # bs, seq_len
        target_mask = batch_input["attention_mask"]  # bs, seq_len
        # seq_len, bs, vocab_size
        vocab_dist_output = model_output["token_output"]

        inv_target_mask = ~(target_mask > 0)  # bs, seq_len

        # seq_len, bs, vocab_size
        l_probs = F.log_softmax(vocab_dist_output, dim=2)
        target_l_probs = l_probs.transpose(
            0, 1).gather(2, target_ids.unsqueeze(2)).squeeze(2)  # bs, seq_len
        target_l_probs = target_l_probs * inv_target_mask
        target_l_probs = target_l_probs.sum(dim=1)

        seq_lengths = inv_target_mask.sum(dim=1)
        neg_normalized_l_probs = -target_l_probs / seq_lengths
        perplexity = torch.pow(2, neg_normalized_l_probs)

        return perplexity.mean(), neg_normalized_l_probs.mean()

    def _predicted_accuracy(self, batch_input, model_output):
        target_ids = batch_input["input_ids"]  # bs, seq_len
        target_mask = batch_input["attention_mask"]  # bs, seq_len
        inv_mask = ~target_mask
        # seq_len, bs, vocab_size
        predicted_logits = model_output["token_output"]

        predicted_ids = torch.argmax(
            predicted_logits, dim=2).transpose(0, 1)  # bs, seq_len

        masked_correct = (predicted_ids == target_ids) & inv_mask
        return torch.sum(masked_correct) / torch.sum(inv_mask)

    def _full_accuracy(self, batch_input, model_output):
        target_ids = batch_input["input_ids"]  # bs, seq_len
        target_mask = batch_input["attention_mask"]  # bs, seq_len
        inv_mask = ~target_mask
        # seq_len, bs, vocab_size
        predicted_logits = model_output["token_output"]

        predicted_ids = torch.argmax(
            predicted_logits, dim=2).transpose(0, 1)  # bs, seq_len

        masked_correct = (predicted_ids == target_ids) & inv_mask
        seq_sum_eq_mask_sum = torch.sum(
            masked_correct, dim=1) == torch.sum(inv_mask, dim=1)
        return seq_sum_eq_mask_sum.float().mean()
    
    def _calc_edit_distance(self, batch_input, model_output):
        target_smiles = batch_input["smiles"]
        token_output = model_output["token_output"]
        output_smiles = self.tokenizer.detokenise(token_output)
        edit_distances = [edit_distance(target, output) for target, output in zip(target_smiles, output_smiles)]
        return torch.tensor(edit_distances).mean()


    def training_step(self, batch, batch_idx):
        collated_smiles = batch

        out = self.forward(batch)
        loss = self._calc_loss(collated_smiles, out)
        with torch.no_grad():
            perplexity = self._calc_perplexity(collated_smiles, out)
            my_perplexity, my_nnll = self._calc_my_perplexity(
                collated_smiles, out)
            accuracy = self._predicted_accuracy(collated_smiles, out)
            full_accuracy = self._full_accuracy(collated_smiles, out)
        self.log("tr/loss", loss)
        metrics = {
            "loss": loss.detach().item(),
            "perplexity": perplexity.detach().item(),
            "my_perplexity": my_perplexity.detach().item(),
            "my_nnll": my_nnll.detach().item(),
            "accuracy": accuracy.detach().item(),
            "full_accuracy": full_accuracy.detach().item()
        }
        self.training_step_outputs.append(metrics)
        return loss

    def validate_or_test_step(self, batch, batch_idx, mode="val"):
        collated_smiles = batch

        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # I dont want to do teacher forcing here!!
        # Be careful, I have batch_first=True but pytorch default is batch_first=False
        # !
        
        out = self.forward(batch)
        loss = self._calc_loss(collated_smiles, out)
        perplexity = self._calc_perplexity(collated_smiles, out)
        my_perplexity, my_nnll = self._calc_my_perplexity(collated_smiles, out)
        accuracy = self._predicted_accuracy(collated_smiles, out)
        full_accuracy = self._full_accuracy(collated_smiles, out)
        edit_distance = self._calc_edit_distance(collated_smiles, out)
        self.log(f"{mode}/loss", loss)
        metrics = {
            "loss": loss.detach().item(),
            "perplexity": perplexity.detach().item(),
            "my_perplexity": my_perplexity.detach().item(),
            "my_nnll": my_nnll.detach().item(),
            "accuracy": accuracy.detach().item(),
            "full_accuracy": full_accuracy.detach().item(),
            "edit_distance": edit_distance.detach().item()
        }
        if mode == "val":
            self.validation_step_outputs.append(metrics)
        elif mode == "test":
            self.test_step_outputs.append(metrics)
        return metrics
    
    
    
    ########
    # Things no need to worry about
    
    
    def validation_step(self, batch, batch_idx):
        return self.validate_or_test_step(batch, batch_idx, mode="val")
    
    def test_step(self, batch, batch_idx):
        return self.validate_or_test_step(batch, batch_idx, mode="test")

    def configure_optimizers(self):
        if not self.scheduler:
            return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay = self.weight_decay)
        elif self.scheduler == "attention":
            optim = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay = self.weight_decay, 
                                     betas=(0.9, 0.98), eps=1e-9)
            
            scheduler = NoamOpt(self.dim_model, self.warm_up_steps, optim, self.noam_factor)
            
            return {
                "optimizer": optim,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": 1,
                }
            }
    
    def on_train_epoch_end(self):
        # return
        if self.training_step_outputs:
            feats = self.training_step_outputs[0].keys()
            di = {}
            for feat in feats:
                di[f"tr/mean_{feat}"] = np.mean([v[feat]
                                                for v in self.training_step_outputs])
            for k, v in di.items():
                self.log(k, v, on_epoch=True)
            self.training_step_outputs.clear()

    def on_validation_epoch_end(self):
        # return
        feats = self.validation_step_outputs[0].keys()
        di = {}
        for feat in feats:
            di[f"val/mean_{feat}"] = np.mean([v[feat]
                                             for v in self.validation_step_outputs])
        for k, v in di.items():
            self.log(k, v, on_epoch=True, prog_bar=k=="val/mean_rank_1")
        self.validation_step_outputs.clear()
        
    def on_test_epoch_end(self):
        feats = self.test_step_outputs[0].keys()
        di = {}
        for feat in feats:
            di[f"test/mean_{feat}"] = np.mean([v[feat]
                                             for v in self.test_step_outputs])
        for k, v in di.items():
            self.log(k, v, on_epoch=True, sync_dist=False)
            # self.log(k, v, on_epoch=True)
        self.test_step_outputs.clear()
