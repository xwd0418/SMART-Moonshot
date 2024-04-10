import logging
import pytorch_lightning as pl
import torch, pickle
import math
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import torch.distributed as dist


class SmilePretrainingDecoder():
    """
      Only parameters, no sampling
    """

    def __init__(self,
                 pad_token_idx,
                 vocab_size,
                 dim_model,
                 num_layers,
                 num_heads,
                 ff_dim,
                 lr,
                 weight_decay,
                 activation,
                 num_steps,
                 max_seq_len,
                 # schedule,
                 warm_up_steps,
                 tokeniser,
                 dropout,
                 parser_args,
                 *args,
                 **kwargs):
        super().__init__(save_params=False, *args, **kwargs)

        self.save_hyperparameters(**parser_args, logger=True)

        # attributes
        self.pad_token_idx = pad_token_idx
        self.vocab_size = vocab_size
        self.dim_model = dim_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        self.max_seq_len = max_seq_len
        self.tokeniser = tokeniser

        # TransformerDecoderModel
        self.emb = nn.Embedding(vocab_size, dim_model, padding_idx=pad_token_idx)
        dec_norm = nn.LayerNorm(dim_model)
        # dec_layer = PreNormDecoderLayer(
        #     dim_model, num_heads, ff_dim, dropout, activation)
        dec_layer = nn.TransformerDecoderLayer(dim_model, num_heads, ff_dim)
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers, norm=dec_norm)
        self.token_fc = nn.Linear(dim_model, vocab_size)
        self.loss_fn = nn.CrossEntropyLoss(
            reduction="none", ignore_index=pad_token_idx)
        self.log_softmax = nn.LogSoftmax(dim=2)
        self.register_buffer("pos_emb", self._positional_embs())

    # Ripped from chemformer
    def _construct_input(self, token_ids):
        """
          Expects tokens in (seq_len, b_s) format

        Returns:
          (seq_len, b_s, dim_model) embedding (with dropout applied)
        """
        seq_len, _ = tuple(token_ids.size())
        token_embs = self.emb(token_ids)

        # Scaling the embeddings like this is done in other transformer libraries
        token_embs = token_embs * math.sqrt(self.dim_model)

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
    #     tgt_mask = self.generate_square_subsequent_mask(s_l).to(self.device)
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
        decoder_inputs = collated_smiles["decoder_inputs"]
        decoder_mask = collated_smiles["decoder_mask"]
        assert(decoder_inputs.size() == decoder_mask.size())
        
        decoder_embs = self._construct_input(decoder_inputs)
        
        tgt_mask = self.generate_square_subsequent_mask(decoder_inputs.size(1)).to(self.device)
        
        model_output = self.decoder(
            decoder_embs,
            memory=None,
            tgt_mask=tgt_mask,  # prevent cheating mask
            tgt_key_padding_mask=decoder_mask,  # padding mask
        )

        token_output = self.token_fc(model_output)
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

        target = batch_input["target"]  # (b_s, s_l)
        target_mask = batch_input["target_mask"]  # (b_s, s_l)
        token_output = model_output["token_output"]  # (s_l, b_s, vocab_size)

        assert (target.size()[0] == token_output.size()[1])

        batch_size, seq_len = tuple(target.size())

        token_pred = token_output.reshape((seq_len * batch_size, -1)).float()
        loss = self.loss_fn(
            token_pred, target.reshape(-1)
        ).reshape((seq_len, batch_size))

        inv_target_mask = ~(target_mask > 0)
        num_tokens = inv_target_mask.sum()
        loss = loss.sum() / num_tokens

        return loss

    def _calc_perplexity(self, batch_input, model_output):
        target_ids = batch_input["target"]  # bs, seq_len
        target_mask = batch_input["target_mask"]  # bs, seq_len
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
        target_ids = batch_input["target"]  # bs, seq_len
        target_mask = batch_input["target_mask"]  # bs, seq_len
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
        target_ids = batch_input["target"]  # bs, seq_len
        target_mask = batch_input["target_mask"]  # bs, seq_len
        inv_mask = ~target_mask
        # seq_len, bs, vocab_size
        predicted_logits = model_output["token_output"]

        predicted_ids = torch.argmax(
            predicted_logits, dim=2).transpose(0, 1)  # bs, seq_len

        masked_correct = (predicted_ids == target_ids) & inv_mask
        return torch.sum(masked_correct) / torch.sum(inv_mask)

    def _full_accuracy(self, batch_input, model_output):
        target_ids = batch_input["target"]  # bs, seq_len
        target_mask = batch_input["target_mask"]  # bs, seq_len
        inv_mask = ~target_mask
        # seq_len, bs, vocab_size
        predicted_logits = model_output["token_output"]

        predicted_ids = torch.argmax(
            predicted_logits, dim=2).transpose(0, 1)  # bs, seq_len

        masked_correct = (predicted_ids == target_ids) & inv_mask
        seq_sum_eq_mask_sum = torch.sum(
            masked_correct, dim=1) == torch.sum(inv_mask, dim=1)
        return seq_sum_eq_mask_sum.float().mean()

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

    def validation_step(self, batch, batch_idx):
        _, collated_smiles = batch

        out = self.forward(batch)
        loss = self._calc_loss(collated_smiles, out)
        perplexity = self._calc_perplexity(collated_smiles, out)
        my_perplexity, my_nnll = self._calc_my_perplexity(collated_smiles, out)
        accuracy = self._predicted_accuracy(collated_smiles, out)
        full_accuracy = self._full_accuracy(collated_smiles, out)
        metrics = {
            "loss": loss.detach().item(),
            "perplexity": perplexity.detach().item(),
            "my_perplexity": my_perplexity.detach().item(),
            "my_nnll": my_nnll.detach().item(),
            "accuracy": accuracy.detach().item(),
            "full_accuracy": full_accuracy.detach().item()
        }
        self.validation_step_outputs.append(metrics)
        return metrics
