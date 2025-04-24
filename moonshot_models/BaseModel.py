import torch
import pytorch_lightning as pl
from utils.transformer_utils import generate_square_subsequent_mask
from torchaudio.functional import edit_distance 
import math
from torch.nn.functional import cross_entropy

class BaseModel(pl.LightningModule):

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

    def _calc_perplexity(self, batch_input, model_output):
        
        # labels = batch_input['smiles']
        input_ids = batch_input['input_ids']
        logits = model_output.logits
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        perplexity = cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        perplexity = math.exp(perplexity)
        return perplexity

    def _predicted_accuracy(self, batch_input, model_output):
        # labels = batch_input['smiles']
        input_ids = batch_input['input_ids']
        logits = model_output.logits
        preds = torch.argmax(logits, dim=-1) 
        # Calculate accuracy
        non_pad_mask = input_ids != self.tokenizer.pad_token_id
        num_correct = (preds == input_ids) & non_pad_mask
        num_correct = num_correct.sum()
        num_tokens = non_pad_mask.sum()       
        accuracy = num_correct.float() / num_tokens.float()
        return accuracy

    
    # def _calc_edit_distance(self, batch_input):
    #     target_smiles = batch_input["smiles"]
    #     inputs = batch_input['input_ids']
    #     attention_mask = batch_input["attention_mask"]
    #     # output_smiles = self.tokenizer.detokenise(token_output)
    #     # self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    #     generated_ids = self.model.generate(inputs, attention_mask=attention_mask, max_length=int(inputs.size(1)*2))
    #     generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    #     edit_distances = [edit_distance(target, output) for target, output in zip(target_smiles, generated_text)]
    #     return torch.tensor(edit_distances).float().mean()

    def _calc_edit_distance_half_given(self, batch_input):
        target_smiles = batch_input["smiles"]
        inputs = batch_input['input_ids'] # shape: batch * seq_len
        attention_mask = batch_input["attention_mask"] # shape: batch * seq_len
        smiles_lengths = attention_mask.sum(dim=1)
        half_point = smiles_lengths // 2
        
        for row_idx, half_point_val in enumerate(half_point):
            inputs[row_idx, half_point_val:] = self.tokenizer.pad_token_id
            attention_mask[row_idx, half_point_val:] = False
        # output_smiles = self.tokenizer.detokenise(token_output)
        # self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        
        generated_ids = self.model.generate(inputs, 
                                            attention_mask=attention_mask, 
                                            max_length=int(inputs.size(1)*2),
                                            num_return_sequences=1
                                            )
        generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        edit_distances = [edit_distance(target, output) for target, output in zip(target_smiles, generated_text)]
        return torch.tensor(edit_distances).float().mean()
        

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
        
    def log(self, name, value, *args, **kwargs):
        # Set 'sync_dist' to True by default
        kwargs['sync_dist'] = True
        
        super().log(name, value, *args, **kwargs)
