import pickle
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, decoders, trainers, processors
from transformers import PreTrainedTokenizerFast


DATASET_DIR = '/root/Moonshot/SMART-Moonshot/dataset'

class SMILESTokenizerBuilder():
    def __init__(self, *args, **kwargs):
        # Define special tokens if needed
        self.special_tokens = ["[PAD]", "[CLS]", "[SEP]", "[MASK]", "[UNK]"]
        
        # Initialize the tokenizer from a simple custom tokenizer
        self.tokenizer = Tokenizer(models.BPE())
        
        # # Normalization: You can add custom normalization steps here
        # tokenizer.normalizer = normalizers.Sequence([

        # ])
        
        # Pre-tokenization: Define how SMILES strings are split into tokens
        self.tokenizer.pre_tokenizer = pre_tokenizers.Split(pattern=r'(\[[^\[\]]*\]|Br|Cl|Si|@@|[=#$()NOPS])',
                                                            behavior='isolated')
        # (\[[^\[\]]*\]): Matches substrings that are enclosed in square brackets.
        # Br|Cl|Si:       two-letter symbols.
        # @@ :            indicate stereochemistry.
        # [=#$()NOPS]:
        #          =, #, $: Common bond types
        #             (, ): Parentheses used for branching.
        #       N, O, P, S: Single-letter element symbols for Nitrogen, Oxygen, Phosphorus, and Sulfur.
        
        # Decoder: Define how tokens are decoded back into a string
        self.tokenizer.decoder = decoders.BPEDecoder()
        
        
    def build_vocab_and_save(self, smiles_list):
        trainer = trainers.BpeTrainer(special_tokens=self.special_tokens, vocab_size=1000)
        
        # def batch_iterator(batch_size=10000):
        #     for i in range(0, len(smiles_list), batch_size):
        #         yield smiles_list[i : i + batch_size]
        # self.tokenizer.train_from_iterator(batch_iterator(), trainer=trainer)
        
        self.tokenizer.train_from_iterator(smiles_list, trainer=trainer)

        self.tokenizer.post_processor = processors.TemplateProcessing(
            single=f"[CLS] $A [SEP]",
            pair=f"[CLS] $A [SEP] $B:1 [SEP]:1",
            special_tokens=[
                ("[CLS]", self.tokenizer.token_to_id("[CLS]")),
                ("[SEP]", self.tokenizer.token_to_id("[SEP]")),
            ],
        )
        self.tokenizer.save(f"{DATASET_DIR}/SMILES_shorter_than_300_tokenizer.json")

    def load_tokenizer(self):
        # self.tokenizer = Tokenizer.from_file(f"{DATASET_DIR}/SMILES_shorter_than_300_tokenizer.json")
        self.tokenizer = PreTrainedTokenizerFast(tokenizer_file=f"{DATASET_DIR}/SMILES_shorter_than_300_tokenizer.json")
        return self.tokenizer
        


# main func  
if  __name__ == "__main__":
    # Load the dataset
    with open(f"{DATASET_DIR}/train_smiles_shorter_than_300.pkl", 'rb') as f:
        smiles = pickle.load(f)
#     smiles = [
#     "CC(=O)N[C@H](C)C(=O)O",  # Example SMILES string
#     "C1=CC=CC=C1",            # Benzene
#     "O=C(NCC1=CC=CC=C1)O",    # Another example
# ]
    # Initialize the tokenizer
    
    SMILESTokenizerBuilder().build_vocab_and_save(smiles)

