
import logging
import pickle, random
import torch, os, pytorch_lightning as pl, glob
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import pathlib
import selfies
from rdkit.Chem import rdFingerprintGenerator
from rdkit import Chem
from pathlib import Path
from Spectre.inference.inference_utils import  build_input
from Spectre.datasets.dataset_utils import isomeric_to_canonical_smiles


repo_path = str(pathlib.Path(__file__).parent.parent.absolute() / "Spectre")
SELFIES_MAX_LEN = 460

class NMR_Selfie_Dataset(Dataset):
    def __init__(self, split="train", input_src=["HSQC"], symbol_to_idx=None, p_args=None):
        
        
        self.dir = f"/workspace/SMILES_dataset/{split}"
        self.split = split
        self.input_src = input_src
        self.symbol_to_idx = symbol_to_idx
        self.max_len_selfies = SELFIES_MAX_LEN
        if p_args['random_smiles']:
            self.max_len_selfies = 600
        self.p_args = p_args
        logger = logging.getLogger("lightning")

        assert os.path.exists(self.dir), f"{self.dir} does not exist"
        assert(split in ["train", "val", "test"])
        for src in input_src:
            assert os.path.exists(os.path.join(self.dir, src)),"{} does not exist".format(os.path.join(self.dir, src))
            
       
        with open(os.path.join(self.dir, "MW/index.pkl"), "rb") as f:
            self.mol_weight_2d = pickle.load(f)
        with open(os.path.join(self.dir, "SMILES/index.pkl"), "rb") as f:
            self.smiles_2d = pickle.load(f)
            
        if p_args['train_on_all_info_set'] or split in ["val", "test"]:
            logger.info(f"[NMR_Selfie_Dataset]: only all info datasets")
            path_to_load_full_info_indices = f"{repo_path}/datasets/{split}_indices_of_full_info_NMRs.pkl"
            self.files = pickle.load(open(path_to_load_full_info_indices, "rb"))
            print("loaded full info indices\n\n\n")
            # print("36690.pt" in self.files, self.files[0])
            # assert (not p_args['combine_oneD_only_dataset'])
        else:    
            self.files = os.listdir(os.path.join(self.dir, "HSQC")) 
        self.files.sort() # sorted because we need to find correct weight mappings 
        if p_args['combine_oneD_only_dataset']: # load 1D dataset as well 
            self.dir_1d = f"/workspace/OneD_Only_Dataset/{split}"
            
            self.mol_weight_1d = pickle.load(open(os.path.join(self.dir_1d, "MW/index.pkl"), 'rb'))
            self.smiles_1d = pickle.load(open(os.path.join(self.dir_1d, "SMILES/index.pkl"), 'rb'))
            self.files_1d = os.listdir(os.path.join(self.dir_1d, "oneD_NMR/"))
            self.files_1d.sort()
            self.NP_classes_1d = None
            
        if dist.is_initialized():
            rank = dist.get_rank()
            if rank != 0:
                # For any process with rank other than 0, set logger level to WARNING or higher
                logger.setLevel(logging.WARNING)
        logger.info(f"[NMR_Selfie_Dataset]: dir={self.dir},input_src={input_src},split={split}")
        
        if self.p_args['only_C_NMR']:
            def filter_unavailable(x):
                if os.path.exists(os.path.join(self.dir, "oneD_NMR", x)) == False:
                    return False 
                c_tensor, h_tensor = torch.load(f"{self.dir}/oneD_NMR/{x}")
                return len(c_tensor)>0
            self.files = list(filter(filter_unavailable, self.files))
        elif self.p_args['only_H_NMR']:
            def filter_unavailable(x):
                if os.path.exists(os.path.join(self.dir, "oneD_NMR", x)) == False:
                    return False 
                c_tensor, h_tensor = torch.load(f"{self.dir}/oneD_NMR/{x}")
                return len(h_tensor)>0
            self.files = list(filter(filter_unavailable, self.files))
            
        elif self.p_args['only_oneD_NMR']:
            def filter_unavailable(x):
                if os.path.exists(os.path.join(self.dir, "oneD_NMR", x)) == False:
                    return False 
                c_tensor, h_tensor = torch.load(f"{self.dir}/oneD_NMR/{x}")
                return len(h_tensor)>0 and len(c_tensor)>0
            self.files = list(filter(filter_unavailable, self.files))

        logger.info(f"[NMR_Selfie_Dataset]: dataset size is {len(self)}")
        
        # print("\n\n\n", "36690.pt" in self.files, self.files[0])
        
        
        
    def __len__(self):
        if self.p_args['debug'] or self.p_args.get('foldername') == "debug":
            return 100
        length = len(self.files)
        if self.p_args['combine_oneD_only_dataset']:
            length += len(self.files_1d)
        return length
        


    def __getitem__(self, idx):
        
        if idx >= len(self.files): 
        ### load 1D dataset
            current_dataset = "1d"
            i = idx - len(self.files)
            # hsqc is empty tensor
            hsqc = torch.empty(0,3)
            c_tensor, h_tensor = torch.load(f"{self.dir_1d}/oneD_NMR/{self.files_1d[i]}")
            if self.p_args['jittering'] >0 and self.split=="train":
                jittering = self.p_args['jittering']
                c_tensor = c_tensor + torch.randn_like(c_tensor) * jittering
                h_tensor = h_tensor + torch.randn_like(h_tensor) * jittering * 0.1
                
            # input dropout
            if self.p_args['optional_inputs'] and len(c_tensor) > 0 and len(h_tensor) > 0:
                if not self.p_args['combine_oneD_only_dataset'] :
                    raise NotImplementedError("optional_inputs is only supported when combine_oneD_only_dataset is True")
                random_num = random.random()
                if random_num <= 0.3984: # drop C rate 
                    c_tensor = torch.tensor([]) 
                elif random_num <= 0.3984+0.2032: # drop H rate
                    h_tensor = torch.tensor([])
                    
        else :
        ### BEGINNING 2D dataset case
            current_dataset = "2d"
            i = idx
            def file_exist(src, filename):
                return os.path.exists(os.path.join(self.dir, src, filename))
            
            # Load HSQC as sequence
            if "HSQC" in self.input_src:
                hsqc = torch.load(f"{self.dir}/HSQC/{self.files[i]}").type(torch.FloatTensor)
                if self.p_args['jittering'] >0  and self.split=="train":
                    jittering = self.p_args['jittering']
                    hsqc[:,0] = hsqc[:,0] + torch.randn_like(hsqc[:,0]) * jittering
                    hsqc[:,1] = hsqc[:,1] + torch.randn_like(hsqc[:,1]) * jittering * 0.1
   
                inputs = hsqc
            
            c_tensor, h_tensor = (torch.tensor([]) , torch.tensor([])) 
            if "oneD_NMR" in self.input_src:
                if file_exist("oneD_NMR", self.files[i]):
                    c_tensor, h_tensor = torch.load(f"{self.dir}/oneD_NMR/{self.files[i]}")  
                    if self.p_args['jittering'] >0 and self.split=="train":
                        jittering = self.p_args['jittering']
                        c_tensor = c_tensor + torch.randn_like(c_tensor) * jittering
                        h_tensor = h_tensor + torch.randn_like(h_tensor) * jittering * 0.1
                    # randomly drop 1D and 2D NMRs if needed
                    if self.p_args['optional_inputs']:
                        # DO NOT drop 2D, cuz we have enough amount of 1D data 
     
                        if len(c_tensor)>0 and len(h_tensor)>0:
                            # it is fine to drop one of the oneD NMRs
                            random_num_for_dropping =  random.random()                            
                            if random_num_for_dropping <= 0.3530:# drop C rate
                                c_tensor = torch.tensor([])
                            elif random_num_for_dropping <= 0.3530+0.2939: # drop H rate
                                h_tensor = torch.tensor([])
                        if random.random() <= 0.5:
                            # the last column of HSQC is all 0
                            hsqc[:,2] = 0
                                          
                        assert (len(hsqc) > 0 or len(c_tensor) > 0 or len(h_tensor) > 0), "all NMRs are dropped"
                            
            if self.p_args['only_oneD_NMR']:
                hsqc = torch.empty(0,3)                
            if self.p_args['only_C_NMR']:
                h_tensor = torch.tensor([])
            if self.p_args['only_H_NMR']:
                c_tensor = torch.tensor([])
        ### ENDING 2D dataset case
                
            
        # loading MW and in different datasets 
        if idx >= len(self.files) : # load 1D dataset    
            if self.p_args['use_MW']:
                mol_weight_dict = self.mol_weight_1d
            smiles_dict = self.smiles_1d
            dataset_files = self.files_1d
            dataset_dir = self.dir_1d

        else:
            if self.p_args['use_MW']:
                mol_weight_dict = self.mol_weight_2d
            smiles_dict = self.smiles_2d
            dataset_files = self.files
            dataset_dir = self.dir

            
        mol_weight = None
        if self.p_args['use_MW']:
            mol_weight = mol_weight_dict[int(dataset_files[i].split(".")[0])]
            mol_weight = torch.tensor([mol_weight,0,0]).float()
            
            if self.p_args['optional_inputs'] and self.p_args["optional_MW"]:
                if random.random() <= 0.5:
                    mol_weight = torch.tensor([])
        
        smiles = smiles_dict[int(dataset_files[i].split(".")[0])] 
        try :
            
            if self.split == "train" and self.p_args['random_smiles'] > 0:
                mol = Chem.MolFromSmiles(smiles)
                smiles_to_get_selfie = Chem.MolToSmiles(mol, canonical=False, doRandom=True)
            else:
                smiles_to_get_selfie = smiles
            selfies_str = selfies.encoder(smiles_to_get_selfie)   
        except:
            return None
        encoded_selfie = [self.symbol_to_idx[symbol] for symbol in selfies.split_selfies(selfies_str)] 
        encoded_selfie = [        self.symbol_to_idx['[START]']] \
                                + encoded_selfie \
                                + [self.symbol_to_idx['[END]']] \
                                + [self.symbol_to_idx['[PAD]'] for _ in range(self.max_len_selfies - 2 - len(encoded_selfie))]
        
        # padding 1D NMRs and stacking： 
        inputs, NMR_type_indicator = self.pad_and_stack_input(hsqc, c_tensor, h_tensor, mol_weight)
        
        
        combined = [inputs, NMR_type_indicator, encoded_selfie]
        if self.split in ["val", "test"]:
            # if self.p_args['combine_oneD_only_dataset']:
            #     combined.append(self.files[i])
            # else:
            combined.append(smiles)
            combined.append(gen_mfp(smiles))
        return combined
    
    def pad_and_stack_input(self, hsqc, c_tensor, h_tensor, mol_weight):
        '''
        embedding mapping:
        0: HSQC
        1: C NMR
        2: H NMR
        3: MW
        4: normal hsqc
        
        future:
        Mass Spectrometry
        
        '''
        
        c_tensor, h_tensor = c_tensor.view(-1, 1), h_tensor.view(-1, 1)
        c_tensor,h_tensor = F.pad(c_tensor, (0, 2), "constant", 0), F.pad(h_tensor, (1, 1), "constant", 0)
        inputs = [hsqc, c_tensor, h_tensor]
        # hsqc_type = 4 if (hsqc[:2]==0).all() else 0 # if multiplicity is all 0s, it is normal hsqc
        hsqc_type = 0
        NMR_type_indicator = [hsqc_type]*len(hsqc)+[1]*len(c_tensor)+[2]*len(h_tensor)
        if mol_weight is not None and len(mol_weight) > 0:
            inputs.append(mol_weight)
            NMR_type_indicator.append(3)
            
        inputs = torch.vstack(inputs)               
        NMR_type_indicator = torch.tensor(NMR_type_indicator).long()
        return inputs, NMR_type_indicator
    
    
class NovelMolDataset(Dataset):
    def __init__(self, symbol_to_idx=None, p_args=None):
        # self.dir = Path
        file_dir = Path(__file__).parent.parent / "Spectre" / "datasets" / "testing_compounds"
        self.file_dir = file_dir
        self.compound_dirs = os.listdir(file_dir)
        self.compound_dirs = [d for d in self.compound_dirs if os.path.isdir(file_dir / d)]
        self.compound_dirs.sort()
        self.symbol_to_idx = symbol_to_idx
        self.max_len_selfies = SELFIES_MAX_LEN
        self.p_args = p_args
    
    def __len__(self):
        return len(self.compound_dirs)
    
    def __getitem__(self, idx):
        compound_dir = self.file_dir / self.compound_dirs[idx]
        inputs, NMR_type_indicator = build_input(compound_dir, 
                                                 include_hsqc = self.p_args["use_HSQC"], 
                                                 include_c_nmr = self.p_args["use_C_NMR"], 
                                                 include_h_nmr = self.p_args["use_H_NMR"], 
                                                 include_MW = self.p_args["use_MW"],)
        
        smiles_file = compound_dir / "SMILES.txt"
        with open(smiles_file, "r") as f:
            smiles = f.read().strip()
            
        smiles = isomeric_to_canonical_smiles(smiles)
        
        selfies_str = selfies.encoder(smiles) 
        encoded_selfie = [self.symbol_to_idx[symbol] for symbol in selfies.split_selfies(selfies_str)] 
        encoded_selfie = [        self.symbol_to_idx['[START]']] \
                                + encoded_selfie \
                                + [self.symbol_to_idx['[END]']] \
                                + [self.symbol_to_idx['[PAD]'] for _ in range(self.max_len_selfies - 2 - len(encoded_selfie))]
                                
        combined = [inputs, NMR_type_indicator, encoded_selfie, smiles ,gen_mfp(smiles)]
        return combined
    
def gen_mfp(smiles):
    MFP_generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    mol = Chem.MolFromSmiles(smiles)
    fp = MFP_generator.GetFingerprint(mol)
    return torch.tensor(fp).float()
                
# used as collate_fn in the dataloader
def collate_fn(batch):
    
    # dataloader may give a None if we met error when converting SMILES to selfie
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    items = tuple(zip(*batch))
    # items[0] : NMRs
    # items[1] : NMR_type_indicator
    # items[2] : smiles
    NMRs = items[0]
    NMRs = pad_sequence([v for v in NMRs], batch_first=True) 
    NMR_type_indicator = pad_sequence([v for v in items[1]], batch_first=True)
    # print(items[2])
    selfie_token_idxs = torch.tensor(items[2]) # no padding needed, padding is done in get_item

    if len(items) == 3:
        combined = (NMRs, NMR_type_indicator, selfie_token_idxs)
        return combined
    elif len(items) == 5: # also smiles string and mfp
        SMILESs = items[3]
        MFPs = torch.stack(items[4])
        combined = (NMRs, NMR_type_indicator, selfie_token_idxs, SMILESs, MFPs)
        return combined
    
class NMR_Selfie_DataModule(pl.LightningDataModule):
    def __init__(self, input_src, batch_size, p_args=None, persistent_workers = True):
        super().__init__()
        self.batch_size = batch_size
        self.input_src = input_src
        self.collate_fn = collate_fn
        self.p_args = p_args
        self.should_persist_workers = persistent_workers
        
        # selfie tokenizer 
        tokenizer_path = pathlib.Path(__file__).parent / 'data' / 'selfies_tokenizer'
        # load symbol_to_idx and idx_to_symbol
        with open(tokenizer_path / 'symbol_to_idx.pkl', 'rb') as f:
            self.symbol_to_idx = pickle.load(f)
        # with open(tokenizer_path / 'idx_to_symbol.pkl', 'rb') as f:
        #     self.idx_to_symbol = pickle.load(f)
            
    def setup(self, stage):
        if stage == "fit" or stage == "validate" or stage is None:
            self.train = NMR_Selfie_Dataset(input_src=self.input_src, split="train", 
                                            symbol_to_idx=self.symbol_to_idx, 
                                            # idx_to_symbol=self.idx_to_symbol,
                                            p_args=self.p_args)
            self.val = NMR_Selfie_Dataset(input_src=self.input_src, split="val", 
                                            symbol_to_idx=self.symbol_to_idx, 
                                            # idx_to_symbol=self.idx_to_symbol,
                                            p_args=self.p_args)
        if stage == "test":
            self.test = NMR_Selfie_Dataset(input_src=self.input_src, split="test", 
                                            symbol_to_idx=self.symbol_to_idx, 
                                            # idx_to_symbol=self.idx_to_symbol,
                                            p_args=self.p_args)
        if stage == "predict" :
            self.test_novel = NovelMolDataset(symbol_to_idx=self.symbol_to_idx, p_args=self.p_args)

    def train_dataloader(self):
            
        return DataLoader(self.train, shuffle=True, batch_size=self.batch_size, collate_fn=self.collate_fn,
                          num_workers=self.p_args['num_workers'], pin_memory=True, persistent_workers=self.should_persist_workers)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, collate_fn=self.collate_fn, 
                          num_workers=self.p_args['num_workers'], pin_memory=True, persistent_workers=self.should_persist_workers)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, collate_fn=self.collate_fn, 
                          num_workers=self.p_args['num_workers'], pin_memory=True, persistent_workers=self.should_persist_workers)
    
    def predict_dataloader(self):
        return DataLoader(self.test_novel, batch_size=1, collate_fn=self.collate_fn, 
                          num_workers=self.p_args['num_workers'], pin_memory=True, persistent_workers=self.should_persist_workers)
