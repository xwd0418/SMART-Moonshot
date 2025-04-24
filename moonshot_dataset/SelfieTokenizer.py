import selfies

import pickle, os
import pathlib


import tqdm
def get_all_train_val_selfies():
    max_len = 0     
    all_selfies = []
    alphabet = set()
    
    for split in ['train', 'val',]:
        dir_1d = f'/workspace/OneD_Only_Dataset/'
        dir = f'/workspace/SMILES_dataset/'
        
        # 2d dataset
        files = os.listdir(os.path.join(dir, split, "HSQC"))
        with open(f'/workspace/SMILES_dataset/{split}/SMILES/index.pkl', 'rb') as f:
            smiles = pickle.load(f)
        for f in tqdm.tqdm(files):
            smile_idx = int(f.split('.')[0])
            smiles_str = smiles[smile_idx]
            try:
                selfies_str = selfies.encoder(smiles_str)
            except:
                print(f"Error encoding SMILES: {smiles_str}")
                continue
            splitted_selfies = list(selfies.split_selfies(selfies_str))
            alphabet.update(splitted_selfies)
            max_len = max(max_len, len(splitted_selfies))
            all_selfies.append(selfies_str)
      
        #1d dataset
        files = os.listdir(os.path.join(dir_1d, split, "oneD_NMR"))
        with open(f'/workspace/OneD_Only_Dataset/{split}/SMILES/index.pkl', 'rb') as f:
            smiles = pickle.load(f)
        for f in tqdm.tqdm(files):
            smile_idx = int(f.split('.')[0])
            smiles_str = smiles[smile_idx]
            try:
                selfies_str = selfies.encoder(smiles_str)
            except:
                print(f"Error encoding SMILES: {smiles_str}")
                continue
            splitted_selfies = list(selfies.split_selfies(selfies_str))
            alphabet.update(splitted_selfies)
            max_len = max(max_len, len(splitted_selfies))
            all_selfies.append(selfies_str)
    
    alphabet = ['[PAD]','[START]','[END]'] + list(sorted(alphabet))
    print(f"Max length of SELFIES: {max_len}")
    return all_selfies, max_len, alphabet

if __name__ == "__main__":
    all_selfies, max_len, alphabet = get_all_train_val_selfies()
    
    symbol_to_idx = {s: i for i, s in enumerate(alphabet)}
    idx_to_symbol = {i: s for i, s in enumerate(alphabet)}
    
    # save the dicts
    save_path = pathlib.Path(__file__).parent / 'data' / 'selfies_tokenizer'
    print(f"save_path: {save_path}")
    os.makedirs(save_path, exist_ok=True)
    with open(save_path / 'symbol_to_idx.pkl', 'wb') as f:
        pickle.dump(symbol_to_idx, f)
    with open(save_path / 'idx_to_symbol.pkl', 'wb') as f:
        pickle.dump(idx_to_symbol, f)
        
    # After running this script, we get max len of selfies is 455