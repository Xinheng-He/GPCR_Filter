from torch.utils.data import Dataset, DataLoader
import pandas as pd
import pickle
import torch
import os

class CPIDataset(Dataset):
    def __init__(self, csv_file, dict_target, dict_ligand):
        self.dataset_raw = pd.read_csv(csv_file)
        # self.idmapping_ligand = pd.read_csv(os.path.join('data', 'idmapping_ligand.csv'))
        # self.idmapping_target = pd.read_csv(os.path.join('data', 'idmapping_target.csv'))
        with open(dict_target, 'rb') as f:
            self.dict_target = pickle.load(f)
        with open(dict_ligand, 'rb') as f:
            self.dict_ligand = pickle.load(f)
        for key, val in self.dict_target.items():
            val_mean = torch.mean(val, dim=0, keepdim=False)
            self.dict_target[key] = val_mean
        wrong_list = ['5617', '25395', '30099', '33848', '38706', '39020', '41526', '46371', '56786', '60853', '61837', '69875', '70076', '70494']
        wrong_list = [f'drug{i}' for i in wrong_list]
        self.dict_ligand_new = {}
        for key, val in self.dict_ligand.items():
            if key in wrong_list:
                continue
            val_dtype = val.to(torch.float32)
            self.dict_ligand_new[key] = val_dtype
        self.dataset = [
            (
                self.dict_target[row['Target UniProt ID']], 
                self.dict_ligand_new[row['Ligand ID']],           
                torch.tensor(row['Label'], dtype=torch.int64) 
            )
            for _, row in self.dataset_raw.iterrows()
        ]
        
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        return self.dataset[idx]