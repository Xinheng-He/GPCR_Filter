import pandas as pd
import pickle
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

class CPIDataset(Dataset):
    def __init__(self, csv_file, dict_target, dict_ligand):
        self.dataset_raw = pd.read_csv(csv_file)
        with open(dict_target, 'rb') as f:
            self.dict_target = pickle.load(f)
        with open(dict_ligand, 'rb') as f:
            self.dict_ligand = pickle.load(f)
        
    def __len__(self):
        return len(self.dataset_raw)

    def __getitem__(self, idx):
        row = self.dataset_raw.iloc[idx]
        protein_id = row['Target UniProt ID']
        ligand_id = row['Ligand ID']
        label = row['Label']

        protein_tensor = self.dict_target[protein_id].clone().detach().to(torch.float32)
        ligand_tensor = torch.tensor(self.dict_ligand[ligand_id], dtype=torch.float32).clone().detach()
        label = torch.tensor(label, dtype=torch.int64).clone().detach()
        protein_data = Data(x=protein_tensor)
        ligand_data = Data(x=ligand_tensor)

        return protein_data, ligand_data, label

