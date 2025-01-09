import pandas as pd
import pickle
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
import torch.nn.utils.rnn as rnn_utils
class CPIDataset(Dataset):
    def __init__(self, csv_file, dict_target, dict_ligand):
        self.dataset_raw = pd.read_csv(csv_file)
        self.id_ligand = pd.read_csv('/datapool/data2/home/majianzhu/xinheng/xiangzhen/DTI2/DTI/data/idmapping_ligand.csv')
        self.id_ligand.set_index('Ligand ID', inplace=True)
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
        ligand_1d_tensor = torch.tensor(self.dict_ligand[ligand_id], dtype=torch.float32).clone().detach()
        label = torch.tensor(label, dtype=torch.int64).clone().detach()

        return protein_tensor, ligand_1d_tensor, label

def custom_collate_fn(batch):
    proteins, ligands, labels = zip(*batch)
    padded_proteins = rnn_utils.pad_sequence(proteins, batch_first=True, padding_value=0)
    padded_ligands = rnn_utils.pad_sequence(ligands, batch_first=True, padding_value=0)
    protein_mask = torch.arange(padded_proteins.size(1))[None, :] < torch.tensor([p.size(0) for p in proteins])[:, None]
    ligand_mask = torch.arange(padded_ligands.size(1))[None, :] < torch.tensor([l.size(0) for l in ligands])[:, None]
    labels = torch.stack(labels)
    return padded_proteins, padded_ligands, protein_mask, ligand_mask, labels
