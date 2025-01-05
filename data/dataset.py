import pandas as pd
import pickle
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from data.utils import *
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
        ligand_2d_tensor = self.create_graph_data(ligand_id)
        label = torch.tensor(label, dtype=torch.int64).clone().detach()
        protein_data = Data(x=protein_tensor)
        ligand_1d_data = Data(x=ligand_1d_tensor)
        ligand_2d_data = ligand_2d_tensor

        return protein_data, ligand_1d_data, ligand_2d_data, label

    def create_graph_data(self, ligand_id):
        ligand_smiles = self.id_ligand.loc[ligand_id, 'SMILES']
        n_features, edge_index = get_mol_edge_list_and_feat_mtx(ligand_smiles)
        return Data(x=n_features, edge_index=edge_index)

