import pandas as pd
import pickle
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from data.utils import *
import os
class CPIDataset(Dataset):
    def __init__(self, args, type):
        self.csv_file = os.path.join(args.data_dir, args.dataset_tag, f'{type}.csv')
        self.dict_target_dir = args.dict_target
        self.dict_ligand_dir = args.dict_ligand
        self.fetch_pretrained_target = args.fetch_pretrained_target
        self.fetch_pretrained_ligand = args.fetch_pretrained_ligand

        self.dataset_raw = pd.read_csv(self.csv_file)
        if self.fetch_pretrained_target:
            with open(self.dict_target_dir, 'rb') as f:
                self.dict_target = pickle.load(f)
        if self.fetch_pretrained_ligand:
            with open(self.dict_ligand_dir, 'rb') as f:
                self.dict_ligand = pickle.load(f)
        self.id_target = pd.read_csv('/datapool/data2/home/majianzhu/xinheng/xiangzhen/DTI2/DTI/data/idmapping_target.csv')
        self.id_target.set_index('Target UniProt ID', inplace=True)
        self.id_ligand = pd.read_csv('/datapool/data2/home/majianzhu/xinheng/xiangzhen/DTI2/DTI/data/idmapping_ligand.csv')
        self.id_ligand.set_index('Ligand ID', inplace=True)
    def __len__(self):
        return len(self.dataset_raw)

    def __getitem__(self, idx):
        # fetch
        row = self.dataset_raw.iloc[idx]
        protein_id = row['Target UniProt ID']
        ligand_id = row['Ligand ID']
        label = row['Label']
        # process
        label = torch.tensor(label, dtype=torch.int64).clone().detach()
        if self.fetch_pretrained_target:
            protein_data = self.dict_target[protein_id].clone().detach().to(torch.float32)
        else:
            target_sequence = self.id_target.loc[protein_id, 'Target Sequence']
            # protein_data = create_target_data(target_sequence)
        if self.fetch_pretrained_ligand:
            ligand_data = torch.tensor(self.dict_ligand[ligand_id], dtype=torch.float32).clone().detach()
        else:
            ligand_smiles = self.id_ligand.loc[ligand_id, 'SMILES']
            ligand_data = create_graph_data(ligand_smiles)
        return protein_data, ligand_data, label


