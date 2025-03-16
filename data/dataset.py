import pandas as pd
import pickle
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from data.utils import *
import os
from esm.models.esm3 import ESM3
from esm.sdk.api import ESMProtein, SamplingConfig
from esm.utils.constants.models import ESM3_OPEN_SMALL
import torch
import pickle
import pandas as pd
from tqdm import tqdm
class CPIDataset(Dataset):
    def __init__(self, args, type):
        if type == 'predict':
            self.csv_file = args.input_data_dir
        else:
            self.csv_file = os.path.join(args.data_dir, args.dataset_tag, f'{type}.csv')
        self.dict_target_dir = args.dict_target
        self.dict_ligand_dir = args.dict_ligand
        self.fetch_pretrained_target = args.fetch_pretrained_target
        self.fetch_pretrained_ligand = args.fetch_pretrained_ligand

        self.id_target = pd.read_csv('data/idmapping_target.csv')
        self.id_target.set_index('Target UniProt ID', inplace=True)
        self.id_ligand = pd.read_csv('data/idmapping_ligand.csv')
        self.id_ligand.set_index('Ligand ID', inplace=True)

        if os.path.exists(self.dict_target_dir):
            with open(self.dict_target_dir, 'rb') as f:
                self.dict_target = pickle.load(f)
        else:
            print(f'Not exist: dict_protein, process...')
            self.dict_target = self.get_pretrain_feature(args)
            print(f'Process done...')
        # self.dict_target = self.get_pretrain_feature(args)
        self.dataset_raw = pd.read_csv(self.csv_file)

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
    
    def get_pretrain_feature(self, args):
        # client = ESM3.from_pretrained(ESM3_OPEN_SMALL, device=torch.device("cpu"))
        device = torch.device(args.cuda_use)
        client = ESM3.from_pretrained(ESM3_OPEN_SMALL, device=device)
        data = self.id_target
        print(f'Process protein nums: {len(data)}...')
        dict_target = {}
        for _, row in tqdm(data.iterrows()):
            protein_id = row.name 
            sequence_item = self.id_target.loc[protein_id, 'Target Sequence']
            protein = ESMProtein(
                sequence=sequence_item
            )
            protein_tensor = client.encode(protein)

            output = client.forward_and_sample(
                protein_tensor, SamplingConfig(return_per_residue_embeddings=True)
            )
            val = output.per_residue_embedding
            dict_target[protein_id] = val
        with open(self.dict_target_dir, 'wb') as f:
            pickle.dump(dict_target, f)

        return dict_target


