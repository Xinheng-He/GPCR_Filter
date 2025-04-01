import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from model import Predictor
from data.dataset import CPIDataset
from data.utils import make_masks_protein
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import argparse
import numpy as np
from datetime import datetime
import os
import random
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem import Draw

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_data_dir', type=str, default='data')
    parser.add_argument('--output_data_dir', type=str, default='data')
    parser.add_argument('--dict_target', type=str, default='data/dict_target.pkl')
    parser.add_argument('--dict_ligand', type=str, default='data/dict_ligand.pkl')
    parser.add_argument('--batchsize', type=int, default=128)
    parser.add_argument('--cuda_use', type=str, default='cuda:7')
    parser.add_argument('--hid_target', type=int, default=1536)
    parser.add_argument('--hid_ligand_1d', type=int, default=512)
    parser.add_argument('--hid_ligand_2d', type=int, default=55)
    parser.add_argument('--hid_dim', type=int, default=256)
    parser.add_argument('--dir_save_model', type=str, default='save/best_model.pth')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--n_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--fetch_pretrained_target', action='store_true', default=False)
    parser.add_argument('--fetch_pretrained_ligand', action='store_true', default=False)
    args = parser.parse_args()
    return args

def seed_torch(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
        
def plot_attention(attn_weights, title="Attention Map"):
    attn_weights = attn_weights.mean(dim=0).detach().cpu().numpy()
    plt.figure(figsize=(10, 8))
    sns.heatmap(attn_weights, cmap="viridis")
    plt.title(title)
    plt.xlabel("Key Positions")
    plt.ylabel("Query Positions")
    plt.savefig(f'{title}.png', dpi=300, bbox_inches='tight')
    plt.close()

def move_to_device(obj, device):
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    elif isinstance(obj, Data):
        return obj.to(device)
    elif isinstance(obj, (tuple, list)):  
        return type(obj)(move_to_device(x, device) for x in obj)
    return obj

def highlight_ligand(Atype, output_plot, batch_idx, ligand_id, ligand_smiles, attention_scores, top_n=5):
    mol = Chem.MolFromSmiles(ligand_smiles)
    if mol is None:
        print(f"Error: Invalid SMILES string for {ligand_id}, skip...")
        return None
    atom_attention = [(atom.GetIdx(), attention_scores[atom.GetIdx()]) for atom in mol.GetAtoms()]
    sorted_attention = sorted(atom_attention, key=lambda x: x[1], reverse=True)
    top_attention_atoms = [atom_idx for atom_idx, _ in sorted_attention[:top_n]]
    highlight_atoms = top_attention_atoms
    img = Draw.MolToImage(mol, highlightAtoms=highlight_atoms, size=(500, 500))
    img.save(os.path.join(output_plot, f"{batch_idx}-{ligand_id}-{Atype}.png"))

def highlight_protein(Atype, output_plot, batch_idx, protein_id, protein_sequence, attention_scores, pdb_id, top_n=5):
    if protein_sequence is None:
        print(f"Error: No sequence found for {protein_id}, skip...")
        return
    residue_attention = [(idx, attention_scores[idx]) for idx, rs in enumerate(protein_sequence)]
    sorted_attention = sorted(residue_attention, key=lambda x: x[1], reverse=True)
    top_amino_acids = [(idx, protein_sequence[idx]) for idx, _ in sorted_attention[:top_n]]
    with open(os.path.join(output_plot, f"{batch_idx}-{protein_id}-{pdb_id}-{Atype}.txt"), 'w')as f:
        for idx, aa in top_amino_acids:
            f.write(f"Position {idx}: {aa}\n")

def get_ligand_smiles(ligand_id, id_ligand):
    # Ensure ligand_id exists in the dictionary
    if ligand_id not in id_ligand.index:
        print(f"Warning: Ligand ID {ligand_id} not found in ligand dictionary.")
        return None
    return id_ligand.loc[ligand_id, 'SMILES']

def get_protein_sequence(protein_id, id_target):
    if protein_id not in id_target.index:
        print(f"Warning: Protein ID {protein_id} not found in target mapping.")
        return None
    return id_target.loc[protein_id, 'Target Sequence']

def get_predict_output(protein_id, ligand_id, batch_idx, outputs, output_plot):
    probs = torch.softmax(outputs, dim=1)[:, 1]
    probs = probs.detach().cpu().numpy()
    with open(os.path.join(output_plot, f'{batch_idx}-{protein_id}-{ligand_id}.txt'), 'w') as f:
        for prob in probs:
            f.write(f"{prob:.6f}\n")

def getAttn(device, model, loader_data, output_data_dir, id_ligand, id_target):
    os.makedirs(output_data_dir, exist_ok=True)
    output_plot = output_data_dir

    model.eval()
    with torch.no_grad():
        for batch_idx, data in enumerate(tqdm(loader_data, desc=f'Extracting Attention', unit='batch')):
            data = move_to_device(data, device)
            inputs = data[:-4]
            labels = data[-4]
            protein_id = data[-3][0]  # fetching 1-st item in batch
            ligand_id = data[-2][0]  
            pdb_id = data[-1][0]  

            outputs, encoder_attn, decoder_self_attn, decoder_cross_attn = model(inputs, return_attn=True)
            encoder_attn, decoder_self_attn, decoder_cross_attn = encoder_attn.squeeze(0), decoder_self_attn.squeeze(0), decoder_cross_attn.squeeze(0)
            ligand_smiles = get_ligand_smiles(ligand_id, id_ligand)
            protein_sequence = get_protein_sequence(protein_id, id_target)
            # before cross-attention
            # print(f'Shape...SA...ligand...{decoder_self_attn.shape}')
            # print(f'Shape...SA...protein...{encoder_attn.shape}')
            # sa_ligand = decoder_self_attn.sum(dim=0, keepdim=True)
            # sa_protein = encoder_attn.sum(dim=0, keepdim=True)
            # highlight_ligand('SA', output_plot, batch_idx, ligand_id, ligand_smiles, sa_ligand, top_n=5)
            # highlight_protein('SA', output_plot, batch_idx, protein_id, protein_sequence, sa_protein, top_n=5)
            # after cross-attention
            decoder_cross_attn_rm = decoder_cross_attn[:, 1:-1]
            # print(f'Shape...CA...{decoder_cross_attn_rm.shape}')
            ca_ligand = decoder_cross_attn_rm.max(dim=1).values
            ca_protein = decoder_cross_attn_rm.mean(dim=0)
            # highlight_ligand('CA', output_plot, batch_idx, ligand_id, ligand_smiles, ca_ligand, top_n=5)
            highlight_protein('CA', output_plot, batch_idx, protein_id, protein_sequence, ca_protein, pdb_id, top_n=20)
            get_predict_output(protein_id, ligand_id, batch_idx, outputs, output_plot)
            if batch_idx == 100:
                    break
if __name__ == '__main__':
    args = get_args()
    current_date = datetime.now().strftime("%Y%m%d-%H%M")
    seed_torch(args.seed)
    
    data_test = CPIDataset(args, 'predict')
    loader_test = DataLoader(data_test, batch_size=args.batchsize, shuffle=False, worker_init_fn=np.random.seed(args.seed), collate_fn=make_masks_protein)
    device = torch.device(args.cuda_use if torch.cuda.is_available() else 'cpu')
    
    model = Predictor(args)
    model.to(device)
    # model.load_state_dict(torch.load(args.dir_save_model))  # leading to different device
    model.load_state_dict(torch.load(args.dir_save_model, map_location=device))

    # Load dictionary
    id_ligand = pd.read_csv('crawl-0401/filtered_idmapping_ligand_visual.csv') #TODO
    id_ligand.set_index('Ligand ID', inplace=True)
    id_target = pd.read_csv('data/idmapping_target.csv')
    id_target.set_index('Target UniProt ID', inplace=True)
    getAttn(device, model, loader_test, args.output_data_dir, id_ligand, id_target)
