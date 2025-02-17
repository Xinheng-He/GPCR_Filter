import torch
from rdkit import Chem
import numpy as np
# from tape import TAPETokenizer
from torch_geometric.data import Data, Batch
import torch.nn.utils.rnn as rnn_utils

def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def atom_features(atom, explicit_H=True, use_chirality=False):
    results =   one_of_k_encoding_unk(
                    atom.GetSymbol(),
                    ['C','N','O', 'S','F','Si','P', 'Cl','Br','Mg','Na','Ca','Fe','As','Al','I','B','V','K','Tl',
                        'Yb','Sb','Sn','Ag','Pd','Co','Se','Ti','Zn','H', 'Li','Ge','Cu','Au','Ni','Cd','In',
                        'Mn','Zr','Cr','Pt','Hg','Pb','Unknown'
                    ]) + \
                [atom.GetDegree()/10, atom.GetImplicitValence(), atom.GetFormalCharge(), atom.GetNumRadicalElectrons()] + \
                one_of_k_encoding_unk(
                    atom.GetHybridization(), 
                    [
                        Chem.rdchem.HybridizationType.SP, 
                        Chem.rdchem.HybridizationType.SP2, 
                        Chem.rdchem.HybridizationType.SP3, 
                        Chem.rdchem.HybridizationType.SP3D, 
                        Chem.rdchem.HybridizationType.SP3D2
                    ]) + \
                [atom.GetIsAromatic()]
    # In case of explicit hydrogen(QM8, QM9), avoid calling `GetTotalNumHs`
    if explicit_H:
        results = results + [atom.GetTotalNumHs()]
    if use_chirality:
        try:
            results = results + one_of_k_encoding_unk(
            atom.GetProp('_CIPCode'),
            ['R', 'S']) + [atom.HasProp('_ChiralityPossible')]
        except:
            results = results + [False, False
                            ] + [atom.HasProp('_ChiralityPossible')]
    results = np.array(results).astype(np.float32)
    return torch.tensor(results)

def get_mol_edge_list_and_feat_mtx(mol_smiles):
    mol_graph = Chem.MolFromSmiles(mol_smiles)
    n_features = [(atom.GetIdx(), atom_features(atom)) for atom in mol_graph.GetAtoms()]
    n_features.sort() # to make sure that the feature matrix is aligned according to the idx of the atom
    _, n_features = zip(*n_features)
    n_features = torch.stack(n_features)

    edge_list = torch.LongTensor([(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol_graph.GetBonds()])
    undirected_edge_list = torch.cat([edge_list, edge_list[:, [1, 0]]], dim=0) if len(edge_list) else edge_list
    if undirected_edge_list.numel() == 0:
        undirected_edge_list = torch.LongTensor([(0, 0),(0, 0)])
    return n_features, undirected_edge_list.T

def seq_cat(prot,tokenizer):
    xs = tokenizer.encode(prot)
    return xs

# def create_target_data(sequence):
#     tokenizer = TAPETokenizer(vocab='iupac')
#     sequence = seq_cat(sequence, tokenizer)
#     with torch.no_grad():
#         protein_embedding = torch.tensor([sequence], dtype=torch.int64)
#     protein_tensor = protein_embedding.squeeze(0)
#     return protein_tensor

def create_graph_data(ligand_smiles):
    n_features, edge_index = get_mol_edge_list_and_feat_mtx(ligand_smiles)
    return Data(x=n_features, edge_index=edge_index)

# def make_masks(batch):
#     proteins, ligands, labels = zip(*batch)
#     padded_proteins = rnn_utils.pad_sequence(proteins, batch_first=True, padding_value=0)
#     padded_ligands = rnn_utils.pad_sequence(ligands, batch_first=True, padding_value=0)
#     protein_mask = torch.arange(padded_proteins.size(1))[None, :] < torch.tensor([p.size(0) for p in proteins])[:, None]
#     ligand_mask = torch.arange(padded_ligands.size(1))[None, :] < torch.tensor([l.size(0) for l in ligands])[:, None]
#     labels = torch.stack(labels)
#     return padded_proteins, padded_ligands, protein_mask, ligand_mask, labels

def make_masks(batch):
    proteins, ligands, labels = zip(*batch)
    proteins = list(proteins)
    ligands = list(ligands)
    padded_proteins = rnn_utils.pad_sequence(proteins, batch_first=True, padding_value=0)
    protein_mask = ~(torch.arange(padded_proteins.size(1))[None, :] < torch.tensor([p.size(0) for p in proteins])[:, None])
    padded_ligands = rnn_utils.pad_sequence(ligands, batch_first=True, padding_value=0)
    ligand_mask = ~(torch.arange(padded_ligands.size(1))[None, :] < torch.tensor([p.size(0) for p in ligands])[:, None])
    labels = torch.stack(labels)
    return padded_proteins, padded_ligands, protein_mask, ligand_mask, labels

def make_masks_protein(batch):
    proteins, ligands, labels = zip(*batch)
    proteins = list(proteins)
    ligands = Batch.from_data_list(list(ligands))
    padded_proteins = rnn_utils.pad_sequence(proteins, batch_first=True, padding_value=0)
    protein_mask = ~(torch.arange(padded_proteins.size(1))[None, :] < torch.tensor([p.size(0) for p in proteins])[:, None])
    labels = torch.stack(labels)
    return padded_proteins, ligands, protein_mask, labels

def pad_graphs_with_mask(data):
    batch = data.batch
    x = data.x
    # obtain max atom num
    num_graphs = batch.max().item() + 1 
    max_nodes = [batch.eq(i).sum().item() for i in range(num_graphs)] 
    max_atom_num = max(max_nodes)
    # for padding
    padded_x = torch.zeros((num_graphs, max_atom_num, x.size(1)), device=x.device)
    padded_mask = torch.ones((num_graphs, max_atom_num), device=x.device)
    for i in range(num_graphs):
        graph_nodes = batch.eq(i)
        num_nodes_in_graph = graph_nodes.sum().item()
        padded_x[i, :num_nodes_in_graph, :] = x[graph_nodes]
        padded_mask[i, :num_nodes_in_graph] = 0
    padded_mask = padded_mask.to(torch.bool)
    
    return padded_x, padded_mask