import torch
from rdkit import Chem
import numpy as np

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