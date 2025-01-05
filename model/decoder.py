import torch.nn as nn
import torch
from torch_geometric.nn import (
                                Set2Set,
                                GATConv, 
                                Linear,
                                SAGPooling,
                                global_add_pool
                                )



class Decoder(nn.Module):
    def __init__(self, protein_dim, atom_dim_1d, atom_dim_2d, hidden, set2set_steps=3):
        super().__init__()
        self.protein_dim = protein_dim
        self.atom_dim_1d = atom_dim_1d
        self.atom_dim_2d = atom_dim_2d
        self.hidden = hidden

        self.ligand_transform = nn.Linear(atom_dim_1d, hidden)
        self.protein_transform = nn.Linear(protein_dim, hidden)

        self.ligand_set2set = Set2Set(hidden, processing_steps=set2set_steps)
        self.protein_set2set = Set2Set(hidden, processing_steps=set2set_steps)

        self.classifier = nn.Sequential(
            nn.Linear(2 * hidden * 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2)
        )

        self.initial_node_feature = nn.Sequential(
            nn.Linear(atom_dim_2d, hidden * 2),
            nn.ReLU()
        )
        self.gnn = GATConv(hidden * 2, hidden * 2 // 4, 4)
        self.relu = nn.ReLU()
        self.readout = SAGPooling(hidden * 2, min_score=-1)


    def forward(self, repr_protein, repr_ligand_1d, repr_ligand_2d):
        repr_protein.x = self.protein_transform(repr_protein.x)  # (batch_size, seq_len_protein, hidden)
        # repr_ligand_1d.x = self.ligand_transform(repr_ligand_1d.x)    # (batch_size, seq_len_ligand, hidden)

        global_protein = self.protein_set2set(repr_protein.x, repr_protein.batch)  # (batch_size, hidden)
        # global_ligand_1d = self.ligand_set2set(repr_ligand_1d.x, repr_ligand_1d.batch)      # (batch_size, hidden)
        repr_ligand_2d.x = self.initial_node_feature(repr_ligand_2d.x)
        repr_ligand_2d.x = self.gnn(repr_ligand_2d.x, repr_ligand_2d.edge_index)
        repr_ligand_2d.x = self.relu(repr_ligand_2d.x)        
        att_x, _, _, att_batch, _, _= self.readout(repr_ligand_2d.x, repr_ligand_2d.edge_index, batch=repr_ligand_2d.batch)
        global_ligand_2d = global_add_pool(att_x, att_batch)

        global_feature = torch.cat([global_protein, global_ligand_2d], dim=-1)  # (batch_size, 2 * hidden)

        x = self.classifier(global_feature)
        return x