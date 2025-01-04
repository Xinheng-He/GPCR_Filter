import torch.nn as nn
import torch
from torch_geometric.nn import Set2Set

class Decoder(nn.Module):
    def __init__(self, protein_dim, atom_dim, hidden, set2set_steps=3):
        super().__init__()
        self.protein_dim = protein_dim
        self.atom_dim = atom_dim
        self.hidden = hidden

        self.ligand_transform = nn.Linear(atom_dim, hidden)
        self.protein_transform = nn.Linear(protein_dim, hidden)

        self.ligand_set2set = Set2Set(hidden, processing_steps=set2set_steps)
        self.protein_set2set = Set2Set(hidden, processing_steps=set2set_steps)

        self.classifier = nn.Sequential(
            nn.Linear(2 * hidden * 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2)
        )

    def forward(self, repr_protein, repr_ligand):
        repr_protein.x = self.protein_transform(repr_protein.x)  # (batch_size, seq_len_protein, hidden)
        repr_ligand.x = self.ligand_transform(repr_ligand.x)    # (batch_size, seq_len_ligand, hidden)

        global_protein = self.protein_set2set(repr_protein.x, repr_protein.batch)  # (batch_size, hidden)
        global_ligand = self.ligand_set2set(repr_ligand.x, repr_ligand.batch)      # (batch_size, hidden)

        global_feature = torch.cat([global_protein, global_ligand], dim=-1)  # (batch_size, 2 * hidden)

        x = self.classifier(global_feature)
        return x