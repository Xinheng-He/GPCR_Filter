import torch.nn as nn
import torch

class Decoder(nn.Module):
    def __init__(self, protein_dim, atom_dim, hidden, num_heads=2, num_layers=2):
        super().__init__()
        self.protein_dim = protein_dim
        self.atom_dim = atom_dim
        self.hidden = hidden

        self.protein_embedding = nn.Linear(self.protein_dim, self.hidden)
        self.ligand_embedding = nn.Linear(self.atom_dim, self.hidden)
        self.transformer = nn.Transformer(
            d_model=self.hidden,       
            nhead=num_heads,
            num_encoder_layers=num_layers, 
            num_decoder_layers=num_layers,
            batch_first=True
        )
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden, self.hidden // 2),
            nn.ReLU(),
            nn.Linear(self.hidden // 2, 2)
        )

    def forward(self, x_protein, x_ligand, protein_mask, ligand_mask):
        src = self.protein_embedding(x_protein)  # (batch_size, seq_len, hidden_dim)
        tgt = self.ligand_embedding(x_ligand)    # (batch_size, seq_len, hidden_dim)
        src_key_padding_mask = ~protein_mask
        tgt_key_padding_mask = ~ligand_mask
        output = self.transformer(
            src, tgt,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )  # (batch_size, seq_len, hidden_dim)
        output = output.mean(dim=1)
        x = self.classifier(output)
        return x
