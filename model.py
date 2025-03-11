import torch.nn as nn
import torch
from torch_geometric.nn import (
                                Set2Set,
                                GATConv, 
                                Linear,
                                SAGPooling,
                                global_add_pool,
                                GCNConv
                                )
import torch.nn.functional as F
from data.utils import pad_graphs_with_mask


class Encoder(nn.Module):
    """protein feature extraction."""
    def __init__(self, args):
        super().__init__()
        self.hid_dim = args.hid_dim
        self.n_layers = args.n_layers
        self.dropout =args.dropout

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.hid_dim, 
                                                        nhead=8, 
                                                        dim_feedforward=self.hid_dim * 4, 
                                                        dropout=self.dropout, 
                                                        batch_first=True
                                                        )
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=self.n_layers)

    def forward(self, protein, mask):
        protein = self.encoder(protein,src_key_padding_mask=mask)
        return protein


class Decoder(nn.Module):
    """ compound feature extraction."""
    def __init__(self, args):
        super().__init__()
        self.n_layers = args.n_layers
        self.dropout =args.dropout
        self.hid_dim = args.hid_dim

        self.decoder_layer = nn.TransformerDecoderLayer(d_model=self.hid_dim, 
                                                        nhead=8, 
                                                        dim_feedforward=self.hid_dim * 4,
                                                        dropout=self.dropout,
                                                        batch_first=True
                                                        )
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=self.n_layers)
        self.fc_1 = nn.Linear(self.hid_dim, self.hid_dim)
        self.fc_2 = nn.Linear(self.hid_dim, 2)
        self.dropout = nn.Dropout(self.dropout)

    def forward(self, trg, src, trg_mask=None,src_mask=None):
        trg = self.decoder(trg, src, tgt_key_padding_mask=trg_mask, memory_key_padding_mask=src_mask)
        x = trg[:,0,:]
        label = F.relu(self.fc_1(x))
        label = self.fc_2(label)
        return label


class Predictor(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.hid_dim = args.hid_dim
        self.hid_target = args.hid_target
        self.hid_ligand_2d = args.hid_ligand_2d

        self.encoder = Encoder(args)
        self.decoder = Decoder(args)
        self.fc_protein = nn.Linear(self.hid_target, self.hid_dim)
        self.fc_compound = nn.Linear(self.hid_ligand_2d, self.hid_dim)
        self.gcn = GCNConv(self.hid_dim, self.hid_dim)
    
    def forward(self, inputs):
        protein, compound, protein_mask = inputs
        # tgt-compound
        compound.x = F.relu(self.fc_compound(compound.x))
        compound.x = self.gcn(compound.x, compound.edge_index)
        compound_tgt, tgt_mask = pad_graphs_with_mask(compound)
        # src-protein
        protein = F.relu(self.fc_protein(protein))
        protein_src = self.encoder(protein, protein_mask)
        out = self.decoder(compound_tgt, protein_src, tgt_mask, protein_mask)
        # out = [batch size, 2]
        return out