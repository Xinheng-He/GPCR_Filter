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


# class Encoder(nn.Module):
#     def __init__(self, args):
#         super().__init__()
#         self.hid_dim = args.hid_dim
#         self.n_layers = args.n_layers
#         self.dropout = args.dropout

#         self.encoder_layer = nn.TransformerEncoderLayer(
#             d_model=self.hid_dim,
#             nhead=8,
#             dim_feedforward=self.hid_dim * 4,
#             dropout=self.dropout,
#             batch_first=True
#         )
#         self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=self.n_layers)

#     def forward(self, protein, mask, return_attn=False):
#         protein = self.encoder(protein, src_key_padding_mask=mask)
        
#         if return_attn:
#             attn_weights = self.encoder_layer.self_attn(protein, protein, protein, need_weights=True)[1]
#             return protein, attn_weights
#         return protein



# class Decoder(nn.Module):
#     def __init__(self, args):
#         super().__init__()
#         self.n_layers = args.n_layers
#         self.dropout = args.dropout
#         self.hid_dim = args.hid_dim

#         self.decoder_layer = nn.TransformerDecoderLayer(
#             d_model=self.hid_dim,
#             nhead=8,
#             dim_feedforward=self.hid_dim * 4,
#             dropout=self.dropout,
#             batch_first=True
#         )
#         self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=self.n_layers)
#         self.fc_1 = nn.Linear(self.hid_dim, self.hid_dim)
#         self.fc_2 = nn.Linear(self.hid_dim, 2)
#         self.dropout = nn.Dropout(self.dropout)

#     def forward(self, trg, src, trg_mask=None, src_mask=None, return_attn=False):
#         trg = self.decoder(trg, src, tgt_key_padding_mask=trg_mask, memory_key_padding_mask=src_mask)

#         x = trg[:, 0, :]
#         label = F.relu(self.fc_1(x))
#         label = self.fc_2(label)

#         if return_attn:
#             self_attn_weights = self.decoder_layer.self_attn(trg, trg, trg, need_weights=True)[1]
#             cross_attn_weights = self.decoder_layer.multihead_attn(trg, src, src, need_weights=True)[1]
#             return label, self_attn_weights, cross_attn_weights 
#         return label

class Encoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.hid_dim = args.hid_dim
        self.n_layers = args.n_layers
        self.dropout = args.dropout

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hid_dim,
            nhead=8,
            dim_feedforward=self.hid_dim * 4,
            dropout=self.dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=self.n_layers)

    def forward(self, protein, mask, return_attn=False):
        """Forward pass with optional attention extraction."""
        self_attn_weights = []

        def hook(module, input, output):
            """Hook to extract self-attention weights from the last layer."""
            attn_output, attn_weights = module.self_attn(input[0], input[0], input[0], need_weights=True)
            self_attn_weights.append(attn_weights.clone().detach())
        last_encoder_layer = self.encoder.layers[-1]
        handle = last_encoder_layer.register_forward_hook(hook)
        protein = self.encoder(protein, src_key_padding_mask=mask)
        handle.remove()
        if return_attn:
            return protein, self_attn_weights[0]
        return protein

    # def forward(self, protein, mask, return_attn=False):
    #     breakpoint()
    #     attn_weights = None
    #     for i, layer in enumerate(self.encoder.layers):
    #         protein = layer(protein, src_key_padding_mask=mask)
    #         if return_attn and i == len(self.encoder.layers) - 1:
    #             attn_weights = layer.self_attn(protein, protein, protein, need_weights=True)[1]

    #     if return_attn:
    #         return protein, attn_weights.detach()
    #     return protein 


class Decoder(nn.Module):
    """Compound feature extraction."""
    def __init__(self, args):
        super().__init__()
        self.n_layers = args.n_layers
        self.dropout = args.dropout
        self.hid_dim = args.hid_dim

        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.hid_dim, 
            nhead=8, 
            dim_feedforward=self.hid_dim * 4,
            dropout=self.dropout,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=self.n_layers)
        self.fc_1 = nn.Linear(self.hid_dim, self.hid_dim)
        self.fc_2 = nn.Linear(self.hid_dim, 2)
        self.dropout = nn.Dropout(self.dropout)

    def forward(self, trg, src, trg_mask=None, src_mask=None, return_attn=False):
        self_attn_weights = []
        cross_attn_weights = []
        def hook(module, input, output):
            """Hook to capture self-attention and cross-attention weights."""
            attn_output, attn_weights = module.self_attn(input[0], input[0], input[0], need_weights=True)
            self_attn_weights.append(attn_weights.clone().detach())

            attn_output, attn_weights = module.multihead_attn(input[0], input[1], input[1], need_weights=True)
            cross_attn_weights.append(attn_weights.clone().detach())

        last_decoder_layer = self.decoder.layers[-1]
        handle = last_decoder_layer.register_forward_hook(hook)
        trg = self.decoder(trg, src, tgt_key_padding_mask=trg_mask, memory_key_padding_mask=src_mask)
        handle.remove()

        x = trg[:, 0, :]
        label = F.relu(self.fc_1(x))
        label = self.fc_2(label)

        if return_attn:
            return label, self_attn_weights[0], cross_attn_weights[0]
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

    def forward(self, inputs, return_attn=False):
        protein, compound, protein_mask = inputs
        # tgt-compound
        compound.x = F.relu(self.fc_compound(compound.x))
        compound.x = self.gcn(compound.x, compound.edge_index)
        compound_tgt, tgt_mask = pad_graphs_with_mask(compound)
        
        # src-protein
        protein = F.relu(self.fc_protein(protein))
        if return_attn:
            protein_src, encoder_attn = self.encoder(protein, protein_mask, return_attn=True)
            out, decoder_self_attn, decoder_cross_attn = self.decoder(compound_tgt, protein_src, tgt_mask, protein_mask, return_attn=True)
            return out, encoder_attn, decoder_self_attn, decoder_cross_attn
        else:
            protein_src = self.encoder(protein, protein_mask)
            out = self.decoder(compound_tgt, protein_src, tgt_mask, protein_mask)
            return out
