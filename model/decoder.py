# import torch.nn as nn
# import torch

# class Decoder(nn.Module):
#     def __init__(self, hidden_target, hidden_ligand, hidden):
#         super().__init__()
#         self.hidden_target = hidden_target
#         self.hidden_ligand = hidden_ligand
#         self.hidden = hidden
#         self.init_Q = nn.Sequential(
#             nn.Linear(self.hidden_target, self.hidden),
#             nn.ReLU(),
#         )
#         self.init_K = nn.Sequential(
#             nn.Linear(self.hidden_ligand, self.hidden),
#             nn.ReLU(),
#         )
#         self.init_V = nn.Sequential(
#             nn.Linear(self.hidden_ligand, self.hidden),
#             nn.ReLU(),
#         )
#         self.attention_layer = nn.MultiheadAttention(self.hidden, num_heads=8)
#         self.classifier = nn.Sequential(
#             nn.Linear(self.hidden, self.hidden // 2),
#             nn.ReLU(),
#             nn.Linear(self.hidden // 2, 2)
#         )
#     def forward(self, x_target, x_ligand):
#         '''
#         co-attention
#         Q: target
#         K, V: ligand
#         '''
#         # (batch_size, hidden_dim)
#         breakpoint()
#         Q_target = self.init_Q(x_target)
#         K_ligand = self.init_K(x_ligand)
#         V_ligand = self.init_V(x_ligand)
#         # To: (batch_size, 1, hidden_dim)
#         Q_target = Q_target.unsqueeze(1)
#         K_ligand = K_ligand.unsqueeze(1)
#         V_ligand = V_ligand.unsqueeze(1)
#         # To: (1, batch_size, hidden_dim)
#         Q_target = Q_target.transpose(0, 1)
#         K_ligand = K_ligand.transpose(0, 1)
#         V_ligand = V_ligand.transpose(0, 1)    
#         # To: (batch_size, hidden_dim)
#         attn_output, attn_output_weights = self.attention_layer(Q_target, K_ligand, V_ligand)
#         attn_output = attn_output.transpose(0, 1)
#         attn_output = attn_output.squeeze()
#         # To: (batch_size, 2)
#         x = self.classifier(attn_output)
#         return x

import torch.nn as nn
import torch

class Decoder(nn.Module):
    def __init__(self, protein_dim, atom_dim, hidden):
        super().__init__()
        self.protein_dim = protein_dim
        self.atom_dim = atom_dim
        self.hidden = hidden

        # 初始化 Q, K, V
        self.init_Q = nn.Sequential(
            nn.Linear(self.protein_dim, self.hidden),
            nn.ReLU(),
        )
        self.init_K = nn.Sequential(
            nn.Linear(self.atom_dim, self.hidden),
            nn.ReLU(),
        )
        self.init_V = nn.Sequential(
            nn.Linear(self.atom_dim, self.hidden),
            nn.ReLU(),
        )

        # 多头注意力机制
        self.attention_layer = nn.MultiheadAttention(self.hidden, num_heads=8)

        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden, self.hidden // 2),
            nn.ReLU(),
            nn.Linear(self.hidden // 2, 2)
        )

    def forward(self, x_protein, x_ligand, protein_mask, ligand_mask):
        """
        Co-attention:
        Q: protein
        K, V: ligand
        """
        # 初始化 Q, K, V
        Q_protein = self.init_Q(x_protein)
        K_ligand = self.init_K(x_ligand)
        V_ligand = self.init_V(x_ligand)

        # 转置以适配多头注意力
        Q_protein = Q_protein.transpose(0, 1)  # (seq_len, batch_size, hidden_dim)
        K_ligand = K_ligand.transpose(0, 1)
        V_ligand = V_ligand.transpose(0, 1)

        # 注意力机制
        attn_output, _ = self.attention_layer(Q_protein, K_ligand, V_ligand, key_padding_mask=~ligand_mask)

        # 汇总序列信息（例如，取平均池化）
        attn_output = attn_output.transpose(0, 1).mean(dim=1)  # (batch_size, hidden_dim)

        # 分类
        x = self.classifier(attn_output)
        return x

