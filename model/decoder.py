import torch.nn as nn
import torch

class Decoder1218(nn.Module):
    def __init__(self, hidden_target, hidden_ligand, hidden):
        super().__init__()
        self.hidden_target = hidden_target
        self.hidden_ligand = hidden_ligand
        self.hidden = hidden
        self.init_Q = nn.Sequential(
            nn.Linear(self.hidden_target, self.hidden),
            nn.ReLU(),
        )
        self.init_K = nn.Sequential(
            nn.Linear(self.hidden_ligand, self.hidden),
            nn.ReLU(),
        )
        self.init_V = nn.Sequential(
            nn.Linear(self.hidden_ligand, self.hidden),
            nn.ReLU(),
        )
        self.attention_layer = nn.MultiheadAttention(self.hidden, num_heads=8)
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden, self.hidden // 2),
            nn.ReLU(),
            nn.Linear(self.hidden // 2, 2)
        )
    def forward(self, x_target, x_ligand):
        '''
        co-attention
        Q: target
        K, V: ligand
        '''
        # (batch_size, hidden_dim)
        Q_target = self.init_Q(x_target)
        K_ligand = self.init_K(x_ligand)
        V_ligand = self.init_V(x_ligand)
        # To: (batch_size, 1, hidden_dim)
        Q_target = Q_target.unsqueeze(1)
        K_ligand = K_ligand.unsqueeze(1)
        V_ligand = V_ligand.unsqueeze(1)
        # To: (1, batch_size, hidden_dim)
        Q_target = Q_target.transpose(0, 1)
        K_ligand = K_ligand.transpose(0, 1)
        V_ligand = V_ligand.transpose(0, 1)    
        # To: (batch_size, hidden_dim)
        attn_output, attn_output_weights = self.attention_layer(Q_target, K_ligand, V_ligand)
        attn_output = attn_output.transpose(0, 1)
        attn_output = attn_output.squeeze()
        # To: (batch_size, 2)
        x = self.classifier(attn_output)
        return x
    
class Decoder(nn.Module):
    def __init__(self, hidden_target, hidden_ligand, hidden):
        super().__init__()
        self.hidden_target = hidden_target
        self.hidden_ligand = hidden_ligand
        self.hidden = hidden
        self.init_target = nn.Sequential(
            nn.Linear(self.hidden_target, self.hidden),
            nn.ReLU(),
        )
        self.init_ligand = nn.Sequential(
            nn.Linear(self.hidden_ligand, self.hidden),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden * 2, self.hidden // 2),
            nn.ReLU(),
            nn.Linear(self.hidden // 2, 2)
        )
    def forward(self, x_target, x_ligand):
        '''
        just cat
        '''
        # (batch_size, hidden_dim)
        # breakpoint()
        x_target = self.init_target(x_target)
        x_ligand = self.init_ligand(x_ligand)
        # To: (batch_size, 2)
        x = torch.cat((x_target, x_ligand), dim=1)
        x = self.classifier(x)
        return x