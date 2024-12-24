# # from torch.utils.data import Dataset, DataLoader
# # import pandas as pd
# # import pickle
# # import torch
# # import os

# # class CPIDataset(Dataset):
# #     def __init__(self, csv_file, dict_target, dict_ligand):
# #         self.dataset_raw = pd.read_csv(csv_file)
# #         # self.idmapping_ligand = pd.read_csv(os.path.join('data', 'idmapping_ligand.csv'))
# #         # self.idmapping_target = pd.read_csv(os.path.join('data', 'idmapping_target.csv'))
# #         with open(dict_target, 'rb') as f:
# #             self.dict_target = pickle.load(f)
# #         with open(dict_ligand, 'rb') as f:
# #             self.dict_ligand = pickle.load(f)
# #         for key, val in self.dict_target.items():
# #             val_mean = torch.mean(val, dim=0, keepdim=False)
# #             self.dict_target[key] = val_mean
# #         wrong_list = ['5617', '25395', '30099', '33848', '38706', '39020', '41526', '46371', '56786', '60853', '61837', '69875', '70076', '70494']
# #         wrong_list = [f'drug{i}' for i in wrong_list]
# #         self.dict_ligand_new = {}
# #         for key, val in self.dict_ligand.items():
# #             if key in wrong_list:
# #                 continue
# #             val_dtype = val.to(torch.float32)
# #             self.dict_ligand_new[key] = val_dtype
# #         self.dataset = [
# #             (
# #                 self.dict_target[row['Target UniProt ID']], 
# #                 self.dict_ligand_new[row['Ligand ID']],           
# #                 torch.tensor(row['Label'], dtype=torch.int64) 
# #             )
# #             for _, row in self.dataset_raw.iterrows()
# #         ]
        
# #     def __len__(self):
# #         return len(self.dataset)
# #     def __getitem__(self, idx):
# #         return self.dataset[idx]

# from torch.utils.data import Dataset
# import pandas as pd
# import pickle
# import torch
# import os
# import torch.nn.utils.rnn as rnn_utils

# class CPIDataset(Dataset):
#     def __init__(self, csv_file, dict_target, dict_ligand):
#         self.dataset_raw = pd.read_csv(csv_file)
#         with open(dict_target, 'rb') as f:
#             self.dict_target = pickle.load(f)  # 蛋白质嵌入字典，{UniProtID: [protein_len, protein_dim]}
#         with open(dict_ligand, 'rb') as f:
#             self.dict_ligand = pickle.load(f)  # 分子嵌入字典，{LigandID: [compound_len, atom_dim]}
        
#         # 去除无效分子
#         wrong_list = ['5617', '25395', '30099', '33848', '38706', '39020', '41526', '46371', '56786', '60853', '61837', '69875', '70076', '70494']
#         wrong_list = [f'drug{i}' for i in wrong_list]
#         # self.dict_ligand_new = {
#         #     key: torch.tensor(val, dtype=torch.float32)
#         #     for key, val in self.dict_ligand.items()
#         #     if key not in wrong_list
#         # }
#         self.dict_ligand_new = {
#             key: torch.tensor(val, dtype=torch.float32).clone().detach()
#             for key, val in self.dict_ligand.items()
#             if key not in wrong_list
#         }

#         # 构建数据集：蛋白质、分子及其标签
#         # self.dataset = []
#         # for _, row in self.dataset_raw.iterrows():
#         #     protein_id = row['Target UniProt ID']
#         #     ligand_id = row['Ligand ID']
#         #     if protein_id in self.dict_target and ligand_id in self.dict_ligand_new:
#         #         protein_tensor = torch.tensor(self.dict_target[protein_id], dtype=torch.float32)
#         #         ligand_tensor = self.dict_ligand_new[ligand_id]
#         #         label = torch.tensor(row['Label'], dtype=torch.int64)
#         #         self.dataset.append((protein_tensor, ligand_tensor, label))
#         self.dataset = []
#         for _, row in self.dataset_raw.iterrows():
#             protein_id = row['Target UniProt ID']
#             ligand_id = row['Ligand ID']
#             if protein_id in self.dict_target and ligand_id in self.dict_ligand_new:
#                 # 如果 protein_tensor 已经是一个 Tensor，不需要重新转换，直接使用 clone().detach()
#                 protein_tensor = self.dict_target[protein_id].clone().detach() if isinstance(self.dict_target[protein_id], torch.Tensor) else torch.tensor(self.dict_target[protein_id], dtype=torch.float32)
#                 ligand_tensor = self.dict_ligand_new[ligand_id].clone().detach() if isinstance(self.dict_ligand_new[ligand_id], torch.Tensor) else torch.tensor(self.dict_ligand_new[ligand_id], dtype=torch.float32)
#                 label = torch.tensor(row['Label'], dtype=torch.int64).clone().detach()
#                 self.dataset.append((protein_tensor, ligand_tensor, label))
        
#     def __len__(self):
#         return len(self.dataset)

#     def __getitem__(self, idx):
#         return self.dataset[idx]

# # 自定义 collate_fn，用于处理变长序列
# def custom_collate_fn(batch):
#     proteins, ligands, labels = zip(*batch)

#     # 填充蛋白质序列和分子序列
#     padded_proteins = rnn_utils.pad_sequence(proteins, batch_first=True, padding_value=0)
#     padded_ligands = rnn_utils.pad_sequence(ligands, batch_first=True, padding_value=0)

#     # 掩码
#     protein_mask = torch.arange(padded_proteins.size(1))[None, :] < torch.tensor([p.size(0) for p in proteins])[:, None]
#     ligand_mask = torch.arange(padded_ligands.size(1))[None, :] < torch.tensor([l.size(0) for l in ligands])[:, None]

#     # 标签
#     labels = torch.stack(labels)
#     return padded_proteins, padded_ligands, protein_mask, ligand_mask, labels


import pandas as pd
import pickle
import torch
from torch.utils.data import Dataset
import torch.nn.utils.rnn as rnn_utils

class CPIDataset(Dataset):
    def __init__(self, csv_file, dict_target, dict_ligand):
        # 仅加载基本信息
        self.dataset_raw = pd.read_csv(csv_file)
        with open(dict_target, 'rb') as f:
            self.dict_target = pickle.load(f)  # {UniProtID: feature_matrix}
        with open(dict_ligand, 'rb') as f:
            self.dict_ligand = pickle.load(f)  # {LigandID: feature_matrix}
        
        # 去除无效分子
        self.wrong_list = {f'drug{i}' for i in [
            '5617', '25395', '30099', '33848', '38706', '39020', '41526',
            '46371', '56786', '60853', '61837', '69875', '70076', '70494']}
        
        # 提取有效样本行索引
        self.valid_indices = [
            idx for idx, row in self.dataset_raw.iterrows()
            if row['Target UniProt ID'] in self.dict_target and 
               row['Ligand ID'] in self.dict_ligand and
               row['Ligand ID'] not in self.wrong_list
        ]

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        # 动态加载特征
        raw_idx = self.valid_indices[idx]
        row = self.dataset_raw.iloc[raw_idx]
        protein_id = row['Target UniProt ID']
        ligand_id = row['Ligand ID']
        label = row['Label']

        protein_tensor = torch.tensor(self.dict_target[protein_id], dtype=torch.float32)
        ligand_tensor = torch.tensor(self.dict_ligand[ligand_id], dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.int64)

        return protein_tensor, ligand_tensor, label_tensor

# 自定义 collate_fn，用于处理变长序列
def custom_collate_fn(batch):
    proteins, ligands, labels = zip(*batch)

    # 填充蛋白质序列和分子序列
    padded_proteins = rnn_utils.pad_sequence(proteins, batch_first=True, padding_value=0)
    padded_ligands = rnn_utils.pad_sequence(ligands, batch_first=True, padding_value=0)

    # 掩码
    protein_mask = torch.arange(padded_proteins.size(1))[None, :] < torch.tensor([p.size(0) for p in proteins])[:, None]
    ligand_mask = torch.arange(padded_ligands.size(1))[None, :] < torch.tensor([l.size(0) for l in ligands])[:, None]

    # 标签
    labels = torch.stack(labels)
    return padded_proteins, padded_ligands, protein_mask, ligand_mask, labels
