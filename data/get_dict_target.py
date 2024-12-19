import pickle
import pandas as pd
import torch

dict_target = {}
df = pd.read_csv('/datapool/data2/home/majianzhu/xinheng/xiangzhen/test/data/split-random/test.csv')
target_set = list(set(df['Target UniProt ID'].to_list()))
for target in target_set:
    key = target
    val = torch.randn(18)
    dict_target[key] = val
with open ('/datapool/data2/home/majianzhu/xinheng/xiangzhen/test/data/dict_target.pkl', 'wb') as f:
    pickle.dump(dict_target, f)
    

dict_ligand = {}
df = pd.read_csv('/datapool/data2/home/majianzhu/xinheng/xiangzhen/test/data/split-random/test.csv')
ligand_set = list(set(df['SMILES'].to_list()))
for ligand in ligand_set:
    key = ligand
    val = torch.randn(18)
    dict_ligand[key] = val 
with open ('/datapool/data2/home/majianzhu/xinheng/xiangzhen/test/data/dict_ligand.pkl', 'wb') as f:
    pickle.dump(dict_ligand, f)

with open('/datapool/data2/home/majianzhu/xinheng/xiangzhen/test/data/dict_target.pkl', 'rb') as f:
    dct = pickle.load(f)
    print(dct)