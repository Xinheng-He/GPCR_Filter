import re
from esm.models.esm3 import ESM3
from esm.sdk.api import ESMProtein, SamplingConfig
from esm.utils.constants.models import ESM3_OPEN_SMALL
import torch
import pickle
import pandas as pd
from tqdm import tqdm
client = ESM3.from_pretrained(ESM3_OPEN_SMALL, device=torch.device("cpu"))
# device = torch.device("cuda:7")
# client = ESM3.from_pretrained(ESM3_OPEN_SMALL, device=device)
data = pd.read_csv('/datapool/data2/home/majianzhu/xinheng/xiangzhen/DTI2/DTI-tfcpi2/data/idmapping_target.csv')
print(f'len {len(data)}')
dict_target = {}
for _, row in tqdm(data.iterrows()):
# for match in matches[:1]:
    protein_id = row['Target UniProt ID']
    sequence_item = row['Target Sequence']
    
    
    protein = ESMProtein(
        sequence=sequence_item
    )
    protein_tensor = client.encode(protein)

    output = client.forward_and_sample(
        protein_tensor, SamplingConfig(return_per_residue_embeddings=True)
    )
    val = output.per_residue_embedding
    dict_target[protein_id] = val

with open('dict_target.pkl', 'wb') as f:
    pickle.dump(dict_target, f)
