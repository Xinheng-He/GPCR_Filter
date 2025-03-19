import pandas as pd
from sklearn.model_selection import train_test_split
import random
'''
first, split train:valid:test = 3:1:1 for each target-group
second, neg-sampling in train/valid/test seperately
''' 
def generate_negative_samples(positive_df, seed):
    positive_samples = set(zip(positive_df['Target UniProt ID'], positive_df['SMILES']))
    target_set = set(positive_df['Target UniProt ID'])
    smiles_set = set(positive_df['SMILES'])
    
    all_combinations = set((t, s) for t in target_set for s in smiles_set)
    negative_samples = all_combinations - positive_samples
            
    num_positive_samples = len(positive_df)
    negative_samples_selected = random.sample(negative_samples, num_positive_samples)
    negative_df = pd.DataFrame(negative_samples_selected, columns=['Target UniProt ID', 'SMILES'])
    negative_df['Label'] = 0
    positive_df_copy = positive_df.copy()
    positive_df_copy['Label'] = 1  
    final_df = pd.concat([positive_df_copy[['Target UniProt ID', 'SMILES', 'Label']], negative_df])
    final_df = final_df.sample(frac=1, random_state=seed).reset_index(drop=True)
    return final_df

# init
SEED = 1
random.seed(SEED)
df = pd.read_csv('/root/gpcr-db/interactions.csv')
# split in target-group
train_data = []
val_data = []
test_data = []
for target_id, group in df.groupby('Target UniProt ID'):
    if len(group) < 5:
        train_data.append(group)
    else:
        train, temp = train_test_split(group, test_size=0.4, random_state=SEED)
        val, test = train_test_split(temp, test_size=0.5, random_state=SEED)
        train_data.append(train)
        val_data.append(val)
        test_data.append(test)
train_df = pd.concat(train_data)
val_df = pd.concat(val_data)
test_df = pd.concat(test_data)
# neg-sampling
train_final_df = generate_negative_samples(train_df, SEED)
val_final_df = generate_negative_samples(val_df, SEED)
test_final_df = generate_negative_samples(test_df, SEED)
# save
train_final_df.to_csv('train.csv', index=False)
val_final_df.to_csv('valid.csv', index=False)
test_final_df.to_csv('test.csv', index=False)
