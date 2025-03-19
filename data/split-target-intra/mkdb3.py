import pandas as pd
from sklearn.model_selection import train_test_split
import random
'''
first, get group for each target
second, neg-sampling in group
third, combine
''' 
def generate_negative_samples(positive_df, seed, smiles_set):
    positive_samples = set(zip(positive_df['Target UniProt ID'], positive_df['Ligand ID']))
    target_set = set(positive_df['Target UniProt ID'])
    
    all_combinations = set((t, s) for t in target_set for s in smiles_set)
    negative_samples = all_combinations - positive_samples
            
    num_positive_samples = len(positive_df)
    negative_samples_selected = random.sample(list(negative_samples), num_positive_samples)
    negative_df = pd.DataFrame(negative_samples_selected, columns=['Target UniProt ID', 'Ligand ID'])
    negative_df['Label'] = 0
    positive_df_copy = positive_df.copy()
    positive_df_copy['Label'] = 1  
    final_df = pd.concat([positive_df_copy[['Target UniProt ID', 'Ligand ID', 'Label']], negative_df])
    final_df = final_df.sample(frac=1, random_state=seed).reset_index(drop=True)
    return final_df

# init
SEED = 1
random.seed(SEED)
df = pd.read_csv('../interactions-pos-1218.csv')
smiles_set = set(df['Ligand ID'])
# split in target-group
train_data = []
val_data = []
test_data = []
for target_id, group in df.groupby('Target UniProt ID'):
    if len(group) < 10:
        group['Label'] = 1
        train_data.append(group)
    else:
        final_df = generate_negative_samples(group, SEED, smiles_set)
        # train, temp = train_test_split(final_df, test_size=0.4, random_state=SEED)
        # val, test = train_test_split(temp, test_size=0.5, random_state=SEED)
        train, temp = train_test_split(final_df, test_size=0.2, random_state=SEED, stratify=final_df['Label'])
        val, test = train_test_split(temp, test_size=0.5, random_state=SEED, stratify=temp['Label'])
        train_data.append(train)
        val_data.append(val)
        test_data.append(test)
train_df = pd.concat(train_data)
val_df = pd.concat(val_data)
test_df = pd.concat(test_data)

# save
train_df.to_csv('train.csv', index=False)
val_df.to_csv('valid.csv', index=False)
test_df.to_csv('test.csv', index=False)
