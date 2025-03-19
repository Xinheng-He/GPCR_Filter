import pandas as pd
import random
from sklearn.model_selection import train_test_split
'''
first, neg-sampling in whole-data
second, split train:valid:test = 3:1:1
'''
def generate_negative_samples(positive_df, seed):
    positive_samples = set(zip(positive_df['Target UniProt ID'], positive_df['Ligand ID']))
    target_set = set(positive_df['Target UniProt ID'])
    smiles_set = set(positive_df['Ligand ID'])
    
    all_combinations = set((t, s) for t in target_set for s in smiles_set)
    negative_samples = all_combinations - positive_samples
            
    num_positive_samples = len(positive_df)
    negative_samples_selected = random.sample(negative_samples, num_positive_samples)
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
# neg-sampling
final_df = generate_negative_samples(df, SEED)
# split
df_train, df_tmp = train_test_split(final_df, test_size=0.2, random_state=SEED, stratify=final_df['Label'])
df_valid, df_test = train_test_split(df_tmp, test_size=0.5, random_state=SEED, stratify=df_tmp['Label'])
# save
df_train.to_csv('train.csv', index=False)
df_valid.to_csv('valid.csv', index=False)
df_test.to_csv('test.csv', index=False)