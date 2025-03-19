import pandas as pd
from sklearn.model_selection import train_test_split
SEED = 1


df = pd.read_csv('/root/gpcr-db/interactions.csv')
target_ids = df['Target UniProt ID'].unique()

train_target_ids, test_target_ids = train_test_split(target_ids, test_size=0.2, random_state=seed)
train_data = df[df['Target UniProt ID'].isin(train_target_ids)]
test_data = df[df['Target UniProt ID'].isin(test_target_ids)]

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

train_final_df = generate_negative_samples(train_data, SEED)
test_final_df = generate_negative_samples(test_data, SEED)

train_data.to_csv('train.csv', index=False)
test_data.to_csv('test.csv', index=False)