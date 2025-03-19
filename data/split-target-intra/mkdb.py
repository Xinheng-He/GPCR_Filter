import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('/root/gpcr-db/interactions.csv')

seed = 1

train_data = []
val_data = []
test_data = []

for target_id, group in df.groupby('Target UniProt ID'):
    if len(group) < 5:
        train_data.append(group)
    else:
        train, temp = train_test_split(group, test_size=0.4, random_state=seed)
        val, test = train_test_split(temp, test_size=0.5, random_state=seed)
        train_data.append(train)
        val_data.append(val)
        test_data.append(test)

train_df = pd.concat(train_data)
val_df = pd.concat(val_data)
test_df = pd.concat(test_data)

print(len(train_df))
print(len(val_df))
print(len(test_df))

train_df.to_csv('train.csv', index=False)
val_df.to_csv('valid.csv', index=False)
test_df.to_csv('test.csv', index=False)