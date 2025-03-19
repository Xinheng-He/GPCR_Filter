import pandas as pd
from sklearn.model_selection import train_test_split

seed = 1

df = pd.read_csv('/root/gpcr-db/interactions.csv')
df_train, df_tmp = train_test_split(df, test_size=0.4, random_state=seed)
df_valid, df_test = train_test_split(df_tmp, test_size=0.5, random_state=seed)
df_train.to_csv('train.csv', index=False)
df_valid.to_csv('valid.csv', index=False)
df_test.to_csv('test.csv', index=False)

print(f"df {df.shape[0]}")
print(f"df_train {df_train.shape[0]}")
print(f"df_valid {df_valid.shape[0]}")
print(f"df_test {df_test.shape[0]}")