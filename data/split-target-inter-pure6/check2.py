import pandas as pd

valid_df = pd.read_csv('valid.csv')
test_df = pd.read_csv('test.csv')

valid_data = set(valid_df['Target UniProt ID'].tolist())
test_data = set(test_df['Target UniProt ID'].tolist())
print(valid_data)
print(test_data)
print(valid_data & test_data)