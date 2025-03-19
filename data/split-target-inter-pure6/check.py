import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('/root/gpcr-db/interactions.csv')

group_sizes = df.groupby('Target UniProt ID').size().reset_index(name='size')

group_sizes_sorted = group_sizes.sort_values(by='size', ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(x='Target UniProt ID', y='size', data=group_sizes_sorted, palette='viridis')
plt.title('Distribution of Target Counts', fontsize=16)
plt.ylabel('Number of Samples', fontsize=14)
plt.xlabel('Target UniProt ID', fontsize=14)

plt.savefig('scale-per-target.png')