# This branch just for Explainability Research
**pipeline**
1. Crawling data from gpcr-structure: crawl-0401/fetch-visual.py
2. Predict collected data, get highlight index in FASTA-Uniprot
3. Align FASTA-Uniprot to PDB-Sequence
4. Check PDB-Sequence Manually
**data collected dir**
```
crawl-0401/filtered_idmapping_ligand_visual.csv # dict: ligand_id to ligand_smiles 
crawl-0401/filtered_test_visual.csv # dataset
```
Notice, for filtered_idmapping_ligand_visual:
1. (These smiles do not in origin dict, so recollect. The dict dir has been updated in dataset.py & predict-3.py)
2. (Some data be wrapped in '"'. The mapping is guaranteed, so no further processing)
**how to use**
```
bash run-predict.sh
bash run-align.sh
```
Notice:
just predict pre-100 data in dataset.