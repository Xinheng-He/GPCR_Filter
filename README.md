# XDTI
# Description
**XDTI** is ...

**Things XDTI can do**
- Discover ...
- Discover ...

----
# Table of contents
1. [Dataset](#dataset)
2. [Setup Environment](#setup-environment)
   1. [For GPU](#For-GPU)
3. [Running on test system](#running-on-test-system)
4. [Retraining](#retraining)
5. [License](#license)
# Dataset
The default data contains the following things:
1. The mapping "Target UniProt ID" to "Target Sequence", place it to `data` such that you have the path `data/idmapping_target.csv`.
2. The mapping "Ligand ID" to "SMILES", place it to `data` such that you have the path `data/idmapping_ligand.csv`.
3. The dataset in format "Target UniProt ID,Ligand ID,Label", place it to `data` such that you have the path 
   ```
   data/dataset_tag/train.csv
   data/dataset_tag/valid.csv
   data/dataset_tag/test.csv
   ```
   which `dataset_tag` is your dataset name.

The dataset can be downloaded from https://huggingface.co/datasets/0soyo0/GPCR-dataset-GL-filter
# Setup Environment
We will set up the environment using Anaconda. Clone the current repo:
`git clone https://github.com/wayyzt/DTI.git`
To use conda or mamba to create the environment, you can use:
```
cd XDTI
conda env create -f environment.yml
```
This is an example of how to set up a working conda environment from scratch to run the code (but make sure to use the correct pytorch, pytorch-geometric, cuda versions, or cpu only versions):
```
conda create --name XDTI python=3.12.8
conda activate XDTI
pip install esm==3.1.1
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu118
pip install https://data.pyg.org/whl/torch-2.3.0%2Bcu118/torch_cluster-1.6.3%2Bpt23cu118-cp312-cp312-linux_x86_64.whl
pip install https://data.pyg.org/whl/torch-2.3.0%2Bcu118/torch_scatter-2.1.2%2Bpt23cu118-cp312-cp312-linux_x86_64.whl
pip install https://data.pyg.org/whl/torch-2.3.0%2Bcu118/torch_sparse-0.6.18%2Bpt23cu118-cp312-cp312-linux_x86_64.whl
pip install https://data.pyg.org/whl/torch-2.3.0%2Bcu118/torch_spline_conv-1.2.2%2Bpt23cu118-cp312-cp312-linux_x86_64.whl
pip install torch_geometric
pip install tensorboard
pip install rdkit==2024.9.5
```
Or use the environment pack from https://huggingface.co/datasets/0soyo0/GPCR-dataset-GL-filter
## In case any version is incompatible, check the environment.yml file.
# Running on test system

```
python predict.py \
    --input_data_dir To-Predict.csv \
    --output_data_dir To-Predict-/output.csv \
    --dir_save_model To-Use-Checkpoint.pth \
    --fetch_pretrained_target \
    --hid_dim 128 \
    --dropout 0 \
    --batchsize 32 \
    --cuda_use cuda:0 \
```
Or run the following script directly:
`bash run-predict.sh`
## If **you meet problem with "no checkpoint file"**, you can also download the checkpoints from **https://huggingface.co/0soyo0/GL-Filter/tree/main**
Toy example (or just run the run-predict-toy-example.sh)
```
python predict.py \
    --input_data_dir data/toy_example/toy_dataset/test.csv \
    --output_data_dir predict-toy-example.csv \
    --dir_save_model split-random.pth \
    --fetch_pretrained_target \
    --batchsize 1 \
    --cuda_use cuda:0 \
```
# Retraining
```
python train.py \
    --dataset_tag split-random\
    --cuda_use cuda:0 \
    --data_dir data \
    --dict_target data/dict_target.pkl \
    --num_epochs 100 \
    --fetch_pretrained_target \
    --lr 1e-5 \
    --batchsize 32
```
Or run the following script directly:
`bash run-train.sh`
# License
MIT