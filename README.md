# GPCRFilter
## Description
**GPCRFilter** is a tool designed to evaluate whether a ligand, represented by its SMILES string, binds to a given GPCR sequence.

---

## Table of Contents
1. [Dataset](#dataset)
2. [Setup Environment](#setup-environment)
3. [Running on Test System](#running-on-test-system)
4. [Retraining](#retraining)
5. [License](#license)

---

## Dataset
The default dataset includes the following components:
1. A mapping of "Target UniProt ID" to "Target Sequence", located at `data/idmapping_target.csv`.
2. A mapping of "Ligand ID" to "SMILES", located at `data/idmapping_ligand.csv`.
3. The dataset in the format "Target UniProt ID,Ligand ID,Label", split into:
   ```
   data/dataset_tag/train.csv
   data/dataset_tag/valid.csv
   data/dataset_tag/test.csv
   ```
   where `dataset_tag` is the name of your dataset.

The dataset is available for download at [https://huggingface.co/datasets/0soyo0/GPCR-dataset-GL-filter](https://huggingface.co/datasets/0soyo0/GPCR-dataset-GL-filter).

---

## Setup Environment
This section outlines how to set up the environment using Anaconda. Start by cloning the repository:
```
git clone https://github.com/wayyzt/DTI.git
```

### Using Conda
Navigate to the cloned directory and create the environment:
```
cd XDTI
conda env create -f environment.yml
```

### Manual Setup
For a manual setup (e.g., if you need specific versions of PyTorch, PyTorch Geometric, CUDA, or CPU-only configurations), follow these steps:
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

Alternatively, use the prebuilt environment pack from [https://huggingface.co/datasets/0soyo0/GPCR-dataset-GL-filter](https://huggingface.co/datasets/0soyo0/GPCR-dataset-GL-filter).

> **Note**: If you encounter version incompatibilities, refer to the `environment.yml` file for guidance.

---

## Running on Test System
> **Important**: Ensure that `id_mapping_csv` does not contain duplicate IDs with the same name.

To run predictions, use the following command:
```
python predict.py \
    --input_data_dir To-Predict.csv \
    --id_mapping_dir id_mapping_csv_dir \
    --output_data_dir To-Predict-/output.csv \
    --dir_save_model To-Use-Checkpoint.pth \
    --fetch_pretrained_target \
    --hid_dim 128 \
    --dropout 0 \
    --batchsize 32 \
    --cuda_use cuda:0
```

Alternatively, execute the provided script:
```
bash run-predict.sh
```

### Troubleshooting
If you encounter a "no checkpoint file" error, download the required checkpoints from [https://huggingface.co/0soyo0/GL-Filter/tree/main](https://huggingface.co/0soyo0/GL-Filter/tree/main).

### Toy Example
To test with a toy example (or run `bash run-predict-toy-example.sh`):
```
python predict.py \
    --input_data_dir data/toy_example/toy_dataset/test.csv \
    --id_mapping_dir data/toy_example/ \
    --output_data_dir predict-toy-example.csv \
    --dir_save_model split-random.pth \
    --fetch_pretrained_target \
    --batchsize 1 \
    --cuda_use cuda:0
```

---

## Retraining
To retrain the model, use the following command:
```
python train.py \
    --dataset_tag split-random \
    --cuda_use cuda:0 \
    --data_dir data \
    --dict_target data/dict_target.pkl \
    --num_epochs 100 \
    --fetch_pretrained_target \
    --lr 1e-5 \
    --batchsize 32
```

Alternatively, run the provided script:
```
bash run-train.sh
```

---

## License
This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).

---

This version maintains all technical details while improving clarity and professionalism. Let me know if you’d like further adjustments!