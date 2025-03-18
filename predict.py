import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from model import Predictor
from data.dataset import CPIDataset
from data.utils import make_masks_protein
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import argparse
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, average_precision_score, matthews_corrcoef, log_loss
import numpy as np
from datetime import datetime
import os
import random
import pandas as pd

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_data_dir', type=str, default='data')
    parser.add_argument('--id_mapping_dir', type=str, default='data')
    parser.add_argument('--output_data_dir', type=str, default='data')
    parser.add_argument('--dict_target', type=str, default='data/dict_target.pkl')
    parser.add_argument('--dict_ligand', type=str, default='data/dict_ligand.pkl')
    parser.add_argument('--batchsize', type=int, default=128)
    parser.add_argument('--cuda_use', type=str, default='cuda:7')
    parser.add_argument('--hid_target', type=int, default=1536)
    parser.add_argument('--hid_ligand_1d', type=int, default=512)
    parser.add_argument('--hid_ligand_2d', type=int, default=55)
    parser.add_argument('--hid_dim', type=int, default=256)
    parser.add_argument('--dir_save_model', type=str, default='save/best_model.pth')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--n_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--fetch_pretrained_target', action='store_true', default=False)
    parser.add_argument('--fetch_pretrained_ligand', action='store_true', default=False)
    args = parser.parse_args()
    return args

def seed_torch(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
        
def getScore(device, model, loader_data, input_file, output_file):
    model.eval()
    all_labels = []
    all_outputs = []    
    with torch.no_grad():
        for data in tqdm(loader_data, desc=f'eval', unit='batch'):
            data = [x.to(device) for x in data] 
            inputs = data[:-1] 
            labels = data[-1]
            outputs = model(inputs)
            all_labels.append(labels.cpu().numpy())
            all_outputs.append(outputs.cpu().numpy())
        all_labels = np.concatenate(all_labels, axis=0)
        all_outputs = np.concatenate(all_outputs, axis=0)
        all_prob = torch.softmax(torch.tensor(all_outputs), dim=1).numpy()[:, 1]
        
        input_df = pd.read_csv(input_file)
        input_df['Score'] = all_prob
        input_df.to_csv(output_file, index=False)

def getMetrics(output_file):
    data = pd.read_csv(output_file)
    y_true = data['Label']
    y_pred = data['Score']
    
    auc = roc_auc_score(y_true, y_pred)
    aupr = average_precision_score(y_true, y_pred)
    acc = accuracy_score(y_true, (y_pred >= 0.5).astype(int))
    precision = precision_score(y_true, (y_pred >= 0.5).astype(int))
    recall = recall_score(y_true, (y_pred >= 0.5).astype(int))
    # f1 = f1_score(y_true, (y_pred >= 0.5).astype(int))
    # mcc = matthews_corrcoef(y_true, (y_pred >= 0.5).astype(int))
    # logloss = log_loss(y_true, y_pred)
    
    print(f'''
            auc: {auc:.4f}, 
            aupr: {aupr:.4f}, 
            acc: {acc:.4f}, 
            precision: {precision:.4f}, 
            recall: {recall:.4f}, 
            \n
            '''
            )

if __name__ == '__main__':
    # init
    args = get_args()
    current_date = datetime.now().strftime("%Y%m%d-%H%M")
    seed_torch(args.seed)
    # prepare-data
    data_test = CPIDataset(args, 'predict')
    loader_test = DataLoader(data_test, batch_size=args.batchsize, shuffle=False, worker_init_fn=np.random.seed(args.seed), collate_fn=make_masks_protein)
    device = torch.device(args.cuda_use if torch.cuda.is_available() else 'cpu')
    model = Predictor(args)
    model.to(device)
    # model.load_state_dict(torch.load(args.dir_save_model))
    model.load_state_dict(torch.load(args.dir_save_model, map_location=device))
    input_file = os.path.join(args.input_data_dir)
    output_file = os.path.join(args.output_data_dir)
    getScore(device, model, loader_test, input_file, output_file)
    # getMetrics(output_file)