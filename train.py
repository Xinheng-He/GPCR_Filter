import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from model.decoder import Decoder
from data.dataset import CPIDataset, custom_collate_fn
from torch.utils.data import DataLoader
import argparse
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, f1_score
import numpy as np
import os
import random

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_tag', type=str, default='split-random')
    parser.add_argument('--dataset_dir', type=str, default='xx')
    parser.add_argument('--dict_target', type=str, default='data/dict_target.pkl')
    parser.add_argument('--dict_ligand', type=str, default='/datapool/data2/home/majianzhu/xinheng/xiangzhen/mk-dict/unimol2/dict_ligand.pkl')
    parser.add_argument('--batchsize', type=int, default=128)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--cuda_use', type=str, default='cuda:7')
    parser.add_argument('--hidden_target', type=int, default=1536)
    parser.add_argument('--hidden_ligand', type=int, default=512)
    parser.add_argument('--hidden', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--dir_save_model', type=str, default='save/best_model.pth')
    parser.add_argument('--seed', type=int, default=1)
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

def train(args, device, model, loader_train, criterion, optimizer, loader_valid):
    acc_best = 0.0
    model.train()
    for epoch in range(args.num_epochs):
        loss_running = 0.0
        for data in tqdm(loader_train, desc=f'epoch: {epoch+1}/{args.num_epochs}', unit='batch'):
            padded_proteins, padded_ligands, protein_mask, ligand_mask, labels = data
            padded_proteins, padded_ligands, protein_mask, ligand_mask, labels = padded_proteins.to(device), padded_ligands.to(device), protein_mask.to(device), ligand_mask.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(padded_proteins, padded_ligands, protein_mask, ligand_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            loss_running += loss.item() * args.batchsize
        loss_epoch = loss_running / len(loader_train.dataset)
        print(f'epoch[{epoch+1}/{args.num_epochs}, Train_loss: {loss_epoch:.4f}]')
        # tensorboard pytorch-lightning
        
        acc_running = eval(args, device, model, loader_valid, criterion)    
        if acc_running > acc_best:
            acc_best = acc_running
            torch.save(model.state_dict(), args.dir_save_model)
        
        
def eval(args, device, model, loader_data, criterion):
    model.eval()
    all_labels = []
    all_outputs = []    
    with torch.no_grad():
        loss_running = 0.0
        for data in tqdm(loader_data, desc=f'eval', unit='batch'):
            padded_proteins, padded_ligands, protein_mask, ligand_mask, labels = data
            padded_proteins, padded_ligands, protein_mask, ligand_mask, labels = padded_proteins.to(device), padded_ligands.to(device), protein_mask.to(device), ligand_mask.to(device), labels.to(device)
            outputs = model(padded_proteins, padded_ligands, protein_mask, ligand_mask)
            loss = criterion(outputs, labels)
            loss_running += loss.item() * args.batchsize
            
            all_labels.append(labels.cpu().numpy())
            all_outputs.append(outputs.cpu().numpy())
        loss_eval = loss_running / len(loader_data.dataset)
        
        all_labels = np.concatenate(all_labels, axis=0)
        all_outputs = np.concatenate(all_outputs, axis=0)
        probs = torch.softmax(torch.tensor(all_outputs), dim=1).numpy()[:, 1]
        preds = (probs >= 0.5).astype(int)
        acc = accuracy_score(all_labels, preds)
        auroc = roc_auc_score(all_labels, probs)
        aupr = average_precision_score(all_labels, probs)
        f1 = f1_score(all_labels, preds)
        print(f'eval[Eval_loss: {loss_eval:.4f}, Accuracy: {acc:.4f}, AUROC: {auroc:.4f}, AUPR: {aupr:.4f}, F1: {f1:.4f}]')           
    
    return acc

if __name__ == '__main__':
    args = get_args()
    args.dataset_dir = os.path.join('data', f'{args.dataset_tag}')
    args.dir_save_model = os.path.join('save', f'{args.dataset_tag}.pth')
    seed_torch(args.seed)
    # prepare-data
    args.dict_ligand = '/datapool/data2/home/majianzhu/xinheng/xiangzhen/mk-dict/unimol2/dict_ligand.pkl'
    data_train = CPIDataset(os.path.join(args.dataset_dir, 'train.csv'), args.dict_target, args.dict_ligand)
    data_valid = CPIDataset(os.path.join(args.dataset_dir, 'valid.csv'), args.dict_target, args.dict_ligand)
    data_test = CPIDataset(os.path.join(args.dataset_dir, 'test.csv'), args.dict_target, args.dict_ligand)
    loader_train = DataLoader(data_train, batch_size=args.batchsize, shuffle=True, worker_init_fn=np.random.seed(args.seed), collate_fn=custom_collate_fn)
    loader_valid = DataLoader(data_valid, batch_size=args.batchsize, shuffle=True, worker_init_fn=np.random.seed(args.seed), collate_fn=custom_collate_fn)
    loader_test = DataLoader(data_test, batch_size=args.batchsize, shuffle=True, worker_init_fn=np.random.seed(args.seed), collate_fn=custom_collate_fn)
    # prepare-model
    device = torch.device(args.cuda_use if torch.cuda.is_available() else 'cpu')
    model = Decoder(args.hidden_target, args.hidden_ligand, args.hidden)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    train(args, device, model, loader_train, criterion, optimizer, loader_valid)
    eval(args, device, model, loader_test, criterion)
    