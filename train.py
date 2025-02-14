import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from model import Predictor
from data.dataset import CPIDataset
from data.utils import make_masks_protein
# from torch_geometric.loader import DataLoader
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import argparse
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, f1_score
import numpy as np
from datetime import datetime
import os
import random

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_tag', type=str, default='split-random')
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--dict_target', type=str, default='data/dict_target.pkl')
    parser.add_argument('--dict_ligand', type=str, default='data/dict_ligand.pkl')
    parser.add_argument('--batchsize', type=int, default=128)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--cuda_use', type=str, default='cuda:7')
    parser.add_argument('--hid_target', type=int, default=1536)
    parser.add_argument('--hid_ligand_1d', type=int, default=512)
    parser.add_argument('--hid_ligand_2d', type=int, default=55)
    parser.add_argument('--hid_dim', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-4)
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

def train(args, device, model, loader_train, loader_valid, loader_test, criterion, optimizer, writer):
    acc_best = 0.0
    model.train()
    if args.dataset_tag == 'split-target-inter':
        loader_test = loader_valid
    for epoch in range(args.num_epochs):
        loss_running = 0.0
        for data in tqdm(loader_train, desc=f'epoch: {epoch+1}/{args.num_epochs}', unit='batch'):
            data = [x.to(device) for x in data] 
            inputs = data[:-1] 
            labels = data[-1]
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            loss_running += loss.item() * args.batchsize
        loss_epoch = loss_running / len(loader_train.dataset)
        print(f'epoch[{epoch+1}/{args.num_epochs}, Train_loss: {loss_epoch:.4f}]')
        writer.add_scalar('Loss/train', loss_epoch, epoch)

        acc_running = eval(args, device, model, loader_valid, criterion, writer, 'valid', epoch)    
        if acc_running > acc_best:
            acc_best = acc_running
            torch.save(model.state_dict(), args.dir_save_model)
        model.train()
    
    # eval(args, device, model, loader_test, criterion, writer, 'test', 0)
        
def eval(args, device, model, loader_data, criterion, writer, phase, epoch):
    model.eval()
    all_labels = []
    all_outputs = []    
    with torch.no_grad():
        loss_running = 0.0
        for data in tqdm(loader_data, desc=f'eval', unit='batch'):
            data = [x.to(device) for x in data] 
            inputs = data[:-1] 
            labels = data[-1]
            outputs = model(inputs)
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

        writer.add_scalar(f'Loss/{phase}', loss_eval, epoch)
        writer.add_scalar(f'Accuracy/{phase}', acc, epoch)
        writer.add_scalar(f'AUROC/{phase}', auroc, epoch)
        writer.add_scalar(f'AUPR/{phase}', aupr, epoch)
        writer.add_scalar(f'F1/{phase}', f1, epoch)           
    
    return acc

if __name__ == '__main__':
    # init
    args = get_args()
    current_date = datetime.now().strftime("%Y%m%d-%H%M")
    run_dir = os.path.join('run', f'{current_date}_{args.dataset_tag}')
    os.makedirs(run_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=run_dir)
    args.dir_save_model = os.path.join(run_dir, f'best_acc.pth')
    seed_torch(args.seed)
    # prepare-data
    data_train = CPIDataset(args, 'train')
    data_valid = CPIDataset(args, 'valid')
    data_test = CPIDataset(args, 'test')
    loader_train = DataLoader(data_train, batch_size=args.batchsize, shuffle=False, worker_init_fn=np.random.seed(args.seed), collate_fn=make_masks_protein)
    loader_valid = DataLoader(data_valid, batch_size=args.batchsize, shuffle=False, worker_init_fn=np.random.seed(args.seed), collate_fn=make_masks_protein)
    loader_test = DataLoader(data_test, batch_size=args.batchsize, shuffle=False, worker_init_fn=np.random.seed(args.seed), collate_fn=make_masks_protein)
    # prepare-model
    device = torch.device(args.cuda_use if torch.cuda.is_available() else 'cpu')
    
    model = Predictor(args)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    train(args, device, model, loader_train, loader_valid, loader_test, criterion, optimizer, writer)
    model.load_state_dict(torch.load(args.dir_save_model))
    eval(args, device, model, loader_test, criterion, writer, 'test', 0)
    writer.close()