import pandas as pd
import pickle
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from data.utils import *
import os


with open('/datapool/data2/home/majianzhu/xinheng/xiangzhen/DTI2/DTI/data/dict_target.pkl', 'rb') as f:
    dict_target = pickle.load(f)
print(dict_target['P21453'])
print(dict_target['P30935'])