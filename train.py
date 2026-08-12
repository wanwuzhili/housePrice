import math
import numpy as np
import pandas as pd
import joblib
import torch

import src.dataset as ds
import src.model as m
import src.trainer as t


# load data
batch_size = 256
train_iter, valid_iter = ds.get_data_loader(root='./data/raw/train.csv', batch_size=batch_size, num_rare=3)

# load model
le_dict = joblib.load('./le_dict.pkl')
static_dict = joblib.load('./static_dict.pkl')
num_types = [len(le.classes_) for le in le_dict.values()]
num_embs = [round(math.sqrt(n)) for n in num_types]
num_numeric = len(static_dict.values())
net = m.HouseNet(num_types=num_types, num_embs=num_embs, num_numeric=num_numeric)

# train
lr = 0.01
num_epochs = 50
t.train(net, train_iter, valid_iter, lr=lr, num_epochs=num_epochs, device=t.try_gpu())

torch.save(net.state_dict(), "model_lr.01_epoch50.pth")