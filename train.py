import math
import numpy as np
import pandas as pd
import joblib
import torch
import yaml

import src.dataset as ds
import src.model as m
import src.trainer as t

#load configs
with open('./configs/configs.yaml', "r") as f:
    configs = yaml.safe_load(f)

# load data
train_iter, valid_iter = ds.get_data_loader(
    batch_size=configs['batch_size'], num_rare=configs['num_rare']
    )

# load model
net = m.load_model()

# train
t.train(
    net, train_iter, valid_iter, lr=configs['lr'],weight_decay=configs['weight_decay'], 
    num_epochs=configs['num_epochs'], device=t.try_gpu()
    )

torch.save(net.state_dict(), configs['model_save_path'])