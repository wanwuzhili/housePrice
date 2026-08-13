import torch
import pandas as pd
import joblib
import math
import yaml

import src.dataset as ds
import src.model as m


#load configs
with open('./configs/configs.yaml', "r") as f:
    configs = yaml.safe_load(f)

test_data_path = configs['raw_data_path'] + 'test.csv'
test_data = pd.read_csv(test_data_path)
test_num, test_cat = ds.preprocess_test(test_data)

net = m.load_model(model_path=configs['model_save_path'])

# predict and submission
if isinstance(net, torch.nn.Module):
    net.eval()
y_pred = torch.expm1(net(test_num, test_cat)).detach().numpy()

test_data['Sold Price'] = pd.Series(y_pred.reshape(1, -1)[0])
submission = pd.concat([test_data['Id'], test_data['Sold Price']], axis=1)
submission.to_csv('submission.csv', index=False)