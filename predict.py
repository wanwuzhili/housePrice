import torch
import pandas as pd
import joblib
import math

import src.dataset as ds
import src.model as m


test_data = pd.read_csv('./data/raw/test.csv')
test_num, test_cat = ds.preprocess_test(test_data)

le_dict = joblib.load('./le_dict.pkl')
static_dict = joblib.load('./static_dict.pkl')
num_types = [len(le.classes_) for le in le_dict.values()]
num_embs = [round(math.sqrt(n)) for n in num_types]
num_numeric = len(static_dict.values())
net = m.HouseNet(num_types=num_types, num_embs=num_embs, num_numeric=num_numeric)
net.load_state_dict(torch.load('model_lr.01_epoch50.pth'))

# predict and submission
if isinstance(net, torch.nn.Module):
    net.eval()
y_pred = torch.expm1(net(test_num, test_cat)).detach().numpy()

test_data['SalePrice'] = pd.Series(y_pred.reshape(1, -1)[0])
submission = pd.concat([test_data['Id'], test_data['SalePrice']], axis=1)
submission.to_csv('submission.csv', index=False)