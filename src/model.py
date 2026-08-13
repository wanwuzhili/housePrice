import torch
from torch import nn
import joblib
import math


class HouseNet(nn.Module):
    def __init__(self, num_types, num_embs, num_numeric):
        super().__init__()
        self.embeddings = nn.ModuleList(
            [nn.Embedding(a, b) for a, b in zip(num_types, num_embs)]
        )
        num_categorical = sum(num_embs)
        self.mlp = nn.Sequential(
            nn.Linear(num_numeric+num_categorical, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, X_num, X_cat):
        X_emb = [emb(X_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        X_cat = torch.cat(X_emb, dim=1)
        X = torch.cat([X_num, X_cat], dim=1)
        return self.mlp(X)

def load_model(model_path=None):
    le_dict = joblib.load('./le_dict.pkl')
    static_dict = joblib.load('./static_dict.pkl')
    num_types = [len(le.classes_) for le in le_dict.values()]
    num_embs = [round(math.sqrt(n)) for n in num_types]
    num_numeric = len(static_dict.values())
    net = HouseNet(num_types=num_types, num_embs=num_embs, num_numeric=num_numeric)
    
    if model_path is not None:
        net.load_state_dict(torch.load(model_path))

    return net