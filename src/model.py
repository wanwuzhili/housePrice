import torch
from torch import nn


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