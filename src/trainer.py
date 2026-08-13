import torch
from torch import nn
from matplotlib import pyplot as plt


def log_rmse(y_hat, y):
    y_hat = torch.clamp(y_hat, 1, float('inf'))
    return (torch.sqrt(((torch.log(y_hat.reshape(y.shape)) - torch.log(y)) ** 2).mean())).item()

def try_gpu(i=0):
     if torch.cuda.device_count() > i:
          return torch.device(f'cuda:{i}')
     return torch.device('cpu')

def evaluate_loss_gpu(net, valid_iter, device):
     ls = []
     for x_num, x_cat, y in valid_iter:
        x_num, x_cat, y = x_num.to(device), x_cat.to(device), y.to(device)
        y_hat = net(x_num, x_cat)
        l = log_rmse(y_hat, y)
        ls.append(l)
     return sum(ls) / len(ls)

def train_epoch(net, train_iter, loss, updater, device):
    ls = []
    for X_num, X_cat, y in train_iter:
        X_num, X_cat, y = X_num.to(device), X_cat.to(device), y.to(device)
        y_hat = net(X_num, X_cat)
        l = loss(y_hat.reshape(y.shape), y)
        updater.zero_grad()
        l.backward()
        updater.step()

        ls.append(log_rmse(y_hat, y))

    return sum(ls) / len(ls)

def train(net, train_iter, valid_iter, lr, weight_decay, num_epochs, device):
    if isinstance(net, nn.Module):
            net.train()

    print(f'train on {device}')
    net = net.to(device)
    loss = nn.MSELoss()
    updater = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)

    l_train, l_valid = [], []
    for epoch in range(num_epochs):
         l = train_epoch(net, train_iter, loss, updater, device)
         lv = evaluate_loss_gpu(net, valid_iter, device)
         l_valid.append(lv)
         l_train.append(l)
         print(f'epoch:{epoch}, train loss: {l:.4f}, valid loss: {lv:.4f}')

    x = [epoch + 1 for epoch in range(num_epochs)]
    plt.plot(x, l_train, label='train loss')
    plt.plot(x, l_valid, label='valid loss')
    plt.legend()
    plt.savefig('loss.png')
    plt.show()