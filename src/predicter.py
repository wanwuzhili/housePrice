
def predict(net, test_num, test_cat):
    return net(test_num, test_cat).detach().numpy()