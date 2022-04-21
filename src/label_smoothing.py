import torch

def label_smoothing(y, C, epsilon):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n = len(y)
    y_onehot = torch.zeros(n, C).to(device)
    y_onehot = y_onehot.scatter_(1, y.unsqueeze(1), 1).to("cpu")
    return y_onehot.apply_(lambda x: epsilon/(C-1) if x == 0 else x-epsilon).to(device)
