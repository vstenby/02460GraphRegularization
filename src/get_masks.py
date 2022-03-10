import numpy as np
import torch

def get_masks(A,B,dataset):
    N = dataset.data.num_nodes
    y = dataset.data.y
    C = dataset.num_classes

    _, counts = np.unique(dataset.data.y.numpy(), return_counts=True)

    assert (A+B) <= len(y), f'(A+B) is larger than the number of nodes.'
    assert not np.any(counts < (A+B)), f'(A+B) is larger than n(class) for one of the classes.'

    train_mask = torch.Tensor([False]).repeat(N) 
    val_mask = torch.Tensor([False]).repeat(N)
    test_mask = torch.Tensor([False]).repeat(N)

    for i in range(C):
        num_in_class = (y == i).sum()
        train_mask[torch.where((y == i))[0][:A]] = True
        val_mask[torch.where((y == i))[0][A:A+B]] = True
        test_mask[torch.where((y == i))[0][A+B:num_in_class]] = True
    return train_mask.bool(), val_mask.bool(), test_mask.bool()