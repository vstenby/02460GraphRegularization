import numpy as np
import torch

def get_masks(A,B,dataset,seed=0):
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
        class_idx = np.argwhere((y==i).numpy()).flatten()
        np.random.seed(seed)
        np.random.shuffle(class_idx)

        train_mask[class_idx[:A]] = True
        val_mask[class_idx[A:A+B]] = True
        test_mask[class_idx[A+B:len(class_idx)]] = True
    return train_mask.bool(), val_mask.bool(), test_mask.bool()