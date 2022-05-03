import torch
import torch_geometric.utils as utils

class LapLoss:
    def __init__(self, edge_index, device=None):
        self.edge_index = edge_index

    def __call__(self, Z):
        #Returns the Laplacian loss.
        Zfrom = Z[self.edge_index[0,:], ]
        Zto   = Z[self.edge_index[1,:], ]
        reg = (torch.norm(Zfrom - Zto, p=2, dim=1)).mean()
        return self.edge_index.size(0) * reg * reg