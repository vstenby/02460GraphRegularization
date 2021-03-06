import torch
import torch.nn.functional as F
from   torch_geometric.nn import GCNConv

class GCN(torch.nn.Module):
    '''
    GCN 
    '''
    def __init__(self, num_node_features, num_classes, num_hidden_features=16):
        super().__init__()
        self.conv1 = GCNConv(num_node_features, num_hidden_features)
        self.conv2 = GCNConv(num_hidden_features, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return x