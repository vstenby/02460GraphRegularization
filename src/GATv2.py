import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GATv2Conv

class GATv2(torch.nn.Module):
    def __init__(self, num_node_features, num_classes, heads=8):
        super().__init__()
        self.gcn1 = GATv2Conv(num_node_features, 16, heads=heads)
        self.gcn2 = GATv2Conv(16 * heads, num_classes, heads=1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.gcn1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, training=self.training)
        x = self.gcn2(x, edge_index)

        return x