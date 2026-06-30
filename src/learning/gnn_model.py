import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class SmartScrapeGNN(nn.Module):
    """
    Graph Neural Network for web information extraction (node classifier).
    Architecture: 2-layer GCN + linear head + log-softmax.

    The architecture must match between training and inference:
      conv1: input_dim  -> hidden_dim
      conv2: hidden_dim -> hidden_dim // 2
      fc:    hidden_dim // 2 -> num_classes
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64, num_classes: int = 3):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim // 2)
        self.fc    = nn.Linear(hidden_dim // 2, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.3, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        out = self.fc(x)

        return F.log_softmax(out, dim=1)