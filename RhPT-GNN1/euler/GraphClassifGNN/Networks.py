import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    TransformerConv,
    Linear
)

"""
In this file there are a few things can be chanegd in the architecture: number of layers, size of hidden layers, etc...
"""


class NetGAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()

        # Main Hidden layers numbers ---------------------------------------------------------------------------------
        mid = 12
        num_heads = 4
        # There is also hidden_channels in the train.py header
        # ------------------------------------------------------------------------------------------------------------

        self.conv1 = TransformerConv(in_channels = in_channels, out_channels = hidden_channels, heads=num_heads)
        self.bn1 = nn.BatchNorm1d(hidden_channels * num_heads)
        self.conv2 = TransformerConv(hidden_channels * num_heads, hidden_channels * num_heads)
        #self.conv2 = TransformerConv(hidden_channels * num_heads, out_channels)
        self.bn2 = nn.BatchNorm1d(hidden_channels * num_heads)
        self.conv3 = TransformerConv(hidden_channels * num_heads, hidden_channels * num_heads)
        self.conv4 = TransformerConv(hidden_channels * num_heads, hidden_channels * num_heads)
        self.conv5 = TransformerConv(hidden_channels * num_heads, hidden_channels * num_heads)
        self.conv6 = TransformerConv(hidden_channels * num_heads, hidden_channels * num_heads)
        self.conv7 = TransformerConv(hidden_channels * num_heads, hidden_channels * num_heads)
        self.conv8 = TransformerConv(hidden_channels * num_heads, hidden_channels * num_heads)

        self.readout = nn.Sequential(

            nn.Linear(hidden_channels * num_heads, mid * num_heads),
            #nn.Dropout(p=0.1),
            nn.Tanhshrink(),

            #nn.Linear(mid * num_heads,mid* num_heads),
            #nn.Dropout(p=0.1),
            #nn.Tanhshrink(),

            nn.Linear(mid * num_heads, out_channels),

        )

    def bn(self, i, x):
        num_nodes, num_channels = x.size()

        x = x.view(-1, num_channels)
        #x = getattr(self, f"bn{i}")(x)
        x = x.view(num_nodes, num_channels)
        return x

    def forward(self, x):
        x0 = x

        x1 = self.bn(1, self.conv1(x0.x, x0.edge_index)) # emb dim: n_nodes * n_feat_node [2] = n_nodes * n_feat_node [hidden_channels]
        #x1 = F.tanhshrink(x1)
        x2 = self.bn(2, self.conv2(x1, x0.edge_index))  # emb dim: n_nodes * n_feat_node [hidden_channels] = n_nodes * n_feat_node [hidden_channels]
        x3 = self.bn(3, self.conv3(x2, x0.edge_index))  # fin emb_dim: n_nodes * n_feat_node [hidden_channels]
        x4 = self.bn(4, self.conv4(x3, x0.edge_index))  # fin emb_dim: n_nodes * n_feat_node [hidden_channels]
        x5 = self.bn(5, self.conv5(x4, x0.edge_index))
        x6 = self.bn(6, self.conv6(x5, x0.edge_index))
        x7 = self.bn(7, self.conv7(x6, x0.edge_index))
        x8 = self.bn(8, self.conv8(x7, x0.edge_index))

        mlp = self.readout(x7) #TODO: record all the layers and hyperparams in the experimental setup
        return mlp
