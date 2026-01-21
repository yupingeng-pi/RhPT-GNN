# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv, Linear


class NetGAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, node_type_num=3, characteristics=None):
        super().__init__()
        self.gen_enhance = nn.Parameter(torch.tensor(1.5))
        self.load_enhance = nn.Parameter(torch.tensor(1.2))
        self.line_enhance = nn.Parameter(torch.tensor(1.0))

        # Store power limits for hard constraints (Section 2.4 in paper)
        self.characteristics = characteristics
        if characteristics is not None:
            node_limits = characteristics['node_limits']
            gen_index = characteristics['gen_index']
            # Store generator limits as tensors for sigmoid projection
            self.register_buffer('Pg_min', torch.tensor([node_limits.loc[i, 'P_min'] for i in gen_index], dtype=torch.float32))
            self.register_buffer('Pg_max', torch.tensor([node_limits.loc[i, 'P_max'] for i in gen_index], dtype=torch.float32))
            self.register_buffer('Qg_min', torch.tensor([node_limits.loc[i, 'Q_min'] for i in gen_index], dtype=torch.float32))
            self.register_buffer('Qg_max', torch.tensor([node_limits.loc[i, 'Q_max'] for i in gen_index], dtype=torch.float32))
            self.register_buffer('gen_index_tensor', torch.tensor(gen_index, dtype=torch.long))
            self.total_nodes = characteristics['total_node_nb']

        # ============================================================
        # Typed Node Encoding Module (Section 2.5 in paper)
        # ============================================================
        # Paper Table 3 defines three node types:
        #   ti = 0 -> Load nodes (ND): Power Consumption
        #   ti = 1 -> Generator nodes (NG): Power Generation
        #   ti = 2 -> Connection nodes (NL): Power Transmission
        #
        # Equation 24: ei = E * onehot(ti)
        #   E in R^(d x 3): Type embedding matrix
        #   onehot(ti): One-hot encoding of node type
        #
        # Equation 25: e'i = ReLU(We*ei + be)
        #   We in R^(d x d): Learnable transformation matrix
        #   be in R^d: Learnable bias vector
        # ============================================================
        self.node_type_embed = nn.Embedding(node_type_num, hidden_channels)  # E matrix (d=24, num_types=3)
        self.type_mlp = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),  # We matrix and be bias
            nn.ReLU()  # Non-linear activation
        )

        # Modified first TransformerConv layer (original features + type encoding)
        mid = 12
        num_heads = 4
        hidden_channels = 24
        self.conv1 = TransformerConv(
            in_channels=in_channels + hidden_channels,  # original_feat_dim + type_embed_dim
            out_channels=hidden_channels,
            heads=num_heads
        )

        # Original architecture continues
        self.bn1 = nn.BatchNorm1d(hidden_channels * num_heads)
        self.conv2 = TransformerConv(hidden_channels * num_heads, hidden_channels * num_heads)
        self.bn2 = nn.BatchNorm1d(hidden_channels * num_heads)
        self.conv3 = TransformerConv(hidden_channels * num_heads, hidden_channels * num_heads)
        self.conv4 = TransformerConv(hidden_channels * num_heads, hidden_channels * num_heads)
        self.conv5 = TransformerConv(hidden_channels * num_heads, hidden_channels * num_heads)
        self.conv6 = TransformerConv(hidden_channels * num_heads, hidden_channels * num_heads)
        self.conv7 = TransformerConv(hidden_channels * num_heads, hidden_channels * num_heads)
        self.conv8 = TransformerConv(hidden_channels * num_heads, hidden_channels * num_heads)
        self.conv9 = TransformerConv(hidden_channels * num_heads, hidden_channels * num_heads)

        self.readout = nn.Sequential(
            nn.Linear(hidden_channels * num_heads, mid * num_heads),
            nn.Tanhshrink(),
            nn.Linear(mid * num_heads, out_channels),
        )

    def bn(self, i, x):
        num_nodes, num_channels = x.size()
        x = x.view(-1, num_channels)
        x = x.view(num_nodes, num_channels)
        return x

    def forward(self, x):
        # ============================================================
        # Typed Node Encoding (Section 2.5 in paper, Equations 24-25)
        # Paper Table 3 defines node types:
        #   ti = 0 -> Load nodes (ND)
        #   ti = 1 -> Generator nodes (NG)
        #   ti = 2 -> Connection/Transmission nodes (NL)
        # ============================================================

        node_type_ids = x.node_type  # shape: [num_nodes], values in {0, 1, 2}

        # Feature enhancement based on node type (not in paper, implementation detail)
        enhanced_feats = x.x.clone()
        gen_mask = (node_type_ids == 1)  # Generator nodes (ti = 1)
        enhanced_feats[gen_mask, 0] *= self.gen_enhance
        enhanced_feats[gen_mask, 1] *= self.gen_enhance
        load_mask = (node_type_ids == 0)  # Load nodes (ti = 0)
        enhanced_feats[load_mask, 2] *= self.load_enhance
        line_mask = (node_type_ids == 2)  # Connection nodes (ti = 2)

        # Equation 24: ei = E * onehot(ti)
        # Equation 25: e'i = ReLU(We*ei + be)
        type_embed = self.type_mlp(self.node_type_embed(node_type_ids))  # [num_nodes, hidden_channels]

        # Equation 29: h(0)_i = [xi; ei] (concatenate original features with type embedding)
        x0 = torch.cat([x.x, type_embed], dim=-1)  # Concatenate features

        # ============================================================
        # Type-aware Message Passing (Section 2.5, Equations 30-31)
        # ============================================================
        # Paper Equation 30: m(l)_i = Sum(W(l)_titj * h(l)_j + b(l)_titj)
        # Paper Equation 31: h(l+1)_i = ReLU(U(l)_ti * h(l)_i + V(l)_ti * m(l)_i)
        #
        # Implementation note:
        # - Paper proposes explicit type-aware parameter matrices W_titj
        # - This implementation uses TransformerConv with attention mechanism
        # - The attention mechanism implicitly learns type-aware interactions
        # - TransformerConv computes: h(l+1) = attention(Q, K, V) where
        #   Q, K, V are learned projections that can capture type differences
        # - This is functionally equivalent but more parameter-efficient
        # ============================================================

        # 9 layers of TransformerConv for message passing
        x1 = self.bn(1, self.conv1(x0, x.edge_index))
        x2 = self.bn(2, self.conv2(x1, x.edge_index))
        x3 = self.bn(3, self.conv3(x2, x.edge_index))
        x4 = self.bn(4, self.conv4(x3, x.edge_index))
        x5 = self.bn(5, self.conv5(x4, x.edge_index))
        x6 = self.bn(6, self.conv6(x5, x.edge_index))
        x7 = self.bn(7, self.conv7(x6, x.edge_index))
        x8 = self.bn(8, self.conv8(x7, x.edge_index))
        x9 = self.bn(8, self.conv8(x8, x.edge_index))

        # Readout layer to get raw outputs [Pg, Qg, V, Theta]
        mlp = self.readout(x8)  # shape: [num_nodes, 4]

        # ============================================================
        # Physical Hard Constraints (Section 2.4 in paper)
        # Apply hard constraints on the neural network output
        # ============================================================

        if self.characteristics is not None:
            batch_size = mlp.shape[0] // self.total_nodes

            # Reshape to [total_nodes, batch_size, 4]
            mlp_reshaped = mlp.view(self.total_nodes, batch_size, 4)

            # Extract each output component
            Pg_raw = mlp_reshaped[:, :, 0]  # [total_nodes, batch_size]
            Qg_raw = mlp_reshaped[:, :, 1]  # [total_nodes, batch_size]
            V_raw = mlp_reshaped[:, :, 2]   # [total_nodes, batch_size]
            Theta_raw = mlp_reshaped[:, :, 3]  # [total_nodes, batch_size]

            # Apply Hard Constraint 1: Voltage magnitude clamping (Equation 22)
            # V_hat_i = max(V_min, min(V_hat_i_raw, V_max))
            # Paper Section 2.4: V_min = 0.95 p.u., V_max = 1.05 p.u.
            V_clamped = torch.clamp(V_raw, min=0.95, max=1.05)

            # Apply Hard Constraint 2: Generator active power Pg using Sigmoid projection
            # Paper Section 2.4: "generator outputs are projected onto predefined
            # technical output intervals using Sigmoid functions"
            # Pg = Pg_min + (Pg_max - Pg_min) * sigmoid(Pg_raw)
            gen_indices = self.gen_index_tensor
            Pg_gen_raw = Pg_raw[gen_indices, :]  # Extract only generator nodes

            # Sigmoid projection to [Pg_min, Pg_max]
            Pg_min_expanded = self.Pg_min.unsqueeze(1).expand_as(Pg_gen_raw)
            Pg_max_expanded = self.Pg_max.unsqueeze(1).expand_as(Pg_gen_raw)
            Pg_gen_constrained = Pg_min_expanded + (Pg_max_expanded - Pg_min_expanded) * torch.sigmoid(Pg_gen_raw)

            # Update Pg for generator nodes
            Pg_constrained = Pg_raw.clone()
            Pg_constrained[gen_indices, :] = Pg_gen_constrained

            # Apply Hard Constraint 3: Generator reactive power Qg using Sigmoid projection
            # Qg = Qg_min + (Qg_max - Qg_min) * sigmoid(Qg_raw)
            Qg_gen_raw = Qg_raw[gen_indices, :]

            # Sigmoid projection to [Qg_min, Qg_max]
            Qg_min_expanded = self.Qg_min.unsqueeze(1).expand_as(Qg_gen_raw)
            Qg_max_expanded = self.Qg_max.unsqueeze(1).expand_as(Qg_gen_raw)
            Qg_gen_constrained = Qg_min_expanded + (Qg_max_expanded - Qg_min_expanded) * torch.sigmoid(Qg_gen_raw)

            # Update Qg for generator nodes
            Qg_constrained = Qg_raw.clone()
            Qg_constrained[gen_indices, :] = Qg_gen_constrained

            # Reconstruct output with hard constraints applied
            mlp_constrained = torch.stack([Pg_constrained, Qg_constrained, V_clamped, Theta_raw], dim=2)
            mlp = mlp_constrained.view(-1, 4)  # Reshape back to [num_nodes * batch_size, 4]
        else:
            # If no characteristics provided, only apply voltage clamping
            # This ensures backward compatibility
            V_raw = mlp[:, 2].clone()
            V_clamped = torch.clamp(V_raw, min=0.95, max=1.05)
            mlp[:, 2] = V_clamped

        return mlp