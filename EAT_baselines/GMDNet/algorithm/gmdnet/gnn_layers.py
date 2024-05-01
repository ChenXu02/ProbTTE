import torch
import torch.nn.functional as F
import torch.nn as nn

class BatchNormNode(nn.Module):

    def __init__(self, hidden_dim):
        super(BatchNormNode, self).__init__()
        self.batch_norm = nn.BatchNorm1d(hidden_dim, track_running_stats=False)

    def forward(self, x):

        x_trans = x.transpose(1, 2).contiguous()
        x_trans_bn = self.batch_norm(x_trans)
        x_bn = x_trans_bn.transpose(1, 2).contiguous()
        return x_bn


class BatchNormEdge(nn.Module):

    def __init__(self, hidden_dim):
        super(BatchNormEdge, self).__init__()
        self.batch_norm = nn.BatchNorm2d(hidden_dim, track_running_stats=False)

    def forward(self, e):

        e_trans = e.transpose(1, 3).contiguous()
        e_trans_bn = self.batch_norm(e_trans)
        e_bn = e_trans_bn.transpose(1, 3).contiguous()
        return e_bn


class NodeFeatures(nn.Module):

    def __init__(self, hidden_dim, aggregation="mean"):
        super(NodeFeatures, self).__init__()
        self.aggregation = aggregation
        self.U = nn.Linear(hidden_dim, hidden_dim, True)
        self.V = nn.Linear(hidden_dim, hidden_dim, True)

    def forward(self, x, edge_gate):

        Ux = self.U(x)
        Vx = self.V(x)
        Vx = Vx.unsqueeze(1)
        gateVx = edge_gate * Vx
        if self.aggregation == "mean":
            x_new = Ux + torch.sum(gateVx, dim=2) / (1e-20 + torch.sum(edge_gate, dim=2))
        elif self.aggregation == "sum":
            x_new = Ux + torch.sum(gateVx, dim=2)
        return x_new


class EdgeFeatures(nn.Module):
    def __init__(self, hidden_dim):
        super(EdgeFeatures, self).__init__()
        self.U = nn.Linear(hidden_dim, hidden_dim, True)
        self.V = nn.Linear(hidden_dim, hidden_dim, True)

    def forward(self, x, e):
        Ue = self.U(e)
        Vx = self.V(x)
        Wx = Vx.unsqueeze(1)
        Vx = Vx.unsqueeze(2)
        e_new = Ue + Vx + Wx
        return e_new


class ResidualGatedGCNLayer(nn.Module):

    def __init__(self, hidden_dim, aggregation="sum"):
        super(ResidualGatedGCNLayer, self).__init__()
        self.node_feat = NodeFeatures(hidden_dim, aggregation)
        self.edge_feat = EdgeFeatures(hidden_dim)
        self.bn_node = BatchNormNode(hidden_dim)
        self.bn_edge = BatchNormEdge(hidden_dim)

    def forward(self, x, e):
        e_in = e
        x_in = x
        e_tmp = self.edge_feat(x_in, e_in)
        edge_gate = F.sigmoid(e_tmp)
        x_tmp = self.node_feat(x_in, edge_gate)
        e_tmp = self.bn_edge(e_tmp)
        x_tmp = self.bn_node(x_tmp)
        e = F.relu(e_tmp)
        x = F.relu(x_tmp)
        x_new = x_in + x
        e_new = e_in + e
        return x_new, e_new


class MLP(nn.Module):

    def __init__(self, hidden_dim, output_dim, L=2):
        super(MLP, self).__init__()
        self.L = L
        U = []
        for layer in range(self.L - 1):
            U.append(nn.Linear(hidden_dim, hidden_dim, True))
        self.U = nn.ModuleList(U)
        self.V = nn.Linear(hidden_dim, output_dim, True)

    def forward(self, x):

        Ux = x
        for U_i in self.U:
            Ux = U_i(Ux)
            Ux = F.relu(Ux)
        y = self.V(Ux)
        return y
