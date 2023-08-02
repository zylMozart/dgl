import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from dgl.nn import GraphConv


# Multi-layer Graph Convolutional Networks
class GCN(nn.Module):
    def __init__(self, in_dim, out_dim, act_fn, num_layers=2):
        super(GCN, self).__init__()

        assert num_layers >= 2
        self.num_layers = num_layers
        self.convs = nn.ModuleList()

        self.convs.append(GraphConv(in_dim, out_dim * 2))
        for _ in range(self.num_layers - 2):
            self.convs.append(GraphConv(out_dim * 2, out_dim * 2))

        self.convs.append(GraphConv(out_dim * 2, out_dim))
        self.act_fn = act_fn

    def forward(self, graph, feat):
        for i in range(self.num_layers):
            feat = self.act_fn(self.convs[i](graph, feat))

        return feat


# Multi-layer(2-layer) Perceptron
class MLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_dim, out_dim)
        self.fc2 = nn.Linear(out_dim, in_dim)

    def forward(self, x):
        z = F.elu(self.fc1(x))
        return self.fc2(z)

def acos_safe(x, eps=1e-4):
    sign = th.sign(x)
    slope = np.arccos(1-eps) / eps
    return th.where(abs(x) <= 1-eps,
                    th.acos(x),
                    th.acos(sign * (1 - eps)) - slope*sign*(abs(x) - 1 + eps))

class Grace(nn.Module):
    r"""
        GRACE model
    Parameters
    -----------
    in_dim: int
        Input feature size.
    hid_dim: int
        Hidden feature size.
    out_dim: int
        Output feature size.
    num_layers: int
        Number of the GNN encoder layers.
    act_fn: nn.Module
        Activation function.
    temp: float
        Temperature constant.
    """

    def __init__(self, in_dim, hid_dim, out_dim, num_layers, act_fn, temp, max_khop, num_nodes, alpha, ranking_type):
        super(Grace, self).__init__()
        self.encoder = GCN(in_dim, hid_dim, act_fn, num_layers)
        self.temp = temp
        self.proj = MLP(hid_dim, out_dim)
        self.max_khop = max_khop
        self.khops = None
        self.alpha = alpha
        self.ranking_type = ranking_type
        self.eps = 1e-5

    def get_khop_neighbors(self, graph):
        adj = graph.adj().to_dense().fill_diagonal_(1).float()
        self.khops = th.zeros((self.max_khop,len(adj),len(adj)))
        kadj = th.zeros((self.max_khop,len(adj),len(adj)))
        kadj[0] = adj
        for i in range(1,self.max_khop): 
            kadj[i] = th.mm(kadj[i-1],adj)
            kadj[i][kadj[i]!=0] = 1
            self.khops[i] = kadj[i]-kadj[i-1]
        for i in range(self.max_khop):
            self.khops[i] = F.normalize(self.khops[i],1,1)
            self.khops[i] = self.khops[i].fill_diagonal_(1)

    def sim(self, z1, z2):
        # normalize embeddings across feature dimension
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        cos = th.mm(z1, z2.t())-self.eps
        return cos

    def sim_arcface(self, z1, z2, m=0.1):
        # normalize embeddings across feature dimension
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        cos = th.mm(z1, z2.t())-self.eps
        sin = th.sqrt(1-th.pow(cos,2))
        return cos*math.cos(m)-sin*math.sin(m)
        # sim[sim>1] = 1 - 1e-6
        # sim[sim<-1] = -1 + 1e-6
        theta = th.acos(sim)+m
        sim = th.cos(theta)
        # sim[th.isnan(sim)] = 1
        return sim

    def get_loss(self, z):

        f = lambda x: th.exp(x / self.temp)
        z_sim = f(self.sim(z, z))
        z_sim_arcface = f(self.sim_arcface(z, z))
        # return z_sim_arcface.sum()
        khop_sim_pos = []
        khop_sim_neg = []
        for i in range(self.max_khop): 
            khop_sim_pos.append(th.mul(z_sim_arcface,self.khops[i]).sum(dim=0))
            khop_sim_neg.append(th.mul(z_sim,self.khops[i]).sum(dim=0))

        loss_sum = 0
        if self.ranking_type=="pair":
            # Gated Pair-Wise
            for i in range(self.max_khop):
                for j in range(i+1,self.max_khop):
                    loss = th.div((khop_sim_pos[i]),(khop_sim_pos[i]+khop_sim_neg[j]))
                    loss_sum += -th.log(loss[loss<self.alpha]).sum()

        return loss_sum

    
    def get_embedding(self, graph, feat):
        # get embeddings from the model for evaluation
        h = self.encoder(graph, feat)

        return h.detach()

    def forward(self, graph, feat):
        # encoding
        h = self.encoder(graph, feat)

        # projection
        z = self.proj(h)

        # get loss
        l = self.get_loss(z)

        return l
