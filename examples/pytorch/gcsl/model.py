import torch as th
import torch.nn as nn
import torch.nn.functional as F

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

    def get_khop_neighbors(self, graph):
        adj = graph.adj().to_dense().fill_diagonal_(1).float()
        self.khops = th.zeros((self.max_khop,len(adj),len(adj)))
        kadj = th.zeros((self.max_khop,len(adj),len(adj)))
        kadj[0] = adj
        self.khops[0] = adj.fill_diagonal_(0)
        for i in range(1,self.max_khop): 
            kadj[i] = th.mm(kadj[i-1],adj)
            kadj[i][kadj[i]!=0] = 1
            self.khops[i] = kadj[i]-kadj[i-1]
        # for i in range(self.max_khop):
        #     self.khops[i] = F.normalize(self.khops[i],1,1)

    def sim(self, z1, z2):
        # normalize embeddings across feature dimension
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)

        s = th.mm(z1, z2.t())
        return s

    def get_loss(self, z):

        f = lambda x: th.exp(x / self.temp)
        z_sim = f(self.sim(z, z))
        # return z_sim.sum()
    
        khop_sim = []
        for i in range(self.max_khop): 
            khop_sim.append(th.mul(z_sim,self.khops[i]).sum(dim=0))
        # return khop_sim[0].sum()+khop_sim[1].sum()+khop_sim[2].sum()
    
        loss_sum = 0
        if self.ranking_type=="pair":
            # Gated Pair-Wise
            for i in range(self.max_khop):
                for j in range(i+1,self.max_khop):
                    idx_nonzero = th.logical_and(khop_sim[i]!=0,khop_sim[j]!=0)
                    loss = th.div((khop_sim[i][idx_nonzero]),(khop_sim[i][idx_nonzero]+khop_sim[j][idx_nonzero]))
                    loss_sum += -th.log(loss[loss<self.alpha]).sum()
        elif self.ranking_type=="list":
            # Gated List-Wise
            for i in range(self.max_khop):
                idx_nonzero = (khop_sim[i]!=0)
                for j in range(i+1,self.max_khop): idx_nonzero = th.logical_and(idx_nonzero,khop_sim[j]!=0)
                neg_term = khop_sim[i][idx_nonzero]
                for j in range(i+1,self.max_khop): neg_term += khop_sim[j][idx_nonzero]
                loss = th.div(khop_sim[i][idx_nonzero],neg_term)
                loss_sum += -th.log(loss[loss<self.alpha]).sum()

                # loss = -th.log(th.div((khop_sim[i]),(khop_sim[i]+khop_sim[j])))
                # for q in range(len(loss)):
                #     if not th.isnan(loss[q]):
                #         loss_min = min(loss[q],self.alpha)
                #         loss_sum += loss_min
                # loss = min(loss[th.logical_not(th.isnan(loss))],self.alpha)
                # # loss[loss>self.alpha] = self.alpha
                # loss_sum += loss.sum()

                # for q in range(len(z_sim)):
                #     loss = -th.log(th.div((khop_sim[i][q]),(khop_sim[i][q]+khop_sim[j][q])))
                #     if not th.isnan(loss):
                #         loss_min = min(loss,self.alpha)
                #         loss_sum += loss_min
                
                # loss = -th.log(th.div((khop_sim[i]),(khop_sim[i]+khop_sim[j])))
                # for q in range(len(loss)):
                #     if (not th.isnan(loss[q])) and loss[q]<1 and loss[q]>1e-2:
                #         loss_sum+=loss[q]
                        # break
                # loss = th.nan_to_num(loss, nan=0)
                # loss[loss>self.alpha] = 0
                # loss_sum += loss.sum()

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
