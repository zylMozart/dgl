# Data augmentation on graphs via edge dropping and feature masking

import dgl
import numpy as np
import torch as th
import torch.nn.functional as F

def knn_aug(graph, x, feat_drop_rate, edge_mask_rate):
    feat = x
    ng = dgl.knn_graph(x, 5)
    ng = ng.add_self_loop()
    return ng, feat


def cos_sim_aug(graph, x, feat_drop_rate, edge_mask_rate):
    n_node = graph.num_nodes()
    feat = x
    
    mul = th.mm(x,x.t())
    norm = th.norm(x,p=2,dim=1).reshape(-1,1)
    sim = th.div(mul,th.matmul(norm,norm.t()))
    sim.fill_diagonal_(0)
    sim_flat = sim.flatten()
    indices = th.topk(sim_flat, 10000).indices
    adj = th.zeros(sim_flat.shape)
    adj[indices] = 1
    adj = adj.reshape(sim.shape)
    adj = adj.to_sparse()
    nsrc = adj.indices()[0]
    ndst = adj.indices()[1]
    ng = dgl.graph((nsrc, ndst), num_nodes=n_node)
    ng = ng.add_self_loop()
    return ng, feat


def no_aug(graph, x, feat_drop_rate, edge_mask_rate):
    return graph, x


def aug(graph, x, feat_drop_rate, edge_mask_rate):
    n_node = graph.num_nodes()

    edge_mask = mask_edge(graph, edge_mask_rate)
    feat = drop_feature(x, feat_drop_rate)

    src = graph.edges()[0]
    dst = graph.edges()[1]

    nsrc = src[edge_mask]
    ndst = dst[edge_mask]

    ng = dgl.graph((nsrc, ndst), num_nodes=n_node)
    ng = ng.add_self_loop()

    return ng, feat


def drop_feature(x, drop_prob):
    drop_mask = (
        th.empty((x.size(1),), dtype=th.float32, device=x.device).uniform_(0, 1)
        < drop_prob
    )
    x = x.clone()
    x[:, drop_mask] = 0

    return x


def mask_edge(graph, mask_prob):
    E = graph.num_edges()

    mask_rates = th.FloatTensor(np.ones(E) * mask_prob)
    masks = th.bernoulli(1 - mask_rates)
    mask_idx = masks.nonzero().squeeze(1)
    return mask_idx
