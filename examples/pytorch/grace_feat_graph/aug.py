# Data augmentation on graphs via edge dropping and feature masking

import dgl
import numpy as np
import torch as th
import torch.nn.functional as F

def cos_combine(graph, feat, cos_topk):
    feat_graph, _ = cos_sim_aug(feat, cos_topk)
    osrc, odst = graph.edges()
    fsrc, fdst = feat_graph.edges()
    new_src = th.concat([osrc, fsrc])
    new_dst = th.concat([odst, fdst])
    new_graph = dgl.graph((new_src, new_dst))
    return new_graph

def knn_combine(graph, feat, knn_clusters):
    feat_graph, _ = knn_aug(feat, knn_clusters)
    osrc, odst = graph.edges()
    fsrc, fdst = feat_graph.edges()
    new_src = th.concat([osrc, fsrc])
    new_dst = th.concat([odst, fdst])
    new_graph = dgl.graph((new_src, new_dst))
    return new_graph

class Fuse:
    def __init__(self, graph, feat, args):
        self.fuse_rate = args.fuse_rate
        self.ori_graph = graph
        if 'knn' in args.aug_type:
            self.feat_graph, _ = knn_aug(feat, args.knn_clusters)
        elif 'cos' in args.aug_type:
            self.feat_graph, _ = cos_sim_aug(feat, args.cos_topk)

    def aug_fuse(self, graph, x, feat_drop_rate, edge_mask_rate):
        osrc, odst = self.ori_graph.edges()
        fsrc, fdst = self.feat_graph.edges()
        oidx = np.random.choice(len(osrc),int(len(osrc)*self.fuse_rate),replace=False)
        kidx = np.random.choice(len(fsrc),int(len(fsrc)*self.fuse_rate),replace=False)
        new_osrc = th.concat([osrc, fsrc[kidx]])
        new_odst = th.concat([odst, fdst[kidx]])
        new_fsrc = th.concat([fsrc, osrc[oidx]])
        new_fdst = th.concat([fdst, odst[oidx]])
        graph1 = dgl.graph((new_osrc, new_odst))
        graph2 = dgl.graph((new_fsrc, new_fdst))
        feat1 = drop_feature(x, feat_drop_rate)
        feat2 = drop_feature(x, feat_drop_rate)
        return graph1, feat1, graph2, feat2

def knn_aug(x, knn_clusters):
    feat = x
    ng = dgl.knn_graph(x, knn_clusters)
    ng = ng.add_self_loop()
    return ng, feat


def cos_sim_aug(x, cos_topk):
    n_node = x.shape[0]
    cos_topk = int(cos_topk*n_node)
    feat = x
    
    mul = th.mm(x,x.t())
    norm = th.norm(x,p=2,dim=1).reshape(-1,1)
    sim = th.div(mul,th.matmul(norm,norm.t()))
    sim.fill_diagonal_(0)
    sim_flat = sim.flatten()
    indices = th.topk(sim_flat, cos_topk).indices
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
