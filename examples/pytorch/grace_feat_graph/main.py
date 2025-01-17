import argparse
import warnings

import numpy as np
import torch as th
import torch.nn as nn
from aug import aug,knn_aug,no_aug,cos_sim_aug,Fuse,cos_combine,knn_combine
from dataset import load
from eval import label_classification
from model import Grace
from tqdm import tqdm
import pandas as pd
import os
from datetime import datetime

warnings.filterwarnings("ignore")


def count_parameters(model):
    return sum(
        [np.prod(p.size()) for p in model.parameters() if p.requires_grad]
    )


parser = argparse.ArgumentParser()

parser.add_argument("--aug_type", type=str, default="knn_combine",help="knn, cos, fuse")
parser.add_argument("--knn_clusters", type=int, default=5)
parser.add_argument("--cos_topk", type=float, default=0.1)
parser.add_argument("--fuse_rate", type=float, default=0.6)
parser.add_argument("--seed", type=int, default=0)

parser.add_argument("--dataname", type=str, default="cora")
parser.add_argument("--gpu", type=int, default=2)
parser.add_argument("--split", type=str, default="public")

parser.add_argument(
    "--epochs", type=int, default=500, help="Number of training periods."
)
parser.add_argument("--lr", type=float, default=0.001, help="Learning rate.")
parser.add_argument("--wd", type=float, default=1e-5, help="Weight decay.")
parser.add_argument("--temp", type=float, default=1.0, help="Temperature.")

parser.add_argument("--act_fn", type=str, default="relu")

parser.add_argument(
    "--hid_dim", type=int, default=256, help="Hidden layer dim."
)
parser.add_argument(
    "--out_dim", type=int, default=256, help="Output layer dim."
)

parser.add_argument(
    "--num_layers", type=int, default=2, help="Number of GNN layers."
)
parser.add_argument(
    "--der1",
    type=float,
    default=0.2,
    help="Drop edge ratio of the 1st augmentation.",
)
parser.add_argument(
    "--der2",
    type=float,
    default=0.2,
    help="Drop edge ratio of the 2nd augmentation.",
)
parser.add_argument(
    "--dfr1",
    type=float,
    default=0.2,
    help="Drop feature ratio of the 1st augmentation.",
)
parser.add_argument(
    "--dfr2",
    type=float,
    default=0.2,
    help="Drop feature ratio of the 2nd augmentation.",
)

args = parser.parse_args()

if args.gpu != -1 and th.cuda.is_available():
    args.device = "cuda:{}".format(args.gpu)
else:
    args.device = "cpu"

def seed_everything(seed):
    import random,os,torch
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

if __name__ == "__main__":
    # Step 1: Load hyperparameters =================================================================== #
    seed_everything(args.seed)
    lr = args.lr
    hid_dim = args.hid_dim
    out_dim = args.out_dim

    num_layers = args.num_layers
    act_fn = ({"relu": nn.ReLU(), "prelu": nn.PReLU()})[args.act_fn]

    drop_edge_rate_1 = args.der1
    drop_edge_rate_2 = args.der2
    drop_feature_rate_1 = args.dfr1
    drop_feature_rate_2 = args.dfr2

    temp = args.temp
    epochs = args.epochs
    wd = args.wd

    # Step 2: Prepare data =================================================================== #
    graph, feat, labels, train_mask, test_mask = load(args.dataname)
    in_dim = feat.shape[1]

    # Step 3: Create model =================================================================== #
    model = Grace(in_dim, hid_dim, out_dim, num_layers, act_fn, temp)
    model = model.to(args.device)
    print(f"# params: {count_parameters(model)}")

    optimizer = th.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    # Step 4: Training =======================================================================
    if args.aug_type=='knn':
        graph1, feat1 = knn_aug(feat, args.knn_clusters)
    elif args.aug_type=='cos':
        graph1, feat1 = cos_sim_aug(feat, args.cos_topk)
    elif args.aug_type=='':
        graph1, feat1 = aug(graph, feat, drop_feature_rate_1, drop_edge_rate_1)
    elif 'fuse' in args.aug_type:
        fuse = Fuse(graph, feat, args)
    elif args.aug_type=='cos_guided':
        feat_graph, _ = cos_sim_aug(feat, args.cos_topk)
        negative_mask = 1 - feat_graph.adj().to_dense()
        negative_mask = negative_mask.to(args.device)
    elif args.aug_type=='knn_guided':
        feat_graph, _ = knn_aug(feat, args.knn_clusters)
        negative_mask = 1 - feat_graph.adj().to_dense()
        negative_mask = negative_mask.to(args.device)
    elif args.aug_type=='cos_combine':
        graph = cos_combine(graph, feat, args.cos_topk)
    elif args.aug_type=='knn_combine':
        graph = knn_combine(graph, feat, args.knn_clusters)
    bar = tqdm(range(epochs))
    for epoch in bar:
        model.train()
        optimizer.zero_grad()
        graph1, feat1 = aug(graph, feat, drop_feature_rate_1, drop_edge_rate_1)
        graph2, feat2 = aug(graph, feat, drop_feature_rate_2, drop_edge_rate_2)
        # if 'fuse' in args.aug_type:
        #     graph1, feat1, graph2, feat2 = fuse.aug_fuse(graph, feat, drop_feature_rate_1, drop_edge_rate_1)

        graph1 = graph1.to(args.device)
        graph2 = graph2.to(args.device)

        feat1 = feat1.to(args.device)
        feat2 = feat2.to(args.device)

        loss = model(graph1, graph2, feat1, feat2)
        # loss = model(graph1, graph2, feat1, feat2, negative_mask)
        loss.backward()
        optimizer.step()

        bar.set_description(f"Dataset: {args.dataname}")
        bar.set_postfix(loss=loss.item())
    
    # Step 5: Linear evaluation ============================================================== #
    print("=== Final ===")

    graph = graph.add_self_loop()
    graph = graph.to(args.device)
    feat = feat.to(args.device)
    embeds = model.get_embedding(graph, feat)

    """Evaluation Embeddings  """
    test_result = label_classification(
        embeds, labels, train_mask, test_mask, split=args.split
    )

    result={}
    for k,v in test_result.items():
        for k2,v2 in v.items():
            result[k+k2]=v2
    result.update(vars(args))
    result['datetime'] = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
    df = pd.DataFrame(columns=result.keys())
    df = df.append(result, ignore_index=True)
    save_path = 'zyl_implement/result/grace_graph_combine.csv'
    if os.path.exists(save_path):
        df.to_csv(save_path,mode='a',header=False) 
    else:
        df.to_csv(save_path,mode='w',header=True) 