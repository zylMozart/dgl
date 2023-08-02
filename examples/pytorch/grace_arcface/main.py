import argparse
import warnings

import numpy as np
import torch as th
import torch.nn as nn
from aug import aug,knn_aug,no_aug,cos_sim_aug
from dataset import load
from eval import label_classification
from model import Grace
from tqdm import tqdm
import datetime
import pandas as pd
warnings.filterwarnings("ignore")


def count_parameters(model):
    return sum(
        [np.prod(p.size()) for p in model.parameters() if p.requires_grad]
    )


parser = argparse.ArgumentParser()

parser.add_argument("--margin", type=float, default=0.1, help="Margin for ArcFace.")

parser.add_argument("--dataname", type=str, default="cora")
parser.add_argument("--gpu", type=int, default=0)
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
    seed_everything(0)
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
    model = Grace(in_dim, hid_dim, out_dim, num_layers, act_fn, temp, args.margin)
    model = model.to(args.device)
    print(f"# params: {count_parameters(model)}")

    optimizer = th.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    # Step 4: Training =======================================================================
    bar = tqdm(range(epochs))
    for epoch in bar:
        model.train()
        optimizer.zero_grad()
        graph1, feat1 = aug(graph, feat, drop_feature_rate_1, drop_edge_rate_1)
        graph2, feat2 = aug(graph, feat, drop_feature_rate_2, drop_edge_rate_2)

        graph1 = graph1.to(args.device)
        graph2 = graph2.to(args.device)

        feat1 = feat1.to(args.device)
        feat2 = feat2.to(args.device)

        loss = model(graph1, graph2, feat1, feat2)
        loss.backward()
        optimizer.step()

        bar.set_description(f"Dataset: {args.dataname} m:{args.margin}")
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

    # Record result with parameters
    e = datetime.datetime.now()
    df = pd.DataFrame(columns=['time','method','dataset','margin','temp','hid_dim','out_dim','F1Mi_mean','F1Mi_std','F1Ma_mean','F1Ma_std'])
    df = df.append({'time':"%s/%s/%s %s:%s:%s"%(e.day,e.month,e.year,e.hour,e.minute,e.second),
                'method':'grace_arcface',
               'dataset':args.dataname,
               'margin':args.margin,
               'temp':args.temp,
               'hid_dim':args.hid_dim,
               'out_dim':args.out_dim,
               'F1Mi_mean':test_result['F1Mi']['mean'],
               'F1Mi_std':test_result['F1Mi']['std'],
               'F1Ma_mean':test_result['F1Ma']['mean'],
               'F1Ma_std':test_result['F1Ma']['std']},ignore_index=True)
    df.to_csv('examples/pytorch/grace_arcface/result.csv',mode='a',header=False) 