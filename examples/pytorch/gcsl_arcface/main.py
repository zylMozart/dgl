import argparse
import warnings

import numpy as np
import torch as th
import torch.nn as nn
from aug import aug
from dataset import load
from eval import label_classification
from model import Grace
from tqdm import tqdm
import pandas as pd
import datetime

warnings.filterwarnings("ignore")


def count_parameters(model):
    return sum(
        [np.prod(p.size()) for p in model.parameters() if p.requires_grad]
    )


parser = argparse.ArgumentParser()

parser.add_argument("--max_khop", type=int, default=2, help="Maximum k-hop neighbors for ranking.")
parser.add_argument("--alpha", type=float, default=0.1, help="Hyper-parameter for filtering large values.")
parser.add_argument("--ranking_type", type=str, default="pair", help="Gated Pair-Wise or List-Wise ranking type.")

parser.add_argument("--dataname", type=str, default="cora")
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--split", type=str, default="public")

parser.add_argument("--epochs", type=int, default=500, help="Number of training periods.")
parser.add_argument("--lr", type=float, default=0.001, help="Learning rate.")
parser.add_argument("--wd", type=float, default=1e-5, help="Weight decay.")
parser.add_argument("--temp", type=float, default=1.0, help="Temperature.")

parser.add_argument("--act_fn", type=str, default="relu")

parser.add_argument("--hid_dim", type=int, default=256, help="Hidden layer dim.")
parser.add_argument("--out_dim", type=int, default=256, help="Output layer dim.")

parser.add_argument("--num_layers", type=int, default=2, help="Number of GNN layers.")
parser.add_argument("--der1",type=float,default=0.2,help="Drop edge ratio of the 1st augmentation.",)
parser.add_argument("--der2",type=float,default=0.2,help="Drop edge ratio of the 2nd augmentation.",)
parser.add_argument("--dfr1",type=float,default=0.2,help="Drop feature ratio of the 1st augmentation.",)
parser.add_argument("--dfr2",type=float,default=0.2,help="Drop feature ratio of the 2nd augmentation.",)

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

    max_khop = args.max_khop
    alpha = args.alpha
    ranking_type = args.ranking_type
    num_layers = args.num_layers
    act_fn = ({"relu": nn.ReLU(), "prelu": nn.PReLU()})[args.act_fn]

    temp = args.temp
    epochs = args.epochs
    wd = args.wd

    # Step 2: Prepare data =================================================================== #
    graph, feat, labels, train_mask, test_mask = load(args.dataname)
    num_nodes = feat.shape[0]
    in_dim = feat.shape[1]

    # Step 3: Create model =================================================================== #
    model = Grace(in_dim, hid_dim, out_dim, num_layers, act_fn, temp, max_khop, num_nodes, alpha, ranking_type)
    model.get_khop_neighbors(graph)
    model = model.to(args.device)
    model.khops = model.khops.to(args.device)
    print(f"# params: {count_parameters(model)}")

    optimizer = th.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    # Step 4: Training =======================================================================
    bar = tqdm(range(epochs))
    for epoch in bar:
        model.train()
        optimizer.zero_grad()

        graph = graph.to(args.device)
        feat = feat.to(args.device)

        loss = model(graph, feat)

        loss.backward()
        optimizer.step()

        bar.set_description(f"k:{max_khop} alpha:{alpha}")
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
    df = pd.DataFrame(columns=['time','method','dataset','max_khop','alpha','ranking_type','temp','hid_dim','out_dim'])
    df = df.append({'time':"%s/%s/%s %s:%s:%s"%(e.day,e.month,e.year,e.hour,e.minute,e.second),
                'method':'GCSL',
               'dataset':args.dataname,
               'max_khop':args.max_khop,
               'alpha':args.alpha,
               'ranking_type':args.ranking_type,
               'temp':args.temp,
               'hid_dim':args.hid_dim,
               'out_dim':args.out_dim,
               'F1Mi_mean':test_result['F1Mi']['mean'],
               'F1Mi_std':test_result['F1Mi']['std'],
               'F1Ma_mean':test_result['F1Ma']['mean'],
               'F1Ma_std':test_result['F1Ma']['std']},ignore_index=True)
    df.to_csv('examples/pytorch/gcsl_arcface/GCSL.csv',mode='a',header=True) 