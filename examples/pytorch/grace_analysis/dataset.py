from dgl.data import CiteseerGraphDataset, CoraGraphDataset, PubmedGraphDataset, YelpDataset, WikiCSDataset, PPIDataset, RedditDataset


def load(name):
    if name == "cora":
        dataset = CoraGraphDataset()
    elif name == "citeseer":
        dataset = CiteseerGraphDataset()
    elif name == "pubmed":
        dataset = PubmedGraphDataset()
    # elif name == "yelp":
    #     dataset = YelpDataset()
    elif name == "wikics":
        dataset = WikiCSDataset()
    elif name == "ppi":
        dataset = PPIDataset()
    elif name == "reddit":
        dataset = RedditDataset()

    graph = dataset[0]

    train_mask = graph.ndata.pop("train_mask")
    test_mask = graph.ndata.pop("test_mask")

    feat = graph.ndata.pop("feat")
    labels = graph.ndata.pop("label")

    return graph, feat, labels, train_mask, test_mask
