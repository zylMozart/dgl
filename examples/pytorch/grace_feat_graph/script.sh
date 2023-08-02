# for dataset in cora citeseer pubmed
# do
# for fuse_rate in 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
# do
# for aug_type in fuse_knn fuse_cos
# do
# python examples/pytorch/grace_feat_graph/main.py --dataname $dataset --aug_type $aug_type --fuse_rate $fuse_rate
# done
# done
# done

for dataset in cora citeseer pubmed
do
for aug_type in cos_combine
do
for cos_topk in 0.01 0.05 0.1 0.15 0.2 0.3 0.5 0.6 0.7 0.8
do
python examples/pytorch/grace_feat_graph/main.py --dataname $dataset --aug_type $aug_type --cos_topk $cos_topk
done
done
done

for dataset in cora citeseer pubmed
do
for aug_type in knn_combine
do
for knn_clusters in 1 2 3 5 10 15 20 50 100 500 1000
do
python examples/pytorch/grace_feat_graph/main.py --dataname $dataset --aug_type $aug_type --knn_clusters $knn_clusters
done
done
done
