for dataset in cora citeseer pubmed
do
for alpha in 1.0 0.5 0.1 0.005 0.001 0.0005 0.0001
do
for temp in 1 0.5 0.2 0.1 0.05 0.02 0.01 0.005 0.002 0.001
do
for dim in 64 128 256 512
do
for max_khop in 2 3 5 10 
do
python examples/pytorch/gcsl/main.py --dataname $dataset --alpha $alpha --temp $temp --hid_dim $dim --out_dim $dim --max_khop $max_khop
done
done
done
done
done