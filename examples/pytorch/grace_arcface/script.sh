# for dataset in citeseer pubmed
# do
# # for margin in 0 0.001 0.003 0.005 0.007 0.009 
# for margin in 0.001 0.005 0.01 0.05 0.1 0.5 1 1.5 2 2.5 3
# do
# python examples/pytorch/grace_arcface/main.py --margin $margin --dataname $dataset
# done
# done

for dataset in citeseer pubmed
do
# for margin in 0 0.001 0.003 0.005 0.007 0.009 
for margin in 0
do
python examples/pytorch/grace_arcface/main.py --margin $margin --dataname $dataset
done
done