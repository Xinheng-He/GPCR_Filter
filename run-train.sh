# split-random
python train.py \
    --dataset_tag split-random\
    --cuda_use cuda:7 \
    --data_dir data \
    --dict_target data/dict_target.pkl \
    --num_epochs 100 \
    --fetch_pretrained_target \
    --lr 1e-5 \
    --batchsize 32

# # split-intra
# python train.py \
#     --dataset_tag split-target-intra\
#     --cuda_use cuda:7 \
#     --data_dir data \
#     --dict_target data/dict_target.pkl \
#     --num_epochs 100 \
#     --fetch_pretrained_target \
#     --lr 1e-5 \
#     --batchsize 32

# # split-inter
# python train.py \
#     --dataset_tag split-target-inter-pure6\
#     --cuda_use cuda:7 \
#     --data_dir data \
#     --dict_target data/dict_target.pkl \
#     --num_epochs 20 \
#     --fetch_pretrained_target \
#     --hid_dim 128 \
#     --dropout 0 \
#     --lr 1e-5 \
#     --batchsize 32
    
    
