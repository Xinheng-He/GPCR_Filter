source ~/miniconda3/bin/activate drugclip

# python train.py \
#     --dataset_tag split-random\
#     --cuda_use cuda:7 \
#     --data_dir /datapool/data2/home/majianzhu/xinheng/xiangzhen/DTI2/DTI/data \
#     --dict_target /datapool/data2/home/majianzhu/xinheng/xiangzhen/DTI2/DTI/data/dict_target.pkl \
#     --dict_ligand /datapool/data2/home/majianzhu/xinheng/xiangzhen/mk-dict/unimol2/dict_ligand.pkl \
#     --num_epochs 100 \
#     --fetch_pretrained_target \
#     --lr 1e-5 \
#     --batchsize 32

# python train.py \
#     --dataset_tag split-target-intra\
#     --cuda_use cuda:7 \
#     --data_dir /datapool/data2/home/majianzhu/xinheng/xiangzhen/DTI2/DTI/data \
#     --dict_target /datapool/data2/home/majianzhu/xinheng/xiangzhen/DTI2/DTI/data/dict_target.pkl \
#     --dict_ligand /datapool/data2/home/majianzhu/xinheng/xiangzhen/mk-dict/unimol2/dict_ligand.pkl \
#     --num_epochs 100 \
#     --fetch_pretrained_target \
#     --lr 1e-5 \
#     --batchsize 32

# python train.py \
#     --dataset_tag split-target-inter\
#     --cuda_use cuda:7 \
#     --data_dir /datapool/data2/home/majianzhu/xinheng/xiangzhen/DTI2/DTI/data \
#     --dict_target /datapool/data2/home/majianzhu/xinheng/xiangzhen/DTI2/DTI/data/dict_target.pkl \
#     --dict_ligand /datapool/data2/home/majianzhu/xinheng/xiangzhen/mk-dict/unimol2/dict_ligand.pkl \
#     --num_epochs 20 \
#     --fetch_pretrained_target \
#     --lr 1e-5 \
#     --batchsize 32

# python train.py \
#     --dataset_tag split-target-inter-pure\
#     --cuda_use cuda:7 \
#     --data_dir /datapool/data2/home/majianzhu/xinheng/xiangzhen/DTI2/DTI/data \
#     --dict_target /datapool/data2/home/majianzhu/xinheng/xiangzhen/DTI2/DTI/data/dict_target.pkl \
#     --dict_ligand /datapool/data2/home/majianzhu/xinheng/xiangzhen/mk-dict/unimol2/dict_ligand.pkl \
#     --num_epochs 20 \
#     --fetch_pretrained_target \
#     --lr 1e-5 \
#     --batchsize 32


# python train.py \
#     --dataset_tag split-target-inter-pure4\
#     --cuda_use cuda:7 \
#     --data_dir /datapool/data2/home/majianzhu/xinheng/xiangzhen/DTI2/DTI/data \
#     --dict_target /datapool/data2/home/majianzhu/xinheng/xiangzhen/DTI2/DTI/data/dict_target.pkl \
#     --dict_ligand /datapool/data2/home/majianzhu/xinheng/xiangzhen/mk-dict/unimol2/dict_ligand.pkl \
#     --num_epochs 20 \
#     --fetch_pretrained_target \
#     --lr 1e-5 \
#     --batchsize 32


# python train.py \
#     --dataset_tag split-target-inter-pure5\
#     --cuda_use cuda:7 \
#     --data_dir /datapool/data2/home/majianzhu/xinheng/xiangzhen/DTI2/DTI/data \
#     --dict_target /datapool/data2/home/majianzhu/xinheng/xiangzhen/DTI2/DTI/data/dict_target.pkl \
#     --dict_ligand /datapool/data2/home/majianzhu/xinheng/xiangzhen/mk-dict/unimol2/dict_ligand.pkl \
#     --num_epochs 20 \
#     --fetch_pretrained_target \
#     --lr 1e-5 \
#     --batchsize 32

# python train.py \
#     --dataset_tag split-target-inter-pure6\
#     --cuda_use cuda:7 \
#     --data_dir /datapool/data2/home/majianzhu/xinheng/xiangzhen/DTI2/DTI/data \
#     --dict_target /datapool/data2/home/majianzhu/xinheng/xiangzhen/DTI2/DTI/data/dict_target.pkl \
#     --dict_ligand /datapool/data2/home/majianzhu/xinheng/xiangzhen/mk-dict/unimol2/dict_ligand.pkl \
#     --num_epochs 20 \
#     --fetch_pretrained_target \
#     --lr 1e-5 \
#     --batchsize 32

# python train.py \
#     --dataset_tag split-target-inter-pure7\
#     --cuda_use cuda:7 \
#     --data_dir /datapool/data2/home/majianzhu/xinheng/xiangzhen/DTI2/DTI/data \
#     --dict_target /datapool/data2/home/majianzhu/xinheng/xiangzhen/DTI2/DTI/data/dict_target.pkl \
#     --dict_ligand /datapool/data2/home/majianzhu/xinheng/xiangzhen/mk-dict/unimol2/dict_ligand.pkl \
#     --num_epochs 20 \
#     --fetch_pretrained_target \
#     --lr 1e-5 \
#     --batchsize 32



python train.py \
    --dataset_tag split-test\
    --cuda_use cuda:7 \
    --data_dir /datapool/data2/home/majianzhu/xinheng/xiangzhen/DTI2/DTI/data \
    --dict_target /datapool/data2/home/majianzhu/xinheng/xiangzhen/DTI2/DTI/data/dict_target.pkl \
    --dict_ligand /datapool/data2/home/majianzhu/xinheng/xiangzhen/mk-dict/unimol2/dict_ligand.pkl \
    --num_epochs 3 \
    --fetch_pretrained_target \
    --batchsize 2 \
    --lr 1e-5 \
    
