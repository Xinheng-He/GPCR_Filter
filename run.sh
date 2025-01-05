python train.py \
    --dataset_tag split-random\
    --cuda_use cuda:7 \
    --data_dir /datapool/data2/home/majianzhu/xinheng/xiangzhen/DTI2/DTI/data \
    --dict_target /datapool/data2/home/majianzhu/xinheng/xiangzhen/DTI2/DTI/data/dict_target.pkl \
    --dict_ligand /datapool/data2/home/majianzhu/xinheng/xiangzhen/mk-dict/unimol2/dict_ligand.pkl \

python train.py \
    --dataset_tag split-target-intra\
    --cuda_use cuda:7 \
    --data_dir /datapool/data2/home/majianzhu/xinheng/xiangzhen/DTI2/DTI/data \
    --dict_target /datapool/data2/home/majianzhu/xinheng/xiangzhen/DTI2/DTI/data/dict_target.pkl \
    --dict_ligand /datapool/data2/home/majianzhu/xinheng/xiangzhen/mk-dict/unimol2/dict_ligand.pkl \

python train.py \
    --dataset_tag split-target-inter\
    --cuda_use cuda:7 \
    --data_dir /datapool/data2/home/majianzhu/xinheng/xiangzhen/DTI2/DTI/data \
    --dict_target /datapool/data2/home/majianzhu/xinheng/xiangzhen/DTI2/DTI/data/dict_target.pkl \
    --dict_ligand /datapool/data2/home/majianzhu/xinheng/xiangzhen/mk-dict/unimol2/dict_ligand.pkl \
    --num_epochs 30

# python train.py \
#     --dataset_tag split-test\
#     --cuda_use cuda:7 \
#     --data_dir /datapool/data2/home/majianzhu/xinheng/xiangzhen/DTI2/DTI/data \
#     --dict_target /datapool/data2/home/majianzhu/xinheng/xiangzhen/DTI2/DTI/data/dict_target.pkl \
#     --dict_ligand /datapool/data2/home/majianzhu/xinheng/xiangzhen/mk-dict/unimol2/dict_ligand.pkl \
#     --num_epochs 3 