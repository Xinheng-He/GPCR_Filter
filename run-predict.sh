# split-inter
python predict.py \
    --input_data_dir data/split-target-inter-pure6/test.csv \
    --output_data_dir predict-split-inter.csv \
    --dir_save_model /datapool/data2/home/majianzhu/xinheng/xiangzhen/DTI2/DTI-tfcpi2/run/20250303-1045_split-target-inter-pure6/best_acc.pth \
    --fetch_pretrained_target \
    --hid_dim 128 \
    --dropout 0 \
    --batchsize 32 \
    --cuda_use cuda:7 \

# split-random
python predict.py \
    --input_data_dir data/split-random/test.csv \
    --output_data_dir predict-split-random.csv \
    --dir_save_model /datapool/data2/home/majianzhu/xinheng/xiangzhen/DTI2/DTI-tfcpi2/run/20250221-1942_split-random/best_acc.pth \
    --fetch_pretrained_target \
    --batchsize 32 \
    --cuda_use cuda:7 \

# split-intra
python predict.py \
    --input_data_dir data/split-target-intra/test.csv \
    --output_data_dir predict-split-intra.csv \
    --dir_save_model /datapool/data2/home/majianzhu/xinheng/xiangzhen/DTI2/DTI-tfcpi2/run/20250222-1318_split-target-intra/best_acc.pth \
    --fetch_pretrained_target \
    --batchsize 32 \
    --cuda_use cuda:7 \