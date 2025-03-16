# split-inter
python predict.py \
    --input_data_dir data/split-target-inter-pure6/test.csv \
    --output_data_dir predict-split-inter.csv \
    --dir_save_model split-target-inter.pth \
    --fetch_pretrained_target \
    --hid_dim 128 \
    --dropout 0 \
    --batchsize 32 \
    --cuda_use cuda:0 \

# split-random
python predict.py \
    --input_data_dir data/split-random/test.csv \
    --output_data_dir predict-split-random.csv \
    --dir_save_model split-random.pth \
    --fetch_pretrained_target \
    --batchsize 32 \
    --cuda_use cuda:0 \

# split-intra
python predict.py \
    --input_data_dir data/split-target-intra/test.csv \
    --output_data_dir predict-split-intra.csv \
    --dir_save_model split-intra.pth \
    --fetch_pretrained_target \
    --batchsize 32 \
    --cuda_use cuda:0 \