# # split-random
# python predict.py \
#     --input_data_dir /datapool/data2/home/majianzhu/xinheng/xiangzhen/DTI2/DTI/data/split-random/test.csv \
#     --output_data_dir predict-split-random \
#     --dir_save_model /datapool/data2/home/majianzhu/xinheng/xiangzhen/DTI2/DTI-tfcpi2/run/20250221-1942_split-random/best_acc.pth \
#     --fetch_pretrained_target \
#     --batchsize 1 \
#     --cuda_use cuda:0 \

# # split-random
# python predict-2.py \
#     --input_data_dir /datapool/data2/home/majianzhu/xinheng/xiangzhen/DTI2/DTI/data/split-random/test.csv \
#     --output_data_dir predict-split-random \
#     --dir_save_model /datapool/data2/home/majianzhu/xinheng/xiangzhen/DTI2/DTI-tfcpi2/run/20250221-1942_split-random/best_acc.pth \
#     --fetch_pretrained_target \
#     --batchsize 1 \
#     --cuda_use cuda:0 \

# # split-random
# python predict-3.py \
#     --input_data_dir /datapool/data2/home/majianzhu/xinheng/xiangzhen/DTI2/DTI/data/split-random/test.csv \
#     --output_data_dir predict-split-random \
#     --dir_save_model /datapool/data2/home/majianzhu/xinheng/xiangzhen/DTI2/DTI-tfcpi2/run/20250221-1942_split-random/best_acc.pth \
#     --fetch_pretrained_target \
#     --batchsize 1 \
#     --cuda_use cuda:0 \

# # split-random
# python predict-3.py \
#     --input_data_dir test-visual-0324.csv \
#     --output_data_dir test-visual-0324 \
#     --dir_save_model /datapool/data2/home/majianzhu/xinheng/xiangzhen/DTI2/DTI-tfcpi2/run/20250221-1942_split-random/best_acc.pth \
#     --fetch_pretrained_target \
#     --batchsize 1 \
#     --cuda_use cuda:0 \

# split-random
python predict-3.py \
    --input_data_dir crawl-0401/filtered_test_visual.csv \
    --output_data_dir crawl-0401/filtered_test_visual \
    --dir_save_model run/20250221-1942_split-random/best_acc.pth \
    --fetch_pretrained_target \
    --batchsize 1 \
    --cuda_use cuda:0 \