# toy-example-test
python predict.py \
    --input_data_dir data/toy_example/toy_dataset/test.csv \
    --output_data_dir predict-toy-example.csv \
    --dir_save_model split-random.pth \
    --fetch_pretrained_target \
    --batchsize 1 \
    --cuda_use cuda:0 \