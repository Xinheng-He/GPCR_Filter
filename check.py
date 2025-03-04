import pickle
import torch

# 定义文件路径
file1 = "/datapool/data2/home/majianzhu/xinheng/xiangzhen/DTI2/DTI/data/dict_target.pkl"
file2 = "/datapool/data2/home/majianzhu/xinheng/xiangzhen/DTI2/DTI-tfcpi2/data/dict_target.pkl"
output_file = "Q9R261_comparison.txt"

# 读取 pkl 文件
def load_pkl(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(f)

# 递归转换所有 Tensor 到 CPU
def move_to_cpu(obj):
    if isinstance(obj, torch.Tensor):
        return obj.cpu()  # 确保所有 Tensor 移动到 CPU
    elif isinstance(obj, dict):
        return {k: move_to_cpu(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [move_to_cpu(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(move_to_cpu(v) for v in obj)
    elif isinstance(obj, set):
        return {move_to_cpu(v) for v in obj}
    else:
        return obj

# 加载并转换数据
dict1 = move_to_cpu(load_pkl(file1))
dict2 = move_to_cpu(load_pkl(file2))

# 检查 "Q9R261" 是否在字典中
if "Q9R261" in dict1 and "Q9R261" in dict2:
    value1 = dict1["Q9R261"]
    value2 = dict2["Q9R261"]

    # 保存到文本文件
    with open(output_file, "w") as f:
        f.write("Q9R261 在 dict1 中的值:\n")
        f.write(str(value1) + "\n\n")  # 转换为字符串写入
        f.write("Q9R261 在 dict2 中的值:\n")
        f.write(str(value2) + "\n\n")  # 转换为字符串写入

    print(f"Q9R261 的值已存入 {output_file}")

else:
    print("Q9R261 不在两个字典中，无法比较！")
