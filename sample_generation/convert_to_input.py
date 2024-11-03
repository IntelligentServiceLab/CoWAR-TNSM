import time
import pandas as pd
import pickle
import torch
from tqdm import tqdm
from sample_generation.data_labeling import get_labeled_dataset


def get_input_data():
    res = []
    labels = []
    labeled_dataset = get_labeled_dataset()  # 此处每次执行一次，负样本的数据都是不同的，因为负样本在选取方式上是随机的
    with open('../data/api_representation_dict.pkl', 'rb') as f_load:
        api_representation_dict = pickle.load(f_load)
    for row in tqdm(labeled_dataset, desc="Convert to input"):
        label = row[0]
        selected_apis = row[1][:-1]
        candidate_api = row[1][-1]
        category = row[2]
        selected_reps = [api_representation_dict[index] for index in selected_apis]
        selected_reps_tensor = torch.stack(selected_reps)
        # 进行表征拼接
        candidate_rep_tensor = api_representation_dict[candidate_api]
        sample = (selected_reps_tensor, candidate_rep_tensor, torch.tensor([category]))
        res.append(sample)
        labels.append(label)
    return res, labels


def save_to_csv(data, filename):
    columns = ['label'] + [f'I{i + 1}' for i in range(1536)] + ['C1', 'C2']
    df = pd.DataFrame(data, columns=columns)
    # 保存到 CSV 文件
    df.to_csv(filename, index=False)


if __name__ == "__main__":
    start_time = time.time()
    input_data, label = get_input_data()
    # save_to_csv(input_data, '../data/input_data.csv')
    end_time = time.time()
    execution_time = end_time - start_time
    print("Execution time:", execution_time, "seconds")