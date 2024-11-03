import random
import time
import pandas as pd
import numpy as np
from tqdm import tqdm
from data_preprocessing.preprocessing import get_preprocessed_data


def get_subs(list):
    res = []
    for i in range(2**len(list)):  # 循环遍历从 1 到该集合长度的平方的所有整数
        sub = []
        for j in range(len(list)):  # 用于移位运算（方便移位的自增）
            if (i >> j) % 2 == 1:  # 移位运算，将二进制位作为传入参数list集合的倒序映射
                sub.append(list[j])
        res.append(sub)
    return res


def get_positive_labeled_dataset(mashup_dict:dict, valid_api_dict:dict):
    positive_labeled_dataset = list()
    # 建立 api_index 到第一个 category 的映射字典
    api_index_to_category = {
        api_info[0]: api_info[2][0] if api_info[2] else None
        for api_name, api_info in valid_api_dict.items()
    }
    # 基于 mashup_dict 数据，并以数据中的子集的形式生成正样本
    for key, value in tqdm(mashup_dict.items(), desc="Get positive labeled dataset"):
        # 这里是限制最大组合的API数量
        if len(value) <= 5:
            apis_index = [valid_api_dict[api_name][0] for api_name in value]
            subs = get_subs(apis_index)
            for sub in subs:
                if len(sub) >= 3:
                    for api in sub:
                        temp = sub.copy()
                        temp.remove(api)
                        temp.append(api)
                        # 直接从映射字典中获取 category
                        category = api_index_to_category.get(api, None)
                        positive_labeled_dataset.append([1, temp, category])
    return positive_labeled_dataset


def get_negative_labeled_dataset(positive_labeled_dataset:list, valid_api_dict:dict):
    # 获取负采样空间，即满足局部非互补性和全局非互补性的负样本采样
    apis_list = list(range(1, len(valid_api_dict) + 1))  # 前闭后开所以要加1，生成所有api的index
    negative_labeled_dataset = list()
    # 建立 api_index 到第一个 category 的映射字典
    api_index_to_category = {
        api_info[0]: api_info[2][0] if api_info[2] else None
        for api_name, api_info in valid_api_dict.items()
    }
    for sample in tqdm(positive_labeled_dataset, desc="Get negative labeled dataset"):
        composite_apis = sample[1]
        # 取除最后一个作为已选api
        select_apis = composite_apis[:-1]
        # 初始化一个集合
        apis_set = set()
        for item in positive_labeled_dataset:
            # 判断当前组合的输入是否是遍历中的组合的子集，如果是将遍历的组合加入到集合中，作为正样本的选择范围，即不是负样本的选择范围
            if set(select_apis).issubset(set(item[1])):
                apis_set = set.union(apis_set, item[1])
        # 生成负采样空间
        apis_set = sorted(apis_set)
        negative_sample_space = list(set(apis_list).difference(set(apis_set)))
        # 获取负样本
        negative_ids = random.sample(negative_sample_space, 1)  # 从选择的负采样空间中选取n个作为负样本
        for negative_id in negative_ids:
            category = api_index_to_category.get(negative_id, None)
            negative_sample = select_apis + [negative_id]
            negative_labeled_dataset.append([0, negative_sample, category])
    return negative_labeled_dataset


def get_labeled_dataset():
    mashup_dict, valid_api_dict = get_preprocessed_data()
    positive_labeled_dataset = get_positive_labeled_dataset(mashup_dict, valid_api_dict)  # 获取正样本数据
    negative_labeled_dataset = get_negative_labeled_dataset(positive_labeled_dataset, valid_api_dict)  # 获取负样本数据
    labeled_dataset = positive_labeled_dataset + negative_labeled_dataset  # 获得所有标签样本数据集
    return labeled_dataset


if __name__ == '__main__':
    start_time = time.time()
    dataset = get_labeled_dataset()
    length = len(dataset)
    print(length)

    count_dict = {i: 0 for i in range(2, 11)}
    for i in range(length):
        label_length = len(dataset[i][1])
        if 2 <= label_length <= 10:
            count_dict[label_length] += 1

    for key, value in count_dict.items():
        print(f"count_{key}: {value}")

    end_time = time.time()
    execution_time = end_time - start_time
    print("Execution time:", execution_time, "seconds")


