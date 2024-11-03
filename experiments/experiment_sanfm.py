import torch
import numpy as np
import pandas as pd
import time
import random
import pickle
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from sample_generation.convert_to_input import get_input_data
from cfrm import CFRM, train, test, collate_fn, collate_fn_predict
from sample_generation.data_labeling import get_positive_labeled_dataset
from data_preprocessing.preprocessing import get_preprocessed_data


def save_to_csv(data, filename):
    columns = ['label'] + [f'I{i + 1}' for i in range(1536)] + ['C1', 'C2']
    df = pd.DataFrame(data, columns=columns)
    # 保存到 CSV 文件
    df.to_csv(filename, index=False)


def recommend(positive_labeled_dataset:list, api_representation_dict:dict, valid_api_dict:dict, model_obj, model_poolingModel):
    # 创建一个 API 索引与其首个类别的映射字典
    api_index_to_category = {api_info[0]: api_info[2][0] for api_info in valid_api_dict.values()}
    # 随机选择 1 个正样本
    sample_id = random.randint(0, len(positive_labeled_dataset) - 1)
    # 获取对应的 api_ids
    list_apis = positive_labeled_dataset[sample_id][1]
    # 取除最后一个以外的其他 API 作为已选 API
    selected_apis = list_apis[:-1]
    # 取最后一个API为候选 API
    candidate_api = list_apis[-1]
    # 获取已选 API 的所有表征
    selected_reps = [api_representation_dict[index] for index in selected_apis]

    # 将用于attention block学习的数据tensor格式化
    selected_reps_tensor = torch.stack(selected_reps)
    candidate_rep_tensor = api_representation_dict[candidate_api]
    category_rep_tensor = torch.tensor([api_index_to_category[candidate_api]])

    true_rep = (selected_reps_tensor, candidate_rep_tensor, category_rep_tensor)
    concat_tensors = [true_rep]

    # 将目前组合过的 API 进行剔除
    apis_id = [api_id for api_id in api_representation_dict if api_id not in selected_apis + [candidate_api]]

    # 将已选 API 的表征与剩余 API 的表征进行拼接
    for api_id in apis_id:
        other_rep = (selected_reps_tensor, api_representation_dict[api_id], torch.tensor([api_index_to_category[api_id]]))
        concat_tensors.append(other_rep)

    # 将所有拼接好的特征向量放到attention operation模块中进行加权求和处理
    predict_loader = DataLoader(dataset=concat_tensors, batch_size=len(concat_tensors), shuffle=False, collate_fn=collate_fn_predict)

    with torch.no_grad():
        for batch in predict_loader:
            # 解包批次内容
            selected_apis, candidate_api, candidate_categories, mask = batch
            attention_reps = model_poolingModel(selected_apis, candidate_api, candidate_categories, mask)

    recom_input = attention_reps
    pred = model_obj(recom_input)

    sorted_indices = torch.sort(pred, descending=True).indices
    # 真实值在预测结果中的排名
    true_result_rank = (sorted_indices == 0).nonzero().item() + 1
    return true_result_rank


def calculate_average_mrr_recall(n, model_obj, model_poolingModel):
    total_mrr = 0
    total_recall = [0] * 5  # 用列表保存每个召回率的累计值
    mashup_dict, valid_api_dict = get_preprocessed_data()
    positive_labeled_dataset = get_positive_labeled_dataset(mashup_dict, valid_api_dict)  # 获取正样本数据
    with open('../data/api_representation_dict.pkl', 'rb') as f_load:
        api_representation_dict = pickle.load(f_load)
    for _ in tqdm(range(n), desc="Predict"):
        rank = recommend(positive_labeled_dataset, api_representation_dict, valid_api_dict, model_obj, model_poolingModel)
        total_mrr += (1 / rank)
        # 更新召回率统计
        for i, k in enumerate(range(2, 12, 2)):  # 从2开始，每次增加2，直到10
            total_recall[i] += (1 if rank <= k else 0)

    # 计算平均值
    average_mrr = total_mrr / n
    average_recall = [r / n for r in total_recall]
    return average_mrr, average_recall


if __name__ == '__main__':

    start = time.time()

    print('---------------读取数据---------------')
    data, data_label = get_input_data()

    apiEmbeddingSize = 768
    hiddenUnits_aa = [128]
    baseVectorSize = 64
    dropoutRate = 0.2
    initStd = 1e-4
    poolingMethod = 'attention'
    batch_size = 256

    embed_dim = 32
    att_dim = 64
    i_num = 1536
    c_num = 2

    x_train, x_test, y_train, y_test = train_test_split(data, data_label, test_size=0.2, random_state=1)

    # 将训练和测试数据转换为 PyTorch 张量
    train_tensor_set = [(item, label) for item, label in zip(x_train, y_train)]
    test_tensor_set = [(item, label) for item, label in zip(x_test, y_test)]

    # 创建数据加载器
    train_loader = DataLoader(train_tensor_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_tensor_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    model_cowar = CFRM(apiEmbeddingSize, baseVectorSize, hiddenUnits_aa, poolingMethod, dropoutRate, initStd,
                      embed_dim, att_dim, i_num, c_num)
    optimizer = torch.optim.Adam(model_cowar.parameters(), lr=0.001)

    loss_train = np.inf
    LOSS_total = np.inf
    endure_count = 0

    print('---------------模型训练---------------')

    for epoch in range(50):
        train(model_cowar, train_loader, optimizer, epoch)
        LOSS, AUC = test(model_cowar, test_loader)

        if LOSS_total > LOSS:
            LOSS_total = LOSS
            AUC_total = AUC
            endure_count = 0
        else:
            endure_count += 1

        print(f"<Test> LOSS: {LOSS:.5f} AUC: {AUC:.5f}")

        if endure_count > 30:
            break

    # LOSS, AUC = test(model_cowar, test_loader)
    # print(f'The best LOSS: {LOSS:.5f} AUC: {AUC:.5f}')

    print('---------------结果预测---------------')

    model_poolingModel = model_cowar.poolingModel
    # torch.save(model_poolingModel.state_dict(), '../model/pooling_model_min.pth')
    model_poolingModel.eval()  # 设置为评估模式

    model_sanfm = model_cowar.sanfm
    # torch.save(model_sanfm.state_dict(), '../model/sanfm_model_min.pth')
    model_sanfm.eval()  # 设置为评估模式

    average_mrr, recall_list = calculate_average_mrr_recall(int(len(data)/2*0.2), model_sanfm, model_poolingModel)  # 2310
    print('SANFM ===>> MRR:', average_mrr)
    for i, recall in enumerate(recall_list):
        print(f'SANFM ===>> R@{(i + 1) * 2}:{recall}')

    print('----------------运行时间----------------')
    end = time.time()
    print('Running time: %.3f Seconds' % (end - start))







