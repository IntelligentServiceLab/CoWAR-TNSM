import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score
from sanfm import SANFM
from datetime import datetime


def collate_fn(batch):
    selected_apis_list, candidate_api_list, candidate_categories_list, labels = [], [], [], []
    for item in batch:
        selected_apis, candidate_api, candidate_category = item[0]
        label = item[1]
        selected_apis_list.append(selected_apis)
        candidate_api_list.append(candidate_api)
        candidate_categories_list.append(candidate_category)
        labels.append(label)

    max_len = max([apis.size(0) for apis in selected_apis_list])

    padded_selected_apis = [F.pad(apis, (0, 0, 0, max_len - apis.size(0)), 'constant', 0) for apis in
                            selected_apis_list]
    masks = [torch.cat([torch.ones(apis.size(0)), torch.zeros(max_len - apis.size(0))]) for apis in
             selected_apis_list]

    return (torch.stack(padded_selected_apis),
            torch.stack(candidate_api_list),
            torch.stack(candidate_categories_list),
            torch.stack(masks)), torch.tensor(labels, dtype=torch.float32)


def collate_fn_predict(batch):
    selected_apis_list, candidate_api_list, candidate_categories_list = zip(*batch)

    max_len = max(apis.size(0) for apis in selected_apis_list)

    padded_selected_apis = [F.pad(apis, (0, 0, 0, max_len - apis.size(0)), 'constant', 0) for apis in
                            selected_apis_list]
    masks = [torch.cat([torch.ones(apis.size(0)), torch.zeros(max_len - apis.size(0))]) for apis in
             selected_apis_list]

    return torch.stack(padded_selected_apis), torch.stack(candidate_api_list), torch.stack(candidate_categories_list), torch.stack(masks)


# PoolingModel
class PoolingModel(nn.Module):
    def __init__(self, apiEmbeddingSize, baseVectorSize, hiddenUnits_aa, poolingMethod, dropoutRate, initStd):
        super(PoolingModel, self).__init__()
        self.apiEmbeddingSize = apiEmbeddingSize  # API的向量维度
        self.baseVectorSize = baseVectorSize  # 交互特征向量的维度
        self.hiddenUnits_aa = hiddenUnits_aa  # 隐藏层神经元个数，格式为[64, 32]
        self.poolingMethod = poolingMethod  # 池化操作类型
        self.dropoutRate = dropoutRate  # dropout
        self.initStd = initStd  # 标准差
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # cpu 或者 gpu

        self.sigmoid_predict = nn.Sigmoid()

        # 根据 poolingMethod 初始化相应的组件
        if self.poolingMethod == 'attention':
            # API之间进行特征交互
            self.inputSize_aa = 2 * self.apiEmbeddingSize  # 已选API与候选API拼接后用于交互的维度，仅考虑已选API和候选API的特征向量
            # self.inputSize_aa = 4 * self.apiEmbeddingSize  # 已选API与候选API拼接后用于交互的维度，考虑了已选API和候选API的互补性和差异性
            self.hiddenUnits_aa = [self.inputSize_aa] + self.hiddenUnits_aa + [self.baseVectorSize]

            self.linears_aa = nn.ModuleList(
                [nn.Linear(self.hiddenUnits_aa[i], self.hiddenUnits_aa[i + 1]) for i in
                 range(len(self.hiddenUnits_aa) - 1)]
            ).to(self.device)
            self.relus_aa = nn.ModuleList([nn.PReLU() for _ in range(len(self.hiddenUnits_aa) - 1)]).to(self.device)

            # Dropout 层
            self.dropout = nn.Dropout(self.dropoutRate)

            # 注意力层初始化
            self.linear_att = nn.Linear(self.baseVectorSize, 1).to(self.device)
            self.relu_att = nn.PReLU().to(self.device)
            self.softmax_att = nn.Softmax(dim=1)

            for name, parameter in self.linear_att.named_parameters():
                if 'weight' in name:
                    nn.init.normal_(parameter, mean=0, std=self.initStd)

    def forward(self, selected_apis, candidate_api, candidate_categories, mask):
        batch_size, max_len, input_dim = selected_apis.size()

        if self.poolingMethod == 'attention':
            # 扩展候选API以匹配已选API的形状
            candidate_api_expanded = candidate_api.unsqueeze(1).expand(batch_size, max_len, input_dim)

            # 创建掩码并应用在候选API上，确保没有达到最大长度的组中填充的向量是0
            mask_expanded = mask.unsqueeze(-1).expand_as(candidate_api_expanded)
            candidate_api_padded = candidate_api_expanded * mask_expanded

            # 拼接已选API和处理后的候选API，仅考虑已选API和候选API的特征向量
            concatenated_features = torch.cat((selected_apis, candidate_api_padded), dim=2)  # [batch_size, max_len, 2*apiEmbeddingSize]

            # 拼接已选API和候选API的特征向量，考虑了已选API和候选API的互补性和差异性
            # concatenated = torch.cat((selected_apis, candidate_api_padded), dim=2)
            # # 逐个元素相乘
            # multiplied = torch.mul(selected_apis, candidate_api_padded)
            # # 逐个元素相减
            # subtracted = torch.sub(selected_apis, candidate_api_padded)
            # # 拼接所有结果
            # concatenated_features = torch.cat((concatenated, multiplied, subtracted), dim=2)

            # 通过全连接网络进行处理
            for linear, relu in zip(self.linears_aa, self.relus_aa):
                concatenated_features = linear(concatenated_features)
                concatenated_features = relu(concatenated_features)
                concatenated_features = self.dropout(concatenated_features)

            # 计算注意力分数
            attn_scores = self.linear_att(concatenated_features)
            attn_scores = self.relu_att(attn_scores).squeeze(2)
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)

            # 使用softmax计算注意力权重
            attn_weights = self.softmax_att(attn_scores)  # [batch_size, max_len]
            # print(attn_weights)

            # 计算已选API的加权特征并求和
            weighted_apis = selected_apis * attn_weights.unsqueeze(2)  # [batch_size, max_len, apiEmbeddingSize]
            output = weighted_apis.sum(dim=1)  # [batch_size, apiEmbeddingSize]

        elif self.poolingMethod == 'average':
            # 计算有效特征的平均值，避免除以0
            valid_len = mask.sum(dim=1, keepdim=True).clamp(min=1)
            output = selected_apis.sum(dim=1) / valid_len  # [batch_size, apiEmbeddingSize]

        elif self.poolingMethod == 'max':
            # 使用掩码处理全0向量，并设置掩码中的填充位置为负无穷大，以排除这些位置的影响
            masked_selected_apis = selected_apis.masked_fill(mask.unsqueeze(-1) == 0, -float('inf'))
            output, _ = masked_selected_apis.max(dim=1)  # [batch_size, apiEmbeddingSize]

        elif self.poolingMethod == 'min':
            # 使用掩码处理全0向量，并设置掩码中的填充位置为正无穷大，以排除这些位置的影响
            masked_selected_apis = selected_apis.masked_fill(mask.unsqueeze(-1) == 0, float('inf'))
            output, _ = masked_selected_apis.min(dim=1)  # [batch_size, apiEmbeddingSize]

        else:
            raise Exception('Unknown pooling method')

        # 拼接候选API的类别编号到最终输出向量后面
        output = torch.cat((output, candidate_api, candidate_categories, candidate_categories), dim=1)  # [batch_size, 2 * apiEmbeddingSize + 2]

        return output


# CFRM (Complementary Function Recommendation Model)，包括PoolingModel和SANFM Model
class CFRM(nn.Module):
    def __init__(self, apiEmbeddingSize, baseVectorSize, hiddenUnits_aa, poolingMethod, dropoutRate, initStd,
                 embedDim_SANFM, attDim_SANFM, i_num, c_num):
        super(CFRM, self).__init__()
        self.poolingModel = PoolingModel(apiEmbeddingSize, baseVectorSize, hiddenUnits_aa, poolingMethod, dropoutRate, initStd)
        self.sanfm = SANFM(embedDim_SANFM, attDim_SANFM, dropoutRate, i_num, c_num)
        self.criterion = nn.BCELoss(weight=None, reduction='mean')

    def forward(self, selected_apis, candidate_api, candidate_categories, mask):
        poolingModel_output = self.poolingModel(selected_apis, candidate_api, candidate_categories, mask)
        final_output = self.sanfm(poolingModel_output)
        return final_output

    def loss(self, selected_apis, candidate_api, candidate_categories, mask, labels):
        pred = self.forward(selected_apis, candidate_api, candidate_categories, mask)
        pred = pred.to(torch.float32)
        labels = labels.to(torch.float32).squeeze()
        loss1 = self.criterion(pred, labels)
        return loss1


# 定义训练函数
def train(model, train_loader, optimizer, epoch):
    model.train()
    avg_loss = 0.0
    for i, batch in enumerate(train_loader):
        (selected_apis, candidate_api, candidate_categories, mask), labels = batch
        optimizer.zero_grad()
        loss2 = model.loss(selected_apis, candidate_api, candidate_categories, mask, labels)
        loss2.backward(retain_graph=True)
        optimizer.step()
        avg_loss += loss2.item()
        if (i + 1) % 10 == 0:
            print('%s Training: [%d epoch, %3d batch] loss: %.5f' % (
                datetime.now(), epoch, i + 1, avg_loss / 10))
            # loss_train = avg_loss
            avg_loss = 0.0
    return 0


# 定义测试函数
def test(model, test_loader):
    model.eval()  # 设置模型为评估模式
    criterion = nn.BCELoss(reduction='mean')
    LOSS = []
    AUC = []

    with torch.no_grad():  # 关闭梯度计算
        for batch in test_loader:
            (selected_apis, candidate_api, candidate_categories, mask), labels = batch
            pred = model(selected_apis, candidate_api, candidate_categories, mask)
            pred = pred.to(torch.float32)
            labels = labels.squeeze().to(torch.float32)
            loss_value = criterion(pred, labels)
            try:
                auc_value = roc_auc_score(labels.detach().tolist(), pred.detach().tolist())
                LOSS.append(loss_value.detach())
                AUC.append(auc_value)
            except ValueError:
                pass
    loss = np.mean(LOSS)
    auc = np.mean(AUC)
    return loss, auc



