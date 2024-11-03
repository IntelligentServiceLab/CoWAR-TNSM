import pandas as pd
from collections import OrderedDict
import pickle


def get_all_api_data(filepath):
    # 读取 APIs 数据
    api_data = pd.read_excel(filepath)
    # 将 API 数据转换为字典，并检查 API 所需信息是否为空，若有一者为空则去除，即保留符合条件的 API 数据
    api_dict = {
        row['APIName']: [row['Description'], row['Categories']]
        for index, row in api_data.iterrows()
        if all(pd.notna([row['APIName'], row['Description'], row['Categories']]))
    }
    return api_dict


def get_mashup_data(filepath, all_api_dict:dict):
    # 读取 Mashups 数据
    mashup_data = pd.read_excel(filepath)
    # 处理 Mashup 数据：检查 Mashup 中的 API 是否有空信息（是否为无效数据），若能在 all_api_dict 中查询到，说明信息都非空
    mashup_dict = {
        row['mashups_name']: [api_name for api_name in row['related_apis'].split('###') if api_name in all_api_dict]
        for index, row in mashup_data.iterrows()
    }
    # 过滤有效的 Mashup：如果 Mashup 中有效 API 数目大于 1 并小于等于 10，则保留该条 Mashup 数据
    mashup_dict = {k: v for k, v in mashup_dict.items() if 1 < len(v) <= 10}
    mashup_dict = OrderedDict(sorted(mashup_dict.items(), key=lambda item: item[0]))
    return mashup_dict


def get_category_data(all_api_dict:dict, valid_api_set:set):
    # 获取 category 集合，并对其进行排序
    category_set = sorted({
        category
        for api_name in valid_api_set
        for category in all_api_dict[api_name][1].split('###')
    })
    # 对 category 处理：category_name 作为 key 值，category_index 作为 value 值
    category_dict = {category: index + 1 for index, category in enumerate(category_set)}
    return category_dict


def get_valid_api_data(all_api_dict:dict, valid_api_set:set, category_dict:dict):
    # 将有效 API 的类别信息数字化，并将有效 API 的数据信息完整化
    valid_api_dict = {
        api_name: [index + 1, all_api_dict[api_name][0], [category_dict[category] for category in all_api_dict[api_name][1].split('###')]]
        for index, api_name in enumerate(sorted(valid_api_set))
    }
    valid_api_dict = OrderedDict(sorted(valid_api_dict.items(), key=lambda item: item[0]))
    return valid_api_dict


def get_preprocessed_data():
    api_filepath = "../data/apisData.xlsx"  # API 文件位置
    mashup_filepath = "../data/mashups.xlsx"  # Mashup 文件位置
    all_api_dict = get_all_api_data(api_filepath)  # 处理后符合条件的 API 数据为 21591 条
    mashup_dict = get_mashup_data(mashup_filepath, all_api_dict)  # 处理后符合条件的 Mashup 数据为 2230 条
    valid_api_set = {api for apis in mashup_dict.values() for api in apis}  # 获取有效 API 数据（即在 mashup_dict 中使用过的 API 数据），有 853 条
    category_dict = get_category_data(all_api_dict, valid_api_set)  # 有效 API 数据中类别的总数为 319
    valid_api_dict = get_valid_api_data(all_api_dict, valid_api_set, category_dict)
    return mashup_dict, valid_api_dict

