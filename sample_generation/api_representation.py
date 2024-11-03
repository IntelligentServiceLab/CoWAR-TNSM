import time
import pickle
from data_preprocessing.preprocessing import get_preprocessed_data
from bert2vec import representation


"""
    说明：实验中使用的 API 表征是将所有 API 描述信息输入到 BERT 模型，由于数据过于庞大，而后根据出现在正样本中的 API 数据作为最后的有效表征
        因此建议使用文件中自带的表征文件（api_representation_dict.pkl）进行实验。如果有需要也可自行生成，但是需要修改相应的代码。
"""


def get_api_representation():
    mashup_dict, valid_api_dict = get_preprocessed_data()
    # 通过遍历已经经过预处理的 API 数据，获取其中的索引值作为 key ，描述信息为 value
    api_description_dict = {
        api_info[0]: api_info[1]
        for api_info in valid_api_dict.values()  # 如果自行生成，该处的valid_api_dict需要修改
    }
    api_representation_dict = representation(api_description_dict)
    # 保存数据
    with open('../data/api_representation_dict.pkl', 'wb') as f_save:
        pickle.dump(api_representation_dict, f_save)


if __name__ == "__main__":
    start_time = time.time()
    get_api_representation()
    end_time = time.time()
    execution_time = end_time - start_time
    print("Execution time:", execution_time, "seconds")



