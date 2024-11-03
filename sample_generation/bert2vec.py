#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : bert2vec.py
# @Author: Your Name
# @Date  : 2021/12/8
# @Desc  :
from transformers import BertModel, BertTokenizer
import pandas as pd
import torch
import numpy as np
# from nlp import load_dataset
# from data_bert_sentence import bert_sentence
# import pprint
import pickle

"""
    BERT模型表征过程
"""
def representation(api_description_dict:dict):
    model = BertModel.from_pretrained('bert-base-uncased')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    name = list(api_description_dict.keys())
    descr = list(api_description_dict.values())

    print(np.shape(descr))

    sentences = descr
    reps = []
    print("program start")
    k = 0

    for sentence in sentences:

        input_ids = tokenizer.encode(sentence,add_special_tokens=True,truncation="longest_first")
        input_ids = torch.tensor([input_ids])

        k += 1
        print(k, "reps")
        with torch.no_grad():
          last_hidden_states = model(input_ids)[0]
          reps.append(last_hidden_states[0][0])  # 训练为768维的向量, 并添加到reps列表中
          # print(last_hidden_states[0][0])
          print(np.shape(last_hidden_states[0][0]))

    api_representation_dict = dict(zip(name, reps))  # name为键，reps为value，两个列表合并成一个字典

    return api_representation_dict


