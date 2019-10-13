import pandas as pd
import numpy as np
import nltk 
import collections
import pickle
import pyhanlp
import json
import math
from pandas import DataFrame
from pyhanlp import *

"""
一、数据加载：
    1.加载预训练好的词向量模型
    2.加载问答数据
    3.加载问题向量化结果
    4.加载需要回答的问题
二、问题向量化：
    1.分词
    2.根据不同要求进行向量化
三、问题匹配：
"""
class question_database():
    def __init__(self, Q_matrix, fdist, singular_v):
        self.Q_matrix = Q_matrix
        self.fdist = fdist
        self.singular_v = singular_v
#数据输入 
"""
输入：csv文件的路径
输出：Dataframe， 列名称为Question Answer
"""
def load_data(path):
    """data = pd.read_excel(path)
    new_data = data.iloc[:, 0:2]
    new_data.columns = [["Question", "Answers"]]
    new_data = new_data[:-1]"""
    new_data = pd.read_csv(path)
    return new_data

def load_QD(path):
    fr = open(path + "/DATA/QD.txt",'rb')  
    QD = pickle.load(fr)
    return (QD.Q_matrix, QD.fdist, QD.singular_v)

def json_decode(path):
    with open(path,'r',encoding='utf-8') as fp:
        qr_js=json.load(fp)
    print(qr_js)
    need=qr_js['question']
    return need

def json_encode(question,answer):
    output={'question':question,
        'answer': answer,
            '{KEY}':'{VALUE}'}
    file = open("../output.json",'w',encoding='utf-8')
    js_str=json.dump(output,file)
    print(output)

def get_raw_vec(sentence, model):
    vec = []
    for word in sentence:
        try:
            tmp = model[word]
        except KeyError:
            tmp = [0 for _ in range(300)]
        vec.append(tmp)
    return vec

def que2vec(query, model, fdist, stop_words, singular_v, pattern = 'SIF', SIF_weight = 0.0001):
    HanLP.Config.ShowTermNature = False
    newSegment = JClass("com.hankcs.hanlp.tokenizer.NLPTokenizer")
    words_list = newSegment.segment(query)
    clean_words = [str(word) for word in words_list if str(word) not in stop_words]
    if pattern == 'SIF':
        vec = np.array(get_raw_vec(clean_words, model))
        weight = np.array([[SIF_weight/( SIF_weight + fdist[word]/fdist.N())  for word in clean_words]])
        tmp = vec * weight.T
        tmp_vec = tmp.mean(axis = 0)
        query_vec = tmp_vec - np.dot(np.dot(singular_v, singular_v.T), tmp_vec)
        #print(weight)
        #print(vec)
        print(tmp_vec.shape)
        #print(query_vec)
    elif pattern == 'AVG':
        vec = np.array(get_raw_vec(clean_words, model))
        query_vec = vec.mean(axis = 0)

    return query_vec 

def cal_similarity(query_vec, Q_matrix):
    sim_dict = {}
    for i in range(Q_matrix.shape[0]):
        sim = np.dot(query_vec, Q_matrix[i].T)/(np.linalg.norm(query_vec)*np.linalg.norm(Q_matrix[i]))       
        if sim > 0:
            sim_dict[i] = sim
    return sim_dict

def top5_question(query_vec, Q_matrix):
    sim_dic = cal_similarity(query_vec, Q_matrix)
    d = sorted(sim_dic, key=lambda x: sim_dic[x], reverse = True)
    return d[:5]


def main():
    stop_words = {'。', ',', '？', '年', '的', ''}
    path = "C:/Users/leo/Desktop/knowledge_quiz"
    
    #model = Word2Vec.load("trained_on_wiki.model")
    model = np.load('simplified_model.npy').item()
    print("load model successfully")

    data = load_data(path + "/DATA/clean.csv")
    print("load data successfully")
    
    Q_matrix, fdist, singular_v = load_QD(path)
    print(data["Question"][3455])
    print(singular_v.shape)
    while True:
        query_path = "../input.json"
        new_qr = json_decode(query_path)
        query_vec = que2vec(new_qr, model, fdist, stop_words, singular_v, 'SIF')
        res = top5_question(query_vec, Q_matrix)
        print(res)
        output_answer = data["Answers"][res[0]]
        json_encode(new_qr, output_answer)

        if_continue = input("是否继续? yes (y) or no (n)")
        if if_continue !='y':
            break

    return 0


if __name__ == '__main__':
    main()