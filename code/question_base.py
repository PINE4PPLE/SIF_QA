import pandas as pd
import numpy as np
import nltk 
import collections
import pickle
import pyhanlp
from collections import Iterable
from collections import Counter 
from pandas import DataFrame
from sklearn.decomposition import PCA
from pyhanlp import *
"""
一、加载数据：
    1.加载问答对
    2.加载预训练的词向量模型
二、问题的向量化：
    1.问题分词，得到语料库
    2.将词转化为向量
    3.将问题由词向量矩阵转化为问题向量
    4.将问题向量组合得到问题向量矩阵
三、将结果保存

"""
#数据输入 
"""
输入：csv文件的路径
输出：Dataframe， 列名称为Question Answer
"""
def load_data(data_path):
    """ 
    data = pd.read_excel(data_path)
    new_data = data.iloc[:, 0:2]
    new_data.columns = [["Question", "Answers"]]
    new_data = new_data[:-1]"""

    data = pd.read_csv(data_path)
    return data

#清洗数据（分词， 删除停用词）
"""
输入：str原始问题，停用词，分词器
输出：str[]，分好词的列表
"""
def clean(sentence, stop_words, segment):
    #words_list = jieba.lcut(sentence, cut_all=True)
    clean_words = []
    words_list = segment.segment(sentence)
    for word in words_list:
        if str(word) not in stop_words:
            clean_words.append(str(word))
    """
    words_list = segment.segment(sentence)
    clean_words = [str(word) for word in words_list if str(word) not in stop_words]   
    """      
    return clean_words


#获得输入文本
"""
输入：导入的文件 Dataframe， 对应的column名称, stop_words停用词
输出：list， 文本集
"""
def get_corpus(source, column, stop_words):
    HanLP.Config.ShowTermNature = False
    newSegment = JClass("com.hankcs.hanlp.tokenizer.NLPTokenizer")
    corpus = []
    for i in source[column]:
        tmp_Q = clean(i, stop_words, newSegment)
        corpus.append(tmp_Q)
    return corpus

#raw sentence vector
"""
输入：词向量模型， 分好词的句子
输出：句子对应的向量
注意，由于有些词的词频过低， 对于keyerror，这里将以0向量代替
"""
def s2v(model, sentence):
    vec = []
    for word in sentence:
        try:
            tmp = model[word]
        except KeyError:
            tmp = [0 for _ in range(300)]
            #print("missing word:%s \n" %(word))
        vec.append(tmp)
    return vec

#获得问题矩阵
def get_Question_matrix(model, corpus, pattern = 'SIF', SIF_weight = 0.0001):
    if pattern == 'AVG':
        Q = []
        for query in corpus:
            tmp_vec = np.array(s2v(model, query))
            Q_vec = tmp_vec.mean(axis = 0)
            Q.append(Q_vec)
        Q_matrix =  np.array(Q)
        return (Q_matrix, 0, 0)
    elif pattern == 'SIF':
        Q = []
        raw = []
        merge = []
        weight = []
        for i in range(len(corpus)):
            merge.extend(corpus[i])
        fdist = nltk.probability.FreqDist(merge)
        count = 0
        for query in corpus:
                tmp_vec = np.array(s2v(model, query))
                weight_matrix = np.array([[SIF_weight/(SIF_weight + fdist[word]/fdist.N())  for word in query]])
                tmp = tmp_vec * weight_matrix.T
                Q_vec = tmp.mean(axis = 0)
                Q.append(Q_vec)
                weight.append(weight_matrix)
                raw.append(tmp_vec)
        #print(weight[3455])
        #print(raw[3455])
        Q_matrix = np.array(Q)
        #print(Q_matrix[3455])
        pca = PCA(n_components = 1)
        u = pca.fit_transform(Q_matrix.T)
        res = Q_matrix - np.dot(Q_matrix, np.dot(u, u.T)) 
        #print(res[3455])
    return (res, fdist, u)

class question_database():
    def __init__(self, Q_matrix, fdist, singular_v):
        self.Q_matrix = Q_matrix
        self.fdist = fdist
        self.singular_v = singular_v
def main():
    stop_words = {'。', ',', '？', '年', '的', ''}
    path = "C:/Users/leo/Desktop/knowledge_quiz"
    #model = Word2Vec.load("trained_on_wiki.model")
    model = np.load('simplified_model.npy').item()
    print("load model successfully")

    data = load_data(path + "/DATA/clean.csv")
    print("load data successfully")

    corpus = get_corpus(data, "Question", stop_words)
    print("generate corpus successfully")

    Q_matrix, fdist, singular_v = get_Question_matrix(model, corpus, 'SIF')
    print("generate question matrix successfully")
    #print(Q_matrix)

    QD = question_database(Q_matrix, fdist, singular_v)
    with open(path+"/DATA/QD.txt", 'wb') as f:
        pickle.dump(QD, f, 0)
    print("question database saved successfully")
    return 0

if __name__ == '__main__':
    main()

