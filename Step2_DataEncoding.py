# python3
# -*- coding:utf-8 -*-

"""
@author:野山羊骑士
@e-mail：thankyoulaojiang@163.com
@file：PycharmProject-PyCharm-DataEncoding.py
@time:2021/9/7 10:04 
"""
import numpy as np
import pandas as pd
import codecs
from subword_nmt.apply_bpe import BPE
#from Step1_getData import GetData

def drug2emb_encoder(smile):
    vocab_path = "./drug_codes_chembl_freq_1500.txt"
    sub_csv = pd.read_csv("./subword_units_map_chembl_freq_1500.csv")

    bpe_codes_drug = codecs.open(vocab_path)
    dbpe = BPE(bpe_codes_drug, merges=-1, separator='')

    idx2word_d = sub_csv['index'].values
    words2idx_d = dict(zip(idx2word_d, range(0, len(idx2word_d))))

    max_d = 50
    t1 = dbpe.process_line(smile).split()  # split
    try:
        i1 = np.asarray([words2idx_d[i] for i in t1])  # index
    except:
        i1 = np.array([0])

    l = len(i1)
    if l < max_d:
        i = np.pad(i1, (0, max_d - l), 'constant', constant_values=0)
        input_mask = ([1] * l) + ([0] * (max_d - l))
    else:
        i = i1[:max_d]
        input_mask = [1] * max_d

    return i, np.asarray(input_mask)

'''
class DataEncoding:
    def __init__(self, args, vocab_dir, cancer_id, sample_id, target_id, drug_id):
        self.vocab_dir = vocab_dir
        # 一个获取数据的函数类
        #self.Getdata = GetData(args, cancer_id, sample_id, target_id, drug_id)

    def _drug2emb_encoder(self, smile):
        vocab_path = "{}/ESPF/drug_codes_chembl_freq_1500.txt".format(  # "{}/ESPF/drug_codes_chembl_freq_1500.txt".format(
            self.vocab_dir)
        sub_csv = pd.read_csv(
            "{}/ESPF/subword_units_map_chembl_freq_1500.csv".format(self.vocab_dir))

        bpe_codes_drug = codecs.open(vocab_path)
        dbpe = BPE(bpe_codes_drug, merges=-1, separator='')

        idx2word_d = sub_csv['index'].values
        words2idx_d = dict(zip(idx2word_d, range(0, len(idx2word_d))))

        max_d = 50
        t1 = dbpe.process_line(smile).split()  # split
        try:
            i1 = np.asarray([words2idx_d[i] for i in t1])  # index
        except:
            i1 = np.array([0])

        l = len(i1)
        if l < max_d:
            i = np.pad(i1, (0, max_d - l), 'constant', constant_values=0)
            input_mask = ([1] * l) + ([0] * (max_d - l))
        else:
            i = i1[:max_d]
            input_mask = [1] * max_d

        return i, np.asarray(input_mask)
'''
