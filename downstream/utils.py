import os
import logging
import json
import matplotlib as plt
import yaml
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from scipy import sparse,stats
from collections import Counter,OrderedDict
from sklearn.metrics import (roc_curve, roc_auc_score, auc, recall_score, average_precision_score, 
                            precision_recall_curve, f1_score, confusion_matrix)



def eval_gini_normalization(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    index = np.argsort(y_pred) #min->max
    y_true = y_true[index]
    cumsum = np.cumsum(y_true)
    ratios = cumsum/sum(y_true)
    ratios = (ratios[:-1]+ratios[1:])/2
    auc_gini = sum(ratios)/(len(ratios))
    return 0.5 - auc_gini
    
def eval_aucpr(y_true, y_pred):
    y_true = np.array(y_true).ravel()
    y_pred = np.array(y_pred).ravel()
    precisions, recalls ,thrs = precision_recall_curve(y_true, y_pred)
    mean_precisions = 0.5*(precisions[:-1]+precisions[1:])
    intervals = recalls[:-1] - recalls[1:]
    auc_pr = np.dot(mean_precisions, intervals)
    return auc_pr


def get_ks(preds, labels):
    fpr,tpr,thresholds = roc_curve(labels,preds)
    ks = (tpr - fpr).max()
    return ks

def get_auc_ks(y_prob, y_label):
    auc = roc_auc_score(y_true=y_label,y_score=y_prob)
    ks = get_ks(y_prob, y_label)
    return {'auc':auc, 'ks':ks}


def plot_roc(labels, predict_prob,file=None):
    false_positive_rate,true_positive_rate,thresholds=roc_curve(labels, predict_prob)
    roc_auc=auc(false_positive_rate, true_positive_rate)
    plt.title('ROC')
    plt.plot(false_positive_rate, true_positive_rate,'b',label='AUC = %0.4f'% roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0,1],[0,1],'r--')
    plt.ylabel('TPR')
    plt.xlabel('FPR')
    if file:
        plt.savefig(file)

def calc_threshold_vs_depth(y_true, y_prob, stats_file=None):
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    # print(y_prob[:100])
    ns = len(y_true)
    index = np.argsort(y_prob)
    index = index[::-1]
    y_prob = y_prob[index]
    # print(y_prob[:100])
    ratios = [0.001,0.002,0.003,0.004,0.005, 0.01,0.05, 0.1,0.15, 0.2, 0.25,0.3,
               0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95,1]
    
    pos_num = sum(y_true)
    pv = pos_num/len(y_true)
    print('pos-to-neg-ratio=%f'%pv)
    depths =[]
    rates =[]
    samples=[]
    covers =[]
    lifts=[]
    p_thresholds=[]
    for ratio in ratios:
        top_k = max(1,int(ns*ratio))
        index1 = index[:top_k]
        top_true =  y_true[index1] 
        hit_rate = sum(top_true)/top_k  
        cover = sum(top_true)/pos_num
        p_threshold = y_prob[top_k-1]
        lift = hit_rate/pv
        
        depths.append(ratio)
        rates.append(hit_rate)
        samples.append(top_k)
        covers.append(cover)
        lifts.append(lift)
        p_thresholds.append(p_threshold)
        
    df = pd.DataFrame({'深度':depths,'命中率': rates, '覆盖率':covers, '样本数':samples,
                  '提升度':lifts, '概率门限':p_thresholds})
    if stats_file is not None:
        df.to_csv(stats_file, encoding='utf-8')   
    return df


def drop_null_columns(df, null_percent=0.95, inplace=False, 
                      null_columns_store=None,
                      anotation_dict={}):
    
    null_cnt = df.isnull().sum(axis=0)/df.index.size
    null_cnt = null_cnt[null_cnt>=null_percent]
    null_cols = null_cnt.index.tolist()
    if inplace:
        df.drop(null_cols,axis=1, inplace =True)
    return null_cols

def drop_corr_columns(df, corr_file, corr_threshold=0.99, inplace=False):
    import operator
    corr = df.corr()
    upper = corr.where(np.triu(np.ones(corr.shape),k=1).astype(np.bool))
    # pos_cols  = [column for column in upper.columns if any(upper[column] > corr_threshold)]
    # neg_cols = [column for column in upper.columns if any(upper[column] < -corr_threshold)]
    # drop_cols = list(set(pos_cols + neg_cols))
    # df.drop(columns = drop_cols, inplace=True)
    #save info
    mask_mat = (upper > corr_threshold) | (upper < -corr_threshold)
    col1 = np.where(mask_mat)[0]
    col2 = np.where(mask_mat)[1]  

    drop_cols=[]
    keep_cols=[]    
    if len(col1)>0:
        pairs = zip(col1, col2)    
        pairs = sorted(pairs, key = operator.itemgetter(0))
        col1, col2 = zip(*pairs)
        col1_name = upper.index[list(col1)]
        col2_name = upper.index[list(col2)]
    
        corr_value = [ ]
        for col1_, col2_ in zip(col1_name, col2_name):
            if col1_ not in drop_cols:
                drop_cols.append(col2_)
                keep_cols.append(col1_)
                corr_value.append(upper.loc[col1_, col2_])
                
        pd.DataFrame({'col1':keep_cols,'col2':drop_cols,'corr':corr_value}).sort_values(
                    by='col1').to_csv(corr_file, index=False)  
                    
        drop_cols =list(set(drop_cols))
        if inplace:
            df.drop(columns= drop_cols, inplace=True)
           
    return drop_cols
    
def drop_invalid_cols(df,corr_file,corr_threshold=0.99,null_percent=0.95):
    df = df.select_dtypes(include=[np.number])
    null_cols = drop_null_columns(df, null_percent=null_percent)
    df = df.drop(columns=null_cols)

    df = df.fillna(df.mean())
    corr_cols = drop_corr_columns(df, corr_file=corr_file, corr_threshold=corr_threshold)
    print('null cols:',null_cols)
    print('corr_cols:',corr_cols)

    return list(set(null_cols+corr_cols))
    

def get_logger(name, filename=None, level=logging.DEBUG, mode='w+'):
    logger = logging.getLogger(name=name)
    logger.setLevel(level)

    if filename is not None:
        handler = logging.FileHandler(filename, mode=mode, encoding=None, delay=False)
        logger.addHandler(handler)

    return logger 

def label_smooth(x):
    """
    x in {0,1}
    """
    EPSILON = 0.1
    return x*(x-EPSILON)+(1-x)*EPSILON

