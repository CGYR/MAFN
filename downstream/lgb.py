from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve
from sklearn import metrics
from pip._internal import main as install_cmd
try:
    import lightgbm as lgb
except:
    
    install_cmd(["install", "lightgbm==2.3.1","-i", "https://pypi.tuna.tsinghua.edu.cn/simple"])
    import lightgbm as lgb

from tqdm import tqdm
import os
import logging
import json
import yaml
import sys
import glob
import gc
import argparse
import shutil
import numpy as np
import pandas as pd
from pathlib import Path

from datetime import datetime
from scipy import sparse,stats
from collections import Counter,OrderedDict
import multiprocessing

from sklearn.model_selection import train_test_split
from utils import *
from data_process import *

import random
import torch

#1. 修改log文件参数，训练，验证，预测文件参数
#2. 修改列名参数（一部分在中间），人群包数量参数 
#3. 修改训练集数量级（在中间）
#4. 修正预测集付费特征

### 参数

# 文件参数
logger_dir = '/home/notebook/code/personal/fix3/'
model_type = ''
val_label_pred_file = logger_dir+'val_label_pred.csv'

use_embs = True

seed = 98 # random seed, fix for exp
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


train_data_file = '/down_train.csv'
forecast_data_file = '/down_forecast.csv'

val_data_path = '/down_forecast.csv'
val_result_file = '/game_backflow_hpjy_imei_train_v1.csv'

# pandas 列名参数
last_pay_total = ''
last_pay_cnt = 'last_pay_cnt'
last_4_pay_cnt = 'last_4_pay_cnt'
last_4_pay_total = ''

# 人群包相关
target_user_num = 10000000

# 修正预测集付费特征
new_pay_file = '/game_val.csv'




def encode_category_feature(df, include=['object','category']):
    """
    df modified in place
    """
    cat_cols = df.select_dtypes(include=include).columns.tolist()
#    cat_cols = list(set(cat_cols + ['age','income_level']))
    cat_cols = [col for col in cat_cols if col in df.columns]
    encode_maps = {}
    for col in cat_cols:
        # if col in ('age','income_level'):
        #     df[col] = df[col].fillna(-999)
        #     df[col] = df[col].astype(int).astype(str)
        # elif col=='model_level_2':
        #     df[col] = df[col].fillna('-999')
        #     df[col] = df[col].astype(str).str.lower().map(convert_model_level_2)
        # else:
        df[col] = df[col].fillna('-999')
        df[col] = df[col].astype(str).str.lower()
        encode_maps[col] = {v:i for i,v in enumerate(df[col].unique())}

    for col in cat_cols:
        df[col] = df[col].map(encode_maps[col])
        df[col] = df[col].astype('category')

    return encode_maps

def lgb_core(xtrain, ytrain, xval, yval, cats_cols=None):
    params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': ['binary_logloss','auc'],
        'num_leaves':53,
        'max_depth':7,
        'learning_rate': 0.02,
        'feature_fraction': 0.8,
        'num_threads':8,
        'bagging_fraction': 0.9,
        'bagging_freq': 5,
        'lambda_l2':6.0,
        'random_state':seed,
        #'min_data_in_leaf':50
        'min_gain_to_split':0.1
    }

    # # === 特征操作
    # xtrain = xtrain.replace(0, np.nan)
    # xval = xval.replace(0, np.nan)

    lgb_train = lgb.Dataset(xtrain.values, ytrain, categorical_feature=cats_cols)
    lgb_eval =  lgb.Dataset(xval.values, yval, categorical_feature=cats_cols,  reference=lgb_train)
    
    lgb_model = lgb.train(params,
                    lgb_train,
                    num_boost_round=3000,
                    # valid_sets=[lgb_train,lgb_eval],
                    valid_sets=[lgb_eval],
                    early_stopping_rounds=50,
                    verbose_eval = 50,
                    feature_name = xtrain.columns.tolist()
                   )
    return lgb_model

def train_lgb_new(train_set, drop_cols, val_set, logger=None, cat_cols=None,res_dir=None):
    drop_cols = [col for col in drop_cols if col in train_set.columns]
    cat_cols = [col for col in cat_cols if col in train_set.columns and col not in drop_cols]

    xtrain = train_set.drop(columns=drop_cols)
    ytrain = train_set['label']

    xval = val_set.drop(columns=drop_cols)
    yval = val_set['label']
    ytest = yval
    
    model = lgb_core(xtrain, ytrain, xval, yval, cats_cols=cat_cols)

    logger.info('val_label.value_counts = {}'.format(val_set.label.value_counts()))
    
    
    ypred = model.predict(xval, num_iteration=model.best_iteration)

    fpr, tpr, thresholds = roc_curve(ytest, ypred)
    val_label_pred = pd.DataFrame()
    val_label_pred['label'] = ytest
    val_label_pred['pred'] = ypred
    val_label_pred.to_csv(val_label_pred_file,index=False)


    gmeans = np.sqrt(tpr * (1-fpr))
    ix = np.argmax(gmeans)


    t = thresholds[ix]
    y_pred_result = [int(i >= t) for i in ypred]
    print(classification_report(ytest, y_pred_result))

    results =get_auc_ks(ypred, ytest)
    print('validation_set performance={}'.format(results).center(100,'-'))
    logger.info('validation_set performance={}'.format(results).center(100,'-'))
    if res_dir is not None:
        calc_threshold_vs_depth(ytest, ypred, stats_file=os.path.join(res_dir,'depth_stats.csv'))  

        logger.info('eval performance={}'.format(results).center(100,'-'))

    return model,results


def calc_feature_importance(model, imp_file=None,imp_type='gain'):
    importance = model.feature_importance(importance_type=imp_type)
    feature_name = model.feature_name()
    feature_importance = pd.DataFrame({'feature_name':feature_name,'importance':importance})
    feature_importance= feature_importance.sort_values('importance', ascending=False)
    feature_importance['importance'] = feature_importance['importance']/feature_importance['importance'].sum()
    if imp_file is not None:
        feature_importance.to_csv(imp_file,index=False)

    return feature_importance

def save_model(model, encode_maps,  res_dir,backup_dir):
    model.save_model(os.path.join(res_dir,'lgb.model'))
    json.dump(encode_maps,open(os.path.join(res_dir,'encode_maps.json'),'w+'), indent=2, ensure_ascii=True) 

    dic = {}
    dic['feature'] = model.feature_name()
    json.dump(dic,open(os.path.join(res_dir, 'topfeatures.json'),'w+'),indent=2)

    if os.path.exists(backup_dir):
        shutil.rmtree(backup_dir)
    shutil.copytree(res_dir, backup_dir)


if __name__ =='__main__':

    ### log 适配
    if os.path.exists(logger_dir):
        shutil.rmtree(logger_dir)
    os.makedirs(logger_dir, mode=0o777)
    logger = get_logger(name=model_type, filename=os.path.join(logger_dir,'log.txt'))
    
    ### 获取和选择数据
    data = pd.read_csv(train_data_file,sep='\t')
    #data = data[data['dayno']==20220623]

    # NOTE 添加embs
    if not use_embs:
        data = data.drop(columns=[str(i) for i in range(64)])
    
    logger.info('总数据 data data.shape={}'.format(data.shape))
    train_set = data[data['dayno']<20230307]
    train_set = train_set.sample(frac=1).reset_index(drop=True)
    val_set = data[(data['dayno']==20230307)]
    val_set = val_set.drop_duplicates(subset='imei', keep='first')    
    
    del data
    logger.info('train_set train_set data.shape={}'.format(train_set.shape))
    logger.info('val_set val_set data.shape={}'.format(val_set.shape))

    #train_set, val_set = train_test_split(data,train_size=0.9, test_size=0.1, random_state=0)

    # 删除特征，屏幕特征有误，这里先进行全部删除  +外加付费标签+dayno
    drop_cols = ['imei','label','label_pay'] \
                +['max_screen_off_cnt','min_screen_off_cnt','avg_screen_off_cnt','max_screen_off_duration','min_screen_off_duration',
                    'avg_screen_off_duration','max_screen_on_cnt','min_screen_on_cnt','avg_screen_on_cnt','max_screen_on_duration',
                    'min_screen_on_duration','avg_screen_on_duration','dayno'] \
                + [last_pay_total, last_pay_cnt,'last_pay_cnt','last_pay_avg', 'last_pay_total','last_1_pay_total','last_1_pay_avg', 'last_3_pay_total', last_4_pay_total, 'last_5_pay_total', 'last_7_pay_total', 
                'last_10_pay_total', 'last_3_pay_cnt', last_4_pay_cnt, 'last_5_pay_cnt', 'last_7_pay_cnt', 'last_10_pay_cnt', 'last_3_pay_avg', 
                'last_4_pay_avg', 'last_5_pay_avg', 'last_7_pay_avg', 'last_10_pay_avg','ly_backflow_cnt','last_14_pay_total','last_14_pay_cnt','last_14_pay_avg'
                ]\
                + ['fly_qpyx_7_uninstall_game_cnt', 'fly_other_3_uninstall_game_cnt', 'fly_yywd_30_act_duration', 'ly_dzmx_90_install_week_cnt', 'fly_yywd_3_install_game_cnt', 'fly_jycl_14_act_duration', 'fly_dzmx_14_install_cnt', 'fly_sjyx_30_uninstall_cnt', 'fly_yywd_30_uninstall_game_cnt', 'other_45_pay_day_cnt', 'fly_qpyx_90_act_duration', 'ly_tyjs_90_uninstall_week_cnt', 'fly_tyjs_30_install_week_cnt', 'ly_other_30_uninstall_day_cnt', 'fly_xxyz_30_act_duration', 'other_7_times_cnt', 'fly_qpyx_7_uninstall_cnt', 'fly_sjyx_3_uninstall_day_cnt', 'fly_tyjs_3_install_cnt', 'fly_sjyx_3_uninstall_game_cnt', 'fly_xxyz_14_install_week_cnt', 'dzmx_90_act_week_cnt', 'fly_yywd_90_uninstall_game_cnt', 'ly_jsby_90_uninstall_week_cnt', 'fly_yywd_60_install_day_cnt', 'ly_qpyx_30_uninstall_week_cnt', 'fly_xxyz_60_install_game_cnt', 'fly_dzmx_30_uninstall_week_cnt', 'ly_sjyx_30_install_week_cnt', 'fly_dzmx_14_uninstall_week_cnt', 'yywd_60_act_week_cnt', 'ly_tyjs_45_uninstall_week_cnt', 'fly_tyjs_3_uninstall_game_cnt', 'fly_other_30_install_week_cnt', 'fly_yywd_14_install_day_cnt', 'ly_other_90_install_cnt', 'fly_tyjs_14_install_day_cnt', 'fly_other_90_install_game_cnt', 'fly_jycl_30_uninstall_game_cnt', 'fly_jycl_3_uninstall_cnt', 'fly_sjyx_45_uninstall_week_cnt', 'fly_dzmx_30_uninstall_cnt', 'fly_dzmx_30_uninstall_game_cnt', 'fly_yywd_30_uninstall_cnt', 'fly_xxyz_30_uninstall_game_cnt', 'ly_tyjs_14_uninstall_week_cnt', 'fly_sjyx_3_install_day_cnt', 'ly_jsby_45_install_week_cnt', 'fly_sjyx_45_install_cnt', 'fly_dzmx_14_uninstall_game_cnt', 'other_14_act_game_cnt', 'ly_jycl_14_install_week_cnt', 'fly_sjyx_45_uninstall_cnt', 'fly_xxyz_90_uninstall_day_cnt', 'ly_other_30_act_duration', 'fly_jycl_200_act_duration', 'fly_other_14_uninstall_cnt', 'dzmx_30_act_week_cnt', 'other_90_pay_day_cnt', 'fly_xxyz_7_act_duration', 'fly_xxyz_90_install_day_cnt', 'dzmx_14_act_week_cnt', 'ly_yywd_90_install_week_cnt', 'fly_yywd_7_install_day_cnt', 'fly_jsby_60_act_duration', 'fly_tyjs_14_uninstall_cnt', 'ly_xxyz_60_uninstall_week_cnt', 'ly_other_7_act_duration', 'fly_sjyx_45_uninstall_day_cnt', 'fly_jsby_45_install_week_cnt', 'fly_qpyx_14_uninstall_cnt', 'ly_jsby_30_install_week_cnt', 'fly_sjyx_90_act_duration', 'fly_qpyx_60_install_game_cnt', 'fly_dzmx_3_install_cnt', 'fly_yywd_14_uninstall_cnt', 'fly_jycl_14_uninstall_cnt', 'fly_other_30_uninstall_week_cnt', 'ly_other_30_install_cnt', 'other_200_pay_cnt', 'fly_qpyx_45_act_duration', 'fly_other_45_install_day_cnt', 'fly_xxyz_3_install_game_cnt', 'ly_sjyx_45_install_week_cnt', 'fly_jycl_90_uninstall_cnt', 'fly_dzmx_7_uninstall_cnt', 'fly_jycl_60_install_day_cnt', 'fly_jsby_90_uninstall_game_cnt', 'fly_tyjs_90_uninstall_week_cnt', 'fly_xxyz_60_uninstall_day_cnt', 'fly_jycl_45_install_game_cnt', 'fly_other_14_uninstall_game_cnt', 'ly_other_45_install_game_cnt', 'fly_yywd_14_uninstall_week_cnt', 'other_60_pay_cnt', 'fly_yywd_60_uninstall_cnt', 'fly_tyjs_3_act_duration', 'fly_dzmx_14_uninstall_cnt', 'ly_other_14_uninstall_cnt', 'fly_sjyx_90_uninstall_cnt', 'yywd_90_act_week_cnt', 'fly_other_90_uninstall_day_cnt', 'fly_tyjs_14_uninstall_day_cnt', 'fly_jsby_14_uninstall_cnt', 'ly_other_7_install_game_cnt', 'fly_xxyz_45_uninstall_game_cnt', 'xxyz_14_act_week_cnt', 'fly_jsby_30_uninstall_week_cnt', 'other_90_act_day_cnt', 'fly_jycl_7_act_duration', 'fly_xxyz_90_install_game_cnt', 'other_30_act_week_cnt', 'fly_jycl_45_install_day_cnt', 'fly_dzmx_7_uninstall_game_cnt', 'fly_other_200_act_duration', 'fly_qpyx_45_uninstall_day_cnt', 'fly_other_7_act_duration', 'fly_yywd_90_install_cnt', 'fly_jycl_60_uninstall_week_cnt', 'fly_xxyz_60_act_duration', 'fly_tyjs_14_act_duration', 'fly_qpyx_45_install_cnt', 'fly_xxyz_60_install_day_cnt', 'fly_sjyx_45_install_game_cnt', 'jycl_30_act_week_cnt', 'fly_other_14_install_game_cnt', 'fly_dzmx_90_install_game_cnt', 'qpyx_45_act_week_cnt', 'ly_xxyz_60_install_week_cnt', 'fly_tyjs_90_install_game_cnt', 'fly_jsby_7_act_duration', 'fly_qpyx_14_install_cnt', 'fly_tyjs_90_install_week_cnt', 'other_90_times_cnt', 'fly_jycl_90_install_cnt', 'other_200_pay_all_cnt', 'fly_other_90_uninstall_game_cnt', 'fly_qpyx_60_uninstall_day_cnt', 'fly_jsby_14_install_cnt', 'fly_qpyx_60_install_week_cnt', 'fly_jycl_7_uninstall_game_cnt', 'other_60_act_week_cnt', 'fly_other_30_act_duration', 'fly_dzmx_3_uninstall_game_cnt', 'ly_tyjs_30_install_week_cnt', 'fly_tyjs_90_uninstall_game_cnt', 'fly_qpyx_90_install_game_cnt', 'other_7_pay_cnt', 'fly_sjyx_30_install_game_cnt', 'fly_tyjs_14_uninstall_week_cnt', 'fly_other_30_install_game_cnt', 'ly_jycl_60_uninstall_week_cnt', 'fly_tyjs_45_install_day_cnt', 'fly_yywd_14_uninstall_game_cnt', 'ly_xxyz_90_uninstall_week_cnt', 'ly_other_60_install_day_cnt', 'fly_sjyx_90_install_week_cnt', 'fly_yywd_3_uninstall_game_cnt', 'fly_tyjs_45_act_duration', 'fly_xxyz_60_uninstall_week_cnt', 'fly_jycl_60_uninstall_day_cnt', 'ly_jycl_60_install_week_cnt', 'fly_jycl_3_uninstall_game_cnt', 'fly_xxyz_45_install_game_cnt', 'ly_tyjs_90_install_week_cnt', 'fly_tyjs_45_uninstall_day_cnt', 'ly_yywd_30_install_week_cnt', 'fly_qpyx_60_install_day_cnt', 'fly_tyjs_30_install_game_cnt', 'fly_qpyx_30_install_week_cnt', 'fly_jycl_7_install_day_cnt', 'fly_xxyz_45_install_day_cnt', 'fly_tyjs_14_install_game_cnt', 'fly_dzmx_14_act_duration', 'fly_dzmx_7_install_game_cnt', 'fly_xxyz_14_uninstall_week_cnt', 'fly_xxyz_60_install_cnt', 'fly_jsby_45_install_day_cnt', 'fly_qpyx_30_install_day_cnt', 'fly_sjyx_14_uninstall_game_cnt', 'fly_sjyx_90_uninstall_week_cnt', 'fly_yywd_30_install_day_cnt', 'ly_qpyx_30_install_week_cnt', 'fly_sjyx_45_act_duration', 'fly_yywd_14_install_game_cnt', 'tyjs_45_act_week_cnt', 'fly_dzmx_60_uninstall_day_cnt', 'fly_qpyx_14_install_game_cnt', 'fly_yywd_60_uninstall_day_cnt', 'fly_xxyz_3_uninstall_cnt', 'ly_jycl_30_install_week_cnt', 'fly_qpyx_14_install_week_cnt', 'other_3_act_game_cnt', 'fly_xxyz_3_uninstall_day_cnt', 'qpyx_60_act_week_cnt', 'fly_qpyx_3_uninstall_day_cnt', 'ly_yywd_30_uninstall_week_cnt', 'ly_dzmx_30_install_week_cnt', 'fly_dzmx_14_install_game_cnt', 'fly_tyjs_3_uninstall_day_cnt', 'fly_other_45_uninstall_day_cnt', 'ly_other_60_uninstall_day_cnt', 'fly_jycl_14_install_day_cnt', 'fly_xxyz_14_uninstall_day_cnt', 'fly_sjyx_30_uninstall_day_cnt', 'ly_other_7_uninstall_day_cnt', 'other_200_pay_day_cnt', 'fly_jsby_30_uninstall_game_cnt', 'fly_dzmx_14_install_week_cnt', 'other_60_act_day_cnt', 'fly_qpyx_60_uninstall_game_cnt', 'fly_tyjs_45_uninstall_game_cnt', 
                'fly_xxyz_45_act_duration', 'other_90_pay_all_cnt', 'fly_yywd_60_install_game_cnt', 'ly_xxyz_90_install_week_cnt', 'other_60_act_game_cnt', 'fly_dzmx_3_uninstall_day_cnt', 'fly_qpyx_90_install_day_cnt', 'other_45_pay_all_cnt', 'ly_other_30_uninstall_cnt', 'fly_sjyx_30_uninstall_week_cnt', 'ly_other_14_uninstall_game_cnt', 'fly_jsby_14_install_game_cnt', 'fly_sjyx_60_uninstall_week_cnt', 'fly_yywd_3_uninstall_cnt', 'other_3_pay_all_cnt', 'ly_other_30_uninstall_game_cnt', 'ly_jsby_60_uninstall_week_cnt', 'fly_sjyx_7_uninstall_cnt', 'fly_other_90_uninstall_cnt', 'fly_other_14_uninstall_week_cnt', 'fly_xxyz_14_install_cnt', 'ly_other_90_uninstall_day_cnt', 'fly_jsby_30_install_cnt', 'fly_tyjs_3_install_game_cnt', 'fly_xxyz_90_uninstall_cnt', 'fly_jsby_90_install_week_cnt', 'fly_jycl_7_install_game_cnt', 'fly_jsby_90_act_duration', 'fly_dzmx_90_install_day_cnt', 'fly_sjyx_3_install_game_cnt', 'fly_dzmx_45_install_week_cnt', 'fly_sjyx_14_install_game_cnt', 'tyjs_90_act_week_cnt', 'ly_dzmx_60_install_week_cnt', 'other_90_act_week_cnt', 'ly_sjyx_60_uninstall_week_cnt', 'ly_tyjs_14_install_week_cnt', 'fly_dzmx_7_install_day_cnt', 'fly_tyjs_3_uninstall_cnt', 'other_3_pay_day_cnt', 'fly_qpyx_60_uninstall_cnt', 'fly_sjyx_45_uninstall_game_cnt', 'fly_jsby_60_uninstall_day_cnt', 'fly_dzmx_45_act_duration', 'fly_tyjs_7_uninstall_cnt', 'other_3_act_day_cnt', 'ly_dzmx_14_uninstall_week_cnt', 'fly_sjyx_60_uninstall_game_cnt', 'yywd_45_act_week_cnt', 'other_14_act_day_cnt', 'fly_other_45_install_cnt', 'fly_tyjs_30_uninstall_day_cnt', 'other_14_pay_cnt', 'fly_dzmx_45_uninstall_day_cnt', 'sjyz_45_act_week_cnt', 'ly_other_7_uninstall_game_cnt', 'ly_other_90_install_week_cnt', 'jycl_90_act_week_cnt', 'fly_qpyx_90_uninstall_day_cnt', 'fly_dzmx_60_install_week_cnt', 'dzmx_45_act_week_cnt', 'ly_sjyx_90_uninstall_week_cnt', 'fly_qpyx_14_act_duration', 'fly_tyjs_45_install_cnt', 'ly_other_3_uninstall_day_cnt', 'fly_tyjs_45_install_game_cnt', 'sjyz_60_act_week_cnt', 'fly_jsby_3_install_cnt', 'fly_qpyx_30_uninstall_game_cnt', 'fly_jycl_3_install_cnt', 'fly_other_45_uninstall_cnt', 'ly_yywd_60_uninstall_week_cnt', 'ly_other_14_uninstall_week_cnt', 'fly_dzmx_60_uninstall_week_cnt', 'fly_yywd_45_install_week_cnt', 'fly_jycl_7_uninstall_cnt', 'fly_other_60_install_week_cnt', 'fly_dzmx_7_uninstall_day_cnt', 'fly_jsby_3_uninstall_day_cnt', 'other_7_pay_all_cnt', 'fly_dzmx_90_uninstall_week_cnt', 'fly_other_60_uninstall_week_cnt', 'other_45_act_game_cnt', 'fly_yywd_60_install_cnt', 'fly_dzmx_60_uninstall_game_cnt', 'fly_qpyx_3_uninstall_game_cnt', 'fly_other_60_install_day_cnt', 'fly_tyjs_60_install_week_cnt', 'ly_qpyx_14_install_week_cnt', 'fly_sjyx_14_install_cnt', 'fly_jycl_3_act_duration', 'fly_sjyx_3_act_duration', 'fly_sjyx_7_uninstall_game_cnt', 'fly_other_14_uninstall_day_cnt', 'fly_yywd_90_uninstall_day_cnt', 'other_30_act_day_cnt', 'fly_xxyz_90_install_cnt', 'yywd_14_act_week_cnt', 'yywd_30_act_week_cnt', 'fly_qpyx_60_act_duration', 'fly_qpyx_60_install_cnt', 'fly_qpyx_3_uninstall_cnt', 'fly_qpyx_7_install_game_cnt', 'sjyz_14_act_week_cnt', 'other_14_pay_day_cnt', 'ly_other_45_act_duration', 'fly_tyjs_7_act_duration', 'fly_tyjs_60_install_cnt', 'ly_other_90_uninstall_game_cnt', 'fly_xxyz_45_uninstall_cnt', 'fly_tyjs_60_uninstall_day_cnt', 'other_60_times_cnt', 'ly_other_3_uninstall_cnt', 'other_30_pay_day_cnt', 'ly_other_14_install_day_cnt', 'fly_jsby_3_act_duration', 'fly_other_90_uninstall_week_cnt', 'ly_jycl_45_install_week_cnt', 'fly_jsby_14_uninstall_game_cnt', 'fly_tyjs_7_install_cnt', 'fly_xxyz_45_install_week_cnt', 'fly_jsby_45_uninstall_game_cnt',
                'fly_xxyz_60_uninstall_cnt', 'fly_qpyx_7_install_cnt', 'ly_other_45_install_week_cnt', 'fly_jsby_90_uninstall_week_cnt', 'ly_other_45_uninstall_week_cnt', 'fly_xxyz_14_uninstall_game_cnt', 'fly_other_60_uninstall_day_cnt', 'fly_tyjs_60_install_game_cnt', 'fly_jsby_14_install_day_cnt', 'fly_jsby_3_install_game_cnt', 'fly_tyjs_200_act_duration', 'ly_dzmx_90_uninstall_week_cnt', 'ly_other_14_install_cnt', 'ly_jsby_45_uninstall_week_cnt', 'fly_jsby_60_uninstall_cnt', 'fly_sjyx_60_install_week_cnt', 'fly_dzmx_3_install_game_cnt', 'fly_qpyx_7_uninstall_day_cnt', 'fly_qpyx_60_uninstall_week_cnt', 'other_3_pay_cnt', 'fly_yywd_45_install_cnt', 'fly_sjyx_7_uninstall_day_cnt', 'other_30_pay_cnt', 'fly_jycl_3_uninstall_day_cnt', 'ly_jycl_90_install_week_cnt', 'other_60_pay_all_cnt', 'fly_jsby_30_install_week_cnt', 'ly_qpyx_90_install_week_cnt', 'fly_dzmx_200_act_duration', 'fly_sjyx_14_install_day_cnt', 'fly_other_3_install_game_cnt', 'fly_tyjs_30_install_day_cnt', 'fly_qpyx_7_act_duration', 'other_3_times_cnt', 'fly_jsby_14_act_duration', 'fly_xxyz_3_uninstall_game_cnt', 'fly_yywd_45_uninstall_game_cnt', 'fly_jsby_45_install_game_cnt', 'fly_yywd_14_install_cnt', 'ly_sjyx_14_uninstall_week_cnt', 'fly_jycl_90_install_day_cnt', 'fly_xxyz_30_install_week_cnt', 'fly_dzmx_60_install_game_cnt', 'fly_xxyz_7_install_day_cnt', 'fly_jsby_14_install_week_cnt', 'ly_other_45_uninstall_day_cnt', 'ly_sjyx_60_install_week_cnt', 'ly_other_90_act_duration', 'qpyx_90_act_week_cnt', 'fly_sjyx_45_install_day_cnt', 'fly_yywd_7_act_duration', 'ly_other_60_uninstall_week_cnt', 'fly_dzmx_14_uninstall_day_cnt', 'fly_yywd_45_install_game_cnt', 'ly_jycl_90_uninstall_week_cnt', 'fly_dzmx_60_uninstall_cnt', 'other_30_pay_all_cnt', 'jsby_45_act_week_cnt', 'ly_sjyx_30_uninstall_week_cnt', 'fly_other_45_uninstall_week_cnt', 'ly_qpyx_45_uninstall_week_cnt', 'fly_jycl_45_act_duration', 'fly_other_60_uninstall_cnt', 'ly_qpyx_45_install_week_cnt', 'ly_other_45_uninstall_cnt', 'fly_sjyx_60_install_cnt', 'fly_jsby_7_uninstall_day_cnt', 'fly_xxyz_90_install_week_cnt', 'fly_xxyz_3_act_duration', 'fly_sjyx_14_uninstall_cnt', 'ly_other_90_uninstall_cnt', 'fly_jycl_30_install_cnt', 'fly_other_45_uninstall_game_cnt', 'ly_other_14_install_game_cnt', 'ly_jycl_45_uninstall_week_cnt', 'fly_qpyx_14_uninstall_week_cnt', 'fly_qpyx_90_install_cnt', 'fly_dzmx_90_uninstall_game_cnt', 'fly_jsby_30_uninstall_cnt', 'fly_jycl_45_install_cnt', 'fly_tyjs_60_uninstall_week_cnt', 'fly_qpyx_3_install_cnt', 'fly_qpyx_90_uninstall_week_cnt', 'fly_jycl_60_uninstall_game_cnt', 'fly_other_30_install_day_cnt', 'fly_qpyx_7_install_day_cnt', 'fly_yywd_60_act_duration', 'ly_jycl_14_uninstall_week_cnt', 'fly_xxyz_60_install_week_cnt', 'fly_jsby_7_install_day_cnt', 'fly_jsby_90_install_cnt', 'fly_xxyz_90_uninstall_week_cnt', 'fly_dzmx_60_install_cnt', 'ly_qpyx_14_uninstall_week_cnt', 'fly_tyjs_30_uninstall_week_cnt', 'ly_yywd_14_uninstall_week_cnt', 'fly_dzmx_30_install_week_cnt', 'fly_yywd_45_install_day_cnt', 'ly_other_7_uninstall_cnt', 'fly_sjyx_45_install_week_cnt', 'ly_dzmx_45_uninstall_week_cnt', 'fly_qpyx_30_uninstall_cnt', 'fly_other_30_uninstall_game_cnt', 'fly_sjyx_90_install_game_cnt', 'fly_other_7_install_cnt', 'fly_other_3_install_cnt', 'ly_other_3_install_cnt', 'fly_jycl_90_act_duration', 'fly_sjyx_30_install_week_cnt', 'ly_xxyz_14_uninstall_week_cnt', 'fly_jsby_90_uninstall_cnt', 'fly_other_3_act_duration', 'other_7_pay_day_cnt', 'ly_jsby_14_uninstall_week_cnt', 'ly_other_3_install_game_cnt', 'fly_jsby_14_uninstall_week_cnt', 'ly_tyjs_60_install_week_cnt', 'fly_dzmx_90_uninstall_day_cnt', 'jsby_60_act_week_cnt', 'ly_xxyz_45_uninstall_week_cnt', 'fly_jycl_90_install_week_cnt', 'fly_qpyx_45_install_day_cnt', 'fly_qpyx_30_install_game_cnt', 'other_90_pay_cnt', 'fly_jycl_90_uninstall_day_cnt', 'fly_jsby_60_install_game_cnt', 'qpyx_30_act_week_cnt', 'ly_xxyz_30_install_week_cnt', 'fly_yywd_3_uninstall_day_cnt', 'fly_yywd_30_install_week_cnt', 'xxyz_60_act_week_cnt', 'fly_yywd_7_uninstall_game_cnt', 'fly_qpyx_30_install_cnt', 'fly_other_3_uninstall_cnt', 'fly_yywd_90_install_game_cnt', 'fly_jsby_60_install_week_cnt', 'other_14_times_cnt', 'xxyz_90_act_week_cnt', 'fly_tyjs_90_uninstall_day_cnt', 'fly_xxyz_3_install_day_cnt', 'fly_dzmx_45_uninstall_game_cnt', 'fly_jycl_14_uninstall_week_cnt', 'fly_xxyz_14_act_duration', 'fly_dzmx_45_install_game_cnt', 'sjyz_90_act_week_cnt', 'fly_xxyz_14_install_day_cnt', 'ly_tyjs_45_install_week_cnt', 'other_60_pay_day_cnt', 'fly_qpyx_45_uninstall_game_cnt', 'fly_jycl_45_uninstall_day_cnt', 'other_45_times_cnt', 'fly_yywd_3_install_day_cnt', 'ly_other_3_install_day_cnt', 'fly_other_7_uninstall_game_cnt', 'fly_other_60_install_game_cnt', 'tyjs_30_act_week_cnt', 'ly_other_90_uninstall_week_cnt', 'fly_jycl_90_install_game_cnt', 'fly_sjyx_14_install_week_cnt', 'ly_xxyz_14_install_week_cnt', 'fly_tyjs_45_uninstall_week_cnt', 'other_7_act_day_cnt', 'fly_yywd_45_uninstall_day_cnt', 'fly_sjyx_14_act_duration', 'fly_yywd_90_install_day_cnt', 'fly_yywd_45_uninstall_week_cnt', 'ly_yywd_14_install_week_cnt', 'fly_qpyx_3_act_duration', 'fly_sjyx_90_uninstall_game_cnt', 'fly_other_45_install_week_cnt', 'fly_sjyx_90_install_cnt', 'fly_xxyz_30_uninstall_week_cnt', 'fly_other_60_install_cnt', 'other_14_act_week_cnt', 'fly_other_60_uninstall_game_cnt', 'fly_dzmx_3_install_day_cnt', 'ly_other_60_act_duration', 'fly_xxyz_90_act_duration', 'fly_sjyx_7_act_duration', 'fly_jycl_45_uninstall_cnt', 'fly_yywd_30_install_cnt', 'ly_other_30_install_game_cnt', 'fly_dzmx_30_install_day_cnt', 'fly_jsby_7_install_cnt', 'jycl_60_act_week_cnt', 'fly_qpyx_45_uninstall_cnt', 'fly_tyjs_3_install_day_cnt', 'fly_other_3_uninstall_day_cnt', 'fly_tyjs_45_uninstall_cnt', 'fly_xxyz_90_uninstall_game_cnt', 'fly_jycl_30_uninstall_cnt', 'fly_tyjs_90_install_day_cnt', 'fly_other_30_uninstall_day_cnt', 'fly_jsby_14_uninstall_day_cnt', 'fly_other_60_act_duration', 'fly_xxyz_30_uninstall_cnt', 'fly_qpyx_3_install_day_cnt', 'fly_sjyx_90_install_day_cnt', 'fly_sjyx_60_act_duration', 'ly_jsby_30_uninstall_week_cnt', 'ly_qpyx_60_uninstall_week_cnt', 'ly_other_60_install_cnt', 'ly_other_30_uninstall_week_cnt', 'fly_yywd_45_uninstall_cnt', 'ly_xxyz_30_uninstall_week_cnt', 'jsby_14_act_week_cnt', 'fly_jsby_45_act_duration', 'ly_other_60_install_game_cnt', 'ly_yywd_90_uninstall_week_cnt', 'sjyz_30_act_week_cnt', 'ly_yywd_45_uninstall_week_cnt', 'ly_yywd_45_install_week_cnt', 'fly_sjyx_30_uninstall_game_cnt',
                'fly_jycl_45_install_week_cnt', 'fly_jycl_30_act_duration', 'fly_jsby_45_uninstall_cnt', 'fly_sjyx_3_install_cnt', 'fly_sjyx_14_uninstall_week_cnt', 'fly_qpyx_14_uninstall_game_cnt', 'fly_other_7_uninstall_cnt', 'fly_dzmx_60_act_duration', 'fly_jsby_3_install_day_cnt', 'fly_other_7_install_game_cnt', 'fly_dzmx_3_uninstall_cnt', 'fly_dzmx_90_act_duration', 'fly_xxyz_45_uninstall_week_cnt', 'fly_tyjs_7_install_game_cnt', 'fly_other_3_install_day_cnt', 'ly_other_3_act_duration', 'fly_jsby_30_uninstall_day_cnt', 'fly_jsby_60_uninstall_week_cnt', 'fly_sjyx_30_install_day_cnt', 'tyjs_14_act_week_cnt', 'fly_yywd_7_uninstall_day_cnt', 'fly_other_90_install_day_cnt', 'jycl_14_act_week_cnt', 'fly_jycl_7_install_cnt', 'fly_jycl_30_uninstall_week_cnt', 'fly_yywd_7_uninstall_cnt', 'ly_other_60_uninstall_game_cnt', 'other_30_times_cnt', 'ly_xxyz_45_install_week_cnt', 'fly_other_45_act_duration', 'fly_xxyz_30_install_cnt', 'fly_jsby_30_act_duration', 'fly_jycl_14_install_cnt', 'fly_dzmx_45_uninstall_cnt', 'other_14_pay_all_cnt', 'fly_jsby_30_install_game_cnt', 'fly_yywd_90_act_duration', 'ly_jycl_30_uninstall_week_cnt', 'ly_other_90_install_day_cnt', 'other_7_act_game_cnt', 'fly_jsby_60_install_cnt', 'ly_other_3_uninstall_game_cnt', 'xxyz_45_act_week_cnt', 'fly_jycl_14_install_game_cnt', 'fly_jycl_30_install_day_cnt', 'fly_sjyx_200_act_duration', 'other_45_pay_cnt', 'fly_other_90_install_week_cnt', 'fly_jycl_60_install_cnt', 'fly_yywd_7_install_cnt', 'fly_tyjs_30_install_cnt', 'ly_dzmx_60_uninstall_week_cnt', 'fly_yywd_14_uninstall_day_cnt', 'fly_other_7_uninstall_day_cnt', 'ly_dzmx_45_install_week_cnt', 'fly_sjyx_60_install_game_cnt', 'fly_yywd_90_uninstall_week_cnt', 'fly_sjyx_7_install_cnt', 'ly_dzmx_14_install_week_cnt', 'fly_other_90_act_duration', 'ly_other_60_install_week_cnt', 'fly_jsby_3_uninstall_cnt', 'fly_jsby_90_install_game_cnt', 'fly_other_90_install_cnt', 'ly_other_7_install_day_cnt', 'fly_other_14_install_cnt', 'fly_jycl_7_uninstall_day_cnt', 'jycl_45_act_week_cnt', 'fly_yywd_30_install_game_cnt', 'fly_sjyx_3_uninstall_cnt', 'ly_other_200_act_duration', 'fly_qpyx_30_uninstall_day_cnt', 'other_45_act_week_cnt', 'fly_jsby_90_install_day_cnt', 'fly_jycl_45_uninstall_game_cnt', 'fly_sjyx_30_install_cnt', 'fly_jycl_3_install_game_cnt', 'jsby_90_act_week_cnt', 'tyjs_60_act_week_cnt', 'fly_dzmx_45_uninstall_week_cnt', 'ly_yywd_60_install_week_cnt', 'fly_yywd_90_install_week_cnt', 'fly_yywd_3_act_duration', 'fly_tyjs_90_install_cnt', 'fly_tyjs_7_install_day_cnt', 'jsby_30_act_week_cnt', 'fly_xxyz_60_uninstall_game_cnt', 'fly_other_45_install_game_cnt', 'fly_jycl_60_install_week_cnt', 'fly_jycl_14_uninstall_day_cnt', 'fly_yywd_45_act_duration', 'fly_jycl_90_uninstall_week_cnt', 'fly_xxyz_7_uninstall_game_cnt', 'fly_other_30_uninstall_cnt', 'fly_qpyx_90_install_week_cnt', 'fly_yywd_60_uninstall_game_cnt', 'fly_xxyz_30_uninstall_day_cnt', 'fly_dzmx_90_uninstall_cnt', 'fly_jsby_60_uninstall_game_cnt', 'fly_xxyz_7_install_game_cnt', 'fly_jycl_30_install_game_cnt', 'fly_jycl_60_act_duration', 'fly_tyjs_60_act_duration', 'fly_xxyz_7_install_cnt', 'other_45_act_day_cnt', 'fly_tyjs_14_install_week_cnt', 'fly_sjyx_7_install_game_cnt', 'fly_jsby_200_act_duration', 'fly_sjyx_60_install_day_cnt', 'fly_tyjs_14_uninstall_game_cnt', 'fly_sjyx_60_uninstall_day_cnt', 'fly_dzmx_7_install_cnt', 'fly_sjyx_60_uninstall_cnt', 'fly_xxyz_30_install_game_cnt', 'fly_jsby_30_install_day_cnt', 'fly_tyjs_7_uninstall_day_cnt', 'fly_jycl_45_uninstall_week_cnt', 'fly_yywd_90_uninstall_cnt', 'fly_dzmx_30_uninstall_day_cnt', 'fly_qpyx_45_install_game_cnt',
                    'fly_xxyz_14_uninstall_cnt', 'fly_xxyz_7_uninstall_cnt', 'fly_jycl_60_install_game_cnt', 'fly_jsby_3_uninstall_game_cnt', 'fly_dzmx_30_install_cnt', 'fly_yywd_200_act_duration', 'fly_jsby_7_uninstall_cnt', 'fly_yywd_30_uninstall_day_cnt', 'fly_qpyx_90_uninstall_cnt', 'fly_jsby_7_install_game_cnt', 'fly_tyjs_7_uninstall_game_cnt', 'fly_jycl_30_install_week_cnt', 'ly_other_45_install_cnt', 'ly_other_7_install_cnt', 'fly_yywd_60_uninstall_week_cnt', 'fly_tyjs_30_act_duration', 'fly_jsby_45_uninstall_week_cnt', 'fly_dzmx_90_install_cnt', 'fly_sjyx_90_uninstall_day_cnt', 'fly_yywd_14_act_duration', 'ly_other_45_uninstall_game_cnt', 'fly_yywd_30_uninstall_week_cnt', 'fly_tyjs_90_act_duration', 'dzmx_60_act_week_cnt', 'xxyz_30_act_week_cnt', 'fly_dzmx_7_act_duration', 'fly_qpyx_45_uninstall_week_cnt', 'fly_qpyx_30_uninstall_week_cnt', 'fly_qpyx_45_install_week_cnt', 'fly_jycl_30_uninstall_day_cnt', 'ly_other_60_uninstall_cnt', 'fly_yywd_7_install_game_cnt', 'fly_jsby_45_install_cnt', 'fly_jycl_60_uninstall_cnt', 'fly_xxyz_30_install_day_cnt', 'other_90_act_game_cnt', 'ly_other_14_act_duration', 'fly_sjyx_14_uninstall_day_cnt', 'fly_xxyz_7_uninstall_day_cnt', 'fly_dzmx_90_install_week_cnt', 'fly_other_14_install_week_cnt', 'fly_qpyx_200_act_duration', 'fly_qpyx_3_install_game_cnt', 'fly_yywd_3_install_cnt', 'fly_dzmx_30_install_game_cnt', 'ly_other_14_install_week_cnt', 'fly_dzmx_30_act_duration', 'fly_tyjs_60_uninstall_cnt', 'fly_jsby_60_install_day_cnt', 'fly_jsby_90_uninstall_day_cnt', 'other_30_act_game_cnt', 'fly_tyjs_45_install_week_cnt', 'ly_tyjs_60_uninstall_week_cnt', 'fly_qpyx_90_uninstall_game_cnt', 'ly_other_90_install_game_cnt', 'ly_qpyx_60_install_week_cnt', 'fly_other_14_act_duration', 'fly_dzmx_3_act_duration', 'fly_other_14_install_day_cnt', 'fly_jycl_90_uninstall_game_cnt', 'fly_jycl_14_uninstall_game_cnt', 'fly_xxyz_14_install_game_cnt', 'fly_tyjs_90_uninstall_cnt', 'fly_tyjs_30_uninstall_cnt', 'fly_sjyx_30_act_duration', 'ly_jsby_14_install_week_cnt', 'fly_other_30_install_cnt', 'ly_other_30_install_day_cnt', 'fly_tyjs_30_uninstall_game_cnt', 'ly_other_14_uninstall_day_cnt', 'fly_dzmx_45_install_day_cnt', 'fly_sjyx_7_install_day_cnt', 'fly_tyjs_60_uninstall_game_cnt', 'ly_dzmx_30_uninstall_week_cnt', 'fly_jycl_3_install_day_cnt', 'fly_jycl_14_install_week_cnt', 'fly_jsby_45_uninstall_day_cnt', 'ly_jsby_90_install_week_cnt', 'fly_dzmx_60_install_day_cnt', 'fly_tyjs_14_install_cnt', 'fly_dzmx_45_install_cnt', 'fly_xxyz_45_install_cnt', 'fly_qpyx_14_install_day_cnt', 'fly_yywd_14_install_week_cnt', 'ly_qpyx_90_uninstall_week_cnt', 'fly_dzmx_14_install_day_cnt', 'fly_qpyx_14_uninstall_day_cnt', 'fly_other_7_install_day_cnt', 'fly_xxyz_45_uninstall_day_cnt', 'qpyx_14_act_week_cnt', 'ly_tyjs_30_uninstall_week_cnt', 'ly_sjyx_45_uninstall_week_cnt', 'fly_xxyz_200_act_duration', 'fly_jsby_7_uninstall_game_cnt', 'fly_yywd_60_install_week_cnt', 'fly_qpyx_30_act_duration', 'fly_tyjs_60_install_day_cnt', 'ly_jsby_60_install_week_cnt', 'ly_sjyx_90_install_week_cnt', 'ly_other_45_install_day_cnt', 'ly_other_30_install_week_cnt', 'ly_sjyx_14_install_week_cnt']
          

            


    # 删除全为空的行
    features =  [col for col in train_set.columns if col not in drop_cols]
    # if use_embs:
    #     features += ["imei"]
    train_set_drop = train_set[features]
    logger.info('行特征都缺失的行数占比 {}'.format(len(train_set_drop[train_set_drop.isnull().T.all()])/len(train_set_drop))) 
    data_delete = train_set_drop[train_set_drop.isnull().T.all()]
    logger.info('删除前行数 {}'.format(len(train_set)))
    train_set = train_set.drop(index=data_delete.index)
    logger.info('删除后行数 {}'.format(len(train_set)))
    del data_delete

    features =  [col for col in val_set.columns if col not in drop_cols ]
    # if use_embs:
    #     features += ["imei"]
    val_set_drop = val_set[features]
    logger.info('行特征都缺失的行数占比 {}'.format(len(val_set_drop[val_set_drop.isnull().T.all()])/len(val_set_drop))) 
    data_delete = val_set_drop[val_set_drop.isnull().T.all()]
    logger.info('删除前行数 {}'.format(len(val_set)))
    val_set = val_set.drop(index=data_delete.index)
    logger.info('删除后行数 {}'.format(len(val_set)))
    del data_delete

    ### 特征预处理
    data_porcess = ML_DataPreprocessor(data = train_set)
    train_set = data_porcess.data_preproess()
    # if use_embs:
    #     train_set = pd.merge(train_set, df_emb, how='left', on='imei')
    #     train_set = train_set.drop(columns=["imei"])

    data_porcess = ML_DataPreprocessor(data = val_set)
    val_set = data_porcess.data_preproess()
    # if use_embs:
    #     val_set = pd.merge(val_set, df_emb, how='left', on='imei')
    #     val_set = val_set.drop(columns=["imei"])

    ### 类别特征处理
    for df in [val_set,train_set]:
        encoder_maps = encode_category_feature(df)

    cat_cols = list(encoder_maps.keys()) 

    ### 模型部分,训练,特征重要性等等
    model,results  = train_lgb_new(train_set, drop_cols, val_set,cat_cols = cat_cols,logger=logger)

    # 特征重要性
    calc_feature_importance(model,imp_file=logger_dir+'feature_importance.csv')
    
    # 保存模型
    backup_dir = logger_dir+'/'+'{}'.format(datetime.now().strftime('%Y%m%d_%H'))
    save_model(model,encoder_maps, res_dir=logger_dir,backup_dir=backup_dir)
    
    '''
    # 读取模型
    base_dir= ''
    model = lgb.Booster(model_file=base_dir+'lgb.model')
    encoder_maps = json.load(open(base_dir+'encode_maps.json','r'))
    cat_cols = list(encoder_maps.keys())
    '''

    ''''''
    import matplotlib.pyplot as plt

    
    ### 验证集预测&输出一系列列结果
    test_path = val_data_path
    table = pd.read_table(test_path,sep='\t',chunksize=1000000)
    ans = pd.DataFrame()
    ret = 0
    for df in table:
        ret += 1
        if ret%3==0:
            print('开始处理第{}个1000000行'.format(ret)) 
        
        logger.info('df.shape={}'.format(df.shape))
        print('df.shape={}'.format(df.shape))

        df_test = df
        features=[ col for col in df_test.columns if col not in drop_cols ]
        xtest = df_test[features]
        ypred = []

         # NOTE 添加embs
        if not use_embs:
            xtest = xtest.drop(columns=[str(i) for i in range(64)])
        
        # xtest = xtest.drop(columns=["imei"])


        # logger.info('行特征都缺失的行数占比 {}'.format(len(xtest[xtest.isnull().T.all()])/len(xtest))) 
        # data_delete = xtest[xtest.isnull().T.all()]
        # logger.info('删除前行数 {}'.format(len(xtest)))
        # xtest = xtest.drop(index=data_delete.index)
        # df_test = df_test.drop(index=data_delete.index)
        # logger.info('删除后行数 {}'.format(len(xtest)))
        # del data_delete

        print('特征预处理')
        data_porcess = ML_DataPreprocessor(data = xtest)
        xtest = data_porcess.data_preproess()
        # # NOTE 添加embs
        # if use_embs:
        #     xtest = pd.merge(xtest, df_emb, how='left', on='imei')
        
        # xtest = xtest.drop(columns=["imei"])

        # 类别特征
        for col in cat_cols:
            if col != 'imei':
                xtest[col] = xtest[col].fillna('-999')
                xtest[col] = xtest[col].str.lower()
                xtest[col] = xtest[col].map(encoder_maps[col])

        # # 特征操作
        # xtest = xtest.replace(0, np.nan)

        chunk = 50000
        for i in range(0,len(xtest),chunk):
            tmp = model.predict(xtest[i:i+chunk], num_iteration=model.best_iteration)
            ypred  = np.concatenate((ypred ,tmp))

        result = pd.DataFrame()
        result['imei'] = df_test['imei']

        # 付费相关
       
        # new_pay = pd.read_csv(new_pay_file,sep='\t')
        # new_pay['imei'] = new_pay['imei'].astype('str')
        # result['imei'] = result['imei'].astype('str')
        # result = result.merge(new_pay, how='left', on='imei')


        result['prob'] = ypred

        ans = ans.append(result)
    

    # 排序去重
    logger.info('forecast ans.shape={}'.format(ans.shape))
    ans.sort_values(by='prob',ascending=False,inplace=True)
    ans = ans.drop_duplicates(subset='imei', keep='first')
    logger.info('forecast drop_duplicates ans.shape={}'.format(ans.shape))
    ans['rank'] = ans.prob.rank(method='first',ascending=False)

    logger.info('===================================验证集验证结论==========================================')
    recall_user = []
    recall_pay = []
    
    for target_num_index in tqdm(range(target_user_num//500000)):
        target_num = (target_num_index+1)*500000

        imei_launch = ans.iloc[:target_num, :]

        val_new = pd.read_csv(val_result_file,sep='\t')
        results = imei_launch
        results['imei'] = results['imei'].astype('str')
        val_new['imei'] = val_new['imei'].astype('str')
        ans['imei'] = ans['imei'].astype('str')

        val_new_tmp = results.merge(val_new, how='inner', on='imei', indicator=True)
        val_new_tmp = val_new_tmp.drop_duplicates(subset='imei', keep='first') 
        
        recall_user.append(val_new_tmp.shape[0])
        # recall_pay.append(val_new_tmp[val_new_tmp[last_pay_cnt]>=1].shape[0])

    ans_forecast = ans.merge(val_new, how='inner', on='imei', indicator=True)
    logger.info('总的回流数据 shape={}'.format(val_new.shape))
    logger.info('和预测集交集 ans shape={}'.format(ans_forecast.shape))
    # logger.info('当天预测集所有活跃用户的付费用户数 = {}'.format(ans[ans[last_pay_cnt]>=1].shape))
    # logger.info('当天回流用户和预测集交集所有活跃用户的付费用户数 = {}'.format(ans_forecast[ans_forecast[last_pay_cnt]>=1].shape))

    logger.info('各个分箱召回的 user 活跃数recall_user {}'.format(recall_user))
    # logger.info('各个分箱召回的 user 付费数recall_pay {}'.format(recall_pay))
    logger.info('===================================验证集验证结论  END==========================================')



    # ### 预测集预测
    # test_path = forecast_data_file
    # table = pd.read_table(test_path,sep='\t',chunksize=3000000)
    # ans = pd.DataFrame()
    # ret = 0
    # for df in table:
    #     ret += 1
    #     if ret%3==0:
    #         print('开始处理第{}个3000000行'.format(ret)) 
        
    #     logger.info('df.shape={}'.format(df.shape))
    #     print('df.shape={}'.format(df.shape))

    #     df_test = df
    #     features=[ col for col in df_test.columns if col not in drop_cols ]
    #     xtest = df_test[features]
    #     ypred = []
    #     # NOTE 添加embs
    #     if use_embs:
    #         xtest = pd.merge(xtest, df_emb, how='left', on='imei')

    #     # logger.info('行特征都缺失的行数占比 {}'.format(len(xtest[xtest.isnull().T.all()])/len(xtest))) 
    #     # data_delete = xtest[xtest.isnull().T.all()]
    #     # logger.info('删除前行数 {}'.format(len(xtest)))
    #     # xtest = xtest.drop(index=data_delete.index)
    #     # df_test = df_test.drop(index=data_delete.index)
    #     # logger.info('删除后行数 {}'.format(len(xtest)))
    #     # del data_delete

    #     print('特征预处理')
    #     data_porcess = ML_DataPreprocessor(data = xtest)
    #     xtest = data_porcess.data_preproess()

    #     # 类别特征
    #     for col in cat_cols:
    #         if col != 'imei':
    #             xtest[col] = xtest[col].fillna('-999')
    #             xtest[col] = xtest[col].str.lower()
    #             xtest[col] = xtest[col].map(encoder_maps[col])

    #     # # 特征操作
    #     # xtest = xtest.replace(0, np.nan)

    #     chunk = 500000
    #     for i in range(0,len(xtest),chunk):
    #         tmp = model.predict(xtest[i:i+chunk], num_iteration=model.best_iteration)
    #         ypred  = np.concatenate((ypred ,tmp))

    #     result = pd.DataFrame()
    #     result['imei'] = df_test['imei']

    #     result['prob'] = ypred

    #     ans = ans.append(result)
    
    # ans = ans.reset_index()
    # ans.to_csv(logger_dir+'backflow_all.csv',index=False)

    # # 排序去重
    # logger.info('forecast ans.shape={}'.format(ans.shape))
    # ans.sort_values(by='prob',ascending=False,inplace=True)
    # ans = ans.drop_duplicates(subset='imei', keep='first')
    # logger.info('forecast drop_duplicates ans.shape={}'.format(ans.shape))
    # ans['rank'] = ans.prob.rank(method='first',ascending=False)

    # plt.cla()
    # plt.hist(ans.prob)
    # plt.savefig(logger_dir+'results_all_pred.jpg')

    # ### 圈取人群包
    # print('开始圈取人群包')
    # imei_launch = ans.iloc[:target_user_num, :]
    # logger.info('imei_launch {}'.format(imei_launch))


    # launch_csv = pd.DataFrame()

    # launch_csv['imei'] = imei_launch['imei']
    # bin_list = []
    # for i in range(target_user_num//500000):
    #     bin_list += [i+1]*500000
    # launch_csv['bin'] = bin_list
    # launch_csv ['game_name'] = '和平精英'


    # imei_launch = imei_launch.drop_duplicates(subset=['imei'])
    # logger.info('imei_launch {}'.format(imei_launch))

    # plt.cla()
    # plt.hist(imei_launch.prob,bins = target_user_num//500000)
    # plt.savefig(logger_dir+'backflow_'+str(target_user_num)+'_val.jpg')
    
    # logger.info(model_type+'_prediction end')

    
    # logging.shutdown()
    
    