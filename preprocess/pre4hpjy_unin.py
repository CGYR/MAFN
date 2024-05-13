# encoding: utf-8

import os
import json
import pandas as pd
import numpy as np
import glob
from datetime import datetime
from tqdm import tqdm
from multiprocessing import Pool
import time
import re
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

def merge_in_un(in_file, un_file, tgt_file):
    # merge install & unstall, select users with both records
    in_df = pd.read_csv(in_file, sep='\t', usecols=["imei", "pack_name", "client_time"], dtype=str)
    un_df = pd.read_csv(un_file, sep='\t', usecols=["imei", "pack_name", "client_time"], dtype=str)
    in_df.rename(columns={"pack_name":"install_seq", "client_time":"install_time"}, inplace=True)
    un_df.rename(columns={"pack_name":"uninstall_seq", "client_time":"uninstall_time"}, inplace=True)
    # in_df.drop(columns=["model", "dayno"], inplace=True)
    # un_df.drop(columns=["model", "dayno"], inplace=True)
    print(len(in_df), len(un_df))
    out_df = in_df.merge(un_df, how="outer", on="imei")
    out_df.fillna("unk", inplace=True)
    print(len(out_df))
    print(out_df.columns)
    out_df.to_csv(tgt_file, index=False, sep='\t')


def sp_n_part(file, n, must_equal=False):
    # split big data to n part, because py3.7 cannot support big file for multi-process...
    df = pd.read_csv(file, sep='\t', dtype=str)
    print(df.columns)
    data = df.values
    if not must_equal:
        data = np.array_split(data, n, axis=0)
    else:
        if len(data) % n != 0:
            data = data[:len(data) - len(data) % n]
        data = np.split(data, n, axis=0)

    for idx in range(len(data)):
        tgt_file = file[:-4] + "_{}.csv".format(idx)
        with open(tgt_file, 'w') as f:
            f.write("\t".join(df.columns) + "\n")# header
            for l in data[idx]:
                f.write("\t".join(l) + '\n')


def check_time(string):
    re_exp = '^[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}$'
    res = re.search(re_exp, string)
    if res:
        return True
    else:
        return False
  

def filter_app_process(input):
    # for multi-process
    app_vocab_file = "/home/notebook/data/group/app-series/pretrain_game/preprocess/app_vocab.json"
    with open(app_vocab_file, 'r') as f:
        app2id = json.load(f)
    app_cate_file = "/home/notebook/data/personal/mete/app_name_to_cate_id.json"
    with open(app_cate_file, 'r') as f:
        app2cate = json.load(f)
    cate_pad_id = '0' # 65 category
    
    out = []
    for row in input:
        imei, in_seq, in_time, un_seq, un_time = row

        in_p_list = in_seq.split('<1>')
        in_t_list = in_time.split('<1>')
        o_in_p_list = []
        o_in_t_list = []
        for i in range(len(in_p_list)):
            app = in_p_list[i]
            time = in_t_list[i]
            if app in app2id:
                o_in_p_list.append(app)
                if check_time(time):
                    o_in_t_list.append(time)
                else:
                    o_in_t_list.append("2023-01-01 00:00:00")
        o_in_c_list = []
        for app in o_in_p_list:
            if app in app2cate:
                o_in_c_list.append(str(app2cate[app]))
            else:
                o_in_c_list.append(cate_pad_id)
            
        un_p_list = un_seq.split('<1>')
        un_t_list = un_time.split('<1>')
        o_un_p_list = []
        o_un_t_list = []
        for i in range(len(un_p_list)):
            app = un_p_list[i]
            time = un_t_list[i]
            if app in app2id:
                o_un_p_list.append(app)
                if check_time(time):
                    o_un_t_list.append(time)
                else:
                    o_un_t_list.append("2023-01-01 00:00:00")
        o_un_c_list = []
        for app in o_un_p_list:
            if app in app2cate:
                o_un_c_list.append(str(app2cate[app]))
            else:
                o_un_c_list.append(cate_pad_id)

        retention_list = []
        retention_list += list(set(o_in_p_list) - set(o_un_p_list))

        and_set = set(o_in_p_list) & set(o_un_p_list)
        in_set = {o_in_p_list[i] : o_in_t_list[i] for i in range(len(o_in_p_list))}
        un_set = {o_un_p_list[i] : o_un_t_list[i] for i in range(len(o_un_p_list))}
        for pack in and_set:
            in_time = int(datetime.timestamp(datetime.strptime(in_set[pack], '%Y-%m-%d %H:%M:%S')))
            un_time = int(datetime.timestamp(datetime.strptime(un_set[pack], '%Y-%m-%d %H:%M:%S')))
            if in_time >= un_time:
                retention_list.append(pack)
        if retention_list == []:
            retention_list = ["unk"]
        if o_in_p_list == []:
            o_in_p_list = ["unk"]
            o_in_t_list = ["unk"]
            o_in_c_list = ["0"]
        if o_un_p_list == []:
            o_un_p_list = ["unk"]
            o_un_t_list = ["unk"]
            o_un_c_list = ["0"]
        # limit seq len
        # o_in_p_list = o_in_p_list[-128:]
        # o_in_t_list = o_in_t_list[-128:]
        # o_in_c_list = o_in_c_list[-128:]
        # o_un_p_list = o_un_p_list[-128:]
        # o_un_t_list = o_un_t_list[-128:]
        # o_un_c_list = o_un_c_list[-128:]
        out.append("\t".join([imei, ",".join(retention_list), ",".join(o_in_p_list), ",".join(o_in_t_list), ",".join(o_in_c_list), ",".join(o_un_p_list), ",".join(o_un_t_list), ",".join(o_un_c_list)]))
    return out
      
    
def filter_app_para(file, tgt_file):
    # filter cold apps, and transfer iaa_code to pack_name, and generate retention list
    df = pd.read_csv(file, sep='\t', dtype=str)
    data_ar = df.values
    data_ar = np.array_split(data_ar, 10, axis=0)
    # multi-process
    pool = Pool(10)
    out_data = pool.map(filter_app_process, data_ar)
    pool.close()
    pool.join()
        # data += out_data
    with open(tgt_file, 'w') as f:
        f.write("\t".join(["imei", "pack_keep_name", "install_seq", "install_time", "install_cate", "uninstall_seq", "uninstall_time", "uninstall_cate"]) + "\n") # header
        for part in out_data:
            for l in part:
                f.write(l + '\n')
    #    idx += 1


def merge_all(files, tgt_file):
    # merge all files to 1 file, add attr
    out_df = pd.DataFrame()
    for file in tqdm(files):
        df = pd.read_csv(file, sep='\t', dtype=str)
        out_df = pd.concat([out_df, df])
    out_df.fillna("unk", inplace=True)
    # attr_file = '/home/notebook/data/group/app-series/pretrain_game/shixu_20230307_quzd.csv'
    # df_attr = pd.read_csv(attr_file, sep='\t', dtype=str, usecols=["imei","age","gender","province"])
    # out_df = pd.merge(out_df, df_attr, how='left', on='imei')
    # out_df.fillna("0", inplace=True)
    out_df = out_df.sort_values(by=["imei"])
    out_df.to_csv(tgt_file, index=False, sep='\t')


def merge_attr(file):
    attr_file = '/home/notebook/data/group/app-series/pretrain_game/shixu_20230307_quzd.csv'
    df_attr = pd.read_csv(attr_file, sep='\t', dtype=str, usecols=["imei","age","gender","province"])
    out_df = pd.read_csv(file, sep='\t', dtype=str)
    out_df = pd.merge(out_df, df_attr, how='left', on='imei')
    out_df.fillna("0", inplace=True)
    out_df.to_csv(file, index=False, sep='\t')


if __name__=="__main__":
    dir = "/home/notebook/data/group/app-series/pretrain_game/"
    tgt_dir = dir + "hpjy202303/"
    ins_files = glob.glob(dir + "*_install*hpjy*")
    uns_files = glob.glob(dir + "*_uninstall*hpjy*")


    in_un_file = tgt_dir + "un_in_forecast.csv"
    tgt_file = tgt_dir + "un_in_forecast_clean.csv"

    vocab_file = "/home/notebook/data/group/app-series/pretrain_game/preprocess/app_vocab.json"
   
    # process: merge in_un data, remain certain apps, construct retention list, add attr info and merge


    # for i in tqdm(range(1, len(uns_files))): # skip forcast part, skip 0301 since only install but not unstall
    #     in_file = ins_files[i]
    #     un_file = uns_files[i]
    #     tgt_file = tgt_dir + "un_in_{}".format(in_file[-12:])
    #     assert in_file[-12:] == un_file[-12:]
    #     print(tgt_file)
    #     merge_in_un(in_file, un_file, tgt_file)
    
    # merge_in_un(ins_files[0], uns_files[0], tgt_dir + "un_in_forecast.csv")

    
    in_un_files = glob.glob(tgt_dir + "un_in_2*")
    for file in tqdm(in_un_files):
        filter_app_para(file, file)

    sp_n_part(in_un_file, 10)
    for i in tqdm(range(10)):
        file_tmp = in_un_file[:-4] + "_{}.csv".format(i)
        filter_app_para(file_tmp, file_tmp)

    merge_all([in_un_file[:-4] + "_{}.csv".format(i) for i in range(10)], tgt_file)

    clean_files = glob.glob(tgt_dir + "un_in_20*") + [tgt_file]
    for file in tqdm(clean_files):
        merge_attr(file)





    
    
