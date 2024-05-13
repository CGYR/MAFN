# encoding: utf-8

import os
import json
import pandas as pd
import numpy as np
import pickle
import glob
from datetime import datetime
from tqdm import tqdm
from multiprocessing import Pool
import time
import re
import copy
from sklearn.preprocessing import LabelEncoder, MinMaxScaler



def clean_sp_data(file, sub_idx):
    print(file)
    if file=="/home/notebook/data/group/app-series/pretrain_game/hpjy_open/xaa":
        df = pd.read_csv(file, sep='\t', dtype=str, usecols=["imei","pack_name","client_time"])
        df.to_csv("{}open_{}.csv".format(file[:-3], sub_idx), sep='\t', index=False)
    else:
        header = ["imei","pack_name","duration","client_time"]
        df = pd.read_csv(file, sep='\t', dtype=str, header=None, names=header)
        df.drop(columns=["duration"], inplace=True)
        df.to_csv("{}open_{}.csv".format(file[:-3], sub_idx), sep='\t', index=False)


def check_timestr(string):
    time_pattern = '^[0-9]{8}$'
    res = re.search(time_pattern, string)
    if res:
        return True
    else:
        return False


def filter_time_process(input):
    # for parallel
    imei_list, pack_list, time_list = input
    # print(len(pack_list))
    for i in range(len(pack_list)):
        pack_seq = pack_list[i]
        time_seq = time_list[i]
        if isinstance(pack_seq, float) or isinstance(time_seq, float):
            pack_list[i] = ""
            time_list[i] = ""
            continue
        p_list = pack_seq.split('<1>')
        t_list = time_seq.split('<1>')
        min_len = min(len(p_list), len(t_list))
        p_out_list = []
        t_out_list = []
        for j in range(min_len):
            date_now = t_list[j].replace("-","")[:8]
            if not check_timestr(date_now):
                continue
            if int(date_now) >= 20230222:
                p_out_list.append(p_list[j])
                t_out_list.append(t_list[j])
        pack_list[i] = "<1>".join(p_out_list)
        time_list[i] = "<1>".join(t_out_list)
    return [imei_list, pack_list, time_list]


def filter_time(file, tgt_file):
    # only remain interaction in [20230222, 20230228]
    out_imei_list = []
    out_pack_list = []
    out_time_list = []

    reader = pd.read_csv(file, sep='\t', dtype=str, chunksize=100000)
    for df in tqdm(reader):
        imei_list = df["imei"]
        pack_list = df["pack_name"]
        time_list = df["client_time"]

        mp_num = 10
        sub_len = len(pack_list) // mp_num

        # input_list = [(imei_list[i * sub_len : (i+1) * sub_len], pack_list[i * sub_len : (i+1) * sub_len], time_list[i * sub_len : (i+1) * sub_len]) for i in range(mp_num)]
        input_list = []
        imei_array = np.array(imei_list)
        pack_array = np.array(pack_list)
        time_array = np.array(time_list)
        ls_im = np.array_split(imei_array, mp_num, axis=0)
        ls_pa = np.array_split(pack_array, mp_num, axis=0)
        ls_ti = np.array_split(time_array, mp_num, axis=0)
    
        for i in range(mp_num):
            imei_part = ls_im[i].tolist()
            pack_part = ls_pa[i].tolist()
            time_part = ls_ti[i].tolist()
            # imei_part = imei_list[i * sub_len : (i+1) * sub_len]
            # pack_part = pack_list[i * sub_len : (i+1) * sub_len]
            # time_part = time_list[i * sub_len : (i+1) * sub_len]
            input_list.append((imei_part, pack_part, time_part))
    
        # multi process
        pool = Pool(mp_num)
        data = pool.map(filter_time_process, input_list)
        pool.close()
        pool.join()
    
        for part in data:
            out_imei_list += part[0]
            out_pack_list += part[1]
            out_time_list += part[2]

    df = pd.DataFrame()
    df["imei"] = out_imei_list
    df["pack_name"] = out_pack_list
    df["client_time"] = out_time_list

    df.to_csv(tgt_file, sep='\t', index=False)


def filter_user_process(input):
    # for parallel
    imei_list, in_pack_list, in_time_list = input
    app_vocab_file = "/home/notebook/data/group/app-series/pretrain_game/preprocess/app_vocab.json"
    with open(app_vocab_file, 'r') as f:
        app2id = json.load(f)
    out = []
    for i in range(len(imei_list)):
        imei = imei_list[i]
        pack_seq = in_pack_list[i]
        time_seq = in_time_list[i]
        pack_list = pack_seq.split('<1>')
        time_list = time_seq.split('<1>')
        use_pack_list = []
        use_time_list = []
        seq_len = len(pack_list)
            
        for j in range(seq_len): # filter useless APP
            pack = pack_list[j]
            if app2id.get(pack) == None:
                continue
            use_pack_list.append(str(app2id[pack]))
            use_time_list.append(time_list[j])
        if len(use_pack_list) < 5: # filter cold user
            continue
        if len(use_pack_list) > 1024:
            use_pack_list = use_pack_list[-1024:]
            use_time_list = use_time_list[-1024:]
        out.append("\t".join([imei, ','.join(use_pack_list), ','.join(use_time_list)]) + '\n')
    return out


def filter_users(file, tgt_file):
    # for each user, only remain records with top APPs, and filter out cold users
    
    out = ["imei\tpack_name\tclient_time\n"] # header
    reader = pd.read_csv(file, sep='\t', dtype=str, chunksize=100000)
    
    for chunk in tqdm(reader):
        chunk.fillna("unk", inplace=True)
        mp_num = 10
        sub_len = len(chunk) // mp_num
        input_list = []
        imei_array = np.array(chunk["imei"])
        pack_array = np.array(chunk["pack_name"])
        time_array = np.array(chunk["client_time"])
        ls_im = np.array_split(imei_array, mp_num, axis=0)
        ls_pa = np.array_split(pack_array, mp_num, axis=0)
        ls_ti = np.array_split(time_array, mp_num, axis=0)
        for i in range(mp_num):
            imei_part = ls_im[i].tolist()
            pack_part = ls_pa[i].tolist()
            time_part = ls_ti[i].tolist()
            input_list.append((imei_part, pack_part, time_part))
        # multi process
        pool = Pool(mp_num)
        data = pool.map(filter_user_process, input_list)
        pool.close()
        pool.join()

        for part in data:
            out += part
    print("remain user: {}, {}".format(len(out), file))
    # out = out[:945001] # remain 94w users

    with open(tgt_file, 'w') as f:
        for l in out:
            f.write(l)



def gather_open_process(input):
    # for parallel
    imei_list, in_pack_list, in_time_list = input
    app_vocab_file = "/home/notebook/data/group/app-series/pretrain_game/preprocess/app_vocab.json"
    with open(app_vocab_file, 'r') as f:
        app2id = json.load(f)
    out = []
    for i in range(len(imei_list)):
        imei = imei_list[i]
        pack_seq = in_pack_list[i]
        time_seq = in_time_list[i]
        pack_list = pack_seq.split('<1>')
        time_list = time_seq.split('<1>')
        seq_len = len(pack_list)
        tmp_set = {}
            
        for j in range(seq_len): # filter useless APP
            pack = pack_list[j]
            if app2id.get(pack) == None:
                continue
            date = time_list[j].replace("-","")[:8]
            if not check_timestr(date):
                continue
            date = int(date)
            if date not in tmp_set:
                tmp_set[date] = []
            tmp_set[date].append(app2id[pack])
        use_pack_list = []
        for date in tmp_set:
            part_list = tmp_set[date]
            use_pack_list += list(set(part_list)) # merge by day
        if len(use_pack_list) > 256:
            use_pack_list = use_pack_list[:256]
            type_list = [1 for k in range(256)]
        else:
            type_list = [1 for k in range(len(use_pack_list))]
            while len(use_pack_list) < 256:
                use_pack_list.append(0)
                type_list.append(0)
        
        out.append([imei, np.array(use_pack_list, dtype='int16'), np.array(type_list, dtype='int8')])
    return out


def gather_open(file, tgt_file):
    # for each user, only remain records with top APPs, and gather open behaviors by day
    
    # out = ["imei\tpack_name\tclient_time\n"] # header
    out = []
    reader = pd.read_csv(file, sep='\t', dtype=str, chunksize=100000)
    
    for chunk in tqdm(reader):
        chunk.fillna("unk", inplace=True)
        mp_num = 10
        sub_len = len(chunk) // mp_num
        input_list = []
        imei_array = np.array(chunk["imei"])
        pack_array = np.array(chunk["pack_name"])
        time_array = np.array(chunk["client_time"])
        ls_im = np.array_split(imei_array, mp_num, axis=0)
        ls_pa = np.array_split(pack_array, mp_num, axis=0)
        ls_ti = np.array_split(time_array, mp_num, axis=0)
        for i in range(mp_num):
            imei_part = ls_im[i].tolist()
            pack_part = ls_pa[i].tolist()
            time_part = ls_ti[i].tolist()
            input_list.append((imei_part, pack_part, time_part))
        # multi process
        pool = Pool(mp_num)
        data = pool.map(gather_open_process, input_list)
        pool.close()
        pool.join()

        for part in data:
            out += part
    print("remain user: {}, {}".format(len(out), file))
    # out = out[:945001] # remain 94w users

    out_dict = {}
    for l in out:
        imei, open_array, type_array = l
        out_dict[imei] = [open_array, type_array]

    with open(tgt_file, 'wb') as f:
        pickle.dump(out_dict, f)


def merge_pickles(files, tgt_file):
    # merge all pickle files into one tgt_file
    out_dict = {}
    for file in tqdm(files):
        with open(file, 'rb') as f:
            tmp_dict = pickle.load(f)
        for key in tmp_dict:
            out_dict[key] = tmp_dict[key]
    with open(tgt_file, 'wb') as f:
        pickle.dump(out_dict, f)
    print(len(out_dict))


def split_pickle(file):
    # split pickle data to 8:2 train/val
    with open(file, 'rb') as f:
        in_dict = pickle.load(f)
    total_len = len(in_dict)
    print(total_len)
    train_num = int(total_len * 0.8)
    idx = 0
    train_dict = {}
    val_dict = {}
    for key in in_dict:
        if idx < train_num:
            train_dict[key] = in_dict[key]
        else:
            val_dict[key] = in_dict[key]
        idx += 1
    with open(file[:-4] + "_train.pkl", 'wb') as f:
        pickle.dump(train_dict, f)
    with open(file[:-4] + "_val.pkl", 'wb') as f:
        pickle.dump(val_dict, f)


def merge_sp_ae_bc(ae_file, bc_files, tgt_path, sp_num):
    # merge aetn & bc pretrain files, and split to n part for multi-gpu
    df_ae = pd.read_csv(ae_file, sep='\t', dtype=str)
    df_bc = pd.DataFrame()
    for file in bc_files:
        df_tmp = pd.read_csv(file, sep='\t', usecols=["imei","pack_name"], dtype=str)
        df_bc = pd.concat([df_bc, df_tmp])
    print("len ae = {}, len bc = {}".format(len(df_ae), len(df_bc)))
    print("same imei: {}".format(len(set(df_ae["imei"]) & set(df_bc["imei"]))))
    df_bc.rename(columns={"pack_name":"open_seq"}, inplace=True)
    df = pd.merge(df_ae, df_bc, on="imei", how="outer")
    df.fillna('unk', inplace=True)
    print(df.columns)
    print("len total = {}".format(len(df)))

    data_ar = df.values
    if len(data_ar) % sp_num != 0:
        data_ar = data_ar[:len(data_ar) - len(data_ar) % sp_num]
    print("len used = {}".format(len(data_ar)))
    data_ar = np.split(data_ar, sp_num, axis=0)
    for i in tqdm(range(sp_num)):
        train_data = data_ar[i]
        with open(tgt_path + "abcn_train_{}.csv".format(i), 'w') as f:
            f.write("\t".join(list(df.columns)) + "\n") # header
            for l in train_data:
                f.write("\t".join(l) + '\n')



def merge_ae_bc(ae_file, bc_file, tgt_file):
    # merge aetn & bc pretrain files, and split to n part for multi-gpu
    df_ae = pd.read_csv(ae_file, sep='\t', dtype=str)
    df_bc = pd.read_csv(bc_file, sep='\t', usecols=["imei","pack_name"], dtype=str)
    print("len ae = {}, len bc = {}".format(len(df_ae), len(df_bc)))
    print("same imei: {}".format(len(set(df_ae["imei"]) & set(df_bc["imei"]))))
    df_bc.rename(columns={"pack_name":"open_seq"}, inplace=True)
    df = pd.merge(df_ae, df_bc, on="imei", how="outer")
    df.fillna('unk', inplace=True)
    print(df.columns)
    print("len total = {}".format(len(df)))
    df.to_csv(tgt_file, sep='\t', index=False)


def add_embs(emb_file, file):
    # add embs to downstream file
    emb_df = pd.read_csv(emb_file)
    df = pd.read_csv(file, sep='\t')
    print(len(df))
    df = pd.merge(df, emb_df, how='left', on='imei')
    df.dropna(inplace=True)
    print(len(df))
    df.to_csv(file[:-4] + "_pre.csv", sep='\t', index=False)


def merge_train_files(emb_files, tgt_file):
    df = pd.DataFrame()
    for file in tqdm(emb_files):
        print(file)
        df_tmp = pd.read_csv(file, dtype=str)
        df_tmp.drop_duplicates(subset=["imei"], inplace=True)
        df_tmp["dayno"] = ["2023" + file[-8:-4] for i in range(len(df_tmp))]
        df = pd.concat([df, df_tmp])
    print("hpjy imei num:{}".format(len(df)))
    df.to_csv(tgt_file, index=False)


if __name__=='__main__':
    data_path = "/home/notebook/data/group/app-series/pretrain_game/hpjy_open/"
    app_file = "/home/notebook/data/group/app-series/pretrain/shixu_app_2021_2022_list_202207.csv"
    app_files = glob.glob(data_path + "x*")
    open_files = glob.glob(data_path + "open*")

    # clean_split_data(app_file, 1000000) # USELESS, use shell [split] command instead
    
    # 1. select 3 useful cols in splited files, and filter to get user records in [20220630~20220715] and limit seq length to [5, 1024]. use SAME app vocab from install/uninstall data.
    
    # for i in range(len(app_files)):
    #     print(i)
    #     clean_sp_data(app_files[i], i)
    # clean_sp_data(app_files[19], 19)

    # for i in range(len(open_files)):
    #     in_file = open_files[i]
    #     out_file = data_path + "bc_{}".format(i)
    #     print(out_file)
    #     filter_users(in_file, out_file)

    # print("filter ok!")
    # # NOTE: only for gather open seq, not for abcn
    # for i in range(len(open_files)):
    #     in_file = open_files[i]
    #     out_file = data_path + "gb_{}.pickle".format(i)
    #     print(in_file)
    #     gather_open(in_file, out_file)
    # pickle_files = glob.glob(data_path + "gb*")
    # pickle_all_file = data_path + "open_data.pkl"
    # merge_pickles(pickle_files, pickle_all_file)
    # split_pickle(pickle_all_file)

    # print("pickle ok!")

    # # 2. transfer to input format
    # open_files = glob.glob(data_path + "bc_*")
    # ae_file = "/home/notebook/data/group/app-series/pretrain_game/hpjy202303/un_in_forecast_clean.csv"
    # merge_sp_ae_bc(ae_file, open_files, data_path, 4)

    # NOTE repeat process, for 0301~0307
    # open_files = glob.glob(data_path + "shixu_20230320_imei_hpjy_train_2023*")
    
    # for i in range(len(open_files)):
    #     in_file = open_files[i]
    #     out_file = data_path + "day_op_{}".format(in_file[-8:])
    #     print(out_file)
    #     filter_users(in_file, out_file)

    # print("filter ok!")
    # # NOTE: only for gather open seq, not for abcn
    # for i in range(len(open_files)):
    #     in_file = open_files[i]
    #     out_file = data_path + "open_data_{}.pkl".format(in_file[-8:-4])
    #     print(out_file)
    #     gather_open(in_file, out_file)
    #     split_pickle(out_file)

    # print("pickle ok!")

    # # 2. transfer to input format
    # open_files = glob.glob(data_path + "day_op_03*")
    # for file in open_files:
    #     part = file[-8:]
    #     ae_file = "/home/notebook/data/group/app-series/pretrain_game/hpjy202303/un_in_2023{}".format(part)
    #     merge_ae_bc(ae_file, file, data_path + "abcn_day_{}".format(part))

    # merge train embs
    train_sum_files = ["/home/notebook/data/group/app-series/pretrain_game/hpjy_open/imei_embs_sum_fine_030{}.csv".format(i+1) for i in range(7)]
    train_ae_files = ["/home/notebook/data/group/app-series/pretrain_game/hpjy_open/imei_embs_sum_last_030{}.csv".format(i+1) for i in range(7)]

    merge_train_files(train_sum_files, "/home/notebook/data/group/app-series/pretrain_game/hpjy_open/imei_embs_sum_fine_train.csv")
    # merge_train_files(train_ae_files, "/home/notebook/data/group/app-series/pretrain_game/hpjy_open/imei_embs_ae_last_train.csv")

    
    

