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
import copy
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


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
        p_list = pack_seq.split('<i>')
        t_list = time_seq.split('<i>')
        min_len = min(len(p_list), len(t_list))
        p_out_list = []
        t_out_list = []
        for j in range(min_len):
            date_now = t_list[j].replace("-","")[:8]
            if not check_timestr(date_now):
                continue
            if int(date_now) <= 20220715:
                p_out_list.append(p_list[j])
                t_out_list.append(t_list[j])
        pack_list[i] = "<i>".join(p_out_list)
        time_list[i] = "<i>".join(t_out_list)
    return [imei_list, pack_list, time_list]


def filter_time(file, tgt_file):
    # only remain interaction in [20220630, 20220715]
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
    app_vocab_file = "app_iaa_vocab_with_cate.json"
    with open(app_vocab_file, 'r') as f:
        app2id = json.load(f)
    out = []
    for i in range(len(imei_list)):
        imei = imei_list[i]
        pack_seq = in_pack_list[i]
        time_seq = in_time_list[i]
        pack_list = pack_seq.split('<i>')
        time_list = time_seq.split('<i>')
        use_pack_list = []
        use_time_list = []
        seq_len = len(pack_list)
            
        for j in range(seq_len): # filter useless APP
            pack = pack_list[j]
            if app2id.get(pack) == None:
                continue
            use_pack_list.append(str(app2id[pack]))
            use_time_list.append(time_list[j])
        if len(use_pack_list) < 5: # padding cold user
            use_pack_list = [str(app2id['unk']) for i in range(5)]
            use_time_list = ['20220831 00:00:00' for i in range(5)]
        if len(use_pack_list) > 256:
            use_pack_list = use_pack_list[-256:]
            use_time_list = use_time_list[-256:]
        out.append("\t".join([imei, ','.join(use_pack_list), ','.join(use_time_list)]) + '\n')
    return out


def filter_users(file, tgt_file):
    # for each user, only remain records with top APPs, and padding for cold users
    
    out = ["imei\topen_seq\topen_time\n"] # header
    reader = pd.read_csv(file, sep='\t', dtype=str, chunksize=120000)
    
    for chunk in tqdm(reader):
        chunk.fillna("unk", inplace=True)
        mp_num = 12
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


def merge_sp_ae_bc(ae_file, bc_files, tgt_path, sp_num):
    # merge aetn & bc pretrain files, and split to n part for multi-gpu
    df_ae = pd.read_csv(ae_file, sep='\t', dtype=str)
    df_bc = pd.DataFrame()
    for file in bc_files:
        df_tmp = pd.read_csv(file, sep='\t', usecols=["imei","open_seq"], dtype=str)
        df_bc = pd.concat([df_bc, df_tmp])
    print("len ae = {}, len bc = {}".format(len(df_ae), len(df_bc)))
    print("same imei: {}".format(len(set(df_ae["imei"]) & set(df_bc["imei"]))))
    df = pd.merge(df_ae, df_bc, on="imei")
    df.dropna(inplace=True)
    print(df.columns)
    print("len total = {}".format(len(df)))

    data_ar = df.values
    if len(data_ar) % sp_num != 0:
        data_ar = data_ar[:len(data_ar) - len(data_ar) % sp_num]
    print("len used = {}".format(len(data_ar)))
    data_ar = np.split(data_ar, sp_num, axis=0)
    for i in tqdm(range(sp_num)):
        data = data_ar[i]
        sp_size = int(0.8 * len(data))
        train_data = data[:sp_size]
        test_data = data[sp_size:]
        with open(tgt_path + "/abcn_train_{}.csv".format(i), 'w') as f:
            f.write("\t".join(["imei", "pack_keep_name", "install_seq", "install_time", "install_cate", "uninstall_seq", "uninstall_time", "uninstall_cate", "open_seq"]) + "\n") # header
            for l in train_data:
                f.write("\t".join(l) + '\n')
        with open(tgt_path + "/abcn_test_{}.csv".format(i), 'w') as f:
            f.write("\t".join(["imei", "pack_keep_name", "install_seq", "install_time", "install_cate", "uninstall_seq", "uninstall_time", "uninstall_cate", "open_seq"]) + "\n") # header
            for l in test_data:
                f.write("\t".join(l) + '\n')


def add_embs(emb_file, file):
    # add embs to downstream file
    emb_df = pd.read_csv(emb_file)
    df = pd.read_csv(file, sep='\t')
    print(len(df))
    df = pd.merge(df, emb_df, how='left', on='imei')
    df.dropna(inplace=True)
    print(len(df))
    df.to_csv(file[:-4] + "_pre.csv", sep='\t', index=False)



if __name__=='__main__':
    data_path = "imei_20221215_quzd_v3_list_v1"
    app_file = "_app_2021_2022_list_202207.csv"
    # app_files = glob.glob(data_path + "/op_*")

    # 1. select 3 useful cols in splited files, and filter to get user records in [20220816~20220831] and limit seq length to [5, 256]. use SAME app vocab from install/uninstall data.

    
    # filter_time(data_path + "/part-0.csv", data_path + "/ad_0.csv")
    # for i in range(9,10):
    #     app_2207_{}.csv".format(i)
    #     app_2207_h_{}.csv".format(i)
    #     filter_time(in_file, out_file)
        #if i == 3:
        #    time.sleep(7200) # wait for clean_sp_data...

    # filter_users(data_path + "/part-2.csv", data_path + "/ad_bc_2.csv")
    # for i in range(5,10):
    #     in_file = data_path + "/part-{}.csv".format(i)
    #     out_file = "op_app_2207/bc_{}.csv".format(i)
    #     filter_users(in_file, out_file)
    #     print(out_file)

    # 2. merge to aetn
    pre_data_path = 'app-series/ad_platform/adT1'
    # aetn_data = pre_data_path+"/aetn_ad_all.csv"
    # merge_sp_ae_bc(aetn_data, glob.glob(data_path + "/ad_bc*"), pre_data_path, 4)
