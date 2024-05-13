# encoding: utf-8
# author: yuren

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


def gen_iaa_app_vocab():
    # generate iaa app vocab
    cate_file = "pretrain/app_type_20221207.csv"
    cate_df = pd.read_csv(cate_file)
    pack_list = cate_df["pack_name"]
    pack_type = cate_df["type"]
    pack2type = {pack_list[i] : pack_type[i] for i in range(len(pack_list))}
    pack_type = cate_df["type"].unique()
    type2id = { pack_type[i] : i+1 for i in range(len(pack_type))} # cate : cate id, 0 for padding
    pack2type_id = { k : type2id[pack2type[k]] for k in pack2type} # pack_name : cate id
    pack2id = {}
    pack2id["pad"] = 0
    pack2id["unk"] = 1
    pack2id["mask"] = 2
    pack2id.update({pack_list[i] : i+3 for i in range(len(pack_list))}) # pack_name : pack id
    with open("app_vocab_with_cate.json", 'w') as f:
        json.dump(pack2id, f, indent=2)
    with open("cate2id.json", 'w') as f:
        json.dump(type2id, f, indent=2)
    
    #app_vocab_file = "/home/notebook/data/personal/mete/aetn_app_down_vocab_new.json"
    iaa_vocab_file = "iaa_id2pack.json"
    tgt_vocab_file = "app_iaa_vocab_with_cate.json"
    # with open(app_vocab_file, 'r') as f:
    #     app2id = json.load(f)
    app2id = pack2id
    with open(iaa_vocab_file, 'r') as f:
        iaa2app = json.load(f)
    print(len(app2id), len(iaa2app))
    idx = 10408 # pack num + 3
    for key in iaa2app:
        app = iaa2app[key]
        if app not in app2id:
            app2id[app] = idx
            idx += 1
        if app not in pack2type_id:
            pack2type_id[app] = type2id["游戏"] # add type for iaa apps
    print(len(app2id))
    with open(tgt_vocab_file, 'w') as f:
        json.dump(app2id, f, indent=2)
    with open("app_name_to_cate_id.json", 'w') as f:
        json.dump(pack2type_id, f, indent=2)


def merge_files(files, tgt_file):
    # merge files to dataset, and transfer to [imei, app, label] version
    out = ["imei,app,label"]
    head = ["imei","ad_id","traceid","pos_id","app_id","dev_app_id","weekday","hour","click","convert","pcvr","pctr","ocpc_type","click_pcvr","click_pcvr_nums","exps_pctr","exps_pctr_nums","diff","dayno","acc_cost"]
    
    id2app = {}
    with open('/home/notebook/data/personal/ad/iaa_app.txt', 'r') as f:
        line = f.readline()
        while line:
            line = line.strip()
            id2app[line.split('\t')[0]] = line.split('\t')[1]
            line = f.readline() 

    for file in files:
        df = pd.read_csv(file, header=None, names=head)
        imei_list = df["imei"]
        app_list = df["dev_app_id"]
        label_list = df["convert"]
        for i in range(len(imei_list)):
            app_id = str(app_list[i])
            app = id2app[app_id]
            out.append(",".join([imei_list[i], app, str(label_list[i])]))
        print("file {} down!".format(file))
    
    with open(tgt_file, 'w') as f:
        for l in out:
            f.write(l+'\n')


def get_imei_list(data_path):
    df = pd.read_csv(data_path + "/train.csv")
    imei_list = list(df["imei"].unique())
    df = pd.read_csv(data_path + "/test.csv")
    imei_list += list(df["imei"].unique())
    imei_list = list(set(imei_list))
    with open(data_path + "/ad_imei.csv", 'w') as f:
        f.write("imei\n")
        for imei in imei_list:
            f.write(imei + '\n')


def merge_in_un(in_file, un_file, tgt_file):
    # merge install & unstall, select users with both records
    in_df = pd.read_csv(in_file, sep='\t')
    un_df = pd.read_csv(un_file, sep='\t')
    in_df.rename(columns={"pack_name":"install_seq", "client_time":"install_time"}, inplace=True)
    un_df.rename(columns={"pack_name":"uninstall_seq", "client_time":"uninstall_time"}, inplace=True)
    in_df.drop(columns=["model"], inplace=True)
    un_df.drop(columns=["model"], inplace=True)
    print(len(in_df), len(un_df))
    out_df = in_df.merge(un_df, how="outer", on="imei")
    out_df.fillna("unk", inplace=True)
    print(len(out_df))
    print(out_df.columns)
    out_df.to_csv(tgt_file, index=False, sep='\t')


def sp_n_part(file, n):
    # split big data to n part
    df = pd.read_csv(file, sep='\t', dtype=str)
    print(df.columns)
    data = df.values
    data = np.array_split(data, n, axis=0)
    for idx in range(len(data)):
        tgt_file = file[:-7] + "{}.csv".format(idx)
        with open(tgt_file, 'w') as f:
            f.write("imei\tinstall_seq\tinstall_time\tuninstall_seq\tuninstall_time\n")# header
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
    app_vocab_file = "app_iaa_vocab_with_cate.json"
    with open(app_vocab_file, 'r') as f:
        app2id = json.load(f)
    app_cate_file = "app_name_to_cate_id.json"
    with open(app_cate_file, 'r') as f:
        app2cate = json.load(f)
    cate_pad_id = '0' # 65 category
    
    out = []
    for row in input:
        imei, in_seq, in_time, un_seq, un_time = row

        in_p_list = in_seq.split('<i>')
        in_t_list = in_time.split('<i>')
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
                    o_in_t_list.append("2022-07-01 00:00:00")
        o_in_c_list = []
        for app in o_in_p_list:
            if app in app2cate:
                o_in_c_list.append(str(app2cate[app]))
            else:
                o_in_c_list.append(cate_pad_id)
            
        un_p_list = un_seq.split('<i>')
        un_t_list = un_time.split('<i>')
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
                    o_un_t_list.append("2022-07-01 00:00:00")
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
        o_in_p_list = o_in_p_list[-128:]
        o_in_t_list = o_in_t_list[-128:]
        o_in_c_list = o_in_c_list[-128:]
        o_un_p_list = o_un_p_list[-128:]
        o_un_t_list = o_un_t_list[-128:]
        o_un_c_list = o_un_c_list[-128:]
        out.append("\t".join([imei, ",".join(retention_list), ",".join(o_in_p_list), ",".join(o_in_t_list), ",".join(o_in_c_list), ",".join(o_un_p_list), ",".join(o_un_t_list), ",".join(o_un_c_list)]))
    return out
      
    
def filter_app_para(file, tgt_file):
    # filter cold apps, and transfer iaa_code to pack_name, and generate retention list
    df = pd.read_csv(file, sep='\t', dtype=str)
    data_ar = df.values
    data_ar = np.array_split(data_ar, 15, axis=0)
    # multi-process
    pool = Pool(15)
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


def merge_aetn_all(files, tgt_file):
    # merge all aetn files to 1 file
    out_df = pd.DataFrame()
    for file in tqdm(files):
        df = pd.read_csv(file, sep='\t', dtype=str)
        out_df = pd.concat([out_df, df])
    out_df = out_df.sort_values(by=["imei"])
    out_df.to_csv(tgt_file, index=False, sep='\t')


def get_train_test(file):
    # select train/test data, 8:2, split to 4 file for multi gpus
    df = pd.read_csv(file, sep='\t', dtype=str, nrows=None)
    print(len(df))
    df.dropna(inplace=True)
    data_ar = df.values
    print(len(data_ar))
    if len(data_ar) % 4 != 0:
        data_ar = data_ar[:len(data_ar) - len(data_ar) % 4]
    print(len(data_ar))
    data_ar = np.split(data_ar, 4, axis=0)
    for i in tqdm(range(len(data_ar))):
        data = data_ar[i]
        sp_size = int(0.8 * len(data))
        train_data = data[:sp_size]
        test_data = data[sp_size:]
        with open(file[:-8] + "_train_{}.csv".format(i), 'w') as f:
            f.write("\t".join(["imei", "pack_keep_name", "install_seq", "install_time", "install_cate", "uninstall_seq", "uninstall_time", "uninstall_cate"]) + "\n") # header
            for l in train_data:
                f.write("\t".join(l) + '\n')
        with open(file[:-8] + "_test_{}.csv".format(i), 'w') as f:
            f.write("\t".join(["imei", "pack_keep_name", "install_seq", "install_time", "install_cate", "uninstall_seq", "uninstall_time", "uninstall_cate"]) + "\n") # header
            for l in test_data:
                f.write("\t".join(l) + '\n')


def add_user_attr(files, attr_file):
    # add user attrs features for pretrain train/test files
    df_attr = pd.read_csv(attr_file, sep='\t', dtype=str)
    for file in tqdm(files):
        print(file)
        df = pd.read_csv(file, sep='\t', dtype=str)
        print(len(df))
        df = pd.merge(df, df_attr, how='left', on='imei')
        print(len(df))
        df.to_csv(file[:-6] + "_attr" + file[-6:], sep='\t', index=False)


def gen_finetune_data(train_file, test_file, pre_files):
    # gen data for finetuning
    pre_data_path = 'app-series/ad_platform/adT1'
    df_pre = pd.DataFrame()
    for file in tqdm(pre_files):
        df_tmp = pd.read_csv(file, sep='\t', dtype=str, nrows=None)
        df_pre = pd.concat([df_pre, df_tmp])
    
    # for train_file
    df_train = pd.read_csv(train_file)
    df_train = pd.merge(df_train, df_pre, how="left", on="imei")
    df_train.dropna(inplace=True)
    print("finetune train len: {}".format(len(df_train)))
    df_train.to_csv(pre_data_path + "/train_finetune.csv", sep='\t', index=False)

    # for test_file
    df_test = pd.read_csv(test_file)
    df_test = pd.merge(df_test, df_pre, how="left", on="imei")
    df_test.dropna(inplace=True)
    print("finetune test len: {}".format(len(df_test)))
    df_test.to_csv(pre_data_path + "/test_finetune.csv", sep='\t', index=False)
                

def merge_embs(file, emb_files, tgt_file):
    # merge embs to file
    df_emb = pd.DataFrame()
    for ef in tqdm(emb_files):
        df_tmp = pd.read_csv(ef, dtype=str)
        df_emb = pd.concat([df_emb, df_tmp])
    print("load embs down.")
    df = pd.read_csv(file, dtype=str)
    print("data len:{}, imei in embs:{}".format(len(df), len(df_emb)))
    df = pd.merge(df, df_emb, how='inner', on='imei')
    print(len(df))
    df.to_csv(tgt_file, index=False)


def get_rand_embs(file):
    # get random embs
    df = pd.read_csv(file)
    df_len = len(df)
    for i in tqdm(range(64)):
        rand_ar = np.random.randn(df_len)
        df[str(i)] = rand_ar
        # df["e_" + str(i)] = rand_ar
    df.to_csv(file[:-7] + "rand_cate.csv", index=False) # for aetn
    # df.to_csv(file[:-10] + "rand_bc.csv", index=False) # for bc


def merge_bc_embs(file, emb_files):
    # merge embs to file
    df_emb = pd.DataFrame()
    for ef in tqdm(emb_files):
        df_tmp = pd.read_csv(ef, nrows=None)
        df_emb = pd.concat([df_emb, df_tmp])
    print("load embs down.")
    emb_list = [eval(e) for e in df_emb["emb"]]
    df_emb["emb"] = emb_list
    for i in tqdm(range(64)):
        df_emb['e_{}'.format(i)] = df_emb['emb'].map(lambda x:x[i])
    df_emb.drop(columns=["emb"], inplace=True)

    df = pd.read_csv(file)
    print("data len:{}, imei in embs:{}".format(len(df), len(df_emb)))
    df = pd.merge(df, df_emb, how='inner', on='imei')
    print(len(df))
    print(df.columns)
    df.to_csv(file[:-4] + "_pre_bc.csv", index=False)


def compare_imei(files1, files2):
    df_emb = pd.DataFrame()
    for ef in tqdm(files1):
        df_tmp = pd.read_csv(ef, usecols=["imei"], dtype=str)
        df_emb = pd.concat([df_emb, df_tmp])
    print(len(df_emb))
    df_emb2 = pd.DataFrame()
    for ef in tqdm(files2):
        df_tmp = pd.read_csv(ef, usecols=["imei"], dtype=str)
        df_emb2 = pd.concat([df_emb2, df_tmp])
    print(len(df_emb2))
    print(len(pd.merge(df_emb, df_emb2, how='inner', on='imei')))
    

def compare_app(files1, files2):
    df_emb = pd.DataFrame()
    for ef in tqdm(files1):
        df_tmp = pd.read_csv(ef, usecols=["pack_name"], dtype=str)
        df_emb = pd.concat([df_emb, df_tmp])
    # print(len(df_emb))
    df_emb.rename(columns={"pack_name":"app"}, inplace=True)
    set1 = set(df_emb["app"].unique())
    print(len(set1))

    df_emb2 = pd.DataFrame()
    for ef in tqdm(files2):
        df_tmp = pd.read_csv(ef, usecols=["app"], dtype=str)
        df_emb2 = pd.concat([df_emb2, df_tmp])
    # print(len(df_emb2))
    set2 = set(df_emb2["app"].unique())
    print(len(set2))
    print(len(set1 & set2))
    # print(len(pd.merge(df_emb, df_emb2, how='inner', on='app')))


def merge_2_embs(files1, files2, tgt_file):
    df_emb = pd.DataFrame()
    for ef in tqdm(files1):
        df_tmp = pd.read_csv(ef, dtype=str)
        df_emb = pd.concat([df_emb, df_tmp])
    print(len(df_emb))

    df_emb2 = pd.DataFrame()
    for ef in tqdm(files2):
        df_tmp = pd.read_csv(ef, dtype=str)
        df_emb2 = pd.concat([df_emb2, df_tmp])
    print(len(df_emb2))
    emb_list = [eval(e) for e in df_emb2["emb"]]
    df_emb2["emb"] = emb_list
    for i in tqdm(range(64)):
        df_emb2['e_{}'.format(i)] = df_emb2['emb'].map(lambda x:x[i])
    df_emb2.drop(columns=["emb"], inplace=True)

    df = pd.merge(df_emb, df_emb2, how='inner', on='imei')
    print(df.columns)
    print(len(df))
    df.to_csv(tgt_file, index=False)


def merge_2_embs_2(files1, files2, tgt_file):
    df_emb = pd.DataFrame()
    for ef in tqdm(files1):
        df_tmp = pd.read_csv(ef, dtype=str)
        df_emb = pd.concat([df_emb, df_tmp])
    print(len(df_emb))

    df_emb2 = pd.DataFrame()
    for ef in tqdm(files2):
        df_tmp = pd.read_csv(ef, dtype=str)
        df_emb2 = pd.concat([df_emb2, df_tmp])
    print(len(df_emb2))
    df_emb2.rename(columns={str(i):'e_{}'.format(i) for i in range(64)}, inplace=True)
    df = pd.merge(df_emb, df_emb2, how='inner', on='imei')
    print(df.columns)
    print(len(df))
    df.to_csv(tgt_file, index=False)


def add_bc_app_embs(file, emb_file):
    # add app_embs to train/test file
    df_emb = pd.read_csv(emb_file, dtype=str)
    emb_list = [eval(e) for e in df_emb["emb"]]
    df_emb["emb"] = emb_list
    for i in tqdm(range(64)):
        df_emb['a_{}'.format(i)] = df_emb['emb'].map(lambda x:x[i])
    df_emb.drop(columns=["emb"], inplace=True)
    df_emb.rename(columns={"pack_name":"app"}, inplace=True)

    df = pd.read_csv(file, dtype=str)
    df = pd.merge(df, df_emb, how="left", on="app")
    df.fillna(0, inplace=True)
    print(df.columns)
    print(df.head())
    df.to_csv(file[:-4] + "_app.csv", index=False)


def add_ae_app_embs(file, emb_file):
    # add app_embs to train/test file
    df_emb = pd.read_csv(emb_file, dtype=str)
    df_emb.rename(columns={"pack_name":"app"}, inplace=True)
    df_emb.rename(columns={str(i):'a_{}'.format(i) for i in range(64)}, inplace=True)

    df = pd.read_csv(file, dtype=str)
    df = pd.merge(df, df_emb, how="left", on="app")
    df.fillna(0, inplace=True)
    # print(df.columns)
    print(df.head())
    df.to_csv(file[:-4] + "_app.csv", index=False)


def merge_pre_all(files, tgt_file):
    # merge all preprocessed files into one tgt_file
    df = pd.DataFrame()
    for file in tqdm(files):
        df_tmp = pd.read_csv(file, sep='\t', dtype=str, nrows=None)
        df = pd.concat([df, df_tmp])
    df.to_csv(tgt_file, sep='\t', index=False)


def merge_2_embs_3(emb_file1, emb_file2, tgt_file):
    # avoid mem usage
    ef1 = pd.read_csv(emb_file1,dtype=str)
    ef2 = pd.read_csv(emb_file2,dtype=str)
    length = len(ef1)
    assert length == len(ef2)
    imei_list = list(ef1["imei"])
    ef1.drop(["imei"], axis=1, inplace=True)
    ef2.drop(["imei"], axis=1, inplace=True)
    value1 = ef1.values
    value2 = ef2.values
    header = "imei," + ",".join([str(i) for i in range(64)] + ["e_" + str(i) for i in range(64)])
    with open(tgt_file, 'w') as f:
        f.write(header + '\n')
        for i in tqdm(range(length)):
            f.write( ','.join([imei_list[i], ','.join(value1[i]), ','.join(value2[i])]) + '\n')




if __name__ == '__main__':
    data_path = "/home/notebook/data/personal/ad/"
    train_files = glob.glob(data_path + "/202209[0-1]*/*.csv") + glob.glob(data_path + "/20220920/*.csv")
    test_files = glob.glob(data_path + "/2022092[1-9]/*.csv")
    # 1. merge train & test files
    # merge_files(train_files, data_path + "/train.csv")
    # merge_files(test_files, data_path + "/test.csv")

    # 2. prepare for pre-train
    pre_data_path = '/home/notebook/data/group/app-series/ad_platform/adT1'
    # install_raw = glob.glob(pre_data_path + '/*install*1209*')[0]
    # unstall_raw = glob.glob(pre_data_path + '/*unstall*1209*')[0]
    # merge_in_un(install_raw, unstall_raw, pre_data_path + "/in_un_all.csv") # merge install & unstall files
    # sp_n_part(pre_data_path + "/in_un_all.csv", 4) # split to 4 file
    # for i in range(4):
    #     print(i)
    #     filter_app_para(pre_data_path + "/in_un_{}.csv".format(i), pre_data_path + "/aetn_ad_{}.csv".format(i)) # select top 10000 app and iaa apps
    # aetn_files = [pre_data_path + "/aetn_ad_{}.csv".format(i) for i in range(4)]
    #aetn_files = glob.glob(pre_data_path + "/aetn*")
    # print(aetn_files)
    # time.sleep(1800)
    # merge_aetn_all(aetn_files, pre_data_path+"/aetn_ad_all.csv")
    # get_train_test(pre_data_path+"/aetn_ad_all.csv") # split train & val, split to 4 file for multi gpus
    # add_user_attr(glob.glob(pre_data_path+"/abcn_test_*.csv"), pre_data_path+"/20230109_imei_shixu_yuren_v1.csv") # # add user attrs features for pretrain train/test files
    train_file = data_path + "/train.csv"
    test_file = data_path + "/test.csv"
    # NOTE gen data for finetune
    # pre_files = glob.glob(pre_data_path + "/*attr*")
    # merge_pre_all(pre_files, pre_data_path + "/pretrain_all.csv") # merge ALL pretrain files into one file
    # gen_finetune_data(train_file, test_file, pre_files) # output: pre_data_path + "/train_finetune.csv" and "/test_finetune.csv"

    # 3. add embs to downstream
    # emb_files = glob.glob('/home/notebook/data/group/app-series/ad_platform/in_un_embs' + "/adT1_imei*")[-4:]
    emb_files = glob.glob('/home/notebook/data/group/app-series/ad_platform/in_un_embs' + "/abcn_cl_imei_embs_ae_*")
    # # emb_files = glob.glob('/home/notebook/data/group/app-series/ad_platform/in_un_embs' + "/abcn_attr_imei_embs_ae_*")
    # # print(emb_files)
    # merge_embs(train_file, emb_files, train_file[:-4] + "_with_ae.csv")
    # merge_embs(test_file, emb_files, test_file[:-4] + "_with_ae.csv")
    emb_files = glob.glob('/home/notebook/data/group/app-series/ad_platform/in_un_embs' + "/abcn_cl_imei_embs_sum_*")
    # print(emb_files)
    # merge_embs(train_file, emb_files, train_file[:-4] + "_with_sum.csv")
    # merge_embs(test_file, emb_files, test_file[:-4] + "_with_sum.csv")
    # # print(emb_files)
    # merge_embs(train_file, emb_files, train_file[:-4] + "_with_sum.csv")
    # merge_embs(test_file, emb_files, test_file[:-4] + "_with_sum.csv")
    # get_rand_embs(train_file[:-4] + "_pre.csv")
    # get_rand_embs(test_file[:-4] + "_pre.csv")
    # add_ae_app_embs(train_file[:-4] + "_pre_cate.csv",'/home/notebook/data/personal/adT1_app_embs.csv')
    # add_ae_app_embs(test_file[:-4] + "_pre_cate.csv",'/home/notebook/data/personal/adT1_app_embs.csv')
    # compare_app(['/home/notebook/data/personal/adT1_app_embs.csv'], [train_file[:-4] + "_pre_cate.csv", test_file[:-4] + "_pre_cate.csv"])
    
    # NOTE for bc model
    # bc_emb_files = glob.glob('/home/notebook/data/group/app-series/ad_platform/ad_emb_202207_v2' + '/user*')
    # merge_bc_embs(train_file, bc_emb_files)
    # merge_bc_embs(test_file, bc_emb_files)
    # get_rand_embs(train_file[:-4] + "_pre_bc.csv")
    # get_rand_embs(test_file[:-4] + "_pre_bc.csv")
    # compare_imei(emb_files, bc_emb_files)
    # add_bc_app_embs(train_file[:-4] + "_pre_bc.csv", '/home/notebook/data/group/app-series/ad_platform/app_emb_ad_202207_v2.csv')
    # add_bc_app_embs(test_file[:-4] + "_pre_bc.csv", '/home/notebook/data/group/app-series/ad_platform/app_emb_ad_202207_v2.csv')


    # NOTE concat ae_embs & bc_embs for each imei
    # merge_2_embs(emb_files, bc_emb_files, '/home/notebook/data/group/app-series/ad_platform/in_un_embs' + "/adT1_whole_imei_embs.csv")
    # merge_embs(train_file, ['/home/notebook/data/group/app-series/ad_platform/in_un_embs' + "/adT1_whole_imei_embs.csv"])
    # merge_embs(test_file, ['/home/notebook/data/group/app-series/ad_platform/in_un_embs' + "/adT1_whole_imei_embs.csv"])
    
    # emb_files = glob.glob('/home/notebook/data/group/app-series/ad_platform/in_un_embs' + "/abcn_cl_imei_embs_ae_*")
    # bc_emb_files = glob.glob('/home/notebook/data/group/app-series/ad_platform/in_un_embs' + "/abcn_cl_imei_embs_bc_*")
    # merge_2_embs_2(emb_files, bc_emb_files, '/home/notebook/data/group/app-series/ad_platform/in_un_embs' + "/abcn_whole_imei_embs.csv")
    # merge_embs(train_file, ['/home/notebook/data/group/app-series/ad_platform/in_un_embs' + "/abcn_whole_imei_embs.csv"], train_file[:-4] + "_with_cat.csv")
    # merge_embs(test_file, ['/home/notebook/data/group/app-series/ad_platform/in_un_embs' + "/abcn_whole_imei_embs.csv"], test_file[:-4] + "_with_cat.csv")

    # NOTE gen imei-emb files
    emb_files = glob.glob('/home/notebook/data/group/app-series/pretrain/clean_train/' + "abcn_imei_embs_sum_*")
    bc_emb_files = glob.glob('/home/notebook/data/group/app-series/pretrain/clean_train/' + "abcn_imei_embs_b_*")

    # df_emb = pd.DataFrame()
    # for ef in tqdm(emb_files):
    #     df_tmp = pd.read_csv(ef, dtype=str)
    #     df_emb = pd.concat([df_emb, df_tmp])
    # print(len(df_emb))
    # df_emb = df_emb.sort_values(by=["imei"])
    # df_emb.to_csv('/home/notebook/data/group/app-series/pretrain/clean_train/imei_embs_sum.csv', index=False)

    # df_emb = pd.DataFrame()
    # for ef in tqdm(bc_emb_files):
    #     df_tmp = pd.read_csv(ef, dtype=str)
    #     df_emb = pd.concat([df_emb, df_tmp])
    # print(len(df_emb))
    # df_emb = df_emb.sort_values(by=["imei"])
    # df_emb.to_csv('/home/notebook/data/group/app-series/pretrain/clean_train/imei_embs_bc.csv', index=False)
    # df = pd.read_csv('/home/notebook/data/group/app-series/pretrain/clean_train/imei_embs_sum.csv', usecols=["imei"])
    # print(df.head())
    # print(df.tail())
    # df = pd.read_csv('/home/notebook/data/group/app-series/pretrain/clean_train/imei_embs_bc.csv', usecols=["imei"])
    # print(df.head())
    # print(df.tail())
    # merge_2_embs_2(emb_files, bc_emb_files, '/home/notebook/data/group/app-series/pretrain/clean_train/' + "abcn_whole_imei_embs.csv")
    # merge_2_embs_3('/home/notebook/data/group/app-series/pretrain/clean_train/imei_embs_sum.csv', '/home/notebook/data/group/app-series/pretrain/clean_train/imei_embs_bc.csv', '/home/notebook/data/group/app-series/pretrain/clean_train/' + "abcn_whole_imei_embs.csv")
    