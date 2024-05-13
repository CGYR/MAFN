
import copy
import yaml
import json
import os
import sys
import random
import logging
import time
import pandas as pd
import numpy as np 
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, lr_scheduler
from torch.utils.data import DataLoader,Dataset,TensorDataset
from torch.utils.data import RandomSampler,SequentialSampler,DistributedSampler
from torch.utils.tensorboard import SummaryWriter


sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from self_attention.config import PretrainedConfig
from self_attention.dataset import HeterAppDataset,HeterAppMaskDataset,AdAppMaskDatasetABC
from self_attention.modelf import ABCN

from utils import EarlyStopping

import atexit
import subprocess as spc


logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s - %(levelname)s： %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)


def set_seed(seed, n_gpu=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu>0:
        torch.cuda.manual_seed_all(seed)


def predict(model, eval_dataset,batch_size=128): 
    # args.eval_batch_size = args.eval.per_gpu_batch_size * max(1, args.n_gpu)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=batch_size, num_workers=2)

    eval_steps = 0
    user_embeddings_ae = []
    user_embeddings_bc = []
    user_embeddings_sum = [] 

    model.eval()
    for idx, batch in enumerate(eval_dataloader):
        keep_app_ids, app_ids, type_ids, time_ids, cate_ids, mask_ids, mask_label_ids, labels, bc_app_ids, bc_masks, bc_label_dist, ages = [x.to(model.encoder.device) for x in batch]
        pooled_out, bc_emb, fuse_emb, dae_loss, decoder_loss, decoder_fuse_loss , bottlneck_loss, mlm_loss, bc_loss, cl_loss, sem_loss = model(keep_app_ids, app_ids, type_ids, time_ids, cate_ids, mask_ids, mask_label_ids, bc_app_ids, bc_masks, bc_label_dist, ages, predict=True)
        # eval_steps +=1 
        user_embeddings_ae.append(pooled_out.detach().cpu().numpy())
        user_embeddings_bc.append(bc_emb.detach().cpu().numpy())
        user_embeddings_sum.append(fuse_emb.detach().cpu().numpy())
        if idx %2000==0:
            print('-----{}-----'.format(idx))

    user_embeddings_ae = np.concatenate(user_embeddings_ae)
    user_embeddings_bc = np.concatenate(user_embeddings_bc)
    user_embeddings_sum = np.concatenate(user_embeddings_sum)

    df_embedding_ae = pd.DataFrame(user_embeddings_ae)
    df_embedding_bc = pd.DataFrame(user_embeddings_bc)
    df_embedding_sum = pd.DataFrame(user_embeddings_sum)

    return df_embedding_ae, df_embedding_bc, df_embedding_sum


def get_app_embs(base_path, vocab_dic, with_cate=False):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'  #torch.cuda.device_count(
    model = torch.load(os.path.join(base_path,'model/abcn_mp/adT1_20221216_11.4388/checkpoint_last/model.pt'), map_location = torch.device(device))
        
    app_embeddings = model.app_embedding(torch.Tensor(range(len(vocab_dic))).long().to(device)).detach().cpu().numpy()
    app_name = [k for k in vocab_dic]
    # add cate embs
    if with_cate:
        app_cate_file = "/home/notebook/data/personal//mete/app_name_to_cate_id.json"
        with open(app_cate_file, 'r') as f:
            app2cate = json.load(f)
            app2cate['pad'] = 66
            app2cate['mask'] = 66
            app2cate['unk'] = 66
        cate_list = [int(app2cate[k]) for k in app_name]
        cate_embeddings = model.cate_embedding(torch.Tensor(cate_list).long().to(device)).detach().cpu().numpy()
        assert app_embeddings.shape == cate_embeddings.shape
        app_embeddings = app_embeddings + cate_embeddings

    df_app = pd.DataFrame(app_embeddings)
    df_app['pack_name'] = app_name
    df_app.to_csv('/home/notebook/data/personal//adT1_app_embs_abcn.csv', index=False)


if __name__ == '__main__':
    # ONLY SUPPORT 1 GPU or CPUs! For more GPUs, see run_aetn_mp.py
    import json
    from utils import parser_yaml

    import argparse
    parser = argparse.ArgumentParser()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    base_path = '/home/notebook/code/personal/appseq'
    # yr: [imei, pack_keep_name, label, install_seq, install_time, uninstall_seq, uninstall_time, cutoff_date]
    # [uid, pack_keep_name, label(means this user whether positive?), install_seq, install_time(split by ',', 2020-03-15 21:10:50), uninstall_seq, uninstall_time, cutoff_date(20200731)]
    # file = '/home/notebook/data/personal/reno5/clean_output/reno5_merge_all1_offdate.csv'

    vocab_dic = json.load(open("/home/notebook/data/personal//mete/app_iaa_vocab_with_cate.json",'r')) # yr: app-id table

    with open('/home/notebook/code/personal/appseq/gen_abcn.yaml','r') as fp:
        cfg = yaml.load(fp)
    
    # NOTE get_app_embs only for output app embs
    # if cfg['eval']['idx'] == 0:
    #     get_app_embs(base_path, vocab_dic)

    cfg['embedding']['app_vocab_size'] = cfg['autoencoder']['app_vocab_size'] = len(vocab_dic)
    # cfg['embedding']['start_time_vocab_size'] = len(cfg['embedding']['start_time_discrete_list']) + 2
    cfg['encoder']['max_position_embeddings'] = cfg['max_seq_len']
    
    pretrain_config = PretrainedConfig(**cfg['encoder'])
    autoencoder_config = cfg['autoencoder']
    common_config = cfg
    common_config.pop('encoder')
    common_config.pop('autoencoder')
    # print(common_config)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'  #torch.cuda.device_count(

    if common_config['MODEL'] == 'TRAIN':
        file = '/home/notebook/data/group/app-series/pretrain/aetn_recent3m_train.csv'
    elif common_config['MODEL'] =='TEST':
        # file = '/home/notebook/data/group/app-series/pretrain/aetn_recent3m_test.csv'
        # file = '/home/notebook/data/group/app-series/ad_platform/un_in_data_20221201/test/aetn_ad_{}.csv'.format(cfg['eval']['idx'])
        train_file = '/home/notebook/data/group/app-series/pretrain/clean_train/abcn_train_{}.csv'.format(cfg['eval']['idx'])
        val_file = '/home/notebook/data/group/app-series/pretrain/clean_train/abcn_test_{}.csv'.format(cfg['eval']['idx'])

    df_train = pd.read_csv(train_file, sep='\t', dtype=str, nrows=None)
    df_eval = pd.read_csv(val_file, sep='\t', dtype=str, nrows=None)
    print(val_file)
    print(len(df_eval))
    df_train["age"].fillna("0", inplace=True)
    df_eval["age"].fillna("0", inplace=True)
    # df = pd.read_csv(file, sep='\t', dtype=str)
    df = pd.concat([df_train, df_eval])
    df.fillna("unk", inplace=True)
    # df = df[df.imei==860079043750010]
    # df['cutoff_date'] = df['cutoff_date'].astype(str)

    if common_config['MODEL'] =='TEST':
        print('TEST'.center(30,'-'))
        # df_train = df.sample(frac=0.8, replace=False,random_state=42) 
        # df_eval = df[~df.index.isin(df_train.index)]
        df_eval = df
        # df_train.reset_index(drop=True, inplace=True)
        df_eval.reset_index(drop=True, inplace=True)
        print(df_eval.shape)
        # train_dataset = HeterAppMaskDataset(df_train, vocab_dic, common_config)
        eval_dataset = AdAppMaskDatasetABC(df_eval, vocab_dic, common_config)
        
        print("load model...")
        # model/aetn/downstream_20201124_4.461/checkpoint_last/model.pt  #--no-appmask-model
        # model = torch.load(os.path.join(base_path,'model/aetn/downstream_20201211_8.0264/checkpoint_last/model.pt'), map_location = torch.device(device)) #20201120(64dim)
        model = torch.load(os.path.join(base_path,'model/abcn_mp/pretrain_880w_20230217_17.3293/checkpoint_last/model.pt'), map_location = torch.device(device))
        model.is_finetune=False

        df_eval_emb_ae, df_eval_emb_bc, df_eval_emb_sum = predict(model, eval_dataset)
        # assert df_eval_embedding.shape[0] == df_eval.shape[0]
        # df_eval_embedding[['imei','label']] = df_eval[['imei','label']]
        df_eval_emb_ae['imei'] = df_eval['imei'].astype(str)
        df_eval_emb_bc['imei'] = df_eval['imei'].astype(str)
        df_eval_emb_sum['imei'] = df_eval['imei'].astype(str)
        # df_eval_embedding['label'] = df_eval['label']
        # df_eval_emb_ae.to_csv('/home/notebook/data/group/app-series/ad_platform/in_un_embs/abcn_cl_imei_embs_ae_{}.csv'.format(cfg['eval']['idx']),index=False)
        # df_eval_emb_bc.to_csv('/home/notebook/data/group/app-series/ad_platform/in_un_embs/abcn_cl_imei_embs_bc_{}.csv'.format(cfg['eval']['idx']),index=False)
        # df_eval_emb_sum.to_csv('/home/notebook/data/group/app-series/ad_platform/in_un_embs/abcn_cl_imei_embs_sum_{}.csv'.format(cfg['eval']['idx']),index=False)
        df_eval_emb_ae.to_csv('/home/notebook/data/group/app-series/pretrain/clean_train/abcn_imei_embs_a_{}.csv'.format(cfg['eval']['idx']),index=False)
        df_eval_emb_bc.to_csv('/home/notebook/data/group/app-series/pretrain/clean_train/abcn_imei_embs_b_{}.csv'.format(cfg['eval']['idx']),index=False)
        df_eval_emb_sum.to_csv('/home/notebook/data/group/app-series/pretrain/clean_train/abcn_imei_embs_sum_{}.csv'.format(cfg['eval']['idx']),index=False)
        print('test dataset done.')



















