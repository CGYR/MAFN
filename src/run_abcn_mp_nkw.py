
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
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam, lr_scheduler
from torch.utils.data import DataLoader,Dataset,TensorDataset
from torch.utils.data import RandomSampler,SequentialSampler,DistributedSampler
from torch.utils.tensorboard import SummaryWriter


sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from self_attention.config import PretrainedConfig
from self_attention.dataset import AdAppMaskDatasetABC
from self_attention.modelf import ABCN

from utils import EarlyStopping

import atexit
import subprocess as spc

import torch.distributed as dist
import torch.multiprocessing as mp

import json
from utils import parser_yaml
import argparse

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s - %(levelname)s： %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)


def set_seed(seed, n_gpu=4):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu>0:
        torch.cuda.manual_seed_all(seed)


def train(local_rank, args, model, train_dataset, eval_dataset=None, writer=None):

    early_stop = EarlyStopping(patience=args.train.early_stop_patience )

    #设置dataloader
    args.n_gpu = torch.cuda.device_count()
    args.train_batch_size = args.train.per_gpu_batch_size * args.n_gpu

    
    sampler = DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=sampler, batch_size=args.train_batch_size, num_workers=2)
    
    if not hasattr(args.train, 'gradient_accum_steps'): 
        args.train.gradient_accum_steps = 1

    # 设置优化器
    #多GPU
    model = DDP(model.to(local_rank), device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    optimizer = Adam(model.parameters(),lr=args.train.learning_rate,  weight_decay=args.train.weight_decay)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',patience=3, min_lr=1e-6,factor=0.2)
    if local_rank == 0:
        logger.info('***********start running************')
        logger.info('batch_size：{}'.format(args.train_batch_size))
        logger.info('max_steps:{}'.format(args.train.max_steps))

    global_step = args.train.start_epoch
    tr_loss,logging_loss, avg_ppl, tr_nb = 0.0,0.0,0.0,0
    step_eval_loss = np.Inf
    epoch_eval_loss = np.Inf
    best_loss = np.Inf
    
    #初始化  (torch.tensor(app_ids), torch.tensor(time_ids),  torch.tensor(duration_ids), torch.tensor(lengths), torch.tensor(labels))
    model.zero_grad()
    set_seed(args.train.seed)

    for idx in range(args.train.start_epoch, args.train.epoch):
        if local_rank == 0:    
            logger.info(('epoch={}'.format(idx)).center(20,'-'))
        for bid, batch in enumerate(train_dataloader):
            keep_app_ids, app_ids, type_ids, time_ids, cate_ids, mask_ids, mask_label_ids, labels, bc_app_ids, bc_masks, bc_label_dist, ages = [x.to(args.device) for x in batch]
            model.train() 
            pooled_out, bc_emb, fusion_emb, dae_loss, decoder_loss, decoder_fuse_loss, bottlneck_loss, mlm_loss, bc_loss, cl_loss, sem_loss = model(keep_app_ids, app_ids, type_ids, time_ids, cate_ids, mask_ids, mask_label_ids, bc_app_ids, bc_masks, bc_label_dist, ages)
        
            loss = dae_loss + decoder_loss + bottlneck_loss + mlm_loss + bc_loss + decoder_fuse_loss + cl_loss + sem_loss

            if args.n_gpu>1:
                loss = loss.mean() #
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.train.max_grad_norm)

            tr_loss += loss.item()

            optimizer.step()
            optimizer.zero_grad()
            
            # print(dae_loss, decoder_loss, bottlneck_loss, mlm_loss)
            
            # train logger 
            if global_step % args.train.logging_step==0:
                if local_rank == 0: # 多gpu，选择其中一个看效果就好
                    avg_ppl =round(np.exp((tr_loss - logging_loss) /(global_step +1- tr_nb)),4) # 分支度量
                    print("step={:d}, dae_loss={:.4f},neck_loss={:.4f},decoder_loss={:.4f},f_decoder_loss={:.4f},mlm_loss={:.4f},bc_loss={:.4f}, cl_loss={:.4f}, sem_loss={:.4f}, loss={:.4f}, avg_ppl={:.4f}".format(
                                    global_step, dae_loss.item(),bottlneck_loss.item(),decoder_loss.item(), decoder_fuse_loss.item(),mlm_loss.item(), bc_loss.item(), cl_loss.item(), sem_loss.item(), loss.item(), avg_ppl))
                    logging_loss = tr_loss
                    tr_nb = global_step 
            global_step += 1    


        # evaluate after each epoch
        with torch.no_grad(): 
            train_result = evaluate(args, model, train_dataset)
            eval_result = evaluate(args, model, eval_dataset)
        if local_rank == 0:
            logger.info('epoch={}, train:{}\n, eval:{}'.format(idx,train_result,eval_result))
            scheduler.step(eval_result['eval_loss'])

        if writer and local_rank == 0:
            writer.add_scalar('eval_loss',eval_result['eval_loss'],global_step=global_step)
            writer.add_scalar('eval_dae_loss',eval_result['dae_loss'],global_step=global_step)
            writer.add_scalar('eval_neck_loss',eval_result['neck_loss'],global_step=global_step)
            writer.add_scalar('eval_decoder_loss',eval_result['decoder_loss'],global_step=global_step)
            writer.add_scalar('eval_mlm_loss',eval_result['mlm_loss'],global_step=global_step)

            writer.add_scalar('train_loss',train_result['eval_loss'],global_step=global_step)
            writer.add_scalar('train_dae_loss',train_result['dae_loss'],global_step=global_step)
            writer.add_scalar('train_neck_loss',train_result['neck_loss'],global_step=global_step)
            writer.add_scalar('train_decoder_loss',train_result['decoder_loss'],global_step=global_step)
            writer.add_scalar('train_mlm_loss',train_result['mlm_loss'], global_step=global_step)
            writer.add_histogram('emb',pooled_out.data.cpu(),global_step=global_step)


        
        if eval_result['eval_loss'] < epoch_eval_loss and local_rank == 0:
            epoch_eval_loss = eval_result['eval_loss']
            best_loss = round(epoch_eval_loss,4)
            logger.info('epoch_{}_update_model'.format(idx))
            
            checkpoint_last = os.path.join(args.checkpoint_dir,'checkpoint_last')
            if not os.path.exists(checkpoint_last):  
                os.makedirs(checkpoint_last, mode=0o777, exist_ok=False)

            logger.info('saving model checkpoint to ...{}'.format(checkpoint_last))
            model_to_save = model.module if hasattr(model,'module') else model
            torch.save(model_to_save,os.path.join(checkpoint_last,'model.pt'))
        
            if (idx+1)%10 == 0:
                checkpoint_last = os.path.join(args.checkpoint_dir,'checkpoint_last')
                if not os.path.exists(checkpoint_last):  
                    os.makedirs(checkpoint_last, mode=0o777, exist_ok=False)

                logger.info('saving model checkpoint to ...{}'.format(checkpoint_last))
                model_to_save = model.module if hasattr(model,'module') else model
                torch.save(model_to_save,os.path.join(checkpoint_last,'model_{}.pt'.format(idx)))

        '''
        early_stop(eval_result['eval_loss'])
        # print(idx, args.train.epoch, args.train.max_steps)

        if early_stop.early_stop or global_step>=args.train.max_steps or idx>args.train.epoch: 
            signature = 'early_stop' if early_stop.early_stop else 'max-steps-done'
            logging.info(('{}'.format(signature)).center(50,'*'))
            break
        '''

    return best_loss


def evaluate(args, model, eval_dataset): 
    
    args.eval_batch_size = args.eval.per_gpu_batch_size * max(1, args.n_gpu)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, num_workers=0)

    eval_loss = 0.0
    tdae_loss = tbottlneck_loss = tdecoder_loss = tdecoder_fuse_loss = tmlm_loss = tbc_loss = tcl_loss = tsem_loss = 0.0
    eval_steps = 0 

    model.eval()
    for batch in eval_dataloader:
        keep_app_ids, app_ids, type_ids, time_ids, cate_ids, mask_ids, mask_label_ids, labels, bc_app_ids, bc_masks, bc_label_dist, ages = [x.to(args.device) for x in batch]
        pooled_out, bc_emb, fuse_emb, dae_loss, decoder_loss, decoder_fuse_loss , bottlneck_loss, mlm_loss, bc_loss, cl_loss, sem_loss = model(keep_app_ids, app_ids, type_ids, time_ids, cate_ids, mask_ids, mask_label_ids, bc_app_ids, bc_masks, bc_label_dist, ages)
        loss = dae_loss + decoder_loss + decoder_fuse_loss + bottlneck_loss + mlm_loss + bc_loss + cl_loss + 0.1 * sem_loss

        if args.n_gpu>1:  
            loss = loss.mean()
        eval_loss += loss.item()
        tdae_loss += dae_loss.item()
        tbottlneck_loss += bottlneck_loss.item()
        tdecoder_loss += decoder_loss.item()
        tmlm_loss += mlm_loss.item()
        tbc_loss += bc_loss.item()
        tdecoder_fuse_loss += decoder_fuse_loss.item()
        tcl_loss += cl_loss.item()
        tsem_loss += sem_loss.item()

        eval_steps +=1 

    eval_loss = eval_loss/eval_steps
    tdae_loss = tdae_loss/eval_steps
    tbottlneck_loss = tbottlneck_loss/eval_steps
    tdecoder_loss = tdecoder_loss/eval_steps
    tmlm_loss = tmlm_loss/eval_steps
    tbc_loss = tbc_loss/eval_steps
    tdecoder_fuse_loss /= eval_steps
    tcl_loss /= eval_steps
    tsem_loss /= eval_steps
    perplexity = np.exp(eval_loss)
    result = {'eval_loss': eval_loss, 'dae_loss':tdae_loss,'neck_loss':tbottlneck_loss,
             'decoder_loss':tdecoder_loss, 'decoder_fuse_loss':tdecoder_fuse_loss,'mlm_loss':tmlm_loss, 'bc_loss':tbc_loss, 'cl_loss':tcl_loss, 'sem_loss':tsem_loss, 'ppl': perplexity}

    return result


def main_worker(local_rank, nprocs):
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl',init_method='tcp://127.0.0.1:23456',
                            rank=local_rank,
                            world_size=nprocs)

    parser = argparse.ArgumentParser()
    
    base_path = '/home/notebook/code/personal/appseq'
    # : [imei, pack_keep_name, label, install_seq, install_time, uninstall_seq, uninstall_time, cutoff_date]
    # [uid, pack_keep_name, label(means this user whether positive?), install_seq, install_time(split by ',', 2020-03-15 21:10:50), uninstall_seq, uninstall_time, cutoff_date(20200731)]
    # file = '/home/notebook/data/personal/reno5/clean_output/reno5_merge_all1_offdate.csv'
    train_file = '/home/notebook/data/group/app-series/pretrain_game/preprocess/abcn_train_{}.csv'.format(local_rank)

    df = pd.read_csv(train_file, sep='\t', dtype=str, nrows=None)
    print(train_file)
    df["age"].fillna("0", inplace=True)
    # df.fillna("unk", inplace=True)
    # df = df[df.imei==860079043750010]
    # df['cutoff_date'] = df['cutoff_date'].astype(str)

    # vocab_dic = json.load(open("/home/notebook/data/personal//mete/aetn_app_iaa_vocab.json",'r')) # : app-id table
    vocab_dic = json.load(open("/home/notebook/data/group/app-series/pretrain_game/preprocess/app_vocab.json",'r')) # : app-id table

    with open('/home/notebook/code/personal/appseq/run_abcn.yaml','r') as fp:
        cfg = yaml.load(fp)

    cfg['embedding']['app_vocab_size'] = cfg['autoencoder']['app_vocab_size'] = len(vocab_dic)
    # cfg['embedding']['start_time_vocab_size'] = len(cfg['embedding']['start_time_discrete_list']) + 2
    cfg['encoder']['max_position_embeddings'] = cfg['max_seq_len']
    
    pretrain_config = PretrainedConfig(**cfg['encoder'])
    autoencoder_config = cfg['autoencoder']
    common_config = cfg
    common_config.pop('encoder')
    common_config.pop('autoencoder')
    # print(common_config)

    if common_config['MODEL'] == 'TRAIN':
        args = parser_yaml(common_config)
        # args.device = local_rank  #torch.cuda.device_count()
        args.device = torch.device('cuda:{}'.format(local_rank))
        if local_rank == 0:
            print('TRAIN'.center(30,'-'))
            args.checkpoint_dir = os.path.join(os.path.join(base_path,'model/abcn_mp','pretrain_nkw_'+time.strftime('%Y%m%d',time.localtime())))
            logger.info('checkpoint_dir:{}'.format(args.checkpoint_dir))

            if not os.path.exists(args.checkpoint_dir):
                os.makedirs(args.checkpoint_dir, mode=0o777, exist_ok=False)
            json.dump(cfg,open(os.path.join(args.checkpoint_dir,'config.json'),'w+'))

            writer = SummaryWriter(log_dir=os.path.join(base_path,'log/abcn_mp'),filename_suffix='_nkw')
            atexit.register(writer.close)


        df_train = df.sample(frac=0.8, replace=False, weights=None, random_state= args.train.seed, axis=None)
        df_eval = df[~df.index.isin(df_train.index)]
        train_dataset = AdAppMaskDatasetABC(df_train, vocab_dic, common_config)
        eval_dataset = AdAppMaskDatasetABC(df_eval,vocab_dic, common_config)
        model = ABCN(pretrain_config,autoencoder_config, common_config)
        if local_rank == 0:
            best_loss= train(local_rank, args,model,train_dataset,eval_dataset, writer= writer)
        else:
            best_loss= train(local_rank, args,model,train_dataset,eval_dataset)
        if local_rank == 0:
            target_dir = args.checkpoint_dir+'_{}'.format(round(best_loss,4))
            spc.call(['mv '+args.checkpoint_dir+' ' + target_dir],shell=True)

    dist.destroy_process_group()

    

if __name__ == '__main__':
    # ONLY FOR Multi GPUs !!
    nprocs = torch.cuda.device_count() # 1 proc use 1 GPU
    if nprocs <= 1:
        assert 1==0, "ONLY FOR Multi GPUs !!"
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(i) for i in range(nprocs)])
    print(nprocs)
    mp.spawn(main_worker, nprocs=nprocs, args=(nprocs, ), join=True) # multi process
    
    