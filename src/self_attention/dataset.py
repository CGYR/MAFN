import os 
import logging
import random
import yaml
import json
import time
import operator
import numpy as np 
import pandas as pd
from datetime import datetime
import torch
from collections import defaultdict,Counter
# import torch.nn as nn
from torch.utils.data import DataLoader,Dataset,TensorDataset
from torch.utils.data import RandomSampler,SequentialSampler,DistributedSampler

class AdAppMaskDataset(Dataset):
    def __init__(self, df, vocab_dic, conf):
        super().__init__()
        self.conf = conf
        self.df = df
        self.vocab_dic = vocab_dic
      
    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):

        # install_max_seq_len = self.conf['install_max_seq_len']
        # uninstall_max_seq_len = self.conf['uninstall_max_seq_len']
    
        # in_app_ids =  np.zeros(install_max_seq_len, dtype=np.int32)
        # un_app_ids =  np.zeros(uninstall_max_seq_len, dtype=np.int32)

        max_seq_len = self.conf['max_seq_len']
        max_seq_len = max_seq_len -1  # keep 1 position for app retention 
        vocab_size = self.conf['embedding']['app_vocab_size'] # -1 exclude pad if necessary

        app_ids = np.zeros(max_seq_len, dtype=np.int32)
        app_time_ids = np.zeros(max_seq_len, dtype=np.int32)
        type_ids = np.zeros(max_seq_len,dtype=np.int32)
        masks = np.zeros(max_seq_len)  # mask attention
        labels = np.zeros(1, dtype=np.float32)  #默认为0
        mask_label_ids = np.zeros(max_seq_len,dtype=np.int32)-1 #默认为-1

        keep_app_ids = torch.zeros(vocab_size, dtype=torch.float)
    
        # labels[0] = self.df.iloc[i]['label']
        unk_id = self.vocab_dic['unk']

        keep_apps = self.df.iloc[i]['pack_keep_name'].split(',')
        app_setbits = torch.tensor([self.vocab_dic.get(app,unk_id) for app in keep_apps])
        keep_app_ids.index_fill_(dim=0,index=app_setbits,value=1.0)
        # assert not torch.equal(keep_app_ids, torch.zeros(keep_app_ids.shape, dtype=keep_app_ids.dtype)), keep_apps
        
        # if self.is_training:
        #      keep_app_ids = torch.mul(self.noise_func.sample(keep_app_ids.shape),keep_app_ids)

        # install and uninstall
        ins_app = self.df.iloc[i]['install_seq'].split(',')
        ins_time = self.df.iloc[i]['install_time'].split(',')
        # ins_time = list(map(lambda x: time.strptime(x, '%Y-%m-%d %H:%M:%S'),ins_time))
        uns_app = self.df.iloc[i]['uninstall_seq'].split(',')
        uns_time = self.df.iloc[i]['uninstall_time'].split(',')
        # cutoff_date = datetime.strptime(self.df.iloc[i]['cutoff_date'],'%Y%m%d')
        cutoff_date = datetime.strptime("20221008",'%Y%m%d')

        ins_seq = list(zip(ins_app, ins_time, [2]*len(ins_app)))  #  2 for install action-type
        uns_seq = list(zip(uns_app, uns_time, [3]*len(uns_app)))  #  3 for uninstall action-type
        seq = ins_seq + uns_seq
        seq = sorted(seq, key=operator.itemgetter(1))[::-1][:max_seq_len] #降序排列

        seq.append(seq[-1]) # only for code friendly 
        # retention app 
        xr = self.conf['embedding']['start_time_discrete_list']
        cnt = 0
        for idx, (app, app_time, action_type) in enumerate(seq[:-1]):
            if app_time == "unk":
                app_time = "2022-09-01 00:00:00"
            cur_time = datetime.strptime(app_time,'%Y-%m-%d %H:%M:%S')
            # if (cutoff_date-cur_time).days<30:
            #     continue
            # if (cutoff_date.date()-cur_time.date()).days>120:  #四个月之内
            #     continue
            # prev_time = datetime.strptime(seq[idx+1][1], '%Y-%m-%d %H:%M:%S')
            masks[cnt] = 1
            type_ids[cnt] = action_type

            #mask app to predict
            app_id = self.vocab_dic.get(app)
            if app_id == None:
                app_id = 1
            
            if random.random()<self.conf['mlm_prob']:
                mask_label_ids[cnt] = app_id  #预测标签

                if random.random() < 0.8:
                    app_ids[cnt] = unk_id  # NOTE 80%的 mask, 理论需要mask_id,因为序列中没有unk_id,用unk_id代替 
                elif random.random() <0.5:
                    app_ids[cnt] = app_id # 10%原始
                else:  
                    sample_item = random.sample(list(self.vocab_dic),1)[0] #10%其他
                    app_ids[cnt] = self.vocab_dic.get(sample_item)
            else:
                app_ids[cnt] = app_id

            # app_ids[cnt] = self.vocab_dic.get(app,unk_id)
            # days = np.searchsorted(xr,(cur_time.date() - prev_time.date()).days, side='right') # 0 for pad
            # days = np.searchsorted(xr,(cutoff_date.date() - cur_time.date()).days, side='right') # 0 for pad
            days = (cutoff_date.date() - cur_time.date()).days # 0 for pad
            days = max(days, 0)
            days = min(days, 92)
            app_time_ids[cnt] = 92 - days # 更合理方式：cur_time - start_date, 后面再改
            cnt +=1
        return (keep_app_ids, torch.tensor(app_ids), torch.tensor(type_ids), torch.tensor(app_time_ids), torch.tensor(masks), torch.tensor(mask_label_ids),torch.tensor(labels))


class AdAppMaskDatasetf(Dataset):
    def __init__(self, df, vocab_dic, conf):
        super().__init__()
        self.conf = conf
        self.df = df
        self.vocab_dic = vocab_dic
      
    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):

        # install_max_seq_len = self.conf['install_max_seq_len']
        # uninstall_max_seq_len = self.conf['uninstall_max_seq_len']
    
        # in_app_ids =  np.zeros(install_max_seq_len, dtype=np.int32)
        # un_app_ids =  np.zeros(uninstall_max_seq_len, dtype=np.int32)

        max_seq_len = self.conf['max_seq_len']
        max_seq_len = max_seq_len -1  # keep 1 position for app retention 
        vocab_size = self.conf['embedding']['app_vocab_size'] # -1 exclude pad if necessary

        app_ids = np.zeros(max_seq_len, dtype=np.int32)
        app_time_ids = np.zeros(max_seq_len, dtype=np.int32)
        type_ids = np.zeros(max_seq_len,dtype=np.int32)
        cate_ids = np.zeros(max_seq_len,dtype=np.int32)
        masks = np.zeros(max_seq_len)  # mask attention
        labels = np.zeros(1, dtype=np.float32)  #默认为0
        mask_label_ids = np.zeros(max_seq_len,dtype=np.int32)-1 #默认为-1

        keep_app_ids = torch.zeros(vocab_size, dtype=torch.float)
    
        # labels[0] = self.df.iloc[i]['label']
        unk_id = self.vocab_dic['unk']

        keep_apps = self.df.iloc[i]['pack_keep_name'].split(',')
        app_setbits = torch.tensor([self.vocab_dic.get(app,unk_id) for app in keep_apps])
        keep_app_ids.index_fill_(dim=0,index=app_setbits,value=1.0)
        # assert not torch.equal(keep_app_ids, torch.zeros(keep_app_ids.shape, dtype=keep_app_ids.dtype)), keep_apps
        
        # if self.is_training:
        #      keep_app_ids = torch.mul(self.noise_func.sample(keep_app_ids.shape),keep_app_ids)

        # install and uninstall
        ins_app = self.df.iloc[i]['install_seq'].split(',')
        ins_time = self.df.iloc[i]['install_time'].split(',')
        ins_cate = self.df.iloc[i]['install_cate'].split(',')
        # ins_time = list(map(lambda x: time.strptime(x, '%Y-%m-%d %H:%M:%S'),ins_time))
        uns_app = self.df.iloc[i]['uninstall_seq'].split(',')
        uns_time = self.df.iloc[i]['uninstall_time'].split(',')
        uns_cate = self.df.iloc[i]['uninstall_cate'].split(',')
        # cutoff_date = datetime.strptime(self.df.iloc[i]['cutoff_date'],'%Y%m%d')
        cutoff_date = datetime.strptime("20220901",'%Y%m%d')

        ins_seq = list(zip(ins_app, ins_time, [2]*len(ins_app), ins_cate))  #  2 for install action-type
        uns_seq = list(zip(uns_app, uns_time, [3]*len(uns_app), uns_cate))  #  3 for uninstall action-type
        seq = ins_seq + uns_seq
        seq = sorted(seq, key=operator.itemgetter(1))[::-1][:max_seq_len] #降序排列

        seq.append(seq[-1]) # only for code friendly 
        # retention app 
        xr = self.conf['embedding']['start_time_discrete_list']
        cnt = 0
        for idx, (app, app_time, action_type, app_cate) in enumerate(seq[:-1]):
            if app_time == "unk":
                app_time = "2022-07-01 00:00:00"
            cur_time = datetime.strptime(app_time,'%Y-%m-%d %H:%M:%S')
            
            masks[cnt] = 1
            type_ids[cnt] = action_type
            cate_ids[cnt] = int(app_cate) # NOTE category save as id

            #mask app to predict
            app_id = self.vocab_dic.get(app)
            if app_id == None:
                app_id = 1
            
            if random.random()<self.conf['mlm_prob']:
                mask_label_ids[cnt] = app_id  #预测标签

                if random.random() < 0.8:
                    app_ids[cnt] = unk_id  # NOTE 80%的 mask, 理论需要mask_id,因为序列中没有unk_id,用unk_id代替 
                elif random.random() <0.5:
                    app_ids[cnt] = app_id # 10%原始
                else:  
                    sample_item = random.sample(list(self.vocab_dic),1)[0] #10%其他
                    app_ids[cnt] = self.vocab_dic.get(sample_item)
            else:
                app_ids[cnt] = app_id

            # app_ids[cnt] = self.vocab_dic.get(app,unk_id)
            # days = np.searchsorted(xr,(cur_time.date() - prev_time.date()).days, side='right') # 0 for pad
            # days = np.searchsorted(xr,(cutoff_date.date() - cur_time.date()).days, side='right') # 0 for pad
            days = (cutoff_date.date() - cur_time.date()).days # 0 for pad
            days = max(days, 0)
            days = min(days, 92)
            app_time_ids[cnt] = 92 - days # 更合理方式：cur_time - start_date, 后面再改
            cnt +=1
        return (keep_app_ids, torch.tensor(app_ids), torch.tensor(type_ids), torch.tensor(app_time_ids), torch.tensor(cate_ids), torch.tensor(masks), torch.tensor(mask_label_ids),torch.tensor(labels))


class AppMaskFinetuneDataset(Dataset):
    def __init__(self, df, vocab_dic, conf):
        super().__init__()
        self.conf = conf
        self.df = df
        self.vocab_dic = vocab_dic
      
    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):

        # install_max_seq_len = self.conf['install_max_seq_len']
        # uninstall_max_seq_len = self.conf['uninstall_max_seq_len']
    
        # in_app_ids =  np.zeros(install_max_seq_len, dtype=np.int32)
        # un_app_ids =  np.zeros(uninstall_max_seq_len, dtype=np.int32)

        max_seq_len = self.conf['max_seq_len']
        max_seq_len = max_seq_len -1  # keep 1 position for app retention 
        vocab_size = self.conf['embedding']['app_vocab_size'] # -1 exclude pad if necessary

        app_ids = np.zeros(max_seq_len, dtype=np.int32)
        app_time_ids = np.zeros(max_seq_len, dtype=np.int32)
        type_ids = np.zeros(max_seq_len,dtype=np.int32)
        masks = np.zeros(max_seq_len)  # mask attention
        mask_label_ids = np.zeros(max_seq_len,dtype=np.int32)-1 #默认为-1
        keep_app_ids = torch.zeros(vocab_size, dtype=torch.float)

        # for finetune classifier
        uids = np.zeros(1, dtype=np.float32)
        aids = np.zeros(1, dtype=np.float32)
        labels = np.zeros(1, dtype=np.float32)  #默认为0
    
        uids[0] = self.df.iloc[i]['uid']
        aids[0] = self.df.iloc[i]['app']
        labels[0] = self.df.iloc[i]['label']

        unk_id = self.vocab_dic['unk']

        keep_apps = self.df.iloc[i]['pack_keep_name'].split(',')
        app_setbits = torch.tensor([self.vocab_dic.get(app,unk_id) for app in keep_apps])
        keep_app_ids.index_fill_(dim=0,index=app_setbits,value=1.0)
        # assert not torch.equal(keep_app_ids, torch.zeros(keep_app_ids.shape, dtype=keep_app_ids.dtype)), keep_apps
        
        # if self.is_training:
        #      keep_app_ids = torch.mul(self.noise_func.sample(keep_app_ids.shape),keep_app_ids)

        # install and uninstall
        ins_app = self.df.iloc[i]['install_seq'].split(',')
        ins_time = self.df.iloc[i]['install_time'].split(',')
        # ins_time = list(map(lambda x: time.strptime(x, '%Y-%m-%d %H:%M:%S'),ins_time))
        uns_app = self.df.iloc[i]['uninstall_seq'].split(',')
        uns_time = self.df.iloc[i]['uninstall_time'].split(',')
        cutoff_date = datetime.strptime(self.df.iloc[i]['cutoff_date'],'%Y%m%d')
        
        ins_seq = list(zip(ins_app, ins_time, [2]*len(ins_app)))  #  2 for install action-type
        uns_seq = list(zip(uns_app, uns_time, [3]*len(uns_app)))  #  3 for uninstall action-type
        seq = ins_seq + uns_seq
        seq = sorted(seq, key=operator.itemgetter(1))[::-1][:max_seq_len] #降序排列

        seq.append(seq[-1]) # only for code friendly 
        # retention app 
        # xr = self.conf['embedding']['start_time_discrete_list']
        cnt = 0
        for idx, (app, app_time, action_type) in enumerate(seq[:-1]):
            cur_time = datetime.strptime(app_time,'%Y-%m-%d %H:%M:%S')
            
            masks[cnt] = 1
            type_ids[cnt] = action_type

            #mask app to predict
            
            if random.random()<self.conf['mlm_prob']:
                mask_label_ids[cnt] = self.vocab_dic.get(app)  #预测标签

                if random.random() < 0.8:
                    app_ids[cnt] = unk_id  # NOTE 80%的 mask, 理论需要mask_id,因为序列中没有unk_id,用unk_id代替 
                elif random.random() <0.5:
                    app_ids[cnt] = self.vocab_dic.get(app) # 10%原始
                else:  
                    sample_item = random.sample(list(self.vocab_dic),1)[0] #10%其他
                    app_ids[cnt] = self.vocab_dic.get(sample_item)
            else:
                app_ids[cnt] = self.vocab_dic.get(app)

            # app_ids[cnt] = self.vocab_dic.get(app,unk_id)
            # days = np.searchsorted(xr,(cur_time.date() - prev_time.date()).days, side='right') # 0 for pad
            # days = np.searchsorted(xr,(cutoff_date.date() - cur_time.date()).days, side='right') # 0 for pad
            days = (cutoff_date.date() - cur_time.date()).days # 0 for pad
            days = max(days, 0)
            days = min(days, 92)
            app_time_ids[cnt] = 92 - days # 更合理方式：cur_time - start_date, 后面再改
            cnt +=1
        
        return (keep_app_ids, torch.tensor(app_ids), torch.tensor(type_ids), torch.tensor(app_time_ids), torch.tensor(masks), torch.tensor(mask_label_ids), torch.tensor(uids), torch.tensor(aids), torch.tensor(labels))


class AdAppMaskDatasetABC(Dataset):
    def __init__(self, df, vocab_dic, conf):
        super().__init__()
        self.conf = conf
        self.df = df
        self.vocab_dic = vocab_dic
      
    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):

        max_seq_len = self.conf['max_seq_len']
        vocab_size = self.conf['embedding']['app_vocab_size'] # -1 exclude pad if necessary
        # gender = self.df.iloc[i]['gender']
        # province = self.df.iloc[i]['province']
        age = self.df.iloc[i]['age']
        if age == "unk":
            age = 0
        else:
            age = int(age)

        # bc data
        bc_app_ids = np.zeros(max_seq_len, dtype=np.int32)
        bc_masks = np.zeros(max_seq_len, dtype=np.int32)
        
        # bc_app = self.df.iloc[i]['open_seq'].split(',')
        # bc_ids = [self.vocab_dic[app] for app in bc_app]
        bc_ids = self.df.iloc[i]['open_seq'].split(',')
        if bc_ids == ['unk']:
            bc_ids = [0]
        bc_ids = list(map(int, bc_ids))
        bc_index = len(bc_ids) // 2 # select [:half] apps to predict the dist of [half:] apps
        if bc_index<2:
            bc_ids = [0,0,0,0]
            bc_index = 2
        bc_label_dict = get_appcnt_dist(bc_ids[bc_index:])
        bc_app_index, bc_app_value = zip(*bc_label_dict.items())
        bc_label_dist = np.zeros(len(self.vocab_dic), dtype=np.float32)
        bc_label_dist.put(bc_app_index, bc_app_value)
        bc_ids = bc_ids[:bc_index]
        for bc_idx, bc_app_id in enumerate(bc_ids[::-1][:max_seq_len]): 
            bc_masks[max_seq_len-1-bc_idx] = 1
            bc_app_ids[max_seq_len-1-bc_idx] = bc_app_id
        
    
        # ae data
        max_seq_len = max_seq_len -1  # keep 1 position for app retention 
        app_ids = np.zeros(max_seq_len, dtype=np.int32)
        app_time_ids = np.zeros(max_seq_len, dtype=np.int32)
        type_ids = np.zeros(max_seq_len,dtype=np.int32)
        cate_ids = np.zeros(max_seq_len,dtype=np.int32)
        masks = np.zeros(max_seq_len)  # mask attention
        labels = np.zeros(1, dtype=np.float32)  #默认为0
        mask_label_ids = np.zeros(max_seq_len,dtype=np.int32)-1 #默认为-1

        keep_app_ids = torch.zeros(vocab_size, dtype=torch.float)

        # labels[0] = self.df.iloc[i]['label']
        unk_id = self.vocab_dic['unk']

        keep_apps = self.df.iloc[i]['pack_keep_name'].split(',')
        app_setbits = torch.tensor([self.vocab_dic.get(app,unk_id) for app in keep_apps])
        keep_app_ids.index_fill_(dim=0,index=app_setbits,value=1.0)
        # assert not torch.equal(keep_app_ids, torch.zeros(keep_app_ids.shape, dtype=keep_app_ids.dtype)), keep_apps
        
        # if self.is_training:
        #      keep_app_ids = torch.mul(self.noise_func.sample(keep_app_ids.shape),keep_app_ids)

        # install and uninstall
        ins_app = self.df.iloc[i]['install_seq'].split(',')
        ins_time = self.df.iloc[i]['install_time'].split(',')
        ins_cate = self.df.iloc[i]['install_cate'].split(',')
        # ins_time = list(map(lambda x: time.strptime(x, '%Y-%m-%d %H:%M:%S'),ins_time))
        uns_app = self.df.iloc[i]['uninstall_seq'].split(',')
        uns_time = self.df.iloc[i]['uninstall_time'].split(',')
        uns_cate = self.df.iloc[i]['uninstall_cate'].split(',')
        # cutoff_date = datetime.strptime(self.df.iloc[i]['cutoff_date'],'%Y%m%d')
        cutoff_date = datetime.strptime("20230514",'%Y%m%d')

        ins_seq = list(zip(ins_app, ins_time, [2]*len(ins_app), ins_cate))  #  2 for install action-type
        uns_seq = list(zip(uns_app, uns_time, [3]*len(uns_app), uns_cate))  #  3 for uninstall action-type
        seq = ins_seq + uns_seq
        seq = sorted(seq, key=operator.itemgetter(1))[::-1][:max_seq_len] #降序排列

        seq.append(seq[-1]) # only for code friendly 
        # retention app 
        xr = self.conf['embedding']['start_time_discrete_list']
        cnt = 0
        for idx, (app, app_time, action_type, app_cate) in enumerate(seq[:-1]):
            if app_time == "unk":
                app_time = "2023-04-01 00:00:00"
            cur_time = datetime.strptime(app_time,'%Y-%m-%d %H:%M:%S')
            
            masks[cnt] = 1
            type_ids[cnt] = action_type
            if app_cate == "unk":
                cate_ids[cnt] = 0
            else:
                cate_ids[cnt] = int(app_cate) # NOTE category save as id

            #mask app to predict
            app_id = self.vocab_dic.get(app)
            if app_id == None:
                app_id = 1
            
            if random.random()<self.conf['mlm_prob']:
                mask_label_ids[cnt] = app_id  #预测标签

                if random.random() < 0.8:
                    app_ids[cnt] = unk_id  # NOTE 80%的 mask, 理论需要mask_id,因为序列中没有unk_id,用unk_id代替 
                elif random.random() <0.5:
                    app_ids[cnt] = app_id # 10%原始
                else:  
                    sample_item = random.sample(list(self.vocab_dic),1)[0] #10%其他
                    app_ids[cnt] = self.vocab_dic.get(sample_item)
            else:
                app_ids[cnt] = app_id

            # app_ids[cnt] = self.vocab_dic.get(app,unk_id)
            # days = np.searchsorted(xr,(cur_time.date() - prev_time.date()).days, side='right') # 0 for pad
            # days = np.searchsorted(xr,(cutoff_date.date() - cur_time.date()).days, side='right') # 0 for pad
            days = (cutoff_date.date() - cur_time.date()).days # 0 for pad
            days = max(days, 0)
            days = min(days, 92)
            app_time_ids[cnt] = 92 - days # 更合理方式：cur_time - start_date, 后面再改
            cnt +=1
        
        return (keep_app_ids, torch.tensor(app_ids), torch.tensor(type_ids), torch.tensor(app_time_ids), torch.tensor(cate_ids), torch.tensor(masks), torch.tensor(mask_label_ids), torch.tensor(labels),\
             torch.tensor(bc_app_ids), torch.tensor(bc_masks), torch.tensor(bc_label_dist), torch.tensor(age))


def get_appcnt_dist(app_ids, log_scale=True):
    """
    p(w|u) = \frac{log(1+w)}{\sum{log(1+w)}}
    """
   
    dict_cnt = Counter(app_ids)
    if log_scale:
        norm = sum(map(np.log1p, dict_cnt.values()))
        ret = {key: np.log1p(dict_cnt[key])/norm for key in dict_cnt}

    else:
        norm  = sum(dict_cnt.values())
        ret = {key: dict_cnt[key]/norm for key in dict_cnt}
    return ret



class AdAppMaskDatasetABC_finetune(Dataset):
    def __init__(self, df, vocab_dic, conf):
        super().__init__()
        self.conf = conf
        self.df = df
        self.vocab_dic = vocab_dic
      
    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):

        max_seq_len = self.conf['max_seq_len']
        vocab_size = self.conf['embedding']['app_vocab_size'] # -1 exclude pad if necessary
        # gender = self.df.iloc[i]['gender']
        # province = self.df.iloc[i]['province']

        # for finetune classifier
        aids = np.zeros(1, dtype=np.float32)
        labels = np.zeros(1, dtype=np.float32)  #默认为0

        # NOTE For downstream tasks which need apps, set use_app=True!!!!!
        use_app = False
        if use_app:
            aids[0] = self.vocab_dic.get(self.df.iloc[i]['app'])
        labels[0] = self.df.iloc[i]['label']

        # bc data
        bc_app_ids = np.zeros(max_seq_len, dtype=np.int32)
        bc_masks = np.zeros(max_seq_len, dtype=np.int32)
        
        # bc_app = self.df.iloc[i]['open_seq'].split(',')
        # bc_ids = [self.vocab_dic[app] for app in bc_app]
        bc_ids = self.df.iloc[i]['open_seq'].split(',')
        if bc_ids == ['unk']:
            bc_ids = [0]
        bc_ids = list(map(int, bc_ids))
        bc_index = len(bc_ids) // 2 # select [:half] apps to predict the dist of [half:] apps
        if bc_index<2:
            bc_ids = [0,0,0,0]
            bc_index = 2
        bc_label_dict = get_appcnt_dist(bc_ids[bc_index:])
        bc_app_index, bc_app_value = zip(*bc_label_dict.items())
        bc_label_dist = np.zeros(len(self.vocab_dic), dtype=np.float32)
        bc_label_dist.put(bc_app_index, bc_app_value)
        bc_ids = bc_ids[:bc_index]
        for bc_idx, bc_app_id in enumerate(bc_ids[::-1][:max_seq_len]): 
            bc_masks[max_seq_len-1-bc_idx] = 1
            bc_app_ids[max_seq_len-1-bc_idx] = bc_app_id
        
    
        # ae data
        max_seq_len = max_seq_len -1  # keep 1 position for app retention 
        app_ids = np.zeros(max_seq_len, dtype=np.int32)
        app_time_ids = np.zeros(max_seq_len, dtype=np.int32)
        type_ids = np.zeros(max_seq_len,dtype=np.int32)
        cate_ids = np.zeros(max_seq_len,dtype=np.int32)
        masks = np.zeros(max_seq_len)  # mask attention
        # labels = np.zeros(1, dtype=np.float32)  #默认为0
        mask_label_ids = np.zeros(max_seq_len,dtype=np.int32)-1 #默认为-1

        keep_app_ids = torch.zeros(vocab_size, dtype=torch.float)

        # labels[0] = self.df.iloc[i]['label']
        unk_id = self.vocab_dic['unk']

        keep_apps = self.df.iloc[i]['pack_keep_name'].split(',')
        app_setbits = torch.tensor([self.vocab_dic.get(app,unk_id) for app in keep_apps])
        keep_app_ids.index_fill_(dim=0,index=app_setbits,value=1.0)
        # assert not torch.equal(keep_app_ids, torch.zeros(keep_app_ids.shape, dtype=keep_app_ids.dtype)), keep_apps
        
        # if self.is_training:
        #      keep_app_ids = torch.mul(self.noise_func.sample(keep_app_ids.shape),keep_app_ids)

        # install and uninstall
        ins_app = self.df.iloc[i]['install_seq'].split(',')
        ins_time = self.df.iloc[i]['install_time'].split(',')
        ins_cate = self.df.iloc[i]['install_cate'].split(',')
        # ins_time = list(map(lambda x: time.strptime(x, '%Y-%m-%d %H:%M:%S'),ins_time))
        uns_app = self.df.iloc[i]['uninstall_seq'].split(',')
        uns_time = self.df.iloc[i]['uninstall_time'].split(',')
        uns_cate = self.df.iloc[i]['uninstall_cate'].split(',')
        # cutoff_date = datetime.strptime(self.df.iloc[i]['cutoff_date'],'%Y%m%d')
        cutoff_date = datetime.strptime("20230301",'%Y%m%d')

        ins_seq = list(zip(ins_app, ins_time, [2]*len(ins_app), ins_cate))  #  2 for install action-type
        uns_seq = list(zip(uns_app, uns_time, [3]*len(uns_app), uns_cate))  #  3 for uninstall action-type
        seq = ins_seq + uns_seq
        seq = sorted(seq, key=operator.itemgetter(1))[::-1][:max_seq_len] #降序排列

        seq.append(seq[-1]) # only for code friendly 
        # retention app 
        xr = self.conf['embedding']['start_time_discrete_list']
        cnt = 0
        for idx, (app, app_time, action_type, app_cate) in enumerate(seq[:-1]):
            if app_time == "unk":
                app_time = "2023-03-01 00:00:00"
            cur_time = datetime.strptime(app_time,'%Y-%m-%d %H:%M:%S')
            
            masks[cnt] = 1
            type_ids[cnt] = action_type
            if app_cate == "unk":
                cate_ids[cnt] = 0
            else:
                cate_ids[cnt] = int(app_cate) # NOTE category save as id

            #mask app to predict
            app_id = self.vocab_dic.get(app)
            if app_id == None:
                app_id = 1
            
            if random.random()<self.conf['mlm_prob']:
                mask_label_ids[cnt] = app_id  #预测标签

                if random.random() < 0.8:
                    app_ids[cnt] = unk_id  # NOTE 80%的 mask, 理论需要mask_id,因为序列中没有unk_id,用unk_id代替 
                elif random.random() <0.5:
                    app_ids[cnt] = app_id # 10%原始
                else:  
                    sample_item = random.sample(list(self.vocab_dic),1)[0] #10%其他
                    app_ids[cnt] = self.vocab_dic.get(sample_item)
            else:
                app_ids[cnt] = app_id

            # app_ids[cnt] = self.vocab_dic.get(app,unk_id)
            # days = np.searchsorted(xr,(cur_time.date() - prev_time.date()).days, side='right') # 0 for pad
            # days = np.searchsorted(xr,(cutoff_date.date() - cur_time.date()).days, side='right') # 0 for pad
            days = (cutoff_date.date() - cur_time.date()).days # 0 for pad
            days = max(days, 0)
            days = min(days, 92)
            app_time_ids[cnt] = 92 - days # 更合理方式：cur_time - start_date, 后面再改
            cnt +=1
        
        return (keep_app_ids, torch.tensor(app_ids), torch.tensor(type_ids), torch.tensor(app_time_ids), torch.tensor(cate_ids), torch.tensor(masks), torch.tensor(mask_label_ids), \
             torch.tensor(bc_app_ids), torch.tensor(bc_masks), torch.tensor(bc_label_dist), torch.tensor(aids), torch.tensor(labels))




if __name__ =='__main__':
    print('----------------------------------')
    

