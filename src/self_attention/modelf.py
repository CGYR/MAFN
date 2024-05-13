import copy
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

import torch
import torch.nn as nn 
from torch.nn import CrossEntropyLoss, MSELoss, BCELoss, BCEWithLogitsLoss
from torch.utils.data import DataLoader,Dataset,TensorDataset
from torch.utils.data import RandomSampler,SequentialSampler,DistributedSampler

from .dataset import HeterAppDataset,HeterAppMaskDataset

from .core import CustomBertModel
from .config import PretrainedConfig
from .mlp import DNN, PredictionLayer

sys.path.append("..")
from base.transformer import get_activation, TransformerEncoder, TransformerLayer,Time2VecEmbedding
from base.merge_input import DropLayerNorm, MergePositionEmbedding


class AutoEncoderLayer(nn.Module):
    # for modeling app retention
    def __init__(self, conf):
        super(AutoEncoderLayer,self).__init__()

        hidden_sizes = conf['hidden_sizes']
        # vocab_size = conf['app_vocab_size']
        # self.pos_weight = conf['loss_pos_weight']

        # self.embedding = nn.Parameter(torch.Tensor(np.random.normal(0.0,0.2,size=(hidden_sizes[0],vocab_size))))
        # self.bias = nn.Parameter(torch.Tensor([0.0]*hidden_sizes[0]))

        # hidden_sizes.insert(0,vocab_size)
        hidden_sizes = hidden_sizes + hidden_sizes[:-1][::-1] # symmetric [256,64,256]
    
        mlists = nn.ModuleList()
        for idx, (d_in, d_out) in enumerate(zip(hidden_sizes[:-1],hidden_sizes[1:])):
            mlists.append(nn.Linear(d_in,d_out))
            mlists.append(nn.LeakyReLU(negative_slope=0.1))
            # if idx == 0:
            #     mlists.append(nn.Linear(d_in,d_out))
            #     mlists.append(nn.LeakyReLU(negative_slope=0.1))
                
            # elif idx !=len(hidden_sizes[:-1])-1:
            #     mlists.append(nn.Linear(d_in,d_out))
            #     mlists.append(nn.BatchNorm1d(d_out,affine=True))
            #     mlists.append(nn.LeakyReLU(negative_slope=0.1))
            # else:
            #     out_layer = nn.Linear(d_in, d_out,bias=False)
            #     mlists.append(out_layer)

        self.feedforward = mlists
        self.feedforward.apply(self._init_weights)
    
        
        # self.feedfoward = nn.Sequential()
        # for idx, (d_in,d_out) in enumerate(zip(hidden_sizes[:-1],hidden_sizes[1:])): #[vocab_size,256,64,256,vocab_size]
        #     if idx != len(hidden_sizes[:-1])-1:
        #         self.feedfoward.add_module('linear'+str(idx), nn.Linear(d_in,d_out))
        #         self.feedfoward.add_module('bias'+str(idx), nn.LeakyReLU(negative_slope=0.1))
        #     else:
        #         self.feedfoward.add_module('linear'+str(idx), nn.Linear(d_in,d_out,bias=False))

    def _init_weights(self,module):
        if isinstance(module,(nn.Linear)):
            # module.weight.data.normal_(0,self.cfg['initializer_range'])  # nn.init.normal_(module.weight.data,0,0.02),module.weight.data.fill_(0)/zero_()
            nn.init.kaiming_normal_(module.weight.data, a=0.1)

    
    def forward(self, inputs):
        # norm = torch.norm(inputs, dim=[1],keepdim=True) #l2
        # inputs = inputs/norm
        # extract_emb = 0
        for idx, layer  in enumerate(self.feedforward):
            inputs = layer(inputs)
            # if idx==0:
            #     apps_emb = inputs 
            # if idx==1:
            #     extract_emb = inputs
        # inputs = F.linear(inputs, self.embedding.T)
        # pos_weight = torch.where(labels==1, torch.tensor(self.pos_weight).float().to(labels.device), torch.tensor(1.0).to(labels.device))
        # loss = F.binary_cross_entropy_with_logits(inputs,labels, pos_weight = pos_weight)
        return inputs


class ABCN(nn.Module):
    # named as MAFN in our paper
    def __init__(self, pretrained_config, autoencoder_config, common_config,  
                       dae_noise=0.1):

        super().__init__()
        assert isinstance(pretrained_config, PretrainedConfig)

        self.pretrained_config = pretrained_config
        self.autoencoder_config = autoencoder_config
        self.common_config = common_config
        
        self.encoder = CustomBertModel(self.pretrained_config)
        self.decoder = CustomBertModel(self.pretrained_config)
        self.fusion_decoder = CustomBertModel(self.pretrained_config)

        self.autoencoder = AutoEncoderLayer(self.autoencoder_config)
        self.noise_func = torch.distributions.Binomial(total_count=1, probs=1-dae_noise)

        self.app_embedding = nn.Embedding(num_embeddings=self.common_config['embedding']['app_vocab_size'],
                                          embedding_dim=self.common_config['embedding']['app_emb_dim'],
                                          padding_idx=0)
        self.time_embedding = nn.Embedding(self.common_config['embedding']['start_time_vocab_size'],
                                           self.common_config['embedding']['start_time_emb_dim'],
                                           padding_idx=0)

        self.type_embedding = nn.Embedding(self.common_config['embedding']['type_vocab_size']+1, # +1 for pad
                                           self.common_config['embedding']['type_vocab_emb_dim'], 
                                            padding_idx=0)
        
        self.cate_embedding = nn.Embedding(self.common_config['embedding']['cate_vocab_size'],
                                           self.common_config['embedding']['cate_emb_dim'], 
                                            padding_idx=0)

        self.codebook = nn.Embedding(num_embeddings=self.common_config['embedding']['codebook_size'],
                                          embedding_dim=self.common_config['embedding']['codeword_dim'],
                                            padding_idx=0)

        self.vocab_ouput_layer = nn.Linear(self.common_config['embedding']['app_emb_dim'], 
                                           self.common_config['embedding']['app_vocab_size'])
        
        self.encoder_mask_output_layer = nn.Linear(self.common_config['embedding']['app_emb_dim'], 
                                                   self.common_config['embedding']['app_vocab_size'])
        
        self.app_embedding.weight.data.normal_(std=0.05)
        self.time_embedding.weight.data.normal_(std=0.05)
        self.type_embedding.weight.data.normal_(std=0.05)
        self.cate_embedding.weight.data.normal_(std=0.05)
        self.codebook.weight.data.normal_(std=0.05)

        self.with_cate = self.common_config['train']['with_cate']
 
        app_vocab_size = self.autoencoder_config['app_vocab_size']
        dae_layer0_size = self.autoencoder_config['hidden_sizes'][0]
        
        assert dae_layer0_size == self.common_config['embedding']['app_emb_dim'],'{} !={}'.format(dae_layer0_size,
                                                                self.common_config['embedding']['app_emb_dim'])

        # tie weight  (seq predict layer)                    
        self.vocab_ouput_layer.weight = self.app_embedding.weight
        self.vocab_ouput_layer.bias.data.fill_(0)
        # tie weight (mask app predict layer)
        self.encoder_mask_output_layer.weight  = self.app_embedding.weight
        self.encoder_mask_output_layer.bias.data.fill_(0)

        # tie weight between app_embedding and dae input layer 
        self.dae_input_layer_weight = self.app_embedding.weight #(vocab_size, emb_dim)
        self.dae_input_layer_bias =  nn.Parameter(torch.randn(dae_layer0_size))
        self.dae_input_layer_bias.data.fill_(0)

        # Bert-based encoder
        hidden_size = self.common_config['embedding']['app_emb_dim']
        max_seq_len = self.common_config['max_seq_len']
        layer_norm_eps = self.common_config['trans_encoder']['layer_norm_eps']
        hidden_dropout_prob = self.common_config['trans_encoder']['hidden_dropout_prob']
        max_position_emb = max_seq_len

        self.lm_head = nn.Linear(hidden_size, self.common_config['embedding']['app_vocab_size'], bias=False)
        self.lm_head.weight = self.app_embedding.weight
        self.merge_embedding = MergePositionEmbedding(hidden_size=hidden_size,
                                              max_position_embeddings=max_position_emb, 
                                              initializer_range=self.common_config['trans_encoder']['initializer_range'],
                                              full_pass=False) # add position_embedding in [max_seq_len-1,....,0] order
        self.dropNorm = DropLayerNorm(hidden_size=hidden_size,
                                      hidden_dropout_prob=hidden_dropout_prob,
                                      layer_norm_eps=layer_norm_eps,
                                      initializer_range=self.common_config['trans_encoder']['initializer_range'])
        self.bc_encoder = TransformerEncoder(self.common_config['trans_encoder'])

        # for finetune, MLP
        self.is_finetune = self.common_config['finetune']['flag']
        # self.user_embedding = nn.Embedding(self.common_config['embedding']['user_vocab_size'],
        #                                    self.common_config['embedding']['user_emb_dim'],
        #                                    padding_idx=0)
        # self.user_embedding.weight.data.normal_(std=0.05)
        dnn_hidden_units = self.common_config['finetune']['mlp_hidden_units']
        dnn_input_dim = self.common_config['embedding']['app_emb_dim'] + self.common_config['embedding']['app_emb_dim'] # app emb + fusion emb
        self.dnn = DNN(dnn_input_dim, dnn_hidden_units)
        self.dnn_linear = nn.Linear(dnn_hidden_units[-1], 1, bias=False)
        self.out = PredictionLayer()

        # for contrast-learning
        self.cons_neg_num = 31 # in-batch neg-sample for each positive sample
        self.cons_sample_num = 5 # for sem_loss
        self.long_encoder = nn.Linear(max_seq_len * hidden_size, hidden_size)
        self.short_encoder = nn.Linear(max_seq_len * hidden_size, hidden_size)

        # for VQ-based adaptive fusion
        # self.fusion_linear = nn.Linear(hidden_size * 2, hidden_size) # for adaptive weight
        self.long_projection = nn.Linear(self.common_config['embedding']['app_emb_dim'], self.common_config['embedding']['codeword_dim'])
        self.short_projection = nn.Linear(self.common_config['embedding']['app_emb_dim'], self.common_config['embedding']['codeword_dim'])
               

    def forward(self, keep_ids, app_ids, type_ids, app_time_ids, app_cate_ids, masks, mask_label_ids, bc_app_ids, bc_masks, bc_label_dist, age, labels=[], predict=False):
        """
        keep_ids, (N,vocab_size)
        app_ids,(N, seq)
        type_ids,(N,seq) # install/unstall/retention/mask
        app_time_ids,(N,seq)
        app_cate_ids,(N,seq)
        masks,(N,seq)
        """
        if self.is_finetune:
            return self.finetune(keep_ids, app_ids, type_ids, app_time_ids, app_cate_ids, masks, mask_label_ids, bc_app_ids, bc_masks, bc_label_dist, age, labels)

        #1. autoencoder for app retention
        dae_input = keep_ids
        dae_target = keep_ids

        dae_input_norm = torch.norm(dae_input, dim=[1],keepdim=True) #l2
        dae_input = dae_input/dae_input_norm

        dae_input = F.dropout(dae_input,p=0.1,training=(not predict)) #add noise 
        # dae_input = torch.mul(self.noise_func.sample(dae_input.shape), dae_input) 

        dae_emb = torch.matmul(dae_input,self.dae_input_layer_weight) + self.dae_input_layer_bias #(batch, layer0_size)
        dae_output = self.autoencoder(dae_emb)  # (batch, last_layer_size)
        dae_output = torch.matmul(dae_output, self.dae_input_layer_weight.transpose(0,1)) #(batch, app_vocab_size)

        #2. Trm-based encoder for app un/install
        app_embs = self.app_embedding(app_ids.long()) # (batch,seq, emb_dim)
        # print(self.time_embedding)
        time_embs = self.time_embedding(app_time_ids.long()) #(batch, seq, emb_dim)
        type_embs = self.type_embedding(type_ids.long())
        cate_embs = self.cate_embedding(app_cate_ids.long())
        if self.with_cate:
            input_embs = app_embs + time_embs + type_embs + cate_embs #(batch,seq, emb_dim)
        else:
            input_embs = app_embs + time_embs + type_embs #(batch,seq, emb_dim)
        
        # for sem loss
        # divide install & unstall behaviors
        ins_mask = type_ids.byte() # bool, (B, L)
        uns_mask = ~ins_mask
        ins_embs = self.del_all_zero(input_embs*ins_mask) # (B, S, E)
        uns_embs = self.del_all_zero(input_embs*uns_mask) # (B, S, E)

        # concatenate rententions
        batch_size = keep_ids.shape[0]
        keep_type_embs = self.type_embedding(torch.tensor([1],dtype=torch.long,device=dae_emb.device).expand(batch_size,1)) #(batch,1,emb_dim), type=1 for retention 
        dae_emb = dae_emb.unsqueeze(1) + keep_type_embs 

        fill_masks = torch.tensor([1],dtype=masks.dtype,device=masks.device).expand(batch_size,1)  # expand return views 
        fill_masks = torch.cat([fill_masks, masks],dim=1) #(B, seq+1)

        input_embs = torch.cat([dae_emb, input_embs],dim=1) #(batch,seq+1, emb_dim)

        encoder_outputs = self.encoder(attention_mask=fill_masks.float(), inputs_embeds=input_embs)
        encoder_seq_outputs, pooled_output = encoder_outputs[0], encoder_outputs[1]  #(batch,seq+1,emb_dim ), (batch, emb_dim)

        # for sem loss
        ins_output = self.encoder(attention_mask=fill_masks.float(), inputs_embeds=ins_embs)[1]
        uns_output = self.encoder(attention_mask=fill_masks.float(), inputs_embeds=uns_embs)[1]

        # 3. Bert-based encoder for app launch
        bc_app_embs = self.app_embedding(bc_app_ids.long())
        bc_input_embs = self.merge_embedding(embs=bc_app_embs) # add position embedding
        bc_input_embs = self.dropNorm(bc_input_embs)

        bc_encoder_outputs = self.bc_encoder(bc_input_embs, attention_mask = bc_masks.float())[0]
        bc_emb_mean = (bc_encoder_outputs * bc_masks.unsqueeze(-1)).sum(1) / bc_masks.sum(1).unsqueeze(-1)
        
        # fusion methods...
        # concatente & FFN & adaptive weight adding
        fusion_emb = pooled_output + bc_emb_mean
        output_embs = torch.cat([pooled_output, bc_emb_mean], dim=1)
        alpha = torch.sigmoid(self.fusion_linear(output_embs))
        fusion_emb = alpha * bc_emb_mean + (1-alpha) * pooled_output

        # 4. VQ-based fusion
        # codebook = self.codebook(torch.Tensor([i for i in range(self.common_config['embedding']['codebook_size'])]).long())
        # codebook, assign, _, _  = kmeans(torch.cat([pooled_output, bc_emb_mean], dim=1), codebook)
        # long_evs = codebook[assign(pooled_output)]
        # long_pro = self.long_projection(pooled_output)
        # with torch.no_grad():
        #     long_d = long_evs - long_pro
        # long_embs = long_pro + long_d
        # short_evs = codebook[assign(bc_emb_mean)]
        # short_pro = self.short_projection(bc_emb_mean)
        # with torch.no_grad():
        #     short_d = short_evs - short_pro
        # short_embs = short_pro + short_d
        # fusion_emb =  long_embs + short_embs

        if not predict:
            # transformer pesudo decoder 
            decoder_input_embs = time_embs + type_embs
            # decoder_fuse_input_embs = torch.cat([fusion_emb.unsqueeze(1), time_embs + type_embs],dim=1) 
            decoder_input_embs = torch.cat([pooled_output.unsqueeze(1),decoder_input_embs],dim=1)

            decoder_hidden_states = self.decoder(attention_mask=fill_masks.float(), inputs_embeds=decoder_input_embs)[0] #(batch, seq+1,emb_dim)
            decoder_hidden_states = decoder_hidden_states[:,1:,:] #(batch,seq, emb_dim)
            # fusion decoder
            # decoder_fuse_hidden_states = self.fusion_decoder(attention_mask=fill_masks.float(), inputs_embeds=decoder_fuse_input_embs)[0] #(batch, seq+1,emb_dim)
            # decoder_fuse_hidden_states = decoder_fuse_hidden_states[:,1:,:]

            # pre-train loss 
            # loss #0: autoencoder loss 
            pos_weight = torch.where(dae_target==1, torch.tensor(self.autoencoder_config['loss_pos_weight']).float().to(dae_target.device), 
                                     torch.tensor(1.0).to(dae_target.device))
            dae_loss = F.binary_cross_entropy_with_logits(dae_output,dae_target, pos_weight=pos_weight)
    
            # loss #2: app reconstruction loss
            decoder_hidden_states = decoder_hidden_states[masks.ne(0)] #(batch*mask)*hidden_size
            decoder_labels = app_ids[masks.ne(0)]

            tmp = mask_label_ids[masks.ne(0)]
            decoder_labels = torch.where(tmp.ne(-1),tmp,decoder_labels)  # replace mask-predict app with orignal app in decoder 

            decoder_logits = self.vocab_ouput_layer(decoder_hidden_states) #(batch*mask) * vocab_size
            decoder_loss  = F.cross_entropy(decoder_logits, decoder_labels.long())
            # alternate reconstruct task for fuse emb
            # decoder_fuse_hidden_states = decoder_fuse_hidden_states[masks.ne(0)]
            # decoder_fuse_logits = self.vocab_ouput_layer(decoder_fuse_hidden_states)
            # decoder_fuse_loss = F.cross_entropy(decoder_fuse_logits, decoder_labels.long())

            # bottleneck_loss 
            bottleneck_output = self.vocab_ouput_layer(pooled_output) # batch*vocab_size 
            bottlneck_loss = F.binary_cross_entropy_with_logits(bottleneck_output,dae_target, pos_weight=pos_weight) 

            # loss alternate: mask app predict loss
            # encoder_seq_outputs = encoder_seq_outputs[:,1:,:]
            # encoder_seq_outputs = encoder_seq_outputs[mask_label_ids.ne(-1)] # (batch*mask_num) * hidden
            # mask_label_ids = mask_label_ids[mask_label_ids.ne(-1)].long() # (batch * mask_num)
            # mask_logits = self.encoder_mask_output_layer(encoder_seq_outputs) #(batch*mask)*vocab_size
            # mlm_loss = F.cross_entropy(mask_logits,mask_label_ids)

            # loss #3: semantic contrastive learning
            cl_loss_sem = self.cl_loss_2(ins_output, uns_output)

            # loss #1: app distribution prediction
            bc_logits = self.lm_head(bc_emb_mean)
            bc_logq = F.log_softmax(bc_logits, dim=1)
            bc_loss = -torch.mean(torch.sum(torch.mul(bc_logq, bc_label_dist), dim=1))
            # bc_emb_mean = torch.tensor(0)

            # loss #4: user contrastive learning: install & unstall seq vs open seq
            cl_loss = self.cl_loss(input_embs, bc_input_embs)
            # cl_loss += self.cl_loss_2(pooled_output, bc_emb_mean)

            # fusion embs predict user attr
            # age_out = self.age_encoder(fusion_emb) # (batch, 6)
            # pre_loss = nn.CrossEntropyLoss()
            # age_loss = pre_loss(age_out, age)

        # total_loss = dae_loss + decoder_loss + bottlneck_loss 
        else:
            dae_loss = decoder_loss = decoder_fuse_loss = bottlneck_loss = mlm_loss = bc_loss = cl_loss = cl_loss_sem = None

        return pooled_output, bc_emb_mean, fusion_emb, dae_loss, decoder_loss, decoder_fuse_loss, bottlneck_loss, mlm_loss, bc_loss, cl_loss, cl_loss_sem


    def cl_loss(self, long_seq_emb, short_seq_emb):
        # contrastive loss for install & unstall seq vs open seq
        # long_seq_emb : [B, L, E]
        # short_seq_emb: [B, L, E]
        long_input = torch.reshape(long_seq_emb, (-1, long_seq_emb.shape[1] * long_seq_emb.shape[2])) # [B, L*E]
        long_input = self.long_encoder(long_input) # [B, L*E] -> [B, E]
        short_input = torch.reshape(short_seq_emb, (-1, short_seq_emb.shape[1] * short_seq_emb.shape[2])) # [B, L*E]
        short_input = self.short_encoder(short_input) # [B, L*E] -> [B, E]

        long_input = torch.unsqueeze(long_input, 1) # [B, 1, E]
        short_inputs = self.gen_con_neg(short_input) # [B, E] -> [B, (1 + neg_num) * E]
        short_inputs = torch.reshape(short_inputs, (-1, 1 + self.cons_neg_num, long_input.shape[2])) # [B, (1 + neg_num) * E] -> [B, 1 + neg_num, E]
        dot_product = torch.mean(long_input * short_inputs, -1) # [B, 1+neg_sample, E] dot [B, 1, E] -> [B, 1+neg_sample, E]
        log_l1 = torch.nn.functional.log_softmax(dot_product, dim=1)[:,0]
        con_loss = -torch.mean(log_l1)
        return con_loss
    

    def cl_loss_2(self, in_inputs, un_inputs):
        # contrastive loss for install, unstall, unseen seqs (represented by other in-batch users' installed seq)
        # in_input : [B, L, E]
        # un_inputs: [B, L, E]
        in_input = in_inputs[0]
        no_input = in_inputs[torch.randperm(in_input.size(0))][0]
        un_inputs = self.sample_con(un_inputs) # [B, L, E] -> [B, neg_sample, E]
        dot_product_pos = in_input * no_input # [B, E]
        dot_product_pos = torch.unsqueeze(dot_product_pos, 1) # [B, 1, E]
        dot_product_neg = torch.mean(in_input * un_inputs, -1) # [B, neg_sample, E] dot [B, 1, E] -> [B, neg_sample, E]
        dot_product = torch.cat((dot_product_pos, dot_product_neg), 1) # [B, 1 + neg_sample, E]
        log_l1 = torch.nn.functional.log_softmax(dot_product, dim=1)[:,0]
        con_loss = -torch.mean(log_l1)
        return con_loss


    def gen_con_neg(self, data):
        # [B, E] -> [B, (1 + neg_sample) * E], neg sample method: shuffle data to let item & hist mis-match
        new_data = data.clone()
        for i in range(self.cons_neg_num):
            random_list_1 = torch.randperm(data.size(0)) # random twice and ensure random_list not in the same order
            random_list_2 = torch.randperm(data.size(0))
            random_list = torch.where(random_list_1 - torch.arange(data.size(0)) == 0, random_list_2,
                                      random_list_1)
            random_data = data[random_list]
            new_data = torch.cat((new_data, random_data), 1)
        return new_data
    
    def sample_con(self, data):
        # [B, L, E] -> [B, S, E]
        random_list = torch.randperm(data.size(1))
        random_data = data[:,random_list[:self.cons_sample_num],:]
        return random_data
    
    def del_all_zero(self, data):
        # [B, L, E] -> [B, S, E]
        idx = torch.where(torch.all(data[..., :] == 0, axis=0))[0]
        all = torch.arange(data.shape[1])
        for i in range(len(idx)):
            all = all[torch.arange(all.size(0)) != idx[i]-i]
        data = torch.index_select(data, 1, all)
        return data

    def finetune(self, keep_ids, app_ids, type_ids, app_time_ids, app_cate_ids, masks, mask_label_ids, bc_app_ids, bc_masks, bc_label_dist, aids, labels, predict=False):
        """
        keep_ids, (N,vocab_size)
        app_ids,(N, seq)
        type_ids,(N,seq) # install/unstall/retention/mask
        app_time_ids,(N,seq)
        app_cate_ids,(N,seq)
        masks,(N,seq)
        """

        #1、autoencoder
        dae_input = keep_ids
        dae_target = keep_ids

        dae_input_norm = torch.norm(dae_input, dim=[1],keepdim=True) #l2
        dae_input = dae_input/dae_input_norm

        dae_input = F.dropout(dae_input,p=0.1,training=(not predict)) #add noise 
        # dae_input = torch.mul(self.noise_func.sample(dae_input.shape), dae_input) 

        dae_emb = torch.matmul(dae_input,self.dae_input_layer_weight) + self.dae_input_layer_bias #(batch, layer0_size)
        dae_output = self.autoencoder(dae_emb)  # (batch, last_layer_size)
        dae_output = torch.matmul(dae_output, self.dae_input_layer_weight.transpose(0,1)) #(batch, app_vocab_size)

        #2、transformer
        app_embs = self.app_embedding(app_ids.long()) # (batch,seq, emb_dim)
        # print(self.time_embedding)
        time_embs = self.time_embedding(app_time_ids.long()) #(batch, seq, emb_dim)
        type_embs = self.type_embedding(type_ids.long())
        cate_embs = self.cate_embedding(app_cate_ids.long())
        if self.with_cate:
            input_embs = app_embs + time_embs + type_embs + cate_embs #(batch,seq, emb_dim)
        else:
            input_embs = app_embs + time_embs + type_embs #(batch,seq, emb_dim)

        # transformer encoder
        # concatenate rententions
        batch_size = keep_ids.shape[0]
        keep_type_embs = self.type_embedding(torch.tensor([1],dtype=torch.long,device=dae_emb.device).expand(batch_size,1)) #(batch,1,emb_dim), type=1 for retention 
        dae_emb = dae_emb.unsqueeze(1) + keep_type_embs 

        fill_masks = torch.tensor([1],dtype=masks.dtype,device=masks.device).expand(batch_size,1)  # expand return views 
        fill_masks = torch.cat([fill_masks, masks],dim=1) #(B, seq+1)

        input_embs = torch.cat([dae_emb, input_embs],dim=1) #(batch,seq+1, emb_dim)

        encoder_outputs = self.encoder(attention_mask=fill_masks.float(), inputs_embeds=input_embs)
        encoder_seq_outputs, pooled_output = encoder_outputs[0], encoder_outputs[1]  #(batch,seq+1,emb_dim ), (batch, emb_dim)

        # bc module
        bc_app_embs = self.app_embedding(bc_app_ids.long())
        bc_input_embs = self.merge_embedding(embs=bc_app_embs) # add position embedding
        bc_input_embs = self.dropNorm(bc_input_embs)

        bc_encoder_outputs = self.bc_encoder(bc_input_embs, attention_mask = bc_masks.float())[0]
        bc_emb_mean = (bc_encoder_outputs * bc_masks.unsqueeze(-1)).sum(1) / bc_masks.sum(1).unsqueeze(-1)
        
        # fusion methods...
        # fusion_emb = pooled_output + bc_emb_mean
        output_embs = torch.cat([pooled_output, bc_emb_mean], dim=1)
        alpha = torch.sigmoid(self.fusion_linear(output_embs))
        fusion_emb = alpha * bc_emb_mean + (1-alpha) * pooled_output
        
        # if finetune, add classifier loss
        if self.is_finetune:
            aid_embs = self.app_embedding(aids.long()) # (batch, 1, emb_dim)
            a_embs = torch.squeeze(aid_embs) # (batch, emb_dim)
            # u_embs = torch.ones(a_embs.shape[0] * 4)
            # u_embs = torch.reshape(u_embs, (-1, 4)) # NOTE useless, just tmp padding
            # dnn_input = torch.cat([u_embs.to(a_embs.device), a_embs, fusion_emb], dim=-1) # (batch, emb_dim + emb_dim)

            dnn_input = torch.cat([a_embs, fusion_emb], dim=-1) # (batch, emb_dim + emb_dim)
            dnn_output = self.dnn(dnn_input)
            dnn_logit = self.dnn_linear(dnn_output)
            y_pred = self.out(dnn_logit)
            down_loss = F.binary_cross_entropy(y_pred.squeeze(), labels.squeeze(), reduction='sum')
        else:
            y_pred = None
            down_loss = None

        if not predict:
            # transformer pesudo decoder 
            decoder_input_embs = time_embs + type_embs
            decoder_fuse_input_embs = torch.cat([fusion_emb.unsqueeze(1), time_embs + type_embs],dim=1) 
            decoder_input_embs = torch.cat([pooled_output.unsqueeze(1),decoder_input_embs],dim=1)

            decoder_hidden_states = self.decoder(attention_mask=fill_masks.float(), inputs_embeds=decoder_input_embs)[0] #(batch, seq+1,emb_dim)
            decoder_hidden_states = decoder_hidden_states[:,1:,:] #(batch,seq, emb_dim)
            # fusion decoder
            decoder_fuse_hidden_states = self.fusion_decoder(attention_mask=fill_masks.float(), inputs_embeds=decoder_fuse_input_embs)[0] #(batch, seq+1,emb_dim)
            decoder_fuse_hidden_states = decoder_fuse_hidden_states[:,1:,:]

            #3、loss 
            # autoencoder_loss 
            pos_weight = torch.where(dae_target==1, torch.tensor(self.autoencoder_config['loss_pos_weight']).float().to(dae_target.device), 
                                     torch.tensor(1.0).to(dae_target.device))
            dae_loss = F.binary_cross_entropy_with_logits(dae_output,dae_target, pos_weight=pos_weight)
    
            # decoder loss
            decoder_hidden_states = decoder_hidden_states[masks.ne(0)] #(batch*mask)*hidden_size
            decoder_labels = app_ids[masks.ne(0)]

            tmp = mask_label_ids[masks.ne(0)]
            decoder_labels = torch.where(tmp.ne(-1),tmp,decoder_labels)  # replace mask-predict app with orignal app in decoder 

            decoder_logits = self.vocab_ouput_layer(decoder_hidden_states) #(batch*mask) * vocab_size
            decoder_loss  = F.cross_entropy(decoder_logits, decoder_labels.long())
            # reconstruct task for fuse emb
            decoder_fuse_hidden_states = decoder_fuse_hidden_states[masks.ne(0)]
            decoder_fuse_logits = self.vocab_ouput_layer(decoder_fuse_hidden_states)
            decoder_fuse_loss = F.cross_entropy(decoder_fuse_logits, decoder_labels.long())

            # bottleneck_loss 
            bottleneck_output = self.vocab_ouput_layer(pooled_output) # batch*vocab_size
            bottlneck_loss = F.binary_cross_entropy_with_logits(bottleneck_output,dae_target, pos_weight=pos_weight) 

            # mask app predict loss
            encoder_seq_outputs = encoder_seq_outputs[:,1:,:]
            encoder_seq_outputs = encoder_seq_outputs[mask_label_ids.ne(-1)] # (batch*mask_num) * hidden
            mask_label_ids = mask_label_ids[mask_label_ids.ne(-1)].long() # (batch * mask_num)
            mask_logits = self.encoder_mask_output_layer(encoder_seq_outputs) #(batch*mask)*vocab_size
            # mlm_loss  = self.loss_fct(mask_logits, labels)  # already divided by batch-size
            mlm_loss = F.cross_entropy(mask_logits,mask_label_ids)

            # bc module loss
            bc_logits = self.lm_head(bc_emb_mean)
            bc_logq = F.log_softmax(bc_logits, dim=1)
            bc_loss = -torch.mean(torch.sum(torch.mul(bc_logq, bc_label_dist), dim=1))
            # bc_emb_mean = torch.tensor(0)

            # contrast: install & unstall seq vs open seq
            cl_loss = self.cl_loss(input_embs, bc_input_embs)
            # cl_loss += self.cl_loss_2(pooled_output, bc_emb_mean)

            # # fusion embs predict downstream tasks
            # age_out = self.age_encoder(fusion_emb) # (batch, 6)
            # pre_loss = nn.CrossEntropyLoss()
            # age_loss = pre_loss(age_out, age)

        # total_loss = dae_loss + decoder_loss + bottlneck_loss 
        else:
            dae_loss = decoder_loss = decoder_fuse_loss = bottlneck_loss = mlm_loss = bc_loss = cl_loss = None

        return pooled_output, bc_emb_mean, fusion_emb, dae_loss, decoder_loss, decoder_fuse_loss, bottlneck_loss, mlm_loss, bc_loss, cl_loss, y_pred, down_loss


def kmeans(X, K_or_center, max_iter=300, verbose=False):
    # sited: https://github.com/ustcml/RecStudio/blob/main/recstudio/ann/sampler.py
    N = X.size(0)
    if isinstance(K_or_center, int):
        K = K_or_center
        C = X[torch.randperm(N)[:K]]
    else:
        K = K_or_center.size(0)
        C = K_or_center
    prev_loss = np.inf
    for iter in range(max_iter):
        dist = torch.sum(X * X, dim=-1, keepdim=True) - 2 * \
            (X @ C.T) + torch.sum(C * C, dim=-1).unsqueeze(0)
        assign = dist.argmin(-1)
        assign_m = X.new_zeros(N, K)
        assign_m[(range(N), assign)] = 1
        loss = torch.sum(torch.square(X - C[assign, :])).item()
        if verbose:
            print(f'step:{iter:<3d}, loss:{loss:.3f}')
        if (prev_loss - loss) < prev_loss * 1e-6:
            break
        prev_loss = loss
        cluster_count = assign_m.sum(0)
        C = (assign_m.T @ X) / cluster_count.unsqueeze(-1)
        empty_idx = cluster_count < .5
        ndead = empty_idx.sum().item()
        C[empty_idx] = X[torch.randperm(N)[:ndead]]
    return C, assign, assign_m, loss



