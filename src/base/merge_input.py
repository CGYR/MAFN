# encoding: utf-8

import math
import numpy as np 
import torch
import torch.nn as nn 
import torch.nn.functional as F


from .transformer import  TimeMaskEmbedding, Time2VecEmbedding

class InitParamBase(nn.Module):

    def __init__(self, initializer_range=0.02):
        super().__init__()
        self.initializer_range = initializer_range
    
    def init_weight(self,module):
        if isinstance(module, torch.nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

        elif isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)

        if isinstance(module, torch.nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


class DropLayerNorm(InitParamBase):
    """
    droupout (layernorm(x))
    """
    def __init__(self, hidden_size, hidden_dropout_prob=0.1, 
                                    layer_norm_eps=1.0e-12,
                                    initializer_range=0.02):
        super(DropLayerNorm,self).__init__(initializer_range)

        self.LayerNorm = torch.nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = torch.nn.Dropout(hidden_dropout_prob)

        self.apply(self.init_weight)
    

    def forward(self, x):
        """
        x: (B,L,H)
        """
        x = self.LayerNorm(x)
        x = self.dropout(x)
        return x


class MergePositionEmbedding(InitParamBase):
    """Construct the embeddings from word, position and token_type embeddings.
    """
    def __init__(self, hidden_size=32,
                       max_position_embeddings=512,
                       padding_idx=None,
                       initializer_range=0.02,
                       full_pass=False,
                       auto_fill=True):

        super(MergePositionEmbedding,self).__init__(initializer_range)
        self.full_pass = full_pass
        self.auto_fill = auto_fill
        # self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=pad_token_id)
        if not self.full_pass:
            self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size, padding_idx=padding_idx)
            # self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

            # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
            # any TensorFlow checkpoint file
            # self.LayerNorm = torch.nn.LayerNorm(hidden_size, eps=layer_norm_eps)
            # self.dropout = nn.Dropout(hidden_dropout_prob)


            self.apply(self.init_weight)
       

    def forward(self, embs, merge_inputs=None):
        """
        auto_fill, automatically generate position index if merge_inputs is None
        
        embs: (B,L,D)
        merge_inputs: (B,L)
        return: (B,L,D)
        """

        input_shape = embs.size()[:-1]
        seq_length = embs.shape[1] # (bsz seq_len)
        device = embs.device

        if not self.full_pass: # identity
            if merge_inputs is not None:
                position_embeddings = self.position_embeddings(merge_inputs.long())

            elif self.auto_fill:
                merge_inputs = torch.arange(start=seq_length-1,end=-1, step=-1, dtype=torch.long, device=device) # 位置0是确定的
                merge_inputs = merge_inputs.unsqueeze(0).expand(input_shape) #(bsz, seq_len)
                position_embeddings = self.position_embeddings(merge_inputs)
            else:
                position_embeddings = torch.tensor(0,dtype=embs.dtype, device=embs.device)

            embs = embs + position_embeddings

            # inputs_embeds = self.LayerNorm(input_embs)
            # inputs_embeds = self.dropout(input_embs)
            
        return embs

class MergeTimeStampEmbedding(InitParamBase):
    """
    construct Embdding from event_id and event_time
    """

    def __init__(self,  hidden_size,include_non_periodic=False, padding_value=-1, 
                        initializer_range=0.02,
                        full_pass=False):

        super(MergeTimeStampEmbedding,self).__init__(initializer_range)

        self.full_pass = full_pass

        if not self.full_pass:
            self.time2vec=Time2VecEmbedding(hidden_size,
                                            include_non_periodic=include_non_periodic,
                                            padding_value=padding_value)
        

            # self.apply(self.init_weight)
            self.time2vec.linear.weight.data.normal_(1.0, std=1.0)  # T= 2pi/wi， input range(0,2*pi)
            self.time2vec.linear.bias.data.zero_()
        
    def forward(self, embs, merge_inputs=None):
        """
        embs: (B,L,D)
        merge_inputs:(B, D)

        return: (B,L,D)
        """
        # input_shape = input_embs.shape[:-1]
        # device = input_embs.device
        if not self.full_pass and merge_inputs is not None:
            # print('merge_inputs',merge_inputs.shape, merge_inputs[0,:10])
            time_embs = self.time2vec(merge_inputs.float())
            embs = embs + time_embs  #注意 embs+=time_embs为原址操作，在分支代码中易出错
        
        return embs


class MergeDurationEmbedding(InitParamBase):
    def __init__(self, hidden_size, context_dim=32, padding_value=None, log_scale=True,full_pass=False):
        super(MergeDurationEmbedding,self).__init__()

        self.full_pass = full_pass
        
        if not self.full_pass:
            self.mask_emb=TimeMaskEmbedding(hidden_size,context_dim,padding_value,log_scale)

    def forward(self, embs, merge_inputs=None):
        """
        embs:   (B, L, D)
        merge_inputs: (B,L)
        return: (B,L,D)


        """
        if not self.full_pass and merge_inputs is not None :
            masks = self.mask_emb(merge_inputs.float()) #(B,L,D)
            assert embs.shape==masks.shape

            # embs = torch.mul(embs, masks)
            embs = embs.mul(masks)

        return embs
            









        
    
