# encoding: utf-8

import math
import numpy as np 
import torch
import torch.nn as nn 
import torch.nn.functional as F



# class TokenEmbedding(nn.Module):
#     def __init__(self, num_tokens, dim, padding_idx=None,  max_nrom = None):
#         super().__init__()
#         self.emb = nn.Embedding(num_tokens, dim,padding_idx=padding_idx, max_norm=max_nrom)

#     def forward(self, x):
#         token_emb = self.emb(x)
#         return token_emb

class AbsolutePositionalEmbedding(nn.Module):
    """
    learnable absolute position encoding 
    """
    def __init__(self, max_seq_len, dim, padding_idx=None,  max_nrom=None):
        super().__init__()
        self.emb = nn.Embedding(max_seq_len, dim, padding_idx=padding_idx, max_norm=max_nrom)

    def forward(self,  pos_ids):
        pos_emb = self.emb(pos_ids)
        return pos_emb

class FixedPositionalEmbedding(nn.Module):
    """
    sinusoid absolute position encoding ， specifically:
            PE(pos,2i) = sin(pos/10000^(2i/dim))
            PE(pos,2i+1) = cos(pos/10000^(2i/dim))

    implemention here diff slightly as we just concat the two parts.
    """
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1. / (10000 ** (torch.arange(0, dim, 2).float() / dim)) #(D//2)
        self.register_buffer('inv_freq', inv_freq)

    def forward(self,pos_ids, offset = 0):
        pos_ids = pos_ids.type_as(self.inv_freq) + offset

        position_emb  = torch.empty(*pos_ids.shape, dim, dtype=self.inv_freq.dtype)
        position_emb[:,:,0:2:]= torch.sin(pos_ids.unsqueeze(-1)*self.inv_freq)
        position_emb[:,:,1:2:]= torch.cos(pos_ids.unsqueeze(-1)*self.inv_freq)
        # sinusoid_inp = pos_ids.unsqueeze(-1) * self.inv_freq
        # emb = torch.cat((sinusoid_inp.sin(), sinusoid_inp.cos()), dim=-1)
        return position_emb


class EventTimeJointEmbedding(nn.Module):
    """
    time duration encoding, inherent from  paper `time-dependent representation for neural event sequence prediction`
    Args:
        d_model (int): dense feature的隐向量维度
        padding_value ([int], optional): padding_value 是填充值，其填充值对应的embed会被填充为 0. Defaults to None.
        hidden_embedding_dim ([type]):
        dropout (float, optional): dropout失活比例. Defaults to 0.1.
    """

    def __init__(self, d_model, padding_value=None, hidden_embed_dim=None, dropout=0.1):

        super().__init__()
        self.d_model = d_model
        self.padding_value = padding_value
        self.hidden_embed_dim = d_model if hidden_embed_dim is None else hidden_embed_dim
        self.w = nn.Parameter(torch.Tensor(self.hidden_embed_dim))
        self.b = nn.Parameter(torch.Tensor(self.hidden_embed_dim))
        self.embedding_matrix = nn.Parameter(torch.Tensor(self.hidden_embed_dim, self.d_model))

        self.dropout = nn.Dropout(p=dropout)

        nn.init.xavier_uniform_(self.embedding_matrix, gain=1.0)
        nn.init.uniform_(self.w)
        nn.init.uniform_(self.b)

    def forward(self, x):
        """
        x: (bsz, seq_len)
        """
        x1 = torch.unsqueeze(x, 2).float()  # [batch_size, series_size, 1]
        x1 = x1 * self.w + self.b  # [batch_size, series_size, hidden_embed_dim], hadamard product
        x1 = F.softmax(x1, dim=-1)

        if self.padding_value is not None:
            mask = torch.unsqueeze(x.ne(self.padding_value), 2).repeat(1,1,x1.shape[-1]) #[bsz,seq, hidden_dim]
            x1 = x1 * mask

        # [batch_size, series_size, d_model]
        output = torch.einsum("bsv,vi->bsi", x1, self.embedding_matrix)
        return output

class TimeMaskEmbedding(nn.Module):
    """
    time duration encoding,  inherent from paper `time-dependent representation for neural event sequence prediction`
    
    returns : (bsz, seq, d_model), as a kind of  mask operation, the  returned value should do dot production with  enetity embedding 
    """
    def __init__(self,  d_model, hidden_dim=None, padding_value=None, log_scale=True):
        super().__init__()
        
        #TODO (app-aware time masking)
        hidden_dim = hidden_dim if hidden_dim else d_model

        # self.W = nn.Parameter(torch.Tensor(size=(hidden_dim,d_model))
   
        self.context_fn = nn.Sequential(nn.Linear(1,hidden_dim,bias=False),
                                        nn.ReLU(),
                                        nn.Linear(hidden_dim,d_model,bias=True),
                                        nn.Sigmoid()
                                        )

        nn.init.xavier_normal_(self.context_fn[2].weight, gain=1.0) # sigmoid /tanh
        nn.init.kaiming_normal_(self.context_fn[0].weight) # ReLU

        self.log_scale=log_scale
        self.padding_value = padding_value

    def forward(self, x):
        """
        x: (B, L)

        returns: (B, L, D)
        """
        if self.log_scale:
            x = torch.log(x+1.e-8) # log_{e}

        x = x.unsqueeze(-1).float() # (B,L,1)
        tx = self.context_fn(x) #(B,L,D)

        if self.padding_value:
            #repeat copy data, while expand just return view
            mask = x.ne(self.padding_value).expand_as(tx) # =expand(-1,-1,d_model),
            tx = tx*mask # hadamard product 
        return tx  

class Time2VecEmbedding(nn.Module):
    """
    timestamp encoding inherent from paper `Time2Vec: Learning a Vector Representation of Time`;

    learnable sin/cos postition encoding;

    returns: (bsz, seq, hidden)
    """
    def __init__(self, d_model, include_non_periodic=False, padding_value=None):
        super().__init__()
        self.d_model = d_model
    
        self.linear = nn.Linear(1, d_model,bias=True)
        self.act = torch.sin  

        self.include_non_periodic = include_non_periodic
        self.padding_value = padding_value

        # nn.init.xavier_normal_(self.linear.weight)

    def forward(self, x):
        """
        x: (bsz,seq)

        return: (bsz, seq, d_model)
        """

        x = x.unsqueeze(-1).float() # (bsz,seq, 1)
        # print('---x---',x.shape,x.dtype,x[0,:10])
        tx = self.linear(x) #(bsz, seq, d_model)
        if self.include_non_periodic:
            x1 = tx[:,:,0]            # non-periodic
            x2 = self.act(tx[:,:,1:]) # periodic
            tx = torch.cat([x1, self.act(x2)], dim=-1)
        else:
            tx = self.act(tx)   # period is 2*pi/wi, input range from (0,1),so wi shoud be (0,2pi)


        if self.padding_value:
            #repeat copy data, while expand just return view
            mask = x.ne(self.padding_value).expand_as(tx) # =expand(-1,-1,d_model),
            tx = tx*mask # hadamard product 
        
        return tx
        














