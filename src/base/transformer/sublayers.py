# encoding: utf-8

import math 
import os 
import torch
# from torch import Tensor, device, dtype
from torch import nn
from  .activations import get_activation

class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class ScaledDotProductAttention(nn.Module):
    """
    softmax(Q.K^T/sqrt(d)).V
    mask填充很大的负数, 比如-1e9 使其权重减少到0
    """
    def __init__(self, attention_query_head_size, attention_probs_dropout_prob=0.1):
        super().__init__()
        self.d_k     = attention_query_head_size
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(attention_probs_dropout_prob)

    def forward(self, Q, K, V, attention_mask=None,head_mask=None):
        """
        Q: [bsz, heads, seq, head_size]
        K: [bsz, heads, seq, head_size]
        V: [bsz, heads, seq, head_size]

        """
        attention_scores = torch.matmul(Q, K.transpose(-1, -2)) #batch-wise matrix multiply  (bsz, heads, seq, seq)  
        attention_scores = attention_scores / math.sqrt(self.d_k) # to avoid saturation and keep variance 

        if attention_mask is not None: #[bsz, heads ,seq ,seq] or [bsz,1,seq,seq]
            # attention_scores.masked_fill_(attention_mask,-1e9) # 将attention_mask元素为1的地方填充value
            attention_scores = attention_scores + attention_mask  # (-1e9)

        attention_probs = self.softmax(attention_scores)  # (bsz, heads, seq, seq)
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None: #[bsz,heads,seq,seq] or [bsz,heads,1,1]
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, V) # (bsz,heads,seq, head_size)

        return context_layer, attention_probs
        
class MultiHeadAttentionLayer(nn.Module):
    """
    multli-head attention  + add&norm
    """
    def __init__(self, hidden_size=None, 
                       num_attention_heads=None, 
                       hidden_dropout_prob = 0.1, 
                       attention_probs_dropout_prob=0.1,
                       layer_norm_eps=1.0e-12):
        super().__init__()
        # attention_head_size 理论可以不等于 hidden_size/attention_heads, 通过最后的一个全连接层匹配回来
        # (Q,K) 和V的 dim也可以不同
        if hidden_size % num_attention_heads != 0:  
            raise ValueError("hidden_size({}) % attention_heads({})!=0" .format(hidden_size, num_attention_heads))
        
        self.attention_head_size = int(hidden_size /num_attention_heads)
        self.num_attention_heads = num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        self.query = nn.Linear(hidden_size, self.all_head_size) #(hidden_size, heads*d_q)
        self.key = nn.Linear(hidden_size, self.all_head_size)   #(hidden_size, heads*d_k)
        self.value = nn.Linear(hidden_size, self.all_head_size) #(hidden_size, heads*d_v)

        self.attention = ScaledDotProductAttention(self.attention_head_size, attention_probs_dropout_prob)

        self.dense = nn.Linear(self.all_head_size, hidden_size)   
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.layernorm = torch.nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        
    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)  #(bsz,seq, heads, head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)  #(bsz, heads, seq, head_size)

    def forward(self, hidden_states, 
                      attention_mask=None, 
                      head_mask=None, 
                      encoder_hidden_states=None,
                      encoder_attention_mask=None,
                      output_attentions=False):
        """
        hidden_states: (bsz,seq, hidden_size)
        attention_mask: (bzs, 1, seq, seq) or (bsz, 1,1,seq)
        head_mask: (bsz, heads, seq, seq)
        return： 
          context_layer:   (bsz, seq, all_head_size)
          attention_probs: (bsz, heads, seq, seq)
        """
        Q = self.query(hidden_states)  #  (bsz,seq, all_head_size),  

        if encoder_hidden_states:  # cross attention
            K = self.key(encoder_hidden_states)
            V = self.value(encoder_hidden_states)
            attention_mask = encoder_attention_mask
        else:
            K = self.key(hidden_states)      #  all_head_size = heads*head_size
            V = self.value(hidden_states)
        
        Q = self.transpose_for_scores(Q) #  #(bsz, heads, seq, head_size)
        K = self.transpose_for_scores(K)
        V = self.transpose_for_scores(V)
        
        # if attention_mask: #(bzs, seq, seq)
        #     attention_mask = attention_mask.unsqueeze(1).repeat(1,self.num_attention_heads,1,1)
        #     # attention_mask = attention_mask[:,None,:,:] # auto broadcast to [bsz,heads,seq,seq]
        # if head_mask and head_mask.dim() == 2:
        #     head_mask = head_mask[:,:,None,None]

        # context_layer: (bsz,heads,seq, head_size)
        # attention_probs:  (bsz, heads, seq, seq)
        context_layer, attention_probs = self.attention(Q, K, V, attention_mask, head_mask) 

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()  #(bsz, seq, heads,head_size)
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,) 
        context_layer = context_layer.view(*new_context_layer_shape)  #(bsz, seq, all_head_size)

        # LayerNorm(dropout(f(x))+x)
        context_layer = self.dense(context_layer) #(bsz, seq, hidden_size)
        context_layer = self.layernorm(self.dropout(context_layer) + hidden_states) 
        

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs

class PositionWiseForwardLayer(nn.Module):
    def __init__(self,  hidden_size, 
                        intermediate_size=None, 
                        hidden_act ='gelu',
                        hidden_dropout_prob=0.1,
                        layer_norm_eps=1.0e-12):
        super().__init__()
        if intermediate_size is None:
            intermediate_size = 4* hidden_size
        
        self.dense1 = nn.Linear(hidden_size, intermediate_size)
        if isinstance(hidden_act,str):
            self.intermediate_act_fn = get_activation(hidden_act)
        else:
            self.intermediate_act_fn = hidden_act

        self.dense2 = nn.Linear(intermediate_size, hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.layernorm = torch.nn.LayerNorm(hidden_size,eps=layer_norm_eps)
    
    def forward(self,hidden_states):
        """
        hidden_states: (bsz, seq, hidden_size)
        """
        shortcut = hidden_states
        hidden_states = self.dense1(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.dense2(hidden_states)

        # add & layerNorm
        output = self.layernorm(self.dropout(hidden_states) + shortcut)

        return output 

class TransformerLayer(nn.Module):
    """
    self_attention_layer + [cross_attention_layer]+ feedforward_layer

    """
    def __init__(self, hidden_size=None, 
                       num_attention_heads=1,
                       hidden_dropout_prob = 0.1, 
                       attention_probs_dropout_prob=0.1,
                       is_decoder=False,
                       intermediate_size = None, 
                       hidden_act ='gelu', 
                       layer_norm_eps = 1.0e-12,
                       ):

        super().__init__()
        
        self.attention = MultiHeadAttentionLayer(hidden_size=hidden_size, 
                                                 num_attention_heads=num_attention_heads,
                                                 hidden_dropout_prob=hidden_dropout_prob, 
                                                 attention_probs_dropout_prob=attention_probs_dropout_prob,
                                                 layer_norm_eps=layer_norm_eps)

        self.is_decoder = is_decoder

        if self.is_decoder:
            self.crossattention = MultiHeadAttentionLayer(hidden_size=hidden_size, 
                                                         num_attention_heads=num_attention_heads,
                                                         hidden_dropout_prob=hidden_dropout_prob, 
                                                         attention_probs_dropout_prob=attention_probs_dropout_prob,
                                                         layer_norm_eps=layer_norm_eps)

        self.ffn = PositionWiseForwardLayer(hidden_size=hidden_size, 
                                            intermediate_size=intermediate_size, 
                                            hidden_act =hidden_act,
                                            hidden_dropout_prob=hidden_dropout_prob,
                                            layer_norm_eps=layer_norm_eps)

    def forward(self,
                hidden_states,   
                attention_mask=None,    #for encoder self-attention or decoder self-attention
                head_mask=None,
                encoder_hidden_states=None,    # for decoder cross attention 
                encoder_attention_mask=None,   # for decoder cross attention 
                output_attentions=False,
               ):
        """
        hidden_states: (bsz, seq,hidden_size)
        attention_mask: (bsz, 1,1, seq) or (bsz, 1, seq, seq)
        head_mask : (num_layers, bsz, heads, seq, seq)
        encoder_hidden_states" (bsz, seq, hidden_size)
        encoder_attention_mask: (bsz, 1,seq,seq) or (bsz, 1,1, seq)
        """

        self_attention_outputs = self.attention(hidden_states, 
                                                attention_mask,   # 考虑了causal + pad_mask
                                                head_mask, 
                                                output_attentions=output_attentions,
                                                )
        attention_output = self_attention_outputs[0]   # (bsz, seq, hidden_size)
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        if self.is_decoder and encoder_hidden_states is not None:
            cross_attention_outputs = self.crossattention(  attention_output,
                                                            attention_mask,
                                                            head_mask,
                                                            encoder_hidden_states,
                                                            encoder_attention_mask, # attention_mask 被覆盖为enconder_attention_mask
                                                            output_attentions,
                                                         )
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:]  # add cross attentions if we output attention weights

        layer_output = self.ffn(attention_output)

        outputs = (layer_output,) + outputs  if output_attentions else (layer_output,) 

        return outputs # (outputs, self_atten_out, cross_atten_out)


