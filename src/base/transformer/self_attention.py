# encoding: utf-8

import logging 
import math 
import os 
import warnings

import torch
import torch.utils.checkpoint
from torch import Tensor, device, dtype
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from typing import List,Tuple,Dict

from .sublayers import ( MultiHeadAttentionLayer,
                         PositionWiseForwardLayer, TransformerLayer)


class PretrainBaseModel(nn.Module):
    """ An abstract class to handle weights initialization and mask mantainance
    """
    def __init__(self, config):

        super().__init__()
        if not isinstance(config, Dict):
            raise ValueError("Parameter config in `{}(config)` should be an instance of dict`. ".format(self.__class__.__name__))

        self.config = config

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config['initializer_range'])
        elif isinstance(module, torch.nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def init_weights(self):
        """ Initialize and prunes weights if needed. """
        # Initialize weights
        self.apply(self._init_weights)
        # Tie weights if needed
        # self.tie_weights()
 
    def tie_weights(self):
        """
        Tie the weights between the input embeddings and the output embeddings.
        """
        output_embeddings = self.get_output_embeddings()
        input_embeddings = self.get_input_embeddings()
        output_embeddings.weight = input_embeddings.weight

    def get_input_embeddings(self):
        return None

    def get_output_embeddings(self):
        return None  # Overwrite for models with output embeddings
    
    @property
    def device(self) -> device:
        """
        Get torch.device from module, assuming that the whole module has one device.
        """
        try:
            return next(self.parameters()).device
        except StopIteration:
            # For nn.DataParallel compatibility in PyTorch 1.5

            def find_tensor_attributes(module: nn.Module) -> List[Tuple[str, Tensor]]:
                tuples = [(k, v) for k, v in module.__dict__.items() if torch.is_tensor(v)]
                return tuples

            gen = self._named_members(get_members_fn=find_tensor_attributes)
            first_tuple = next(gen)
            return first_tuple[1].device

    @property
    def dtype(self) -> dtype:
        """
        Get torch.dtype from module, assuming that the whole module has one dtype.
        """
        try:
            return next(self.parameters()).dtype
        except StopIteration:
            # For nn.DataParallel compatibility in PyTorch 1.5

            def find_tensor_attributes(module: nn.Module) -> List[Tuple[str, Tensor]]:
                tuples = [(k, v) for k, v in module.__dict__.items() if torch.is_tensor(v)]
                return tuples

            gen = self._named_members(get_members_fn=find_tensor_attributes)
            first_tuple = next(gen)
            return first_tuple[1].dtype

    def invert_attention_mask(self, encoder_attention_mask: Tensor) -> Tensor:
        """type: torch.Tensor -> torch.Tensor
        input:  (bsz, seq,seq)  or (bsz,seq)
        return: (bsz,1,seq,seq) or (bsz, 1,1,seq) with 0 filled with -10000
        """

        if encoder_attention_mask.dim() == 3: #(bsz, seq,seq)
            encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]
        if encoder_attention_mask.dim() == 2: #(bsz,seq)
            encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]  #batch中一条纪录每个head的每个序列位置的mask都是一样的
        encoder_extended_attention_mask = encoder_extended_attention_mask.to(dtype=self.dtype)  # fp16 compatibility
        if self.dtype == torch.float16:
            encoder_extended_attention_mask = (1.0 - encoder_extended_attention_mask) * -1e4
        elif self.dtype == torch.float32:
            encoder_extended_attention_mask = (1.0 - encoder_extended_attention_mask) * -1e9
        else:
            raise ValueError("{} not recognized. `dtype` should be set to either `torch.float32` or `torch.float16`".format(self.dtype))

        return encoder_extended_attention_mask

    def get_extended_attention_mask(self, attention_mask: Tensor, input_shape: Tuple, device: device) -> Tensor:
        """Makes broadcastable attention mask and causal mask so that future and maked tokens are ignored.

        attention_mask: (bsz,seq,seq) or  (bsz, seq) ,  1 indicating tokens to ATTEND to
        input_shape: (bsz,seq)  shape of input_ids

        Returns: (bsz,1,seq,seq) or (bsz,1,1,seq)
           
        """
        if attention_mask.dim() == 3:   # [bsz, seq, seq]
            extended_attention_mask = attention_mask[:, None, :, :]  #broadcastable to all heads.
        elif attention_mask.dim() == 2: # [bsz, seq]
            # Provided a padding mask of dimensions 
            # - if the model is a decoder, apply a causal mask in addition to the padding mask
            # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
            if self.config.get('is_decoder',False):
                batch_size, seq_length = input_shape
                seq_ids = torch.arange(seq_length, device=device)
                causal_mask = seq_ids[None, None, :].repeat(batch_size, seq_length, 1) <= seq_ids[None, :, None] #一条序列不同位置的mask并不相同
                # causal and attention masks must have same dtype with pytorch version < 1.3
                causal_mask = causal_mask.to(attention_mask.dtype)
                extended_attention_mask = causal_mask[:, None, :, :] * attention_mask[:, None, None, :] # 有0的一定为0
            else:
                extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError("Wrong shape for attention_mask (shape {})".format(input_shape, attention_mask.shape))
        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        extended_attention_mask = extended_attention_mask.to(dtype=self.dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def check_legal_layer_params(self, config):
        allow_kws  = ('hidden_size',
                      'num_attention_heads',
                      'hidden_dropout_prob', 
                      'attention_probs_dropout_prob',
                      'is_decoder',
                      'intermediate_size', 
                      'hidden_act', 
                      'layer_norm_eps'
                      )
        kws = {key: config[key] for key in config if key in allow_kws}

        return kws

class TransformerEncoder(PretrainBaseModel):
    """
       N x TransformerLayer
    """
    def __init__(self, config: Dict):
        super().__init__(config)

        num_hidden_layers = config['num_hidden_layers']
        params = self.check_legal_layer_params(config)
        
        self.layers = nn.ModuleList([TransformerLayer(**params) for _ in range(num_hidden_layers)])

        self.init_weights()

    def forward(self, hidden_states,
                    attention_mask=None,
                    head_mask=None,
                    encoder_hidden_states=None,
                    encoder_attention_mask=None,
                    output_attentions=False,
                    output_hidden_states=False):
        """     
        hidden_states: (bsz, seq,hidden_size)
        attention_mask: (bsz, seq) or (bsz, seq, seq)
        head_mask : (num_layers,heads) or (heads)
        encoder_hidden_states" (bsz, seq, hidden_size)
        encoder_attention_mask: (bsz,seq) or (bsz, seq,seq)
        """

        input_shape = hidden_states.size()[:-1]
        device = hidden_states.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)  #(batch, seq)
        
        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads. [batch_size, num_heads, seq_length, seq_length]
        # pad mask for encoder self, pad & causl mask for decoder self-attention and pad mask for decoder cross_attention
        #  (bsz,1,seq,seq) or (bsz,1,1,seq), -10000填充0值
        if attention_mask is not None:
            attention_mask = self.get_extended_attention_mask(attention_mask, input_shape, device)

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        # head_mask = self.get_head_mask(head_mask, num_hidden_layers)
        head_mask = [None]* len(self.layers)

        # If a 2D ou 3D  mask is provided for the  decoder cross-attention
        # we need to make broadcastabe to [batch_size, num_heads, seq_length, seq_length]
        if self.config.get('is_decoder',False) and encoder_hidden_states is not None:   
            encoder_hidden_shape = encoder_hidden_states.size()[:2]
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_attention_mask = self.invert_attention_mask(encoder_attention_mask) # (bsz,1,seq,seq) or (bsz,1,1,seq)
        else:
            encoder_attention_mask = None


        all_hidden_states = ()
        all_attentions = ()
        for i, layer_module in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(hidden_states,
                                        attention_mask,
                                        head_mask[i],
                                        encoder_hidden_states,
                                        encoder_attention_mask,
                                        output_attentions
                                        )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if output_attentions:
            outputs = outputs + (all_attentions,)
        return outputs  # (last-layer hidden state, all hidden states, all attentions)

class TransformerDecoder(nn.Module):
    pass

