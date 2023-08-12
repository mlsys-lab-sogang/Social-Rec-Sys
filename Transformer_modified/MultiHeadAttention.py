# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# This source code is little modified version of original one.
    # Graphormer Encoder (Encoding 방식, graph data의 처리 방식) 참고를 위해 우선 작성 진행 중.

import math
import numpy as np
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from Utils import DropoutModule

class MultiHeadAttention(nn.Module):
    """
    Multi-head Attention
    """
    def __init__(self, embed_dim, num_heads, dim_K, dim_V, dropout=0.0, bias=True, self_attention=False):
        super(MultiHeadAttention, self).__init__()

        # Embedding size
        self.embed_dim = embed_dim
        self.dim_K = dim_K if dim_K is not None else embed_dim
        self.dim_V = dim_V if dim_V is not None else embed_dim
        self.QKV_same_dim = self.dim_K == embed_dim and self.dim_V == embed_dim

        self.num_heads = num_heads

        self.dropout_module = DropoutModule(
            dropout, module_name=self.__class__.__name__
        )

        self.head_dim = embed_dim // num_heads
        assert(
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads."
        self.scaling = self.head_dim ** -0.5
        
        self.self_attention = self_attention

        assert self.self_attention, "Supporting self attention now."

        assert not self.self_attention or self.QKV_same_dim, (
            "Self-attention requires Q, K and " "V to be of the same size."
        ) 

        self.K_proj = nn.Linear(self.dim_K, embed_dim, bias=bias)
        self.V_proj = nn.Linear(self.dim_V, embed_dim, bias=bias)
        self.Q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.layer_norm = nn.LayerNorm(embed_dim)

        self.reset_parameters()

    def reset_parameters(self):
        if self.QKV_same_dim:
            """
            (From Graphormer Implementation)
            Empirically observed the convergence to be much better with
            the scaled initialization
            """
            nn.init.xavier_uniform_(self.K_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.V_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.Q_proj.weight, gain=1 / math.sqrt(2))
        else:
            nn.init.xavier_uniform_(self.K_proj.weight)
            nn.init.xavier_uniform_(self.V_proj.weight)
            nn.init.xavier_uniform_(self.Q_proj.weight)
        
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.0)
    
    def forward(
            self, 
            query, 
            key: Optional[torch.Tensor], 
            value: Optional[torch.Tensor], 
            attn_bias: Optional[torch.Tensor], 
            key_padding_mask: Optional[torch.Tensor] = None, 
            need_weights: bool = True, 
            attn_mask: Optional[torch.Tensor] = None, 
            before_softmax: bool = False,
            need_head_weights: bool = False,
        ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            key_padding_mask (ByteTensor): mask to exclude
                keys that are pads, of shape `(batch, src_len)`, where
                padding elements are indicated by 1s.
            need_weights (bool): return the attention weights,
                averaged over heads (default: False)
            attn_mask (ByteTensor): typically used to 
                implement causal attention, where the mask prevents the
                attention from looking forward in time (default: None).
            before_softmax (bool): return the raw attention weights and values
                before the attention softmax.
            need_head_weights (bool): return the attention
                weights for each head. Implies *neeg_weights*.
                (default: return the average attention weights over all heads.)
        """
        if need_head_weights:
            need_weights = True

        len_target, batch_size, embed_dim = query.size()
        len_source = len_target

        assert embed_dim == self.embed_dim, f"query dim {embed_dim} != {self.embed_dim}"
        assert list(query.size()) == [len_target, batch_size, embed_dim]

        ##### TODO: ?
        if key is not None:
            len_source, key_batch_size, _ = key.size()
        #####

        # query is x (node feature).
        q = self.Q_proj(query)  # (feature_dim, embed_dim)
        k = self.K_proj(query)  # (feature_dim, embed_dim)
        v = self.V_proj(query)  # (feature_dim, embed_dim)

        q *= self.scaling       # scale by sqrt

        # Make q,k,v to contiguous for reshaping.
            # See: 
            # https://stackoverflow.com/questions/48915810/what-does-contiguous-do-in-pytorch 
            # https://discuss.pytorch.kr/t/contiguous-tensor/889
        
        # (embed_dim, embed_dim) -> (len_target, batch_size*num_head, head_dim)
        q = (
            q.contiguous()
            .view(len_target, batch_size * self.num_heads, self.head_dim)
            .transpose(0,1)
        )

        # (embed_dim, embed_dim) -> (-1, batch_size*num_head, head_dim)
        if k is not None:
            k = (
                k.contiguous()
                .view(-1, batch_size * self.num_heads, self.head_dim)
                .transpose(0,1)
            )   

        # (embed_dim, embed_dim) -> (-1, batch_size*num_head, head_dim)
        if v is not None:
            v = (
                v.contiguous()
                .view(-1, batch_size * self.num_heads, self.head_dim)
                .transpose(0,1)
            )   
        
        assert k is not None
        assert k.size(1) == len_source

        ##### (from Graphormer) 
        # This is part of a workaround to get aroung fork/join parallelism
        # not supporting Optional types.
        if key_padding_mask is not None and key_padding_mask.dim() == 0:
            key_padding_mask = None
        
        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == batch_size
            assert key_padding_mask.size(1) == len_source

        # Compute attention weight, by Q*K
        # (batch_size*num_head, len_target, head_dim) * (batch_size*num_head, head_dim, len_source)
        # => (batch_size*num_head, len_target, len_source)
        attn_weights = torch.bmm(q, k.transpose(1, 2))
        attn_weights = self.apply_sparse_mask(attn_weights, len_target, len_source, batch_size)

        # att_weight shape will be like below.
        assert list(attn_weights.size()) == [batch_size * self.num_heads, len_target, len_source]

        # add attention bias
        if attn_bias is not None:
            attn_weights += attn_bias.view(batch_size * self.num_heads, len_target, len_source)

        # apply attention mask to attention weight
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0)
            attn_weights += attn_mask
        
        # don't attend to padding symbols
        if key_padding_mask is not None:
            attn_weights = attn_weights.view(batch_size, self.num_heads, len_target, len_source)
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool),
                float("-inf")
            )
            attn_weights = attn_weights.view(batch_size * self.num_heads, len_target, len_source)
        
        # get non-normalized attn_weight.
        if before_softmax:
            return attn_weights, v
        
        # compute nornalized attn_weights (& apply dropout)
        attn_weights_float = F.softmax(
            input = attn_weights,
            dim = -1
        )
        attn_weights = attn_weights_float.type_as(attn_weights)
        attn_probs = self.dropout_module(attn_weights)

        assert v is not None    # for checking

        # Compute attention score by A*V, where A=softmax(Q,K^T) 
        # (batch_size*num_head, len_target, len_source) * (batch_size*num_head, len_source, head_dim)
        # => (batch_size*num_head, len_target, head_dim)
        attn = torch.bmm(attn_probs, v)
        assert list(attn.size()) == [batch_size * self.num_heads, len_target, self.head_dim]

        attn = attn.transpose(0, 1).contiguous().view(len_target, batch_size, embed_dim)
        attn = self.out_proj(attn)

        # reshape attention weight.
        attn_weights: Optional[torch.Tensor] = None
        if need_weights:
            attn_weights = attn_weights_float.view(
                batch_size, self.num_heads, len_target, len_source
            ).transpose(1, 0)

            # average attention weights over heads
            if not need_head_weights:
                attn_weights = attn_weights.mean(dim=0)
        
        return attn, attn_weights
    
    ##### TODO: 그냥 받은 attn_weight를 그대로 넘겨주는데 masking이 어디?
    def apply_sparse_mask(self, attn_weights, len_target: int, len_source: int, batch_size: int):
        return attn_weights
    ##### 

