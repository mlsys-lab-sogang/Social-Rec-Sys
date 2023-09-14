import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class ScaledDotProductAttention(nn.Module):
    """
    Perform scaled dot product attention
    """
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()
    
    def forward(self, Q, K, V, mask=None, attn_bias=None, last_layer_flag=False):
        # Input is 4-d tensor
            # [batch_size, head, length, d_tensor]
        batch_size, head, length, d_tensor = K.size()

        # 1. Compute similarity by Q.dot(K^T)
            # d_tensor = d_model // num_head
            # [batch_size, num_heads, seq_length, d_tensor] ==> [batch_size, num_heads, d_tensor, seq_length]
        K_T = K.transpose(2, 3)
        score = torch.matmul(Q, K_T) / math.sqrt(d_tensor)
            # ==> [batch_size, num_heads, seq_length, seq_length]
        # print(f"////// After Q*KT: {score.shape}")

        # 2. Apply attention mask
        if mask is not None:
            score = score.masked_fill(mask == 0, -10000)

        # 3. Apply attention bias (spatial encoding)
        # TODO: add attention bias before softmax
            # [batch_size, num_head, seq_length, seq_length]
        if attn_bias is not None:
            score += attn_bias

        ### Decoder 마지막 layer에서 Q * K.T 한 결과를 output으로 출력
        if last_layer_flag:
            score = torch.mean(score, dim=1)
            return score
        ###

        # 3. Pass score to softmax for making [0, 1] range.
        score = torch.softmax(score, dim=-1)

        # 4. Dot product with V
            # [batch_size, num_heads, seq_length, d_tensor]
        V = torch.matmul(score, V)

        # print(f"////// After (Q*KT)V: score {score.shape} V(return) {V.shape}")

        return V, score

class MultiHeadAttention(nn.Module):
    """
    Perform multi-head attention
    """
    def __init__(self, d_model, num_heads, last_layer_flag=False):
        super(MultiHeadAttention, self).__init__()

        self.num_heads = num_heads
        self.attention = ScaledDotProductAttention()
        self.last_layer_flag = last_layer_flag

        # Input projection
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)

        self.W_concat = nn.Linear(d_model, d_model)

    def forward(self, Q, K, V, mask=None, attn_bias=None):
        # 1. Dot produt with weight matrices
            # [batch_size, seq_length, d_model]
        Q, K, V = self.W_Q(Q), self.W_K(K), self.W_V(V)

        # 2. Split tensor by number of heads
            # d_tensor = d_model // num_heads
            # [batch_size, num_heads, seq_lengths, d_tensor]
        Q, K, V = self.split(Q), self.split(K), self.split(V)

        # Apply mask for multi-head attention
            # [batch_size, len_q, len_k] ==> [batch_size, num_heads, len_q, len_k]
        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)

        ####### Decoder의 마지막 layer (cross-attn)는 rating prediction을 수행
        if not self.last_layer_flag:
            # 3. Perform scaled-dot product attention
            out, attn = self.attention(Q, K, V, mask, attn_bias)
        else:
            # print(f"        Last Layer shapes : Q ({Q.shape})   K ({K.shape})   V ({V.shape})")
            out = self.attention(Q, K, V, mask, attn_bias, self.last_layer_flag)
            return out
        #######

        # print(f"$$$$$$$$$$$$$$ After MHA: out {out.shape}")

        # 4. Concat and pass to linear layer
            # (batch_size, seq_length, d_model)
        out = self.concat(out)
        out = self.W_concat(out)

        # print(f"$$$$$$$$$$$$$$ After MHA and concat: out {out.shape}")

        return out
    
    def split(self, tensor):
        """
        Split tensor by number of heads

        Input tensor shape: 
            (batch_size, length, d_model)
        Outout tensor shape:
            (batch_size, num_head, length, d_tensor)
        """
        batch_size, length, d_model = tensor.size()

        d_tensor = d_model // self.num_heads
        tensor = tensor.view(batch_size, length, self.num_heads, d_tensor).transpose(1, 2)

        return tensor

    def concat(self, tensor):
        """
        Inverse function of self.split()

        Input tensor shape:
            (batch_size, num_head, length, d_tensor)
        Output tensor shape:
            (batch_size, length, d_model)
        """
        batch_size, num_head, length, d_tensor = tensor.size()
        
        d_model = num_head * d_tensor
        tensor = tensor.transpose(1, 2).contiguous().view(batch_size, length, d_model)

        return tensor