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
    
    def forward(self, Q, K, V, mask=None):
        # Input is 4-d tensor
            # (batch_size, head, length, d_tensor)
        batch_size, head, length, d_tensor = K.size()

        # 1. Compute similarity by Q.dot(K^T)
        K_T = K.transpose(2, 3)
        score = torch.matmul(Q, K_T) / math.sqrt(d_tensor)

        # 2. Apply attention mask
        if mask is not None:
            score = score.masked_fill(mask == 0, -10000)

        # 3. Pass score to softmax for making [0, 1] range.
        score = torch.softmax(score, dim=-1)

        # 4. Dot product with V
        V = torch.matmul(score, V)

        return V, score

class MultiHeadAttention(nn.Module):
    """
    Perform multi-head attention
    """
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()

        self.num_heads = num_heads
        self.attention = ScaledDotProductAttention()

        # Input projection
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)

        self.W_concat = nn.Linear(d_model, d_model)

    def forward(self, Q, K, V, mask=None):
        # 1. Dot produt with weight matrices
        Q, K, V = self.W_Q(Q), self.W_K(K), self.W_V(V)

        # 2. Split tensor by number of heads
        Q, K, V = self.split(Q), self.split(K), self.split(V)

        # 3. Perform scaled-dot product attention
        out, attn = self.attention(Q, K, V, mask=mask)

        # 4. Concat and pass to linear layer
        out = self.concat(out)
        out = self.W_concat(out)

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