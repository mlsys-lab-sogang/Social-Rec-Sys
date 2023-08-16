import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size, ffn_size):
        super(FeedForwardNetwork, self).__init__()

        self.layer1 = nn.Linear(hidden_size, ffn_size)
        self.gelu = nn.GELU()
        self.layer2 = nn.Linear(ffn_size, hidden_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.gelu(x)
        x = self.layer2(x)
        
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, attention_dropout_rate, num_heads):
        super(MultiHeadAttention, self).__init__()

        # Initial settings
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.attn_size = head_dim = embed_dim // num_heads
        self.scaling = head_dim ** -0.5

        # Input embedding -> Q,K,V projection
        self.Q_proj = nn.Linear(embed_dim, num_heads * head_dim)
        self.K_proj = nn.Linear(embed_dim, num_heads * head_dim)
        self.V_proj = nn.Linear(embed_dim, num_heads * head_dim)
        
        self.attn_dropout = nn.Dropout(attention_dropout_rate)

        self.out_proj = nn.Linear(num_heads * head_dim, embed_dim)

        self.reset_parameters()
    
    def reset_parameters(self):
        """
        (From Graphormer Implementation)
        Empirically observed the convergence to be much better with
        the scaled initialization
        """
        nn.init.xavier_uniform_(self.Q_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.K_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.V_proj.weight, gain=1 / math.sqrt(2))

        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.0)
    
    def forward(self, Q, K, V, attn_bias=None, before_softmax=False):
        """
        Input shape: [batch, head, input_length, dim_tensor]
        """
        original_Q_size = Q.size()

        dim_K = self.attn_size
        dim_V = self.attn_size
        batch_size = Q.size(0)

        # Project Q, K, V
        Q = self.Q_proj(Q).view(batch_size, -1, self.num_heads, dim_K)
        K = self.K_proj(K).view(batch_size, -1, self.num_heads, dim_K)
        V = self.V_proj(V).view(batch_size, -1, self.num_heads, dim_V)

        # Reshape for multiplication
        Q = Q.transpose(1, 2)                   # [batch, head, len_Q, dim_K]
        K = K.transpose(1, 2).transpose(2, 3)   # [batch, head, dim_K, len_K]
        V = K.transpose(1, 2)                   # [batch, head, len_V, dim_V]

        ####### Perform Scaled Dot-Product Attention #######
                # A = (Q * K^T) / sqrt(dim_K)
                # Attention = softmax(A) * V
        Q = Q * self.scaling
        attn_weights = torch.matmul(Q, K)                   # [batch, head, len_Q, len_K]

        # Add Attention bias (spatial encoding, edge encoding)
        if attn_bias is not None:
            attn_weights = attn_weights + attn_bias
        
        if before_softmax:
            return attn_weights, V

        attn_weights = F.softmax(attn_weights, dim=3)   
        attn_weights = self.attn_dropout(attn_weights)

        attn_weights = torch.matmul(attn_weights, V)        # [batch, head, len_Q, attn]

        attn_weights = attn_weights.transpose(1, 2).contiguous()    # [batch, len_Q, head, attn]
        attn_weights = attn_weights.view(batch_size, -1, self.num_heads * dim_V)

        attn_weights = self.out_proj(attn_weights)

        assert (attn_weights.size() == original_Q_size), f"Shape mismatch: attn_weights {attn_weights.size()} != Q: {original_Q_size}"
        
        return attn_weights
        ####### Perform Scaled Dot-Product Attention #######

class EncoderLayer(nn.Module):
    """
    Input: social graph (trustnetwork)
    """
    def __init__(self, embed_dim, ffn_size, dropout_rate, attention_dropout_rate, num_heads):
        super(EncoderLayer, self).__init__()

        self.self_attention_norm = nn.LayerNorm(embed_dim)
        self.self_attention = MultiHeadAttention(
            embed_dim, attention_dropout_rate, num_heads
        )
        self.self_attention_dropout = nn.Dropout(dropout_rate)

        self.ffn_norm = nn.LayerNorm(embed_dim)
        self.ffn = FeedForwardNetwork(embed_dim, ffn_size)
        self.ffn_dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x, attn_bias=None):
        y = self.self_attention_norm(x)
        y = self.self_attention(y, y, y, attn_bias)     # y -> Q, K, V
        y = self.self_attention_dropout(y)
        x = x + y

        y = self.ffn_norm(x)
        y = self.ffn(y)
        y = self.ffn_dropout(y)
        x = x + y

        return x

class DecoderLayer(nn.Module):
    """
    Input: rating matrix (user-item graph)
    """
    pass