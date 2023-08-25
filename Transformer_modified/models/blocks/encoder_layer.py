import torch.nn as nn

from layers.multi_head_attention import MultiHeadAttention
from layers.feed_forward_network import FeedForwardNetwork

class EncoderLayer(nn.Module):
    """
    Input:
        fixed-length random walk sequence (generated from social graph)
    """
    def __init__(self, d_model, d_ffn, num_heads, dropout=0.1):
        super(EncoderLayer, self).__init__()

        self.attention = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(p=dropout)

        self.ffn = FeedForwardNetwork(d_model=d_model, ffn_size=d_ffn, dropout=dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(p=dropout)
    
    def forward(self, x, src_mask):
        # 1. Perform self attention
        residual = x
        x = self.attention(Q=x, K=x, V=x, mask=src_mask)

        # 2. Add & Norm
        x = self.dropout1(x)
        x = self.norm1(x + residual)

        # 3. FFN
        residual = x
        x = self.ffn(x)

        # 4. Add & Norm
        x = self.dropout2(x)
        x = self.norm2(x + residual)

        return x
