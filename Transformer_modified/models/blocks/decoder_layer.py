import torch.nn as nn

from models.layers.multi_head_attention import MultiHeadAttention
from models.layers.feed_forward_network import FeedForwardNetwork

class DecoderLayer(nn.Module):
    """
    Input:
        fixed-length item sequences \n
        This items are interacted items of users in encoder's input random walk sequence.
    """
    def __init__(self, d_model, d_ffn, num_heads, dropout=0.1, last_layer:bool=False):
        super(DecoderLayer, self).__init__()

        self.last_layer_flag = last_layer

        # Self attention
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.dropout1 = nn.Dropout(p=dropout)

        # Multi-head attntion
        self.norm2 = nn.LayerNorm(d_model)
        self.cross_attention = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.dropout2 = nn.Dropout(p=dropout)

        if not self.last_layer_flag:
            # FFN
            self.norm3 = nn.LayerNorm(d_model)
            self.ffn = FeedForwardNetwork(d_model=d_model, ffn_size=d_ffn, dropout=dropout)
            self.dropout3 = nn.Dropout(p=dropout)
        
        if self.last_layer_flag:
            # prediction layer
                # [batch_size, seq_len_user, seq_len_item]
                # encoder의 입력으로 들어온 user들에 대해, decoder에 들어온 item들에 대한 예측된 평점을 출력
            # self.linear = nn.Linear(d_model, 1)
            self.last_attn = MultiHeadAttention(d_model=d_model, num_heads=num_heads, last_layer_flag=True)

    def forward(self, x, enc_output, trg_mask, src_mask, attn_bias):
        # print("//////// In Decoder Layer ////////")
        # 1. Perform self attention

        # print('\n[[[[[[[[ Decoder Self Attention 시작 ]]]]]]]]')
        residual = x
        x = self.norm1(x)
        x = self.attention(Q=x, K=x, V=x, mask=trg_mask, attn_bias=None)
        # quit()

        # 2. Add & Norm
        x = self.dropout1(x)
        x = x + residual

        # 3. Perform Encoder-Decoder cross attention (bias here)
        if enc_output is not None:
            # print('\n[[[[[[[[ Cross Attention 시작 ]]]]]]]]')
            residual = x
            
            enc_output = self.norm2(enc_output)
            x = self.norm2(x)

            # print("     Start cross attention....")
            # print(f"     enc_output: {enc_output.shape} /// self-attn output: {x.shape} /// src_mask: {src_mask.shape}")
            x = self.cross_attention(Q=x, K=enc_output, V=enc_output, mask=src_mask, attn_bias=attn_bias)

            # 4. Add & Norm
            # print(f"********* After cross attn: x {x.shape}, residual {residual.shape}")
            x = self.dropout2(x)
            x = x + residual
            
        if not self.last_layer_flag:
            # 5. FFN
            residual = x
            x = self.norm3(x)
            x = self.ffn(x)

            # 6. Add & Norm
            x = self.dropout3(x)
            x = x + residual
            return x

        else:
            # print('\n[[[[[[[[ Last layer 진입 ]]]]]]]]')
            # 7. last layer returns predicted ratings.
            # x = self.linear(x)
            # print(f"Entering Last Layer: x {x.shape}    enc_output {enc_output.shape}")
            x = self.last_attn(Q=x, K=enc_output, V=enc_output, mask=src_mask, attn_bias=attn_bias)
            return x