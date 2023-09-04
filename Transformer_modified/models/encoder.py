import torch.nn as nn

from models.blocks.encoder_layer import EncoderLayer
from models.layers.encoding_modules import SocialNodeEncoder, SpatialEncoder

class Encoder(nn.Module):
    """
    Encoder for modeling user representation (in social graph)
    """
    def __init__(self, data_path, spd_file, max_degree, num_user, d_model, d_ffn, num_heads, dropout, num_layers):
        """
        Args:
            data_path: path to dataset (ciao or epinions)
            spd_file: path to spd file (.npy)
            max_degree: max degree in social graph (can be fetched from `degree_table_social.csv`).
            num_user: number of total users in social graph (also can be fetched from `degree_table_social.csv`)
            d_model: embedding dimension (attention module)
            d_ffn: embedding dimension (FFN module)
            num_heads: number of heads in multi-headed attention
            dropout: dropout rate
            num_layers: number of encoder layers
        """
        super(Encoder, self).__init__()

        self.max_degree = max_degree
        self.num_user = num_user

        self.input_embed = SocialNodeEncoder(
            num_nodes = self.num_user,
            max_degree = self.max_degree,
            d_model = d_model
        )

        self.enc_layers = nn.ModuleList(
            [EncoderLayer(
                d_model = d_model,
                d_ffn = d_ffn,
                num_heads = num_heads,
                dropout = dropout
            ) for _ in range(num_layers)]
        )

        self.spatial_pos_bias = SpatialEncoder(
            data_path = data_path,
            spd_file = spd_file,
            num_heads = num_heads
        )
    
    def forward(self, batched_data, src_mask):
        # Input Encoding : Node id encoding + Degree encoding
            # [batch_size, seq_length, d_model]
        x = self.input_embed(batched_data)

        # Spatial Encoding
            # [batch_size, seq_length, seq_length, num_heads] ==> [batch_size, num_heads, seq_length, seq_length]
        attn_bias = self.spatial_pos_bias(batched_data).permute(0, 3, 2, 1)

        # Encoder layer forward pass (MHA, FFN)
        for layer in self.enc_layers:
            x = layer(x, src_mask, attn_bias)

        # [batch_size, seq_length, d_model]
        return x