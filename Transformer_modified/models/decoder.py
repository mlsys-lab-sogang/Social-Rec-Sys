import torch.nn as nn

from models.blocks.decoder_layer import DecoderLayer
from models.layers.encoding_modules import ItemNodeEncoder, RatingEncoder

class Decoder(nn.Module):
    """
    Decoder for modeling item representation (in user-item graph),
    and perform rating prediction
    """
    def __init__(self, data_path, num_item, max_degree, d_model, d_ffn, num_heads, dropout, num_layers):
        """
        Args:
            data_path: path to dataset (ciao or epinions)
            num_item: number of total items in user-item graph (can be fetched from `degree_table_item.csv`)
            max_degree: max degree in user-item graph (also can be fetched from `degree_table_item.csv`)
            d_model: embedding dimension (attention module)
            d_ffn: embedding dimension (FFN module)
            num_heads: number of heads in multi-headed attention
            dropout: dropout rate
            num_layers: number of encoder layers
        """
        super(Decoder, self).__init__()

        self.num_item = num_item
        self.max_degree = max_degree

        self.input_embed = ItemNodeEncoder(
            num_nodes = self.num_item,
            max_degree = self.max_degree,
            d_model = d_model
        )

        self.dec_layers = nn.ModuleList(
            [DecoderLayer(
                d_model = d_model,
                d_ffn = d_ffn,
                num_heads = num_heads,
                dropout = dropout,
                last_layer = False
            ) for _ in range(num_layers)]
        )

        self.relation_bias = RatingEncoder()

        # rating prediction layer
        self.pred_layer = DecoderLayer(
            d_model = d_model,
            d_ffn = d_ffn,
            num_heads = num_heads,
            dropout = dropout,
            last_layer = True
        )
    
    def forward(self, batched_data, enc_output, trg_mask):
        # Input Encoding: Node it encoding + degree encoding
            # [batch_size, seq_length, item_length]
        x = self.input_embed(batched_data)

        # Rating encoding
            # [batch_size, seq_length, seq_length, num_heads]
        attn_bias = self.relation_bias(batched_data)#.permute(0, 3, 2, 1)

        # Decoder layer forward pass (MHA, FFN)
        for layer in self.dec_layers:
            x = layer(x, enc_output, trg_mask, attn_bias)

        # Pass to prediction layer
        output = self.pred_layer(x)

        return output