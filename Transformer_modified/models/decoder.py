import torch
import torch.nn as nn

from models.blocks.decoder_layer import DecoderLayer
from models.layers.encoding_modules import ItemNodeEncoder, RatingEncoder

from model_utils import generate_attn_pad_mask

class Decoder(nn.Module):
    """
    Decoder for modeling item representation (in user-item graph),
    and perform rating prediction
    """
    def __init__(self, num_item, max_degree, d_model, d_ffn, num_heads, dropout, num_layers):
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
                last_layer = False,
                is_dec_layer = True
            ) for _ in range(num_layers)]
        )

        self.relation_bias = RatingEncoder(num_heads=num_heads)

        # rating prediction layer
        self.pred_layer = DecoderLayer(
            d_model = d_model,
            d_ffn = d_ffn,
            num_heads = num_heads,
            dropout = dropout,
            last_layer = True,
            is_dec_layer = True
        )
    
    def forward(self, batched_data, enc_output, src_mask):
        # Input Encoding: Node it encoding + degree encoding
            # [batch_size, seq_length, item_length]
        x = self.input_embed(batched_data)

        # Generate mask for padded data
            # FIXME: 현재 데이터/task 에선 subsequent masking에 의미가 X.
        dec_self_attn_mask = generate_attn_pad_mask(batched_data['item_list'], batched_data['item_list']).cuda()    # [batch_size, seq_len_item, seq_len_item]
        # print('\n<<<<<<<<<< Decoder의 self attention pad mask >>>>>>>>>>')
        # print(dec_self_attn_mask[0][:][0].data)
        # print(dec_self_attn_mask[0][:][0].shape)
        # print(dec_self_attn_mask.shape)
        # quit()
        # dec_self_attn_subsequent_mask = generate_attn_subsequent_mask(batched_data['item_list']).cuda()             # [batch_size, seq_len_item, seq_len_item]
        # print('\n<<<<<<<<<< Decoder의 self attention subsequent mask >>>>>>>>>>')
        # print(dec_self_attn_subsequent_mask[0][:][0].data)
        # quit()

        # trg_mask = torch.gt((dec_self_attn_mask + dec_self_attn_subsequent_mask), 0)    # [batch_size, seq_len_item, seq_len_item]
        trg_mask = dec_self_attn_mask

        dec_enc_mask = generate_attn_pad_mask(batched_data['item_list'], batched_data['user_seq']).cuda()
        src_mask = dec_enc_mask     # [batch_size, seq_len_item, seq_len_user]

        # del dec_self_attn_mask, dec_self_attn_subsequent_mask, dec_enc_mask
        del dec_self_attn_mask, dec_enc_mask

        # Rating encoding
            # [batch_size, seq_length_user, seq_length_item, num_heads]
            # ==> [batch_size, num_heads, seq_length_user, seq_length_item]
        attn_bias = self.relation_bias(batched_data).permute(0, 3, 2, 1)
        # attn_bias=None

        # Decoder layer forward pass (MHA, FFN)
        for layer in self.dec_layers:
            x = layer(x, enc_output, trg_mask, src_mask, attn_bias)

        # Pass to prediction layer
        output = self.pred_layer(x, enc_output, trg_mask, src_mask, attn_bias)

        return output