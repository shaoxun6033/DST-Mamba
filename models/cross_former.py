import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from layers.cross_encoder import Encoder
from layers.cross_decoder import Decoder
from layers.attn import FullAttention, AttentionLayer, TwoStageAttentionLayer
from layers.cross_embed import DSW_embedding

from math import ceil

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        # ===== 从 configs 读取（对齐 PatchTST）=====
        self.data_dim = configs.enc_in
        self.in_len = configs.seq_len
        self.out_len = configs.pred_len

        self.seg_len = configs.seg_len
        self.merge_win = configs.win_size

        d_model = configs.d_model
        d_ff = configs.d_ff
        n_heads = configs.n_heads
        e_layers = configs.e_layers
        dropout = configs.dropout
        factor = configs.factor

        self.baseline = getattr(configs, 'baseline', False)
        self.device = getattr(configs, 'device', torch.device('cuda:0'))

        # ===== padding（保持原逻辑）=====
        self.pad_in_len = ceil(1.0 * self.in_len / self.seg_len) * self.seg_len
        self.pad_out_len = ceil(1.0 * self.out_len / self.seg_len) * self.seg_len
        self.in_len_add = self.pad_in_len - self.in_len

        # ===== Embedding =====
        self.enc_value_embedding = DSW_embedding(self.seg_len, d_model)

        self.enc_pos_embedding = nn.Parameter(
            torch.randn(1, self.data_dim, (self.pad_in_len // self.seg_len), d_model)
        )

        self.pre_norm = nn.LayerNorm(d_model)

        # ===== Encoder =====
        self.encoder = Encoder(
            e_layers,
            self.merge_win,
            d_model,
            n_heads,
            d_ff,
            block_depth=1,
            dropout=dropout,
            in_seg_num=(self.pad_in_len // self.seg_len),
            factor=factor
        )

        # ===== Decoder =====
        self.dec_pos_embedding = nn.Parameter(
            torch.randn(1, self.data_dim, (self.pad_out_len // self.seg_len), d_model)
        )

        self.decoder = Decoder(
            self.seg_len,
            e_layers + 1,
            d_model,
            n_heads,
            d_ff,
            dropout,
            out_seg_num=(self.pad_out_len // self.seg_len),
            factor=factor
        )

    def forward(self, x_seq):
        if self.baseline:
            base = x_seq.mean(dim=1, keepdim=True)
        else:
            base = 0

        batch_size = x_seq.shape[0]

        if self.in_len_add != 0:
            x_seq = torch.cat(
                (x_seq[:, :1, :].expand(-1, self.in_len_add, -1), x_seq),
                dim=1
            )

        x_seq = self.enc_value_embedding(x_seq)
        x_seq += self.enc_pos_embedding
        x_seq = self.pre_norm(x_seq)

        enc_out = self.encoder(x_seq)

        dec_in = repeat(
            self.dec_pos_embedding,
            'b ts_d l d -> (repeat b) ts_d l d',
            repeat=batch_size
        )

        predict_y = self.decoder(dec_in, enc_out)

        return base + predict_y[:, :self.out_len, :]