import math
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F

import utils
from attention import OriginalFullAttention, BigbirdSimulatedAttention, BigbirdBlockSpareAttention


def create_pad_mask(t, pad):
    mask = (t == pad)
    return mask


def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


class Linear(nn.Module):
    """
    Linear Module
    """

    def __init__(self, in_dim, out_dim, activation=None, bias=True, w_init='linear'):
        """
        :param in_dim: dimension of input
        :param out_dim: dimension of output
        :param bias: boolean. if True, bias is included.
        :param w_init: str. weight inits with xavier initialization.
        """
        super(Linear, self).__init__()
        self.linear_layer = nn.Linear(in_dim, out_dim, bias=bias)
        self.activation = activation

        nn.init.xavier_uniform_(
            self.linear_layer.weight,
            gain=nn.init.calculate_gain(w_init))

    def forward(self, x):
        x = self.linear_layer(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class Conv(nn.Module):
    """
    Convolution Module
    """

    def __init__(self, in_channels, out_channels, kernel_size=(1, 1), stride=(2, 2),
                 padding=(0, 0), dilation=(1, 1), bias=True, w_init='linear'):
        """
        :param in_channels: dimension of input
        :param out_channels: dimension of output
        :param kernel_size: size of kernel
        :param stride: size of stride
        :param padding: size of padding
        :param dilation: dilation rate
        :param bias: boolean. if True, bias is included.
        :param w_init: str. weight inits with xavier initialization.
        """
        super(Conv, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation,
                              bias=bias)

        nn.init.xavier_uniform_(
            self.conv.weight, gain=nn.init.calculate_gain(w_init))

    def forward(self, x):
        x = self.conv(x)
        return x


class MultiHeadedAttentionLayer(nn.Module):
    def __init__(self,
                 attention_type,
                 hidden_size=768,
                 num_attention_heads=1,
                 num_rand_blocks=3,
                 size_per_head=512,
                 from_block_size=16,
                 to_block_size=16,
                 attention_probs_dropout_prob=0.0,
                 use_bias=True,
                 seed=None,
                 query_act=None,
                 key_act=None,
                 value_act=None):

        super(MultiHeadedAttentionLayer, self).__init__()

        self.num_attention_heads = num_attention_heads
        self.size_per_head = size_per_head

        self.query_layer = Linear(hidden_size, size_per_head * num_attention_heads, query_act, bias=use_bias)

        self.key_layer = Linear(hidden_size, size_per_head * num_attention_heads, key_act, bias=use_bias)

        self.value_layer = Linear(hidden_size, size_per_head * num_attention_heads, value_act, bias=use_bias)

        self.attention_type = attention_type
        self.attn_impl = None

        if attention_type == "original_full":
            self.attn_impl = OriginalFullAttention(size_per_head=size_per_head,
                                                   attn_dropout=attention_probs_dropout_prob)
        elif attention_type == "simulated_sparse":
            self.attn_impl = BigbirdSimulatedAttention(num_attention_heads=num_attention_heads,
                                                       size_per_head=size_per_head,
                                                       num_rand_blocks=num_rand_blocks,
                                                       from_block_size=from_block_size,
                                                       to_block_size=to_block_size,
                                                       seed=seed,
                                                       attn_dropout=attention_probs_dropout_prob)
        elif attention_type == "block_sparse":
            self.attn_impl = BigbirdBlockSpareAttention(num_attention_heads=num_attention_heads,
                                                        size_per_head=size_per_head,
                                                        num_rand_blocks=num_rand_blocks,
                                                        from_block_size=from_block_size,
                                                        to_block_size=to_block_size,
                                                        seed=seed)
        else:
            raise NotImplementedError("Attention type {} is not implemented".format(attention_type))

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = x.view(batch_size, -1, self.num_attention_heads, self.size_per_head)
        return x.permute(0, 2, 1, 3)

    def forward(self,
                from_tensor,
                to_tensor,
                attention_mask=None,
                band_mask=None,
                from_mask=None,
                to_mask=None,
                from_blocked_mask=None,
                to_blocked_mask=None,
                cache=None,
                decode_i=None):

        from_shape = utils.get_shape_list(from_tensor, expected_rank=3)
        to_shape = utils.get_shape_list(to_tensor, expected_rank=3)

        if len(from_shape) != len(to_shape):
            raise ValueError(
                "The rank of `from_tensor` must match the rank of `to_tensor`.")

        batch_size = from_shape[0]
        if len(from_shape) == 3:
            from_seq_length = from_shape[1]
            to_seq_length = to_shape[1]
        else:
            raise ValueError(
                "Need rank 3 tensors to attention_layer.")

        # `query` = [b, h, m, d]
        query = self.query_layer(from_tensor)
        query = self.split_heads(query, batch_size)

        # `key` = [b, h, n, d]
        key = self.key_layer(to_tensor)
        key = self.split_heads(key, batch_size)

        # `value_layer` = [b, h, n, d]
        value = self.value_layer(to_tensor)
        value = self.split_heads(value, batch_size)

        if cache is not None and decode_i is not None:
            max_len = utils.get_shape_list(cache["k"])[2]
            indices_select = torch.view(
                F.one_hot(decode_i, max_len),
                [1, 1, max_len, 1])
            key = cache["k"] + key * indices_select
            value = cache["v"] + value * indices_select
            cache["k"] = key
            cache["v"] = value

        if self.attention_type == "original_full":
            return self.attn_impl(
                query, key, value, attention_mask
            )
        elif self.attention_type == "simulated_sparse":
            return self.attn_impl(
                query, key, value, attention_mask,
                from_seq_length, to_seq_length
            )
        elif self.attention_type == "block_sparse":
            return self.attn_impl(
                query, key, value, band_mask,
                from_mask, to_mask, from_blocked_mask,
                to_blocked_mask, batch_size, from_seq_length,
                to_seq_length
            )


class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size, filter_size, dropout_rate):
        super(FeedForwardNetwork, self).__init__()

        self.layer1 = Linear(hidden_size, filter_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.layer2 = Linear(filter_size, hidden_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.layer2(x)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, hidden_size, filter_size, nhead, dropout_rate):
        super(EncoderLayer, self).__init__()

        self.self_attention_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.self_attention = MultiHeadedAttentionLayer(hidden_size=hidden_size,
                                                        attention_type='block_sparse',
                                                        num_attention_heads=nhead,
                                                        size_per_head=int(hidden_size/nhead),
                                                        attention_probs_dropout_prob=dropout_rate)
        self.self_attention_dropout = nn.Dropout(dropout_rate)
        self.projection_layer = Linear(hidden_size, hidden_size)
        self.ffn_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.ffn = FeedForwardNetwork(hidden_size, filter_size, dropout_rate)
        self.ffn_dropout = nn.Dropout(dropout_rate)

    def forward(self,
                x,
                attention_mask=None,
                band_mask=None,
                from_mask=None,
                to_mask=None,
                input_blocked_mask=None):
        y = self.self_attention_norm(x)
        y = self.self_attention(from_tensor=y,
                                to_tensor=y,
                                attention_mask=attention_mask,
                                band_mask=band_mask,
                                from_mask=from_mask,
                                to_mask=to_mask,
                                from_blocked_mask=input_blocked_mask,
                                to_blocked_mask=input_blocked_mask)
        y = y.reshape(y.size(0), y.size(1), -1)
        y = self.projection_layer(y)
        y = self.self_attention_dropout(y)
        x = x + y

        y = self.ffn_norm(x)
        y = self.ffn(y)
        y = self.ffn_dropout(y)
        x = x + y
        return x


class DecoderLayer(nn.Module):
    def __init__(self, hidden_size, filter_size, nhead, dropout_rate):
        super(DecoderLayer, self).__init__()

        self.self_attention_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.self_attention = nn.MultiheadAttention(hidden_size, nhead, dropout_rate)
        self.self_attention_dropout = nn.Dropout(dropout_rate)

        self.enc_dec_attention_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.enc_dec_attention = nn.MultiheadAttention(hidden_size, nhead, dropout_rate)
        self.enc_dec_attention_dropout = nn.Dropout(dropout_rate)

        self.ffn_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.ffn = FeedForwardNetwork(hidden_size, filter_size, dropout_rate)
        self.ffn_dropout = nn.Dropout(dropout_rate)

    def forward(self, x, enc_output, padding_mask, attn_mask, memory_mask):
        y = self.self_attention_norm(x)
        y, self_att_weights = self.self_attention(y, y, y, attn_mask=attn_mask, key_padding_mask=padding_mask)
        y = self.self_attention_dropout(y)
        x = x + y

        y = self.enc_dec_attention_norm(x)
        y, mutihead_att_weights = self.enc_dec_attention(y, enc_output, enc_output, attn_mask=None,
                                                         key_padding_mask=memory_mask)
        y = self.enc_dec_attention_dropout(y)
        x = x + y

        y = self.ffn_norm(x)
        y = self.ffn(y)
        y = self.ffn_dropout(y)
        x = x + y
        return x, self_att_weights, mutihead_att_weights


class Encoder(nn.Module):
    def __init__(self, hidden_size, filter_size, dropout_rate, n_layers, nhead):
        super(Encoder, self).__init__()

        encoders = [EncoderLayer(hidden_size, filter_size, nhead, dropout_rate)
                    for _ in range(n_layers)]
        self.layers = nn.ModuleList(encoders)

        self.last_norm = nn.LayerNorm(hidden_size, eps=1e-6)

    def forward(self,
                inputs,
                band_mask,
                from_mask,
                to_mask,
                input_blocked_mask):
        encoder_output = inputs
        self_attentions = []
        for enc_layer in self.layers:
            encoder_output = enc_layer(x=inputs,
                                       attention_mask=None,
                                       band_mask=band_mask,
                                       from_mask=from_mask,
                                       to_mask=to_mask,
                                       input_blocked_mask=input_blocked_mask)
            # self_attentions.append(self_att_weights)
        encoder_output = encoder_output.permute([1, 0, 2])
        return self.last_norm(encoder_output), self_attentions


class Decoder(nn.Module):
    def __init__(self, hidden_size, filter_size, dropout_rate, n_layers, nhead):
        super(Decoder, self).__init__()

        decoders = [DecoderLayer(hidden_size, filter_size, nhead, dropout_rate)
                    for _ in range(n_layers)]
        self.layers = nn.ModuleList(decoders)

        self.last_norm = nn.LayerNorm(hidden_size, eps=1e-6)

    def forward(self, targets, enc_output, padding_mask, attn_mask, memory_mask):
        decoder_output = targets
        self_attentions = []
        mutihead_attentions = []
        decoder_output = decoder_output.permute([1, 0, 2])
        enc_output = enc_output.permute([1, 0, 2])
        for i, dec_layer in enumerate(self.layers):
            decoder_output, self_att_weights, mutihead_att_weights = dec_layer(decoder_output, enc_output,
                                                                               padding_mask, attn_mask, memory_mask)
            self_attentions.append(self_att_weights)
            mutihead_attentions.append(mutihead_att_weights)
        decoder_output = decoder_output.permute([1, 0, 2])
        return self.last_norm(decoder_output), self_attentions, mutihead_attentions


class Transformer(nn.Module):
    def __init__(self, encoder_layers=6, decoder_layers=6,
                 hidden_size=512, filter_size=2048,
                 encoder_nhead=8, decoder_nhead=8, dropout_rate=0.1):
        super(Transformer, self).__init__()

        self.hidden_size = hidden_size

        self.decoder = Decoder(hidden_size, filter_size,
                               dropout_rate, decoder_layers, encoder_nhead)

        self.encoder = Encoder(hidden_size, filter_size,
                               dropout_rate, encoder_layers, decoder_nhead)

    def forward(self, inputs, targets, i_padding_mask, t_padding_mask, t_attn_mask):
        batch_size = i_padding_mask.size(0)
        encoder_length = i_padding_mask.size(1)
        blocked_encoder_mask = i_padding_mask.view(batch_size, encoder_length // 16, 16)
        encoder_from_mask = i_padding_mask.view(batch_size, 1, encoder_length, 1)
        encoder_to_mask = i_padding_mask.view(batch_size, 1, 1, encoder_length)
        band_mask = utils.create_band_mask_from_inputs(blocked_encoder_mask, blocked_encoder_mask)
        enc_output, en_self_attns = self.encoder(inputs=inputs,
                                                 band_mask=band_mask,
                                                 from_mask=encoder_from_mask,
                                                 to_mask=encoder_to_mask,
                                                 input_blocked_mask=blocked_encoder_mask)
        dec_output, de_self_attns, mutihead_attns = self.decoder(targets, enc_output, t_padding_mask, t_attn_mask,
                                                                 i_padding_mask)
        return enc_output, dec_output, en_self_attns, de_self_attns, mutihead_attns


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class Speech_Transformer(nn.Module):
    def __init__(self, d_model, vocal_size,
                 nhead=4, num_encoder_layers=12,
                 num_decoder_layers=12,
                 dim_feedforward=2048,
                 conv_kenel_size=3, rate=0.1):
        super(Speech_Transformer, self).__init__()

        self.d_model = d_model

        self.trg_pad_idx = 0

        self.embed = nn.Embedding(vocal_size, d_model, padding_idx=0)
        self.embed_dropout = nn.Dropout(p=0.1)

        self.pos_encoder = PositionalEncoding(d_model, dropout=rate)
        self.pos_decoder = PositionalEncoding(d_model, dropout=rate)

        self.transfomer = Transformer(encoder_layers=num_encoder_layers,
                                      decoder_layers=num_decoder_layers,
                                      hidden_size=d_model, filter_size=dim_feedforward,
                                      encoder_nhead=nhead, decoder_nhead=nhead,
                                      dropout_rate=rate)

        self.t_encoder = Linear(d_model, vocal_size)

    def forward(self, input_, target_):
        # mask
        i_padding_mask = create_pad_mask(input_, self.trg_pad_idx)
        t_padding_mask = create_pad_mask(target_, self.trg_pad_idx)

        target_size = target_.size()[1]
        t_attn_mask = generate_square_subsequent_mask(target_size)

        # input enbedding
        input_embedded = self.embed(input_)
        input_embedded = input_embedded[:, :-1]
        input_embedded = F.pad(input_embedded, (0, 0, 1, 0))

        input_embedded *= math.sqrt(self.d_model)
        input_embedded = self.pos_decoder(input_embedded)
        input_embedded = self.embed_dropout(input_embedded)

        # target enbedding
        target_embedded = self.embed(target_)
        target_embedded = target_embedded[:, :-1]
        target_embedded = F.pad(target_embedded, (0, 0, 1, 0))

        target_embedded *= math.sqrt(self.d_model)
        target_embedded = self.pos_decoder(target_embedded)
        target_embedded = self.embed_dropout(target_embedded)

        enc_output, dec_output, en_self_attns, de_self_attns, mutihead_attns = self.transfomer(input_embedded,
                                                                                               target_embedded,
                                                                                               i_padding_mask,
                                                                                               t_padding_mask,
                                                                                               t_attn_mask)

        dec_output = self.t_encoder(dec_output)

        return dec_output, en_self_attns, de_self_attns, mutihead_attns


model = Speech_Transformer(512, 300)

input_tensor = torch.ones(32, 640).long()
target_tensor = torch.ones(32, 200).long()

dec_output, _, _, _ = model(input_tensor, target_tensor)

print('dec_output.shape ', dec_output.shape)
