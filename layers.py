import numpy as np

import torch
from torch import nn
from torch.nn import functional as F

import utils
from attention import OriginalFullAttention, BigbirdSimulatedAttention, BigbirdBlockSpareAttention


class Linear(nn.Module):
    """
    Linear Module
    """

    def __init__(self, in_dim, out_dim, activation, bias=True, w_init='linear'):
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

        print('from_seq_length ', from_seq_length)
        print('to_seq_length ', to_seq_length)

        # `query` = [b, h, m, d]
        query = self.query_layer(from_tensor)
        query = self.split_heads(query, batch_size)

        # `key` = [b, h, n, d]
        key = self.key_layer(to_tensor)
        key = self.split_heads(key, batch_size)

        # `value_layer` = [b, h, n, d]
        value = self.value_layer(to_tensor)
        value = self.split_heads(value, batch_size)

        print(query.shape, key.shape, value.shape)

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


# model = MultiHeadedAttentionLayer(attention_type='block_sparse', num_attention_heads=12)
#
# batch_size = 8
# encoder_length = 1024
# real_length = 800
#
# query_layer = torch.from_numpy(np.load('query_layer.npy'))
# key_layer = torch.from_numpy(np.load('key_layer.npy'))
# value_layer = torch.from_numpy(np.load('value_layer.npy'))
#
# encoder_inputs_mask = torch.from_numpy(np.concatenate([np.zeros((batch_size, real_length), np.float32),
#                                                        np.ones((batch_size, encoder_length - real_length), np.float32)],
#                                                       1))
#
# encoder_block_size = 16
# blocked_encoder_mask = encoder_inputs_mask.view(batch_size, encoder_length // encoder_block_size, encoder_block_size)
# encoder_from_mask = encoder_inputs_mask.view(batch_size, 1, encoder_length, 1)
# encoder_to_mask = encoder_inputs_mask.view(batch_size, 1, 1, encoder_length)
#
# # create band padding
# attention_mask = None
# band_mask = utils.create_band_mask_from_inputs(blocked_encoder_mask, blocked_encoder_mask)
#
# attn_impl = BigbirdBlockSpareAttention(num_attention_heads=12,
#                                        size_per_head=512,
#                                        num_rand_blocks=3,
#                                        from_block_size=16,
#                                        to_block_size=16,
#                                        seed=None)
#
# output_tensor = attn_impl(
#     query_layer=query_layer, key_layer=key_layer, value_layer=value_layer, band_mask=band_mask,
#     from_mask=encoder_from_mask, to_mask=encoder_to_mask, from_blocked_mask=blocked_encoder_mask,
#     to_blocked_mask=blocked_encoder_mask, batch_size=batch_size, from_seq_length=1024, to_seq_length=1024
# )
#
# result = np.load('result.npy')
# result_torch = output_tensor.detach().numpy()
#
# print(np.sum(result - result_torch))
