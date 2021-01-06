import torch
from torch import nn

from attention import original_full_attention


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
                 num_attention_heads=1,
                 num_rand_blocks=3,
                 size_per_head=512,
                 initializer_range=0.02,
                 from_block_size=64,
                 to_block_size=64,
                 attention_probs_dropout_prob=0.0,
                 use_bias=True,
                 seed=None,
                 query_act=None,
                 key_act=None,
                 value_act=None,
                 name=None,
                 **kwargs):

        super(MultiHeadedAttentionLayer, self).__init__(name=name, **kwargs)

        self.query_layer = Linear(num_attention_heads, size_per_head, query_act, bias=use_bias)

        self.key_layer = Linear(num_attention_heads, size_per_head, key_act, bias=use_bias)

        self.value_layer = Linear(num_attention_heads, size_per_head, value_act, bias=use_bias)

        def attn_impl(
                query, key, value, attention_mask,
                band_mask, from_mask, to_mask, from_blocked_mask, to_blocked_mask,
                batch_size, from_seq_length, to_seq_length, training):
            if attention_type == "original_full":
                attn_fn = original_full_attention(
                    query, key, value,
                    attention_mask, size_per_head,
                    attention_probs_dropout_prob if training else 0.0)
            elif attention_type == "simulated_sparse":
                attn_fn = bigbird_simulated_attention(
                    query, key, value,
                    attention_mask, num_attention_heads, num_rand_blocks, size_per_head,
                    from_seq_length, to_seq_length, from_block_size, to_block_size,
                    seed)
            elif attention_type == "block_sparse":
                logging.info("**** Using block sparse attention ****")
                attn_fn = bigbird_block_sparse_attention(
                    query, key, value,
                    band_mask, from_mask, to_mask, from_blocked_mask, to_blocked_mask,
                    num_attention_heads, num_rand_blocks, size_per_head, batch_size,
                    from_seq_length, to_seq_length, from_block_size, to_block_size,
                    seed)
            else:
                raise NotImplementedError(
                    "Attention type {} is not implemented".format(attention_type))
            return attn_fn

        self.attn_impl = attn_impl

    @property
    def trainable_weights(self):
        tvar_list = (self.query_layer.trainable_weights +
                     self.key_layer.trainable_weights +
                     self.value_layer.trainable_weights)
        self._trainable_weights = list({v.name: v for v in tvar_list}.values())
        return self._trainable_weights

    def call(self,
             from_tensor,
             to_tensor,
             attention_mask=None,
             band_mask=None,
             from_mask=None,
             to_mask=None,
             from_blocked_mask=None,
             to_blocked_mask=None,
             cache=None,
             decode_i=None,
             training=None):
        """Implements a multi-headed attention layer from from_tensor to to_tensor.
    Args:
      from_tensor: float Tensor of shape [batch_size, from_seq_length,
        from_width]
      to_tensor: float Tensor of shape [batch_size, to_seq_length, to_width].
      attention_mask: (optional) int32 Tensor of shape [batch_size,
        from_seq_length, to_seq_length]. The values should be 1 or 0. The
        attention scores will effectively be set to -infinity for any positions
        in the mask that are 0, and will be unchanged for positions that are 1.
      band_mask: (optional) int32 Tensor of shape [batch_size, 1,
        from_seq_length//from_block_size-4, from_block_size, 3*to_block_size].
        The values should be 1 or 0. The attention scores will effectively be
        set to -infinity for any positions in the mask that are 0, and will be
        unchanged for positions that are 1.
      from_mask: (optional) int32 Tensor of shape [batch_size, 1,
        from_seq_length, 1]. The values should be 1 or 0. The
        attention scores will effectively be set to -infinity for any positions
        in the mask that are 0, and will be unchanged for positions that are 1.
      to_mask: (optional) int32 Tensor of shape [batch_size, 1, 1,
        to_seq_length]. The values should be 1 or 0. The
        attention scores will effectively be set to -infinity for any positions
        in the mask that are 0, and will be unchanged for positions that are 1.
      from_blocked_mask: (optional) int32 Tensor of shape [batch_size,
        from_seq_length//from_block_size, from_block_size].
        Same as from_mask, just reshaped.
      to_blocked_mask: (optional) int32 Tensor of shape [batch_size,
        to_seq_length//to_block_size, to_block_size].
        Same as to_mask, just reshaped.
      cache: (Used during prediction) A dictionary with tensors containing
        results of previous attentions. The dictionary must have the items:
            {"k": tensor with shape
                  [batch_size, max_len, num_attention_heads, size_per_head],
             "v": tensor with shape
                  [batch_size, max_len, num_attention_heads, size_per_head]}
      decode_i: (Used during prediction) current location of decoding
      training: Boolean indicating whether the call is training or inference.
    Returns:
      float Tensor of shape [batch_size, from_seq_length, num_attention_heads,
        size_per_head].
    Raises:
      ValueError: Any of the arguments or tensor shapes are invalid.
      NotImplementedError: For unknown attention type.
    """
        from_shape = utils.get_shape_list(from_tensor, expected_rank=3)
        to_shape = utils.get_shape_list(to_tensor, expected_rank=3)

        if len(from_shape) != len(to_shape):
            raise ValueError(
                "The rank of `from_tensor` must match the rank of `to_tensor`.")

        if len(from_shape) == 3:
            batch_size = from_shape[0]
            from_seq_length = from_shape[1]
            to_seq_length = to_shape[1]
        else:
            raise ValueError(
                "Need rank 3 tensors to attention_layer.")

        # Scalar dimensions referenced here:
        #   b = batch size (number of sequences)
        #   m = `from_tensor` sequence length
        #   n = `to_tensor` sequence length
        #   h = `num_attention_heads`
        #   d = `size_per_head`

        # `query` = [b, h, m, d]
        query = self.query_layer(from_tensor)

        # `key` = [b, h, n, d]
        key = self.key_layer(to_tensor)

        # `value_layer` = [b, h, n, d]
        value = self.value_layer(to_tensor)

        if cache is not None and decode_i is not None:
            max_len = utils.get_shape_list(cache["k"])[2]
            indices_select = tf.reshape(
                tf.one_hot(decode_i, max_len, dtype=to_tensor.dtype),
                [1, 1, max_len, 1])
            key = cache["k"] + key * indices_select
            value = cache["v"] + value * indices_select
            cache["k"] = key
            cache["v"] = value

        contextual_output = self.attn_impl(
            query, key, value, attention_mask,
            band_mask, from_mask, to_mask, from_blocked_mask, to_blocked_mask,
            batch_size, from_seq_length, to_seq_length, training)

        return contextual_output
