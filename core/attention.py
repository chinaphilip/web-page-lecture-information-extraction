# Copyright 2021 The BigBird Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""BigBird Attention Layers."""

from absl import logging
from bigbird.core import recompute_grad
from bigbird.core import utils
import numpy as np
import tensorflow.compat.v2 as tf


MAX_SEQ_LEN = 4096







def create_band_mask_from_inputs(from_blocked_mask, to_blocked_mask):
  """Create 4D attention mask from a 3D blocked tensor mask.

  Args:
    from_blocked_mask: 3D Tensor of shape [batch_size,
      from_seq_length//from_block_size, from_block_size].
    to_blocked_mask: 3D Tensor of shape [batch_size,
      to_seq_length//to_block_size, to_block_size].

  Returns:
    float Tensor of shape [batch_size, 1, from_seq_length//from_block_size-4,
                           from_block_size,  3*to_block_size].
  """
  exp_blocked_to_pad = tf.concat(
      [to_blocked_mask[:, :-2], to_blocked_mask[:, 1:-1],to_blocked_mask[:, 2:]], 2)
  band_mask = tf.einsum(
      "BLQ,BLK->BLQK", from_blocked_mask[:, 1:-1], exp_blocked_to_pad)
  band_mask = tf.expand_dims(band_mask, 1)
  return band_mask


def create_attention_mask_from_input_mask(from_mask, to_mask):
  """Create attention mask from a 2D tensor mask.

  Args:
    from_mask: float32 Tensor of shape [batch_size, from_seq_length].
    to_mask: float32 Tensor of shape [batch_size, to_seq_length].

  Returns:
    float32 Tensor of shape [batch_size, 1, from_seq_length, to_seq_length].
  """
  mask = tf.einsum("BF,BT->BFT", from_mask, to_mask)

  # expand to create a slot for heads.
  mask = tf.expand_dims(mask, 1)

  return mask


def bigbird_block_sparse_attention(query_layer,
                                   key_layer,
                                   value_layer,
                                   band_mask,
                                   from_mask,
                                   to_mask,
                                   from_blocked_mask,
                                   to_blocked_mask,
                                   num_attention_heads,
                                   size_per_head,
                                   from_seq_length,
                                   to_seq_length,
                                   from_block_size,
                                   to_block_size):
  """BigBird attention sparse calculation using blocks in linear time.

  Assumes from_seq_length//from_block_size == to_seq_length//to_block_size.
  A pure function with a long argument list to allow easy use outside our
  framework.

  Args:
    query_layer: float Tensor of shape [batch_size, num_attention_heads,
      from_seq_length, size_per_head]
    key_layer: float Tensor of shape [batch_size, num_attention_heads,
      to_seq_length, size_per_head]
    value_layer: float Tensor of shape [batch_size, num_attention_heads,
      to_seq_length, size_per_head]
    band_mask: float32 Tensor of shape [batch_size, 1,
      from_seq_length//from_block_size-2, from_block_size, 3*to_block_size].
      The values should be 1 or 0. The attention scores will effectively be
      set to -infinity for any positions in the mask that are 0, and will be
      unchanged for positions that are 1.
    from_mask: float32 Tensor of shape [batch_size, 1, from_seq_length, 1].
      The values should be 1 or 0. The attention scores will effectively be set
      to -infinity for any positions in the mask that are 0, and will be
      unchanged for positions that are 1.
    to_mask: float32 Tensor of shape [batch_size, 1, 1, to_seq_length].
      The values should be 1 or 0. The attention scores will effectively be set
      to -infinity for any positions in the mask that are 0, and will be
      unchanged for positions that are 1.
    from_blocked_mask: float32 Tensor of shape [batch_size,
      from_seq_length//from_block_size, from_block_size].
      Same as from_mask, just reshaped.
    to_blocked_mask: float32 Tensor of shape [batch_size,
      to_seq_length//to_block_size, to_block_size].
      Same as to_mask, just reshaped.
    rand_attn: int32 Tensor of shape [num_attention_heads,
      from_seq_length//from_block_size-2, num_rand_blocks] specifying which
      blocks to attend to for each from sequence block (except 2 global ones).
    num_attention_heads: int. Number of attention heads.
    size_per_head: int. Size of each attention head.
    num_rand_blocks: int. Number of random chunks per row.
    from_seq_length: int. length of from sequence.
    to_seq_length: int. length of to sequence.
    from_block_size: int. size of block in from sequence.
    to_block_size: int. size of block in to sequence.

  Returns:
    float Tensor of shape [batch_size, from_seq_length, num_attention_heads,
      size_per_head].
  """
  assert from_seq_length//from_block_size == to_seq_length//to_block_size

  # repeat for batch size
  batch_size = utils.get_shape_list(query_layer)[0]




  # Define shorthands
  # b = batch_size
  h = num_attention_heads
  d = size_per_head
  m = from_seq_length
  n = to_seq_length
  wm = from_block_size
  wn = to_block_size

  blocked_query_matrix = tf.reshape(query_layer, (-1, h, m // wm, wm, d))#(batchsize,num_attention_heads,num_of_blocks,from_block_size,size_per_head)
  blocked_key_matrix = tf.reshape(key_layer, (-1, h, n // wn, wn, d))
  blocked_value_matrix = tf.reshape(value_layer, (-1, h, n // wn, wn, d))

#这个地方是连全局的attentiond也去掉了
  first_key_matrix=tf.reshape(blocked_key_matrix[:, :, 0:2],(-1,h,2*wn,d))
  first_value_matrix=tf.reshape(blocked_key_matrix[:, :, 0:2],(-1,h,2*wn,d))
  first_product = tf.einsum(
      "BHQD,BHKD->BHQK", blocked_query_matrix[:, :, 0],
      first_key_matrix)  # [b, h, wm, d] x [b, h, 2*wn, d] ==> [b, h, wm, 2*wn]
  first_product = tf.multiply(first_product, 1.0 / np.sqrt(d))
  first_product += (1.0 - to_mask[:,:,:,:2*wn]) * -10000.0#(b,h,wm,2*wn)-(b,1,1,2*wn)==>
  first_attn_weights = tf.nn.softmax(first_product)  # [b, h, wm, n]
  first_context_layer = tf.einsum(
      "BHQK,BHKD->BHQD", first_attn_weights,
      first_value_matrix)  # [b, h, wm, 2*wn] x [b, h, 2*wn, d] ==> [b, h, wm, d]
  first_context_layer = tf.expand_dims(first_context_layer, 2)#[b, h, m//wm-2, wm, d]


#开始计算中间层
  exp_blocked_key_matrix = tf.concat([
      blocked_key_matrix[:, :, 0:-2], blocked_key_matrix[:, :, 1:-1],
      blocked_key_matrix[:, :, 2:]], 3)  # [b, h, m//wm-2, 3*wn, -1]
  exp_blocked_value_matrix = tf.concat([
      blocked_value_matrix[:, :, 0:-2], blocked_value_matrix[:, :, 1:-1],
      blocked_value_matrix[:, :, 2:]], 3)  # [b, h, m//wm-2, 3*wn, -1]
  middle_query_matrix = blocked_query_matrix[:, :, 1:-1]
  inner_band_product = tf.einsum(
      "BHLQD,BHLKD->BHLQK", middle_query_matrix, exp_blocked_key_matrix
  )  # [b, h, m//wm-2, wm, -1] x [b, h, m//wm-2, 3*wn, -1]
  #     ==> [b, h, m//wm-2, wm, 3*wn]
  inner_band_product = tf.multiply(inner_band_product, 1.0 / np.sqrt(d))
  inner_band_product += (1.0 - band_mask) * -10000.0
  attn_weights = tf.nn.softmax(inner_band_product)  # [b, h, m//wm-2, wm, 3*wn]
  context_layer = tf.einsum(
      "BHLQK,BHLKD->BHLQD", attn_weights,
      exp_blocked_value_matrix
  )  # [b, h, m//wm-2, wm, 3*wn] x [b, h, m//wm-2, 3*wn, -1]
  #     ==> [b, h, m//wm-2, wm, d]

#去掉了全局attention
  last_key_matrix=tf.reshape(blocked_key_matrix[:, :, -2:],(-1,h,2*wn,d))
  last_value_matrix=tf.reshape(blocked_key_matrix[:, :, -2:],(-1,h,2*wn,d))
  last_product = tf.einsum(
      "BHQD,BHKD->BHQK", blocked_query_matrix[:, :, -1],
      last_key_matrix)  # [b, h, wm, -1] x [b, h, 2*wn, -1] ==> [b, h, wm, n]
  last_product = tf.multiply(last_product, 1.0 / np.sqrt(d))
  last_product += (1.0 - to_mask[:,:,:,-2*wn:]) * -10000.0
  last_attn_weights = tf.nn.softmax(last_product)  # [b, h, wm, n]
  last_context_layer = tf.einsum(
      "BHQK,BHKD->BHQD", last_attn_weights,
      last_value_matrix)  # [b, h, wm, 2*wn] x [b, h, 2*wn, d] ==> [b, h, wm, d]
  last_context_layer = tf.expand_dims(last_context_layer, 2)#[b, h, m//wm-2, wm, d]

  context_layer = tf.concat([
      first_context_layer, context_layer, last_context_layer
  ], 2)
  context_layer = tf.reshape(context_layer, (-1, h, m, d)) * from_mask#(batch_size,num_attention_heads,from_seq_length,size_per_head)
  context_layer = tf.transpose(context_layer, (0, 2, 1, 3))
  return context_layer


class MultiHeadedAttentionLayer(tf.keras.layers.Layer):
  """A multi-headed attention layer.

  It implements following types of multi-headed attention:
  - original_full attention from "Attention is all you Need".
  - simulated_sparse attention from BigBird with full quadratic implemention.
  - block_sparse attention from BigBird with memory efficient linear impl.
  """

  def __init__(self,
               attention_type,
               num_attention_heads=1,
               size_per_head=512,
               from_seq_length=1024,
               to_seq_length=1024,
               from_block_size=64,
               to_block_size=64,
               attention_probs_dropout_prob=0.0,
               initializer_range=0.02,
               use_bias=True,
               query_act=None,
               key_act=None,
               value_act=None,
               name=None):
    """Constructor for a multi-headed attention layer.

    Args:
      attention_type: Type of attention, needs to be one of ['original_full',
        'simulated_sparse', 'block_sparse'].
      num_attention_heads: (optional) int. Number of attention heads.
      size_per_head: (optional) int. Size of each attention head.
      from_seq_length: int. (optional) length of from sequence.
      to_seq_length: int. (optional) length of to sequence.
      from_block_size: (optional) int. size of block in from sequence.
      to_block_size: (optional) int. size of block in to sequence.
      attention_probs_dropout_prob: (optional) float. Dropout probability of the
        attention probabilities.
      initializer_range: (optional) float. Range of the weight initializer.
      use_bias: Whether the layer uses a bias vector.
      query_act: (optional) Activation function for the query transform.
      key_act: (optional) Activation function for the key transform.
      value_act: (optional) Activation function for the value transform.
      name: The name scope of this layer.
    """
    super(MultiHeadedAttentionLayer, self).__init__(name=name)
    self.num_attention_heads = num_attention_heads
    self.size_per_head = size_per_head
    self.from_seq_length = from_seq_length
    self.to_seq_length = to_seq_length
    self.from_block_size = from_block_size
    self.to_block_size = to_block_size

    with tf.compat.v1.variable_scope(name):
      self.query_layer = utils.Dense3dLayer(
          num_attention_heads, size_per_head,
          utils.create_initializer(initializer_range), query_act,
          "query", head_first=True, use_bias=use_bias)

      self.key_layer = utils.Dense3dLayer(
          num_attention_heads, size_per_head,
          utils.create_initializer(initializer_range), key_act,
          "key", head_first=True, use_bias=use_bias)

      self.value_layer = utils.Dense3dLayer(
          num_attention_heads, size_per_head,
          utils.create_initializer(initializer_range), value_act,
          "value", head_first=True, use_bias=use_bias)

    if attention_type == "block_sparse":
      logging.info("**** Using block sparse attention ****")
      assert from_seq_length//from_block_size == to_seq_length//to_block_size, (
          "Error the number of blocks needs to be same!")
      self.attention_dropout = None

      self.attn_impl = self.bigbird_block_sparse_attention
    else:
      raise NotImplementedError(
          "Attention type {} is not implemented".format(attention_type))




  def bigbird_block_sparse_attention(self,
                                     query_layer,
                                     key_layer,
                                     value_layer,
                                     masks,
                                     training=None):
    """BigBird attention sparse calculation using blocks in linear time.

    Args:
      query_layer: float Tensor of shape [batch_size, num_attention_heads,
        from_seq_length, size_per_head]
      key_layer: float Tensor of shape [batch_size, num_attention_heads,
        to_seq_length, size_per_head]
      value_layer: float Tensor of shape [batch_size, num_attention_heads,
        to_seq_length, size_per_head]
      masks: A list of 5 masks used in BigBird attention at position 1 to 5.
        Position 0 (first element) is not used can be left as none. In the mask,
        the values should be 1 or 0. The attention scores will effectively
        be set to -infinity for any positions in the mask that are 0,
        and will be unchanged for positions that are 1.
           "None": Not needed.
            "band_mask": (optional) float32 Tensor of shape
              [batch_size, 1, from_seq_length//from_block_size-4,
              from_block_size, 3*to_block_size].
            "from_mask": (optional) float32 Tensor of shape
              [batch_size, 1, from_seq_length, 1].
            "to_mask": (optional) float32 Tensor of shape
              [batch_size, 1, 1, to_seq_length].
            "from_blocked_mask": (optional) float32 Tensor of shape
              [batch_size, from_seq_length//from_block_size, from_block_size].
              Same as from_mask, just reshaped.
            "to_blocked_mask": (optional) float32 Tensor of shape
              [batch_size, to_seq_length//to_block_size, to_block_size].
              Same as to_mask, just reshaped.}
      training: Boolean indicating whether the call is training or inference.

    Returns:
      float Tensor of shape [batch_size, from_seq_length, num_attention_heads,
        size_per_head].
    """

    (_, band_mask, from_mask, to_mask,
     from_blocked_mask, to_blocked_mask) = masks#把attentions mask去掉了

    return bigbird_block_sparse_attention(
        query_layer, key_layer, value_layer,
        band_mask, from_mask, to_mask, from_blocked_mask, to_blocked_mask,
        self.num_attention_heads, self.size_per_head,
        self.from_seq_length, self.to_seq_length,
        self.from_block_size, self.to_block_size)

  def call(self,
           from_tensor,
           to_tensor,
           masks,
           cache=None,
           decode_i=None,
           training=None):
    """Implements a multi-headed attention layer from from_tensor to to_tensor.

    Args:
      from_tensor: float Tensor of shape [batch_size, from_seq_length,
        from_width]
      to_tensor: float Tensor of shape [batch_size, to_seq_length, to_width].
      masks: A list of masks used in different attention. Only relevant masks
        need to be supplied and at other positions place None. In the mask,
        the values should be 1 or 0. The attention scores will effectively
        be set to -infinity for any positions in the mask that are 0,
        and will be unchanged for positions that are 1.
           "attention_mask": (optional) float32 Tensor of shape
              [batch_size, from_seq_length, to_seq_length].
            "band_mask": (optional) float32 Tensor of shape
              [batch_size, 1, from_seq_length//from_block_size-4,
              from_block_size, 3*to_block_size].
            "from_mask": (optional) float32 Tensor of shape
              [batch_size, 1, from_seq_length, 1].
            "to_mask": (optional) float32 Tensor of shape
              [batch_size, 1, 1, to_seq_length].
            "from_blocked_mask": (optional) float32 Tensor of shape
              [batch_size, from_seq_length//from_block_size, from_block_size].
              Same as from_mask, just reshaped.
            "to_blocked_mask": (optional) float32 Tensor of shape
              [batch_size, to_seq_length//to_block_size, to_block_size].
              Same as to_mask, just reshaped.}
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
        query, key, value, masks, training=training)

    return contextual_output
