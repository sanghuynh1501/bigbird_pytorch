import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

MAX_SEQ_LEN = 4096


def get_single_block_row_attention(block_id,
                                   to_start_block_id,
                                   to_end_block_id,
                                   num_rand_blocks,
                                   window_block_left=1,
                                   window_block_right=1,
                                   global_block_left=1,
                                   global_block_right=1):
    """
    For a single row block get random row attention.
    Args:
        block_id: int. block id of row.
        to_start_block_id: int. random attention coloum start id.
        to_end_block_id: int. random attention coloum end id.
        num_rand_blocks: int. number of random blocks to be selected.
        window_block_left: int. number of blocks of window to left of a block.
        window_block_right: int. number of blocks of window to right of a block.
        global_block_left: int. Number of blocks globally used to the left.
        global_block_right: int. Number of blocks globally used to the right.
    Returns:
        row containing the random attention vector of size num_rand_blocks.
    """

    # list of to_blocks from which to choose random attention
    to_block_list = np.arange(to_start_block_id, to_end_block_id,
                              dtype=np.int32)
    # permute the blocks
    perm_block = np.random.permutation(to_block_list)
    # print(perm_block)

    # illegal blocks for the current block id, using window
    illegal_blocks = list(
        range(block_id - window_block_left, block_id + window_block_right + 1))

    # Add blocks at the start and at the end
    illegal_blocks.extend(list(range(global_block_left)))
    illegal_blocks.extend(
        list(range(to_end_block_id - global_block_right, to_end_block_id)))

    # The second from_block cannot choose random attention on second last to_block
    if block_id == 1:
        illegal_blocks.append(to_end_block_id - 2)

    # The second last from_block cannot choose random attention on second to_block
    if block_id == to_end_block_id - 2:
        illegal_blocks.append(1)

    selected_random_blokcs = []

    for i in range(to_end_block_id - to_start_block_id):
        if perm_block[i] not in illegal_blocks:
            selected_random_blokcs.append(perm_block[i])
        if len(selected_random_blokcs) == num_rand_blocks:
            break
    return np.array(selected_random_blokcs, dtype=np.int32)


def bigbird_block_rand_mask_with_head(from_seq_length,
                                      to_seq_length,
                                      from_block_size,
                                      to_block_size,
                                      num_heads,
                                      plan_from_length,
                                      plan_num_rand_blocks,
                                      window_block_left=1,
                                      window_block_right=1,
                                      global_block_top=1,
                                      global_block_bottom=1,
                                      global_block_left=1,
                                      global_block_right=1):
    """Create adjacency list of random attention.
  Args:
    from_seq_length: int. length of from sequence.
    to_seq_length: int. length of to sequence.
    from_block_size: int. size of block in from sequence.
    to_block_size: int. size of block in to sequence.
    num_heads: int. total number of heads.
    plan_from_length: list. plan from lenght where num_rand are choosen from.
    plan_num_rand_blocks: list. number of rand blocks within the plan.
    window_block_left: int. number of blocks of window to left of a block.
    window_block_right: int. number of blocks of window to right of a block.
    global_block_top: int. number of blocks at the top.
    global_block_bottom: int. number of blocks at the bottom.
    global_block_left: int. Number of blocks globally used to the left.
    global_block_right: int. Number of blocks globally used to the right.
  Returns:
    adjacency list of size num_head where each element is of size
    from_seq_length//from_block_size-2 by num_rand_blocks
  """
    assert from_seq_length // from_block_size == to_seq_length // to_block_size, \
        "Error the number of blocks needs to be same!"

    assert from_seq_length in plan_from_length, \
        "Error from sequence length not in plan!"

    # Total number of blocks in the mmask
    num_blocks = from_seq_length // from_block_size
    # Number of blocks per plan
    plan_block_length = np.array(plan_from_length) // from_block_size
    # till when to follow plan
    max_plan_idx = plan_from_length.index(from_seq_length)
    # Random Attention adjajency list
    rand_attn = [np.zeros((num_blocks,
                           np.sum(plan_num_rand_blocks[:max_plan_idx + 1])),
                          dtype=np.int32) for i in range(num_heads)]

    # We will go iteratively over the plan blocks and pick random number of
    # Attention blocks from the legally allowed blocks
    for plan_idx in range(max_plan_idx + 1):
        rnd_r_cnt = 0
        if plan_idx > 0:
            # set the row for all from_blocks starting from 0 to
            # plan_block_length[plan_idx-1]
            # column indx start fromm plan_block_length[plan_idx-1] and ends at
            # plan_block_length[plan_idx]
            if plan_num_rand_blocks[plan_idx] > 0:
                rnd_r_cnt = int(np.sum(plan_num_rand_blocks[:plan_idx]))
                curr_r_cnt = int(np.sum(plan_num_rand_blocks[:plan_idx + 1]))
                for blk_rw_idx in range(global_block_top,
                                        plan_block_length[plan_idx - 1]):
                    for h in range(num_heads):
                        # print("head", h, "blk_rw_idx", blk_rw_idx)
                        rand_attn[h][blk_rw_idx,
                        rnd_r_cnt:curr_r_cnt] = get_single_block_row_attention(
                            block_id=blk_rw_idx,
                            to_start_block_id=plan_block_length[plan_idx - 1],
                            to_end_block_id=plan_block_length[plan_idx],
                            num_rand_blocks=plan_num_rand_blocks[plan_idx],
                            window_block_left=window_block_left,
                            window_block_right=window_block_right,
                            global_block_left=global_block_left,
                            global_block_right=global_block_right)

            for pl_id in range(plan_idx):
                if plan_num_rand_blocks[pl_id] == 0:
                    continue
                for blk_rw_idx in range(plan_block_length[plan_idx - 1],
                                        plan_block_length[plan_idx]):
                    rnd_r_cnt = 0
                    to_start_block_id = 0
                    if pl_id > 0:
                        rnd_r_cnt = int(np.sum(plan_num_rand_blocks[:pl_id]))
                        to_start_block_id = plan_block_length[pl_id - 1]
                    curr_r_cnt = int(np.sum(plan_num_rand_blocks[:pl_id + 1]))
                    for h in range(num_heads):
                        # print("head", h, "blk_rw_idx", blk_rw_idx)
                        rand_attn[h][blk_rw_idx,
                        rnd_r_cnt:curr_r_cnt] = get_single_block_row_attention(
                            block_id=blk_rw_idx,
                            to_start_block_id=to_start_block_id,
                            to_end_block_id=plan_block_length[pl_id],
                            num_rand_blocks=plan_num_rand_blocks[pl_id],
                            window_block_left=window_block_left,
                            window_block_right=window_block_right,
                            global_block_left=global_block_left,
                            global_block_right=global_block_right)

        if plan_num_rand_blocks[plan_idx] == 0:
            continue
        # print("Start from here")
        curr_r_cnt = int(np.sum(plan_num_rand_blocks[:plan_idx + 1]))
        from_start_block_id = global_block_top
        to_start_block_id = 0
        if plan_idx > 0:
            rnd_r_cnt = int(np.sum(plan_num_rand_blocks[:plan_idx]))
            from_start_block_id = plan_block_length[plan_idx - 1]
            to_start_block_id = plan_block_length[plan_idx - 1]

        for blk_rw_idx in range(from_start_block_id, plan_block_length[plan_idx]):
            for h in range(num_heads):
                # print("head", h, "blk_rw_idx", blk_rw_idx)
                rand_attn[h][blk_rw_idx,
                rnd_r_cnt:curr_r_cnt] = get_single_block_row_attention(
                    block_id=blk_rw_idx,
                    to_start_block_id=to_start_block_id,
                    to_end_block_id=plan_block_length[plan_idx],
                    num_rand_blocks=plan_num_rand_blocks[plan_idx],
                    window_block_left=window_block_left,
                    window_block_right=window_block_right,
                    global_block_left=global_block_left,
                    global_block_right=global_block_right)

    for nh in range(num_heads):
        rand_attn[nh] = rand_attn[nh][global_block_top:num_blocks -
                                                       global_block_bottom, :]
    return rand_attn


def get_rand_attn_plan(from_seq_length, from_block_size, num_rand_blocks):
    """Gives the plan of where to put random attention.
    Args:
        from_seq_length: int. length of from sequence.
        from_block_size: int. size of block in from sequence.
        num_rand_blocks: int. Number of random chunks per row.
        Returns:
        plan_from_length: ending location of from block
        plan_num_rand_blocks: number of random ending location for each block
    """
    # general plan
    plan_from_length = []
    plan_num_rand_blocks = []
    if (2 * num_rand_blocks + 5) < (from_seq_length // from_block_size):
        plan_from_length.append(int((2 * num_rand_blocks + 5) * from_block_size))
        plan_num_rand_blocks.append(num_rand_blocks)
        plan_from_length.append(from_seq_length)
        plan_num_rand_blocks.append(0)
    elif (num_rand_blocks + 5) < (from_seq_length // from_block_size):
        plan_from_length.append(int((num_rand_blocks + 5) * from_block_size))
        plan_num_rand_blocks.append(num_rand_blocks // 2)
        plan_from_length.append(from_seq_length)
        plan_num_rand_blocks.append(num_rand_blocks - (num_rand_blocks // 2))
    else:
        plan_from_length.append(from_seq_length)
        plan_num_rand_blocks.append(num_rand_blocks)

    return plan_from_length, plan_num_rand_blocks


def bigbird_block_rand_mask(from_seq_length,
                            to_seq_length,
                            from_block_size,
                            to_block_size,
                            num_rand_blocks,
                            last_idx=-1):
    """Create adjacency list of random attention.
  Args:
    from_seq_length: int. length of from sequence.
    to_seq_length: int. length of to sequence.
    from_block_size: int. size of block in from sequence.
    to_block_size: int. size of block in to sequence.
    num_rand_blocks: int. Number of random chunks per row.
    last_idx: if -1 then num_rand_blocks blocks chosen anywhere in to sequence,
      if positive then num_rand_blocks blocks choosen only upto last_idx.
  Returns:
    adjacency list of size from_seq_length//from_block_size-2 by num_rand_blocks
  """
    assert from_seq_length // from_block_size == to_seq_length // to_block_size, \
        "Error the number of blocks needs to be same!"

    rand_attn = np.zeros(
        (from_seq_length // from_block_size - 2, num_rand_blocks), dtype=np.int32)
    middle_seq = np.arange(1, to_seq_length // to_block_size - 1, dtype=np.int32)
    last = to_seq_length // to_block_size - 1
    if last_idx > (2 * to_block_size):
        last = (last_idx // to_block_size) - 1

    r = num_rand_blocks  # shorthand
    for i in range(1, from_seq_length // from_block_size - 1):
        start = i - 2
        end = i
        if i == 1:
            rand_attn[i - 1, :] = np.random.permutation(middle_seq[2:last])[:r]
        elif i == 2:
            rand_attn[i - 1, :] = np.random.permutation(middle_seq[3:last])[:r]
        elif i == from_seq_length // from_block_size - 3:
            rand_attn[i - 1, :] = np.random.permutation(middle_seq[:last])[:r]
            # Missing -3: should have been sliced till last-3
        elif i == from_seq_length // from_block_size - 2:
            rand_attn[i - 1, :] = np.random.permutation(middle_seq[:last])[:r]
            # Missing -4: should have been sliced till last-4
        else:
            if start > last:
                start = last
                rand_attn[i - 1, :] = np.random.permutation(middle_seq[:start])[:r]
            elif (end + 1) == last:
                rand_attn[i - 1, :] = np.random.permutation(middle_seq[:start])[:r]
            else:
                rand_attn[i - 1, :] = np.random.permutation(
                    np.concatenate((middle_seq[:start], middle_seq[end + 1:last])))[:r]
    return rand_attn


def full_bigbird_mask(from_seq_length,
                      to_seq_length,
                      from_block_size,
                      to_block_size,
                      num_rand_blocks,
                      rand_attn=None,
                      focus=1024):
    """Calculate BigBird attention pattern as a full dense matrix.
  Args:
    from_seq_length: int. length of from sequence.
    to_seq_length: int. length of to sequence.
    from_block_size: int. size of block in from sequence.
    to_block_size: int. size of block in to sequence.
    num_rand_blocks: int. Number of random chunks per row.
    rand_attn: adjajency matrix for random attention.
    focus: pick random mask within focus
  Returns:
    attention mask matrix of shape [from_seq_length, to_seq_length]
  """
    if rand_attn is None:
        rand_attn = bigbird_block_rand_mask(MAX_SEQ_LEN, MAX_SEQ_LEN,
                                            from_block_size, to_block_size,
                                            num_rand_blocks, focus)

    attn_mask = np.zeros((MAX_SEQ_LEN, MAX_SEQ_LEN), dtype=np.int32)
    for i in range(1, (MAX_SEQ_LEN // from_block_size) - 1):
        attn_mask[(i) * from_block_size:(i + 1) * from_block_size,
        (i - 1) * to_block_size:(i + 2) * to_block_size] = 1
        for j in rand_attn[i - 1, :]:
            attn_mask[i * from_block_size:(i + 1) * from_block_size,
            j * to_block_size:(j + 1) * to_block_size] = 1

    attn_mask[:from_block_size, :] = 1
    attn_mask[:, :to_block_size] = 1
    attn_mask[:, -to_block_size:] = 1
    attn_mask[-from_block_size:, :] = 1
    clipped_attn_mask = attn_mask[:from_seq_length, :to_seq_length]
    return np.array(clipped_attn_mask, dtype=bool)


def create_rand_mask_from_inputs(from_blocked_mask,
                                 to_blocked_mask,
                                 rand_attn,
                                 num_attention_heads,
                                 num_rand_blocks,
                                 batch_size,
                                 from_seq_length,
                                 from_block_size):
    """Create 3D attention mask from a 2D tensor mask.
  Args:
    from_blocked_mask: 2D Tensor of shape [batch_size,
      from_seq_length//from_block_size, from_block_size].
    to_blocked_mask: int32 Tensor of shape [batch_size,
      to_seq_length//to_block_size, to_block_size].
    rand_attn: [batch_size, num_attention_heads,
      from_seq_length//from_block_size-2, num_rand_blocks]
    num_attention_heads: int. Number of attention heads.
    num_rand_blocks: int. Number of random chunks per row.
    batch_size: int. Batch size for computation.
    from_seq_length: int. length of from sequence.
    from_block_size: int. size of block in from sequence.
  Returns:
    float Tensor of shape [batch_size, num_attention_heads,
                           from_seq_length//from_block_size-2,
                           from_block_size, num_rand_blocks*to_block_size].
  """
    num_windows = from_seq_length // from_block_size - 2
    rand_mask = torch.view(
        torch.gather(to_blocked_mask, 1, rand_attn), [
            batch_size, num_attention_heads, num_windows,
            num_rand_blocks * from_block_size
        ])
    rand_mask = torch.einsum("blq,bhlk->bhlqk", from_blocked_mask[:, 1:-1],
                             rand_mask)
    return rand_mask


def create_band_mask_from_inputs(from_blocked_mask, to_blocked_mask):
    """Create 3D attention mask from a 2D tensor mask.
  Args:
    from_blocked_mask: 2D Tensor of shape [batch_size,
      from_seq_length//from_block_size, from_block_size].
    to_blocked_mask: int32 Tensor of shape [batch_size,
      to_seq_length//to_block_size, to_block_size].
  Returns:
    float Tensor of shape [batch_size, 1, from_seq_length//from_block_size-4,
                           from_block_size,  3*to_block_size].
  """
    exp_blocked_to_pad = torch.cat(
        (to_blocked_mask[:, 1:-3], to_blocked_mask[:, 2:-2],
         to_blocked_mask[:, 3:-1]), 2)
    band_mask = torch.einsum("blq,blk->blqk",
                             from_blocked_mask[:, 2:-2].float(),
                             exp_blocked_to_pad.float32())
    band_mask = torch.unsqueeze(band_mask, 1)
    return band_mask


def create_attention_mask_from_input_mask(from_mask, to_mask):
    """Create attention mask from a 2D tensor mask.
  Args:
    from_mask: int32 Tensor of shape [batch_size, from_seq_length].
    to_mask: int32 Tensor of shape [batch_size, to_seq_length].
  Returns:
    int32 Tensor of shape [batch_size, 1, from_seq_length, to_seq_length].
  """
    mask = torch.einsum("BF,BT->BFT", from_mask, to_mask)

    # expand to create a slot for heads.
    mask = torch.unsqueeze(mask, 1)

    return mask


class OriginalFullAttention(nn.Module):
    def __init__(self, attn_dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, query_layer,
                key_layer,
                value_layer,
                attention_mask,
                size_per_head):
        # Directly take n^2 dot product between "query" and "key".
        attention_scores = torch.einsum("bnfh,bnth->bnft", query_layer, key_layer)
        attention_scores = torch.multiply(attention_scores,
                                          1.0 / np.sqrt(float(size_per_head)))

        if attention_mask is not None:
            # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
            # masked positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and -10000.0 for masked positions.
            adder = (1.0 - attention_mask.float()) * -10000.0

            # Since we are adding it to the raw scores before the softmax, this is
            # effectively the same as removing these entirely.
            attention_scores += adder

        # Normalize the attention scores to probabilities.
        # `attention_probs` = [B, N, F, T]
        attention_probs = F.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # `context_layer` = [B, F, N, H]
        context_layer = torch.einsum("bnft,bnth->bfnh", attention_probs, value_layer)
        return context_layer


class BigbirdSimulatedAttention(nn.Module):
    def __init__(self, attn_dropout=0.1):
        super().__init__()
        self.original_full_attention = OriginalFullAttention(attn_dropout)

    def forward(self, query_layer,
                key_layer,
                value_layer,
                attention_mask,
                num_attention_heads,
                num_rand_blocks,
                size_per_head,
                from_seq_length,
                to_seq_length,
                from_block_size,
                to_block_size,
                seed=None):
        if seed:
            np.random.seed(seed)

        plan_from_length, plan_num_rand_blocks = get_rand_attn_plan(
            from_seq_length, from_block_size, num_rand_blocks)

        rand_attn = bigbird_block_rand_mask_with_head(
            from_seq_length=from_seq_length,
            to_seq_length=to_seq_length,
            from_block_size=from_block_size,
            to_block_size=to_block_size,
            num_heads=num_attention_heads,
            plan_from_length=plan_from_length,
            plan_num_rand_blocks=plan_num_rand_blocks)
        temp_mask = [
            full_bigbird_mask(  # pylint: disable=g-complex-comprehension
                from_seq_length, to_seq_length, from_block_size, to_block_size,
                num_rand_blocks, rand_attn=rand_attn[i], focus=1024)
            for i in range(num_attention_heads)
        ]
        temp_mask = np.stack(temp_mask, axis=0)
        temp_mask = np.array(temp_mask, dtype=bool)

        rand_block_mask = torch.from_numpy(temp_mask).long()
        rand_block_mask = torch.unsqueeze(rand_block_mask, 0)  # [1, N, F, T]
        if attention_mask is not None:
            attention_mask = torch.minimum(attention_mask, rand_block_mask)
        else:
            attention_mask = rand_block_mask
        return self.original_full_attention(query_layer,
                                            key_layer,
                                            value_layer,
                                            attention_mask,
                                            size_per_head)

class BigbirdBlockSpareAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, query_layer,
                key_layer,
                value_layer,
                band_mask,
                from_mask,
                to_mask,
                from_blocked_mask,
                to_blocked_mask,
                num_attention_heads,
                num_rand_blocks,
                size_per_head,
                batch_size,
                from_seq_length,
                to_seq_length,
                from_block_size,
                to_block_size,
                seed=None,
                plan_from_length=None,
                plan_num_rand_blocks=None):

        assert from_seq_length // from_block_size == to_seq_length // to_block_size

        # cast masks to float
        from_mask = from_mask.float()
        to_mask = to_mask.float()
        band_mask = band_mask.float()
        from_blocked_mask = from_blocked_mask.float()
        to_blocked_mask = to_blocked_mask.float()

        # generate random attention and corresponding masks
        np.random.seed(seed)
        if from_seq_length in [1024, 3072, 4096]:  # old plans used in paper
            rand_attn = [
                bigbird_block_rand_mask(  # pylint: disable=g-complex-comprehension
                    MAX_SEQ_LEN, MAX_SEQ_LEN,
                    from_block_size, to_block_size, num_rand_blocks,
                    last_idx=1024)[:(from_seq_length // from_block_size - 2)]
                for _ in range(num_attention_heads)
            ]
        else:
            if plan_from_length is None:
                plan_from_length, plan_num_rand_blocks = get_rand_attn_plan(
                    from_seq_length, from_block_size, num_rand_blocks)

            rand_attn = bigbird_block_rand_mask_with_head(
                from_seq_length=from_seq_length,
                to_seq_length=to_seq_length,
                from_block_size=from_block_size,
                to_block_size=to_block_size,
                num_heads=num_attention_heads,
                plan_from_length=plan_from_length,
                plan_num_rand_blocks=plan_num_rand_blocks)

        rand_attn = np.stack(rand_attn, axis=0)
        rand_attn = torch.from_numpy(rand_attn).long()
        rand_attn = torch.unsqueeze(rand_attn, 0)
        rand_attn = tf.repeat(rand_attn, batch_size, 0)

        rand_mask = create_rand_mask_from_inputs(
            from_blocked_mask, to_blocked_mask, rand_attn,
            num_attention_heads, num_rand_blocks,
            batch_size, from_seq_length, from_block_size, )

        # Define shorthands
        h = num_attention_heads
        r = num_rand_blocks
        d = size_per_head
        b = batch_size
        m = from_seq_length
        n = to_seq_length
        wm = from_block_size
        wn = to_block_size

        blocked_query_matrix = tf.reshape(query_layer, (b, h, m // wm, wm, -1))
        blocked_key_matrix = tf.reshape(key_layer, (b, h, n // wn, wn, -1))
        blocked_value_matrix = tf.reshape(value_layer, (b, h, n // wn, wn, -1))
        gathered_key = tf.reshape(
            tf.gather(blocked_key_matrix, rand_attn, batch_dims=2, name="gather_key"),
            (b, h, m // wm - 2, r * wn, -1))  # [b, h, n//wn-2, r, wn, -1]
        gathered_value = tf.reshape(
            tf.gather(
                blocked_value_matrix, rand_attn, batch_dims=2, name="gather_value"),
            (b, h, m // wm - 2, r * wn, -1))  # [b, h, n//wn-2, r, wn, -1]

        first_product = tf.einsum(
            "BHQD,BHKD->BHQK", blocked_query_matrix[:, :, 0],
            key_layer)  # [b, h, wm, -1] x [b, h, n, -1] ==> [b, h, wm, n]
        first_product = tf.multiply(first_product, 1.0 / np.sqrt(d))
        first_product += (1.0 - to_mask) * -10000.0
        first_attn_weights = tf.nn.softmax(first_product)  # [b, h, wm, n]
        first_context_layer = tf.einsum(
            "BHQK,BHKD->BHQD", first_attn_weights,
            value_layer)  # [b, h, wm, n] x [b, h, n, -1] ==> [b, h, wm, -1]
        first_context_layer = tf.expand_dims(first_context_layer, 2)

        second_key_mat = tf.concat([
            blocked_key_matrix[:, :, 0], blocked_key_matrix[:, :, 1],
            blocked_key_matrix[:, :, 2], blocked_key_matrix[:, :, -1],
            gathered_key[:, :, 0]], 2)  # [b, h, (4+r)*wn, -1]
        second_value_mat = tf.concat([
            blocked_value_matrix[:, :, 0], blocked_value_matrix[:, :, 1],
            blocked_value_matrix[:, :, 2], blocked_value_matrix[:, :, -1],
            gathered_value[:, :, 0]], 2)  # [b, h, (4+r)*wn, -1]
        second_product = tf.einsum(
            "BHQD,BHKD->BHQK", blocked_query_matrix[:, :, 1], second_key_mat
        )  # [b, h, wm, -1] x [b, h, (4+r)*wn, -1] ==> [b, h, wm, (4+r)*wn]
        second_seq_pad = tf.concat([
            to_mask[:, :, :, :3 * wn], to_mask[:, :, :, -wn:],
            tf.ones([b, 1, 1, r * wn], dtype=tf.float32)], 3)
        second_rand_pad = tf.concat(
            [tf.ones([b, h, wm, 4 * wn], dtype=tf.float32), rand_mask[:, :, 0]], 3)
        second_product = tf.multiply(second_product, 1.0 / np.sqrt(d))
        second_product += (1.0 -
                           tf.minimum(second_seq_pad, second_rand_pad)) * -10000.0
        second_attn_weights = tf.nn.softmax(second_product)  # [b , h, wm, (4+r)*wn]
        second_context_layer = tf.einsum(
            "BHQK,BHKD->BHQD", second_attn_weights, second_value_mat
        )  # [b, h, wm, (4+r)*wn] x [b, h, (4+r)*wn, -1] ==> [b, h, wm, -1]
        second_context_layer = tf.expand_dims(second_context_layer, 2)

        exp_blocked_key_matrix = tf.concat([
            blocked_key_matrix[:, :, 1:-3], blocked_key_matrix[:, :, 2:-2],
            blocked_key_matrix[:, :, 3:-1]], 3)  # [b, h, m//wm-4, 3*wn, -1]
        exp_blocked_value_matrix = tf.concat([
            blocked_value_matrix[:, :, 1:-3], blocked_value_matrix[:, :, 2:-2],
            blocked_value_matrix[:, :, 3:-1]], 3)  # [b, h, m//wm-4, 3*wn, -1]
        middle_query_matrix = blocked_query_matrix[:, :, 2:-2]
        inner_band_product = tf.einsum(
            "BHLQD,BHLKD->BHLQK", middle_query_matrix, exp_blocked_key_matrix
        )  # [b, h, m//wm-4, wm, -1] x [b, h, m//wm-4, 3*wn, -1]
        #     ==> [b, h, m//wm-4, wm, 3*wn]
        inner_band_product = tf.multiply(inner_band_product, 1.0 / np.sqrt(d))
        rand_band_product = tf.einsum(
            "BHLQD,BHLKD->BHLQK", middle_query_matrix, gathered_key[:, :, 1:-1]
        )  # [b, h, m//wm-4, wm, -1] x [b, h, m//wm-4, r*wn, -1]
        #     ==> [b, h, m//wm-4, wm, r*wn]
        rand_band_product = tf.multiply(rand_band_product, 1.0 / np.sqrt(d))
        first_band_product = tf.einsum(
            "BHLQD,BHKD->BHLQK", middle_query_matrix, blocked_key_matrix[:, :, 0]
        )  # [b, h, m//wm-4, wm, -1] x [b, h, wn, -1] ==> [b, h, m//wm-4, wm, wn]
        first_band_product = tf.multiply(first_band_product, 1.0 / np.sqrt(d))
        last_band_product = tf.einsum(
            "BHLQD,BHKD->BHLQK", middle_query_matrix, blocked_key_matrix[:, :, -1]
        )  # [b, h, m//wm-4, wm, -1] x [b, h, wn, -1] ==> [b, h, m//wm-4, wm, wn]
        last_band_product = tf.multiply(last_band_product, 1.0 / np.sqrt(d))
        inner_band_product += (1.0 - band_mask) * -10000.0
        first_band_product += (
                                      1.0 - tf.expand_dims(to_mask[:, :, :, :wn], 3)) * -10000.0
        last_band_product += (
                                     1.0 - tf.expand_dims(to_mask[:, :, :, -wn:], 3)) * -10000.0
        rand_band_product += (1.0 - rand_mask[:, :, 1:-1]) * -10000.0
        band_product = tf.concat([
            first_band_product, inner_band_product, rand_band_product,
            last_band_product], -1)  # [b, h, m//wm-4, wm, (5+r)*wn]
        attn_weights = tf.nn.softmax(band_product)  # [b, h, m//wm-4, wm, (5+r)*wn]
        context_layer = tf.einsum(
            "BHLQK,BHLKD->BHLQD", attn_weights[:, :, :, :, wn:4 * wn],
            exp_blocked_value_matrix
        )  # [b, h, m//wm-4, wm, 3*wn] x [b, h, m//wm-4, 3*wn, -1]
        #     ==> [b, h, m//wm-4, wm, -1]
        context_layer += tf.einsum(
            "BHLQK,BHLKD->BHLQD", attn_weights[:, :, :, :, 4 * wn:-wn],
            gathered_value[:, :, 1:-1]
        )  # [b, h, m//wm-4, wm, r*wn] x [b, h, m//wm-4, r*wn, -1]
        #     ==> [b, h, m//wm-4, wm, -1]
        context_layer += tf.einsum(
            "BHLQK,BHKD->BHLQD", attn_weights[:, :, :, :, :wn],
            blocked_value_matrix[:, :, 0]
        )  # [b, h, m//wm-4, wm, wn] x [b, h, wn, -1] ==> [b, h, m//wm-4, wm, -1]
        context_layer += tf.einsum(
            "BHLQK,BHKD->BHLQD", attn_weights[:, :, :, :, -wn:],
            blocked_value_matrix[:, :, -1]
        )  # [b, h, m//wm-4, wm, wn] x [b, h, wn, -1] ==> [b, h, m//wm-4, wm, -1]

        second_last_key_mat = tf.concat([
            blocked_key_matrix[:, :, 0], blocked_key_matrix[:, :, -3],
            blocked_key_matrix[:, :, -2], blocked_key_matrix[:, :, -1],
            gathered_key[:, :, -1]], 2)  # [b, h, (4+r)*wn, -1]
        second_last_value_mat = tf.concat([
            blocked_value_matrix[:, :, 0], blocked_value_matrix[:, :, -3],
            blocked_value_matrix[:, :, -2], blocked_value_matrix[:, :, -1],
            gathered_value[:, :, -1]], 2)  # [b, h, (4+r)*wn, -1]
        second_last_product = tf.einsum(
            "BHQD,BHKD->BHQK", blocked_query_matrix[:, :, -2], second_last_key_mat
        )  # [b, h, wm, -1] x [b, h, (4+r)*wn, -1] ==> [b, h, wm, (4+r)*wn]
        second_last_seq_pad = tf.concat([
            to_mask[:, :, :, :wn], to_mask[:, :, :, -3 * wn:],
            tf.ones([b, 1, 1, r * wn], dtype=tf.float32)], 3)
        second_last_rand_pad = tf.concat(
            [tf.ones([b, h, wm, 4 * wn], dtype=tf.float32), rand_mask[:, :, -1]], 3)
        second_last_product = tf.multiply(second_last_product, 1.0 / np.sqrt(d))
        second_last_product += (
                                       1.0 - tf.minimum(second_last_seq_pad, second_last_rand_pad)) * -10000.0
        second_last_attn_weights = tf.nn.softmax(
            second_last_product)  # [b, h, wm, (4+r)*wn]
        second_last_context_layer = tf.einsum(
            "BHQK,BHKD->BHQD", second_last_attn_weights, second_last_value_mat
        )  # [b, h, wm, (4+r)*wn] x [b, h, (4+r)*wn, -1] ==> [b, h, wm, -1]
        second_last_context_layer = tf.expand_dims(second_last_context_layer, 2)

        last_product = tf.einsum(
            "BHQD,BHKD->BHQK", blocked_query_matrix[:, :, -1],
            key_layer)  # [b, h, wm, -1] x [b, h, n, -1] ==> [b, h, wm, n]
        last_product = tf.multiply(last_product, 1.0 / np.sqrt(d))
        last_product += (1.0 - to_mask) * -10000.0
        last_attn_weights = tf.nn.softmax(last_product)  # [b, h, wm, n]
        last_context_layer = tf.einsum(
            "BHQK,BHKD->BHQD", last_attn_weights,
            value_layer)  # [b, h, wm, n] x [b, h, n, -1] ==> [b, h, wm, -1]
        last_context_layer = tf.expand_dims(last_context_layer, 2)

        context_layer = tf.concat([
            first_context_layer, second_context_layer, context_layer,
            second_last_context_layer, last_context_layer
        ], 2)
        context_layer = tf.reshape(context_layer, (b, h, m, -1)) * from_mask
        context_layer = tf.transpose(context_layer, (0, 2, 1, 3))
        return context_layer
