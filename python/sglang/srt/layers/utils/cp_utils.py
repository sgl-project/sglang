# temp NSA debugging environ
from dataclasses import dataclass
from itertools import accumulate
from typing import TYPE_CHECKING, List

import torch
import torch.nn.functional as F

from sglang.srt.layers.dp_attention import get_attention_cp_group
from sglang.srt.server_args import get_global_server_args

if TYPE_CHECKING:
    pass


@dataclass
class ContextParallelMetadata:
    split_list: List[int] = None
    max_rank_len: List[int] = None
    zigzag_index: List[int] = None
    per_rank_actual_token: List[int] = None
    reverse_split_len: List[int] = None
    cp_reverse_index: List[int] = None

    # metadata for attention
    kv_len_prev: int = -1
    kv_len_next: int = -1
    actual_seq_q_prev: int = -1
    actual_seq_q_next: int = -1
    kv_len_prev_tensor: torch.Tensor = None
    kv_len_next_tensor: torch.Tensor = None
    actual_seq_q_prev_tensor: torch.Tensor = None
    actual_seq_q_next_tensor: torch.Tensor = None

    total_seq_lens: torch.Tensor = None


def is_prefill_context_parallel_enabled():
    return get_global_server_args().enable_prefill_context_parallel


def is_prefill_cp_in_seq_split():
    return (
        is_prefill_context_parallel_enabled()
        and get_global_server_args().prefill_cp_mode == "in-seq-split"
    )


def can_cp_split(seq_len: int, cp_size: int, forward_batch):
    # TODO current just support prefill batch=1 and len(input_ids) > self.cp_size * 2
    # Note: (self.cp_size * 2) To achieve load balancing for seq computation,
    # the seq data needs to be divided and recombined at twice the size of cp_size.
    cur_cp_seq_len = seq_len // (cp_size * 2)
    # print("DEBUG: can_cp_split", cur_cp_seq_len, cp_size, forward_batch.forward_mode.is_context_parallel_extend(), is_prefill_context_parallel_enabled(), flush=True)
    if (
        cur_cp_seq_len != 0
        and cp_size > 1
        and forward_batch.forward_mode.is_context_parallel_extend()
        and is_prefill_context_parallel_enabled()
    ):
        return True
    else:
        return False


def cp_split_and_rebuild_data(forward_batch, input_: torch.Tensor):
    input_list = list(
        torch.split(input_, forward_batch.attn_cp_metadata.split_list, dim=0)
    )
    result = torch.cat(
        [input_list[i] for i in forward_batch.attn_cp_metadata.zigzag_index], dim=0
    ).view(-1, input_.shape[-1])
    return result


def cp_split_and_rebuild_position(forward_batch, positions: torch.Tensor):
    position_id_list = list(
        torch.split(positions, forward_batch.attn_cp_metadata.split_list, dim=-1)
    )
    positions = torch.cat(
        [position_id_list[i] for i in forward_batch.attn_cp_metadata.zigzag_index],
        dim=-1,
    )
    return positions


def cp_all_gather_reorganazied_into_tensor(
    input_tensor, total_len, cp_size, forward_batch, stream
):
    """
    Allgather communication for context_parallel hidden_states.
    """
    # The input tensor should already be padded to the same length for allgather communication.
    # No need to pad again.
    # step1
    max_len = (total_len + cp_size - 1) // cp_size
    pad_size = max_len - input_tensor.shape[0]
    if pad_size > 0:
        input_tensor = F.pad(
            input_tensor, (0, 0, 0, pad_size), mode="constant", value=0
        )
    input_tensor_full = torch.empty(
        max_len * cp_size,
        input_tensor.shape[1],
        device=input_tensor.device,
        dtype=input_tensor.dtype,
    )

    get_attention_cp_group().cp_all_gather_into_tensor_async(
        input_tensor_full, input_tensor, stream
    )

    outputs_list_max = list(
        torch.split(
            input_tensor_full, forward_batch.attn_cp_metadata.max_rank_len, dim=0
        )
    )
    outputs = torch.cat(
        [
            outputs_list_max[index][:per_rank_len]
            for index, per_rank_len in enumerate(
                forward_batch.attn_cp_metadata.per_rank_actual_token
            )
        ],
        dim=0,
    )

    return outputs


def cp_all_gather_reorganazied_into_tensor_kv_cache(
    input_tensor, total_len, cp_size, forward_batch, stream
):
    """
    Allgather communication for context_parallel KV cache.
    Handles multi-dimensional tensors (e.g., [seq_len, num_heads, head_dim]).
    """
    max_len = (total_len + cp_size - 1) // cp_size
    pad_size = max_len - input_tensor.shape[0]
    if pad_size > 0:
        # Pad the first dimension (seq_len). F.pad expects padding in reverse dimension order.
        # For n dimensional tensor, we need 2*n values: (last_dim_left, last_dim_right, ..., first_dim_left, first_dim_right)
        # To pad only the first dimension: [0, 0] * (ndim - 1) + [0, pad_size]
        padding = [0, 0] * (input_tensor.ndim - 1) + [0, pad_size]
        input_tensor = F.pad(input_tensor, padding, mode="constant", value=0)

    # Create output tensor with proper shape for all dimensions
    input_tensor_full = torch.empty(
        max_len * cp_size,
        *input_tensor.shape[1:],
        device=input_tensor.device,
        dtype=input_tensor.dtype,
    )
    # print(
    #     f"DEBUG: Rank {torch.distributed.get_rank()} cp_all_gather_reorganazied_into_tensor_kv_cache input: {input_tensor.shape}, output: {input_tensor_full.shape}, pad: {pad_size}",
    #     flush=True,
    # )

    get_attention_cp_group().cp_all_gather_into_tensor_async(
        input_tensor_full, input_tensor, stream
    )

    outputs_list_max = list(
        torch.split(
            input_tensor_full, forward_batch.attn_cp_metadata.max_rank_len, dim=0
        )
    )
    outputs = torch.cat(
        [
            outputs_list_max[index][:per_rank_len]
            for index, per_rank_len in enumerate(
                forward_batch.attn_cp_metadata.per_rank_actual_token
            )
        ],
        dim=0,
    )

    return outputs


def cp_all_gather_rerange_output(input_tensor, cp_size, forward_batch, stream):
    """
    # for in-seq-split
    |   +-----------before allgather------------+|
    |   | dp_atten_tp0: block0, block7 |
    |   | dp_atten_tp1: block1, block6 |
    |   | dp_atten_tp2: block2, block5 |
    |   | dp_atten_tp3: block3, block4 |
    |
    |   +----------before rerange---------------+|
    | block0 | block7 | block1 | block6 | block2 | block5 | block3 | block4 |
    |
    |   +--------------result-------------------+
    | block0 | block1 | block2 | block3 | block4 | block5 | block6 | block7 |
    |   +-------------------------+
    """

    # TODO: Do we need to remove the padding here?
    bs_seq_len, hidden_size = input_tensor.shape
    print(
        f"Rank {torch.distributed.get_rank()} DEBUG: cp_all_gather_rerange_output",
        bs_seq_len,
        hidden_size,
        flush=True,
    )
    output_tensor = cp_all_gather_reorganazied_into_tensor(
        input_tensor,
        forward_batch.attn_cp_metadata.total_seq_lens,
        cp_size,
        forward_batch,
        stream,
    )
    outputs_list = list(
        torch.split(
            output_tensor, forward_batch.attn_cp_metadata.reverse_split_len, dim=0
        )
    )
    output_tensor = torch.cat(
        [outputs_list[i] for i in forward_batch.attn_cp_metadata.cp_reverse_index],
        dim=0,
    )
    output_tensor = output_tensor.view(-1, hidden_size)
    return output_tensor


def cp_all_gather_rerange_kv_cache(input_tensor, cp_size, forward_batch, stream):
    """
    Allgather and reorganize KV cache from all ranks in context parallel group.

    # for in-seq-split
    |   +-----------before allgather------------+|
    |   | dp_atten_tp0: block0, block7 |
    |   | dp_atten_tp1: block1, block6 |
    |   | dp_atten_tp2: block2, block5 |
    |   | dp_atten_tp3: block3, block4 |
    |
    |   +----------before rerange---------------+|
    | block0 | block7 | block1 | block6 | block2 | block5 | block3 | block4 |
    |
    |   +--------------result-------------------+
    | block0 | block1 | block2 | block3 | block4 | block5 | block6 | block7 |
    |   +-------------------------+
    """
    output_tensor = cp_all_gather_reorganazied_into_tensor_kv_cache(
        input_tensor,
        forward_batch.attn_cp_metadata.total_seq_lens,
        cp_size,
        forward_batch,
        stream,
    )
    outputs_list = list(
        torch.split(
            output_tensor, forward_batch.attn_cp_metadata.reverse_split_len, dim=0
        )
    )
    output_tensor = torch.cat(
        [outputs_list[i] for i in forward_batch.attn_cp_metadata.cp_reverse_index],
        dim=0,
    )
    # No need to reshape - output_tensor already has the correct shape [seq_len, ...]
    return output_tensor


def prepare_context_parallel_metadata(
    kv_len,
    cp_rank,
    cp_size,
    seqs_len,
):
    """prepare_input_dp_with_cp_dsa-zigzag index
    Example (DP_ATTENT_TP == CP_SIZE == 4):
    Description:
    1. Start with a full-length request.
    2. Split the request into multiple blocks (block0 to block7).
    3. Rearrange these blocks to balance computational
        load across different DP ranks.
    4. Assign the rearranged blocks to different DP attention
        time points (dp_atten_tp0 to dp_atten_tp3).
    +---------------------------------+
    |        cp_split_tokens         |
    +---------------------------------+
    |                                 |
    |   request_with_full_length     |
    |             | split (cp_size * 2) |
    |   +-------------------------+  |
    |   | block0 | block1 | block2 | block3 | block4 | block5 | block6 | block7 |
    |   +-------------------------+  |
    |             | rerange          |
    |   +---------------------------------+
    |   | block0 | block7 | block1 | block6 | block2 | block5 | block3 | block4 |
    |   +---------------------------------+
    |             |
    |   +-------------------------+
    |   | dp_atten_tp0: block0, block7 |
    |   | dp_atten_tp1: block1, block6 |
    |   | dp_atten_tp2: block2, block5 |
    |   | dp_atten_tp3: block3, block4 |
    |   +-------------------------+

    Why zigzag rearrange?
    - Attention calculations must follow causal attention principles.
    - Simply slicing by rank order can lead to computational load imbalance:
        * First rank may focus on fewer historical key-value tokens (less computation)
        * Last rank may focus on more tokens (more computation)
    - To mitigate uneven load, the input hissenstate needs to be sliced by cp_size*2 and rearranged.
    """
    # just support batch = 1
    # kv_len: the number of tokens *computed in this extend pass* (i.e. the
    # "new" tokens). When radix/prefix cache hits, the effective KV length
    # visible to attention is: prefix_len + kv_len. CP attention must use the
    # full visible KV length, otherwise queries won't attend to cached prefix.
    kv_len = torch.tensor(kv_len)
    bs_per_cp_group = 1
    kv_len_origin = kv_len

    # Derive prefix offset from the full sequence length on CPU.
    # NOTE: forward_batch.seq_lens_cpu includes cached prefix + extend tokens.
    # In CP we only split the extend tokens, but cache_seqlens passed to FA must
    # include the cached prefix.
    prefix_len = 0
    try:
        if seqs_len is not None and len(seqs_len) == 1:
            prefix_len = int(seqs_len[0]) - int(kv_len_origin.item())
            if prefix_len < 0:
                prefix_len = 0
    except Exception:
        prefix_len = 0
    # get zigzag index
    cp_segment_num = cp_size * 2
    seq_per_batch = kv_len // cp_segment_num  # seq_len for each batch and segment
    split_list = seq_per_batch.repeat_interleave(cp_segment_num).int().tolist()
    remainder = kv_len % (cp_segment_num)
    if remainder > 0:
        split_list[:remainder] = [x + 1 for x in split_list[:remainder]]

    print(
        f"Rank {torch.distributed.get_rank()} DEBUG: split_list: {split_list}",
        flush=True,
    )

    seq_max_rank_len = (kv_len + cp_size - 1) // cp_size
    max_rank_len = seq_max_rank_len.repeat_interleave(cp_size).int().tolist()
    zigzag_index = list(
        range(cp_rank, cp_rank + bs_per_cp_group * cp_segment_num, cp_segment_num)
    ) + list(
        range(
            cp_segment_num - cp_rank - 1,
            bs_per_cp_group * cp_segment_num,
            cp_segment_num,
        )
    )

    per_rank_actual_token = list(
        split_list[i] + split_list[cp_size * 2 - i - 1] for i in range(cp_size)
    )
    reverse_split_len = [
        element
        for i in range(cp_size)
        for element in (split_list[i], split_list[cp_size * 2 - i - 1])
    ]
    # get zigzag reverse index
    cp_reverse_index = []
    for batch_id in range(bs_per_cp_group):
        cp_reverse_index.extend(
            list(range(batch_id, cp_segment_num * bs_per_cp_group, 2 * bs_per_cp_group))
            + list(
                range(
                    (cp_segment_num - 1) * bs_per_cp_group + batch_id,
                    0,
                    -2 * bs_per_cp_group,
                )
            )
        )
    prefix_sum_list = list(accumulate(split_list))

    # TODO Support multi-batch-cp-split, multi-batch-cp support has accuracy issues
    # Prefix offset is critical when radix cache hits (prefix_len > 0).
    # These "cache_seqlens" values represent how many KV tokens are visible to
    # each query segment during CP attention.
    kv_len_prev = prefix_len + prefix_sum_list[cp_rank]
    kv_len_next = prefix_len + prefix_sum_list[cp_size * 2 - cp_rank - 1]
    actual_seq_q_prev = split_list[cp_rank]
    actual_seq_q_next = split_list[cp_size * 2 - cp_rank - 1]
    # Flash Attention expects cache_seqlens to have shape (batch_size,), not scalar
    kv_len_prev_tensor = torch.tensor([kv_len_prev], device="cuda", dtype=torch.int32)
    kv_len_next_tensor = torch.tensor([kv_len_next], device="cuda", dtype=torch.int32)
    actual_seq_q_prev_tensor = torch.tensor(
        [actual_seq_q_prev], device="cuda", dtype=torch.int32
    )
    actual_seq_q_next_tensor = torch.tensor(
        [actual_seq_q_next], device="cuda", dtype=torch.int32
    )

    attn_cp_metadata = ContextParallelMetadata(
        split_list=split_list,
        max_rank_len=max_rank_len,
        zigzag_index=zigzag_index,
        per_rank_actual_token=per_rank_actual_token,
        reverse_split_len=reverse_split_len,
        cp_reverse_index=cp_reverse_index,
        kv_len_prev=kv_len_prev,
        kv_len_next=kv_len_next,
        actual_seq_q_prev=actual_seq_q_prev,
        actual_seq_q_next=actual_seq_q_next,
        kv_len_prev_tensor=kv_len_prev_tensor,
        kv_len_next_tensor=kv_len_next_tensor,
        actual_seq_q_prev_tensor=actual_seq_q_prev_tensor,
        actual_seq_q_next_tensor=actual_seq_q_next_tensor,
        total_seq_lens=kv_len_origin,
    )
    return attn_cp_metadata
