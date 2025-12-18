# temp NSA debugging environ
from dataclasses import dataclass
from itertools import accumulate
from typing import List

import torch
import torch.nn.functional as F

from sglang.srt.layers.dp_attention import get_attention_tp_group
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import get_bool_env_var

NSA_DUAL_STREAM = get_bool_env_var("SGLANG_NSA_DUAL_STREAM", "true")
NSA_FUSE_TOPK = get_bool_env_var("SGLANG_NSA_FUSE_TOPK", "true")

NSA_FLASHMLA_BACKEND_DECODE_COMPUTE_FP8 = get_bool_env_var(
    "SGLANG_NSA_FLASHMLA_BACKEND_DECODE_COMPUTE_FP8", "true"
)
NSA_QUANT_K_CACHE_FAST = get_bool_env_var("SGLANG_NSA_QUANT_K_CACHE_FAST", "true")
NSA_DEQUANT_K_CACHE_FAST = get_bool_env_var("SGLANG_NSA_DEQUANT_K_CACHE_FAST", "true")


def print_nsa_bool_env_vars():
    msg = ""
    for k, v in globals().items():
        if k.startswith("NSA_") and isinstance(v, bool):
            msg += f"{k}={v} "
    print(msg, flush=True)


def compute_nsa_seqlens(original_seq_lens, nsa_index_topk: int):
    return original_seq_lens.clamp(max=nsa_index_topk)


def is_nsa_enable_prefill_cp():
    return get_global_server_args().enable_nsa_prefill_context_parallel


@dataclass
class NSAContextParallelMetadata:

    split_list: List[int] = None
    max_rank_len: List[int] = None
    zigzag_index: List[int] = None
    per_rank_actual_token: List[int] = None
    reverse_split_len: List[int] = None
    cp_reverse_index: List[int] = None
    kv_len_prev: int = -1
    kv_len_next: int = -1
    actual_seq_q_prev: int = -1
    actual_seq_q_next: int = -1
    kv_len_prev_tensor: torch.Tensor = None
    kv_len_next_tensor: torch.Tensor = None
    actual_seq_q_prev_tensor: torch.Tensor = None
    actual_seq_q_next_tensor: torch.Tensor = None
    total_seq_lens: torch.Tensor = None


def can_cp_split(cur_cp_seq_len: int, cp_size: int, use_nsa: bool, forward_batch):
    if (
        cur_cp_seq_len != 0
        and cp_size > 1
        and use_nsa
        and forward_batch.forward_mode.is_context_parallel_extend()
        and is_nsa_enable_prefill_cp()
    ):
        return True
    else:
        return False


def cp_split_and_rebuild_data(forward_batch, input_: torch.Tensor):
    input_list = list(
        torch.split(input_, forward_batch.nsa_cp_metadata.split_list, dim=0)
    )
    result = torch.cat(
        [input_list[i] for i in forward_batch.nsa_cp_metadata.zigzag_index], dim=0
    ).view(-1, input_.shape[-1])
    return result


def cp_split_and_rebuild_position(forward_batch, positions: torch.Tensor):
    position_id_list = list(
        torch.split(positions, forward_batch.nsa_cp_metadata.split_list, dim=-1)
    )
    positions = torch.cat(
        [position_id_list[i] for i in forward_batch.nsa_cp_metadata.zigzag_index],
        dim=-1,
    )
    return positions


def enable_prefill_cp(forward_batch, nsa_enable_prefill_cp):
    if (
        forward_batch.nsa_cp_metadata is not None
        and nsa_enable_prefill_cp
        and forward_batch.forward_mode.is_context_parallel_extend()
    ):
        return True
    else:
        return False


def cp_attn_tp_all_gather_reorganazied_into_tensor(
    input_: torch.Tensor, total_len, attn_tp_size, forward_batch, stream_op
):
    """
    Allgather communication for context_parallel(kv_cache, index_k, hidden_states).
    This implementation mainly consists of three parts:
    Step 1, padding the input shape to unify the shape for allgather communication (the shape must be the same).
    Step 2, allgather communication(async).
    Step 3, removing the padding and reassembling the data according to the actual tokens.
    """
    # step1
    max_len = (total_len + attn_tp_size - 1) // attn_tp_size
    pad_size = max_len - input_.shape[0]
    if pad_size > 0:
        input_ = F.pad(input_, (0, 0, 0, pad_size), mode="constant", value=0)
    input_tensor_all = torch.empty(
        max_len * attn_tp_size,
        input_.shape[1],
        device=input_.device,
        dtype=input_.dtype,
    )
    # step2
    get_attention_tp_group().cp_all_gather_into_tensor_async(
        input_tensor_all, input_, stream_op
    )
    # step3
    outputs_list_max = list(
        torch.split(input_tensor_all, forward_batch.nsa_cp_metadata.max_rank_len, dim=0)
    )
    outputs = torch.cat(
        [
            outputs_list_max[index][:per_rank_len]
            for index, per_rank_len in enumerate(
                forward_batch.nsa_cp_metadata.per_rank_actual_token
            )
        ],
        dim=0,
    )
    return outputs


def cp_all_gather_rerange_output(input_tensor, cp_size, forward_batch, stream):
    """
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
    bs_seq_len, hidden_size = input_tensor.shape
    output_tensor = cp_attn_tp_all_gather_reorganazied_into_tensor(
        input_tensor,
        forward_batch.nsa_cp_metadata.total_seq_lens,
        cp_size,
        forward_batch,
        stream,
    )
    outputs_list = list(
        torch.split(
            output_tensor, forward_batch.nsa_cp_metadata.reverse_split_len, dim=0
        )
    )
    output_tensor = torch.cat(
        [outputs_list[i] for i in forward_batch.nsa_cp_metadata.cp_reverse_index], dim=0
    )
    output_tensor = output_tensor.view(-1, hidden_size)
    return output_tensor


def calculate_cp_seq_idx(cp_chunks_len, seqs_len):
    """Used to obtain the index of the seq corresponding
    to each cp block in the forwardbatch, and the starting
    and ending positions of the corresponding seq in the cp block"""
    j = 0
    tuple_len = []  # Only keep this result list
    cumulative = {}  # Used to track cumulative values for each index

    for i in range(len(cp_chunks_len)):
        current_dict = {}
        current_tuples = []
        c_val = cp_chunks_len[i]

        while j < len(seqs_len):
            s_val = seqs_len[j]
            if s_val == c_val:
                idx = j
                current_dict[idx] = s_val
                # Update cumulative value for this index
                cumulative[idx] = cumulative.get(idx, 0) + s_val
                j += 1
                break
            elif s_val > c_val:
                idx = j
                current_dict[idx] = c_val
                # Update cumulative value for this index
                cumulative[idx] = cumulative.get(idx, 0) + c_val
                seqs_len[j] = s_val - c_val
                break
            else:  # s_val < c_val
                idx = j
                current_dict[idx] = s_val
                # Update cumulative value for this index
                cumulative[idx] = cumulative.get(idx, 0) + s_val
                c_val -= s_val
                j += 1

        # Build tuple: (index, historical cumulative, historical+current)
        for idx, val in current_dict.items():
            # Subtract current value to get historical cumulative
            prev_cum = cumulative.get(idx, 0) - val
            current_cum = prev_cum + val
            current_tuples.append((idx, prev_cum, current_cum))

        tuple_len.append(current_tuples)
    return tuple_len


def prepare_input_dp_with_cp_dsa(
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
    bs_per_cp_group = 1
    kv_len_origin = kv_len
    # get zigzag index
    cp_segment_num = cp_size * 2
    seq_per_batch = kv_len // cp_segment_num  # seq_len for each batch and segment
    split_list = seq_per_batch.repeat_interleave(cp_segment_num).int().tolist()
    remainder = kv_len % (cp_segment_num)
    if remainder > 0:
        split_list[:remainder] = [x + 1 for x in split_list[:remainder]]

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
    # cp_seq_index = calculate_cp_seq_idx(split_list[:], seqs_len[:])
    kv_len_prev = prefix_sum_list[cp_rank]
    kv_len_next = prefix_sum_list[cp_size * 2 - cp_rank - 1]
    actual_seq_q_prev = split_list[cp_rank]
    actual_seq_q_next = split_list[cp_size * 2 - cp_rank - 1]
    kv_len_prev_tensor = torch.tensor(kv_len_prev).to(device="cuda", dtype=torch.int32)
    kv_len_next_tensor = torch.tensor(kv_len_next).to(device="cuda", dtype=torch.int32)
    actual_seq_q_prev_tensor = torch.tensor(actual_seq_q_prev).to(
        device="cuda", dtype=torch.int32
    )
    actual_seq_q_next_tensor = torch.tensor(actual_seq_q_next).to(
        device="cuda", dtype=torch.int32
    )

    nsa_cp_metadata = NSAContextParallelMetadata(
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
    return nsa_cp_metadata
