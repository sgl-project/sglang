# temp NSA debugging environ
from dataclasses import dataclass
from itertools import accumulate
from typing import TYPE_CHECKING, List, Tuple, Union

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from sglang.srt.layers.dp_attention import (
    DpPaddingMode,
    attn_tp_all_gather_into_tensor,
    get_attention_dp_rank,
    get_attention_tp_group,
    get_attention_tp_rank,
    get_attention_tp_size,
)
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils.common import ceil_align, ceil_div

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


def compute_nsa_seqlens(original_seq_lens, nsa_index_topk: int):
    return original_seq_lens.clamp(max=nsa_index_topk)


def is_nsa_enable_prefill_cp():
    return get_global_server_args().enable_nsa_prefill_context_parallel


def is_nsa_prefill_cp_in_seq_split():
    return (
        is_nsa_enable_prefill_cp()
        and get_global_server_args().nsa_prefill_cp_mode == "in-seq-split"
    )


def is_nsa_prefill_cp_round_robin_split():
    return (
        is_nsa_enable_prefill_cp()
        and get_global_server_args().nsa_prefill_cp_mode == "round-robin-split"
    )


def can_nsa_prefill_cp_round_robin_split(forward_batch: "ForwardBatch"):
    if not forward_batch.forward_mode.is_context_parallel_extend():
        return False
    cp_size = get_attention_tp_size()
    seq_len = sum(forward_batch.extend_seq_lens_cpu)
    return is_nsa_prefill_cp_round_robin_split() and seq_len > 0 and cp_size > 1


def nsa_cp_round_robin_split_data(input_: Union[torch.Tensor, List]):
    """
    # for round-robin-split, split the tokens evenly according to the rule of token_idx % cp_size.
    |   +-----------before split------------+|
    | token0, token1, token2, token3, token4, token5, token6, token7, ...
    |
    |   +--------------result-------------------+
    | dp_atten_tp0: token0, token4, token8, token12, token16, ... |
    | dp_atten_tp1: token1, token5, token9, token13, token17, ... |
    | dp_atten_tp2: token2, token6, token10, token14, token18, ... |
    | dp_atten_tp3: token3, token7, token11, token15, token19, ... |
    |   +-------------------------+
    """
    cp_size = get_attention_tp_size()
    cp_rank = get_attention_tp_rank()
    if isinstance(input_, (tuple, list)):
        indices = range(cp_rank, len(input_), cp_size)
        return input_[indices]

    tokens = len(input_)
    if tokens % cp_size != 0:
        cur_len = tokens // cp_size + (tokens % cp_size > cp_rank)
        if cur_len == 0:
            return input_.new_empty(0, *input_.shape[1:])
        indices = torch.arange(cp_rank, tokens, cp_size, device=input_.device)
        return input_[indices]

    # for torch device tensor
    return input_.view(-1, cp_size, *input_.shape[1:])[:, cp_rank].contiguous()


def cal_padded_tokens(forward_batch: "ForwardBatch"):
    # Consistent with the padding calculation logic in ForwardBatch.prepare_mlp_sync_batch,
    # calculate the actual token length after padding when attn_tp_size > 1 or in the MAX_LEN padding mode.
    global_num_tokens = forward_batch.global_num_tokens_cpu.copy()
    sync_group_size = len(global_num_tokens)
    attn_tp_size = get_attention_tp_size()
    for i in range(sync_group_size):
        global_num_tokens[i] = ceil_align(global_num_tokens[i], attn_tp_size)
    dp_padding_mode = DpPaddingMode.get_dp_padding_mode(
        forward_batch.is_extend_in_batch, global_num_tokens
    )
    if dp_padding_mode.is_max_len():
        tokens = max(global_num_tokens)
    elif len(global_num_tokens) > 1:
        tokens = global_num_tokens[get_attention_dp_rank()]
    else:
        tokens = global_num_tokens[0]
    if can_nsa_prefill_cp_round_robin_split(forward_batch):
        tokens = ceil_div(tokens, attn_tp_size)
    return tokens


def pad_nsa_cache_seqlens(forward_batch: "ForwardBatch", nsa_cache_seqlens):
    if forward_batch.global_num_tokens_cpu is None:
        return nsa_cache_seqlens
    tokens = cal_padded_tokens(forward_batch)
    pad_len = tokens - nsa_cache_seqlens.shape[0]
    if pad_len > 0:
        nsa_cache_seqlens = torch.cat(
            [
                nsa_cache_seqlens,
                nsa_cache_seqlens.new_zeros(pad_len, *nsa_cache_seqlens.shape[1:]),
            ]
        )
    return nsa_cache_seqlens


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


def can_cp_split(seq_len: int, cp_size: int, use_nsa: bool, forward_batch):
    if is_nsa_prefill_cp_round_robin_split():
        cur_cp_seq_len = seq_len // cp_size
        assert (
            seq_len % cp_size == 0
        ), f"seq_len {seq_len} is not divisible by cp_size {cp_size} when nsa_prefill_cp_mode is round-robin-split"
    else:
        # TODO current just support prefill batch=1 and len(input_ids) > self.cp_size * 2
        # Note: (self.cp_size * 2) To achieve load balancing for seq computation,
        # the seq data needs to be divided and recombined at twice the size of cp_size.
        cur_cp_seq_len = seq_len // (cp_size * 2)
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
    if is_nsa_prefill_cp_round_robin_split():
        cp_size = get_attention_tp_size()
        assert (
            input_.shape[0] % cp_size == 0
        ), f"Expect input shape 0 can divided by cp size, but got input shape {input_.shape}, cp size {cp_size}"
        return nsa_cp_round_robin_split_data(input_)

    input_list = list(
        torch.split(input_, forward_batch.nsa_cp_metadata.split_list, dim=0)
    )
    result = torch.cat(
        [input_list[i] for i in forward_batch.nsa_cp_metadata.zigzag_index], dim=0
    ).view(-1, input_.shape[-1])
    return result


def cp_split_and_rebuild_position(forward_batch, positions: torch.Tensor):
    if is_nsa_prefill_cp_round_robin_split():
        cp_size = get_attention_tp_size()
        assert positions.shape[0] % cp_size == 0, (
            f"Expect positions shape 0 can divided by cp size, but got positions shape {positions.shape}, "
            f"cp size {cp_size}"
        )
        return nsa_cp_round_robin_split_data(positions)

    position_id_list = list(
        torch.split(positions, forward_batch.nsa_cp_metadata.split_list, dim=-1)
    )
    positions = torch.cat(
        [position_id_list[i] for i in forward_batch.nsa_cp_metadata.zigzag_index],
        dim=-1,
    )
    return positions


@triton.jit
def nsa_cp_round_robin_split_q_seqs_kernel(
    in_seqs_ptr,
    out_seqs_ptr,
    bs_idx_ptr,
    tokens: tl.constexpr,
    cp_size: tl.constexpr,
    cp_rank: tl.constexpr,
):
    extra_seq = 0
    bs_idx = 0
    for bs in range(tokens):
        cur_len = tl.load(in_seqs_ptr + bs)
        cur_len += extra_seq
        cur_seq = cur_len // cp_size + (cur_len % cp_size > cp_rank)
        if cur_seq > 0:
            tl.store(bs_idx_ptr + bs_idx, bs)
            tl.store(out_seqs_ptr + bs_idx, cur_seq)
            bs_idx += 1
        extra_seq = cur_len - cur_seq * cp_size


def nsa_cp_round_robin_split_q_seqs_cpu(extend_seqs):
    cp_size = get_attention_tp_size()
    cp_rank = get_attention_tp_rank()
    extra_seq = 0
    q_seqs = []
    for bs, cur_len in enumerate(extend_seqs):
        cur_len += extra_seq
        cur_seq = cur_len // cp_size + int(cur_len % cp_size > cp_rank)
        q_seqs.append(cur_seq)
        extra_seq = cur_len - cur_seq * cp_size
    bs_idx = list([i for i, x in enumerate(q_seqs) if x > 0])
    q_seqs = [q_len for q_len in q_seqs if q_len > 0]
    return q_seqs, bs_idx


def nsa_cp_round_robin_split_q_seqs(
    extend_seqs_cpu, extend_seqs
) -> Tuple[List, torch.Tensor, List, torch.Tensor]:
    """
    round-robin-split distributes tokens across ranks based on token_idx % cp_size.

    Return:
    ret_q_lens_cpu(List) and ret_q_lens(torch.Tensor): the partitioned length (excluding zeros) on the current cp rank
        for each sequence after distribution across cp ranks.
    bs_idx_cpu(List) and bs_idx(torch.Tensor): marks which sequences are ultimately selected,
        i.e., those with a partitioned length greater than zero.
    """
    cp_size = get_attention_tp_size()
    cp_rank = get_attention_tp_rank()
    # len(ret_q_lens_cpu) == len(bs_idx_cpu)
    ret_q_lens_cpu, bs_idx_cpu = nsa_cp_round_robin_split_q_seqs_cpu(extend_seqs_cpu)
    ret_q_lens = torch.empty(
        (len(bs_idx_cpu),), device=extend_seqs.device, dtype=extend_seqs.dtype
    )
    bs_idx = torch.empty(
        (len(bs_idx_cpu),), device=extend_seqs.device, dtype=torch.int32
    )
    grid = (1,)
    nsa_cp_round_robin_split_q_seqs_kernel[grid](
        extend_seqs, ret_q_lens, bs_idx, len(extend_seqs), cp_size, cp_rank
    )
    return ret_q_lens_cpu, ret_q_lens, bs_idx_cpu, bs_idx


def nsa_use_prefill_cp(forward_batch, nsa_enable_prefill_cp=None):
    if nsa_enable_prefill_cp is None:
        nsa_enable_prefill_cp = is_nsa_enable_prefill_cp()
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

    # for round-robin-split
    |   +-----------before allgather------------+|
    | dp_atten_tp0: token0, token4, token8, token12, token16, ... |
    | dp_atten_tp1: token1, token5, token9, token13, token17, ... |
    | dp_atten_tp2: token2, token6, token10, token14, token18, ... |
    | dp_atten_tp3: token3, token7, token11, token15, token19, ... |
    |
    |   +--------------result-------------------+
    | token0, token1, token2, token3, token4, token5, token6, token7, ...
    |   +-------------------------+
    """
    if is_nsa_prefill_cp_round_robin_split():
        output_tensor = input_tensor.new_empty(
            (input_tensor.shape[0] * cp_size, *input_tensor.shape[1:]),
        )
        attn_tp_all_gather_into_tensor(
            output_tensor,
            input_tensor,
        )
        out_shape = output_tensor.shape
        output_tensor = (
            output_tensor.view(cp_size, -1, *out_shape[1:])
            .transpose(0, 1)
            .reshape(out_shape)
        )
        return output_tensor

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
    if is_nsa_prefill_cp_round_robin_split():
        return True
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
    kv_len = torch.tensor(kv_len)
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
