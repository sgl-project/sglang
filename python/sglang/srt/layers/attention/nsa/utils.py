from typing import TYPE_CHECKING, List, Tuple, Union

import torch
import triton
import triton.language as tl

from sglang.srt.layers.dp_attention import (
    DpPaddingMode,
    get_attention_cp_rank,
    get_attention_cp_size,
    get_attention_dp_rank,
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
    cp_size = get_attention_cp_size()
    seq_len = sum(forward_batch.extend_seq_lens_cpu)
    return (
        is_nsa_prefill_cp_round_robin_split()
        and seq_len > 0
        and seq_len >= cp_size
        and cp_size > 1
    )


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
    cp_size = get_attention_cp_size()
    cp_rank = get_attention_cp_rank()
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
    attn_cp_size = get_attention_cp_size()
    for i in range(sync_group_size):
        global_num_tokens[i] = ceil_align(global_num_tokens[i], attn_cp_size)
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
        tokens = ceil_div(tokens, attn_cp_size)
    return tokens


def pad_nsa_cache_seqlens(forward_batch: "ForwardBatch", nsa_cache_seqlens):
    attn_cp_size = get_attention_cp_size()
    needs_cp_pad = attn_cp_size > 1 and can_nsa_prefill_cp_round_robin_split(
        forward_batch
    )
    needs_dp_pad = forward_batch.global_num_tokens_cpu is not None
    if not needs_cp_pad and not needs_dp_pad:
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


def can_nsa_cp_split(seq_len: int, cp_size: int, use_nsa: bool, forward_batch):
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
        and sum(forward_batch.extend_seq_lens_cpu) >= cp_size
    ):
        return True
    else:
        return False


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
    cp_size = get_attention_cp_size()
    cp_rank = get_attention_cp_rank()
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
    cp_size = get_attention_cp_size()
    cp_rank = get_attention_cp_rank()
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
        forward_batch.attn_cp_metadata is not None
        and nsa_enable_prefill_cp
        and forward_batch.forward_mode.is_context_parallel_extend()
    ):
        return True
    else:
        return False
