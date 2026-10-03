"""Greedy draft-only vocabulary sharding on a synchronized DP graph bucket."""

import logging

import torch
import triton
import triton.language as tl
from torch import nn


@triton.jit
def _partial_max(
    X,
    V,
    I,
    N: tl.constexpr,
    STRIDE: tl.constexpr,
    START: tl.constexpr,
    SPLITS: tl.constexpr,
    SENTINEL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    split = tl.program_id(1)
    col = split * BLOCK + tl.arange(0, BLOCK)
    valid = col < N
    x = tl.load(X + row * STRIDE + col, valid, other=-float("inf")).to(tl.float32)
    if SENTINEL:
        x = tl.where(x == x, x, -1e30)
    nan = valid & (x != x)
    first_nan = tl.min(tl.where(nan, col, 2147483647), 0)
    maximum = tl.max(tl.where(nan, -float("inf"), x), 0)
    index = tl.min(tl.where(valid & (x == maximum), col, 2147483647), 0)
    index = tl.where(first_nan != 2147483647, first_nan, index)
    maximum = tl.where(first_nan != 2147483647, float("nan"), maximum)
    tl.store(V + row * SPLITS + split, maximum)
    tl.store(I + row * SPLITS + split, index + START)


@triton.jit
def _merge(
    V,
    I,
    OUT,
    ROWS: tl.constexpr,
    PARTS: tl.constexpr,
    RANK_MAJOR: tl.constexpr,
    PACK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    part = tl.arange(0, BLOCK)
    if RANK_MAJOR:
        off = (part * ROWS + row) * 2
        value = tl.load(V + off, part < PARTS, other=-float("inf"))
        index = tl.load(V + off + 1, part < PARTS, other=2147483647).to(tl.int32)
    else:
        off = row * PARTS + part
        value = tl.load(V + off, part < PARTS, other=-float("inf"))
        index = tl.load(I + off, part < PARTS, other=2147483647)
    nan = (part < PARTS) & (value != value)
    nan_index = tl.min(tl.where(nan, index, 2147483647), 0)
    maximum = tl.max(tl.where(nan, -float("inf"), value), 0)
    result = tl.min(tl.where((part < PARTS) & (value == maximum), index, 2147483647), 0)
    result = tl.where(nan_index != 2147483647, nan_index, result)
    if PACK:
        maximum = tl.where(nan_index != 2147483647, float("nan"), maximum)
        tl.store(OUT + row * 2, maximum)
        tl.store(OUT + row * 2 + 1, result.to(tl.float32))
    else:
        tl.store(OUT + row, result.to(tl.int64))


def local_argmax_pair(
    logits, vocab_start=0, valid_vocab=None, draft_nan_sentinel=False
):
    rows, width = logits.shape
    width = width if valid_vocab is None else valid_vocab
    pairs = torch.empty((rows, 2), device=logits.device, dtype=torch.float32)
    if rows == 0:
        return pairs
    if width == 0:
        pairs[:, 0] = -float("inf")
        pairs[:, 1] = 2**24 - 1
        return pairs
    splits = triton.cdiv(width, 8192)
    vals = torch.empty((rows, splits), device=logits.device, dtype=torch.float32)
    ids = torch.empty((rows, splits), device=logits.device, dtype=torch.int32)
    _partial_max[(rows, splits)](
        logits,
        vals,
        ids,
        width,
        logits.stride(0),
        vocab_start,
        splits,
        draft_nan_sentinel,
        8192,
        num_warps=8,
    )
    _merge[(rows,)](
        vals,
        ids,
        pairs,
        rows,
        splits,
        False,
        True,
        triton.next_power_of_2(splits),
        num_warps=1,
    )
    return pairs


def merge_argmax_pairs(pairs):
    ranks, rows, _ = pairs.shape
    result = torch.empty((rows, 1), dtype=torch.int64, device=pairs.device)
    if rows:
        _merge[(rows,)](
            pairs,
            pairs,
            result,
            rows,
            ranks,
            True,
            False,
            triton.next_power_of_2(ranks),
            num_warps=1,
        )
    return result


class DraftHeadTP(nn.Module):
    def __init__(self, group, vocab_size):
        super().__init__()
        self.group = group
        self.vocab_size = vocab_size
        self.calls = []

    def forward(self, hidden, weight, draft_nan_sentinel=False):
        size, rank = self.group.world_size, self.group.rank_in_group
        rows = hidden.shape[0]
        gathered = self.group.all_gather(hidden.contiguous(), dim=0)
        shard = triton.cdiv(self.vocab_size, size)
        start, end = (
            min(rank * shard, self.vocab_size),
            min((rank + 1) * shard, self.vocab_size),
        )
        logits = torch.matmul(gathered.to(weight.dtype), weight[start:end].T)
        pairs = local_argmax_pair(logits, start, draft_nan_sentinel=draft_nan_sentinel)
        gathered_pairs = self.group.all_gather(pairs, dim=0).view(size, size * rows, 2)
        selected = merge_argmax_pairs(gathered_pairs)
        self.calls.append(
            dict(
                rows=rows,
                hidden_allgather=1,
                argmax_allgather=1,
                hidden_input_bytes=hidden.numel() * hidden.element_size(),
                argmax_input_bytes=pairs.numel() * pairs.element_size(),
                draft_nan_sentinel=draft_nan_sentinel,
            )
        )
        return selected[rank * rows : (rank + 1) * rows]


def head_tp4_rank_groups(rank_hosts):
    if len(rank_hosts) not in (4, 8, 16):
        return None
    groups = []
    for offset in range(0, len(rank_hosts), 4):
        members = rank_hosts[offset : offset + 4]
        if len({host for _, host, _ in members}) != 1:
            return None
        if len({device for _, _, device in members}) != 4:
            return None
        groups.append([rank for rank, _, _ in members])
    return groups


def configure_draft_head_tp(model, hot_token_id):
    from sglang.srt.runtime_context import get_disagg, get_lora, get_parallel, get_spec
    from sglang.srt.models.qwen3_5_mtp import Qwen3_5ForCausalLMMTP

    parallel, spec = get_parallel(), get_spec()
    if parallel.pp_size != 1 or parallel.tp_size not in (4, 8, 16):
        return
    weight = getattr(getattr(model, "lm_head", None), "weight", None)
    lp = getattr(model, "logits_processor", None)
    eligible = (
        isinstance(model, Qwen3_5ForCausalLMMTP)
        and get_disagg().disaggregation_mode == "decode"
        and parallel.enable_dp_lm_head
        and parallel.attn_dp_size == parallel.tp_size
        and spec.speculative_eagle_topk == 1
        and not spec.speculative_use_rejection_sampling
        and not spec.speculative_adaptive
        and hot_token_id is None
        and not get_lora().enable_lora
        and weight is not None
        and weight.is_cuda
        and weight.dtype == torch.bfloat16
        and weight.ndim == 2
        and weight.shape[0] >= model.config.vocab_size
        and model.config.vocab_size < 2**24 - 1
        and not lp.use_fp32_lm_head
        and lp.logit_scale is None
        and not lp.final_logit_softcapping
        and not lp.return_full_logits
    )
    # Agree once before capture; eager/prefill never opt into the graph-only path.
    agreement = torch.tensor([int(eligible)], dtype=torch.int32, device="cpu")
    torch.distributed.all_reduce(
        agreement, op=torch.distributed.ReduceOp.MIN, group=parallel.tp_group.cpu_group
    )
    if agreement.item():
        group = parallel.tp_group
        if parallel.tp_size > 4:
            import socket
            from sglang.srt.distributed.parallel_state import init_mtp_head_tp_group

            if group.ranks != list(range(torch.distributed.get_world_size())):
                return
            # Startup-only topology agreement; replay never inspects device values.
            hosts = [None] * parallel.tp_size
            torch.distributed.all_gather_object(
                hosts,
                (group.rank, socket.gethostname(), group.local_rank),
                group=group.cpu_group,
            )
            groups = head_tp4_rank_groups(hosts)
            if groups is None:
                return
            group = init_mtp_head_tp_group(groups, group)
        model.draft_head_tp = DraftHeadTP(group, model.config.vocab_size)
        logging.getLogger(__name__).info(
            "MTP draft head TP4 enabled: rank=%d group=%s full_weight_shape=%s "
            "shared_weight_storage_bytes=%d",
            group.rank,
            group.ranks,
            tuple(weight.shape),
            weight.untyped_storage().nbytes(),
        )
