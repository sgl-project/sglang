"""Greedy draft-only vocabulary sharding on a synchronized DP graph bucket."""

import logging
import socket

import torch
import triton
import triton.language as tl
from torch import nn

from sglang.srt.utils.device import is_hip


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
    def __init__(self, group, vocab_size, presharded=None):
        super().__init__()
        self.group = group
        self.vocab_size = vocab_size
        # presharded: (first token id, valid rows) of this rank's stock TP lm_head shard.
        self.presharded = presharded

    def forward(self, hidden, weight, draft_nan_sentinel=False):
        size, rank = self.group.world_size, self.group.rank_in_group
        rows = hidden.shape[0]
        gathered = self.group.all_gather(hidden.contiguous(), dim=0)
        if self.presharded:
            start, width = self.presharded
            shard_weight = weight
        else:
            shard = triton.cdiv(self.vocab_size, size)
            start, end = (
                min(rank * shard, self.vocab_size),
                min((rank + 1) * shard, self.vocab_size),
            )
            shard_weight = weight[start:end]
            width = end - start
        logits = torch.matmul(gathered.to(weight.dtype), shard_weight.T)
        pairs = local_argmax_pair(logits, start, width, draft_nan_sentinel)
        gathered_pairs = self.group.all_gather(pairs, dim=0).view(size, size * rows, 2)
        selected = merge_argmax_pairs(gathered_pairs)
        return selected[rank * rows : (rank + 1) * rows]


def lm_head_shard_range(lm_head):
    """(first token id, number of real vocabulary rows) held by this rank's TP shard."""
    indices = lm_head.shard_indices
    return indices.org_vocab_start_index, indices.num_org_elements


def draft_head_tp_of(model):
    return getattr(getattr(model, "logits_processor", None), "draft_head_tp", None)


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


# Architectures whose draft head sharding was validated end to end (accuracy + serving).
# `--draft-head-tp on` forces any other architecture that satisfies the conditions below.
VALIDATED_DRAFT_MODELS = frozenset({"Qwen3_5ForCausalLMMTP"})


def _ineligible_reason(model, hot_token_id, parallel, spec):
    from sglang.srt.kernel_plugins.nvidia.binding import megakernel_entry
    from sglang.srt.runtime_context import get_disagg, get_lora
    from sglang.srt.speculative.eagle_megakernel import _offers_eagle_entries

    if parallel.pp_size != 1 or parallel.tp_size not in (4, 8, 16):
        return "needs pp_size 1 and tp_size 4, 8 or 16"
    if _offers_eagle_entries(draft_model=model) or (
        megakernel_entry(model, "draft_step_fold") is not None
    ):
        # The megakernel draft folds top-1 itself and never reaches draft_topk_index.
        return "the draft runs a megakernel entry"
    # Graph-only path: the DP draft graph buckets keep every rank's row count in step.
    if get_disagg().disaggregation_mode != "decode":
        return "not a decode worker of a disaggregated deployment"
    if not (parallel.attn_dp_size == parallel.tp_size and parallel.attn_tp_size == 1):
        return "needs pure attention DP (attn_dp_size == tp_size)"
    if (
        spec.speculative_eagle_topk != 1
        or spec.speculative_use_rejection_sampling
        or spec.speculative_adaptive
        or hot_token_id is not None
    ):
        return "needs greedy top-1 drafting without adaptive spec or a token map"
    if get_lora().enable_lora:
        return "LoRA is enabled"
    lm_head = getattr(model, "lm_head", None)
    lp = getattr(model, "logits_processor", None)
    weight = getattr(lm_head, "weight", None)
    if lp is None or not (
        weight is not None
        and weight.is_cuda
        and weight.dtype == torch.bfloat16
        and weight.ndim == 2
        and model.config.vocab_size < 2**24 - 1
        and lm_head.num_added_embeddings == 0
        # A tied head is the embedding module, which has no bias attribute.
        and getattr(lm_head, "bias", None) is None
        and lm_head.org_vocab_size == model.config.vocab_size
    ):
        return "the lm_head is not a plain bf16 CUDA matrix"
    if parallel.enable_dp_lm_head:
        # --enable-dp-lm-head: every rank holds the whole vocabulary.
        if lm_head.tp_size != 1 or weight.shape[0] < model.config.vocab_size:
            return "the lm_head is not replicated"
    elif (
        lm_head.tp_size != parallel.tp_size
        or weight.shape[0] != lm_head.num_embeddings_per_partition
    ):
        return "the lm_head is not sharded over the TP group"
    if (
        lp.use_fp32_lm_head
        or lp.logit_scale is not None
        or lp.final_logit_softcapping
        or lp.return_full_logits
    ):
        return "the logits processor alters the logits"
    # Last, so that "not validated" is only reported for deployments that otherwise qualify.
    if spec.draft_head_tp == "auto":
        if is_hip():
            return "not validated on ROCm; use --draft-head-tp on to force it"
        if type(model).__name__ not in VALIDATED_DRAFT_MODELS:
            return (
                f"not validated for {type(model).__name__}; "
                "use --draft-head-tp on to force it"
            )
    return None


def configure_draft_head_tp(model, hot_token_id):
    from sglang.srt.runtime_context import get_parallel, get_spec

    log = logging.getLogger(__name__)
    parallel = get_parallel()
    spec = get_spec()
    # "auto" explains itself at info level, an explicit "on" that cannot apply warns.
    declined = log.warning if spec.draft_head_tp == "on" else log.info
    reason = _ineligible_reason(model, hot_token_id, parallel, spec)
    # Agree once before capture; eager/prefill never opt into the graph-only path.
    agreement = torch.tensor([int(reason is None)], dtype=torch.int32, device="cpu")
    torch.distributed.all_reduce(
        agreement, op=torch.distributed.ReduceOp.MIN, group=parallel.tp_group.cpu_group
    )
    group = parallel.tp_group
    if not agreement.item():
        if group.rank == 0:
            declined(
                "Draft head TP not enabled: %s",
                reason or "another TP rank is ineligible",
            )
        return
    presharded = not parallel.enable_dp_lm_head
    if (
        not presharded
        and parallel.tp_size > 4
        and group.ranks != list(range(torch.distributed.get_world_size()))
    ):
        if group.rank == 0:
            declined("Draft head TP not enabled: the TP group is not the world")
        return
    # Startup-only topology agreement; replay never inspects device values.
    hosts = [None] * parallel.tp_size
    torch.distributed.all_gather_object(
        hosts,
        (group.rank, socket.gethostname(), group.local_rank),
        group=group.cpu_group,
    )
    if presharded:
        # The stock TP shard already splits the vocabulary over the whole TP group.
        groups = (
            [[rank for rank, _, _ in hosts]]
            if len({h for _, h, _ in hosts}) == 1
            else None
        )
    else:
        groups = head_tp4_rank_groups(hosts)
    if groups is None:
        if group.rank == 0:
            declined("Draft head TP not enabled: a head group spans several nodes")
        return
    if not presharded and parallel.tp_size > 4:
        from sglang.srt.distributed.parallel_state import init_mtp_head_tp_group

        group = init_mtp_head_tp_group(groups, group)
    weight = model.lm_head.weight
    model.logits_processor.draft_head_tp = DraftHeadTP(
        group,
        model.config.vocab_size,
        lm_head_shard_range(model.lm_head) if presharded else None,
    )
    log.info(
        "Draft head TP enabled: rank=%d group=%s presharded=%s weight_shape=%s",
        group.rank,
        group.ranks,
        presharded,
        tuple(weight.shape),
    )
