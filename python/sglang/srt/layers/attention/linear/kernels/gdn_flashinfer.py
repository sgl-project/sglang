"""FlashInfer-based kernels for GDN (Gated Delta Network) linear attention.

Both SM90 and SM100 use the same pool layout: [pool, HV, V, K] (K-last).

SM90 (Hopper): full support — decode, prefill, MTP.  State dtype: fp32.
SM100 (Blackwell): full support — decode, prefill, MTP.

Requires flashinfer >= 0.6.14.
"""

from __future__ import annotations

import logging
import math
import os
from typing import TYPE_CHECKING, Optional

import msgspec
import torch

from sglang.srt.layers.attention.linear.kernels.kernel_backend import (
    LinearAttnKernelBase,
)
from sglang.srt.runtime_context import (
    mamba_cache_chunk_size,
)
from sglang.srt.environ import envs
from sglang.srt.utils import is_cuda
from sglang.srt.utils.common import is_pin_memory_available

if TYPE_CHECKING:
    from sglang.srt.layers.attention.linear.gdn_prefill_graph import (
        GDNFlashInferGraphTables,
    )
    from sglang.srt.layers.attention.mamba.mamba2_metadata import ForwardMetadata
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

logger = logging.getLogger(__name__)

_FLASHINFER_GDN_ALIGNMENT = 32


def copy_verify_intermediate_rows(
    destination: torch.Tensor,
    positional_source: torch.Tensor,
    dst_indices_raw: torch.Tensor,
) -> None:
    """Move FlashInfer's positional verify snapshots to their owned rows."""
    count = dst_indices_raw.shape[0]
    destination.index_copy_(
        0,
        dst_indices_raw.to(device=destination.device, dtype=torch.long),
        positional_source[:count],
    )


def _empty_aligned_like(
    tensor: torch.Tensor, alignment: int = _FLASHINFER_GDN_ALIGNMENT
) -> torch.Tensor:
    element_size = tensor.dtype.itemsize
    alignment_elements = max(1, alignment // element_size)
    storage = torch.empty(
        tensor.numel() + alignment_elements - 1,
        dtype=tensor.dtype,
        device=tensor.device,
    )
    start_bytes = (-storage.data_ptr()) % alignment
    if start_bytes % element_size:
        raise RuntimeError(
            f"Cannot align {tensor.dtype} storage at {storage.data_ptr()} "
            f"to {alignment} bytes"
        )
    start = start_bytes // element_size
    return storage[start : start + tensor.numel()].view(tensor.shape)


# ---------------------------------------------------------------------------
# Lazy import for FlashInfer GDN kernels
# ---------------------------------------------------------------------------
_flashinfer_gdn_available: Optional[bool] = None
_flashinfer_chunk_gated_delta_rule = None
_flashinfer_gated_delta_rule_mtp = None
_flashinfer_gated_delta_rule_decode = None
_flashinfer_gated_delta_rule_mtp_bf16 = None


def maybe_build_flashinfer_checkpoint_plan(
    forward_batch: ForwardBatch,
    forward_metadata: ForwardMetadata,
    device: str,
) -> Optional[torch.Tensor]:
    """Populate packed FlashInfer checkpoint metadata when tracking requires it.
    Returns the host copy of track_ssm_h_src when built from the host lists."""
    if (
        forward_metadata.track_ssm_h_src is None
        or forward_metadata.track_ssm_h_src.numel() == 0
    ):
        return

    checkpoint_every_n_tokens = mamba_cache_chunk_size()
    # SGLANG_GDN_PREFILL_SYNC_FREE_METADATA (default off): build the plan from the batch's host lists.
    host_lists = (
        _host_track_lists(forward_batch)
        if envs.SGLANG_GDN_PREFILL_SYNC_FREE_METADATA.get()
        else None
    )
    if host_lists is not None:
        # The device tensors below are H2D copies of these lists; reading them back syncs the stream.
        seq_lens_cpu, mask_cpu, track_seqlens_cpu, prefix_lens_cpu = host_lists
        extend_seq_lens = torch.tensor(seq_lens_cpu, dtype=torch.int64)
        track_mask = torch.tensor(mask_cpu, dtype=torch.bool)
        relative_track_lens = torch.tensor(
            track_seqlens_cpu, dtype=torch.int64
        ) - torch.tensor(prefix_lens_cpu, dtype=torch.int64)
    else:
        extend_seq_lens = forward_batch.extend_seq_lens.to(
            device="cpu", dtype=torch.int64
        )
        track_mask = forward_batch.mamba_track_mask.to(device="cpu", dtype=torch.bool)
        relative_track_lens = forward_batch.mamba_track_seqlens.to(
            device="cpu", dtype=torch.int64
        ) - forward_batch.extend_prefix_lens.to(device="cpu", dtype=torch.int64)

    checkpoint_counts = extend_seq_lens // checkpoint_every_n_tokens
    checkpoint_cu_starts = torch.zeros(checkpoint_counts.numel() + 1, dtype=torch.int64)
    checkpoint_cu_starts[1:] = torch.cumsum(checkpoint_counts, dim=0)

    use_checkpoint = track_mask & (relative_track_lens % checkpoint_every_n_tokens != 0)
    track_checkpoint_src = checkpoint_cu_starts[:-1][use_checkpoint] + (
        relative_track_lens[use_checkpoint] // checkpoint_every_n_tokens - 1
    )
    if track_checkpoint_src.numel() and track_checkpoint_src.min() < 0:
        raise ValueError("Tracked GDN state precedes the first FlashInfer checkpoint.")
    assert track_checkpoint_src.numel() == forward_metadata.track_ssm_h_dst.numel()

    from sglang.srt.utils import is_pin_memory_available

    if host_lists is not None and is_pin_memory_available():
        # A pageable H2D copy may wait for the stream; a pinned one is only enqueued.
        track_checkpoint_src = track_checkpoint_src.pin_memory()
        checkpoint_cu_starts = checkpoint_cu_starts.pin_memory()
    forward_metadata.track_ssm_h_src = track_checkpoint_src.to(
        device, non_blocking=True
    )
    forward_metadata.state_checkpoint_cu_starts = checkpoint_cu_starts.to(
        device, non_blocking=True
    )
    forward_metadata.num_state_checkpoints = int(checkpoint_cu_starts[-1])
    forward_metadata.state_checkpoint_every_n_tokens = checkpoint_every_n_tokens
    if host_lists is not None:
        _log_host_checkpoint_plan_once(
            num_checkpoints=forward_metadata.num_state_checkpoints
        )
        return track_checkpoint_src
    return None


_host_checkpoint_plan_logged = False


def _log_host_checkpoint_plan_once(*, num_checkpoints: int) -> None:
    global _host_checkpoint_plan_logged
    if not _host_checkpoint_plan_logged:
        _host_checkpoint_plan_logged = True
        logger.info(
            "FlashInfer GDN checkpoint plan: built from the batch's host lists, "
            "no device read (first: %d checkpoints)",
            num_checkpoints,
        )


def _host_track_lists(forward_batch: ForwardBatch) -> Optional[tuple]:
    """(extend_seq_lens, track mask, track seqlens, extend_prefix_lens) as host lists when all cover
    the batch: MambaAttnBackendBase._has_cpu_prefill_track_metadata, under which the track plan
    (track_ssm_h_dst included) is already built from these lists."""
    mode = forward_batch.forward_mode
    lists = (
        forward_batch.extend_seq_lens_cpu,
        forward_batch.mamba_prefill_track_mask_cpu,
        forward_batch.mamba_track_seqlens_cpu,
        forward_batch.extend_prefix_lens_cpu,
    )
    if not mode.is_extend() or mode.is_target_verify():
        return None
    if any(v is None or len(v) != forward_batch.batch_size for v in lists):
        return None
    return lists


def _install_gdn_prefill_tmem_dealloc_fix() -> None:
    """FlashInfer #8854 on FlashInfer 0.7.2's Blackwell (SM10x) GDN prefill kernels: TMEM is freed only after every
    reading warp's last tcgen05.ld (sglang/kernels/ops/mamba/gdn_prefill_tmem_dealloc.py). Bitwise. With another
    FlashInfer than 0.7.2 nothing changes and the kernels run as shipped, with a warning."""
    if torch.cuda.get_device_capability()[0] != 10:
        return
    try:
        from sglang.kernels.ops.mamba.gdn_prefill_tmem_dealloc import install

        install()
    except Exception as e:
        logger.warning(f"GDN prefill TMEM dealloc fix (FlashInfer #8854) not applied: {e}")


def _get_flashinfer_gdn_kernels():
    """Lazy import for FlashInfer GDN prefill, decode and verify (MTP) kernels.

    Returns (available, prefill_fn, mtp_fn, decode_fn, mtp_bf16_fn).
    """
    global \
        _flashinfer_gdn_available, \
        _flashinfer_chunk_gated_delta_rule, \
        _flashinfer_gated_delta_rule_mtp, \
        _flashinfer_gated_delta_rule_decode, \
        _flashinfer_gated_delta_rule_mtp_bf16
    if _flashinfer_gdn_available is None:
        try:
            os.environ.setdefault("FLASHINFER_DISABLE_VERSION_CHECK", "1")

            from flashinfer.gdn_decode import (
                gated_delta_rule_decode_pretranspose,
                gated_delta_rule_mtp,
            )
            from flashinfer.gdn_kernels.gdn_decode_bf16_state import (
                gated_delta_rule_mtp as gated_delta_rule_mtp_bf16,
            )
            from flashinfer.gdn_prefill import chunk_gated_delta_rule

            _flashinfer_chunk_gated_delta_rule = chunk_gated_delta_rule
            _flashinfer_gated_delta_rule_mtp = gated_delta_rule_mtp
            _flashinfer_gated_delta_rule_mtp_bf16 = gated_delta_rule_mtp_bf16
            _flashinfer_gated_delta_rule_decode = gated_delta_rule_decode_pretranspose
            _flashinfer_gdn_available = (
                is_cuda() and torch.cuda.get_device_capability()[0] >= 9
            )
            if _flashinfer_gdn_available:
                logger.info("FlashInfer GDN kernels loaded successfully")
                _install_gdn_prefill_tmem_dealloc_fix()
        except (ImportError, RuntimeError) as e:
            logger.warning(f"FlashInfer GDN kernels not available: {e}")
            _flashinfer_gdn_available = False
            _flashinfer_gated_delta_rule_decode = None
    return (
        _flashinfer_gdn_available,
        _flashinfer_chunk_gated_delta_rule,
        _flashinfer_gated_delta_rule_mtp,
        _flashinfer_gated_delta_rule_decode,
        _flashinfer_gated_delta_rule_mtp_bf16,
    )


def is_flashinfer_gdn_prefill_available() -> bool:
    """Return whether the kernel loader can construct the prefill path."""
    available, prefill_fn, *_ = _get_flashinfer_gdn_kernels()
    return bool(available and prefill_fn is not None)


# ---------------------------------------------------------------------------
# SGLANG_ENABLE_GDN_PREFILL_CP_SPLIT: FlashInfer takes its CP kernel only for a
# one-sequence call (sequences x heads x 4 < SMs); its non-CP kernel walks each
# sequence's chunks serially, so a call lasts as long as its longest sequence.
# With the switch on, an eager prefill of several sequences runs its longest
# sequences as one-sequence CP calls (the split extend_in_graph makes with
# tables.cp_single) and the others as one non-CP call per contiguous run.
# Same kernels and state layout; only the grouping of sequences changes.
# ---------------------------------------------------------------------------
GDN_PREFILL_CP_SPLIT = os.environ.get(
    "SGLANG_ENABLE_GDN_PREFILL_CP_SPLIT", "0"
).lower() in ("1", "true")
# Shorter sequences never take a CP call.
GDN_PREFILL_CP_SPLIT_MIN_LEN = int(
    os.environ.get("SGLANG_GDN_PREFILL_CP_SPLIT_MIN_LEN", "4096")
)
# "0": every sequence of at least the minimum length takes a CP call, unmodeled.
GDN_PREFILL_CP_SPLIT_MODEL = (
    os.environ.get("SGLANG_GDN_PREFILL_CP_SPLIT_MODEL", "1") != "0"
)

# Per-layer us = a + b * kilotokens for a one-sequence CP call and for one
# (sequence, head) walk of the non-CP kernel, keyed by whether the pass writes
# state checkpoints. Measured on VR200 (212 SMs), FlashInfer 0.7.2, 48 value
# heads, fp16 state, checkpoints every 64 tokens; re-fit for other hardware.
_CP_SPLIT_FIT = {
    True: ((74.1, 19.66), (10.3, 41.91)),
    False: ((55.4, 14.49), (9.1, 22.66)),
}
# Modeled gain required over the stock call; arbitrary margin for the fit.
_CP_SPLIT_MARGIN = 0.10


def _ncp_call_us(lens, walk_us, *, heads: int, sms: int) -> float:
    """Modeled time of one non-CP call: its CTAs take the (sequence, head) walks
    in order, so a sequence past the first wave of SMs starts late."""
    if not lens:
        return 0.0
    free = [[0.0, sms]]  # [time SMs free up, count], by time
    end = 0.0
    for n in lens:
        dur = walk_us(n)
        need = heads
        started = []
        while need:
            t, count = free[0]
            take = min(count, need)
            need -= take
            if take == count:
                free.pop(0)
            else:
                free[0][1] -= take
            started.append([t + dur, take])
            end = max(end, t + dur)
        free = sorted(free + started)
    # Measured: +5% per further sequence in flight, up to one wave.
    return end * (1 + 0.05 * (min(len(lens), max(sms // heads, 1)) - 1))


def _runs_without(num_seqs: int, cp_seqs) -> list:
    """Contiguous [lo, hi) runs of the sequences not in cp_seqs."""
    runs = []
    for i in range(num_seqs):
        if i in cp_seqs:
            continue
        if runs and runs[-1][1] == i:
            runs[-1][1] = i + 1
        else:
            runs.append([i, i + 1])
    return runs


def _choose_cp_seqs(
    lens, *, checkpoints: bool, heads: int, sms: int, cp_checkpoints=None
):
    """The sequences to run as one-sequence CP calls (empty: the stock call),
    and the modeled stock and split times in us per layer. cp_checkpoints: per
    sequence, whether its CP call stores a checkpoint (default: checkpoints)."""
    min_len = max(GDN_PREFILL_CP_SPLIT_MIN_LEN, 1)
    longest = sorted(
        (i for i, n in enumerate(lens) if n >= min_len), key=lambda i: -lens[i]
    )
    if not GDN_PREFILL_CP_SPLIT_MODEL:
        return frozenset(longest), 0.0, 0.0
    (cp_a, cp_b), (walk_a, walk_b) = _CP_SPLIT_FIT[checkpoints]

    def walk_us(n: int) -> float:
        return walk_a + walk_b * n / 1000

    stock = _ncp_call_us(lens, walk_us, heads=heads, sms=sms)
    best, chosen, cp_us = stock * (1 - _CP_SPLIT_MARGIN), frozenset(), 0.0
    for k, i in enumerate(longest, 1):
        if cp_checkpoints is not None:
            # A CP call storing any checkpoint costs about the every-64 fit (measured).
            cp_a, cp_b = _CP_SPLIT_FIT[bool(cp_checkpoints[i])][0]
        cp_us += cp_a + cp_b * lens[i] / 1000
        if cp_us >= best:
            break
        cp_seqs = frozenset(longest[:k])
        split = cp_us + sum(
            _ncp_call_us(lens[a:b], walk_us, heads=heads, sms=sms)
            for a, b in _runs_without(len(lens), cp_seqs)
        )
        if split < best:
            best, chosen = split, cp_seqs
    return chosen, stock, (best if chosen else stock)


class GDNPrefillCPSplit(msgspec.Struct, frozen=True):
    """Host plan of one eager prefill forward: the stock packed call as calls on
    contiguous sequence ranges, each (use_cp, seq lo, seq hi, token lo, token hi,
    checkpoint lo, checkpoint hi, cu_seqlens, checkpoint cu); the int64 tables
    are device views of one copy."""

    lens: tuple
    # Checkpoint interval the per-sequence checkpoint counts were taken at.
    every: int
    num_checkpoints: int
    calls: tuple
    modeled_stock_us: float
    modeled_split_us: float

    @property
    def num_tokens(self) -> int:
        return sum(self.lens)


def build_gdn_prefill_cp_split(
    extend_seq_lens_cpu,
    *,
    checkpoints: bool,
    heads: int,
    sms: int,
    device,
    cp_checkpoints=None,
) -> Optional[GDNPrefillCPSplit]:
    """The SGLANG_ENABLE_GDN_PREFILL_CP_SPLIT plan of a plain extend batch, or
    None for the stock call (one sequence, which FlashInfer already runs on CP,
    or no split worth its modeled cost)."""
    lens = tuple(int(n) for n in extend_seq_lens_cpu)
    if len(lens) < 2:
        return None
    cp_seqs, stock_us, split_us = _choose_cp_seqs(
        lens,
        checkpoints=checkpoints,
        heads=heads,
        sms=sms,
        cp_checkpoints=cp_checkpoints,
    )
    if not cp_seqs:
        return None
    every = mamba_cache_chunk_size()
    cu = [0]
    ck = [0]
    for n in lens:
        cu.append(cu[-1] + n)
        ck.append(ck[-1] + (n // every if every > 0 else 0))
    ranges = [(i, i + 1, True) for i in cp_seqs]
    ranges += [(a, b, False) for a, b in _runs_without(len(lens), cp_seqs)]
    ranges.sort()
    # Per call: cu_seqlens and checkpoint cu rebased to its first sequence.
    host = []
    for a, b, _ in ranges:
        host += [c - cu[a] for c in cu[a : b + 1]]
        host += [c - ck[a] for c in ck[a : b + 1]]
    table = torch.tensor(
        host, dtype=torch.int64, device="cpu", pin_memory=is_pin_memory_available()
    ).to(device, non_blocking=True)
    calls = []
    o = 0
    for a, b, use_cp in ranges:
        n = b - a + 1
        calls.append(
            (
                use_cp,
                a,
                b,
                cu[a],
                cu[b],
                ck[a],
                ck[b],
                table[o : o + n],
                table[o + n : o + 2 * n],
            )
        )
        o += 2 * n
    return GDNPrefillCPSplit(
        lens=lens,
        every=every,
        num_checkpoints=ck[-1],
        calls=tuple(calls),
        modeled_stock_us=stock_us,
        modeled_split_us=split_us,
    )


def _cp_split_mismatch(plan: GDNPrefillCPSplit, kw: dict) -> Optional[str]:
    """Why the stock call's arguments do not match the plan (host checks only)."""
    num_seqs = len(plan.lens)
    if kw["q"].shape[0] < plan.num_tokens:
        return f"{kw['q'].shape[0]} rows < {plan.num_tokens} planned tokens"
    if kw["cu_seqlens"].shape[0] != num_seqs + 1:
        return f"cu_seqlens of {kw['cu_seqlens'].shape[0]} for {num_seqs} sequences"
    for name in ("initial_state", "output_state"):
        if kw[name] is None or kw[name].shape[0] != num_seqs:
            return f"{name} is not one packed row per sequence"
    every = kw["checkpoint_every_n_tokens"] or 0
    if every:
        if every != plan.every:
            return f"checkpoints every {every} tokens, planned {plan.every}"
        ckpts = kw["state_checkpoints"]
        if ckpts is None or ckpts.shape[0] != plan.num_checkpoints:
            return f"checkpoint rows != {plan.num_checkpoints} planned"
    return None


def _run_prefill_cp_split(prefill_fn, plan: GDNPrefillCPSplit, kw: dict):
    """The stock packed call ``prefill_fn(**kw)`` as the plan's calls on row
    slices of the same tensors; returns (output, output_state) as it does."""
    q, k, v, g, beta = kw["q"], kw["k"], kw["v"], kw["g"], kw["beta"]
    init, out_state = kw["initial_state"], kw["output_state"]
    out = kw["output"]
    if out is None:
        out = q.new_empty((q.shape[0], max(q.shape[1], v.shape[1]), q.shape[2]))
    every = kw["checkpoint_every_n_tokens"] or 0
    ckpts = kw["state_checkpoints"] if every else None
    for use_cp, s0, s1, t0, t1, c0, c1, cu, ck_cu in plan.calls:
        prefill_fn(
            q=q[t0:t1],
            k=k[t0:t1],
            v=v[t0:t1],
            g=g[t0:t1],
            beta=beta[t0:t1],
            scale=kw["scale"],
            initial_state=init[s0:s1],
            output_final_state=True,
            cu_seqlens=cu,
            use_qk_l2norm_in_kernel=kw["use_qk_l2norm_in_kernel"],
            output=out[t0:t1],
            output_state=out_state[s0:s1],
            use_cp=use_cp,
            **({"max_seqlen": t1 - t0} if use_cp else {}),
            **(
                dict(
                    state_checkpoints=ckpts[c0:c1],
                    checkpoint_cu_starts=ck_cu,
                    checkpoint_every_n_tokens=every,
                )
                if ckpts is not None
                else {}
            ),
        )
    return out, out_state


# ---------------------------------------------------------------------------
# SGLANG_ENABLE_GDN_PREFILL_SPARSE_CHECKPOINTS: once any sequence of a prefill
# is tracked off the chunk grid, the stock call stores a state checkpoint every
# N tokens for every sequence, while tracking reads one row per such sequence.
# With the switch on, each call stores only those rows: a one-sequence call
# takes N = its tracked token (one row), and a non-CP call of several sequences
# runs FlashInfer's kernel with per-sequence checkpoint tokens
# (kernels/ops/mamba/gdn_prefill_ckpt_targets.py). Outputs, final states and
# the rows read keep the stock call's bits; only the rows nobody reads go.
# ---------------------------------------------------------------------------
GDN_PREFILL_SPARSE_CHECKPOINTS = os.environ.get(
    "SGLANG_ENABLE_GDN_PREFILL_SPARSE_CHECKPOINTS", "0"
).lower() in ("1", "true")


def gdn_prefill_checkpoint_targets(
    *, track_mask, track_seqlens, prefix_lens, every: int
) -> tuple:
    """Per sequence, the token whose state tracking reads from the checkpoints
    (0: none): maybe_build_flashinfer_checkpoint_plan's row, the last checkpoint
    at or before a tracked offset off the checkpoint grid."""
    targets = []
    for mask, track, prefix in zip(track_mask, track_seqlens, prefix_lens):
        rel = int(track) - int(prefix)
        targets.append((rel // every) * every if mask and rel % every else 0)
    return tuple(targets)


def _sparse_checkpoint_calls(lens, targets, ranges) -> tuple:
    """Host layout for (use_cp, seq lo, seq hi) calls in sequence order: per call
    (mode, row lo, row hi, every, table) with mode "none", "every" (one sequence,
    N = its target, table [0, rows]) or "targets" (several sequences, table the
    codes row * 65536 + chunks, rows relative to the call); the rows tracking
    reads, in tracked-sequence order; the row count."""
    calls = []
    src = []
    row = 0
    for _, a, b in ranges:
        if not any(targets[a:b]):
            calls.append(("none", row, row, 0, ()))
        elif b - a == 1:
            n = lens[a] // targets[a]
            calls.append(("every", row, row + n, targets[a], (0, n)))
            src.append(row)
            row += n
        else:
            codes = []
            local = 0
            for i in range(a, b):
                if targets[i]:
                    codes.append(local * 65536 + targets[i] // 64)
                    src.append(row + local)
                    local += 1
                else:
                    codes.append(0)
            calls.append(("targets", row, row + local, 0, tuple(codes)))
            row += local
    return tuple(calls), tuple(src), row


class GDNPrefillSparseCheckpoints(msgspec.Struct, frozen=True):
    """Host plan of one eager prefill forward's checkpoint rows. calls: (use_cp,
    seq lo, seq hi, token lo, token hi, cu_seqlens, mode, row lo, row hi, every,
    table), with token hi and cu_seqlens None for the stock call over the whole
    batch; src: the rows tracking reads; stock_*: the stock layout, for the
    fallback that gathers those rows from it."""

    lens: tuple
    num_rows: int
    calls: tuple
    src: torch.Tensor
    stock_src: torch.Tensor
    stock_every: int
    stock_num: int
    cp_split: Optional[GDNPrefillCPSplit]

    @property
    def num_tokens(self) -> int:
        return sum(self.lens)


def build_gdn_prefill_sparse_checkpoints(
    lens,
    targets,
    *,
    cp_split: Optional[GDNPrefillCPSplit],
    stock_src: torch.Tensor,
    stock_every: int,
    stock_num: int,
    device,
) -> GDNPrefillSparseCheckpoints:
    """The SGLANG_ENABLE_GDN_PREFILL_SPARSE_CHECKPOINTS plan over the CP split's
    calls, or over one stock call (FlashInfer's own CP choice) without a split."""
    lens = tuple(int(n) for n in lens)
    if cp_split is not None:
        spans = [(c[0], c[1], c[2], c[3], c[4], c[7]) for c in cp_split.calls]
    else:
        spans = [("auto", 0, len(lens), 0, None, None)]
    calls, src, num_rows = _sparse_checkpoint_calls(
        lens, targets, [(s[0], s[1], s[2]) for s in spans]
    )
    host = list(src)
    for _, _, _, _, table in calls:
        host += table
    table = torch.tensor(
        host, dtype=torch.int64, device="cpu", pin_memory=is_pin_memory_available()
    ).to(device, non_blocking=True)
    o = len(src)
    out = []
    for (use_cp, a, b, t0, t1, cu), (mode, r0, r1, every, tab) in zip(spans, calls):
        out.append(
            (use_cp, a, b, t0, t1, cu, mode, r0, r1, every, table[o : o + len(tab)])
        )
        o += len(tab)
    return GDNPrefillSparseCheckpoints(
        lens=lens,
        num_rows=num_rows,
        calls=tuple(out),
        src=table[: len(src)],
        stock_src=stock_src,
        stock_every=stock_every,
        stock_num=stock_num,
        cp_split=cp_split,
    )


def _sparse_checkpoints_mismatch(
    plan: GDNPrefillSparseCheckpoints, kw: dict
) -> Optional[str]:
    """Why the stock call's arguments do not match the plan (host checks only)."""
    num_seqs = len(plan.lens)
    if kw["q"].shape[0] < plan.num_tokens:
        return f"{kw['q'].shape[0]} rows < {plan.num_tokens} planned tokens"
    if kw["cu_seqlens"].shape[0] != num_seqs + 1:
        return f"cu_seqlens of {kw['cu_seqlens'].shape[0]} for {num_seqs} sequences"
    for name in ("initial_state", "output_state"):
        if kw[name] is None or kw[name].shape[0] != num_seqs:
            return f"{name} is not one packed row per sequence"
    ckpts = kw["state_checkpoints"]
    if kw["checkpoint_every_n_tokens"] != plan.stock_every:
        return f"checkpoints every {kw['checkpoint_every_n_tokens']} tokens"
    if ckpts is None or ckpts.shape[0] != plan.num_rows:
        return f"checkpoint rows != {plan.num_rows} planned"
    return None


def _run_prefill_sparse_checkpoints(
    prefill_fn, plan: GDNPrefillSparseCheckpoints, kw: dict
):
    """The stock call ``prefill_fn(**kw)`` with only the plan's checkpoint rows
    stored into ``kw["state_checkpoints"]``; returns (output, output_state)."""
    from sglang.kernels.ops.mamba.gdn_prefill_ckpt_targets import (
        chunk_gated_delta_rule_sm100_checkpoint_targets,
    )

    q, k, v, g, beta = kw["q"], kw["k"], kw["v"], kw["g"], kw["beta"]
    init, out_state = kw["initial_state"], kw["output_state"]
    out = kw["output"]
    if out is None:
        out = q.new_empty((q.shape[0], max(q.shape[1], v.shape[1]), q.shape[2]))
    ckpts = kw["state_checkpoints"]
    for use_cp, s0, s1, t0, t1, cu, mode, r0, r1, every, table in plan.calls:
        tok = slice(t0, t1)  # the stock call passes every row, padding included
        cu = kw["cu_seqlens"] if cu is None else cu
        if mode == "targets":
            # FlashInfer's non-CP path, as chunk_gated_delta_rule dispatches it.
            scale = kw["scale"]
            chunk_gated_delta_rule_sm100_checkpoint_targets(
                q[tok],
                k[tok],
                v[tok],
                g[tok],
                beta[tok],
                out[tok],
                cu,
                init[s0:s1],
                out_state[s0:s1],
                scale if scale else 1.0 / math.sqrt(q.shape[-1]),
                table,
                ckpts[r0:r1],
            )
            continue
        prefill_fn(
            q=q[tok],
            k=k[tok],
            v=v[tok],
            g=g[tok],
            beta=beta[tok],
            scale=kw["scale"],
            initial_state=init[s0:s1],
            output_final_state=True,
            cu_seqlens=cu,
            use_qk_l2norm_in_kernel=kw["use_qk_l2norm_in_kernel"],
            output=out[tok],
            output_state=out_state[s0:s1],
            use_cp=use_cp,
            **({"max_seqlen": t1 - t0} if use_cp is True else {}),
            **(
                dict(
                    state_checkpoints=ckpts[r0:r1],
                    checkpoint_cu_starts=table,
                    checkpoint_every_n_tokens=every,
                )
                if mode == "every"
                else {}
            ),
        )
    return out, out_state


# ---------------------------------------------------------------------------
# Kernel implementation
# ---------------------------------------------------------------------------


class FlashInferGDNKernel(LinearAttnKernelBase):
    """FlashInfer kernel for GDN with K-last SSM state layout.

    SM90 (Hopper): decode uses gather/scatter; prefill and MTP verify supported.
    SM100 (Blackwell): decode uses gather/scatter; prefill and MTP verify supported.

    Requires flashinfer >= 0.6.14.
    """

    uses_state_checkpoints = True
    _cp_split_fallback_warned = False
    _sparse_ckpt_fallback_warned = False

    def __init__(self):
        (
            available,
            self._prefill_fn,
            self._mtp_fn,
            self._decode_fn,
            mtp_bf16_fn,
        ) = _get_flashinfer_gdn_kernels()

        if not available:
            raise RuntimeError(
                "FlashInfer GDN kernels are not available. "
                "Requires SM90+ and FlashInfer with GDN kernel support."
            )
        if self._decode_fn is None:
            raise RuntimeError("FlashInfer GDN decode kernel is unavailable.")

        sm_major = torch.cuda.get_device_capability()[0]
        self.use_state_pool = sm_major >= 10
        # The SM120 chunked-prefill kernel only accepts float32 initial
        # states; SM100 accepts the state-pool dtype directly.
        self._prefill_needs_fp32_state = sm_major >= 12
        self.supports_target_verify = sm_major in (9, 10)
        self._aligned_input_buffers: dict[tuple, torch.Tensor] = {}
        self._aligned_parameter_cache: dict[
            tuple, tuple[torch.Tensor, torch.Tensor]
        ] = {}
        self._verify_intermediate_buffers: dict[tuple, torch.Tensor] = {}
        self._alignment_fallback_warned = False
        # FlashInfer writes through mutable state/workspace pointers, so misaligned
        # inputs must fall back to Triton rather than use temporary copies.
        from sglang.srt.layers.attention.linear.kernels.gdn_triton import (
            TritonGDNKernel,
        )

        self._alignment_fallback_kernel = TritonGDNKernel()

        if sm_major == 9 and self._prefill_fn is None:
            raise RuntimeError("FlashInfer GDN prefill kernel is unavailable.")
        if self._mtp_fn is None:
            raise RuntimeError("FlashInfer GDN MTP (verify) kernel is unavailable.")

        if self.use_state_pool and mtp_bf16_fn is not None:
            # Adapt bf16 kernel to fp32 kernel interface so target_verify needs no branching.
            def _mtp_bf16_adapted(
                q,
                k,
                v,
                initial_state,
                initial_state_indices,
                A_log,
                a,
                dt_bias,
                b,
                use_qk_l2norm=True,
                **kw,
            ):
                out = mtp_bf16_fn(
                    A_log=A_log.float(),
                    a=a,
                    dt_bias=dt_bias,
                    softplus_beta=1.0,
                    softplus_threshold=20.0,
                    q=q,
                    k=k,
                    v=v,
                    b=b,
                    initial_state_source=initial_state,
                    initial_state_indices=initial_state_indices,
                    use_qk_l2norm_in_kernel=use_qk_l2norm,
                    **kw,
                )
                return out, None

            self._mtp_fn = _mtp_bf16_adapted

        logger.info("Using FlashInfer GDN kernels")

    def _prepare_dynamic_input(self, name: str, tensor: torch.Tensor) -> torch.Tensor:
        # Reuse per-stream aligned scratch for uncommon read-only views; stable
        # addresses avoid allocator churn and remain safe for CUDA graph capture.
        if tensor.data_ptr() % _FLASHINFER_GDN_ALIGNMENT == 0:
            return tensor

        stream_key = (
            torch.cuda.current_stream(tensor.device).cuda_stream
            if tensor.device.type == "cuda"
            else None
        )
        key = (
            name,
            tensor.device,
            tensor.dtype,
            tuple(tensor.shape),
            stream_key,
        )
        aligned = self._aligned_input_buffers.get(key)
        if aligned is None:
            aligned = _empty_aligned_like(tensor)
            self._aligned_input_buffers[key] = aligned
        aligned.copy_(tensor)
        return aligned

    def _prepare_parameter(
        self,
        name: str,
        tensor: torch.Tensor,
        *,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        key = (
            name,
            id(tensor),
            dtype,
        )
        cached_entry = self._aligned_parameter_cache.get(key)
        if cached_entry is not None and cached_entry[0] is tensor:
            return cached_entry[1]

        prepared = tensor.detach().reshape(-1)
        if dtype is not None:
            prepared = prepared.to(dtype=dtype, copy=False)
        if (
            not prepared.is_contiguous()
            or prepared.data_ptr() % _FLASHINFER_GDN_ALIGNMENT
        ):
            aligned = _empty_aligned_like(prepared)
            aligned.copy_(prepared)
            prepared = aligned
        # Retaining the source also prevents a recycled Python id from
        # colliding with an older cache entry.
        self._aligned_parameter_cache[key] = (tensor, prepared)
        return prepared

    def _prepare_gate_parameters(
        self,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        *,
        A_log_dtype: Optional[torch.dtype] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            self._prepare_parameter("A_log", A_log, dtype=A_log_dtype),
            self._prepare_parameter("dt_bias", dt_bias),
        )

    def _mutable_inputs_are_aligned(
        self, *named_tensors: tuple[str, Optional[torch.Tensor]]
    ) -> bool:
        for name, tensor in named_tensors:
            if tensor is None or tensor.data_ptr() % _FLASHINFER_GDN_ALIGNMENT == 0:
                continue
            if not self._alignment_fallback_warned:
                logger.warning(
                    "FlashInfer GDN mutable buffer %r has data_ptr %d "
                    "(mod 32 = %d); falling back to Triton for this call.",
                    name,
                    tensor.data_ptr(),
                    tensor.data_ptr() % _FLASHINFER_GDN_ALIGNMENT,
                )
                self._alignment_fallback_warned = True
            return False
        return True

    def _prepare_verify_intermediate_buffer(
        self,
        intermediate_states_buffer: torch.Tensor,
        batch_size: int,
        force_scratch: bool = False,
    ) -> tuple[torch.Tensor, bool]:
        # FlashInfer requires exact capture B, which may exceed the pool-scoped
        # buffer; padded tiers use stable scratch and copy owned rows back.
        if not force_scratch:
            direct = intermediate_states_buffer[:batch_size]
            if direct.shape[0] == batch_size:
                return direct, False

        stream_key = (
            torch.cuda.current_stream(intermediate_states_buffer.device).cuda_stream
            if intermediate_states_buffer.device.type == "cuda"
            else None
        )
        shape = (batch_size, *intermediate_states_buffer.shape[1:])
        key = (
            intermediate_states_buffer.device,
            intermediate_states_buffer.dtype,
            shape,
            stream_key,
        )
        scratch = self._verify_intermediate_buffers.get(key)
        if scratch is None:
            template = intermediate_states_buffer.new_empty(shape)
            scratch = _empty_aligned_like(template)
            self._verify_intermediate_buffers[key] = scratch
        return scratch, True

    # ---- decode ----

    def decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        *,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        if not self._mutable_inputs_are_aligned(("ssm_states", ssm_states)):
            return self._alignment_fallback_kernel.decode(
                q,
                k,
                v,
                a,
                b,
                A_log=A_log,
                dt_bias=dt_bias,
                ssm_states=ssm_states,
                cache_indices=cache_indices,
                query_start_loc=query_start_loc,
                **kwargs,
            )

        batch_size = cache_indices.shape[0]
        num_heads = q.shape[2]
        head_k_dim = q.shape[3]
        num_v_heads = v.shape[2]
        head_v_dim = v.shape[3]

        query_fi = q.view(batch_size, 1, num_heads, head_k_dim)
        key_fi = k.view(batch_size, 1, num_heads, head_k_dim)
        value_fi = v.view(batch_size, 1, num_v_heads, head_v_dim)
        a_fi = a.view(batch_size, 1, num_v_heads)
        b_fi = b.view(batch_size, 1, num_v_heads)
        query_fi = self._prepare_dynamic_input("decode_q", query_fi)
        key_fi = self._prepare_dynamic_input("decode_k", key_fi)
        value_fi = self._prepare_dynamic_input("decode_v", value_fi)
        a_fi = self._prepare_dynamic_input("decode_a", a_fi)
        b_fi = self._prepare_dynamic_input("decode_b", b_fi)
        A_log_fi, dt_bias_fi = self._prepare_gate_parameters(
            A_log,
            dt_bias,
            # Preserve the original backend contract: the SM100 state-pool
            # kernel consumes float32 A_log, while SM90 uses the source dtype.
            A_log_dtype=torch.float32 if self.use_state_pool else None,
        )

        if self.use_state_pool:
            cache_indices_fi = self._prepare_dynamic_input(
                "decode_cache_indices", cache_indices
            )
            output_fi, _ = self._decode_fn(
                q=query_fi,
                k=key_fi,
                v=value_fi,
                state=None,
                A_log=A_log_fi,
                a=a_fi,
                dt_bias=dt_bias_fi,
                b=b_fi,
                use_qk_l2norm=True,
                initial_state=ssm_states,
                initial_state_indices=cache_indices_fi,
            )
        else:
            # TODO: Once FlashInfer PR#2521 is merged for SM90, gather/scatter
            # will no longer be needed here.
            state_batch = ssm_states[cache_indices]
            output_fi, new_state = self._decode_fn(
                q=query_fi,
                k=key_fi,
                v=value_fi,
                state=state_batch,
                A_log=A_log_fi,
                a=a_fi,
                dt_bias=dt_bias_fi,
                b=b_fi,
                scale=None,
                output=None,
                use_qk_l2norm=True,
            )
            ssm_states[cache_indices] = new_state

        return output_fi.view(1, batch_size, num_v_heads, head_v_dim)

    # ---- extend (prefill) ----

    def _run_prefill(
        self,
        cp_split: Optional[GDNPrefillCPSplit],
        sparse_ckpt: Optional[GDNPrefillSparseCheckpoints] = None,
        **kw,
    ):
        """``self._prefill_fn(**kw)``, or its SGLANG_ENABLE_GDN_PREFILL_CP_SPLIT split
        when the backend planned one for this forward and the call matches it."""
        if sparse_ckpt is not None:
            return self._run_prefill_sparse(cp_split, sparse_ckpt, kw)
        if cp_split is None:
            return self._prefill_fn(**kw)
        reason = (
            "needs the SM100 state-pool prefill"
            if not self.use_state_pool or self._prefill_needs_fp32_state
            else _cp_split_mismatch(cp_split, kw)
        )
        if reason is not None:
            if not self._cp_split_fallback_warned:
                self._cp_split_fallback_warned = True
                logger.warning(
                    "SGLANG_ENABLE_GDN_PREFILL_CP_SPLIT: stock call (%s)", reason
                )
            return self._prefill_fn(**kw)
        return _run_prefill_cp_split(self._prefill_fn, cp_split, kw)

    def _run_prefill_sparse(
        self,
        cp_split: Optional[GDNPrefillCPSplit],
        plan: GDNPrefillSparseCheckpoints,
        kw: dict,
    ):
        """SGLANG_ENABLE_GDN_PREFILL_SPARSE_CHECKPOINTS: the call storing only the rows
        tracking reads (plan.src of kw's checkpoint rows, which the backend sized)."""
        reason = (
            "needs the SM100 state-pool prefill"
            if not self.use_state_pool or self._prefill_needs_fp32_state
            else _sparse_checkpoints_mismatch(plan, kw)
        )
        if reason is None:
            return _run_prefill_sparse_checkpoints(self._prefill_fn, plan, kw)
        ckpts = kw["state_checkpoints"]
        if ckpts is None or ckpts.shape[0] != plan.num_rows:
            raise RuntimeError(
                f"SGLANG_ENABLE_GDN_PREFILL_SPARSE_CHECKPOINTS: {reason}; tracking reads "
                "plan rows the call cannot fill"
            )
        if not self._sparse_ckpt_fallback_warned:
            self._sparse_ckpt_fallback_warned = True
            logger.warning(
                "SGLANG_ENABLE_GDN_PREFILL_SPARSE_CHECKPOINTS: stock rows, gathered (%s)",
                reason,
            )
        # The stock layout, then the rows tracking reads gathered into the plan's.
        full = ckpts.new_empty((plan.stock_num, *ckpts.shape[1:]))
        result = self._run_prefill(cp_split, **dict(kw, state_checkpoints=full))
        ckpts.index_copy_(0, plan.src, full.index_select(0, plan.stock_src))
        return result

    def extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        *,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        state_checkpoint_cu_starts: Optional[torch.Tensor] = None,
        num_state_checkpoints: int = 0,
        state_checkpoint_every_n_tokens: int = 0,
        **kwargs,
    ) -> tuple:
        from sglang.kernels.ops.attention.fla.l2norm import (
            gdn_prefill_qkv_prepare_fwd,
        )

        total_seq_len = q.shape[1]
        num_v_heads = v.shape[2]
        head_v_dim = v.shape[3]

        q_fi, k_fi, v_fi = gdn_prefill_qkv_prepare_fwd(q[0], k[0], v[0])

        output = kwargs.get("output")
        output_fi = None
        if output is not None:
            expected_output_shape = (
                1,
                total_seq_len,
                num_v_heads,
                head_v_dim,
            )
            if tuple(output.shape) != expected_output_shape:
                raise ValueError(
                    "FlashInfer GDN prefill output shape mismatch: "
                    f"expected {expected_output_shape}, got {tuple(output.shape)}"
                )
            if output.dtype != v.dtype or output.device != v.device:
                raise ValueError(
                    "FlashInfer GDN prefill output must match v dtype/device, "
                    f"got output=({output.dtype}, {output.device}) and "
                    f"v=({v.dtype}, {v.device})"
                )
            output_fi = output[0]
            if not output_fi.is_contiguous():
                raise ValueError("FlashInfer GDN prefill output must be contiguous")

        # g (alpha) and beta: [1, seq, HV] -> [seq, HV], float32 for FlashInfer
        # g_is_alpha: the gating kernel already wrote exp(g) (SGLANG_GDN_PREFILL_ALPHA_FUSED).
        alpha_fi = (
            g[0] if kwargs.get("g_is_alpha") else torch.exp(g[0].to(torch.float32))
        )
        beta_fi = beta[0].to(torch.float32)

        state_copy = kwargs.get("state_copy")
        if (
            state_copy is not None
            and self.use_state_pool
            and not self._prefill_needs_fp32_state
        ):
            fused = self._extend_fused_state_copy(
                q_fi=q_fi,
                k_fi=k_fi,
                v_fi=v_fi,
                alpha_fi=alpha_fi,
                beta_fi=beta_fi,
                output_fi=output_fi,
                ssm_states=ssm_states,
                state_copy=state_copy,
                state_checkpoint_cu_starts=state_checkpoint_cu_starts,
                num_state_checkpoints=num_state_checkpoints,
                state_checkpoint_every_n_tokens=state_checkpoint_every_n_tokens,
                out_shape=(1, total_seq_len, num_v_heads, head_v_dim),
                cp_split=kwargs.get("cp_split"),
                sparse_ckpt=kwargs.get("sparse_ckpt"),
            )
            if fused is not None:
                return fused[0], None, fused[1]

        if self.use_state_pool:
            # Negative indices (e.g. -1) are padding markers for slots not yet
            # assigned to a real sequence; clamp them to 0 (the reserved dummy
            # slot) so the FlashInfer kernel never reads out-of-bounds state.
            ssm_cache_indices = cache_indices.clamp(min=0).to(torch.int64)
            initial_state_fi = (
                ssm_states[ssm_cache_indices].to(torch.float32)
                if self._prefill_needs_fp32_state
                else ssm_states[ssm_cache_indices].contiguous()
            )
            cu_seqlens = query_start_loc.to(torch.int64)  # kernel requires int64
        else:
            # SM90: preserve original negative-index handling (remap to last slot).
            ssm_cache_indices = torch.where(
                cache_indices >= 0,
                cache_indices,
                ssm_states.shape[0] - 1,
            ).to(torch.int64)
            # State must be float32; kernel requires int64 cu_seqlens.
            initial_state_fi = ssm_states[ssm_cache_indices].to(torch.float32)
            cu_seqlens = query_start_loc.to(torch.int64)

        # Keep final state and checkpoints in the same kernel state dtype.
        output_state_fi = torch.empty_like(initial_state_fi)
        state_checkpoints = (
            initial_state_fi.new_empty(
                (num_state_checkpoints, *initial_state_fi.shape[1:])
            )
            if num_state_checkpoints > 0
            else None
        )
        output_fi, output_state_fi = self._run_prefill(
            kwargs.get("cp_split"),
            sparse_ckpt=kwargs.get("sparse_ckpt"),
            q=q_fi,
            k=k_fi,
            v=v_fi,
            g=alpha_fi,
            beta=beta_fi,
            scale=None,
            initial_state=initial_state_fi,
            output_final_state=True,
            cu_seqlens=cu_seqlens,
            use_qk_l2norm_in_kernel=False,
            output=output_fi,
            output_state=output_state_fi,
            state_checkpoints=state_checkpoints,
            checkpoint_cu_starts=state_checkpoint_cu_starts,
            checkpoint_every_n_tokens=state_checkpoint_every_n_tokens,
        )

        # Write back state to pool
        ssm_states.index_copy_(
            0,
            ssm_cache_indices,
            output_state_fi.to(ssm_states.dtype),
        )

        # Output: [seq, HV, V] -> [1, seq, HV, V]
        core_attn_out = output_fi.view(1, total_seq_len, num_v_heads, head_v_dim)

        # Match Triton's [1, checkpoints, H, V, K] intermediate-state layout.
        h = state_checkpoints.unsqueeze(0) if state_checkpoints is not None else None
        return core_attn_out, None, h

    def _extend_fused_state_copy(
        self,
        *,
        q_fi: torch.Tensor,
        k_fi: torch.Tensor,
        v_fi: torch.Tensor,
        alpha_fi: torch.Tensor,
        beta_fi: torch.Tensor,
        output_fi: Optional[torch.Tensor],
        ssm_states: torch.Tensor,
        state_copy: tuple[torch.Tensor, torch.Tensor],
        state_checkpoint_cu_starts: Optional[torch.Tensor],
        num_state_checkpoints: int,
        state_checkpoint_every_n_tokens: int,
        out_shape: tuple,
        cp_split: Optional[GDNPrefillCPSplit] = None,
        sparse_ckpt: Optional[GDNPrefillSparseCheckpoints] = None,
    ) -> Optional[tuple]:
        """extend's stock FlashInfer call with its pool-row gather and index_copy_ back as one copy
        launch each (SGLANG_ENABLE_GDN_PREFILL_FUSED_STATE_COPY); state_copy: the per-forward int64
        (clamped slots, cu_seqlens) the stock path derives per layer. None: pool rows not dense."""
        from sglang.kernels.ops.mamba.gdn_prefill_state_glue import (
            gdn_gather_state_rows,
            gdn_scatter_state_rows,
        )

        slots, cu_seqlens = state_copy
        initial_state_fi = gdn_gather_state_rows(ssm_states, slots)
        if initial_state_fi is None:
            return None
        output_state_fi = torch.empty_like(initial_state_fi)
        state_checkpoints = (
            initial_state_fi.new_empty((num_state_checkpoints, *initial_state_fi.shape[1:]))
            if num_state_checkpoints > 0
            else None
        )
        output_fi, output_state_fi = self._run_prefill(
            cp_split,
            sparse_ckpt=sparse_ckpt,
            q=q_fi,
            k=k_fi,
            v=v_fi,
            g=alpha_fi,
            beta=beta_fi,
            scale=None,
            initial_state=initial_state_fi,
            output_final_state=True,
            cu_seqlens=cu_seqlens,
            use_qk_l2norm_in_kernel=False,
            output=output_fi,
            output_state=output_state_fi,
            state_checkpoints=state_checkpoints,
            checkpoint_cu_starts=state_checkpoint_cu_starts,
            checkpoint_every_n_tokens=state_checkpoint_every_n_tokens,
        )
        gdn_scatter_state_rows(ssm_states, slots, output_state_fi.to(ssm_states.dtype))
        h = state_checkpoints.unsqueeze(0) if state_checkpoints is not None else None
        return output_fi.view(*out_shape), h

    def extend_in_graph(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        *,
        ssm_states: torch.Tensor,
        tables: GDNFlashInferGraphTables,
        output: torch.Tensor,
        checkpoint_rows: int,
        checkpoint_every_n_tokens: int,
    ) -> Optional[torch.Tensor]:
        """The prefill core of a captured GDN extend on the bucket's static tables:
        the CP kernel takes a one-sequence batch (if ``tables.cp_single``), the
        non-CP kernel any other, and both update the pool in place by slot. ``output`` is [tokens, HV, V];
        returns the [1, checkpoint_rows, HV, V, K] state checkpoints, if any."""
        from sglang.kernels.ops.mamba.gdn_prefill_track import (
            gdn_cp_tail_neutralize,
        )
        from sglang.kernels.ops.attention.fla.l2norm import (
            gdn_prefill_qkv_prepare_fwd,
        )

        q_fi, k_fi, v_fi = gdn_prefill_qkv_prepare_fwd(q[0], k[0], v[0])
        alpha = torch.exp(g[0].to(torch.float32))
        beta_fi = beta[0].to(torch.float32)
        common = dict(
            q=q_fi,
            k=k_fi,
            v=v_fi,
            g=alpha,
            beta=beta_fi,
            scale=None,
            initial_state=ssm_states,
            output_final_state=True,
            use_qk_l2norm_in_kernel=False,
            output=output,
            output_state=ssm_states,
        )
        checkpoints = (
            ssm_states.new_empty((checkpoint_rows, *ssm_states.shape[1:]))
            if checkpoint_rows > 0
            else None
        )

        def checkpoint_args(cu: torch.Tensor) -> dict:
            if checkpoints is None:
                return {}
            return dict(
                state_checkpoints=checkpoints,
                checkpoint_cu_starts=cu,
                checkpoint_every_n_tokens=checkpoint_every_n_tokens,
            )

        if tables.cp_single:
            # Rows past T are bucket padding (stale graph-pool memory, possibly not finite).
            gdn_cp_tail_neutralize(
                q_fi, k_fi, v_fi, alpha, beta_fi, tables.cp_cu_seqlens
            )
            # max_seqlen bounds the CP launch grid; the bucket bounds any sequence.
            self._prefill_fn(
                **common,
                cu_seqlens=tables.cp_cu_seqlens,
                state_indices=tables.cp_state_indices,
                use_cp=True,
                max_seqlen=q_fi.shape[0],
                **checkpoint_args(tables.cp_checkpoint_cu),
            )
        self._prefill_fn(
            **common,
            cu_seqlens=tables.ncp_cu_seqlens,
            state_indices=tables.ncp_state_indices,
            use_cp=False,
            **checkpoint_args(tables.ncp_checkpoint_cu),
        )
        return checkpoints.unsqueeze(0) if checkpoints is not None else None

    # ---- target_verify (MTP) ----

    def target_verify(
        self,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        *,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        intermediate_states_buffer: torch.Tensor,
        intermediate_state_indices: torch.Tensor,
        cache_steps: int,
        retrieve_parent_token: torch.Tensor,
        stable_rows: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        # MTP verify using FlashInfer gated_delta_rule_mtp kernel (SM90 + SM100+).
        if retrieve_parent_token is not None:
            raise RuntimeError(
                "FlashInfer GDN verify kernel only supports topk=1 "
                "(retrieve_parent_token must be None)."
            )

        seq_len = q.shape[1]
        batch_size = query_start_loc.shape[0] - 1
        draft_token_num = seq_len // batch_size

        num_heads = q.shape[2]
        head_k_dim = q.shape[3]
        num_v_heads = v.shape[2]
        head_v_dim = v.shape[3]

        if a is None or b is None or A_log is None or dt_bias is None:
            raise RuntimeError(
                "FlashInfer GDN MTP kernel requires a, b, A_log, dt_bias."
            )

        intermediate_states_buffer_mtp = intermediate_states_buffer
        copy_verify_intermediate_back = False
        if intermediate_states_buffer is not None and (
            self.use_state_pool or stable_rows
        ):
            # FlashInfer writes positional rows; isolate them in exact-B scratch.
            (
                intermediate_states_buffer_mtp,
                copy_verify_intermediate_back,
            ) = self._prepare_verify_intermediate_buffer(
                intermediate_states_buffer,
                batch_size,
                force_scratch=stable_rows,
            )
        if not self._mutable_inputs_are_aligned(
            ("ssm_states", ssm_states),
            ("intermediate_states_buffer", intermediate_states_buffer_mtp),
        ):
            return self._alignment_fallback_kernel.target_verify(
                A_log=A_log,
                dt_bias=dt_bias,
                q=q,
                k=k,
                v=v,
                a=a,
                b=b,
                ssm_states=ssm_states,
                cache_indices=cache_indices,
                query_start_loc=query_start_loc,
                intermediate_states_buffer=intermediate_states_buffer,
                intermediate_state_indices=intermediate_state_indices,
                cache_steps=cache_steps,
                retrieve_parent_token=retrieve_parent_token,
                **kwargs,
            )

        query_mtp = self._prepare_dynamic_input(
            "verify_q",
            q.view(batch_size, draft_token_num, num_heads, head_k_dim),
        )
        key_mtp = self._prepare_dynamic_input(
            "verify_k",
            k.view(batch_size, draft_token_num, num_heads, head_k_dim),
        )
        value_mtp = self._prepare_dynamic_input(
            "verify_v",
            v.view(batch_size, draft_token_num, num_v_heads, head_v_dim),
        )

        a_mtp = self._prepare_dynamic_input(
            "verify_a", a.view(batch_size, draft_token_num, num_v_heads)
        )
        b_mtp = self._prepare_dynamic_input(
            "verify_b", b.view(batch_size, draft_token_num, num_v_heads)
        )
        A_log_fi, dt_bias_fi = self._prepare_gate_parameters(A_log, dt_bias)
        cache_indices_fi = self._prepare_dynamic_input(
            "verify_cache_indices", cache_indices
        )

        output_fi, _ = self._mtp_fn(
            q=query_mtp,
            k=key_mtp,
            v=value_mtp,
            initial_state=ssm_states,
            initial_state_indices=cache_indices_fi,
            A_log=A_log_fi,
            a=a_mtp,
            dt_bias=dt_bias_fi,
            b=b_mtp,
            scale=None,
            output=None,
            intermediate_states_buffer=intermediate_states_buffer_mtp,
            disable_state_update=True,
            use_qk_l2norm=True,
        )

        if stable_rows and intermediate_states_buffer is not None:
            # Persist positional output under PP-stable request rows.
            copy_verify_intermediate_rows(
                intermediate_states_buffer,
                intermediate_states_buffer_mtp,
                intermediate_state_indices[:batch_size],
            )
        elif copy_verify_intermediate_back:
            intermediate_states_buffer.copy_(
                intermediate_states_buffer_mtp[: intermediate_states_buffer.shape[0]]
            )

        return output_fi.view(1, seq_len, num_v_heads, head_v_dim)
