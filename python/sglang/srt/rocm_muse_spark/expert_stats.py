"""Opt-in recorder of the routed experts of Muse Spark decode / target-verify forwards (``SGLANG_MUSE_EXPERT_STATS_PATH``).

The input of the expert placement (``expert_placement``). TP rank 0 keeps, per recorded forward and MoE layer, how
many real rows and how many graph-padding rows each expert slot received (saturating uint8), in a device ring that
one Triton launch per MoE layer writes; the ring slot comes from a device step counter, so the launches are
graph-capturable and nothing is read on the host inside a replay. Eager forwards (prefill, or every forward with
CUDA graphs off) drain the ring to host at most every ``_DRAIN_SECONDS`` and append ``expert_stats_<pid>_<n>.npz``
files to the directory; steps older than the ring when a drain comes are counted as lost.

Prefill (EXTEND / MIXED) forwards, which run eagerly, are recorded separately: per forward and MoE layer the rows
each expert slot receives (uint16; padding rows included), copied to host at the end of the forward and appended
as ``expert_stats_ext_<pid>_<n>.npz`` (``counts`` [forwards, layers, experts], ``tokens``, ``wall``).
"""

from __future__ import annotations

import json
import logging
import os
import time

import numpy as np
import torch
import triton
import triton.language as tl

from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.model_executor.runner_utils import capture_mode

logger = logging.getLogger(__name__)

# Arbitrary: 4096 steps x 62 layers x 256 slots x 2 B = 130 MB of device memory, about 100 s of C32 verify steps.
_RING_STEPS = 4096
_DRAIN_SECONDS = 5.0
_STEPS_PER_FILE = 2048
_WRITE_SECONDS = 60.0
_KIND_DECODE = 1
_KIND_VERIFY = 2
# Prefill forwards buffered on host before an extend file is written; arbitrary.
_EXT_FORWARDS_PER_FILE = 256
# Per-layer counters of the top-8 boundary on real rows: biased-score gap between the 8th and 9th expert in
# units of the 8th's FP32 spacing (exact tie, <= 1, <= 4, <= 64, <= 1024, more), rows whose routed set differs
# from an FP32 torch recompute of the router, rows checked.
MARGIN_BINS = ("tie", "ulp1", "ulp4", "ulp64", "ulp1024", "wide", "set_differs", "rows")


# Row count and layer stay runtime values so one binary per (kind, padding info) serves every launch.
@triton.jit(do_not_specialize=["M", "LAYER"])
def _record_kernel(
    IDS,
    VALID,
    STEP,
    REAL,
    PAD,
    META,
    M,
    LAYER,
    KIND: tl.constexpr,
    HAS_VALID: tl.constexpr,
    L: tl.constexpr,
    S: tl.constexpr,
    E: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    slot = tl.load(STEP) % S
    if HAS_VALID:
        valid = tl.load(VALID).to(tl.int32)
    else:
        valid = M
    e = tl.arange(0, E)
    real = tl.zeros((E,), tl.int32)
    pad = tl.zeros((E,), tl.int32)
    n = M * TOPK
    for c0 in range(0, n, BLOCK):
        s = c0 + tl.arange(0, BLOCK)
        ids = tl.load(IDS + s, mask=s < n, other=-1).to(tl.int32)
        hit = (ids[:, None] == e[None, :]).to(tl.int32)
        is_real = ((s // TOPK) < valid).to(tl.int32)
        real += tl.sum(hit * is_real[:, None], axis=0)
        pad += tl.sum(hit * (1 - is_real)[:, None], axis=0)
    base = (slot * L + LAYER) * E
    tl.store(REAL + base + e, tl.minimum(real, 255).to(tl.uint8))
    tl.store(PAD + base + e, tl.minimum(pad, 255).to(tl.uint8))
    f = tl.arange(0, 4)
    # kind + 4 when the real row count came from the batch (else every row counts as real).
    tag = KIND + 4 * HAS_VALID
    meta = tl.where(f == 0, M, tl.where(f == 1, valid, tl.where(f == 2, tag, LAYER)))
    tl.store(META + slot * 4 + f, meta)


# Row count and layer stay runtime values so one binary per padding mode serves every launch.
@triton.jit(do_not_specialize=["M", "LAYER"])
def _margin_kernel(
    LOGITS,
    BIAS,
    IDS,
    VALID,
    HIST,
    M,
    LAYER,
    MULT,
    HAS_VALID: tl.constexpr,
    E: tl.constexpr,
    NB: tl.constexpr,
):
    row = tl.program_id(0)
    if HAS_VALID:
        valid = tl.load(VALID).to(tl.int32)
    else:
        valid = M
    if row < valid:
        n = tl.arange(0, E)
        # The routing kernels' score sequence: sigmoid(logit * MULT) + bias, top-8 by repeated argmax.
        biased = tl.sigmoid(tl.load(LOGITS + row * E + n) * MULT) + tl.load(BIAS + n)
        chosen = tl.zeros((E,), tl.int32)
        b8 = tl.max(biased, axis=0)
        for _ in tl.static_range(8):
            b8 = tl.max(biased, axis=0)
            hit = n == tl.argmax(biased, axis=0)
            chosen = tl.where(hit, 1, chosen)
            biased = tl.where(hit, -float("inf"), biased)
        gap = b8 - tl.max(biased, axis=0)
        bits = tl.abs(b8).to(tl.int32, bitcast=True)
        spacing = tl.exp2(((bits >> 23) & 255).to(tl.float32) - 150.0)
        r = gap / spacing
        b = tl.where(
            gap == 0,
            0,
            tl.where(
                r <= 1,
                1,
                tl.where(r <= 4, 2, tl.where(r <= 64, 3, tl.where(r <= 1024, 4, 5))),
            ),
        )
        ids = tl.load(IDS + row * 8 + tl.arange(0, 8))
        routed = tl.sum((n[:, None] == ids[None, :]).to(tl.int32), axis=1)
        differs = (tl.sum(tl.abs(routed - chosen), axis=0) > 0).to(tl.int32)
        tl.atomic_add(HIST + LAYER * NB + b, 1)
        tl.atomic_add(HIST + LAYER * NB + 6, differs)
        tl.atomic_add(HIST + LAYER * NB + 7, 1)


class ExpertStatsRecorder:
    """Device ring of per-step expert slot loads; ``begin_forward`` / ``record`` / ``end_forward`` per forward."""

    def __init__(
        self,
        out_dir: str,
        num_layers: int,
        num_experts: int,
        device: torch.device,
        header: dict,
    ):
        self.out_dir = out_dir
        self.num_layers = num_layers
        self.num_experts = num_experts
        self.real = torch.zeros(
            (_RING_STEPS, num_layers, num_experts), dtype=torch.uint8, device=device
        )
        self.pad = torch.zeros_like(self.real)
        self.meta = torch.zeros((_RING_STEPS, 4), dtype=torch.int32, device=device)
        self.step = torch.zeros((1,), dtype=torch.int64, device=device)
        self.margin = torch.zeros(
            (num_layers, len(MARGIN_BINS)), dtype=torch.int32, device=device
        )
        self._kind = 0
        self._valid = None
        self._drained = 0
        self._lost = 0
        self._last_drain = time.monotonic()
        self._last_write = self._last_drain
        self._pending: list[tuple[np.ndarray, ...]] = []
        self._pending_steps = 0
        self._files = 0
        # Prefill: per-layer route counts of the current forward (device), then host buffers.
        self._extend_tokens = 0
        self._ext_counts: dict[int, torch.Tensor] = {}
        self._ext_pending: list[tuple[np.ndarray, int, float]] = []
        self._ext_files = 0
        self._prefix = os.path.join(out_dir, f"expert_stats_{os.getpid()}")
        os.makedirs(out_dir, exist_ok=True)
        with open(self._prefix + "_header.json", "w") as f:
            json.dump(
                {
                    **header,
                    "num_layers": num_layers,
                    "num_experts": num_experts,
                    "ring_steps": _RING_STEPS,
                    "meta_fields": ["rows", "valid_rows", "kind", "last_layer"],
                    "margin_bins": list(MARGIN_BINS),
                    "kinds": {
                        "decode": _KIND_DECODE,
                        "verify": _KIND_VERIFY,
                        "plus_if_valid_rows_known": 4,
                    },
                },
                f,
                indent=1,
            )
        self._compile()
        logger.info(
            "Muse expert stats: recording decode / verify routing to %s", out_dir
        )

    def _compile(self) -> None:
        """JIT every kernel variant now: the first real launches happen under graph capture."""
        dev = self.step.device
        ids = torch.zeros((8, 8), dtype=torch.int32, device=dev)
        valid = torch.zeros((1,), dtype=torch.int32, device=dev)
        x = torch.zeros((8, 64), dtype=torch.bfloat16, device=dev)
        router = torch.zeros((64, self.num_experts), dtype=torch.float32, device=dev)
        bias = torch.zeros((self.num_experts,), dtype=torch.float32, device=dev)
        for kind in (_KIND_DECODE, _KIND_VERIFY):
            for has_valid in (False, True):
                self._kind, self._valid = kind, valid if has_valid else None
                self.record(0, ids, x, router, bias, 1.0)
        self._kind, self._valid = 0, None
        for ring in (self.real, self.pad, self.meta, self.margin):
            ring.zero_()

    def begin_forward(self, forward_batch) -> None:
        mode = forward_batch.forward_mode
        if mode.is_target_verify():
            self._kind = _KIND_VERIFY
        elif mode.is_decode():
            self._kind = _KIND_DECODE
        else:
            self._kind = 0
        if self._kind and not _eager_warmup():
            self._valid = forward_batch.moe_num_token_non_padded()
        else:
            self._kind = 0
        prefill = mode == ForwardMode.EXTEND or mode == ForwardMode.MIXED
        self._extend_tokens = (
            int(forward_batch.extend_num_tokens or 0)
            if prefill and not torch.cuda.is_current_stream_capturing()
            else 0
        )

    def record_extend(self, layer_idx: int, topk_ids: torch.Tensor) -> None:
        """Per-expert routed rows of a prefill MoE layer over all ``topk_ids`` rows (gathered under SP)."""
        if not self._extend_tokens:
            return
        self._ext_counts[layer_idx] = torch.bincount(
            topk_ids.flatten().to(torch.int64), minlength=self.num_experts
        )

    def record(
        self,
        layer_idx: int,
        topk_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        router_weight: torch.Tensor,
        expert_bias: torch.Tensor,
        router_multiplier: float,
    ) -> None:
        """Slot loads of ``topk_ids`` and the top-8 boundary margins of an FP32 recompute of the routing."""
        if not self._kind:
            return
        ids = topk_ids if topk_ids.is_contiguous() else topk_ids.contiguous()
        rows, topk = ids.shape
        logits = torch.mm(hidden_states.float(), router_weight)
        _margin_kernel[(rows,)](
            logits,
            expert_bias,
            ids,
            self._valid if self._valid is not None else self.step,
            self.margin,
            rows,
            layer_idx,
            float(router_multiplier),
            HAS_VALID=self._valid is not None,
            E=self.num_experts,
            NB=len(MARGIN_BINS),
            num_warps=4,
        )
        _record_kernel[(1,)](
            ids,
            self._valid if self._valid is not None else self.step,
            self.step,
            self.real,
            self.pad,
            self.meta,
            rows,
            layer_idx,
            KIND=self._kind,
            HAS_VALID=self._valid is not None,
            L=self.num_layers,
            S=_RING_STEPS,
            E=self.num_experts,
            TOPK=topk,
            BLOCK=32,
            num_warps=4,
        )

    def end_forward(self) -> None:
        if self._extend_tokens and self._ext_counts:
            counts = torch.zeros(
                (self.num_layers, self.num_experts),
                dtype=torch.int64,
                device=self.step.device,
            )
            for layer, c in self._ext_counts.items():
                counts[layer] = c
            self._ext_pending.append(
                (
                    counts.clamp_(max=65535).cpu().numpy().astype(np.uint16),
                    self._extend_tokens,
                    time.time(),
                )
            )
            if len(self._ext_pending) >= _EXT_FORWARDS_PER_FILE:
                self._write_extend()
        self._ext_counts = {}
        self._extend_tokens = 0
        if self._kind:
            self.step.add_(1)
        self._kind = 0
        self._valid = None
        if not torch.cuda.is_current_stream_capturing() and not _eager_warmup():
            self._maybe_drain()

    def _maybe_drain(self) -> None:
        now = time.monotonic()
        if now - self._last_drain < _DRAIN_SECONDS:
            return
        self._last_drain = now
        total = int(self.step.item())
        new = total - self._drained
        if new <= 0:
            return
        keep = min(new, _RING_STEPS)
        self._lost += new - keep
        steps = torch.arange(total - keep, total, device=self.step.device)
        slots = steps % _RING_STEPS
        self._pending.append(
            (
                steps.cpu().numpy(),
                self.real.index_select(0, slots).cpu().numpy(),
                self.pad.index_select(0, slots).cpu().numpy(),
                self.meta.index_select(0, slots).cpu().numpy(),
                np.full(keep, time.time()),
            )
        )
        self._drained = total
        self._pending_steps += keep
        if (
            self._pending_steps >= _STEPS_PER_FILE
            or now - self._last_write >= _WRITE_SECONDS
        ):
            self._write()
            if self._ext_pending:
                self._write_extend()
            self._last_write = now

    def _write_extend(self) -> None:
        counts, tokens, wall = zip(*self._ext_pending)
        path = os.path.join(
            self.out_dir, f"expert_stats_ext_{os.getpid()}_{self._ext_files:05d}.npz"
        )
        np.savez_compressed(
            path, counts=np.stack(counts), tokens=np.array(tokens), wall=np.array(wall)
        )
        logger.info("Muse expert stats: %d prefill forwards -> %s", len(tokens), path)
        self._ext_files += 1
        self._ext_pending = []

    def _write(self) -> None:
        steps, real, pad, meta, wall = (np.concatenate(a) for a in zip(*self._pending))
        path = f"{self._prefix}_{self._files:05d}.npz"
        np.savez_compressed(
            path,
            step=steps,
            real=real,
            pad=pad,
            meta=meta,
            wall=wall,
            lost=self._lost,
            margin=self.margin.cpu().numpy(),
        )
        logger.info(
            "Muse expert stats: %d steps -> %s (%d lost so far)",
            len(steps),
            path,
            self._lost,
        )
        self._files += 1
        self._pending = []
        self._pending_steps = 0


def _eager_warmup() -> bool:
    """Graph-capture warmup runs (eager, dummy batches): not recorded."""
    return capture_mode.is_capture_mode and not torch.cuda.is_current_stream_capturing()


def make_recorder(
    out_dir: str | None,
    *,
    tp_rank: int,
    num_layers: int,
    num_experts: int,
    device: torch.device,
    header: dict,
) -> ExpertStatsRecorder | None:
    """The recorder on TP rank 0 when ``out_dir`` is set; every rank routes the same rows."""
    if not out_dir or tp_rank != 0:
        return None
    return ExpertStatsRecorder(out_dir, num_layers, num_experts, device, header)
