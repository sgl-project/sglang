# SPDX-License-Identifier: Apache-2.0
"""Candidate-pruned DSpark Markov walk and persistent FP32 proposal cache.

The algorithm follows vLLM's DSpark candidate walk (Apache-2.0, contributors to
vLLM). The separate K-only online projection, duplicate ownership, token-keyed
noise, and ordered sparse cache publication here are SGLang implementations.

Weights and static tables must be immutable for this object's lifetime. Callers
must consume corrected_logits before the next sample on the same object; cache
state belongs to physical buffer slots, not request IDs or graph tier IDs.
"""

from __future__ import annotations

import logging
import math
import time
from typing import Optional

import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    # Static table validation can run with CPU Torch.
    triton = None
    tl = None

from sglang.kernels.ops.speculative.dspark.dspark_markov_topk_reference import (
    MarkovCandidateResult,
    validate_mapping,
)

logger = logging.getLogger(__name__)
_FLOAT_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
# Resource bounds, not a claim of a measured speedup. GPU validation and resource
# profiling are required on the serving hardware before performance acceptance.
MAX_WALK_RANK = 1024
MAX_BASE_K = 64
MAX_STATIC_M = 128
MAX_STEPS = 32
_UNCHANGED_MAPPING = object()


def _jit_if_available(function):
    return function if triton is None else triton.jit(function)


class CandidateCapacityError(RuntimeError):
    """Initialization cannot fit the candidate table's minimum workspace."""


def candidate_support_reason(
    w1: torch.Tensor,
    w2: torch.Tensor,
    *,
    topk: int,
    bias_topk: int,
    tp_size: int = 1,
) -> Optional[str]:
    """Return an explicit fast-path fallback reason, or None when supported."""
    if topk == 0:
        return "K=0 selects the existing full-vocabulary path"
    if tp_size != 1:
        return "candidate Markov walk currently requires TP=1"
    if w1.ndim != 2 or w2.ndim != 2 or w1.shape[1] != w2.shape[1]:
        return "Markov weights must be two dense matrices with matching rank"
    if w1.device != w2.device or w1.device.type != "cuda" or torch.version.hip:
        return "candidate Markov walk requires NVIDIA CUDA weights"
    if triton is None:
        return "candidate Markov walk requires Triton"
    if w1.dtype not in _FLOAT_DTYPES or w2.dtype not in _FLOAT_DTYPES:
        return "Markov weights must be ordinary FP16/BF16/FP32 tensors"
    if w1.stride(1) != 1 or w2.stride(1) != 1:
        return "Markov weight rows must have contiguous rank dimensions"
    if w1.stride(0) < w1.shape[1] or w2.stride(0) < w2.shape[1]:
        return "overlapping Markov weight rows are unsupported"
    if not 0 < w1.shape[1] <= MAX_WALK_RANK:
        return f"Markov rank is outside the bounded range 1..{MAX_WALK_RANK}"
    if not 0 < topk <= MAX_BASE_K or not 0 <= bias_topk <= MAX_STATIC_M:
        return f"fused budget bounds are K<= {MAX_BASE_K}, M<= {MAX_STATIC_M}"
    return None


@torch.no_grad()
def build_static_candidates(
    w1: torch.Tensor,
    w2: torch.Tensor,
    bias_topk: int,
    alpha: float,
    *,
    memory_budget_bytes: Optional[int] = None,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Build unscaled FP32 static biases with a bounded chunked GEMM.

    FP32 GEMM uses the process's existing matmul precision policy. The policy is
    recorded, never changed globally. Cached scores are authoritative for static
    candidates even if GEMM and the online FP32 reduction round differently.
    """
    vp, rank = w1.shape
    vd = w2.shape[0]
    if not 0 <= bias_topk <= vd:
        raise ValueError("DSpark static candidate budget must satisfy 0 <= M <= Vd")
    precision = torch.get_float32_matmul_precision()
    allow_tf32 = bool(torch.backends.cuda.matmul.allow_tf32)
    if bias_topk == 0:
        return (
            torch.empty((vp, 0), dtype=torch.int32, device=w1.device),
            torch.empty((vp, 0), dtype=torch.float32, device=w1.device),
            dict(
                build_seconds=0.0,
                temporary_bytes=0,
                precision=precision,
                allow_tf32=allow_tf32,
            ),
        )
    # alpha=0 has tied static rankings. Select the first M draft IDs and avoid
    # converting the other W2 rows, while retaining their true unscaled bias.
    projection_width = bias_topk if alpha == 0 else vd
    converted_w2_bytes = 0 if w2.dtype == torch.float32 else projection_width * rank * 4
    bytes_per_row = (2 * projection_width + rank + 3 * bias_topk) * 4
    minimum_workspace = converted_w2_bytes + bytes_per_row
    table_bytes = vp * bias_topk * 8
    start = time.perf_counter()
    if w1.is_cuda:
        torch.cuda.synchronize(w1.device)
        start = time.perf_counter()
        free_bytes, _ = torch.cuda.mem_get_info(w1.device)
        after_table = free_bytes - table_bytes
        budget = min(converted_w2_bytes + (256 << 20), max(0, after_table // 8))
    else:
        budget = 64 << 20
    if memory_budget_bytes is not None:
        if memory_budget_bytes < 0:
            raise ValueError("DSpark static workspace budget must be nonnegative")
        budget = min(budget, memory_budget_bytes)
    if budget < minimum_workspace:
        raise CandidateCapacityError(
            "DSpark static table needs at least "
            f"{minimum_workspace} workspace bytes ({converted_w2_bytes} for FP32 W2 "
            f"conversion), but its workspace budget is {budget} bytes; "
            f"the persistent table additionally needs {table_bytes} bytes"
        )
    # Validate before allocating either the persistent table or conversion.
    ids = torch.empty((vp, bias_topk), dtype=torch.int32, device=w1.device)
    bias_values = torch.empty((vp, bias_topk), dtype=torch.float32, device=w1.device)
    # Includes the converted W2, chunk input, projection, and a conservative
    # top-k workspace allowance. Never materialize [Vprev,Vd].
    chunk = min(512, (budget - converted_w2_bytes) // bytes_per_row)
    w2t = w2[:projection_width].float().T
    for begin in range(0, vp, chunk):
        stop = min(begin + chunk, vp)
        if alpha == 0:
            chosen = torch.arange(bias_topk, device=w1.device)
            values = w1[begin:stop].float() @ w2t
            ids[begin:stop].copy_(chosen.expand(stop - begin, -1))
        else:
            projection = w1[begin:stop].float() @ w2t
            values, chosen = torch.topk(
                projection, bias_topk, dim=-1, largest=alpha > 0
            )
            ids[begin:stop].copy_(chosen)
            del projection
        bias_values[begin:stop].copy_(values)
        del values, chosen
    if w1.is_cuda:
        torch.cuda.synchronize(w1.device)
    info = dict(
        build_seconds=time.perf_counter() - start,
        temporary_bytes=converted_w2_bytes + chunk * bytes_per_row,
        precision=precision,
        allow_tf32=allow_tf32,
    )
    return ids, bias_values, info


@_jit_if_available
def _token_uniform(seed, token):
    x = token.to(tl.uint32) ^ seed.to(tl.uint32) ^ tl.full((), 0xA511E9B3, tl.uint32)
    x = ((x ^ (x >> 16)) * tl.full((), 0x7FEB352D, tl.uint32)).to(tl.uint32)
    x = ((x ^ (x >> 15)) * tl.full((), 0x846CA68B, tl.uint32)).to(tl.uint32)
    x = x ^ (x >> 16)
    return ((x >> 9).to(tl.float32) + 0.5) * (1.0 / 8388608.0)


@_jit_if_available
def _clear_candidate_cache(
    cache, previous_ids, V: tl.constexpr, C: tl.constexpr, BLOCK_C: tl.constexpr
):
    flat = tl.program_id(0).to(tl.int64)
    col = tl.arange(0, BLOCK_C)
    old = tl.load(previous_ids + flat * C + col, col < C, other=-1)
    tl.store(cache + flat * V + old, float("-inf"), (col < C) & (old >= 0))


@_jit_if_available
def _candidate_walk(
    base,
    top_values,
    top_ids,
    w1,
    w2,
    static_ids,
    static_bias,
    d2t,
    anchors,
    temperatures,
    greedy_mask,
    seeds,
    num_valid,
    tokens,
    predecessors,
    cache,
    previous_ids,
    BASE_STRIDE_B: tl.constexpr,
    BASE_STRIDE_N: tl.constexpr,
    W1_STRIDE: tl.constexpr,
    W2_STRIDE: tl.constexpr,
    V: tl.constexpr,
    N: tl.constexpr,
    R: tl.constexpr,
    K: tl.constexpr,
    M: tl.constexpr,
    ALPHA: tl.constexpr,
    HAS_D2T: tl.constexpr,
    BLOCK_BASE_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    kb = tl.arange(0, BLOCK_BASE_K)
    ms = tl.arange(0, BLOCK_M)
    active = row < tl.load(num_valid)
    # Inactive graph rows must never dereference their stale or sentinel anchor.
    prev = tl.load(anchors + row, active, other=0).to(tl.int64)
    temperature = tl.load(temperatures + row, active, other=1).to(tl.float32)
    greedy = tl.load(greedy_mask + row, active, other=1) != 0
    for step in range(N):
        flat = row * N + step
        output = cache + flat * V
        old = previous_ids + flat * (K + M)
        if active:
            tl.store(predecessors + flat, prev)
            bid = tl.load(top_ids + flat * K + kb, kb < K, other=0).to(tl.int64)
            bbase = tl.load(top_values + flat * K + kb, kb < K, other=0).to(tl.float32)
            if M > 0:
                sid = tl.load(static_ids + prev * M + ms, ms < M, other=-1).to(tl.int64)
                sbias = tl.load(static_bias + prev * M + ms, ms < M, other=0)
                sbase = tl.load(
                    base + row * BASE_STRIDE_B + step * BASE_STRIDE_N + sid,
                    ms < M,
                    other=0,
                ).to(tl.float32)
                # Static ownership guarantees one score and one cache writer per
                # token, including rounding differences between GEMM/reduction.
                duplicate = (
                    tl.sum(
                        ((bid[:, None] == sid[None, :]) & (ms[None, :] < M)).to(
                            tl.int32
                        ),
                        axis=1,
                    )
                    > 0
                )
                bvalid = (kb < K) & ~duplicate
            else:
                bvalid = kb < K
            # Critically, online W2 loads and multiplies are [BLOCK_BASE_K,32],
            # independent of M and any padded final candidate union size.
            bias = tl.full((BLOCK_BASE_K,), 0, tl.float32)
            for rank_start in range(tl.cdiv(R, BLOCK_R)):
                ro = rank_start * BLOCK_R + tl.arange(0, BLOCK_R)
                embed = tl.load(w1 + prev * W1_STRIDE + ro, ro < R, other=0).to(
                    tl.float32
                )
                weight = tl.load(
                    w2 + bid[:, None] * W2_STRIDE + ro[None, :],
                    bvalid[:, None] & (ro[None, :] < R),
                    other=0,
                ).to(tl.float32)
                bias += tl.sum(weight * embed[None, :], axis=1)
            bscore = tl.where(bvalid, bbase + ALPHA * bias, float("-inf"))
            bt = bid
            if HAS_D2T:
                bt += tl.load(d2t + bid, bvalid, other=0)
            bkey = bscore
            seed = tl.load(seeds + flat)
            if not greedy:
                bkey = bscore / temperature - tl.log(-tl.log(_token_uniform(seed, bt)))
            best = tl.max(bkey, axis=0)
            winner = tl.min(tl.where(bvalid & (bkey == best), bt, 2147483647), axis=0)
            finite = tl.max(tl.where(bvalid, bscore, float("-inf")), axis=0) > float(
                "-inf"
            )
            if M > 0:
                st = sid
                if HAS_D2T:
                    st += tl.load(d2t + sid, ms < M, other=0)
                sscore = tl.where(ms < M, sbase + ALPHA * sbias, float("-inf"))
                skey = sscore
                if not greedy:
                    skey = sscore / temperature - tl.log(
                        -tl.log(_token_uniform(seed, st))
                    )
                sbest = tl.max(skey, axis=0)
                swinner = tl.min(
                    tl.where((ms < M) & (skey == sbest), st, 2147483647), axis=0
                )
                winner = tl.where(
                    sbest > best,
                    swinner,
                    tl.where(sbest == best, tl.minimum(winner, swinner), winner),
                )
                finite = finite | (tl.max(sscore, axis=0) > float("-inf"))
            if finite:
                tl.store(output + bt, bscore, bvalid)
                tl.store(old + kb, tl.where(bvalid, bt, -1), kb < K)
                if M > 0:
                    tl.store(output + st, sscore, ms < M)
                    tl.store(old + K + ms, st, ms < M)
            else:
                # All -inf rows have an explicit point mass; downstream dense
                # softmax stays finite. Candidate winner is the smallest ID.
                winner = tl.min(tl.where(bvalid, bt, 2147483647), axis=0)
                if M > 0:
                    winner = tl.minimum(
                        winner, tl.min(tl.where(ms < M, st, 2147483647), axis=0)
                    )
                tl.store(output + winner, 0.0)
                tl.store(old + kb, tl.where(kb == 0, winner, -1), kb < K)
                if M > 0:
                    tl.store(old + K + ms, -1, ms < M)
            tl.store(tokens + flat, winner)
            prev = winner
        else:
            # Padding has a harmless point mass, never an all -inf softmax row.
            tl.store(output, 0.0)
            tl.store(old + kb, tl.where(kb == 0, 0, -1), kb < K)
            if M > 0:
                tl.store(old + K + ms, -1, ms < M)
            tl.store(tokens + flat, 0)
            tl.store(predecessors + flat, 0)


class MarkovCandidateSampler:
    """Persistent proposal storage; construct after weights load, before capture."""

    def __init__(
        self,
        w1: torch.Tensor,
        w2: torch.Tensor,
        *,
        alpha: float,
        topk: int,
        bias_topk: int,
        target_vocab_size: int,
        gamma: int,
        capacity: int,
        d2t_offset: Optional[torch.Tensor] = None,
        topk_backend: str = "auto",
        logits_dtype: Optional[torch.dtype] = None,
    ):
        if not math.isfinite(alpha):
            raise ValueError("DSpark Markov alpha must be finite")
        if w1.ndim != 2 or w2.ndim != 2 or w1.shape[1] != w2.shape[1]:
            raise ValueError(
                "DSpark Markov weights must be matrices with matching rank"
            )
        if w1.shape[0] < target_vocab_size or w1.shape[1] == 0:
            raise ValueError(
                "DSpark W1 must cover all target predecessors with positive rank"
            )
        if not 0 < topk <= w2.shape[0] or not 0 <= bias_topk <= w2.shape[0]:
            raise ValueError("DSpark candidates require 0 < K <= Vd and 0 <= M <= Vd")
        if not 0 < gamma <= MAX_STEPS or capacity <= 0:
            raise ValueError(
                f"DSpark candidate gamma must be 1..{MAX_STEPS}, capacity positive"
            )
        if w1.dtype not in _FLOAT_DTYPES or w2.dtype not in _FLOAT_DTYPES:
            raise ValueError("DSpark candidate weights must be FP16/BF16/FP32")
        if w1.device != w2.device:
            raise ValueError("DSpark candidate weights must be on the same device")
        reason = candidate_support_reason(w1, w2, topk=topk, bias_topk=bias_topk)
        if reason:
            raise ValueError(reason)
        validate_mapping(d2t_offset, w2.shape[0], target_vocab_size)
        self.w1, self.w2 = w1, w2
        self.alpha, self.topk, self.bias_topk = float(alpha), topk, bias_topk
        self.target_vocab_size, self.gamma, self.capacity = (
            target_vocab_size,
            gamma,
            capacity,
        )
        self.d2t_offset = (
            None if d2t_offset is None else d2t_offset.to(w1.device).contiguous()
        )
        self._mapping_snapshot = (
            None if self.d2t_offset is None else self.d2t_offset.detach().cpu().clone()
        )
        self._weight_layouts = tuple(self._weight_layout(weight) for weight in (w1, w2))
        self.path = "triton"
        self.logits_dtype = w1.dtype if logits_dtype is None else logits_dtype
        if self.logits_dtype not in _FLOAT_DTYPES:
            raise ValueError("DSpark base logits must be FP16/BF16/FP32")
        self._flashinfer_top_k = None
        if topk_backend not in ("auto", "torch", "flashinfer"):
            raise ValueError("topk_backend must be auto, torch, or flashinfer")
        if topk_backend != "torch":
            try:
                from flashinfer import top_k
            except ImportError:
                if topk_backend == "flashinfer":
                    raise
            else:
                # Probe before capture: older FlashInfer builds may lack dtype,
                # device, or deterministic support. Runtime never retries inside
                # a graph or silently changes backend during a replay.
                try:
                    probe = torch.zeros(
                        (1, w2.shape[0]), dtype=self.logits_dtype, device=w1.device
                    )
                    top_k(probe, topk, sorted=True, deterministic=True)
                    self._flashinfer_top_k = top_k
                    del probe
                except (RuntimeError, TypeError, NotImplementedError) as exc:
                    if topk_backend == "flashinfer":
                        raise
                    logger.info(
                        "DSpark FlashInfer top-k unavailable: %s; using Torch", exc
                    )
        self.topk_backend = "flashinfer" if self._flashinfer_top_k else "torch"
        self.static_ids, self.static_bias, info = build_static_candidates(
            w1, w2, bias_topk, alpha
        )
        self.build_seconds = info["build_seconds"]
        self.static_temporary_bytes = info["temporary_bytes"]
        self.static_precision = info["precision"]
        self.static_allow_tf32 = info["allow_tf32"]
        shape = (capacity, gamma)
        self.tokens = torch.empty(shape, dtype=torch.int64, device=w1.device)
        self.prev_tokens = torch.empty_like(self.tokens)
        self.corrected_logits = torch.full(
            (*shape, target_vocab_size),
            float("-inf"),
            dtype=torch.float32,
            device=w1.device,
        )
        self.cached_ids = torch.full(
            (*shape, topk + bias_topk), -1, dtype=torch.int32, device=w1.device
        )
        self.seeds = torch.empty(shape, dtype=torch.int64, device=w1.device)
        self._num_valid = torch.zeros((), dtype=torch.int32, device=w1.device)
        self.static_table_bytes = self.static_ids.numel() * 8
        self.storage_bytes = self.static_table_bytes + sum(
            t.numel() * t.element_size()
            for t in (
                self.tokens,
                self.prev_tokens,
                self.corrected_logits,
                self.cached_ids,
                self.seeds,
                self._num_valid,
            )
        )
        self._storage_addresses = self._buffer_addresses()
        self.refresh_count = 0
        self.last_refresh_seconds = 0.0
        logger.info(
            "DSpark candidate sampler path=%s K=%d M=%d Vt=%d Vd=%d R=%d gamma=%d capacity=%d "
            "topk=%s static=%.2f MiB build=%.3fs temporary_estimate=%.2f MiB "
            "fp32_matmul_precision=%s allow_tf32=%s logits_dtype=%s proposal=%.2f MiB persistent_total=%.2f MiB",
            self.path,
            topk,
            bias_topk,
            target_vocab_size,
            w2.shape[0],
            w1.shape[1],
            gamma,
            capacity,
            self.topk_backend,
            self.static_table_bytes / 2**20,
            self.build_seconds,
            self.static_temporary_bytes / 2**20,
            self.static_precision,
            self.static_allow_tf32,
            self.logits_dtype,
            self.corrected_logits.numel() * 4 / 2**20,
            self.storage_bytes / 2**20,
        )

    @staticmethod
    def _weight_layout(weight):
        return tuple(weight.shape), weight.stride(), weight.dtype, weight.device

    def _buffer_addresses(self):
        return {
            name: getattr(self, name).data_ptr()
            for name in (
                "w1",
                "w2",
                "static_ids",
                "static_bias",
                "tokens",
                "prev_tokens",
                "corrected_logits",
                "cached_ids",
                "seeds",
                "_num_valid",
            )
        } | {
            "d2t_offset": (
                None if self.d2t_offset is None else self.d2t_offset.data_ptr()
            )
        }

    @torch.no_grad()
    def refresh_weights(
        self,
        w1: torch.Tensor,
        w2: torch.Tensor,
        *,
        alpha: Optional[float] = None,
        d2t_offset=_UNCHANGED_MAPPING,
    ) -> None:
        """Refresh loaded weights and static values without invalidating graphs.

        The model's weight-update barrier must pause requests/replays throughout
        the reload and this call. A raised exception aborts that reload; callers
        must not resume old graphs after a failed or incompatible weight update.
        New source tensors are accepted with the original layout; their values
        are copied into the original graph-visible weight storage. Changing the
        layout, scale, mapping, or any captured address requires a restart and
        graph recapture. An omitted mapping retains the configured mapping;
        explicit None requests identity and is checked like any other change.
        """
        if torch.cuda.is_current_stream_capturing():
            raise ValueError("DSpark Markov weights cannot be refreshed during capture")
        layouts = tuple(self._weight_layout(weight) for weight in (w1, w2))
        current_layouts = tuple(
            self._weight_layout(weight) for weight in (self.w1, self.w2)
        )
        if layouts != self._weight_layouts or current_layouts != self._weight_layouts:
            raise ValueError(
                "DSpark Markov weight shape/stride/dtype/device changed; "
                "restart the worker and recapture CUDA graphs"
            )
        if self._buffer_addresses() != self._storage_addresses:
            raise ValueError(
                "DSpark candidate storage addresses changed; restart the worker "
                "and recapture CUDA graphs"
            )
        if alpha is not None and float(alpha) != self.alpha:
            raise ValueError(
                "DSpark Markov alpha changed; restart the worker and recapture CUDA graphs"
            )
        mapping = self.d2t_offset if d2t_offset is _UNCHANGED_MAPPING else d2t_offset
        same_mapping = mapping is None and self._mapping_snapshot is None
        if mapping is not None and self._mapping_snapshot is not None:
            same_mapping = (
                mapping.dtype == self.d2t_offset.dtype
                and mapping.shape == self.d2t_offset.shape
                and self.d2t_offset.is_contiguous()
                and torch.equal(mapping.detach().cpu(), self._mapping_snapshot)
                and torch.equal(self.d2t_offset.detach().cpu(), self._mapping_snapshot)
            )
        if not same_mapping:
            raise ValueError(
                "DSpark draft-to-target mapping changed; restart the worker "
                "and recapture CUDA graphs"
            )
        torch.cuda.synchronize(self.w1.device)
        # Build first, so a workspace failure never publishes a partial table.
        # Existing table storage remains allocated and is included in the free
        # memory observation used to budget the temporary replacement table.
        ids, bias, info = build_static_candidates(w1, w2, self.bias_topk, self.alpha)
        if w1.data_ptr() != self.w1.data_ptr():
            self.w1.copy_(w1)
        if w2.data_ptr() != self.w2.data_ptr():
            self.w2.copy_(w2)
        self.static_ids.copy_(ids)
        self.static_bias.copy_(bias)
        torch.cuda.synchronize(self.w1.device)
        if self._buffer_addresses() != self._storage_addresses:
            raise RuntimeError("DSpark refresh unexpectedly replaced captured storage")
        self.refresh_count += 1
        self.last_refresh_seconds = info["build_seconds"]
        self.static_precision = info["precision"]
        self.static_allow_tf32 = info["allow_tf32"]
        self.static_temporary_bytes = info["temporary_bytes"]
        logger.info(
            "DSpark candidate table refreshed count=%d build=%.3fs "
            "graph_buffer_addresses_preserved=True fp32_matmul_precision=%s allow_tf32=%s",
            self.refresh_count,
            self.last_refresh_seconds,
            self.static_precision,
            self.static_allow_tf32,
        )

    @torch.no_grad()
    def prepare_topk(
        self, base_logits: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One batched Top-K across all draft positions, without a softmax."""
        if base_logits.ndim != 3 or base_logits.shape[1:] != (
            self.gamma,
            self.w2.shape[0],
        ):
            raise ValueError("DSpark base logits must have shape [B, gamma, Vd]")
        if (
            base_logits.device != self.w1.device
            or base_logits.dtype not in _FLOAT_DTYPES
        ):
            raise ValueError(
                "DSpark base logits must be floating tensors on the weight device"
            )
        if base_logits.shape[0] > self.capacity or base_logits.stride(-1) != 1:
            raise ValueError(
                "DSpark base logits exceed capacity or have non-contiguous vocabulary"
            )
        if self._flashinfer_top_k is not None:
            if base_logits.dtype != self.logits_dtype:
                raise ValueError(
                    "DSpark FlashInfer Top-K received an unprobed dtype; construct "
                    "the sampler with logits_dtype matching the model's actual logits"
                )
            # FlashInfer requires contiguous rows. This is a necessary copy for
            # padded/cropped LM-head output and belongs in wrapper benchmarks.
            flat = base_logits.reshape(-1, base_logits.shape[-1]).contiguous()
            values, ids = self._flashinfer_top_k(
                flat, self.topk, sorted=True, deterministic=True
            )
            return values.reshape(*base_logits.shape[:2], self.topk), ids.reshape(
                *base_logits.shape[:2], self.topk
            )
        return torch.topk(base_logits, self.topk, dim=-1)

    def sample(
        self,
        base_logits,
        anchor,
        temperatures,
        greedy_mask,
        *,
        seeds=None,
        num_valid=None,
    ) -> MarkovCandidateResult:
        values, ids = self.prepare_topk(base_logits)
        return self.sample_prepared(
            base_logits,
            values,
            ids,
            anchor,
            temperatures,
            greedy_mask,
            seeds=seeds,
            num_valid=num_valid,
        )

    @torch.no_grad()
    def sample_prepared(
        self,
        base_logits,
        values,
        ids,
        anchor,
        temperatures,
        greedy_mask,
        *,
        seeds=None,
        num_valid=None,
    ) -> MarkovCandidateResult:
        bs = base_logits.shape[0]
        if (
            bs > self.capacity
            or base_logits.shape[1:] != (self.gamma, self.w2.shape[0])
            or base_logits.stride(-1) != 1
        ):
            raise ValueError(
                "DSpark prepared base logits have unsupported shape or strides"
            )
        if any(
            t.device != self.w1.device
            for t in (base_logits, values, ids, anchor, temperatures, greedy_mask)
        ):
            raise ValueError("DSpark candidate inputs must be on the weight device")
        if values.shape != (bs, self.gamma, self.topk) or ids.shape != values.shape:
            raise ValueError("DSpark prepared top-k values/IDs have incorrect shapes")
        if not values.is_contiguous() or not ids.is_contiguous():
            raise ValueError("DSpark prepared top-k values/IDs must be contiguous")
        if (
            anchor.shape != (bs,)
            or temperatures.numel() != bs
            or greedy_mask.numel() != bs
        ):
            raise ValueError("DSpark anchor/temperature/greedy staging shape mismatch")
        if any(not t.is_contiguous() for t in (anchor, temperatures, greedy_mask)):
            raise ValueError("DSpark input staging vectors must be contiguous")
        if num_valid is None:
            self._num_valid.fill_(bs)
            num_valid = self._num_valid
        if (
            num_valid.device != self.w1.device
            or num_valid.numel() != 1
            or num_valid.dtype not in (torch.int32, torch.int64)
        ):
            raise ValueError("DSpark num_valid must be an integer device scalar")
        if seeds is None:
            seeds = self.seeds[:bs]
            # Captured Torch RNG operations advance the CUDA generator on every
            # replay. Only B*N seeds are generated; there is no [B,V] noise.
            seeds.random_(0, 2**31)
        elif (
            seeds.shape != (bs, self.gamma)
            or not seeds.is_contiguous()
            or seeds.device != self.w1.device
            or seeds.dtype not in (torch.int32, torch.int64)
        ):
            raise ValueError("DSpark proposal seeds must be contiguous [B, gamma]")
        if bs == 0:
            return MarkovCandidateResult(
                self.tokens[:0], self.corrected_logits[:0], self.prev_tokens[:0]
            )
        _clear_candidate_cache[(bs * self.gamma,)](
            self.corrected_logits,
            self.cached_ids,
            self.target_vocab_size,
            self.topk + self.bias_topk,
            triton.next_power_of_2(self.topk + self.bias_topk),
            num_warps=4,
        )
        _candidate_walk[(bs,)](
            base_logits,
            values,
            ids,
            self.w1,
            self.w2,
            self.static_ids,
            self.static_bias,
            self.d2t_offset if self.d2t_offset is not None else ids,
            anchor,
            temperatures,
            greedy_mask,
            seeds,
            num_valid,
            self.tokens,
            self.prev_tokens,
            self.corrected_logits,
            self.cached_ids,
            BASE_STRIDE_B=base_logits.stride(0),
            BASE_STRIDE_N=base_logits.stride(1),
            W1_STRIDE=self.w1.stride(0),
            W2_STRIDE=self.w2.stride(0),
            V=self.target_vocab_size,
            N=self.gamma,
            R=self.w1.shape[1],
            K=self.topk,
            M=self.bias_topk,
            ALPHA=self.alpha,
            HAS_D2T=self.d2t_offset is not None,
            BLOCK_BASE_K=triton.next_power_of_2(self.topk),
            BLOCK_M=triton.next_power_of_2(max(1, self.bias_topk)),
            BLOCK_R=32,
            num_warps=4,
            enable_fp_fusion=False,
        )
        return MarkovCandidateResult(
            self.tokens[:bs], self.corrected_logits[:bs], self.prev_tokens[:bs]
        )
