# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
"""FlashInfer Prims-TS sparse MLA for the DSA backend (``prims_ts``).

The kernel reads the raw FP8 MLA cache rows that the ``trtllm`` DSA backend writes
(576 B/token: 512 NoPE + 64 RoPE E4M3, one per-tensor scale) with an E4M3 query
and FP8 QK/PV MMA, and returns BF16 output with a natural-log LSE. It shares the
``trtllm`` route's query preparation, KV store, top-k transforms and DCP slot
localization; only the attention kernel differs.
"""

from __future__ import annotations

import contextlib
import dataclasses
import os
import sys
import logging
from typing import Optional

import torch

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

# Decode/verify graphs plan their exact row count; eager batches above this share
# power-of-two capacity plans, so prefill chunks do not plan one kernel per size.
_EXACT_PLAN_MAX_ROWS = 1024


# --- K31HT_MLA_PROFILE (default off = served) -------------------------------------------
# 'br': exact (non-packed) decode / verify plans with rows >= _K31HT_MLA_BR_MIN_ROWS whose served FP8 schedule is the
# '2cta static, 1 gather warp' profile get the same profile with balanced_registers=True (a shipped schedule option of
# the same kernel; bitwise to served). Every other plan resolves exactly as served.
# Read once at import; any other value raises.
_K31HT_MLA_PROFILE = os.environ.get("K31HT_MLA_PROFILE", "0")
if _K31HT_MLA_PROFILE not in ("0", "br"):
    raise ValueError(f"K31HT_MLA_PROFILE must be '0' or 'br', got {_K31HT_MLA_PROFILE!r}")
_K31HT_MLA_BR_MIN_ROWS = 192
K31HT_MLA_ENGAGED_ROWS: set[int] = set()


@contextlib.contextmanager
def _k31ht_mla_profile(*, rows: int, packed: bool):
    if _K31HT_MLA_PROFILE == "0" or packed or rows < _K31HT_MLA_BR_MIN_ROWS:
        yield
        return
    # Fail closed: a missing policy module or profile type raises here instead of silently serving.
    import flashinfer.attention.prims_ts.kernels.mla_decode.sparse_policy as sp

    served_form = sp.SparseMlaProfile(family="2cta", scheduler="static", gather_issue_warps=1)
    orig = sp.choose_fp8_profile

    def choose(r, heads, query_length, capacity, sm_count, rope_dim):
        prof = orig(r, heads, query_length, capacity, sm_count, rope_dim)
        if prof != served_form:
            return prof
        new = dataclasses.replace(prof, balanced_registers=True)
        if not K31HT_MLA_ENGAGED_ROWS:
            print(f"K31HT_MLA_PROFILE ENGAGE form=br rows_min={_K31HT_MLA_BR_MIN_ROWS} "
                  f"pid={os.getpid()}", file=sys.stderr, flush=True)
        K31HT_MLA_ENGAGED_ROWS.add(rows)
        return new

    sp.choose_fp8_profile = choose
    try:
        yield
    finally:
        sp.choose_fp8_profile = orig


def _plan_capacity(rows: int) -> Optional[int]:
    """None plans exactly ``rows`` fixed rows; otherwise the packed row capacity."""
    if rows <= _EXACT_PLAN_MAX_ROWS:
        return None
    return 1 << (rows - 1).bit_length()


# SM107 (VR200, Rubin) is opt-in: its sparse MLA planning needs FlashInfer with
# flashinfer-ai/flashinfer#8186.
_PRIMS_TS_SM107 = (10, 7)
_logged_capabilities: set[tuple[int, int]] = set()


def prims_ts_supports_capability(capability: tuple[int, int]) -> bool:
    """SM100/SM103, plus SM107 when SGLANG_ENABLE_PRIMS_TS_SM107=1."""
    capability = tuple(capability)
    if capability in ((10, 0), (10, 3)):
        return True
    return capability == _PRIMS_TS_SM107 and envs.SGLANG_ENABLE_PRIMS_TS_SM107.get()


def check_prims_ts_sparse_mla_support(
    *, device: torch.device, num_heads: int, kv_lora_rank: int, qk_rope_head_dim: int
) -> None:
    """Fail at startup, not at the first forward, when the installed FlashInfer
    cannot plan the D576 native-E4M3 sparse MLA this backend needs."""
    if (kv_lora_rank, qk_rope_head_dim) != (512, 64):
        raise ValueError(
            "--dsa-*-backend prims_ts supports D576 MLA "
            f"(kv_lora_rank 512, qk_rope_head_dim 64); got {kv_lora_rank}, "
            f"{qk_rope_head_dim}."
        )
    capability = tuple(torch.cuda.get_device_capability(device))
    if not prims_ts_supports_capability(capability):
        raise ValueError(
            "Prims-TS sparse MLA requires SM100 or SM103 (SM107 with "
            "SGLANG_ENABLE_PRIMS_TS_SM107=1)."
        )
    try:
        from flashinfer.attention.prims_ts import (
            get_prims_ts_sparse_mla_decode_workspace_size,
        )

        get_prims_ts_sparse_mla_decode_workspace_size(
            device,
            1,
            num_heads,
            max_topk=2048,
            q_dtype=torch.float8_e4m3fn,
            qk_rope_head_dim=qk_rope_head_dim,
            kv_cache_format="native",
            return_lse=True,
        )
    except (ImportError, TypeError, ValueError, NotImplementedError) as exc:
        needs = "#7410 and #7984"
        if capability == _PRIMS_TS_SM107:
            needs += ", and #8186 for SM107"
        raise RuntimeError(
            "--dsa-*-backend prims_ts needs FlashInfer Prims-TS with D576 native-FP8 "
            f"sparse MLA (flashinfer-ai/flashinfer {needs}); "
            f"device capability {capability}: {exc}"
        ) from exc
    if capability == _PRIMS_TS_SM107 and capability not in _logged_capabilities:
        _logged_capabilities.add(capability)
        logger.info(
            "prims_ts sparse MLA enabled on SM107 (%d heads; "
            "SGLANG_ENABLE_PRIMS_TS_SM107=1)",
            num_heads,
        )


class PrimsTSSparseMLA:
    """Planned Prims-TS sparse MLA launches, keyed by row plan, heads and top-k width.

    Plans compile and allocate, so they are made on eager calls; the CUDA graph
    runner's eager warmups plan every captured row count before its capture.
    """

    def __init__(self, device: torch.device):
        # Plans compare tensor devices with this one, so it carries the index.
        if device.index is None:
            device = torch.device(device.type, torch.cuda.current_device())
        self.device = device
        self._plans: dict[tuple[int, int, int, bool], object] = {}
        # packed plans' [0, rows] query offsets, keyed like _plans
        self._qo_indptr: dict[tuple[int, int, int, bool], torch.Tensor] = {}

    def _wrapper(self, *, rows: int, heads: int, width: int):
        from flashinfer.attention.prims_ts import BatchSparseMLADecodePagedTSWrapper

        capacity = _plan_capacity(rows)
        packed = capacity is not None
        key = (capacity if packed else rows, heads, width, packed)
        wrapper = self._plans.get(key)
        if wrapper is not None:
            return wrapper, self._qo_indptr.get(key)
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                f"Prims-TS sparse MLA has no plan for {rows} rows x {heads} heads; "
                "graph capture must follow an eager run of the same shape."
            )
        wrapper = BatchSparseMLADecodePagedTSWrapper()
        if _K31HT_MLA_PROFILE != "0":
            with _k31ht_mla_profile(rows=rows, packed=packed):
                wrapper.plan(
                    self.device,
                    1 if packed else rows,
                    heads,
                    max_topk=width,
                    max_q_len=capacity if packed else 1,
                    packed_query=packed,
                    q_dtype=torch.float8_e4m3fn,
                    qk_rope_head_dim=64,
                    kv_cache_format="native",
                    return_lse=True,
                )
        else:
            wrapper.plan(
                self.device,
                1 if packed else rows,
                heads,
                max_topk=width,
                max_q_len=capacity if packed else 1,
                packed_query=packed,
                q_dtype=torch.float8_e4m3fn,
                qk_rope_head_dim=64,
                kv_cache_format="native",
                return_lse=True,
            )
        self._plans[key] = wrapper
        if packed:
            self._qo_indptr[key] = torch.zeros(2, dtype=torch.int32, device=self.device)
        return wrapper, self._qo_indptr.get(key)

    def forward(
        self,
        *,
        q: torch.Tensor,
        kv_cache: torch.Tensor,
        indices: torch.Tensor,
        lengths: torch.Tensor,
        sm_scale: float,
        kv_scale: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """q [rows, H, 576] E4M3; kv_cache raw rows viewable as [N, 576] E4M3;
        indices [rows, K] int32 cache rows (-1 holes); lengths [rows] int32 <= K.
        Returns BF16 [rows, H, 512] and natural-log LSE [rows, H] (-inf, zero
        output for a row with no keys)."""
        rows, heads, head_dim = q.shape
        out = torch.empty((rows, heads, 512), dtype=torch.bfloat16, device=q.device)
        lse = torch.empty((rows, heads), dtype=torch.float32, device=q.device)
        if rows == 0:
            return out, lse
        width = indices.shape[1]
        wrapper, qo_indptr = self._wrapper(rows=rows, heads=heads, width=width)
        kv_rows = kv_cache.view(torch.float8_e4m3fn).view(-1, head_dim)
        # The per-tensor KV descale multiplies both the QK logits and the
        # weighted V sum; the query is cast unscaled.
        kwargs = dict(
            softmax_scale=sm_scale * kv_scale,
            output_scale=kv_scale,
            validate=False,
        )
        if qo_indptr is not None:
            qo_indptr[1:].fill_(rows)
            wrapper.run(
                q,
                kv_rows,
                indices,
                lengths,
                qo_indptr=qo_indptr,
                out=out,
                lse=lse,
                **kwargs,
            )
        else:
            wrapper.run(
                q.view(rows, 1, heads, head_dim),
                kv_rows,
                indices,
                lengths,
                out=out.view(rows, 1, heads, 512),
                lse=lse.view(rows, 1, heads),
                **kwargs,
            )
        return out, lse
