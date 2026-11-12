# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
"""One-shot all-reduce and packed top-k gather over epoch-tagged mailboxes.

The caller owns the mailboxes (symmetric memory, one per process) and the
per-CTA epoch counters; see ``sglang.srt.speculative.dflash_draft_collectives``.
A rank still missing a peer's record after ``deadline_ns`` (> 0) traps: the
CUDA context fails and no output, epoch advance or dependent launch follows.
"""

from __future__ import annotations

from typing import List

import torch
from tvm_ffi import Module

from sglang.kernels.jit.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)

# Threads per CTA of the two kernels (draft_collectives.cuh); the counters hold one entry per CTA.
_AR_THREADS = 64
_GATHER_THREADS = 256
# Gather records keep 24 bits of token id.
MAX_GATHER_TOKEN_ID = (1 << 24) - 1


@cache_once
def _module(num_slots: int, ranks_per_gpu: int) -> Module:
    args = make_cpp_args(num_slots, ranks_per_gpu, is_arch_support_pdl())
    cls = f"draft_collectives::DraftCollectives<{args}>"
    return load_jit(
        "draft_collectives",
        *args,
        cuda_files=["distributed/draft_collectives.cuh"],
        cuda_wrappers=[
            ("all_reduce", f"{cls}::all_reduce"),
            ("topk_gather", f"{cls}::topk_gather"),
        ],
    )


def load_draft_collectives(*, num_slots: int, ranks_per_gpu: int) -> None:
    """JIT-compile or load the module now, so the first call's peers are not left polling while it builds."""
    _module(num_slots, ranks_per_gpu)


def all_reduce_num_ctas(numel: int) -> int:
    return -(-numel // 8 // _AR_THREADS)


def all_reduce_slot_bytes(numel: int) -> int:
    # 24 bytes per 8 BF16 (three tagged records), rounded up to 128 bytes.
    return -(-(numel // 8 * 24) // 128) * 128


def topk_gather_num_ctas(num_slots: int, num_rows: int, k: int) -> int:
    return -(-(num_slots * num_rows * k) // _GATHER_THREADS)


def topk_gather_slot_bytes(num_rows: int, k: int) -> int:
    return -(-(num_rows * k * 8) // 128) * 128


def draft_all_reduce_pk3(
    *,
    inputs: torch.Tensor,
    output: torch.Tensor,
    counters: torch.Tensor,
    peers: List[int],
    gpu: int,
    slot_bytes: int,
    num_slots: int,
    deadline_ns: int,
) -> None:
    """``output`` = rank-order FP32 sum of the ``num_slots`` ranks' BF16 inputs.

    ``inputs`` is ``[ranks_per_gpu, *output.shape]``: the partials of the ranks
    this process hosts (one in serving).
    """
    ranks_per_gpu = num_slots // len(peers)
    _module(num_slots, ranks_per_gpu).all_reduce(
        inputs, output, counters, peers, gpu, slot_bytes, deadline_ns
    )


def draft_topk_gather(
    *,
    vals: torch.Tensor,
    ids: torch.Tensor,
    out_vals: torch.Tensor,
    out_ids: torch.Tensor,
    counters: torch.Tensor,
    peers: List[int],
    gpu: int,
    slot_bytes: int,
    num_slots: int,
    deadline_ns: int,
) -> None:
    """``out_*[row, r * k + j]`` = rank r's ``*[row, j]``: an all-gather along the
    last dim of fp32 values and int64 token ids (each id <= MAX_GATHER_TOKEN_ID)."""
    ranks_per_gpu = num_slots // len(peers)
    _module(num_slots, ranks_per_gpu).topk_gather(
        vals,
        ids,
        out_vals,
        out_ids,
        counters,
        peers,
        gpu,
        slot_bytes,
        deadline_ns,
    )
