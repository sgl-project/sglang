# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
"""The DFlash2 selector tail in one launch (SGLANG_OPT_DFLASH_SELECTOR_ONE_LAUNCH, default off).

``dflash_selector_one_launch`` replaces FusedSelectorHead's three launches after the candidate gather
(``dflash_selector_select``, the lattice ``torch.einsum`` and ``selector_walk_fused``) with one kernel
(jit/csrc/speculative/dflash_selector_one_launch.cuh). candidates, tokens and q_rows are bitwise equal to that chain;
the unary logits, the lattice operands and the lattice itself never leave the SM.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.kernels.jit.utils import cache_once, load_jit, make_cpp_args

if TYPE_CHECKING:
    from tvm_ffi.module import Module

# One warp per draft slot: the served drafts block_size - 1 = 7 slots; 8 is the CTA's warp count.
ONE_LAUNCH_MAX_SLOTS = 8


@cache_once
def _one_launch_module(width: int, top_k: int, rank: int, id_dtype: torch.dtype) -> Module:
    args = make_cpp_args(width, top_k, rank, id_dtype, ONE_LAUNCH_MAX_SLOTS)
    return load_jit(
        "dflash_selector_one_launch",
        *args,
        cuda_files=["speculative/dflash_selector_one_launch.cuh"],
        cuda_wrappers=[("run", f"DFlashSelectorOneLaunch<{args}>::run")],
    )


def one_launch_supported(*, top_k: int, rank: int, slots: int) -> bool:
    """The kernel's compile-time form: top_k 16 (torch's 32-slot sort), rank a multiple of 64, <= 8 slots."""
    return top_k == 16 and rank % 64 == 0 and 1 <= slots <= ONE_LAUNCH_MAX_SLOTS


def dflash_selector_one_launch(
    *,
    gathered_vals: torch.Tensor,
    gathered_ids: torch.Tensor,
    hidden: torch.Tensor,
    anchors: torch.Tensor,
    pred_table: torch.Tensor,
    succ_table: torch.Tensor,
    uniforms: torch.Tensor,
    temperatures: torch.Tensor,
    greedy_mask: torch.Tensor,
    candidates: torch.Tensor,
    tokens: torch.Tensor,
    q_rows: torch.Tensor,
    lattice: Optional[torch.Tensor] = None,
) -> None:
    """gathered_vals / gathered_ids [bs * slots, width], hidden [bs, slots, rank] (the projected hidden states),
    anchors [bs] (any stride), uniforms [bs, slots], temperatures [bs] fp32, greedy_mask [bs] bool or uint8 ->
    candidates [bs * slots, top_k] int64, tokens [bs * slots] int64, q_rows [bs, slots, top_k] fp32, in place.
    lattice (tests only): [bs, slots, top_k, top_k] bf16 copy of the on-chip lattice."""
    _, top_k = candidates.shape
    rank = hidden.shape[-1]
    if greedy_mask.dtype == torch.bool:
        greedy_mask = greedy_mask.view(torch.uint8)
    module = _one_launch_module(gathered_vals.shape[1], top_k, rank, gathered_ids.dtype)
    module.run(
        gathered_vals,
        gathered_ids,
        hidden,
        anchors,
        pred_table,
        succ_table,
        uniforms,
        temperatures,
        greedy_mask,
        candidates,
        tokens,
        q_rows,
        q_rows if lattice is None else lattice,
        0 if lattice is None else 1,
    )
