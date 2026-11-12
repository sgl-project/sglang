# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
"""K31HT_DRAFT_SITE_EPI=1 (default off = stock): one epilogue kernel per HT DFlash draft-layer site.

Stock chain per site (models/dflash.py DFlashDecoderLayer.forward): the row-parallel projection's all-reduce (CustomAllReduceV2)
-> DFlashGroupedConv.finish (side 1) -> RMSNorm(x, residual) (sgl_kernel fused_add_rmsnorm). The all-reduce stays; the epilogue
kernel (kernels/jit/csrc/distributed/k31ht_draft_ar_site.cuh) replaces conv finish + fused add-RMSNorm with the same arithmetic,
so the normed rows and the updated residual are bit-identical to the chain. K31HT_DRAFT_AR_LEAN=<threads> (with
K31HT_DRAFT_SITE_EPI=1) runs the stock 2shot_push all-reduce kernel with that many threads per CTA (k31ht_draft_ar_lean.cuh) where
the stock dispatch would run 2shot_push on the group's CustomAllReduceV2; elsewhere the stock all-reduce runs and the reason is
logged once.
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from typing import Optional

import torch

from sglang.kernels.jit.utils import cache_once, is_arch_support_pdl, load_jit, make_cpp_args

logger = logging.getLogger(__name__)

HIDDEN = 7168
TAPS = 2
GROUP = 16
_IN_TREE = "distributed/k31ht_draft_ar_site.cuh"
_DECLINED: set = set()
_ENGAGED: set = set()


def k31ht_draft_site_epi_requested() -> bool:
    """K31HT_DRAFT_SITE_EPI=1 (default off = stock): keep the stock all-reduce and replace conv finish + fused add-RMSNorm with
    one epilogue kernel (bit-identical; reads the reduced rows exactly as the stock conv does). K31HT_DRAFT_AR_SITE wins if both."""
    return os.environ.get("K31HT_DRAFT_SITE_EPI", "0").strip() == "1"


def _cuh_path() -> str:
    from sglang.kernels.jit.utils.compile.paths import KERNEL_PATH

    in_tree = KERNEL_PATH / "csrc" / _IN_TREE
    if in_tree.exists():
        return _IN_TREE
    # standalone copies keep the header next to this module
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "k31ht_draft_ar_site.cuh")


@cache_once
def _epi_module():
    args = make_cpp_args(is_arch_support_pdl())
    return load_jit(
        "k31ht_draft_site_epi",
        *args,
        cuda_files=[_cuh_path()],
        cuda_wrappers=[("run", f"k31ht_arsite::DraftSiteEpilogue<{args}>::run")],
        extra_cuda_cflags=["-O3"],
    )


def draft_site_epilogue(
    y: torch.Tensor,
    delta1: torch.Tensor,
    base1: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    block_size: int,
) -> torch.Tensor:
    """conv finish (side 1) + fused add-RMSNorm on the all-reduced rows ``y`` in one kernel: returns the normed rows,
    ``residual`` updated in place (as fused_add_rmsnorm)."""
    out = torch.empty_like(y)
    _epi_module().run(y, delta1, base1, residual, weight, out, float(eps), int(block_size))
    key = ("epi", y.shape[0])
    if key not in _ENGAGED:
        _ENGAGED.add(key)
        logger.info("K31HT_DRAFT_SITE_EPI engaged at %d rows (stock all-reduce + one epilogue kernel)", y.shape[0])
    return out


def decline(reason: str) -> None:
    """Log one line per distinct reason the stock chain runs instead of the epilogue (every decline path logs)."""
    if reason not in _DECLINED:
        _DECLINED.add(reason)
        logger.info("K31HT_DRAFT_SITE_EPI declined (stock chain runs): %s", reason)


def stock_push_algo(ca, x: torch.Tensor) -> Optional[str]:
    """The algorithm CustomAllReduceV2.custom_all_reduce would run for ``x`` now ('1shot_push' / '2shot_push' /
    another name), mirroring its eager-plan / override / config branches."""
    nbytes = x.numel() * x.element_size()
    from sglang.srt.distributed.device_communicators import custom_all_reduce_v2 as carv2

    if ca.override_algo is None and not ca._graph_mode_allowed and not getattr(carv2, "_KERNEL_API_LOG_LEVEL", 0):
        plan = ca._eager_plan(nbytes, x.dtype)
        return None if plan is None else plan[1]
    if ca.override_algo is not None:
        return ca.override_algo.algo_name
    config = ca._pick_config(nbytes, ca._can_use_graph())
    return None if config is None else config.algo.algo_name


def stock_group_for(linear):
    """The GroupCoordinator RowParallelLinear.forward would all-reduce on (same branch order)."""
    from sglang.srt.distributed import get_tp_group
    from sglang.srt.runtime_context import get_parallel

    if getattr(linear, "use_dp_attention_reduce", False):
        return get_parallel().attn_tp_group
    if getattr(linear, "parallel_group", None) == "tp":
        return get_parallel().tp_group
    return get_tp_group()


def site_route(group, x: torch.Tensor) -> Optional[str]:
    """None when the lean launch may replace the stock all-reduce of ``x`` on ``group`` bit for bit; else the decline reason."""
    if group.world_size <= 1:
        return "world_size 1"
    if torch.compiler.is_compiling():
        return "torch.compile tracing"
    if not x.is_cuda or x.dtype != torch.bfloat16 or x.dim() != 2 or x.shape[1] != HIDDEN or not x.is_contiguous():
        return f"input {tuple(x.shape)} {x.dtype}"
    ms = getattr(group, "pymscclpp_comm", None)
    if ms is not None and ms.should_mscclpp_allreduce(x):
        return "pymscclpp route"
    ca = getattr(group, "ca_comm", None)
    if ca is None or ca.disabled or not ca.should_custom_ar(x):
        return "no custom all-reduce for this input"
    if group._can_use_flashinfer_allreduce(x):
        return "flashinfer all-reduce route"
    algo = stock_push_algo(ca, x)
    if algo not in ("1shot_push", "2shot_push"):
        return f"stock algo {algo}"
    rows = x.shape[0]
    need = 2 * rows * (HIDDEN // 8 // group.world_size) * 16
    if not ca.supports_push(nbytes=need, num_blocks=1) or (HIDDEN // 8) % group.world_size:
        return f"push slot too small for {rows} rows"
    return None


@contextmanager
def no_reduce(linear):
    """Run a RowParallelLinear without its all-reduce (the site reduces its partial)."""
    prev = linear.reduce_results
    linear.reduce_results = False
    try:
        yield
    finally:
        linear.reduce_results = prev


def k31ht_draft_ar_lean_threads() -> int:
    """K31HT_DRAFT_AR_LEAN=<threads per CTA> (default unset / 0 = off; acts only with K31HT_DRAFT_SITE_EPI=1): the draft-layer
    all-reduce runs the stock 2shot_push kernel with this many threads per CTA (multiple of 32 in [32, 1024]) on the group's own
    communicator and grid, so it is bit-identical; the stock launcher's 1024-thread CTA takes a whole SM's register file."""
    v = os.environ.get("K31HT_DRAFT_AR_LEAN", "").strip()
    if v in ("", "0"):
        return 0
    n = int(v)
    if n % 32 or not 32 <= n <= 1024:
        raise ValueError(f"K31HT_DRAFT_AR_LEAN must be a multiple of 32 in [32, 1024], got {v!r}")
    return n


_LEAN_IN_TREE = "distributed/k31ht_draft_ar_lean.cuh"


def _lean_cuh_path() -> str:
    from sglang.kernels.jit.utils.compile.paths import KERNEL_PATH

    if (KERNEL_PATH / "csrc" / _LEAN_IN_TREE).exists():
        return _LEAN_IN_TREE
    # standalone copies keep the header (and a copy of the stock custom_all_reduce.cuh it includes) next to this module
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "k31ht_draft_ar_lean.cuh")


@cache_once
def _lean_module(world_size: int):
    args = make_cpp_args(world_size, is_arch_support_pdl())
    return load_jit(
        "k31ht_draft_ar_lean",
        *args,
        cuda_files=[_lean_cuh_path()],
        cuda_wrappers=[("run", f"k31ht_arlean::DraftArLean<{args}>::run")],
        extra_cuda_cflags=["-O3"],
    )


def lean_all_reduce(group, partial: torch.Tensor, threads: int) -> torch.Tensor:
    """The projection's all-reduce of ``partial`` on ``group``: the lean launch of the stock 2shot_push kernel where the stock dispatch
    would run 2shot_push (same kernel, same grid), else the stock all-reduce (logs why once)."""
    reason = site_route(group, partial)
    if reason is None:
        algo = stock_push_algo(group.ca_comm, partial)
        if algo == "2shot_push":
            out = torch.empty_like(partial)
            _lean_module(group.world_size).run(group.ca_comm.obj, partial, out, int(threads))
            key = ("lean", partial.shape[0], int(threads))
            if key not in _ENGAGED:
                _ENGAGED.add(key)
                logger.info(
                    "K31HT_DRAFT_AR_LEAN engaged at %d rows (stock 2shot_push kernel, %d threads per CTA, world %d)",
                    partial.shape[0], int(threads), group.world_size,
                )
            return out
        reason = f"stock algo {algo}"
    if ("lean", reason) not in _DECLINED:
        _DECLINED.add(("lean", reason))
        logger.info("K31HT_DRAFT_AR_LEAN declined (stock all-reduce runs): %s", reason)
    return group.all_reduce(partial)


