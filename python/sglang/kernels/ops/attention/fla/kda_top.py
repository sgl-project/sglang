# Adapted from https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/kda/chunk.py
# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

"""The TOP KDA program: ``chunk_kda`` with the safe gate, the fused intra kernel and one pinned config per autotuned
kernel, so rollout prefill, rollout decode (``kda_rows``) and the trainer's prefill run the same arithmetic.

Selected by ``SGLANG_TOP_KDA_MODE=chunked``, which a TOP engine sets for a KDA family at launch (and Miles' TOP launch
for the trainer); the kernels read it when they are imported (their autotune config lists are built then), as
``SGLANG_TOP_GDN_MODE`` pins the GDN kkt kernel.
"""

import msgspec
import triton

CHUNKED = "chunked"


class TopKDAConfig(msgspec.Struct, frozen=True):
    """Tile sizes and warp counts that set the reduction order of each stage; the row engine replays them."""

    # kda_gate_chunk_cumsum_vector_kernel; 8 warps, as the row engine's prep kernel also runs l2norm_fwd's tile.
    gate_bs: int = 64
    gate_warps: int = 8
    # chunk_kda_fwd_kernel_inter_solve_fused (fused diagonal and recompute).
    intra_bk: int = 64
    intra_bv: int = 64
    intra_warps: int = 4
    # chunk_gla_fwd_kernel_o.
    out_bk: int = 64
    out_bv: int = 64
    out_warps: int = 4
    out_stages: int = 2
    # log2(e) as kda.RCP_LN2 scales the gate cumsum.
    gate_scale: float = 1.4426950216293335


TOP_KDA_CONFIG = TopKDAConfig()


def top_kda_chunked() -> bool:
    from sglang.srt.environ import envs

    mode = envs.SGLANG_TOP_KDA_MODE.get()
    if mode and mode != CHUNKED:
        raise ValueError(f"SGLANG_TOP_KDA_MODE must be '' or {CHUNKED!r}, got {mode!r}")
    return mode == CHUNKED


# The mode the kernels importing this module build their config lists under.
CHUNKED_AT_IMPORT = top_kda_chunked()


def gate_configs(default):
    if top_kda_chunked():
        c = TOP_KDA_CONFIG
        return [triton.Config({"BS": c.gate_bs}, num_warps=c.gate_warps)]
    return default


def intra_configs(default):
    if top_kda_chunked():
        c = TOP_KDA_CONFIG
        return [
            triton.Config({"BK": c.intra_bk, "BV": c.intra_bv}, num_warps=c.intra_warps)
        ]
    return default


def out_configs(default):
    if top_kda_chunked():
        c = TOP_KDA_CONFIG
        return [
            triton.Config(
                {"BK": c.out_bk, "BV": c.out_bv},
                num_warps=c.out_warps,
                num_stages=c.out_stages,
            )
        ]
    return default
