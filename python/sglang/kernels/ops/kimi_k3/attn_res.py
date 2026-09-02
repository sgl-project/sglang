"""CUDA JIT wrapper for the Kimi-K3 SM100 attention-residual kernel."""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING, NamedTuple, Optional

import torch

from sglang.kernels.jit.utils import (
    cache_once,
    load_jit,
    make_cpp_args,
    override_jit_cuda_arch,
)
from sglang.kernels.ops.kimi_k3 import comm as k3_comm
from sglang.srt.utils.custom_op import register_custom_op

if TYPE_CHECKING:
    from tvm_ffi.module import Module


_DIM: int = 7168  # K3 hidden size, template parameter of the TMA kernel
_MAX_BANK_ROWS: int = 8  # K3 has <= 8 snapshots, upper bound of the nvb dispatch tables


class FusionDispatch(NamedTuple):
    strategy: str
    max_blocks: int


_CONFIG_DIR = os.path.join(os.path.dirname(__file__), "configs", "attn_res")
_TABLES: dict[str, Optional[dict]] = {}


def _device_name(device: torch.device) -> str:
    return torch.cuda.get_device_name(device).replace(" ", "_").replace("/", "_")


def get_fusion_dispatch(
    kind: str,
    world_size: int,
    hidden_size: int,
    num_tokens: int,
    device: torch.device,
) -> Optional[FusionDispatch]:
    """Return a measured fused strategy, or None for the separate path."""
    path = os.path.join(
        _CONFIG_DIR,
        f"world={world_size},H={hidden_size},device_name={_device_name(device)}.json",
    )
    if path not in _TABLES:
        _TABLES[path] = json.load(open(path)) if os.path.exists(path) else None
    table = _TABLES[path]
    if table is None:
        return None
    raw = table["configs"].get(kind)
    if raw is None:
        return None
    configs = {int(k): v for k, v in raw.items()}
    bucket = min(configs)
    for candidate in sorted(configs):
        if candidate <= num_tokens:
            bucket = candidate
        else:
            break
    config = configs[bucket]
    if config["strategy"] == "separate":
        return None
    return FusionDispatch(config["strategy"], config["max_blocks"])


def _make_name(*args):
    return "kimi_k3_attn_res_" + "_".join(str(a) for a in args)


@cache_once
def _jit_fused_tma_module(
    chunk_rows: int, occupancy: int, consumer_regs: int
) -> Module:
    """Compile and cache the warp-specialized TMA aggregation kernel (per-row
    bulk copies into chunk slots; chunk_rows / occupancy / consumer_regs are
    tuning knobs). The smem ring is frozen at 2 chunk slots and PDL is always
    on: the kernel targets SM100+ except SM12x."""
    major, minor = torch.cuda.get_device_capability()
    if major < 10 or major == 12:
        raise RuntimeError(
            "attn_res_fused_tma requires SM100+ excluding SM12x; "
            f"SM{major}{minor} is unsupported"
        )
    args = make_cpp_args(
        _DIM,
        _MAX_BANK_ROWS,
        chunk_rows,
        occupancy,
        consumer_regs,
    )
    with override_jit_cuda_arch(major, minor, suffix="a"):
        return load_jit(
            _make_name("fused_tma"),
            *args,
            cuda_files=["kimi_k3/attn_res/fused_tma.cuh"],
            cuda_wrappers=[
                ("run", f"AttnResFusedTmaKernel<{args}>::run"),
                ("run_pull_rs", f"AttnResFusedTmaKernel<{args}>::run_pull_rs"),
                ("run_direct_ag", f"AttnResFusedTmaKernel<{args}>::run_direct_ag"),
            ],
            extra_cuda_cflags=["-O3", "--use_fast_math"],
        )


# Benchmarked-best (chunk_rows, occupancy, consumer_regs) per nvb
# (GB200/GB300-class, H=7168). nvb=1 is latency-bound per token, so 2 CTAs/SM
# (occupancy=2, which excludes setmaxnreg: 4*Nc + 2*Np > 512) wins large T by
# ~17%; nvb=4/8 fill 5-row chunks exactly ((nvb+1) % 5 == 0 or covers it in
# 1-2 chunks); everything else is fastest on the balanced 4-row chunk. All
# entries use the setmaxnreg producer/consumer split. consumer_regs sits on
# the 200..232 performance plateau (below 200 the consumer loop starves); we
# take 200, not the 232 budget cap, so each SMSP keeps 2K registers free for
# PDL-overlapped neighbor kernels instead of allocating them idle.
_TMA_BEST_CONFIG: dict[int, tuple[int, int, int]] = {
    1: (2, 2, 0),
    2: (4, 1, 200),
    3: (4, 1, 200),
    4: (5, 1, 200),
    5: (3, 1, 200),
    6: (4, 1, 200),
    7: (4, 1, 200),
    8: (5, 1, 200),
}


def _tuning(nvb: int, num_tokens: int) -> tuple[int, int, int]:
    """(chunk_rows, occupancy, consumer_regs) for this aggregation point.

    Shared by all three entry points below: they run the same kernel template
    and differ only in the collective fused onto it."""
    if not 1 <= nvb <= _MAX_BANK_ROWS:
        raise ValueError(f"attn_res: nvb must be in [1, {_MAX_BANK_ROWS}], got {nvb}")
    best = _TMA_BEST_CONFIG[nvb]
    if best[1] > 1 and num_tokens < 128:
        # occupancy=2 only pays off once there are enough tokens to fill both
        # CTAs per SM; below that its tighter register budget just costs ~10%.
        best = (4, 1, 200)
    return best


@register_custom_op(mutates_args=["out", "prefix_out"])
def _attn_res_fused_pull_rs_op(
    world_size: int,
    input: torch.Tensor,
    residual: torch.Tensor | None,
    bank: torch.Tensor,
    cw: torch.Tensor,
    ow: torch.Tensor,
    out: torch.Tensor,
    prefix_out: torch.Tensor,
    nvb: int,
    eps: float,
    input_mc_ptr: int,
    max_blocks: int,
    chunk_rows: int,
    occupancy: int,
    consumer_regs: int,
) -> None:
    _jit_fused_tma_module(chunk_rows, occupancy, consumer_regs).run_pull_rs(
        k3_comm.get(world_size),
        input,
        residual,
        bank,
        cw,
        ow,
        out,
        prefix_out,
        nvb,
        eps,
        input_mc_ptr,
        max_blocks,
    )


def attn_res_fused_tma(
    prefix_sum: torch.Tensor,
    bank: torch.Tensor,
    cw: torch.Tensor,
    ow: torch.Tensor,
    out: torch.Tensor,
    nvb: int,
    eps: float,
    *,
    write_prefix: bool = False,
) -> None:
    """Warp-specialized TMA aggregation (score -> online softmax -> weighted
    combine -> fused output RMSNorm), one persistent CTA per SM: a producer
    warp (group) fetches rows with one bulk copy each into chunk slots of
    `chunk_rows` rows (one barrier pair per chunk, double-buffered ring of 2
    slots); the 8 consumer warps score and fold one chunk per rendezvous,
    with cw / ow staged in TMEM.

    Restrictions: H == 7168, nvb in [1, 8], SM100a+. The launch config comes
    from _TMA_BEST_CONFIG via _tuning() (each combination compiles its own
    module).

    Parameters
    ----------
    prefix_sum : [T, H] bf16
    bank       : [T, NB, H] bf16 (rows 0..nvb-1 are aggregated)
    cw         : [H] bf16 — precomputed score_norm_weight * proj_weight
    ow         : [H] bf16 — output RMSNorm weight
    out        : [T, H] bf16 output buffer
    nvb        : number of valid bank rows (1..8)
    eps        : RMSNorm epsilon (shared by score and output norms)
    write_prefix : also snapshot the prefix row into bank[:, nvb, :]
                 (bit-exact copy, fused into the score pass which already
                 has the row in registers); requires NB > nvb
    """
    _jit_fused_tma_module(*_tuning(nvb, prefix_sum.shape[0])).run(
        prefix_sum, bank, cw, ow, out, nvb, eps, write_prefix
    )


@register_custom_op(mutates_args=["bank", "out"])
def _attn_res_fused_direct_ag_op(
    world_size: int,
    prefix_sum: torch.Tensor,
    bank: torch.Tensor,
    cw: torch.Tensor,
    ow: torch.Tensor,
    out: torch.Tensor,
    nvb: int,
    eps: float,
    output_mc_ptr: int,
    max_blocks: int,
    write_prefix: bool,
    chunk_rows: int,
    occupancy: int,
    consumer_regs: int,
) -> None:
    _jit_fused_tma_module(chunk_rows, occupancy, consumer_regs).run_direct_ag(
        k3_comm.get(world_size),
        prefix_sum,
        bank,
        cw,
        ow,
        out,
        nvb,
        eps,
        output_mc_ptr,
        max_blocks,
        write_prefix,
    )


def attn_res_fused_direct_ag(
    world_size: int,
    prefix_sum: torch.Tensor,
    bank: torch.Tensor,
    cw: torch.Tensor,
    ow: torch.Tensor,
    out: torch.Tensor,
    nvb: int,
    eps: float,
    *,
    output_mc_ptr: int,
    max_blocks: int = 128,
    write_prefix: bool = False,
) -> torch.Tensor:
    """Fuse local attention-residual aggregation with direct multicast AG.

    `out` is the full symmetric [world * local_tokens, H] output. Each rank
    aggregates its local token shard and multicast-stores normalized vectors
    directly from consumer registers into that rank's slice on every peer.
    """
    _attn_res_fused_direct_ag_op(
        world_size,
        prefix_sum,
        bank,
        cw,
        ow,
        out,
        nvb,
        eps,
        output_mc_ptr,
        max_blocks,
        write_prefix,
        *_tuning(nvb, prefix_sum.shape[0]),
    )
    return out


def attn_res_fused_pull_rs(
    world_size: int,
    input: torch.Tensor,
    residual: torch.Tensor | None,
    bank: torch.Tensor,
    cw: torch.Tensor,
    ow: torch.Tensor,
    out: torch.Tensor,
    prefix_out: torch.Tensor,
    nvb: int,
    eps: float,
    *,
    input_mc_ptr: int,
    max_blocks: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused NVLS pull RS + local residual + TMA attention aggregation.

    `input` is the full TP-partial o_proj tensor in multicast symmetric
    memory. Every rank reduces only its contiguous token shard, optionally
    adds its already-local residual, materializes that prefix in `prefix_out`,
    and feeds it directly to the fused attention-residual aggregation/norm.
    """
    _attn_res_fused_pull_rs_op(
        world_size,
        input,
        residual,
        bank,
        cw,
        ow,
        out,
        prefix_out,
        nvb,
        eps,
        input_mc_ptr,
        max_blocks,
        # prefix_out, not input: the shard this rank actually aggregates.
        *_tuning(nvb, prefix_out.shape[0]),
    )
    return out, prefix_out
