"""K3 MoE front: merged gate + routed_expert_down_proj GEMM, and the fp32 router.

`KimiK3MoE._forward_unfused` -- the path every EP-a2a / WideEP deployment takes --
runs three ops over the same `hidden_states [T, 7168]`:

    router_logits = gate(hidden_states)                     # [896, 7168]  12.85 MB
    topk_output   = topk(hidden_states, router_logits)
    routed_input  = routed_expert_down_proj(hidden_states)   # [3584, 7168] 51.4 MB

Plain [M, 896] fp32 logits go to route_radix (which took fp32 support in #237).
This module covers what that cannot: the merged front.

**fused_front** -- the two GEMMs share their input, so their weights are merged
and one cuBLAS GEMM emits `[T, 896 + 3584]` fp32; a single epilogue kernel then
runs the top-k on the gate slice and casts the latent slice to bf16.  Routing
stays bit-identical to the fp32 path and routed_input comes out dense.

Measured in-graph on a GB300, us per MoE layer, with the copy a dense-input runner
(deep_gemm) needs:

    T            1     16    256   1024   1280   2560   4096  16384
    baseline   22.3   21.9   31.2   61.4   65.9  125.2  185.1  735.8
    merged     13.1   12.0   17.6   47.2   54.8  106.4  181.8  702.8
    radix-only 17.3   17.5   24.2   51.3   55.0  100.7  154.5  621.8

The merge stops paying once the GEMM is compute-bound: it saves one read of
`hidden_states` but costs doubled fp32 output traffic, and past ~1024 tokens the
second outweighs the first.  1024 is where the merged path still wins clearly
(47.2 vs 51.3); 1280-2048 is a wash and 2560 up belongs to the router-only path,
so the threshold sits at the last power of two that is unambiguous.

A bf16-output merged GEMM was measured too (fastest at T=1, 11.8 us).  It is not
used: bf16 rounds the router logits and moves the selected expert set on 2-25% of
rows depending on T, for ~1.2 us -- and from T>=8 it loses to the fp32 variant
anyway, because its routed_input is a strided slice a dense-input runner must copy.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple

import torch

from sglang.kernels.jit.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)

if TYPE_CHECKING:
    from tvm_ffi.module import Module

import json
import os

NUM_EXPERTS = 896
TOPK = 16

# Above this token count the merged GEMM stops paying; see the table above.
MERGED_FRONT_MAX_TOKENS = 1024

_CONFIG_DIR = os.path.join(os.path.dirname(__file__), "configs", "moe_front")

# Kernel tunables, per token bucket, from the JSON table.
#   block_size  threads per CTA; sets experts-per-thread in the radix select
#               (896 / block_size). 224 -> 4, 448 -> 2.
#   cast_vec    fp32 elements each thread converts per step in the latent cast.
#   cast_first  issue the cast before the select (loads in flight during the
#               radix rounds) or after it.
# Fallbacks when no tuned table matches the device. Both were the sweep's most
# common winners: cast_vec 8 is 32 B/thread, the Blackwell vector-load limit, and
# it won at every one of the 28 token counts measured; issuing the cast after the
# select beat issuing it before almost everywhere.
DEFAULT_EPILOGUE_CONFIG = {"block_size": 224, "cast_vec": 8, "cast_first": False}

_tables = {}


def _table(kind: str, device_name: str):
    path = os.path.join(
        _CONFIG_DIR,
        f"{kind},E={NUM_EXPERTS},topk={TOPK},device_name={device_name}.json",
    )
    if path not in _tables:
        table = None
        if os.path.exists(path):
            with open(path) as f:
                table = {int(k): v for k, v in json.load(f)["configs"].items()}
        _tables[path] = table
    return _tables[path]


def _device_name(device) -> str:
    return torch.cuda.get_device_name(device).replace(" ", "_").replace("/", "_")


def get_config(kind: str, num_tokens: int, device, default: dict) -> dict:
    """Tuned config for the nearest token bucket at or below `num_tokens`."""
    table = _table(kind, _device_name(device))
    if not table:
        return dict(default)
    pick = min(table)
    for k in sorted(table):
        if k <= num_tokens:
            pick = k
        else:
            break
    return dict(table[pick])


@cache_once
def _jit_module() -> Module:
    args = make_cpp_args(is_arch_support_pdl())
    return load_jit(
        "moe_front",
        *args,
        cuda_files=["moe/moe_front.cuh"],
        cuda_wrappers=[
            ("front_epilogue", f"FusedFrontEpilogueKernel<{args}>::run"),
        ],
        # No fast-math: scoring and expert-id selection must stay comparable to
        # route_radix / the Triton router under ties and NaN.
        extra_cuda_cflags=["-O3"],
    )


@cache_once
def available() -> bool:
    import logging

    try:
        _jit_module()
        return True
    except Exception as e:  # pragma: no cover - toolchain dependent
        logging.getLogger(__name__).warning(
            f"Failed to load the JIT MoE front kernels: {e}"
        )
        return False


# --------------------------------------------------------------------------
# merged front: [gate | down] GEMM -> top-k + routed_input
# --------------------------------------------------------------------------


def fused_front_covered(
    hidden_states: torch.Tensor,
    merged_weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    topk: int,
    latent: int,
) -> bool:
    """[T<=MERGED_FRONT_MAX_TOKENS, 7168] bf16 x [896 + latent, 7168] bf16, fp32
    bias, top-16, latent a multiple of 4."""
    return (
        hidden_states.dim() == 2
        and merged_weight.dim() == 2
        and hidden_states.dtype == torch.bfloat16
        and merged_weight.dtype == torch.bfloat16
        and bias is not None
        and bias.dtype == torch.float32
        and bias.numel() == NUM_EXPERTS
        and int(topk) == TOPK
        and merged_weight.shape[0] == NUM_EXPERTS + latent
        and merged_weight.shape[1] == hidden_states.shape[1]
        and latent % 4 == 0
        and 0 < hidden_states.shape[0] <= MERGED_FRONT_MAX_TOKENS
        and hidden_states.stride(1) == 1
        and merged_weight.stride(1) == 1
    )


def fused_front_epilogue_only(
    merged: torch.Tensor,
    correction_bias: torch.Tensor,
    latent: int,
    topk: int = TOPK,
    renormalize: bool = True,
    routed_scaling_factor: float = 1.0,
    apply_routed_scaling_factor_on_output: bool = False,
    config: Optional[dict] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The epilogue alone, over an already-computed merged GEMM output.

    Exists so the kernel can be tuned without the cuBLAS GEMM dominating the
    measurement; :func:`fused_front` is the production entry point.
    """
    M = merged.shape[0]
    device = merged.device
    if config is None:
        config = get_config("epilogue", M, device, DEFAULT_EPILOGUE_CONFIG)

    weights = torch.empty((M, topk), dtype=torch.float32, device=device)
    ids = torch.empty((M, topk), dtype=torch.int32, device=device)
    routed = torch.empty((M, latent), dtype=torch.bfloat16, device=device)

    _jit_module().front_epilogue(
        merged,
        correction_bias,
        weights,
        ids,
        routed,
        topk,
        float(routed_scaling_factor if routed_scaling_factor is not None else 1.0),
        bool(renormalize),
        bool(apply_routed_scaling_factor_on_output),
        int(config["block_size"]),
        int(config["cast_vec"]),
        bool(config["cast_first"]),
    )
    return weights, ids, routed


def fused_front(
    hidden_states: torch.Tensor,
    merged_weight: torch.Tensor,
    correction_bias: torch.Tensor,
    latent: int,
    topk: int = TOPK,
    renormalize: bool = True,
    routed_scaling_factor: float = 1.0,
    apply_routed_scaling_factor_on_output: bool = False,
    config: Optional[dict] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Merged front GEMM + fused top-k/cast epilogue.

    Returns ``(topk_weights [M, topk] fp32, topk_ids [M, topk] int32,
    routed_input [M, latent] bf16)``.
    """
    M = hidden_states.shape[0]
    device = hidden_states.device
    if config is None:
        config = get_config("epilogue", M, device, DEFAULT_EPILOGUE_CONFIG)

    # fp32 out keeps routing exact; the extra output traffic versus bf16 is
    # M x (896 + latent) x 2 bytes, negligible against the 64 MB weight read at
    # the sizes this path serves.
    merged = torch.mm(hidden_states, merged_weight.t(), out_dtype=torch.float32)

    weights = torch.empty((M, topk), dtype=torch.float32, device=device)
    ids = torch.empty((M, topk), dtype=torch.int32, device=device)
    routed = torch.empty((M, latent), dtype=torch.bfloat16, device=device)

    _jit_module().front_epilogue(
        merged,
        correction_bias,
        weights,
        ids,
        routed,
        topk,
        float(routed_scaling_factor if routed_scaling_factor is not None else 1.0),
        bool(renormalize),
        bool(apply_routed_scaling_factor_on_output),
        int(config["block_size"]),
        int(config["cast_vec"]),
        bool(config["cast_first"]),
    )
    return weights, ids, routed
