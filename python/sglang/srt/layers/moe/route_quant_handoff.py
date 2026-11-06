"""Attempt-and-verify handoff for the fused K3 MoE-front prep launch.

At decode batch sizes the chain between the K3 fused-front GEMM and the
trtllm-gen SiTU MoE op is three tiny back-to-back kernels on the critical
path — route_radix (top-16 of 896), the triton ``(id << 16) | bf16(weight)``
pack, and per_token_group_quant (mxfp8, ``[T, 3584]``) — about 7.5 us busy
plus two extra launches per MoE layer, each leaving the SMs near idle. The
fused kernel (kernels/ops/moe/moe_route_quant_fused.py) runs all three in one
launch: routing CTAs and one quant CTA per token, concurrently.

The inputs live in different modules (router logits reach the router through
TopK, activations through the MoE runner), so the fusion is wired as a
consume-once stash instead of new signatures:

    KimiK3MoE._forward_routed*      stage(x) before self.topk, clear() after
                                    the experts call
    biased_grouped_topk_gpu         try_route_quant_fused() replaces the
                                    moe_fused_gate call on a hit
    Mxfp4MoEMethod.apply (situ)     take(x) skips the quant and the pack

Every step is fallback-safe: if the routing dispatch never consumes the staged
request (different model, uncovered shape, triton fallback) or the runner's
``take`` misses (activations re-viewed or copied), the unfused chain runs as
before. The stage/clear bracket in the model layer guarantees a published
entry can never leak into another layer whose allocator reused the same
activation address.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Optional, Tuple

import msgspec
import torch

logger = logging.getLogger(__name__)


class _Handoff(msgspec.Struct):
    # staged by the model layer: the activation rows the runner will quantize
    request_x: Optional[torch.Tensor] = None
    # published by the routing dispatch, keyed by the staged activations
    produced_x: Optional[torch.Tensor] = None
    packed: Optional[torch.Tensor] = None
    x_q: Optional[torch.Tensor] = None
    x_s: Optional[torch.Tensor] = None
    # HIP route+MXFP4 publish. Its own produced pointer, so a CUDA take()
    # cannot observe or clear it.
    hip_produced: Optional[torch.Tensor] = None
    hip_fp4: Optional[torch.Tensor] = None
    hip_scale: Optional[torch.Tensor] = None
    moe_inter_dim: Optional[int] = None
    num_experts: Optional[int] = None


_handoff = _Handoff()


def stage(
    x: torch.Tensor,
    moe_inter_dim: Optional[int] = None,
    num_experts: Optional[int] = None,
) -> None:
    """Publish the routed activations for the upcoming topk call. Caller pairs
    this with clear() after the experts call (try/finally)."""
    _handoff.request_x = x
    _handoff.produced_x = None
    _handoff.hip_produced = None
    _handoff.hip_fp4 = None
    _handoff.hip_scale = None
    _handoff.moe_inter_dim = moe_inter_dim
    _handoff.num_experts = num_experts


def clear() -> None:
    _handoff.request_x = None
    _handoff.produced_x = None
    _handoff.packed = None
    _handoff.x_q = None
    _handoff.x_s = None
    _handoff.hip_produced = None
    _handoff.hip_fp4 = None
    _handoff.hip_scale = None
    _handoff.moe_inter_dim = None
    _handoff.num_experts = None


def staged_activation() -> Optional[torch.Tensor]:
    """The activation row the model staged for this topk, or None."""
    return _handoff.request_x


def staged_moe_shape() -> Tuple[Optional[int], Optional[int]]:
    """(intermediate size per rank, expert count) staged with the activation."""
    return _handoff.moe_inter_dim, _handoff.num_experts


def publish_hip_mxfp4(x: torch.Tensor, fp4: torch.Tensor, scale: torch.Tensor) -> None:
    """Publish token-order MXFP4 for the staged rows. Consume-once."""
    _handoff.request_x = None
    _handoff.hip_produced = x
    _handoff.hip_fp4 = fp4
    _handoff.hip_scale = scale


def take_hip_mxfp4(
    x: torch.Tensor,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """Consume the published (fp4 uint8, e8m0 uint8) for these exact rows."""
    produced = _handoff.hip_produced
    if produced is None or _handoff.hip_fp4 is None or _handoff.hip_scale is None:
        return None
    if (
        produced.data_ptr() != x.data_ptr()
        or produced.shape != x.shape
        or produced.dtype != x.dtype
        or produced.stride() != x.stride()
    ):
        return None
    out = (_handoff.hip_fp4, _handoff.hip_scale)
    _handoff.hip_produced = None
    _handoff.hip_fp4 = None
    _handoff.hip_scale = None
    return out


def _padded_token_bucket(num_tokens: int) -> int:
    """Same bucket fused_moe uses: next power of two below 32768."""
    if num_tokens <= 1:
        return 1
    if num_tokens < 32768:
        return 1 << (num_tokens - 1).bit_length()
    return 32768 if num_tokens < 131072 else 131072


_prequant_rows_cache: Optional[frozenset[tuple[str, str, int, int, int]]] = None


def _prequant_rows() -> frozenset[tuple[str, str, int, int, int]]:
    """Rows whose GEMM1 consumes host MXFP4.

    Keyed by (gfx, cu_num, token bucket, inter_dim, expert count) for hidden
    3584 and top-16. block_m 16 ``_f16in`` rows are omitted: that kernel reads
    bf16 and faults on fp4. A missing config, a different GPU, or any other
    shape is absent, and the caller keeps the unfused route and quant.
    """
    global _prequant_rows_cache
    if _prequant_rows_cache is not None:
        return _prequant_rows_cache
    try:
        import aiter
        from aiter.jit.utils.chip_info import get_cu_num, get_gfx_runtime

        gfx = str(get_gfx_runtime())
        cu = str(get_cu_num())
        path = (
            Path(aiter.__file__).parent
            / "configs"
            / "model_configs"
            / "kimik3_a4w4_tuned_fmoe.csv"
        )
        rows: set[tuple[str, str, int, int, int]] = set()
        with path.open() as f:
            for row in csv.DictReader(f):
                if (
                    row.get("gfx") != gfx
                    or row.get("cu_num") != cu
                    or row.get("model_dim") != "3584"
                    or row.get("topk") != "16"
                    or row.get("block_m") == "16"
                    or "_f16in" in row.get("kernelName1", "")
                ):
                    continue
                rows.add(
                    (
                        gfx,
                        cu,
                        int(row["token"]),
                        int(row["inter_dim"]),
                        int(row["expert"]),
                    )
                )
        _prequant_rows_cache = frozenset(rows)
        return _prequant_rows_cache
    except Exception:
        # Do not cache the failure: a missing import during startup must not
        # disable the replacement for the rest of the process.
        logger.warning("K3 MXFP4 route-quant config unavailable", exc_info=True)
        return frozenset()


def hip_prequant_bucket(
    num_tokens: int,
    inter_dim: Optional[int],
    num_experts: Optional[int],
) -> bool:
    """Whether this shape's tuned GEMM1 consumes prequantized MXFP4."""
    if inter_dim is None or num_experts is None:
        return False
    rows = _prequant_rows()
    if not rows:
        return False
    gfx, cu = next(iter(rows))[:2]
    return (
        gfx,
        cu,
        _padded_token_bucket(num_tokens),
        int(inter_dim),
        int(num_experts),
    ) in rows


def try_route_quant_fused(
    gating_output: torch.Tensor,
    correction_bias: torch.Tensor,
    topk: int,
    *,
    num_fused_shared_experts: int,
    renormalize: bool,
    routed_scaling_factor: Optional[float],
    apply_routed_scaling_factor_on_output: bool,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """Fused replacement for the ungrouped-sigmoid moe_fused_gate call when a
    staged request covers it. Returns (weights, ids) on a hit, None otherwise
    (caller falls through to the unfused router)."""
    x = _handoff.request_x
    if x is None or num_fused_shared_experts != 0:
        return None

    from sglang.kernels.ops.moe import moe_route_quant_fused

    if (
        not moe_route_quant_fused.covered(gating_output, correction_bias, topk, x)
        or not moe_route_quant_fused.available()
    ):
        return None

    weights, ids, packed, x_q, x_s = moe_route_quant_fused.route_quant_fused(
        gating_output,
        correction_bias,
        x,
        topk,
        renormalize=renormalize,
        routed_scaling_factor=(
            routed_scaling_factor if routed_scaling_factor is not None else 1.0
        ),
        apply_scale=apply_routed_scaling_factor_on_output,
    )
    _handoff.request_x = None
    _handoff.produced_x = x
    _handoff.packed = packed
    _handoff.x_q = x_q
    _handoff.x_s = x_s
    return weights, ids


def take(
    x: torch.Tensor,
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Consume the published (packed_topk, x_q, x_s int32) for these exact
    activation rows, or None. Storage identity is verified so a re-viewed or
    copied tensor simply misses."""
    produced = _handoff.produced_x
    # A HIP publish never sets packed. Returning the Nones would look like a
    # hit to the CUDA consumer.
    if produced is None or _handoff.packed is None:
        return None
    if (
        produced.data_ptr() != x.data_ptr()
        or produced.shape != x.shape
        or produced.dtype != x.dtype
        or produced.stride() != x.stride()
    ):
        return None
    out = (_handoff.packed, _handoff.x_q, _handoff.x_s)
    _handoff.produced_x = None
    _handoff.packed = None
    _handoff.x_q = None
    _handoff.x_s = None
    return out
