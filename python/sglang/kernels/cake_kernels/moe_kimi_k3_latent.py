"""Cake Kimi-K3 Stable LatentMoE front / tail projections via FlashInfer.

FlashInfer entries: ``flashinfer.kimi_k3_latent_moe.prepare_kimi_k3_latent_moe_front``
/ ``kimi_k3_latent_moe_front`` / ``prepare_kimi_k3_latent_moe_tail`` /
``kimi_k3_latent_moe_tail`` (host planner and launch binding in
``flashinfer.experimental.kimi_k3_latent_moe.cake_backend``, JIT registry
``...kimi_k3_latent_moe.cake_jit``). Contract at FlashInfer ``46340689a5ab``:
sm_100a / sm_103a AND exactly 148 SMs (plans frozen for ``SM_COUNT = 148``;
``prepare_*`` raises on other counts). All operands contiguous BF16 on one
device.

Front: ``x [T, 7168]``, ``gate_weight [896, 7168]``, ``down_weight [3584, 7168]``,
``shared_gate_up_weight [2*6144/tp, 7168]`` (gate rows then up rows, ``tp in
{1, 8}`` inferred from the row count); outputs ``logits [T, 896]`` f32,
``latent [T, 3584]`` BF16, ``shared_act [T, 6144/tp]`` BF16 =
``SiTU(g, u) = 4 tanh(g/4) sigmoid(g) * 25 tanh(u/25)``. ``T <= 128`` runs the
weight-streaming decode kernel (one launch), ``T > 128`` a persistent 2-CTA
GEMM (validated up to 16384).

Tail: ``routed [P, T, 3584]`` (P un-reduced partials summed in FP32),
``norm_weight [3584]``, ``up_weight [7168, 3584]`` replicated,
``shared_act [T, 6144/tp]``, ``shared_down_weight [7168, 6144/tp]``,
``out [T, 7168]`` (rank partial when ``tp > 1``; caller all-reduces),
``y_workspace [T, 3584]`` receives the KimiRMSNorm (eps 1e-5) latent;
``tp in {1, 8}``, ``0 <= rank < tp``. ``T <= 128``: one launch; ``T > 128``:
``tail_norm`` + programmatic-dependent ``tail_gemm`` (trailing-wave stream-K).

CUDA graphs: prepare outside capture (JIT build, per-device scratch); the
returned ``KimiK3LatentMoeRunner`` launches allocation-free and is capturable.

Not supported here: packed / quantized weights, EP, other SM counts, TP other
than 1 or 8.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Tuple

from sglang.kernels.cake_kernels._support import SM100, SM103, cuda_tensor_on
from sglang.kernels.cake_kernels.moe_common import (
    contiguous_cuda,
    device_sm_count,
    modules_available,
    same_device,
)

if TYPE_CHECKING:
    import torch

FI_MODULE = "flashinfer.kimi_k3_latent_moe"
FI_JIT_MODULE = "flashinfer.experimental.kimi_k3_latent_moe.cake_jit"
ARCHS = (SM100, SM103)
SM_COUNT = 148
HIDDEN = 7168
LATENT = 3584
NUM_EXPERTS = 896
SHARED_INTERMEDIATE = 6144
SUPPORTED_TP = (1, 8)
RMS_EPS = 1.0e-5


def _device_ok(t: torch.Tensor) -> bool:
    return cuda_tensor_on(t, ARCHS) and device_sm_count(t.device.index) == SM_COUNT


def supports_kimi_k3_latent_moe_front(
    x: torch.Tensor,
    gate_weight: torch.Tensor,
    down_weight: torch.Tensor,
    shared_gate_up_weight: torch.Tensor,
    logits: torch.Tensor,
    latent: torch.Tensor,
    shared_act: torch.Tensor,
) -> bool:
    """Admission check mirroring the FlashInfer front contract; never raises."""
    import torch

    try:
        if not (modules_available(FI_MODULE, FI_JIT_MODULE) and _device_ok(x)):
            return False
        if x.ndim != 2 or shared_gate_up_weight.ndim != 2:
            return False
        T = int(x.shape[0])
        i_local = int(shared_gate_up_weight.shape[0]) // 2
        if T <= 0 or i_local not in {SHARED_INTERMEDIATE // tp for tp in SUPPORTED_TP}:
            return False
        bf16 = torch.bfloat16
        return (
            contiguous_cuda(x, shape=(T, HIDDEN), dtype=bf16)
            and contiguous_cuda(gate_weight, shape=(NUM_EXPERTS, HIDDEN), dtype=bf16)
            and contiguous_cuda(down_weight, shape=(LATENT, HIDDEN), dtype=bf16)
            and contiguous_cuda(
                shared_gate_up_weight, shape=(2 * i_local, HIDDEN), dtype=bf16
            )
            and contiguous_cuda(logits, shape=(T, NUM_EXPERTS), dtype=torch.float32)
            and contiguous_cuda(latent, shape=(T, LATENT), dtype=bf16)
            and contiguous_cuda(shared_act, shape=(T, i_local), dtype=bf16)
            and same_device(
                x,
                gate_weight,
                down_weight,
                shared_gate_up_weight,
                logits,
                latent,
                shared_act,
            )
        )
    except Exception:
        return False


def supports_kimi_k3_latent_moe_tail(
    routed: torch.Tensor,
    norm_weight: torch.Tensor,
    up_weight: torch.Tensor,
    shared_act: torch.Tensor,
    shared_down_weight: torch.Tensor,
    out: torch.Tensor,
    *,
    tp: int,
    rank: int,
    y_workspace: torch.Tensor,
) -> bool:
    """Admission check mirroring the FlashInfer tail contract; never raises."""
    import torch

    try:
        if not (modules_available(FI_MODULE, FI_JIT_MODULE) and _device_ok(routed)):
            return False
        if tp not in SUPPORTED_TP or not 0 <= int(rank) < tp or routed.ndim != 3:
            return False
        P, T = int(routed.shape[0]), int(routed.shape[1])
        if P <= 0 or T <= 0:
            return False
        i_local = SHARED_INTERMEDIATE // tp
        bf16 = torch.bfloat16
        return (
            contiguous_cuda(routed, shape=(P, T, LATENT), dtype=bf16)
            and contiguous_cuda(norm_weight, shape=(LATENT,), dtype=bf16)
            and contiguous_cuda(up_weight, shape=(HIDDEN, LATENT), dtype=bf16)
            and contiguous_cuda(shared_act, shape=(T, i_local), dtype=bf16)
            and contiguous_cuda(shared_down_weight, shape=(HIDDEN, i_local), dtype=bf16)
            and contiguous_cuda(out, shape=(T, HIDDEN), dtype=bf16)
            and contiguous_cuda(y_workspace, shape=(T, LATENT), dtype=bf16)
            and same_device(
                routed,
                norm_weight,
                up_weight,
                shared_act,
                shared_down_weight,
                out,
                y_workspace,
            )
        )
    except Exception:
        return False


def prepare_kimi_k3_latent_moe_front(
    x: torch.Tensor,
    gate_weight: torch.Tensor,
    down_weight: torch.Tensor,
    shared_gate_up_weight: torch.Tensor,
    logits: torch.Tensor,
    latent: torch.Tensor,
    shared_act: torch.Tensor,
) -> Any:
    """Forward to FlashInfer; returns ``KimiK3LatentMoeRunner`` (call -> ``(logits, latent, shared_act)``)."""
    from flashinfer.kimi_k3_latent_moe import (
        prepare_kimi_k3_latent_moe_front as fi_prepare,
    )

    return fi_prepare(
        x,
        gate_weight,
        down_weight,
        shared_gate_up_weight,
        logits,
        latent,
        shared_act,
        backend="cake",
    )


def kimi_k3_latent_moe_front(
    x: torch.Tensor,
    gate_weight: torch.Tensor,
    down_weight: torch.Tensor,
    shared_gate_up_weight: torch.Tensor,
    logits: torch.Tensor,
    latent: torch.Tensor,
    shared_act: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prepare + one launch; returns ``(logits, latent, shared_act)``."""
    from flashinfer.kimi_k3_latent_moe import kimi_k3_latent_moe_front as fi_front

    return fi_front(
        x,
        gate_weight,
        down_weight,
        shared_gate_up_weight,
        logits,
        latent,
        shared_act,
        backend="cake",
    )


def prepare_kimi_k3_latent_moe_tail(
    routed: torch.Tensor,
    norm_weight: torch.Tensor,
    up_weight: torch.Tensor,
    shared_act: torch.Tensor,
    shared_down_weight: torch.Tensor,
    out: torch.Tensor,
    *,
    tp: int,
    rank: int,
    y_workspace: torch.Tensor,
) -> Any:
    """Forward to FlashInfer; returns ``KimiK3LatentMoeRunner`` (call -> ``(y_workspace, out)``)."""
    from flashinfer.kimi_k3_latent_moe import (
        prepare_kimi_k3_latent_moe_tail as fi_prepare,
    )

    return fi_prepare(
        routed,
        norm_weight,
        up_weight,
        shared_act,
        shared_down_weight,
        out,
        tp=tp,
        rank=rank,
        y_workspace=y_workspace,
        backend="cake",
    )


def kimi_k3_latent_moe_tail(
    routed: torch.Tensor,
    norm_weight: torch.Tensor,
    up_weight: torch.Tensor,
    shared_act: torch.Tensor,
    shared_down_weight: torch.Tensor,
    out: torch.Tensor,
    *,
    tp: int,
    rank: int,
    y_workspace: torch.Tensor,
) -> torch.Tensor:
    """Prepare + one launch; returns ``out``."""
    from flashinfer.kimi_k3_latent_moe import kimi_k3_latent_moe_tail as fi_tail

    return fi_tail(
        routed,
        norm_weight,
        up_weight,
        shared_act,
        shared_down_weight,
        out,
        tp=tp,
        rank=rank,
        y_workspace=y_workspace,
        backend="cake",
    )


def device_supports_kimi_k3_latent_moe(device: Optional[torch.device] = None) -> bool:
    """Arch + 148-SM + module probe without tensors."""
    from sglang.kernels.cake_kernels.moe_common import (
        cuda_device_in,
        current_cuda_index,
    )

    try:
        index = current_cuda_index(device)
        return (
            modules_available(FI_MODULE, FI_JIT_MODULE)
            and cuda_device_in(index, ARCHS)
            and device_sm_count(index) == SM_COUNT
        )
    except Exception:
        return False
