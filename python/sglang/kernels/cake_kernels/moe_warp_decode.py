"""Cake NVFP4 warp-decode MoE runner (``CakeWarpDecodeConfig``) via FlashInfer.

FlashInfer entries: ``flashinfer.fused_moe.CakeWarpDecodeConfig`` (frozen
backend config, ``api.py``), ``flashinfer.fused_moe.CakeWarpDecodeRunner``
(``MoERunner`` subclass, ``runners.py``), JIT module
``flashinfer.jit.cake_fused_moe_warp_decode``. Contract at FlashInfer
``46340689a5ab``: exact SM100 / SM103; NVFP4 x NVFP4 only, in the TRTLLM
``trtllm_fp4_routed`` physical view (``CakeWarpDecodeConfig.prepare_weights`` /
``prepare_activations`` delegate to ``TrtllmFp4Config``); linear (non-swizzled)
scale layout, no per-token scale; ``1 <= num_tokens <= 32``;
``RoutingInputMode.UnpackedPrecomputed`` only; ``do_finalize=True``,
``enable_pdl=True``, ``local_expert_offset=0``, ``local_num_experts=E``;
calibrated (activation, H, I, E, top_k) tuples listed in ``GEOMETRIES``.

Usage: build ``MoEConfig(..., backend=BackendOptions((cake_warp_decode_config(),)),
execution=ExecutionConfig(enable_pdl=True))`` and hand it to
``flashinfer.fused_moe.MoELayer`` (which maps the config to the runner), or
construct the runner directly with :func:`warp_decode_runner`.

CUDA graphs: the runner prepares one workspace receipt per (stream, geometry)
through ``cake_fused_moe_warp_decode_prepare_workspace``; run one eager
forward (warmup) BEFORE capture, otherwise capture raises. ``topk_ids`` are
range-validated on the host before capture (at most 64 receipts / 64 stream
workspaces are cached). Single tactic (-1), ``TuningConfig(use_cuda_graph=True)``.

Not supported here: other geometries or activations, EP / expert offsets,
bias, LoRA, FromLogits routing, swizzled scales, per-token scales, >32 tokens.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Tuple

from sglang.kernels.cake_kernels._support import SM100, SM103
from sglang.kernels.cake_kernels.moe_common import (
    cuda_device_in,
    current_cuda_index,
    modules_available,
)

if TYPE_CHECKING:
    import torch

FI_MODULE = "flashinfer.fused_moe.runners"
FI_JIT_MODULE = "flashinfer.jit.cake_fused_moe_warp_decode"
ARCHS = (SM100, SM103)
MAX_TOKENS = 32

# (activation key, hidden_size, intermediate_size, num_experts, top_k).
# Activation keys: "swiglu" = default SwiGLU(); "silu" = SiLU();
# "swiglu_1.702_1.0_7.0" = SwiGLU(alpha=1.702, beta=1.0, limit=7.0);
# "situ_4.0_25.0" = SiTU(gate_scale=4.0, linear_scale=25.0).
GEOMETRIES: Tuple[Tuple[str, int, int, int, int], ...] = (
    ("swiglu", 2048, 512, 512, 10),
    ("swiglu", 2048, 1536, 60, 4),
    ("swiglu", 2560, 768, 384, 4),
    ("silu", 6144, 1536, 192, 4),
    ("swiglu", 2048, 768, 128, 8),
    ("swiglu", 4096, 1536, 128, 8),
    ("swiglu", 2048, 512, 256, 8),
    ("swiglu", 4096, 1024, 512, 10),
    ("swiglu", 3072, 1536, 256, 8),
    ("swiglu_1.702_1.0_7.0", 6144, 3072, 128, 4),
    ("situ_4.0_25.0", 3584, 3072, 896, 16),
    # Sharded per-partition slices: Qwen3.5-397B TP2 / TP4 and MiniMax-M2 TP2 / TP4.
    # (MiniMax-M2 TP8, I = 192, is not admissible: the TRT-LLM FP4 weight view and
    # the trtllm-gen launcher require intermediate_size % 128 == 0.)
    ("swiglu", 4096, 512, 512, 10),
    ("swiglu", 4096, 256, 512, 10),
    ("swiglu", 3072, 768, 256, 8),
    ("swiglu", 3072, 384, 256, 8),
)


def activation_key(activation: Any) -> Optional[str]:
    """Map a FlashInfer ``ActivationConfig`` instance (or key string) to a GEOMETRIES key.

    Instances are compared by equality against the exact table entries of
    ``CakeWarpDecodeRunner._SUPPORTED_CONFIGURATIONS``; ``None`` when the
    activation is outside the table or FlashInfer is not installed.
    """
    if isinstance(activation, str):
        return activation if activation in {g[0] for g in GEOMETRIES} else None
    try:
        from flashinfer.fused_moe import SiLU, SiTU, SwiGLU
    except ImportError:
        return None
    table = (
        ("swiglu", SwiGLU()),
        ("silu", SiLU()),
        ("swiglu_1.702_1.0_7.0", SwiGLU(alpha=1.702, beta=1.0, limit=7.0)),
        ("situ_4.0_25.0", SiTU(gate_scale=4.0, linear_scale=25.0)),
    )
    for key, instance in table:
        if activation == instance:
            return key
    return None


def supports_warp_decode(
    *,
    activation: Any,
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
    top_k: int,
    num_tokens: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> bool:
    """Admission check mirroring ``CakeWarpDecodeRunner._check_support``; never raises."""
    try:
        key = activation_key(activation)
        if key is None:
            return False
        if (key, hidden_size, intermediate_size, num_experts, top_k) not in GEOMETRIES:
            return False
        if num_tokens is not None and not 1 <= int(num_tokens) <= MAX_TOKENS:
            return False
        return modules_available(FI_MODULE, FI_JIT_MODULE) and cuda_device_in(
            current_cuda_index(device), ARCHS
        )
    except Exception:
        return False


def get_warp_decode_config_class():
    from flashinfer.fused_moe import CakeWarpDecodeConfig

    return CakeWarpDecodeConfig


def get_warp_decode_runner_class():
    from flashinfer.fused_moe import CakeWarpDecodeRunner

    return CakeWarpDecodeRunner


def warp_decode_config():
    """``CakeWarpDecodeConfig(backend="cake")`` for ``MoEConfig.backend``."""
    from flashinfer.fused_moe import CakeWarpDecodeConfig

    return CakeWarpDecodeConfig(backend="cake")


def warp_decode_runner(config: Any, device: torch.device):
    """Construct ``CakeWarpDecodeRunner(config, device)`` (call ``check_support()`` / ``build()`` as ``MoELayer`` does)."""
    from flashinfer.fused_moe import CakeWarpDecodeRunner

    return CakeWarpDecodeRunner(config, device)


def warp_decode_prepare_weights(
    w1_bf16: torch.Tensor,
    w2_bf16: torch.Tensor,
    *,
    num_local_experts: int,
    hidden_size: int,
    intermediate_size: int,
    activation: Any = None,
    quant: Any = None,
    device: Optional[torch.device] = None,
    permute_cache: Any = None,
):
    """Shared TRTLLM NVFP4 weight view; register with ``MoEWeightPack.prepare_for("cake", view)``."""
    from flashinfer.fused_moe import CakeWarpDecodeConfig

    kwargs = dict(
        num_local_experts=num_local_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        activation=activation,
        device=device,
        permute_cache=permute_cache,
    )
    if quant is not None:
        kwargs["quant"] = quant
    return CakeWarpDecodeConfig.prepare_weights(w1_bf16, w2_bf16, **kwargs)


def warp_decode_prepare_activations(
    hidden_states_bf16: torch.Tensor, *, quant: Any = None
):
    """Shared TRTLLM NVFP4 packed activation view ``(hidden_states_q, hidden_states_scale)``."""
    from flashinfer.fused_moe import CakeWarpDecodeConfig

    if quant is None:
        return CakeWarpDecodeConfig.prepare_activations(hidden_states_bf16)
    return CakeWarpDecodeConfig.prepare_activations(hidden_states_bf16, quant=quant)
