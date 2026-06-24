# SPDX-License-Identifier: Apache-2.0
"""Phase 2: PyTorch-native FP8-compute linears for the OmniDreams DiT.

Replaces the DiT's bf16 GEMMs (self/cross-attn projections + MLP) with
FP8-compute matmuls via ``torch._scaled_mm`` on Blackwell (sm_120+), without any
custom CUDA. Weights are quantized once (lazily, on first call) to FP8 e4m3 with
a **per-output-channel (row-wise) scale** -- the same per-channel scheme as the
offline FP8 artifact (``omnidreams_cosmos_fp8_utils.quantize_fp8_per_out_channel``)
-- and activations are quantized dynamically **per token (per row)** before each
matmul. The rowwise ``torch._scaled_mm(a_fp8, w_fp8.t(), scale_a=[M,1],
scale_b=[N], out_dtype=bf16)`` runs on FP8 tensor cores (~2x bf16 on sm_120).

Design:
* TP linears (``MergedColumnParallelLinear``/``ColumnParallelLinear``/
  ``RowParallelLinear``) delegate the matmul to ``self.quant_method.apply``; the
  TP gather/reduce lives in the linear's ``forward``. Swapping ``quant_method``
  to :class:`OmniDreamsFP8ComputeLinearMethod` FP8s the matmul while preserving
  all TP semantics -- no weight-loader / sharding reimplementation needed.
* Plain ``nn.Linear`` (the GPT2FeedForward ``mlp.layer1``/``layer2`` are not
  TP-sharded) is replaced with :class:`OmniDreamsFP8ComputeLinear`, a thin
  wrapper holding the original bf16 weight and the same FP8 forward.

The DiT is loaded normally (bf16 checkpoint, TP-sharded, ``post_load_weights``);
:func:`install_fp8_compute_on_dit` then swaps the linears in place. Each rank
quantizes its own TP shard, so the per-channel scales are TP-correct. On non-FP8
HW (CPU, no ``_scaled_mm``) the install is a no-op and the DiT runs eager bf16.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from sglang.multimodal_gen.runtime.layers.quantization.configs.base_config import (
    QuantizeMethodBase,
)
from sglang.multimodal_gen.runtime.models.dits.omnidreams_cosmos_fp8_utils import (
    FP8_MAX_E4M3,
    FP8_SCALE_EPS,
    quantize_fp8_per_out_channel,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


def _scaled_mm_available() -> bool:
    """True if the runtime supports FP8 e4m3 ``torch._scaled_mm`` on a CUDA device."""
    if not torch.cuda.is_available():
        return False
    if not hasattr(torch, "float8_e4m3fn"):
        return False
    fn = getattr(torch, "_scaled_mm", None)
    return callable(fn)


def _quantize_activation_per_token(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token (per-row) dynamic E4M3 activation quant.

    ``x`` is reshaped to ``[M, K]``; returns ``(x_fp8 [M,K] float8_e4m3fn,
    scale [M,1] float32)``.
    """
    orig_shape = x.shape
    K = orig_shape[-1]
    x2 = x.reshape(-1, K).to(torch.float32)
    amax = x2.abs().amax(dim=-1, keepdim=True).clamp(min=FP8_SCALE_EPS)
    scale = amax / FP8_MAX_E4M3  # [M,1]
    q = (x2 / scale).clamp(-FP8_MAX_E4M3, FP8_MAX_E4M3).to(torch.float8_e4m3fn)
    return q, scale.to(torch.float32)


def _fp8_matmul(
    x: torch.Tensor,
    w_fp8: torch.Tensor,
    w_scale: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    """FP8-compute matmul: ``x @ w_fp8.t()`` via rowwise ``torch._scaled_mm``.

    Args:
        x: activation ``[..., K]`` (bf16/fp32).
        w_fp8: weight ``[N, K]`` as ``float8_e4m3fn`` (per-row scale ``w_scale``).
        w_scale: per-output-channel weight scale ``[N]``.
        bias: optional ``[N]``.

    Returns: ``[..., N]`` bf16.
    """
    orig_shape = x.shape
    x_fp8, a_scale = _quantize_activation_per_token(x)  # [M,K], [M,1]
    # rowwise scaled_mm: scale_a [M,1], scale_b [N]; weight passed as [K,N] (w.t()).
    out = torch._scaled_mm(
        x_fp8,
        w_fp8.t(),
        scale_a=a_scale,
        scale_b=w_scale,
        out_dtype=torch.bfloat16,
    )  # [M, N]
    out = out.reshape(*orig_shape[:-1], out.shape[-1])
    if bias is not None:
        out = out + bias
    return out


def _ensure_weight_cache(module: nn.Module) -> tuple[torch.Tensor, torch.Tensor]:
    """Lazily quantize ``module.weight`` (bf16) to FP8 per-channel; cache on module.

    Returns ``(w_fp8 [N,K] float8_e4m3fn, w_scale [1,N] float32)``. The scale is
    reshaped to 2-D and cast to float32 because the rowwise ``torch._scaled_mm``
    path requires ``scale_a=[M,1]`` / ``scale_b=[1,N]`` float32 contiguous.
    """
    weight = module.weight
    cache = getattr(module, "_fp8_compute_cache", None)
    if cache is not None and cache[0] is weight:
        return cache[1], cache[2]
    w_u8, w_scale = quantize_fp8_per_out_channel(weight)  # uint8 [N,K], scale [N] fp16
    w_fp8 = w_u8.view(torch.float8_e4m3fn)
    w_scale = w_scale.to(torch.float32).reshape(1, -1).contiguous()  # [1, N]
    module._fp8_compute_cache = (weight, w_fp8, w_scale)
    return w_fp8, w_scale


class OmniDreamsFP8ComputeLinearMethod(QuantizeMethodBase):
    """Drop-in ``quant_method`` that FP8-computes an existing TP linear's matmul.

    ``create_weights`` is a no-op: the linear already holds its bf16 weight
    (loaded by the normal checkpoint path). ``apply`` lazily quantizes that
    weight per-output-channel and runs ``torch._scaled_mm``. The linear's
    ``forward`` (TP gather/reduce) is unchanged, so all TP semantics survive.
    """

    def create_weights(self, layer: nn.Module, *args, **kwargs) -> None:  # noqa: D401
        # Weight already exists on the layer (post-load swap); nothing to create.
        return None

    def apply(
        self, layer: nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None
    ) -> torch.Tensor:
        w_fp8, w_scale = _ensure_weight_cache(layer)
        return _fp8_matmul(x, w_fp8, w_scale, bias)


class OmniDreamsFP8ComputeLinear(nn.Module):
    """Thin FP8-compute replacement for a plain ``nn.Linear`` (non-TP).

    Holds the original bf16 weight (and optional bias) and runs the same rowwise
    ``torch._scaled_mm`` forward. Used for the GPT2FeedForward MLP linears, which
    are plain ``nn.Linear`` (no TP sharding).
    """

    def __init__(self, weight: torch.Tensor, bias: torch.Tensor | None = None) -> None:
        super().__init__()
        self.weight = nn.Parameter(weight.detach().to(torch.bfloat16), requires_grad=False)
        if bias is not None:
            self.bias = nn.Parameter(bias.detach().to(torch.bfloat16), requires_grad=False)
        else:
            self.bias = None
        self._fp8_compute_cache: tuple | None = None

    @classmethod
    def from_linear(cls, linear: nn.Linear) -> "OmniDreamsFP8ComputeLinear":
        return cls(linear.weight.data, getattr(linear, "bias", None))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_fp8, w_scale = _ensure_weight_cache(self)
        return _fp8_matmul(x, w_fp8, w_scale, self.bias)


def install_fp8_compute_on_dit(dit: nn.Module) -> bool:
    """Swap the DiT's linears to FP8-compute in place (post-load).

    Returns True if installed, False if skipped (non-CUDA / no ``_scaled_mm`` ->
    the DiT keeps running eager bf16). Idempotent: re-installing is a no-op
    (guarded by ``dit._fp8_compute_applied``).
    """
    if getattr(dit, "_fp8_compute_applied", False):
        return True
    if not _scaled_mm_available():
        logger.info(
            "OmniDreams fp8_compute: torch._scaled_mm (FP8 e4m3) unavailable on "
            "this device; keeping eager bf16 DiT."
        )
        return False

    method = OmniDreamsFP8ComputeLinearMethod()
    n_swapped = 0
    for block in dit.blocks:
        # Self-attn: fused QKV (MergedColumnParallel) + output (RowParallel).
        block.self_attn.to_qkv.quant_method = method
        block.self_attn.output_proj.quant_method = method
        # Cross-attn: q (ColumnParallel) + fused KV (MergedColumnParallel) + output.
        block.cross_attn.q_proj.quant_method = method
        block.cross_attn.to_kv.quant_method = method
        block.cross_attn.output_proj.quant_method = method
        n_swapped += 5
        # MLP: plain nn.Linear (not TP-sharded) -> replace with the FP8 wrapper.
        block.mlp.layer1 = OmniDreamsFP8ComputeLinear.from_linear(block.mlp.layer1)
        block.mlp.layer2 = OmniDreamsFP8ComputeLinear.from_linear(block.mlp.layer2)
        n_swapped += 2

    dit._fp8_compute_applied = True
    logger.info(
        "OmniDreams fp8_compute: swapped %d linears/block x %d blocks "
        "(%d total) to torch._scaled_mm (rowwise per-channel weight + per-token "
        "activation).",
        n_swapped // max(len(dit.blocks), 1),
        len(dit.blocks),
        n_swapped,
    )
    return True
