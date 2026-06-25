# SPDX-License-Identifier: Apache-2.0
"""OmniDreams FP8 weight utilities (pure Python).

Phase 1 dropped the vendored native CUDA FP8 DiT tree. What remains is the
CPU-runnable FP8 weight surface used by the offline exporter and the
``weight_only_fp8`` runtime path:

* :func:`prepare_fp8_dit_weights` -- per-output-channel E4M3 quantization of the
  DiT linears (relocated Cosmos helpers, no native code).
* :func:`dequantize_fp8_weights_to_bf16` -- reverse the quantization so the
  standard eager bf16 DiT can run on the dequantized weights (Ideogram 4 style
  weight-only FP8).
* :func:`_unfuse_self_attn_qkv_for_cosmos` -- split the fused ``to_qkv`` back
  into q/k/v for the Cosmos weight-prep helpers.

The native FP8 DiT dispatch (FP8 tensor-core GEMMs + Sage3/Sparge attention via
the vendored C++ extension) was removed with the native tree. A PyTorch-native
``fp8_compute`` mode may be added in Phase 2.
"""

from __future__ import annotations

from typing import Any

import torch

from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


# --------------------------------------------------------------------------- #
# FP8 weight preparation (CPU-runnable)                                       #
# --------------------------------------------------------------------------- #
def _unfuse_self_attn_qkv_for_cosmos(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Split fused ``self_attn.to_qkv`` back into q/k/v for the Cosmos prep.

    The sglang OmniDreamsDiT stores self-attention Q/K/V as a single
    ``MergedColumnParallelLinear`` (``to_qkv`` = ``cat([q, k, v], dim=0)``,
    q/k/v row order -- see ``OmniDreamsAttention.__init__``), but the Cosmos
    weight-prep / FP8-quant helpers
    (``prepare_cosmos_streaming_weights``, ``quantize_cosmos_fp8_weights``)
    expect the legacy split ``q_proj``/``k_proj``/``v_proj`` keys. This splits
    the fused weight back into three equal row shards so those helpers run
    unchanged and produce byte-identical artifacts to the pre-merge path:
    per-output-channel FP8 scales are row-independent, so quantizing the fused
    tensor then splitting == splitting then quantizing (verified EXACT on the
    30 self-attn QKV tensors).

    The fused ``to_qkv`` key is dropped from the output: the FP8 runtime
    consumes the rebuilt ``qkv_proj`` and does not read ``to_qkv``, so keeping
    it would only ship ~1.3 GB of dead bf16 in the artifact.
    """
    suffix = "self_attn.to_qkv.weight"
    fused_keys = [k for k in state_dict if k.endswith(suffix)]
    if not fused_keys:
        return state_dict
    out = dict(state_dict)
    for fused_key in fused_keys:
        prefix = fused_key[: -len(suffix)]  # "blocks.{i}."
        fused = out.pop(fused_key)
        if fused.dim() != 2 or fused.shape[0] % 3 != 0:
            raise ValueError(
                f"cannot split {fused_key!r} (shape {tuple(fused.shape)}) into "
                "3 equal q/k/v row shards"
            )
        q_weight, k_weight, v_weight = torch.chunk(fused, 3, dim=0)
        out[f"{prefix}self_attn.q_proj.weight"] = q_weight.contiguous()
        out[f"{prefix}self_attn.k_proj.weight"] = k_weight.contiguous()
        out[f"{prefix}self_attn.v_proj.weight"] = v_weight.contiguous()
    return out


def prepare_fp8_dit_weights(
    state_dict: dict[str, torch.Tensor],
    num_blocks: int,
    *,
    cosmos_fp8_utils: Any | None = None,
    linear_policy: str = "all",
) -> dict[str, torch.Tensor]:
    if cosmos_fp8_utils is None:
        from sglang.multimodal_gen.runtime.models.dits.omnidreams_cosmos_fp8_utils import (
            prepare_cosmos_quantized_streaming_weights,
        )
    else:
        prepare_cosmos_quantized_streaming_weights = (
            cosmos_fp8_utils.prepare_cosmos_quantized_streaming_weights
        )
    cpu_state = {k: v.detach().cpu() for k, v in state_dict.items()}
    # sglang's DiT fuses self-attn Q/K/V into ``to_qkv``; the Cosmos prep/quant
    # helpers expect the legacy split q/k/v keys. Unfuse first so they run
    # unchanged (see _unfuse_self_attn_qkv_for_cosmos).
    cpu_state = _unfuse_self_attn_qkv_for_cosmos(cpu_state)
    return prepare_cosmos_quantized_streaming_weights(
        cpu_state, num_blocks=num_blocks, device=None, linear_policy=linear_policy,
    )


# --------------------------------------------------------------------------- #
# FP8 → bf16 dequantization (weight-only FP8, Ideogram 4 style)               #
# --------------------------------------------------------------------------- #
_WEIGHT_SUFFIX = ".weight"
_SCALE_SUFFIX = ".weight_scale"


def dequantize_fp8_weights_to_bf16(
    fp8_weights: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Dequantize FP8 quantized weights back to bf16 for standard PyTorch inference.

    The FP8 artifact stores each quantized linear weight as:
      - ``K.weight``: raw E4M3 bytes (torch.uint8, [out, in])
      - ``K.weight_scale``: per-output-channel scale (float16, [out])

    This function reverses the quantization:
      weight_bf16 = weight.view(float8_e4m3fn).to(float32) * scale.unsqueeze(1)

    Returns a dict mapping original ``.weight`` keys to dequantized bf16 tensors,
    suitable for ``model.load_state_dict(strict=False)``.
    """
    result: dict[str, torch.Tensor] = {}
    dequant_count = 0

    for key, value in fp8_weights.items():
        # Already dequantized or not a weight — pass through
        if key.endswith(_SCALE_SUFFIX):
            continue  # scales consumed by their weight key
        if key.endswith(".weight") and value.dtype == torch.uint8:
            # FP8 raw E4M3 bytes — dequantize
            scale_key = key + "_scale"
            if scale_key in fp8_weights:
                scale = fp8_weights[scale_key]
                weight_f32 = value.view(torch.float8_e4m3fn).to(torch.float32)
                dequant = (weight_f32 * scale.to(torch.float32).unsqueeze(1)).to(
                    torch.bfloat16
                )
                result[key] = dequant.contiguous()
                dequant_count += 1
            else:
                logger.warning("FP8 weight %s has no scale %s; skipping", key, scale_key)
        elif key.endswith(".weight_prepared"):
            # bf16 transposed prepared weight — skip (model uses .weight layout)
            continue
        elif key.endswith("_fp8_prepared"):
            # FP8 prepared alias — skip (base .weight key covers it)
            continue
        elif key.endswith("_fp8_prepared_scale"):
            # FP8 prepared alias scale — skip
            continue
        else:
            # Non-FP8 key (biases, norms, embeddings, bf16 weights) — pass through
            if key not in result:
                result[key] = value

    logger.info(
        "Dequantized %d FP8 weight pairs to bf16 (%d total keys in result)",
        dequant_count,
        len(result),
    )
    return result

# ============================================================================
# FP8-compute linears (folded from omnidreams_fp8_compute.py)
# ============================================================================
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
