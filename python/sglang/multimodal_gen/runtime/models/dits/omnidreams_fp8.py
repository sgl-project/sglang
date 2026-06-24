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
