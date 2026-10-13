# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Load-time FP8 quantization for the BF16 linears a K3 checkpoint ships dense.

The official ``moonshotai/Kimi-K3`` release packs only ``block_sparse_moe.experts.*``
as MXFP4 and leaves ``self_attn.*`` and ``shared_experts.*`` in BF16, so those
GEMMs stream twice the weight bytes at decode and run at half the MFMA rate at
prefill. Quark's export of the same model quantizes ``self_attn.*`` to
per-output-channel FP8 with dynamic per-token activations; this module builds
that exact layout in memory instead, which is what every downstream K3 ROCm fast
path keys off:

* ``kimi_k3_rocm_quant._k3_merge_kda_inproj_fp8`` reads ``[out, in]`` e4m3 plus an
  ``[out]`` fp32 ``weight_scale`` and packs one PTPC GEMM for the whole KDA input
  projection.
* ``deepseek_common.utils_rocm.accepts_ptpc_fp8_tuple`` keys off ``scheme``, so the
  RMSNorm/gate producer fusions can hand ``(fp8, per-token scale)`` straight to
  ``o_proj``.
* ``QuarkW8A8Fp8.process_weights_after_loading`` does the transpose, the
  bpreshuffle and the narrow-N dequant fallback, so the layout stays byte-identical
  to a Quark checkpoint's.

The conversion therefore runs *before* the merges in ``post_load_weights`` and
leaves ``process_weights_after_loading`` to the loader, matching the order a Quark
checkpoint goes through.
"""

import logging
from typing import Iterator, Optional

import torch
from torch import nn
from torch.nn import Parameter

from sglang.srt.environ import envs
from sglang.srt.layers.linear import LinearBase
from sglang.srt.layers.quantization.base_config import LinearMethodBase
from sglang.srt.utils import is_hip

logger = logging.getLogger(__name__)

_is_hip = is_hip()

# The three switches -- SGLANG_ROCM_K3_ONLINE_FP8_ATTN, _KDA_INPROJ and
# _SHARED_EXPERTS -- are registered and documented in ``srt/environ.py``. They
# are read per call rather than cached at import so ``.override()`` reaches
# them.

# Rows per quantization chunk. A KDA input projection is [6144, 7168] at TP=8;
# promoting all of it to fp32 at once costs 176 MiB of transient VRAM right
# where the loader's own peak sits, and the amax is row-independent anyway.
_QUANT_CHUNK_ROWS = 1024

# ``_prepare_fused_decode`` only arms the AITER fused KDA decode when
# ``f_b_proj.weight`` is a BF16 [12*128, 128]. At 196K parameters the FP8 copy
# would save nothing and cost that kernel, so f_b_proj is never converted.
_SKIP_SUFFIXES = ("f_b_proj",)

# The three projections ``_merge_bfa_weights`` fuses into one KDA input GEMM.
# They convert together or not at all: ``_k3_merge_kda_inproj_fp8`` needs all
# three in the same layout, and a partial conversion would leave the merge on
# the slower unfused path.
_KDA_INPROJ_SUFFIXES = ("fused_qkvg_proj", "f_a_proj", "b_proj")

# The shared-expert down projection is the one weight K3 consumes as a raw
# dense tensor rather than through ``quant_method``:
#
# * ``_run_shared_down`` falls back to ``_k3_bf16_gemm(x, down_proj.weight)``,
#   which reads ``[out, in]``. The loader transposes a quantized weight to
#   ``[in, out]``, so converting it turns that fallback into a shape error.
# * ``_eligible_for_fused_front`` requires ``down_proj.weight.dtype`` to be
#   bf16/fp16, so converting it silently drops the fused-front collective.
#
# FP8 for this weight is already available, and done correctly, via
# ``SGLANG_ROCM_K3_PTPC_FP8_SHARED_DOWN``: it packs a *separate* FP8 copy into
# ``_shared_down_fp8_w`` and leaves the BF16 weight live for both paths above.
_SHARED_SKIP_SUFFIXES = ("down_proj",)


class _K3OnlineFp8LinearMethod(LinearMethodBase):
    """Linear method for a layer quantized after its weights were loaded.

    ``create_weights`` is unreachable by construction: the layer already owns
    real weights by the time this method replaces the unquantized one.
    """

    def __init__(self, scheme):
        self.scheme = scheme
        self.quant_config = None

    def create_weights(self, *args, **kwargs):
        raise RuntimeError(
            "K3 online FP8 quantization runs after create_weights; the layer "
            "cannot allocate through it."
        )

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        # Loaders invoke this once, but the capture-overlap path runs the whole
        # post-load sequence twice (sentinel weights, then real ones) and the
        # scheme transposes in place, so make the second call a no-op.
        if getattr(layer, "_k3_online_fp8_processed", False):
            return
        layer.scheme.process_weights_after_loading(layer)
        layer._k3_online_fp8_processed = True

    def apply(
        self,
        layer: nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return layer.scheme.apply_weights(layer, x, bias)


def _build_scheme():
    """A Quark W8A8-FP8 scheme with per-channel weights and per-token acts."""
    from sglang.srt.layers.quantization.quark.schemes.quark_w8a8_fp8 import QuarkW8A8Fp8

    return QuarkW8A8Fp8(
        weight_config={"qscheme": "per_channel"},
        input_config={"is_dynamic": True, "qscheme": "per_channel"},
    )


def _quantize_per_output_channel(
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric per-output-channel e4m3 quantization of an ``[out, in]`` weight.

    Row-wise is the only split that stays exact under tensor parallelism: a
    column-parallel shard owns whole output rows, and a row-parallel shard's
    scales apply to its own partial product, which the all-reduce then sums.
    """
    finfo = torch.finfo(torch.float8_e4m3fn)
    out_features = weight.shape[0]
    scale = torch.empty(out_features, dtype=torch.float32, device=weight.device)
    quantized = torch.empty_like(weight, dtype=torch.float8_e4m3fn)
    for start in range(0, out_features, _QUANT_CHUNK_ROWS):
        stop = min(start + _QUANT_CHUNK_ROWS, out_features)
        rows = weight[start:stop].to(torch.float32)
        # clamp(min) keeps an all-zero row (padding, or a pruned head) from
        # producing a zero scale the dequant would divide by.
        chunk_scale = (rows.abs().amax(dim=1) / finfo.max).clamp(min=1e-12)
        scale[start:stop] = chunk_scale
        quantized[start:stop] = (
            rows.div_(chunk_scale.unsqueeze(1))
            .clamp_(finfo.min, finfo.max)
            .to(torch.float8_e4m3fn)
        )
    return quantized, scale


def _is_convertible(module: nn.Module) -> bool:
    if not isinstance(module, LinearBase):
        return False
    if getattr(module, "scheme", None) is not None:
        return False
    weight = getattr(module, "weight", None)
    if not isinstance(weight, torch.Tensor) or weight.dim() != 2:
        return False
    if weight.dtype != torch.bfloat16:
        return False
    # A quantized checkpoint carries its scales alongside; never re-quantize.
    for name in ("weight_scale", "weight_scale_inv", "weight_packed"):
        if getattr(module, name, None) is not None:
            return False
    return True


def _selected_modules(model: nn.Module) -> Iterator[tuple[str, nn.Module]]:
    do_attn = envs.SGLANG_ROCM_K3_ONLINE_FP8_ATTN.get()
    do_kda_inproj = envs.SGLANG_ROCM_K3_ONLINE_FP8_KDA_INPROJ.get()
    do_shared = envs.SGLANG_ROCM_K3_ONLINE_FP8_SHARED_EXPERTS.get()
    for name, module in model.named_modules():
        # The vision tower and projector run replicated on a handful of tokens;
        # the router gate must stay BF16 because FP8 logits move the top-k pick.
        if "vision_tower" in name or "mm_projector" in name:
            continue
        leaf = name.rsplit(".", 1)[-1]
        if leaf in _SKIP_SUFFIXES or leaf == "gate":
            continue
        in_attn = ".self_attn." in name
        in_shared = ".shared_experts." in name
        if in_attn and leaf in _KDA_INPROJ_SUFFIXES:
            wanted = do_kda_inproj
        elif in_attn:
            wanted = do_attn
        elif in_shared:
            wanted = do_shared and leaf not in _SHARED_SKIP_SUFFIXES
        else:
            wanted = False
        if not wanted or not _is_convertible(module):
            continue
        yield name, module


def maybe_quantize_bf16_linears_fp8(model: nn.Module) -> None:
    """Quantize the selected BF16 linears to per-channel FP8 in place.

    A no-op off ROCm, with both switches unset, or on a checkpoint that already
    ships quantized weights.
    """
    if not _is_hip:
        return
    if not any(
        field.get()
        for field in (
            envs.SGLANG_ROCM_K3_ONLINE_FP8_ATTN,
            envs.SGLANG_ROCM_K3_ONLINE_FP8_KDA_INPROJ,
            envs.SGLANG_ROCM_K3_ONLINE_FP8_SHARED_EXPERTS,
        )
    ):
        return

    converted = 0
    saved_bytes = 0
    for _, module in _selected_modules(model):
        weight = module.weight.data
        quantized, scale = _quantize_per_output_channel(weight)
        saved_bytes += weight.numel()  # bf16 -> fp8 halves the footprint
        module.weight = Parameter(quantized, requires_grad=False)
        module.weight_scale = Parameter(scale, requires_grad=False)
        module.input_scale = None
        module.scheme = _build_scheme()
        # The loader's postprocess pass runs after post_load_weights, so the
        # transpose/bpreshuffle lands there -- exactly where a Quark checkpoint
        # gets it, and after the merges below have read the [out, in] layout.
        module.quant_method = _K3OnlineFp8LinearMethod(module.scheme)
        module._k3_online_fp8_processed = False
        converted += 1

    if converted:
        logger.info(
            "K3 online FP8: quantized %d BF16 linears to per-channel FP8 "
            "(%.2f GiB of weights freed per rank)",
            converted,
            saved_bytes / (1 << 30),
        )
