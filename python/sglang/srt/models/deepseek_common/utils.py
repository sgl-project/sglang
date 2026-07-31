# Copyright 2026 SGLang Team
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
import logging
import math
from typing import Iterable, List, Optional, Tuple, Union

import torch

from sglang.kernels.ops.quantization.fp8_kernel import is_fp8_fnuz
from sglang.srt.environ import envs
from sglang.srt.layers.moe.fused_moe_triton.layer import get_moe_runner_backend
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.utils import (
    cpu_has_amx_support,
    get_bool_env_var,
    get_device_sm,
    get_hip_version,
    is_cpu,
    is_cuda,
    is_gfx95_supported,
    is_hip,
    is_musa,
    is_npu,
    is_nvidia_cublas_version_ge_12_9,
    is_xpu,
)

_is_hip = is_hip()
_is_cuda = is_cuda()
_is_npu = is_npu()
_is_musa = is_musa()
_is_fp8_fnuz = is_fp8_fnuz()
_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip
_is_cpu_amx_available = cpu_has_amx_support()
_is_cpu = is_cpu()
_is_xpu = is_xpu()
_device_sm = get_device_sm()
_is_gfx95_supported = is_gfx95_supported()
_use_aiter_gfx95 = _use_aiter and _is_gfx95_supported
_use_aiter_bpreshuffle_gfx95 = _use_aiter_gfx95 and get_hip_version() >= (7, 2, 0)


_is_cublas_ge_129 = is_nvidia_cublas_version_ge_12_9()

logger = logging.getLogger(__name__)

NVFP4_CKPT_FP8_ATTN_QUANT_MODULES = ["q_b_proj"]

FORWARD_ABSORB_CORE_ATTENTION_BACKENDS = [
    "fa3",
    "fa4",
    "dsa",
    "nsa",  # Deprecated alias for "dsa"
    "flashinfer",
    "cutlass_mla",
    "trtllm_mla",
    "cutedsl_mla",
    "tokenspeed_mla",
    "ascend",
    "intel_xpu",
]


def get_dsv4_c4_layer_ids(compress_ratios: Iterable[int]) -> List[int]:
    return [idx for idx, ratio in enumerate(compress_ratios) if ratio == 4]


def compute_dsv4_index_topk_flags(
    compress_ratios: List[int],
    layer_id: int,
    index_topk_freq: int = 1,
    index_topk_pattern: Optional[Union[str, List[str]]] = None,
) -> Tuple[bool, bool]:
    if index_topk_freq is None:
        index_topk_freq = 1
    if (
        isinstance(index_topk_freq, bool)
        or not isinstance(index_topk_freq, int)
        or index_topk_freq <= 0
    ):
        raise ValueError(
            f"index_topk_freq must be a positive integer, got {index_topk_freq}"
        )
    c4_layer_ids = get_dsv4_c4_layer_ids(compress_ratios)
    c4_layer_rank = c4_layer_ids.index(layer_id)

    if index_topk_pattern is None:
        skip_topk = c4_layer_rank % index_topk_freq != 0
        next_skip_topk = (
            c4_layer_rank + 1 < len(c4_layer_ids)
            and (c4_layer_rank + 1) % index_topk_freq != 0
        )
        return skip_topk, next_skip_topk

    if isinstance(index_topk_pattern, str):
        index_topk_pattern = list(index_topk_pattern)
    invalid_pattern_values = set(index_topk_pattern) - {"F", "S"}
    if invalid_pattern_values:
        raise ValueError(
            "index_topk_pattern only supports 'F' for full indexer "
            f"layers and 'S' for shared layers, got "
            f"{sorted(invalid_pattern_values)}"
        )
    if len(index_topk_pattern) == len(compress_ratios):
        pattern_idx = layer_id
        has_next_pattern = c4_layer_rank < len(c4_layer_ids) - 1
        next_pattern_idx = c4_layer_ids[c4_layer_rank + 1] if has_next_pattern else None
    elif len(index_topk_pattern) == len(c4_layer_ids):
        pattern_idx = c4_layer_rank
        has_next_pattern = c4_layer_rank < len(c4_layer_ids) - 1
        next_pattern_idx = c4_layer_rank + 1 if has_next_pattern else None
    else:
        raise ValueError(
            "index_topk_pattern length must either match "
            f"num_hidden_layers ({len(compress_ratios)}) or "
            f"the number of C4 layers ({len(c4_layer_ids)}), got "
            f"{len(index_topk_pattern)}"
        )

    skip_topk = index_topk_pattern[pattern_idx] == "S"
    if c4_layer_rank == 0 and skip_topk:
        raise ValueError(
            "index_topk_pattern marks the first C4 layer as 'S' (shared), but "
            "the first C4 layer must be 'F' (a full producer): there is no "
            "prior producer whose raw top-k it could reuse. Set the first C4 "
            f"layer to 'F'. (pattern={index_topk_pattern})"
        )
    next_skip_topk = (
        index_topk_pattern[next_pattern_idx] == "S"
        if next_pattern_idx is not None
        else False
    )
    return skip_topk, next_skip_topk


def compute_dsv4_index_topk_flags_for_all_c4_layers(
    compress_ratios: List[int],
    index_topk_freq: int = 1,
    index_topk_pattern: Optional[Union[str, List[str]]] = None,
) -> "dict[int, Tuple[bool, bool]]":
    """Map each C4 layer id -> (skip_topk, next_skip_topk)."""
    return {
        layer_id: compute_dsv4_index_topk_flags(
            compress_ratios, layer_id, index_topk_freq, index_topk_pattern
        )
        for layer_id in get_dsv4_c4_layer_ids(compress_ratios)
    }


def dsv4_index_cache_enabled(
    compress_ratios: List[int],
    index_topk_freq: int = 1,
    index_topk_pattern: Optional[Union[str, List[str]]] = None,
) -> bool:
    flags = compute_dsv4_index_topk_flags_for_all_c4_layers(
        compress_ratios, index_topk_freq, index_topk_pattern
    )
    return any(skip_topk for skip_topk, _ in flags.values())


def dsv4_index_cache_producer_layer_ids(
    compress_ratios: List[int],
    index_topk_freq: int = 1,
    index_topk_pattern: Optional[Union[str, List[str]]] = None,
) -> List[int]:
    flags = compute_dsv4_index_topk_flags_for_all_c4_layers(
        compress_ratios, index_topk_freq, index_topk_pattern
    )
    return sorted(
        layer_id for layer_id, (skip_topk, _) in flags.items() if not skip_topk
    )


def compute_dsv4_index_cache_descriptor(
    hf_config,
    *,
    fp4_indexer_enabled: bool,
    page_size: int,
) -> "Tuple[str, List[int]]":
    import hashlib
    import json

    compress_ratios = list(getattr(hf_config, "compress_ratios", None) or [])
    index_topk_freq = getattr(hf_config, "index_topk_freq", 1)
    index_topk_pattern = getattr(hf_config, "index_topk_pattern", None)

    layout = {
        "schema_version": 1,
        "num_hidden_layers": hf_config.num_hidden_layers,
        "compress_ratios": compress_ratios,
        "index_topk": getattr(hf_config, "index_topk", 512),
        "index_head_dim": getattr(hf_config, "index_head_dim", 128),
        "index_n_heads": getattr(hf_config, "index_n_heads", 64),
        "fp4_indexer_enabled": bool(fp4_indexer_enabled),
        "page_size": page_size,
    }
    layout_signature = hashlib.sha256(
        json.dumps(layout, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    producer_layer_ids = dsv4_index_cache_producer_layer_ids(
        compress_ratios, index_topk_freq, index_topk_pattern
    )
    return layout_signature, producer_layer_ids


def validate_dsv4_index_cache_pd_compatibility(
    *,
    prefill_layout_signature: str,
    prefill_producer_layer_ids: List[int],
    decode_layout_signature: str,
    decode_producer_layer_ids: List[int],
) -> None:
    if prefill_layout_signature != decode_layout_signature:
        raise RuntimeError(
            "DSV4 IndexCache layout mismatch between prefill and decode: "
            f"prefill={prefill_layout_signature} decode={decode_layout_signature}"
        )
    missing = set(decode_producer_layer_ids) - set(prefill_producer_layer_ids)
    if missing:
        raise RuntimeError(
            "DSV4 IndexCache producer coverage violation: prefill does not "
            f"provide required indexer cache for layers {sorted(missing)}. "
            "A decode engine's producer set must be a subset of prefill's."
        )


def awq_dequantize_func():
    """
    Get the AWQ dequantize function for the current device

    Return:
        - The AWQ dequantize function for the current device.
        - None if the current device is not supported.
    """
    if _is_cuda:
        from sgl_kernel import awq_dequantize

        return awq_dequantize
    elif _is_hip:
        from sglang.kernel_api_logging import debug_kernel_api
        from sglang.kernels.ops.quantization.awq_triton import (
            awq_dequantize_triton as awq_dequantize,
        )

        return debug_kernel_api(awq_dequantize, op_name="DeepseekCommon.awq_dequantize")
    elif _is_npu:
        from sglang.kernel_api_logging import debug_kernel_api
        from sglang.kernels.ops.quantization.awq_triton import (
            awq_dequantize_decomposition as awq_dequantize,
        )

        return debug_kernel_api(awq_dequantize, op_name="DeepseekCommon.awq_dequantize")
    else:
        return None


def enable_nextn_moe_bf16_cast_to_fp8(
    quant_config: Optional[QuantizationConfig],
) -> bool:
    return (
        envs.SGLANG_NVFP4_CKPT_FP8_NEXTN_MOE.get()
        and quant_config is not None
        and quant_config.get_name() == "modelopt_fp4"
        and get_moe_runner_backend().is_deep_gemm()
    )


def is_wint4afp8_or_wint4a16_config(
    quant_config: Optional[QuantizationConfig],
) -> bool:
    if quant_config is None:
        return False
    if quant_config.get_name() == "w4afp8":
        return True

    from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
        CompressedTensorsConfig,
    )

    if not isinstance(quant_config, CompressedTensorsConfig):
        return False
    linear_scheme = quant_config.target_scheme_map.get("Linear", {})
    weight_quant = linear_scheme.get("weights")
    input_quant = linear_scheme.get("input_activations")
    return quant_config._is_wint4afp8(
        weight_quant, input_quant
    ) or quant_config._is_wint4abf16(weight_quant, input_quant)


def yarn_get_mscale(scale: float = 1, mscale: float = 1) -> float:
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


def _get_llama_4_scaling(
    original_max_position_embeddings: int, scaling_beta: float, positions: torch.Tensor
) -> torch.Tensor:
    scaling = 1 + scaling_beta * torch.log(
        1 + torch.floor(positions / original_max_position_embeddings)
    )
    return scaling[..., None, None]
