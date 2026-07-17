# Copyright 2023-2026 SGLang Team
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

"""MiMo V2 adapters owned by the context-parallel runtime."""

from __future__ import annotations

import math
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator, List, Optional, Sequence, Tuple

import torch

from sglang.srt.configs.model_config import get_mimo_v2_fused_qkv_expected_tp_size
from sglang.srt.layers.linear import QKVParallelLinear
from sglang.srt.layers.quantization.fp8_utils import block_quant_dequant
from sglang.srt.runtime_context import get_parallel


def maybe_get_mimo_v2_cp_input_embedding(
    model, input_ids: torch.Tensor
) -> Optional[torch.Tensor]:
    """Return CP-ready draft embeddings for MiMo V2 MTP models."""

    config = getattr(model, "config", None)
    architectures = getattr(config, "architectures", None) or []
    if "MiMoV2MTP" not in architectures:
        return None

    embedding = model.get_input_embeddings()
    if embedding is None:
        raise AttributeError("MiMoV2MTP has no input embedding layer")
    vocab_size = int(config.vocab_size)
    return embedding(input_ids.clamp(min=0, max=vocab_size - 1))


def _block_quantize_fp8(
    weight: torch.Tensor,
    block_size: Sequence[int],
    fp8_dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    block_n, block_k = (int(block_size[0]), int(block_size[1]))
    n, k = weight.shape
    padded_n = math.ceil(n / block_n) * block_n
    padded_k = math.ceil(k / block_k) * block_k
    padded = torch.zeros(
        (padded_n, padded_k),
        dtype=torch.float32,
        device=weight.device,
    )
    padded[:n, :k] = weight.to(torch.float32)
    blocks = padded.view(
        padded_n // block_n,
        block_n,
        padded_k // block_k,
        block_k,
    )

    fp8_info = torch.finfo(fp8_dtype)
    scale = (blocks.abs().amax(dim=(1, 3)) / fp8_info.max).clamp(min=1e-12)
    quantized = (
        (blocks / scale[:, None, :, None])
        .clamp(min=fp8_info.min, max=fp8_info.max)
        .to(fp8_dtype)
    )
    return (
        quantized.view(padded_n, padded_k)[:n, :k].contiguous(),
        scale.to(torch.float32).contiguous(),
    )


def repack_mimo_v2_fused_qkv_block_fp8(
    checkpoint_weight: torch.Tensor,
    checkpoint_scale: torch.Tensor,
    *,
    q_rows: int,
    k_rows: int,
    v_rows: int,
    checkpoint_tp_size: int,
    block_size: Sequence[int],
    output_dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert checkpoint ``[Q_i,K_i,V_i]`` groups to runtime ``[Q,K,V]``.

    Each serialized TP group restarts blockwise FP8 scaling. Since Q/K/V
    boundaries are generally not block-aligned, the conversion must dequantize
    each group before globally requantizing the canonical runtime layout.
    """

    if checkpoint_tp_size <= 1:
        raise ValueError("checkpoint_tp_size must be greater than one")
    if any(rows % checkpoint_tp_size for rows in (q_rows, k_rows, v_rows)):
        raise ValueError(
            "MiMo V2 fused QKV row counts must be divisible by checkpoint TP size"
        )

    q_group_rows = q_rows // checkpoint_tp_size
    k_group_rows = k_rows // checkpoint_tp_size
    v_group_rows = v_rows // checkpoint_tp_size
    group_rows = q_group_rows + k_group_rows + v_group_rows
    expected_weight_shape = (q_rows + k_rows + v_rows, checkpoint_weight.shape[1])
    if tuple(checkpoint_weight.shape) != expected_weight_shape:
        raise ValueError(
            f"Unexpected MiMo V2 fused QKV weight shape "
            f"{tuple(checkpoint_weight.shape)}; expected {expected_weight_shape}."
        )

    block_n, block_k = (int(block_size[0]), int(block_size[1]))
    group_scale_rows = math.ceil(group_rows / block_n)
    expected_scale_shape = (
        checkpoint_tp_size * group_scale_rows,
        math.ceil(checkpoint_weight.shape[1] / block_k),
    )
    if tuple(checkpoint_scale.shape) != expected_scale_shape:
        raise ValueError(
            f"Unexpected MiMo V2 fused QKV scale shape "
            f"{tuple(checkpoint_scale.shape)}; expected {expected_scale_shape}."
        )

    q_parts: List[torch.Tensor] = []
    k_parts: List[torch.Tensor] = []
    v_parts: List[torch.Tensor] = []
    for group_index in range(checkpoint_tp_size):
        weight_start = group_index * group_rows
        scale_start = group_index * group_scale_rows
        group = block_quant_dequant(
            checkpoint_weight[weight_start : weight_start + group_rows],
            checkpoint_scale[scale_start : scale_start + group_scale_rows].to(
                torch.float32
            ),
            list(block_size),
            output_dtype,
        )
        q_part, k_part, v_part = group.split(
            [q_group_rows, k_group_rows, v_group_rows], dim=0
        )
        q_parts.append(q_part)
        k_parts.append(k_part)
        v_parts.append(v_part)

    canonical_weight = torch.cat([*q_parts, *k_parts, *v_parts], dim=0)
    return _block_quantize_fp8(
        canonical_weight,
        block_size,
        checkpoint_weight.dtype,
    )


@dataclass
class _MiMoQKVAdaptation:
    module_name: str
    linear: QKVParallelLinear
    checkpoint_tp_size: int
    block_size: List[int]
    original_scale_data: torch.Tensor
    original_scale_format_ue8m0: bool

    @property
    def checkpoint_scale_shape(self) -> Tuple[int, int]:
        group_rows = (
            self.linear.q_proj_shard_size
            + self.linear.kv_proj_shard_size
            + self.linear.v_proj_shard_size
        ) // self.checkpoint_tp_size
        block_n, block_k = self.block_size
        return (
            self.checkpoint_tp_size * math.ceil(group_rows / block_n),
            math.ceil(self.linear.weight.shape[1] / block_k),
        )

    @property
    def target_scale_shape(self) -> Tuple[int, int]:
        block_n, block_k = self.block_size
        return (
            math.ceil(self.linear.weight.shape[0] / block_n),
            math.ceil(self.linear.weight.shape[1] / block_k),
        )

    def prepare(self) -> None:
        scale = self.linear.weight_scale_inv
        scale.data = torch.empty(
            self.checkpoint_scale_shape,
            dtype=torch.float32,
            device=scale.device,
        )
        scale.format_ue8m0 = False

    def restore(self) -> None:
        scale = self.linear.weight_scale_inv
        scale.data = self.original_scale_data
        scale.format_ue8m0 = self.original_scale_format_ue8m0

    def finish(self) -> None:
        scale = self.linear.weight_scale_inv
        qweight, qscale = repack_mimo_v2_fused_qkv_block_fp8(
            self.linear.weight.data,
            scale.data,
            q_rows=self.linear.q_proj_shard_size,
            k_rows=self.linear.kv_proj_shard_size,
            v_rows=self.linear.v_proj_shard_size,
            checkpoint_tp_size=self.checkpoint_tp_size,
            block_size=self.block_size,
            output_dtype=torch.float32,
        )
        if tuple(qweight.shape) != tuple(self.linear.weight.shape):
            raise ValueError(
                f"Repacked {self.module_name} weight has shape "
                f"{tuple(qweight.shape)}, expected {tuple(self.linear.weight.shape)}."
            )
        if tuple(qscale.shape) != self.target_scale_shape:
            raise ValueError(
                f"Repacked {self.module_name} scale has shape "
                f"{tuple(qscale.shape)}, expected "
                f"{self.target_scale_shape}."
            )
        self.linear.weight.data = qweight.to(
            device=self.linear.weight.device, dtype=self.linear.weight.dtype
        )
        scale.data = qscale.to(device=scale.device, dtype=torch.float32)
        scale.format_ue8m0 = False


def _collect_mimo_qkv_adaptations(model) -> List[_MiMoQKVAdaptation]:
    config = getattr(model, "config", None)
    if config is None:
        return []
    architectures = getattr(config, "architectures", None) or []
    if not any(
        architecture in ("MiMoV2ForCausalLM", "MiMoV2FlashForCausalLM", "MiMoV2MTP")
        for architecture in architectures
    ):
        return []
    checkpoint_tp_size = get_mimo_v2_fused_qkv_expected_tp_size(config)
    if checkpoint_tp_size is None or checkpoint_tp_size <= 1:
        return []

    parallel = get_parallel()
    if parallel.attn_cp_size <= 1 or parallel.attn_tp_size != 1:
        return []

    adaptations: List[_MiMoQKVAdaptation] = []
    for module_name, module in model.named_modules():
        if not isinstance(module, QKVParallelLinear):
            continue
        if not module_name.startswith("model.") or not module_name.endswith(
            ".self_attn.qkv_proj"
        ):
            continue
        scale = getattr(module, "weight_scale_inv", None)
        quant_method = getattr(module, "quant_method", None)
        quant_config = getattr(quant_method, "quant_config", None)
        block_size = getattr(quant_config, "weight_block_size", None)
        if scale is None or block_size is None or len(block_size) != 2:
            raise ValueError(
                "Collapsed MiMo V2 attention TP currently requires a serialized "
                f"block-FP8 qkv_proj; unsupported module {module_name}."
            )
        if getattr(quant_method, "use_mxfp8", False):
            raise ValueError(
                "Collapsed MiMo V2 attention TP does not support serialized "
                f"MXFP8 QKV scales in {module_name}."
            )
        adaptations.append(
            _MiMoQKVAdaptation(
                module_name=module_name,
                linear=module,
                checkpoint_tp_size=checkpoint_tp_size,
                block_size=[int(block_size[0]), int(block_size[1])],
                original_scale_data=scale.data,
                original_scale_format_ue8m0=bool(getattr(scale, "format_ue8m0", False)),
            )
        )

    if not adaptations:
        raise ValueError(
            "Collapsed MiMo V2 attention TP was requested, but no language "
            "qkv_proj modules were available for the CP weight adapter."
        )
    return adaptations


@contextmanager
def maybe_adapt_mimo_v2_fused_qkv_for_cp(model) -> Iterator[None]:
    """Adapt serialized MiMo TP-interleaved QKV weights for collapsed CP TP."""

    adaptations = _collect_mimo_qkv_adaptations(model)
    for adaptation in adaptations:
        adaptation.prepare()
    try:
        yield
        for adaptation in adaptations:
            adaptation.finish()
    except BaseException:
        for adaptation in adaptations:
            adaptation.restore()
        raise
