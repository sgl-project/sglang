# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from: https://github.com/vllm-project/vllm/blob/ab3e80042eac24dd362408e6d63ad98768046359/vllm/model_executor/layers/quantization/gguf.py
from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING, Any, List, Optional

import gguf
import torch
from gguf import GGMLQuantizationType as WeightType
from torch.nn.parameter import Parameter, UninitializedParameter

from sglang.srt.hardware_backend.npu.quantization.moe_methods import (
    NPUUnquantMoEMethod,
)
from sglang.srt.hardware_backend.npu.utils import npu_format_cast
from sglang.srt.layers.linear import LinearBase
from sglang.srt.layers.moe.moe_runner import MoeRunner, MoeRunnerConfig
from sglang.srt.layers.moe.utils import MoeRunnerBackend, get_moe_runner_backend
from sglang.srt.layers.quantization.base_config import (
    FusedMoEMethodBase,
    LinearMethodBase,
    QuantizationConfig,
    QuantizeMethodBase,
)
from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
from sglang.srt.utils import is_cuda, is_hip, is_musa, is_npu, is_xpu, set_weight_attrs

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher import (
        CombineInput,
        StandardDispatchOutput,
    )

_is_cuda = is_cuda()
_is_hip = is_hip()
_is_xpu = is_xpu()
_is_musa = is_musa()
_is_npu = is_npu()

if _is_cuda:
    from sgl_kernel import moe_align_block_size, moe_sum
    from sgl_kernel.quantization import (
        ggml_dequantize,
        ggml_moe_a8,
        ggml_moe_a8_vec,
        ggml_moe_get_block_size,
        ggml_mul_mat_a8,
        ggml_mul_mat_vec_a8,
    )

    from sglang.kernels.ops.activation.activation import gelu_and_mul, silu_and_mul
elif _is_musa:
    from sgl_kernel import gelu_and_mul, moe_align_block_size, moe_sum, silu_and_mul
    from sgl_kernel.quantization import (
        ggml_dequantize,
        ggml_moe_a8,
        ggml_moe_a8_vec,
        ggml_moe_get_block_size,
        ggml_mul_mat_a8,
        ggml_mul_mat_vec_a8,
    )
elif _is_xpu:
    from gguf import dequantize as gguf_dequantize
elif _is_npu:
    from gguf import dequantize as gguf_dequantize
else:
    if not _is_hip:
        warnings.warn(
            "Only CUDA, XPU, MUSA and NPU support GGUF quantization currently."
        )

logger = logging.getLogger(__name__)


def _ordered_gguf_shard_ids(shard_ids: list) -> list:
    """Return checkpoint shards in the fused layer's logical output order."""
    if len(shard_ids) == 3 and set(shard_ids) == {"q", "k", "v"}:
        return ["q", "k", "v"]
    if all(isinstance(shard_id, int) for shard_id in shard_ids) and set(
        shard_ids
    ) == set(range(len(shard_ids))):
        return sorted(shard_ids)
    return list(shard_ids)


class GGUFConfig(QuantizationConfig):
    """Config class for GGUF."""

    def __init__(self, modules_to_not_convert: list[str] | None = None) -> None:
        super().__init__()
        if _is_hip:
            warnings.warn(f"Only CUDA and MUSA support GGUF quantization currently.")
        self.modules_to_not_convert = modules_to_not_convert or []

    def __repr__(self) -> str:
        return "GGUFConfig()"

    def get_scaled_act_names(self) -> List[str]:
        return []

    def get_name(self) -> str:
        return "gguf"

    def get_supported_act_dtypes(self) -> list[torch.dtype]:
        return [torch.half, torch.bfloat16, torch.float32]

    @classmethod
    def get_min_capability(cls) -> int:
        return 60 if not _is_musa else 21

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return []  # no extra configs.

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> GGUFConfig:
        modules_to_not_convert = cls.get_from_keys_or(
            config, ["modules_to_not_convert"], None
        )
        return cls(modules_to_not_convert)

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Optional[QuantizeMethodBase]:
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
        from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding

        if isinstance(layer, LinearBase):
            if is_layer_skipped_gguf(prefix, self.modules_to_not_convert):
                return UnquantizedLinearMethod()
            if _is_xpu:
                return GGUFLinearXPUMethod(self)
            if _is_npu:
                return GGUFLinearAscendMethod(self)
            return GGUFLinearMethod(self)
        elif isinstance(layer, VocabParallelEmbedding):
            if _is_xpu:
                return GGUFEmbeddingXPUMethod(self)
            if _is_npu:
                return GGUFEmbeddingAscendMethod(self)
            return GGUFEmbeddingMethod(self)
        elif isinstance(layer, FusedMoE):
            if _is_npu:
                return GGUFMoEAscendMethod(self)
            if _is_xpu:
                return GGUFMoEXPUMethod(self)
            return GGUFMoEMethod(self)
        return None


def is_layer_skipped_gguf(prefix: str, modules_to_not_convert: list[str]):
    return any(module_name in prefix for module_name in modules_to_not_convert)


UNQUANTIZED_TYPES = {WeightType.F32, WeightType.F16, WeightType.BF16}
STANDARD_QUANT_TYPES = {
    WeightType.Q4_0,
    WeightType.Q4_1,
    WeightType.Q5_0,
    WeightType.Q5_1,
    WeightType.Q8_0,
    WeightType.Q8_1,
}
KQUANT_TYPES = {
    WeightType.Q2_K,
    WeightType.Q3_K,
    WeightType.Q4_K,
    WeightType.Q5_K,
    WeightType.Q6_K,
}
IMATRIX_QUANT_TYPES = {
    WeightType.IQ1_M,
    WeightType.IQ1_S,
    WeightType.IQ2_XXS,
    WeightType.IQ2_XS,
    WeightType.IQ2_S,
    WeightType.IQ3_XXS,
    WeightType.IQ3_S,
    WeightType.IQ4_XS,
    WeightType.IQ4_NL,
}
# TODO(Isotr0py): Currently, we don't have MMQ kernel for I-Matrix quantization.
# Consolidate DEQUANT_TYPES, MMVQ_QUANT_TYPES and MMQ_QUANT_TYPES after we add
# MMQ kernel for I-Matrix quantization.
DEQUANT_TYPES = STANDARD_QUANT_TYPES | KQUANT_TYPES | IMATRIX_QUANT_TYPES
MMVQ_QUANT_TYPES = STANDARD_QUANT_TYPES | KQUANT_TYPES | IMATRIX_QUANT_TYPES
MMQ_QUANT_TYPES = STANDARD_QUANT_TYPES | KQUANT_TYPES


def dequantize_gguf_weight(
    qweight: torch.Tensor, qweight_type: int, dtype: torch.dtype
) -> torch.Tensor:
    """Dequantize a packed GGUF matrix using its inferred logical shape."""
    block_size, type_size = gguf.GGML_QUANT_SIZES[qweight_type]
    shape = (qweight.shape[0], qweight.shape[1] // type_size * block_size)
    return ggml_dequantize(qweight, qweight_type, *shape, dtype)


def fused_mul_mat_gguf(
    x: torch.Tensor, qweight: torch.Tensor, qweight_type: int
) -> torch.Tensor:
    if qweight_type in IMATRIX_QUANT_TYPES:
        mmvq_safe = 8 if qweight.shape[0] > 5120 else 16
    else:
        mmvq_safe = 2 if qweight.shape[0] > 5120 else 6
    # HACK: when doing chunked prefill we don't generate output tokens
    # so input to logits generator is empty which causes invalid parameter
    if x.shape[0] == 0:
        return torch.empty(x.shape[0], qweight.shape[0], dtype=x.dtype, device=x.device)
    # there is no need to call any kernel for fp16/bf16
    if qweight_type in UNQUANTIZED_TYPES:
        return x @ qweight.T
    # enable MMVQ in contiguous batching with batch_size=1
    if x.shape[0] <= mmvq_safe and qweight_type in MMVQ_QUANT_TYPES:
        y = ggml_mul_mat_vec_a8(qweight, x, qweight_type, qweight.shape[0])
    # Use MMQ Kernel if it's available (standard + k-quants)
    elif qweight_type in MMQ_QUANT_TYPES:
        y = ggml_mul_mat_a8(qweight, x, qweight_type, qweight.shape[0])
    # If there is no available MMQ kernel, fallback to dequantize
    elif qweight_type in DEQUANT_TYPES:
        weight = dequantize_gguf_weight(qweight, qweight_type, x.dtype)
        y = x @ weight.T
    else:
        # Raise an error if the quantization type is not supported.
        # Might be useful if llama.cpp adds a new quantization type.
        # Wrap to GGMLQuantizationType IntEnum to make sure it's a valid type.
        qweight_type = WeightType(qweight_type)
        raise NotImplementedError(f"Unsupported GGUF quantization type: {qweight_type}")
    return y


def fused_moe_gguf(
    x: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    qweight_type: int,
    qweight_type2: int,
    activation: str,
) -> torch.Tensor:
    def act(x: torch.Tensor):
        if activation == "silu":
            return silu_and_mul(x)
        elif activation == "gelu":
            return gelu_and_mul(x)
        raise ValueError(f"Unsupported activation: {activation}")

    out_hidden_states = torch.empty_like(x)
    # unless we decent expert reuse we are better off running moe_vec kernel
    if (
        qweight_type2 in MMQ_QUANT_TYPES
        and qweight_type in MMQ_QUANT_TYPES
        and x.shape[0] > 64
    ):
        num_tokens, _ = x.shape
        E, N, _ = w1.shape
        top_k = topk_ids.shape[1]
        BLOCK_SIZE = ggml_moe_get_block_size(qweight_type)

        sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
            topk_ids, BLOCK_SIZE, E
        )
        out = ggml_moe_a8(
            x,
            w1,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            qweight_type,
            N,
            top_k,
            num_tokens,
        )
        out = act(out)
        out = ggml_moe_a8(
            out,
            w2,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            qweight_type2,
            w2.shape[1],
            1,
            num_tokens * top_k,
        )
        out = out.reshape(num_tokens, top_k, w2.shape[1]).mul_(
            topk_weights.view(num_tokens, top_k, 1)
        )
        # TODO(FlamingoPg): maybe we can use moe_sum_reduce here?
        moe_sum(out, out_hidden_states)
    elif qweight_type2 in MMVQ_QUANT_TYPES and qweight_type in MMVQ_QUANT_TYPES:
        num_tokens, _ = x.shape
        E, N, _ = w1.shape
        top_k = topk_ids.shape[1]

        out = ggml_moe_a8_vec(x, w1, topk_ids, top_k, qweight_type, N, num_tokens)
        out = act(out)

        out = ggml_moe_a8_vec(
            out, w2, topk_ids, 1, qweight_type2, w2.shape[1], num_tokens * top_k
        )
        out = out.reshape(num_tokens, top_k, w2.shape[1]).mul_(
            topk_weights.view(num_tokens, top_k, 1)
        )
        moe_sum(out, out_hidden_states)
    else:
        logger.warning_once(
            "There is no support for fast MoE kernel "
            "for current quantization method. "
            "Falling back to slow implementation. "
        )
        for tok, (w, idx) in enumerate(zip(topk_weights, topk_ids)):
            inp = x[tok].reshape((1,) + x.shape[1:])
            current_hidden_state = None
            for ww, ii in zip(w, idx):
                expert_up = w1[ii]

                out = fused_mul_mat_gguf(inp, expert_up, qweight_type)
                out = act(out)

                expert_down = w2[ii]
                current_state = fused_mul_mat_gguf(
                    out, expert_down, qweight_type2
                ).mul_(ww)
                if current_hidden_state is None:
                    current_hidden_state = current_state
                else:
                    current_hidden_state.add_(current_state)
            out_hidden_states[tok] = current_hidden_state
    return out_hidden_states


def apply_gguf_embedding(
    x: torch.Tensor,
    qweight: torch.Tensor,
    qweight_type: int,
    hidden_size: int,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    if qweight_type in UNQUANTIZED_TYPES:
        return torch.embedding(qweight, x)
    elif qweight_type in DEQUANT_TYPES:
        block_size, type_size = gguf.GGML_QUANT_SIZES[qweight_type]
        x_flat = x.flatten()
        assert hidden_size == qweight.shape[1] // type_size * block_size
        quant = torch.index_select(qweight, dim=0, index=x_flat)
        dequant = ggml_dequantize(
            quant, qweight_type, hidden_size, x_flat.shape[0], dtype
        )
        return dequant.view(*x.shape, hidden_size)
    else:
        qweight_type = WeightType(qweight_type)
        raise NotImplementedError(f"Unsupported GGUF quantization type: {qweight_type}")


class GGUFLinearMethod(LinearMethodBase):
    """Linear method for GGUF.

    Args:
        quant_config: The GGUF quantization config.
    """

    def __init__(self, quant_config: GGUFConfig):
        self.quant_config = quant_config

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        self.params_dtype = params_dtype
        output_size_per_partition = sum(output_partition_sizes)

        tensor_shape = (output_size_per_partition, input_size_per_partition)
        qweight = GGUFUninitializedParameter(requires_grad=False)
        set_weight_attrs(
            qweight,
            {
                "input_dim": 1,
                "output_dim": 0,
                "tensor_shape": tensor_shape,
                "is_gguf_weight": True,
                "data_container": [],
                "shard_id": [],
                "shard_id_map": {},
            },
        )
        set_weight_attrs(qweight, extra_weight_attrs)
        layer.register_parameter("qweight", qweight)

        qweight_type = Parameter(
            torch.empty(len(output_partition_sizes), dtype=torch.uint8),
            requires_grad=False,
        )
        set_weight_attrs(
            qweight_type,
            {
                "is_gguf_weight_type": True,
                "weight_type": 0,
                "shard_weight_type": {},
                "ignore_warning": True,
            },
        )
        set_weight_attrs(qweight_type, extra_weight_attrs)
        layer.register_parameter("qweight_type", qweight_type)

    def process_weights_after_loading(self, layer: torch.nn.Module):
        qweight_type = layer.qweight_type.weight_type
        if not (qweight_type in UNQUANTIZED_TYPES or qweight_type in DEQUANT_TYPES):
            qweight_type = WeightType(qweight_type)
            raise ValueError(
                f"Unsupported GGUF quantization type {qweight_type} in layer {layer}."
            )
        # For MergedColumnParallelLinear and QKVParallelLinear, we need to
        # materialize the padded weight parameter for CUDA Graph compatibility.
        self._create_padded_weight_param(layer)

    def _create_padded_weight_param(self, layer: torch.nn.Module):
        """Create padded weight parameter for GGUF MergedLinear layer."""
        qweight = layer.qweight
        shard_id_map = qweight.shard_id_map
        shard_id = qweight.shard_id
        if len(data_container := qweight.data_container) > 1:
            dtype = {data.dtype for data in data_container}
            assert len(dtype) == 1, ValueError(
                f"Data container has mixed dtypes: {dtype}"
            )
            dtype = next(iter(dtype))
            # concat dim0 and pad dim1
            padded_side = max(x.size(1) for x in data_container)
            concat_side = sum(x.size(0) for x in data_container)
            # Pad the quantized weights to dense tensor, and create a map
            # with the location of each shard in the padded tensor.
            padded_data = torch.zeros(
                (concat_side, padded_side), dtype=dtype, device=qweight.device
            )
            # (dim0_start, dim0_end, dim1_size)
            shard_offset_map = dict[str, tuple[int, int, int]]()
            ordered_shard_ids = _ordered_gguf_shard_ids(shard_id)
            cursor = 0
            for idx in ordered_shard_ids:
                id_in_container = shard_id_map[idx]
                start = cursor
                end = start + data_container[id_in_container].size(0)
                size = data_container[id_in_container].size(1)
                padded_data[start:end, :size] = data_container[id_in_container]
                shard_offset_map[idx] = (start, end, size)
                cursor = end
            qweight.data_container.clear()
            padded_param = Parameter(padded_data, requires_grad=False)
            set_weight_attrs(padded_param, vars(qweight))
            padded_param.shard_id = ordered_shard_ids
            set_weight_attrs(padded_param, {"shard_offset_map": shard_offset_map})
            layer.register_parameter("qweight", padded_param)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        shard_id = layer.qweight.shard_id

        if shard_id:
            # dequantize shard weights respectively
            shard_id = _ordered_gguf_shard_ids(shard_id)
            qweight = layer.qweight
            result = []
            for idx in shard_id:
                start, end, offset = layer.qweight.shard_offset_map[idx]
                qweight_type = layer.qweight_type.shard_weight_type[idx]
                result.append(
                    fused_mul_mat_gguf(
                        x, qweight[start:end, :offset].contiguous(), qweight_type
                    )
                )
            out = torch.cat(result, axis=1)
        else:
            qweight = layer.qweight
            qweight_type = layer.qweight_type.weight_type
            out = fused_mul_mat_gguf(x, qweight, qweight_type)
        if bias is not None:
            out.add_(bias)
        return out


class GGUFMoEMethod(FusedMoEMethodBase):
    """MoE method for GGUF.

    Args:
        quant_config: The GGUF quantization config.
    """

    def __init__(self, quant_config: GGUFConfig):
        self.quant_config = quant_config

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        self.params_dtype = params_dtype
        tensor_shape = (num_experts, 2 * intermediate_size_per_partition, hidden_size)
        # gate up proj
        w13_qweight = GGUFUninitializedParameter(requires_grad=False)
        set_weight_attrs(
            w13_qweight,
            {
                "input_dim": 1,
                "output_dim": 0,
                "tensor_shape": tensor_shape,
                "is_gguf_weight": True,
                "data_container": [],
            },
        )
        set_weight_attrs(w13_qweight, extra_weight_attrs)
        layer.register_parameter("w13_qweight", w13_qweight)

        w13_qweight_type = Parameter(
            torch.empty(1, dtype=torch.uint8), requires_grad=False
        )
        set_weight_attrs(
            w13_qweight_type,
            {"is_gguf_weight_type": True, "weight_type": 0, "ignore_warning": True},
        )
        set_weight_attrs(w13_qweight_type, extra_weight_attrs)
        layer.register_parameter("w13_qweight_type", w13_qweight_type)

        tensor_shape = (num_experts, intermediate_size_per_partition, hidden_size)
        # gate down proj
        w2_qweight = GGUFUninitializedParameter(requires_grad=False)
        set_weight_attrs(
            w2_qweight,
            {
                "input_dim": 1,
                "output_dim": 0,
                "tensor_shape": tensor_shape,
                "is_gguf_weight": True,
                "data_container": [],
            },
        )
        set_weight_attrs(w2_qweight, extra_weight_attrs)
        layer.register_parameter("w2_qweight", w2_qweight)

        w2_qweight_type = Parameter(
            torch.empty(1, dtype=torch.uint8), requires_grad=False
        )
        set_weight_attrs(
            w2_qweight_type,
            {"is_gguf_weight_type": True, "weight_type": 0, "ignore_warning": True},
        )

        set_weight_attrs(w2_qweight_type, extra_weight_attrs)
        layer.register_parameter("w2_qweight_type", w2_qweight_type)

    def create_moe_runner(
        self, layer: torch.nn.Module, moe_runner_config: MoeRunnerConfig
    ):
        self.moe_runner_config = moe_runner_config

    def apply(
        self,
        layer: torch.nn.Module,
        dispatch_output: StandardDispatchOutput,
    ) -> CombineInput:
        assert self.fused_experts is None

        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

        assert self.moe_runner_config.activation == "silu", (
            "Only SiLU activation is supported."
        )

        x = dispatch_output.hidden_states
        topk_output = dispatch_output.topk_output

        moe_runner_config = self.moe_runner_config

        topk_weights, topk_ids, _ = topk_output
        output = fused_moe_gguf(
            x=x,
            w1=layer.w13_qweight,
            w2=layer.w2_qweight,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            qweight_type=layer.w13_qweight_type.weight_type,
            qweight_type2=layer.w2_qweight_type.weight_type,
            activation=moe_runner_config.activation,
        )
        return StandardCombineInput(hidden_states=output)


class GGUFMoEXPUMethod(GGUFMoEMethod):
    """Hybrid non-ESIMD GGUF MoE implementation for Intel XPU.

    The Q4_K gate/up projection stays quantized and runs through
    sgl-kernel-xpu's INT4 W4A16 grouped GEMM.  GGUF files commonly use a
    higher precision type (Q5_K/Q6_K/Q5_1/Q8_0) for the down projection; it
    is dequantized once at load time and consumed by the native grouped FP16
    GEMM.  This preserves the checkpoint values instead of silently
    requantizing the down projection to four bits.
    """

    def process_weights_after_loading(self, layer: torch.nn.Module):
        from sglang.srt.layers.quantization.xpu_gguf_q4_k import (
            W4A16_GROUP_SIZE,
            repack_q4_k_to_w4a16,
        )

        if hasattr(layer, "materialize_gguf_weights"):
            layer.materialize_gguf_weights()

        w13 = layer.w13_qweight
        w13_type = WeightType(layer.w13_qweight_type.weight_type)
        if isinstance(w13, UninitializedParameter):
            raise ValueError("GGUF XPU MoE gate/up weights were not loaded")
        if w13_type != WeightType.Q4_K:
            raise ValueError(
                "GGUF XPU MoE currently requires Q4_K gate/up weights, got "
                f"{w13_type.name}"
            )

        device = w13.device
        num_experts, output_features, packed_width = w13.shape
        repacked = repack_q4_k_to_w4a16(
            w13.detach()
            .to(device="cpu", dtype=torch.uint8)
            .reshape(num_experts * output_features, packed_width)
        )
        logical_width = repacked.qweight.shape[-1]
        scale_width = repacked.scales.shape[-1]
        layer.register_buffer(
            "w13_xpu_qweight",
            repacked.qweight.reshape(num_experts, output_features, logical_width).to(
                device=device
            ),
            persistent=False,
        )
        layer.register_buffer(
            "w13_xpu_scale",
            repacked.scales.reshape(num_experts, output_features, scale_width).to(
                device=device, dtype=self.params_dtype
            ),
            persistent=False,
        )
        layer.register_buffer(
            "w13_xpu_zero",
            repacked.zeros.reshape(num_experts, output_features, scale_width).to(
                device=device, dtype=self.params_dtype
            ),
            persistent=False,
        )
        layer._gguf_xpu_w13_group_size = W4A16_GROUP_SIZE

        w2 = layer.w2_qweight
        w2_type = WeightType(layer.w2_qweight_type.weight_type)
        if isinstance(w2, UninitializedParameter):
            raise ValueError("GGUF XPU MoE down weights were not loaded")
        if w2_type not in UNQUANTIZED_TYPES | DEQUANT_TYPES:
            raise ValueError(
                f"Unsupported GGUF XPU MoE down quantization type: {w2_type.name}"
            )
        # sgl-kernel-xpu's dense grouped GEMM currently accepts BF16 only.
        # Keep this conversion local to GEMM2; gate/up continues to use the
        # model activation dtype required by the W4A16 kernel.
        down_dtype = torch.bfloat16
        if w2_type in UNQUANTIZED_TYPES:
            w2_dequant = w2.detach().to(dtype=down_dtype)
        else:
            w2_dequant = torch.stack(
                [
                    _xpu_reference_dequantize(w2[expert_id], w2_type, down_dtype)
                    for expert_id in range(w2.shape[0])
                ],
                dim=0,
            ).to(device=device)
        layer.register_buffer(
            "w2_xpu_dequant", w2_dequant.contiguous(), persistent=False
        )

        # The converted runtime buffers own the weights.  Do not retain the
        # original GGUF byte tensors as a second device-resident copy.
        del layer.w13_qweight
        del layer.w2_qweight

    def apply(
        self,
        layer: torch.nn.Module,
        dispatch_output: StandardDispatchOutput,
    ) -> CombineInput:
        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

        activation = self.moe_runner_config.activation
        if activation not in ("silu", "gelu"):
            raise ValueError(f"GGUF XPU MoE supports SiLU and GELU, got {activation!r}")

        x = dispatch_output.hidden_states
        if x.ndim != 2:
            raise ValueError(f"GGUF XPU MoE expects rank-2 input, got {x.shape}")
        topk_weights, topk_ids, _ = dispatch_output.topk_output
        if topk_ids.shape != topk_weights.shape or topk_ids.shape[0] != x.shape[0]:
            raise ValueError("GGUF XPU MoE top-k tensors do not match input")

        num_experts = layer.w13_xpu_qweight.shape[0]
        num_routes = topk_ids.numel()
        hidden_size = x.shape[-1]
        intermediate_size = layer.w2_xpu_dequant.shape[-1]
        device = x.device

        topk_ids_int = topk_ids.to(dtype=torch.int32).contiguous()
        expert_rows = torch.empty((num_experts,), dtype=torch.int32, device=device)
        problem_sizes1 = torch.empty((num_experts, 3), dtype=torch.int32, device=device)
        problem_sizes2 = torch.empty_like(problem_sizes1)
        a_map = torch.empty((num_routes,), dtype=torch.int32, device=device)
        c_map = torch.empty_like(a_map)
        torch.ops.sgl_kernel.prepare_moe_input.default(
            topk_ids_int,
            expert_rows,
            None,
            problem_sizes1,
            problem_sizes2,
            a_map,
            c_map,
            num_experts,
            hidden_size,
            topk_ids.shape[1],
        )

        routed_x = torch.empty((num_routes, hidden_size), dtype=x.dtype, device=device)
        torch.ops.sgl_kernel.scatter_tokens_to_experts.default(x, c_map, routed_x)
        gate_up = torch.empty(
            (num_routes, 2 * intermediate_size), dtype=x.dtype, device=device
        )
        torch.ops.sgl_kernel.moe_grouped_mm_nt_xe20_w4a16(
            gate_up,
            routed_x,
            layer.w13_xpu_qweight,
            layer.w13_xpu_scale,
            layer.w13_xpu_zero,
            None,
            expert_rows,
            num_experts,
            True,
            layer._gguf_xpu_w13_group_size,
        )
        activated = torch.empty(
            (num_routes, intermediate_size), dtype=x.dtype, device=device
        )
        if activation == "silu":
            torch.ops.sgl_kernel.silu_and_mul(activated, gate_up)
        else:
            torch.ops.sgl_kernel.gelu_tanh_and_mul(activated, gate_up)

        activated_bf16 = activated.to(dtype=torch.bfloat16)
        routed_output = torch.empty(
            (num_routes, hidden_size), dtype=torch.bfloat16, device=device
        )
        torch.ops.sgl_kernel.moe_grouped_mm_nt_xe20(
            routed_output,
            activated_bf16,
            layer.w2_xpu_dequant,
            None,
            expert_rows,
            num_experts,
            0,
            fuse_act=False,
            gemm1_alpha=1.702,
            gemm1_limit=7.0,
        )

        combined_bf16 = torch.empty(x.shape, dtype=torch.bfloat16, device=device)
        routed_scale = self.moe_runner_config.routed_scaling_factor
        torch.ops.sgl_kernel.apply_shuffle_mul_sum.default(
            routed_output,
            combined_bf16,
            c_map,
            1.0 if routed_scale is None else routed_scale,
            topk_weights,
        )
        return StandardCombineInput(hidden_states=combined_bf16.to(dtype=x.dtype))


class GGUFEmbeddingMethod(GGUFLinearMethod):
    """Embedding method for GGUF.

    Args:
        quant_config: The GGUF quantization config.
    """

    def embedding(self, layer: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
        qweight = layer.qweight
        qweight_type = layer.qweight_type.weight_type
        hidden_size = qweight.tensor_shape[1]

        return apply_gguf_embedding(
            x, qweight, qweight_type, hidden_size, dtype=self.params_dtype
        )


def _xpu_gguf_weight_shards(layer: torch.nn.Module):
    """Yield logical shards without forcing mixed GGUF dtypes into one tensor."""
    qweight = layer.qweight
    data_container = getattr(qweight, "data_container", None)
    if data_container:
        shard_ids = _ordered_gguf_shard_ids(qweight.shard_id)
        for shard_id in shard_ids:
            if shard_id not in qweight.shard_id_map:
                raise ValueError(f"Missing GGUF data for shard {shard_id}")
            # gguf's iterator emits no companion type tensor for F32 payloads.
            # A missing per-shard type therefore unambiguously means F32.
            qweight_type = layer.qweight_type.shard_weight_type.get(
                shard_id, WeightType.F32
            )
            yield (
                shard_id,
                data_container[qweight.shard_id_map[shard_id]],
                qweight_type,
            )
        return

    if isinstance(qweight, UninitializedParameter):
        raise ValueError("GGUF weight was not loaded")
    shard_ids = getattr(qweight, "shard_id", None)
    if shard_ids:
        for shard_id in _ordered_gguf_shard_ids(shard_ids):
            start, end, width = qweight.shard_offset_map[shard_id]
            yield (
                shard_id,
                qweight[start:end, :width].contiguous(),
                layer.qweight_type.shard_weight_type[shard_id],
            )
    else:
        yield None, qweight, layer.qweight_type.weight_type


def _xpu_reference_dequantize(
    qweight: torch.Tensor, qweight_type: int, dtype: torch.dtype
) -> torch.Tensor:
    """Dequantize a GGUF tensor once on CPU for the XPU fallback path."""
    import numpy as np

    raw = qweight.detach().to(device="cpu").contiguous().numpy()
    dequantized = gguf_dequantize(raw, WeightType(qweight_type))
    # Some numpy views returned by gguf are read-only.  Own the storage before
    # exposing it to torch and moving it to the accelerator.
    return torch.from_numpy(np.array(dequantized, copy=True)).to(dtype=dtype)


class GGUFLinearXPUMethod(GGUFLinearMethod):
    """GGUF linear implementation backed by non-ESIMD XPU kernels.

    Q4_K weights are repacked once into the unsigned INT4 layout consumed by
    sgl-kernel-xpu's W4A16 grouped GEMM.  Other GGUF types are dequantized once
    at load time and use the native XPU matmul as a correctness fallback.
    """

    def process_weights_after_loading(self, layer: torch.nn.Module):
        from sglang.srt.layers.quantization.xpu_gguf_q4_k import (
            W4A16_GROUP_SIZE,
            repack_q4_k_to_w4a16,
        )

        shards = list(_xpu_gguf_weight_shards(layer))
        if not shards:
            raise ValueError("GGUF XPU linear has no loaded weight shards")
        for _, _, qweight_type in shards:
            if qweight_type not in UNQUANTIZED_TYPES | DEQUANT_TYPES:
                raise ValueError(
                    "Unsupported GGUF quantization type on XPU: "
                    f"{WeightType(qweight_type)}"
                )

        device = shards[0][1].device
        gdn_col_perm = getattr(layer, "_gguf_gdn_col_perm", None)
        representations = []
        for index, (shard_id, raw_weight, qweight_type) in enumerate(shards):
            prefix = f"_gguf_xpu_{index}"
            if qweight_type == WeightType.Q4_K and gdn_col_perm is None:
                repacked = repack_q4_k_to_w4a16(
                    raw_weight.detach().to(device="cpu", dtype=torch.uint8)
                )
                layer.register_buffer(
                    f"{prefix}_weight",
                    repacked.qweight.unsqueeze(0).to(device=device),
                    persistent=False,
                )
                layer.register_buffer(
                    f"{prefix}_scale",
                    repacked.scales.unsqueeze(0).to(
                        device=device, dtype=self.params_dtype
                    ),
                    persistent=False,
                )
                layer.register_buffer(
                    f"{prefix}_zero",
                    repacked.zeros.unsqueeze(0).to(
                        device=device, dtype=self.params_dtype
                    ),
                    persistent=False,
                )
                representations.append(
                    (
                        "q4_k",
                        prefix,
                        shard_id,
                        W4A16_GROUP_SIZE,
                    )
                )
            else:
                weight = _xpu_reference_dequantize(
                    raw_weight, qweight_type, self.params_dtype
                ).to(device=device)
                if gdn_col_perm is not None:
                    ratio, num_key_heads, value_head_dim = gdn_col_perm
                    expected = ratio * num_key_heads * value_head_dim
                    if weight.shape[1] != expected:
                        raise ValueError(
                            "GGUF GDN column permutation does not match weight: "
                            f"K={weight.shape[1]}, expected={expected}"
                        )
                    weight = (
                        weight.reshape(
                            weight.shape[0],
                            ratio,
                            num_key_heads,
                            value_head_dim,
                        )
                        .transpose(1, 2)
                        .reshape(weight.shape)
                        .contiguous()
                    )
                layer.register_buffer(f"{prefix}_weight", weight, persistent=False)
                representations.append(("dense", prefix, shard_id, None))

        layer._gguf_xpu_representations = representations
        # The converted buffers own the runtime representation.  Release the
        # original GGUF byte tensor rather than keeping both copies resident.
        layer.register_parameter(
            "qweight",
            Parameter(
                torch.empty(0, dtype=torch.uint8, device=device),
                requires_grad=False,
            ),
        )

    @staticmethod
    def _apply_q4_k(
        layer: torch.nn.Module,
        x: torch.Tensor,
        prefix: str,
        group_size: int,
    ) -> torch.Tensor:
        packed_weight = getattr(layer, f"{prefix}_weight")
        scales = getattr(layer, f"{prefix}_scale")
        zeros = getattr(layer, f"{prefix}_zero")
        if x.dtype != scales.dtype:
            raise TypeError(
                f"GGUF XPU W4A16 expects {scales.dtype} activations, got {x.dtype}"
            )
        output_features = packed_weight.shape[1]
        x_2d = x.reshape(-1, x.shape[-1])
        output = torch.empty(
            (x_2d.shape[0], output_features), dtype=x.dtype, device=x.device
        )
        rows_per_expert = torch.full(
            (1,), x_2d.shape[0], dtype=torch.int32, device=x.device
        )
        torch.ops.sgl_kernel.moe_grouped_mm_nt_xe20_w4a16(
            output,
            x_2d,
            packed_weight,
            scales,
            zeros,
            None,
            rows_per_expert,
            1,
            True,
            group_size,
        )
        return output.reshape(*x.shape[:-1], output_features)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        outputs = []
        for kind, prefix, _shard_id, group_size in layer._gguf_xpu_representations:
            weight = getattr(layer, f"{prefix}_weight")
            if kind == "q4_k":
                outputs.append(self._apply_q4_k(layer, x, prefix, group_size))
            else:
                outputs.append(torch.matmul(x, weight.T))
        output = outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=-1)
        if bias is not None:
            output.add_(bias)
        return output


class GGUFEmbeddingXPUMethod(GGUFLinearMethod):
    """XPU embedding fallback that dequantizes the table once at load time."""

    def process_weights_after_loading(self, layer: torch.nn.Module):
        super().process_weights_after_loading(layer)
        shards = list(_xpu_gguf_weight_shards(layer))
        if len(shards) != 1:
            raise ValueError("GGUF XPU embedding expects one weight shard")
        _, raw_weight, qweight_type = shards[0]
        device = raw_weight.device
        weight = _xpu_reference_dequantize(
            raw_weight, qweight_type, self.params_dtype
        ).to(device=device)
        layer.register_buffer("_gguf_xpu_embedding", weight, persistent=False)
        layer.register_parameter(
            "qweight",
            Parameter(
                torch.empty(0, dtype=torch.uint8, device=device),
                requires_grad=False,
            ),
        )

    def embedding(self, layer: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
        return torch.embedding(layer._gguf_xpu_embedding, x)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Use the same dequantized table when this module is an LM head."""
        output = torch.matmul(x, layer._gguf_xpu_embedding.T)
        if bias is not None:
            output.add_(bias)
        return output


class GGUFUninitializedParameter(UninitializedParameter):
    cls_to_become = Parameter
    data_container: list[torch.Tensor]


# =============================================================================
# NPU-specific implementations for Ascend hardware
# =============================================================================
def ggml_dequantize_ascend(
    qweight: torch.Tensor,
    qweight_type: int,
    rows: int,
    cols: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Dequantize GGML quantized weights for NPU.

    Uses gguf library's reference implementation which supports all GGML formats
    and is guaranteed to be correct. The dequantization runs on CPU during model
    loading, then the dequantized weights are transferred to NPU for inference.
    """

    # Move to CPU for dequantization using gguf library
    qweight_cpu = qweight.cpu().numpy()

    # Use gguf library's dequantize (supports all GGML formats)
    dequant_np = gguf_dequantize(qweight_cpu, qweight_type)

    # Convert to torch and move to target device
    result = torch.from_numpy(dequant_np).to(dtype=dtype, device=qweight.device)
    result = result.reshape(rows, cols)

    return result


class GGUFLinearAscendMethod(LinearMethodBase):
    """Linear method for GGUF on Ascend NPU."""

    def __init__(self, quant_config: GGUFConfig):
        self.quant_config = quant_config

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        self.params_dtype = params_dtype
        output_size_per_partition = sum(output_partition_sizes)

        tensor_shape = (output_size_per_partition, input_size_per_partition)
        qweight = GGUFUninitializedParameter(requires_grad=False)
        set_weight_attrs(
            qweight,
            {
                "input_dim": 1,
                "output_dim": 0,
                "tensor_shape": tensor_shape,
                "is_gguf_weight": True,
                "data_container": [],
                "shard_id": [],
                "shard_id_map": {},
            },
        )
        set_weight_attrs(qweight, extra_weight_attrs)
        layer.register_parameter("qweight", qweight)

        qweight_type = Parameter(
            torch.empty(len(output_partition_sizes), dtype=torch.uint8),
            requires_grad=False,
        )
        set_weight_attrs(
            qweight_type,
            {
                "is_gguf_weight_type": True,
                "weight_type": 0,
                "shard_weight_type": {},
                "ignore_warning": True,
            },
        )
        set_weight_attrs(qweight_type, extra_weight_attrs)
        layer.register_parameter("qweight_type", qweight_type)

    def process_weights_after_loading(self, layer: torch.nn.Module):
        qweight_type = layer.qweight_type.weight_type
        if not (qweight_type in UNQUANTIZED_TYPES or qweight_type in DEQUANT_TYPES):
            raise ValueError(
                f"Unsupported GGUF quantization type {WeightType(qweight_type)} in layer."
            )
        self._create_padded_weight_param(layer)
        # Pre-dequantize weights for faster inference
        self._pre_dequantize_weights(layer)

    def _create_padded_weight_param(self, layer: torch.nn.Module):
        """Create padded weight parameter for GGUF MergedLinear layer."""
        qweight = layer.qweight
        shard_id_map = qweight.shard_id_map
        shard_id = qweight.shard_id
        if len(data_container := qweight.data_container) > 1:
            dtype = {data.dtype for data in data_container}
            assert len(dtype) == 1
            dtype = next(iter(dtype))
            padded_side = max(x.size(1) for x in data_container)
            concat_side = sum(x.size(0) for x in data_container)
            padded_data = torch.zeros(
                (concat_side, padded_side), dtype=dtype, device=qweight.device
            )
            shard_offset_map = dict[str, tuple[int, int, int]]()
            for idx in shard_id:
                id_in_container = shard_id_map[idx]
                start = sum(x.size(0) for x in data_container[:id_in_container])
                end = start + data_container[id_in_container].size(0)
                size = data_container[id_in_container].size(1)
                padded_data[start:end, :size] = data_container[id_in_container]
                shard_offset_map[idx] = (start, end, size)
            qweight.data_container.clear()
            padded_param = Parameter(padded_data, requires_grad=False)
            set_weight_attrs(padded_param, vars(qweight))
            set_weight_attrs(padded_param, {"shard_offset_map": shard_offset_map})
            layer.register_parameter("qweight", padded_param)

    def _pre_dequantize_weights(self, layer: torch.nn.Module):
        """Pre-dequantize GGML weights to FP16 for faster inference.

        This eliminates runtime dequantization overhead at the cost of more memory.
        """
        qweight = layer.qweight
        qweight_type = layer.qweight_type.weight_type

        if qweight_type in UNQUANTIZED_TYPES and qweight.dtype in (
            torch.float16,
            torch.bfloat16,
            torch.float32,
        ):
            layer.dequantized_weight = qweight
            return

        shard_id = getattr(qweight, "shard_id", None)
        has_shard_offset = hasattr(qweight, "shard_offset_map")

        if shard_id and has_shard_offset:
            # Handle sharded weights (QKV merged)
            shard_id = ["q", "k", "v"] if "q" in shard_id else shard_id
            dequant_shards = []
            for idx in shard_id:
                start, end, offset = qweight.shard_offset_map[idx]
                shard_qtype = layer.qweight_type.shard_weight_type[idx]
                shard_data = qweight[start:end, :offset].contiguous()

                block_size, type_size = gguf.GGML_QUANT_SIZES[shard_qtype]
                shape = (
                    shard_data.shape[0],
                    shard_data.shape[1] // type_size * block_size,
                )
                dequant = ggml_dequantize_ascend(
                    shard_data, shard_qtype, *shape, self.params_dtype
                )
                dequant_shards.append(dequant)

            dequant_weight = torch.cat(dequant_shards, dim=0)
        else:
            # Handle single weight
            block_size, type_size = gguf.GGML_QUANT_SIZES[qweight_type]
            shape = (qweight.shape[0], qweight.shape[1] // type_size * block_size)
            dequant_weight = ggml_dequantize_ascend(
                qweight, qweight_type, *shape, self.params_dtype
            )

        layer.dequantized_weight = dequant_weight

        if hasattr(layer, "qweight"):
            del layer.qweight
        if hasattr(layer, "qweight_type"):
            del layer.qweight_type

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Use pre-dequantized weight (always available after process_weights_after_loading)
        weight = layer.dequantized_weight
        out = x @ weight.T
        if bias is not None:
            out.add_(bias)
        return out


class GGUFMoEAscendMethod(FusedMoEMethodBase):
    """MoE method for GGUF on Ascend NPU."""

    def __init__(self, quant_config: GGUFConfig):
        self.quant_config = quant_config
        self.w13_kernel = NPUUnquantMoEMethod()
        self.w2_kernel = NPUUnquantMoEMethod()

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        tensor_shape = (num_experts, 2 * intermediate_size_per_partition, hidden_size)
        w13_qweight = GGUFUninitializedParameter(requires_grad=False)
        set_weight_attrs(
            w13_qweight,
            {
                "input_dim": 1,
                "output_dim": 0,
                "tensor_shape": tensor_shape,
                "is_gguf_weight": True,
                "data_container": [],
            },
        )
        set_weight_attrs(w13_qweight, extra_weight_attrs)
        layer.register_parameter("w13_qweight", w13_qweight)

        w13_qweight_type = Parameter(
            torch.empty(1, dtype=torch.uint8), requires_grad=False
        )
        set_weight_attrs(
            w13_qweight_type,
            {"is_gguf_weight_type": True, "weight_type": 0, "ignore_warning": True},
        )
        set_weight_attrs(w13_qweight_type, extra_weight_attrs)
        layer.register_parameter("w13_qweight_type", w13_qweight_type)

        tensor_shape = (num_experts, intermediate_size_per_partition, hidden_size)
        w2_qweight = GGUFUninitializedParameter(requires_grad=False)
        set_weight_attrs(
            w2_qweight,
            {
                "input_dim": 1,
                "output_dim": 0,
                "tensor_shape": tensor_shape,
                "is_gguf_weight": True,
                "data_container": [],
            },
        )
        set_weight_attrs(w2_qweight, extra_weight_attrs)
        layer.register_parameter("w2_qweight", w2_qweight)

        w2_qweight_type = Parameter(
            torch.empty(1, dtype=torch.uint8), requires_grad=False
        )
        set_weight_attrs(
            w2_qweight_type,
            {"is_gguf_weight_type": True, "weight_type": 0, "ignore_warning": True},
        )
        set_weight_attrs(w2_qweight_type, extra_weight_attrs)
        layer.register_parameter("w2_qweight_type", w2_qweight_type)

        # Store params_dtype for pre-dequantization
        self.params_dtype = params_dtype

    def process_weights_after_loading(self, layer: torch.nn.Module):
        """Pre-dequantize MoE weights to FP16 for faster inference."""

        if hasattr(layer, "materialize_gguf_weights"):
            layer.materialize_gguf_weights()

        # Check if weights are actually loaded (not still UninitializedParameter/empty)
        w13_qweight = layer.w13_qweight
        w13_qtype = layer.w13_qweight_type.weight_type

        # Pre-dequantize w13 weights (gate+up projections)
        if w13_qtype not in UNQUANTIZED_TYPES:
            num_experts = w13_qweight.shape[0]
            w13_dequant_list = []

            block_size, type_size = gguf.GGML_QUANT_SIZES[w13_qtype]

            for e in range(num_experts):
                qweight_cpu = w13_qweight[e].cpu().numpy()
                rows = w13_qweight[e].shape[0]
                cols = w13_qweight[e].shape[1] // type_size * block_size

                dequant_np = gguf_dequantize(qweight_cpu.flatten(), w13_qtype)
                dequant = (
                    torch.from_numpy(dequant_np)
                    .to(dtype=self.params_dtype, device=w13_qweight.device)
                    .reshape(rows, cols)
                    .contiguous()
                )
                w13_dequant_list.append(dequant)

            w13_full = torch.stack(w13_dequant_list, dim=0)
            layer.register_buffer(
                "w13_dequant", npu_format_cast(w13_full), persistent=False
            )
        else:
            layer.register_buffer(
                "w13_dequant", npu_format_cast(w13_qweight.data), persistent=False
            )

        # Pre-dequantize w2 weights (down projection)
        w2_qweight = layer.w2_qweight
        w2_qtype = layer.w2_qweight_type.weight_type

        if w2_qtype not in UNQUANTIZED_TYPES:
            num_experts = w2_qweight.shape[0]
            w2_dequant_list = []

            block_size, type_size = gguf.GGML_QUANT_SIZES[w2_qtype]

            for e in range(num_experts):
                qweight_cpu = w2_qweight[e].cpu().numpy()
                rows = w2_qweight[e].shape[0]
                cols = w2_qweight[e].shape[1] // type_size * block_size

                dequant_np = gguf_dequantize(qweight_cpu.flatten(), w2_qtype)
                dequant = (
                    torch.from_numpy(dequant_np)
                    .to(dtype=self.params_dtype, device=w2_qweight.device)
                    .reshape(rows, cols)
                    .contiguous()
                )
                w2_dequant_list.append(dequant)

            w2_full = torch.stack(w2_dequant_list, dim=0)

            layer.register_buffer(
                "w2_dequant", npu_format_cast(w2_full), persistent=False
            )
        else:
            layer.register_buffer(
                "w2_dequant", npu_format_cast(w2_qweight.data), persistent=False
            )

        if hasattr(layer, "w2_qweight"):
            del layer.w2_qweight
        if hasattr(layer, "w13_qweight"):
            del layer.w13_qweight

        if hasattr(layer, "dispatcher"):
            layer.dispatcher.set_quant_config({"quant_type": "gguf"})

    def create_moe_runner(
        self, layer: torch.nn.Module, moe_runner_config: MoeRunnerConfig
    ):
        layer.w13_kernel = self.w13_kernel
        layer.w2_kernel = self.w2_kernel
        moe_runner_config.layer = layer
        moe_runner_config.use_tp_all_gather_activation = True
        self.moe_runner_config = moe_runner_config
        backend = get_moe_runner_backend()
        if backend.is_auto():
            backend = MoeRunnerBackend.ASCEND
        self.runner = MoeRunner(backend, moe_runner_config)

    def apply(
        self,
        layer: torch.nn.Module,
        dispatch_output: StandardDispatchOutput,
    ) -> CombineInput:
        from sglang.srt.layers.moe.moe_runner.ascend import AscendQuantInfo

        quant_info = AscendQuantInfo(
            w13_weight=layer.w13_dequant,
            w2_weight=layer.w2_dequant,
            w13_weight_bias=getattr(layer, "w13_weight_bias", None),
            w2_weight_bias=getattr(layer, "w2_weight_bias", None),
            w13_scale_bias=getattr(layer, "w13_scale_bias", None),
            w2_scale_bias=getattr(layer, "w2_scale_bias", None),
        )
        return self.runner.run(dispatch_output, quant_info)


class GGUFEmbeddingAscendMethod(GGUFLinearAscendMethod):
    """Embedding method for GGUF on Ascend NPU."""

    def embedding(self, layer: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
        return torch.embedding(layer.dequantized_weight, x)
