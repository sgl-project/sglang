# SPDX-License-Identifier: Apache-2.0
"""Shared producer policy for explicitly opted-in dense/shared-expert callers."""

from functools import lru_cache

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.quantization.modelopt_fp8_input import ModelOptFp8Input
from sglang.srt.runtime_context import (
    get_exec,
    get_forward,
    get_lora,
    get_parallel,
    get_spec,
)


@lru_cache(maxsize=1)
def silu_static_fp8_supported() -> bool:
    if not torch.cuda.is_available() or torch.version.hip is not None:
        return False
    if torch.cuda.get_device_capability() != (8, 9):
        return False
    from sglang.srt.layers.quantization.fp8_utils import cutlass_fp8_supported

    # Reuse the consumer's CUDA/architecture requirement (SM89: CUDA >= 12.4).
    return cutlass_fp8_supported()


def supports_silu_fp8_graph(graph) -> bool:
    # Admission depends on the graph backend; tc_compiler alone does not imply
    # torch.compile execution for the full/breakable runners.
    return graph is None or all(
        getattr(graph, phase).backend in ("disabled", "full", "breakable")
        for phase in ("prefill", "decode")
    )


def make_silu_fp8_fusion(down, *, allowed: bool):
    if not allowed or not envs.SGLANG_ENABLE_SILU_STATIC_FP8_FUSION.get():
        return None
    # Linear construction imports the quantization registry. Keep its concrete
    # method imports here to avoid coupling module import to registry loading.
    from sglang.srt.layers.linear import RowParallelLinear
    from sglang.srt.layers.moe import get_moe_a2a_backend
    from sglang.srt.layers.quantization.modelopt_quant import ModelOptFp8LinearMethod

    method = down.quant_method
    parallel, execution = get_parallel(), get_exec()
    if (
        # Consumer/input contract: retain the validated linear wrapper and
        # channelwise scale path. Aligned shapes also retain its AUTO choices.
        type(down) is not RowParallelLinear
        or type(method) is not ModelOptFp8LinearMethod
        or not method.supports_static_fp8_input(down)
        or method.enable_flashinfer_bmm
        or not method.quant_config.is_checkpoint_fp8_serialized
        or not down.input_is_parallel
        or down.use_dp_attention_reduce
        or down.use_decode_attn_tp
        or down.input_size_per_partition % 16
        or down.output_size_per_partition % 16
        # Require the expected TP group. Mixed topologies are not yet supported
        # by this integration.
        or down.tp_size != parallel.tp_size
        or parallel.dp_size != 1
        or parallel.pp_size != 1
        or parallel.moe_ep_size != 1
        or parallel.attn_cp_size != 1
        or parallel.dcp_size != 1
        or parallel.enable_dp_attention
        # SP g-bar requires Tensor input for contiguous() and token padding,
        # even when it eventually calls apply; its fused path can bypass apply.
        or parallel.enable_layernorm_sp
        # Special execution modes are not yet supported by this integration.
        # RL/compile can select a different activation reference implementation.
        or get_lora().enable_lora
        or get_spec().speculative_algorithm is not None
        or execution.deterministic.enable_deterministic_inference
        or execution.deterministic.rl_on_policy_target is not None
        or execution.graph.enable_torch_compile
        or not get_moe_a2a_backend().is_none()
        or down.orig_dtype not in (torch.float16, torch.bfloat16)
    ):
        return None
    graph = execution.graph.cuda_graph_config
    if not supports_silu_fp8_graph(graph):
        return None
    if not silu_static_fp8_supported():
        return None
    from sglang.kernels.ops.quantization.silu_and_mul_static_fp8 import (
        silu_and_mul_static_fp8_module,
    )

    silu_and_mul_static_fp8_module(
        down.orig_dtype
    )  # Compile before model forward / Graph capture.
    return SiluFp8Fusion(down)


class SiluFp8Fusion:
    # Not an nn.Module: do not register the linear twice or change state_dict.
    def __init__(self, down):
        # Cache the producer callable to avoid imports in forward.
        from sglang.kernels.ops.quantization.silu_and_mul_static_fp8 import (
            silu_and_mul_static_fp8,
        )

        self.down = down
        self._producer = silu_and_mul_static_fp8

    def __call__(self, gate_up):
        """Return prequantized input, or None to use the unfused activation."""
        down = self.down
        scale = getattr(down, "input_scale", None)
        if (
            not isinstance(gate_up, torch.Tensor)
            or get_forward().sp_active
            or gate_up.ndim != 2
            or not gate_up.is_contiguous()
            or gate_up.storage_offset() % 8
            or gate_up.dtype != down.orig_dtype
            or not gate_up.is_cuda
            or gate_up.shape[1] != 2 * down.input_size_per_partition
            or gate_up.shape[1] == 0
            or gate_up.shape[1] % 32
            or gate_up.numel() > 2**32 - 1
            or not isinstance(scale, torch.Tensor)
            or scale.numel() != 1
            or scale.dtype != torch.float32
            or scale.device != gate_up.device
            or not scale.is_contiguous()
            or down.weight.device != gate_up.device
            or down.weight.dtype != torch.float8_e4m3fn
            or down.weight.shape
            != (down.input_size_per_partition, down.output_size_per_partition)
            or down.weight_scale.numel() != down.output_size_per_partition
            or down.output_size_per_partition % 16
            or down.use_flashinfer_bmm
        ):
            return None
        qx, rows = self._producer(gate_up, scale)
        return ModelOptFp8Input(qx, scale, gate_up.dtype, rows)
