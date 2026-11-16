# Adapted from https://github.com/vllm-project/vllm/tree/main/vllm/model_executor/layers/quantization/compressed_tensors
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import logging
from collections.abc import Callable
from typing import Optional

import torch

from sglang.srt.layers.parameter import (
    GroupQuantScaleParameter,
    ModelWeightParameter,
    PerTensorScaleParameter,
)
from sglang.srt.layers.quantization.compressed_tensors.schemes.compressed_tensors_scheme import (
    CompressedTensorsLinearScheme,
)
from sglang.srt.layers.quantization.fp4_utils import get_fp4_gemm_runner_backend
from sglang.srt.layers.quantization.marlin_utils_fp4 import (
    apply_fp4_marlin_linear,
    prepare_nvfp4_layer_for_marlin,
)
from sglang.srt.layers.quantization.utils import swizzle_blockscale
from sglang.srt.layers.utils.common import copy_or_rebind_param
from sglang.srt.runtime_context import get_platform

logger = logging.getLogger(__name__)

__all__ = ["CompressedTensorsW4A16Fp4"]

NVFP4_GROUP_SIZE = 16
# The first cuDNN backend with the bf16 x fp4 block-scaled GEMM that FlashInfer's
# cudnn backend calls; older libraries do not implement it at all.
CUDNN_BF16_FP4_MIN_VERSION = 92301

_warned: set[str] = set()


def _warn_once(key: str, msg: str, *args) -> None:
    # The scheme is built per layer; log a backend fallback once per model.
    if key not in _warned:
        _warned.add(key)
        logger.warning(msg, *args)


def _check_cudnn_bf16_fp4() -> None:
    """Fail at load rather than on the first forward: FlashInfer checks the
    cuDNN version only when the GEMM runs, and torch pins an older cuDNN."""
    try:
        import cudnn

        version = cudnn.backend_version()
    except ImportError:
        version = None
    if version is None or version < CUDNN_BF16_FP4_MIN_VERSION:
        found = "not installed" if version is None else f"found {version}"
        raise ValueError(
            "--fp4-gemm-backend flashinfer_cudnn needs cuDNN >= 9.23.1 for the "
            f"bf16 x fp4 GEMM ({found}). Use --fp4-gemm-backend flashinfer_cutedsl "
            "(the SM100 default), or upgrade the nvidia-cudnn wheel and "
            "nvidia-cudnn-frontend."
        )


def _use_flashinfer_bf16_fp4(params_dtype: torch.dtype) -> bool:
    """Whether to serve this weight-only layer with FlashInfer ``mm_bf16_fp4``
    instead of FP4 Marlin, following ``--fp4-gemm-backend``.

    ``auto`` already resolves to ``flashinfer_cutedsl`` on SM100 and to
    ``marlin`` on SM80-SM90 (``initialize_fp4_gemm_config``). Only the cuDNN and
    CuTe-DSL FlashInfer backends implement bf16 x fp4, and only for bf16
    activations, so every other case keeps the Marlin path.
    """
    backend = get_fp4_gemm_runner_backend()
    if not (backend.is_flashinfer_cutedsl() or backend.is_flashinfer_cudnn()):
        if backend.is_flashinfer():
            _warn_once(
                "no_bf16_fp4_backend",
                "--fp4-gemm-backend %s has no bf16 x fp4 GEMM; serving NVFP4 "
                "weight-only linears with FP4 Marlin.",
                backend.value,
            )
        return False
    if not get_platform().is_blackwell:
        raise ValueError(
            f"--fp4-gemm-backend {backend.value} for NVFP4 weight-only linears "
            "requires SM100+. Use --fp4-gemm-backend marlin on SM80-SM90."
        )
    if backend.is_flashinfer_cudnn():
        _check_cudnn_bf16_fp4()
    if params_dtype != torch.bfloat16:
        _warn_once(
            "non_bf16",
            "FlashInfer mm_bf16_fp4 supports only bfloat16 activations; serving "
            "NVFP4 weight-only linears with FP4 Marlin for this %s model.",
            params_dtype,
        )
        return False
    return True


class _Nvfp4MarlinQuantConfig:
    """Carries the one field `prepare_nvfp4_layer_for_marlin` reads off
    `layer.quant_config`. Setting it is what enables that helper's group_size
    validation, which is skipped entirely when the attribute is absent."""

    group_size = NVFP4_GROUP_SIZE


class CompressedTensorsW4A16Fp4(CompressedTensorsLinearScheme):
    """Weight-only NVFP4: FP4 weights, FP16/BF16 activations.

    The GEMM follows ``--fp4-gemm-backend``: FlashInfer ``mm_bf16_fp4`` for
    ``flashinfer_cutedsl`` / ``flashinfer_cudnn`` (the SM100 default) with bf16
    activations, FP4 Marlin otherwise (the SM80-SM90 default).

    Serves two config shapes:
      - nvfp4a16, which has no ``input_activations`` at all;
      - nvfp4 (w4a4) on a pre-Blackwell GPU, where the activation quantization
        is dropped and the checkpoint's ``input_global_scale`` goes unused.

    Kept separate from ``CompressedTensorsW4A4Fp4`` rather than folded in behind
    a flag because ``get_min_capability`` is a classmethod resolved on the
    instance's class, so a shared class would also lower the SM100 gate that
    guards the native w4a4 path.
    """

    def __init__(self, has_input_global_scale: bool = False):
        # True for a w4a4 checkpoint being served weight-only: the parameter
        # must still be registered so the loader finds a home for it.
        self.has_input_global_scale = has_input_global_scale
        self.group_size = NVFP4_GROUP_SIZE

    @classmethod
    def get_min_capability(cls) -> int:
        # FP4 Marlin is a weight-only kernel and needs no FP4 tensor cores.
        return 80

    def create_weights(
        self,
        layer: torch.nn.Module,
        output_partition_sizes: list[int],
        input_size_per_partition: int,
        params_dtype: torch.dtype,
        weight_loader: Callable,
        **kwargs,
    ):
        output_size_per_partition = sum(output_partition_sizes)
        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        # prepare_nvfp4_layer_for_marlin reads both of these; without
        # params_dtype it raises on a None activation dtype.
        layer.params_dtype = params_dtype
        layer.quant_config = _Nvfp4MarlinQuantConfig()

        weight = ModelWeightParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition // 2,
                dtype=torch.uint8,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight_packed", weight)

        weight_global_scale = PerTensorScaleParameter(
            data=torch.empty(len(output_partition_sizes), dtype=torch.float32),
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight_global_scale", weight_global_scale)

        weight_scale = GroupQuantScaleParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition // self.group_size,
                dtype=torch.float8_e4m3fn,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight_scale", weight_scale)

        if self.has_input_global_scale:
            input_global_scale = PerTensorScaleParameter(
                data=torch.empty(len(output_partition_sizes), dtype=torch.float32),
                weight_loader=weight_loader,
            )
            layer.register_parameter("input_global_scale", input_global_scale)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # The activation scale is meaningless for a weight-only kernel.
        if self.has_input_global_scale:
            del layer.input_global_scale

        # Both kernels take one global scale per layer. With fused projections
        # (q/k/v, gate/up) that only matches the checkpoint if every projection
        # shares it: substituting one scale while keeping the other projections'
        # group scales would silently change their dequantized weights.
        # llm-compressor fuses these global scales when exporting.
        global_scale = layer.weight_global_scale.data
        if torch.unique(global_scale).numel() != 1:
            raise ValueError(
                "NVFP4 weight-only linear requires all fused projections to share "
                f"one weight_global_scale, got {global_scale.tolist()}. Re-export "
                "the checkpoint with fused global scales."
            )
        # compressed-tensors stores the global scale as a divisor (1/scale),
        # while both kernels expect the scale itself. Skipping the inversion
        # overflows the bf16 bias multiply to inf, zeroing all logits.
        weight_scale_2 = (1 / global_scale[0]).to(torch.float32)

        layer.use_flashinfer_bf16_fp4 = _use_flashinfer_bf16_fp4(layer.params_dtype)
        if layer.use_flashinfer_bf16_fp4:
            from flashinfer import prepare_bf16_fp4_weights

            layer.flashinfer_backend = (
                get_fp4_gemm_runner_backend().get_flashinfer_backend()
            )
            weight, weight_scale, alpha = prepare_bf16_fp4_weights(
                layer.weight_packed.data,
                swizzle_blockscale(layer.weight_scale.data),
                weight_scale_2.reshape(1),
                backend=layer.flashinfer_backend,
            )
            del layer.weight_packed, layer.weight_scale, layer.weight_global_scale
            copy_or_rebind_param(layer, "weight", weight)
            copy_or_rebind_param(layer, "weight_scale_interleaved", weight_scale)
            # The backend may fold the global scale into the block scales.
            if alpha is None:
                layer.alpha = None
            else:
                copy_or_rebind_param(layer, "alpha", alpha)
            return

        copy_or_rebind_param(layer, "weight_global_scale", weight_scale_2)

        # prepare_nvfp4_layer_for_marlin operates on `weight`; compressed-tensors
        # names the packed weight `weight_packed`.
        copy_or_rebind_param(layer, "weight", layer.weight_packed.data)
        del layer.weight_packed

        prepare_nvfp4_layer_for_marlin(layer)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if layer.use_flashinfer_bf16_fp4:
            from flashinfer import mm_bf16_fp4

            out = mm_bf16_fp4(
                x.reshape(-1, x.shape[-1]),
                layer.weight,
                layer.weight_scale_interleaved,
                layer.alpha,
                backend=layer.flashinfer_backend,
                out_dtype=x.dtype,
            )
            if bias is not None:
                out = out + bias
            return out.view(*x.shape[:-1], layer.output_size_per_partition)

        return apply_fp4_marlin_linear(
            input=x,
            weight=layer.weight,
            weight_scale=layer.weight_scale,
            weight_global_scale=layer.weight_global_scale,
            workspace=layer.workspace,
            size_n=layer.output_size_per_partition,
            size_k=layer.input_size_per_partition,
            bias=bias,
        )
