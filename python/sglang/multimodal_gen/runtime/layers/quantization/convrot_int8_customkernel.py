# SPDX-License-Identifier: Apache-2.0
"""INT8 W8A8 linear backed by sgl-kernel's fused ConvRot ops.

One op takes a BF16 activation and performs the group-wise Hadamard rotation,
dynamic per-row INT8 quantization, INT8 GEMM, dequantization and bias add
without materializing the intermediates. Weights receive the same rotation and
per-output-channel quantization once, after loading a stock BF16 checkpoint.

Two variants of that op are exposed to the model code as helpers below: a
shared-input form that rotates and quantizes an activation once for several
linears consuming it (a q/k/v trio), and a gelu-input form that applies
GELU(tanh) inside the rotate kernel for an FFN down-projection. Both are
bitwise identical to the plain op on the equivalent eager input.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch.nn.parameter import Parameter

from sglang.multimodal_gen.runtime.layers.linear import (
    ColumnParallelLinear,
    LinearBase,
    LinearMethodBase,
    ReplicatedLinear,
    RowParallelLinear,
)
from sglang.multimodal_gen.runtime.layers.quantization.configs.convrot_int8_customkernel_config import (
    ConvRotInt8CustomKernelConfig,
    check_convrot_int8_capability,
)
from sglang.multimodal_gen.runtime.utils.weight_attrs import set_weight_attrs

__all__ = [
    "ConvRotInt8CustomKernelConfig",
    "ConvRotInt8CustomKernelLinearMethod",
    "apply_convrot_int8_gelu_input",
    "apply_convrot_int8_shared_input",
    "apply_convrot_int8_shared_input_out",
    "convrot_int8_fuses_gelu_input",
    "convrot_int8_shares_input",
]

_REQUIRED_OPS = (
    "convrot_int8_supported_sm_versions",
    "convrot_rotate_quantize_activation",
    "convrot_int8_fused_linear",
    "convrot_int8_fused_linear_gelu_input",
    "convrot_int8_linear_prequant",
    "convrot_int8_linear_prequant_out",
)


def _load_sgl_kernel() -> None:
    import sgl_kernel  # noqa: F401 -- registers torch.ops.sgl_kernel.*

    # Ops first: the capability table is read from the kernel itself.
    for op_name in _REQUIRED_OPS:
        if not hasattr(torch.ops.sgl_kernel, op_name):
            raise RuntimeError(
                "convrot_int8_customkernel quantization requires an sgl_kernel "
                f"build that registers torch.ops.sgl_kernel.{op_name}; the "
                "installed sgl_kernel does not."
            )
    check_convrot_int8_capability(torch.cuda.get_device_capability())


def _as_rows(x: torch.Tensor) -> torch.Tensor:
    # The ops take BF16 only. FP16 activations (autocast-fp16 pipelines) are
    # cast on the way in and back on the way out; the BF16 rounding is an
    # order of magnitude below the INT8 quantization error.
    if x.dtype == torch.float16:
        x = x.to(torch.bfloat16)
    elif x.dtype != torch.bfloat16:
        raise ValueError(
            f"convrot_int8_customkernel does not support activation dtype {x.dtype}"
        )
    return x.reshape(-1, x.shape[-1]).contiguous()


def _like_input(out: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    out = out.reshape(*x.shape[:-1], out.shape[-1])
    return out if out.dtype == x.dtype else out.to(x.dtype)


class ConvRotInt8CustomKernelLinearMethod(LinearMethodBase):
    """Quantizes BF16 weights to rotated INT8 after load and runs the fused op."""

    def __init__(self, quant_config: ConvRotInt8CustomKernelConfig) -> None:
        self.quant_config = quant_config
        _load_sgl_kernel()

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        # get_quant_method already screened the per-shard sizes (both parallel
        # linears set them before asking for the method), so these are last-
        # resort assertions for layers that pass different sizes here.
        if input_size_per_partition % self.quant_config.group_size:
            raise ValueError(
                f"convrot_int8_customkernel needs input_size_per_partition "
                f"({input_size_per_partition}) divisible by group_size "
                f"{self.quant_config.group_size}; leave the layer in BF16 with "
                "--quantization-ignored-layers"
            )
        if sum(output_partition_sizes) % 8:
            raise ValueError(
                f"convrot_int8_customkernel needs the output size per partition "
                f"({sum(output_partition_sizes)}) to be a multiple of 8; leave the "
                "layer in BF16 with --quantization-ignored-layers"
            )
        # Matches UnquantizedLinearMethod so the source weights load in BF16
        # before quantization.
        weight = Parameter(
            torch.empty(
                sum(output_partition_sizes),
                input_size_per_partition,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        set_weight_attrs(weight, {"input_dim": 1, "output_dim": 0})
        layer.register_parameter("weight", weight)
        set_weight_attrs(weight, extra_weight_attrs)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        weight = layer.weight.data
        if weight.dtype == torch.int8:
            return

        # Quantization runs on CUDA; a layer staged on CPU is round-tripped on
        # its own. (The transformer loader currently materialises every online-
        # quantized component on the GPU before this runs, so INT8 saves memory
        # after load, not during it.)
        home = weight.device
        weight_q, weight_scale = (
            torch.ops.sgl_kernel.convrot_rotate_quantize_activation(
                weight.to("cuda", non_blocking=True).to(torch.bfloat16),
                self.quant_config.group_size,
            )
        )
        layer.weight = Parameter(weight_q.to(home), requires_grad=False)
        layer.register_parameter(
            "weight_scale",
            Parameter(weight_scale.to(home), requires_grad=False),
        )
        self.quant_config.note_quantized(weight.numel() * weight.element_size())
        del weight_q, weight_scale
        torch.cuda.empty_cache()

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        out = torch.ops.sgl_kernel.convrot_int8_fused_linear(
            _as_rows(x),
            layer.weight,
            layer.weight_scale,
            bias,
            self.quant_config.group_size,
        )
        return _like_input(out=out, x=x)


def _forward_is_plain_apply(layer: LinearBase) -> bool:
    # True only when layer(x) is exactly quant_method.apply(layer, x, layer.bias):
    # no collective, no deferred bias, no rank-dependent bias.
    if layer.skip_bias_add:
        return False
    if isinstance(layer, ReplicatedLinear):
        return True
    if isinstance(layer, ColumnParallelLinear):
        return not layer.gather_output
    if isinstance(layer, RowParallelLinear):
        return layer.tp_size == 1
    return False


def _is_convrot_int8(layer: torch.nn.Module) -> bool:
    # LoRA wrappers and other non-LinearBase modules carry no quant_method.
    return isinstance(layer, LinearBase) and isinstance(
        layer.quant_method, ConvRotInt8CustomKernelLinearMethod
    )


def convrot_int8_shares_input(layers: Sequence[torch.nn.Module]) -> bool:
    """Whether ``apply_convrot_int8_shared_input`` reproduces ``layer(x)`` for
    every layer in ``layers``."""
    return len(layers) > 0 and all(
        _is_convrot_int8(layer) and _forward_is_plain_apply(layer) for layer in layers
    )


def apply_convrot_int8_shared_input(
    x: torch.Tensor, layers: Sequence[LinearBase]
) -> list[torch.Tensor]:
    """``[layer(x)[0] for layer in layers]`` with ``x`` rotated and quantized once.

    Bitwise identical to applying each layer on its own; see
    ``convrot_int8_linear_prequant`` in sgl-kernel.
    """
    group_size = layers[0].quant_method.quant_config.group_size
    x_q, x_scale = torch.ops.sgl_kernel.convrot_rotate_quantize_activation(
        _as_rows(x), group_size
    )
    outs = []
    for layer in layers:
        out = torch.ops.sgl_kernel.convrot_int8_linear_prequant(
            x_q, x_scale, layer.weight, layer.weight_scale, layer.bias, group_size
        )
        outs.append(_like_input(out=out, x=x))
    return outs


def apply_convrot_int8_shared_input_out(
    x: torch.Tensor, layers: Sequence[LinearBase], outs: Sequence[torch.Tensor]
) -> None:
    """Writes ``layer(x)[0]`` into ``out`` for each (layer, out) pair with ``x``
    rotated and quantized once; ``out`` must be a contiguous BF16 slice shaped
    like ``layer(x)[0]``. Bitwise identical to ``apply_convrot_int8_shared_input``
    and to ``layer(x)``; see ``convrot_int8_linear_prequant_out`` in sgl-kernel.
    """
    group_size = layers[0].quant_method.quant_config.group_size
    x_q, x_scale = torch.ops.sgl_kernel.convrot_rotate_quantize_activation(
        _as_rows(x), group_size
    )
    for layer, out in zip(layers, outs, strict=True):
        # view() rather than reshape(): a copy here would silently drop the write.
        torch.ops.sgl_kernel.convrot_int8_linear_prequant_out(
            x_q,
            x_scale,
            layer.weight,
            layer.weight_scale,
            layer.bias,
            group_size,
            out.view(-1, out.shape[-1]),
        )


def convrot_int8_fuses_gelu_input(layer: torch.nn.Module) -> bool:
    """Whether ``apply_convrot_int8_gelu_input`` reproduces
    ``layer(F.gelu(x, approximate="tanh"))[0]``."""
    return _is_convrot_int8(layer) and _forward_is_plain_apply(layer)


def apply_convrot_int8_gelu_input(layer: LinearBase, x: torch.Tensor) -> torch.Tensor:
    """``layer(F.gelu(x, approximate="tanh"))[0]`` as one op, bitwise identical
    to the eager GELU followed by the layer; see
    ``convrot_int8_fused_linear_gelu_input`` in sgl-kernel."""
    out = torch.ops.sgl_kernel.convrot_int8_fused_linear_gelu_input(
        _as_rows(x),
        layer.weight,
        layer.weight_scale,
        layer.bias,
        layer.quant_method.quant_config.group_size,
    )
    return _like_input(out=out, x=x)
