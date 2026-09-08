# SPDX-License-Identifier: Apache-2.0
"""kitchen_int8's sgl-kernel backend: INT8 W8A8 linear on the fused ConvRot ops.

One op takes a BF16 activation and performs the group-wise Hadamard rotation,
dynamic per-row INT8 quantization, INT8 GEMM, dequantization and bias add
without materializing the intermediates. Weights receive the same rotation and
per-output-channel quantization once, after loading a stock BF16 checkpoint; a
serialized Comfy ConvRot INT8 checkpoint loads its INT8 weights and row scales
directly, the same tensors comfy_kitchen consumes.

Two variants of that op are exposed to the model code as helpers below: a
shared-input form that rotates and quantizes an activation once for several
linears consuming it (a q/k/v trio), and a gelu-input form that applies
GELU(tanh) inside the rotate kernel for an FFN down-projection. Both are
bitwise identical to the plain op on the equivalent eager input.
"""

from __future__ import annotations

import functools
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
from sglang.multimodal_gen.runtime.layers.quantization.configs.kitchen_int8_config import (
    KitchenInt8Config,
)
from sglang.multimodal_gen.runtime.utils.weight_attrs import set_weight_attrs

__all__ = [
    "ConvRotInt8SglKernelLinearMethod",
    "REFUSED_CAPABILITY_REASONS",
    "apply_convrot_int8_gelu_input",
    "apply_convrot_int8_shared_input",
    "apply_convrot_int8_shared_input_out",
    "check_convrot_int8_capability",
    "convrot_int8_fuses_gelu_input",
    "convrot_int8_shares_input",
    "convrot_int8_supported_capabilities",
    "sgl_kernel_convrot_available",
    "sgl_kernel_convrot_unavailable_reason",
]

_REQUIRED_OPS = (
    "convrot_int8_supported_sm_versions",
    "convrot_rotate_quantize_activation",
    "convrot_int8_fused_linear",
    "convrot_int8_fused_linear_gelu_input",
    "convrot_int8_linear_prequant",
    "convrot_int8_linear_prequant_out",
)

# Parts the ops deliberately leave out, with the reason shown at load time. The
# supported table itself lives in sgl-kernel (convrot_int8_supported_sm_versions);
# nothing here duplicates it.
REFUSED_CAPABILITY_REASONS: dict[tuple[int, int], str] = {
    (10, 3): (
        "Blackwell Ultra cuts INT8 tensor-core throughput to a fraction of its "
        "BF16 rate, so W8A8 INT8 would be a slowdown there; use FP8 or NVFP4 "
        "quantization on this GPU"
    ),
}


def _ops_registered() -> bool:
    try:
        import sgl_kernel  # noqa: F401 -- registers torch.ops.sgl_kernel.*
    except ImportError:
        return False
    return all(hasattr(torch.ops.sgl_kernel, op) for op in _REQUIRED_OPS)


def convrot_int8_supported_capabilities() -> frozenset[tuple[int, int]]:
    """(major, minor) pairs the installed sgl_kernel carries convrot code for."""
    import sgl_kernel  # noqa: F401 -- registers torch.ops.sgl_kernel.*

    versions = torch.ops.sgl_kernel.convrot_int8_supported_sm_versions()
    return frozenset((int(v) // 10, int(v) % 10) for v in versions)


def check_convrot_int8_capability(capability: tuple[int, int]) -> None:
    """Raise with the specific reason when `capability` is not in the kernel's table."""
    supported = convrot_int8_supported_capabilities()
    if capability in supported:
        return
    major, minor = capability
    reason = REFUSED_CAPABILITY_REASONS.get(
        capability, "the convrot_int8_* ops carry no code for it"
    )
    supported_text = ", ".join(f"{a}.{b}" for a, b in sorted(supported))
    raise RuntimeError(
        f"kitchen_int8 backend sgl_kernel does not support CC {major}.{minor}: "
        f"{reason}. Supported compute capabilities: {supported_text}"
    )


@functools.cache
def sgl_kernel_convrot_unavailable_reason() -> str | None:
    """Why the sgl-kernel backend cannot run here, or None when it can."""
    if not torch.cuda.is_available():
        return "no CUDA device"
    if not _ops_registered():
        return "the installed sgl_kernel does not register the convrot_int8_* ops"
    capability = torch.cuda.get_device_capability()
    if capability in convrot_int8_supported_capabilities():
        return None
    major, minor = capability
    reason = REFUSED_CAPABILITY_REASONS.get(
        capability, "the convrot_int8_* ops carry no code for it"
    )
    return f"CC {major}.{minor} is not supported: {reason}"


def sgl_kernel_convrot_available() -> bool:
    """Whether the installed sgl_kernel carries the convrot ops for the current GPU."""
    return sgl_kernel_convrot_unavailable_reason() is None


def _load_sgl_kernel() -> None:
    # Ops first: the capability table is read from the kernel itself.
    if not _ops_registered():
        raise RuntimeError(
            "kitchen_int8 backend sgl_kernel requires an sgl_kernel build that "
            "registers the torch.ops.sgl_kernel.convrot_int8_* ops; the installed "
            "sgl_kernel does not"
        )
    check_convrot_int8_capability(torch.cuda.get_device_capability())


def _as_rows(x: torch.Tensor) -> torch.Tensor:
    # The ops take BF16 only. FP16 activations (--dit-precision fp16) are
    # cast on the way in and back on the way out; the BF16 rounding is an
    # order of magnitude below the INT8 quantization error.
    if x.dtype == torch.float16:
        x = x.to(torch.bfloat16)
    elif x.dtype != torch.bfloat16:
        raise ValueError(
            f"kitchen_int8 backend sgl_kernel does not support activation dtype {x.dtype}"
        )
    return x.reshape(-1, x.shape[-1]).contiguous()


def _like_input(out: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    out = out.reshape(*x.shape[:-1], out.shape[-1])
    return out if out.dtype == x.dtype else out.to(x.dtype)


class ConvRotInt8SglKernelLinearMethod(LinearMethodBase):
    """Loads or creates ConvRot INT8 weights and runs sgl-kernel's fused ops."""

    def __init__(
        self,
        quant_config: KitchenInt8Config,
        *,
        group_size: int,
        is_checkpoint_serialized: bool,
    ) -> None:
        self.quant_config = quant_config
        self.group_size = group_size
        self.is_checkpoint_serialized = is_checkpoint_serialized
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
        # get_quant_method screened the unsharded input size and the per-shard
        # output width, so these fire only under TP > 1, where a row-parallel
        # layer splits the very dimension the rotation groups over.
        if input_size_per_partition % self.group_size:
            raise ValueError(
                f"kitchen_int8 needs input_size_per_partition "
                f"({input_size_per_partition}) divisible by group_size "
                f"{self.group_size}; leave the layer in BF16 with "
                "--quantization-ignored-layers"
            )
        if sum(output_partition_sizes) % 8:
            raise ValueError(
                f"kitchen_int8 backend sgl_kernel needs the output size per "
                f"partition ({sum(output_partition_sizes)}) to be a multiple of 8"
            )
        # The online path initially matches UnquantizedLinearMethod so the
        # source weights load in BF16 before quantization. Serialized weights
        # allocate their final INT8 storage immediately.
        weight = Parameter(
            torch.empty(
                sum(output_partition_sizes),
                input_size_per_partition,
                dtype=(torch.int8 if self.is_checkpoint_serialized else params_dtype),
            ),
            requires_grad=False,
        )
        set_weight_attrs(weight, {"input_dim": 1, "output_dim": 0})
        layer.register_parameter("weight", weight)
        set_weight_attrs(weight, extra_weight_attrs)
        if self.is_checkpoint_serialized:
            # Comfy stores the per-row scale as [N, 1]; the ops read N contiguous
            # floats and accept that shape unchanged.
            weight_scale = Parameter(
                torch.empty(sum(output_partition_sizes), 1, dtype=torch.float32),
                requires_grad=False,
            )
            set_weight_attrs(weight_scale, {"output_dim": 0})
            set_weight_attrs(weight_scale, extra_weight_attrs)
            layer.register_parameter("weight_scale", weight_scale)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        weight = layer.weight.data
        if self.is_checkpoint_serialized:
            self.quant_config.note_loaded()
        elif weight.dtype != torch.int8:
            # Quantization runs on CUDA; a layer staged on CPU is round-tripped
            # on its own. (The transformer loader currently materialises every
            # online-quantized component on the GPU before this runs, so INT8
            # saves memory after load, not during it.)
            home = weight.device
            weight_q, weight_scale = (
                torch.ops.sgl_kernel.convrot_rotate_quantize_activation(
                    weight.to("cuda", non_blocking=True).to(torch.bfloat16),
                    self.group_size,
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
        # The ops add a BF16 bias; an FP16 pipeline loads FP16 parameters. A
        # deferred bias (skip_bias_add) never reaches the ops and keeps its dtype.
        if (
            layer.bias is not None
            and layer.bias.dtype != torch.bfloat16
            and not layer.skip_bias_add
        ):
            layer.bias = Parameter(
                layer.bias.data.to(torch.bfloat16), requires_grad=False
            )

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
            self.group_size,
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


def _is_sgl_kernel_int8(layer: torch.nn.Module) -> bool:
    # LoRA wrappers and other non-LinearBase modules carry no quant_method.
    return isinstance(layer, LinearBase) and isinstance(
        layer.quant_method, ConvRotInt8SglKernelLinearMethod
    )


def convrot_int8_shares_input(layers: Sequence[torch.nn.Module]) -> bool:
    """Whether ``apply_convrot_int8_shared_input`` reproduces ``layer(x)`` for
    every layer in ``layers``."""
    if not layers or not all(
        _is_sgl_kernel_int8(layer) and _forward_is_plain_apply(layer)
        for layer in layers
    ):
        return False
    # One rotated input serves every layer only at one group width.
    return len({layer.quant_method.group_size for layer in layers}) == 1


def apply_convrot_int8_shared_input(
    x: torch.Tensor, layers: Sequence[LinearBase]
) -> list[torch.Tensor]:
    """``[layer(x)[0] for layer in layers]`` with ``x`` rotated and quantized once.

    Bitwise identical to applying each layer on its own; see
    ``convrot_int8_linear_prequant`` in sgl-kernel.
    """
    group_size = layers[0].quant_method.group_size
    x_q, x_scale = torch.ops.sgl_kernel.convrot_rotate_quantize_activation(
        _as_rows(x), group_size
    )
    outs = []
    for layer in layers:
        out = torch.ops.sgl_kernel.convrot_int8_linear_prequant(
            x_q,
            x_scale,
            layer.weight,
            layer.weight_scale,
            layer.bias,
            group_size,
        )
        outs.append(_like_input(out=out, x=x))
    return outs


def apply_convrot_int8_shared_input_out(
    x: torch.Tensor, layers: Sequence[LinearBase], outs: Sequence[torch.Tensor]
) -> None:
    """Writes ``layer(x)[0]`` into ``out`` for each (layer, out) pair with ``x``
    rotated and quantized once; ``out`` must be a contiguous BF16 slice shaped
    like ``layer(x)[0]``, so ``x`` must be BF16 as well (an FP16 ``x`` would
    have ``layer(x)`` return FP16; use ``apply_convrot_int8_shared_input`` for
    it). Bitwise identical to ``apply_convrot_int8_shared_input`` and to
    ``layer(x)``; see ``convrot_int8_linear_prequant_out`` in sgl-kernel.
    """
    if x.dtype != torch.bfloat16:
        raise ValueError(
            "apply_convrot_int8_shared_input_out writes BF16 outputs, so its "
            f"input must be BF16, got {x.dtype}"
        )
    group_size = layers[0].quant_method.group_size
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
    return _is_sgl_kernel_int8(layer) and _forward_is_plain_apply(layer)


def apply_convrot_int8_gelu_input(layer: LinearBase, x: torch.Tensor) -> torch.Tensor:
    """``layer(F.gelu(x, approximate="tanh"))[0]`` as one op, bitwise identical
    to the eager GELU followed by the layer for a BF16 ``x`` (an FP16 ``x`` is
    cast to BF16 before the GELU); see ``convrot_int8_fused_linear_gelu_input``
    in sgl-kernel."""
    out = torch.ops.sgl_kernel.convrot_int8_fused_linear_gelu_input(
        _as_rows(x),
        layer.weight,
        layer.weight_scale,
        layer.bias,
        layer.quant_method.group_size,
    )
    return _like_input(out=out, x=x)
