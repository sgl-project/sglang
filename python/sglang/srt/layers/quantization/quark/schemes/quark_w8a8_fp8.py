# SPDX-License-Identifier: Apache-2.0

from typing import Any, Callable, Optional, cast

import torch
from torch.nn import Parameter

from sglang.kernels.ops.quantization.fp8_kernel import is_fp8_fnuz
from sglang.srt.layers.parameter import (
    BlockQuantScaleParameter,
    ChannelQuantScaleParameter,
    ModelWeightParameter,
    PerTensorScaleParameter,
)
from sglang.srt.layers.quantization.fp8_utils import (
    _use_aiter_bpreshuffle_gfx95,
    aiter_w8a8_block_fp8_linear,
    apply_fp8_linear,
    cutlass_fp8_supported,
    dispatch_w8a8_block_fp8_linear,
    normalize_e4m3fn_to_e4m3fnuz,
    use_aiter_triton_gemm_w8a8_tuned_gfx950,
    validate_fp8_block_shape,
)
from sglang.srt.layers.quantization.quark.schemes import QuarkLinearScheme
from sglang.srt.layers.quantization.utils import requantize_with_max_scale
from sglang.srt.utils import get_bool_env_var, is_hip, set_weight_attrs

__all__ = ["QuarkW8A8Fp8", "QuarkW8A8Fp8Block"]

_is_fp8_fnuz = is_fp8_fnuz()
_is_hip = is_hip()
_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip
if _use_aiter:
    from aiter.ops.shuffle import shuffle_weight


class QuarkW8A8Fp8(QuarkLinearScheme):

    def __init__(
        self, weight_config: dict[str, Any], input_config: Optional[dict[str, Any]]
    ):
        self.cutlass_fp8_supported = cutlass_fp8_supported()
        self.weight_qscheme = cast(str, weight_config.get("qscheme"))
        self.is_static_input_scheme: bool = False
        self.input_qscheme: Optional[str] = None
        if input_config is not None:
            self.is_static_input_scheme = not cast(bool, input_config.get("is_dynamic"))
            self.input_qscheme = cast(str, input_config.get("qscheme"))

        self.per_token = (
            not self.is_static_input_scheme and self.input_qscheme == "per_channel"
        )
        self.out_dtype = torch.get_default_dtype()

    @classmethod
    def get_min_capability(cls) -> int:
        # lovelace and up
        return 89

    def process_weights_after_loading(self, layer) -> None:
        # If per tensor, when we have a fused module (e.g. QKV) with per
        # tensor scales (thus N scales being passed to the kernel),
        # requantize so we can always run per tensor
        if self.weight_qscheme == "per_tensor":
            if _is_fp8_fnuz:
                input_scale = getattr(layer, "input_scale", None)
                weight, max_w_scale, input_scale = normalize_e4m3fn_to_e4m3fnuz(
                    weight=layer.weight,
                    weight_scale=layer.weight_scale,
                    input_scale=input_scale,
                )
                if input_scale is not None:
                    layer.input_scale = Parameter(input_scale, requires_grad=False)
            else:
                max_w_scale = layer.weight_scale
                weight = layer.weight

            max_w_scale, weight = requantize_with_max_scale(
                weight=weight,
                weight_scale=max_w_scale,
                logical_widths=layer.logical_widths,
            )

            layer.weight = Parameter(weight.t(), requires_grad=False)
            layer.weight_scale = Parameter(max_w_scale, requires_grad=False)

        # If channelwise, scales are already lined up, so just transpose.
        elif self.weight_qscheme == "per_channel":
            weight = layer.weight

            if _is_fp8_fnuz:
                input_scale = getattr(layer, "input_scale", None)
                weight, weight_scale, input_scale = normalize_e4m3fn_to_e4m3fnuz(
                    weight=weight,
                    weight_scale=layer.weight_scale,
                    input_scale=input_scale,
                )
                if input_scale is not None:
                    layer.input_scale = Parameter(input_scale, requires_grad=False)
            else:
                weight_scale = layer.weight_scale.data
            if self.per_token:
                weight_scale = weight_scale.view(-1, 1)
            if _use_aiter:
                layer.weight = Parameter(
                    shuffle_weight(weight, (16, 16)).t(), requires_grad=False
                )
            else:
                layer.weight = Parameter(weight.t(), requires_grad=False)
            # required by torch.compile to be torch.nn.Parameter
            layer.weight_scale = Parameter(weight_scale, requires_grad=False)

        else:
            raise ValueError(f"Unknown quantization scheme {self.weight_qscheme}")

        # INPUT SCALE
        if self.is_static_input_scheme:
            layer.input_scale = Parameter(layer.input_scale.max(), requires_grad=False)
        else:
            layer.input_scale = None

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

        # WEIGHT
        weight = ModelWeightParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition,
                dtype=torch.float8_e4m3fn,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight", weight)

        # WEIGHT SCALE
        if self.weight_qscheme == "per_channel":
            weight_scale = ChannelQuantScaleParameter(
                data=torch.empty((sum(output_partition_sizes)), dtype=torch.float32),
                output_dim=0,
                weight_loader=weight_loader,
            )
        else:
            assert self.weight_qscheme == "per_tensor"
            weight_scale = PerTensorScaleParameter(
                data=torch.empty(len(output_partition_sizes), dtype=torch.float32),
                weight_loader=weight_loader,
            )
            set_weight_attrs(weight_scale, {"needs_scalar_to_array": True})

        # min requirement for fp8 kernels
        weight_scale[:] = torch.finfo(torch.float32).min
        layer.register_parameter("weight_scale", weight_scale)

        # INPUT SCALE
        if self.is_static_input_scheme:
            input_scale = PerTensorScaleParameter(
                data=torch.empty(len(output_partition_sizes), dtype=torch.float32),
                weight_loader=weight_loader,
            )
            input_scale[:] = torch.finfo(torch.float32).min
            set_weight_attrs(input_scale, {"needs_scalar_to_array": True})
            layer.register_parameter("input_scale", input_scale)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        return apply_fp8_linear(
            x,
            layer.weight,
            layer.weight_scale,
            input_scale=layer.input_scale,
            bias=bias,
            cutlass_fp8_supported=self.cutlass_fp8_supported,
            use_per_token_if_dynamic=self.per_token,
        )


class QuarkW8A8Fp8Block(QuarkLinearScheme):
    """Quark per-block FP8 linear scheme.

    The activation is per-group dynamic FP8 with ``group_size == block_k``;
    the quantization happens inside the kernel (``per_token_group_quant_fp8``
    on the Triton path, ``aiter_per1x128_quant`` on the AITER path).
    """

    # BlockQuantScaleParameter (and the merged/QKV block-scale offset math it
    # needs) is only implemented on the v2 loader path.
    requires_weight_loader_v2 = True

    def __init__(
        self,
        weight_config: dict[str, Any],
        input_config: Optional[dict[str, Any]],
    ):
        self.weight_block_size = tuple(weight_config["block_size"])
        self.scale_storage = weight_config.get("scale_type", "float32")
        assert self.scale_storage in (
            "float32",
            "float8_e8m0fnu",
        ), f"unsupported weight scale_type {self.scale_storage!r}"
        self.w8a8_block_fp8_linear = dispatch_w8a8_block_fp8_linear()

    @classmethod
    def get_min_capability(cls) -> int:
        return 89

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size: int,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        output_size: int,
        params_dtype: torch.dtype,
        weight_loader,
        **kwargs,
    ) -> None:
        output_size_per_partition = sum(output_partition_sizes)
        layer.logical_widths = output_partition_sizes
        layer.weight_block_size = list(self.weight_block_size)
        layer.orig_dtype = params_dtype

        validate_fp8_block_shape(
            layer,
            input_size,
            output_size,
            input_size_per_partition,
            output_partition_sizes,
            self.weight_block_size,
        )

        weight = ModelWeightParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition,
                dtype=torch.float8_e4m3fn,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight", weight)

        block_n, block_k = self.weight_block_size
        # Register the parameter directly in its on-disk dtype.
        # ``copy_with_check`` (parameter.py) shares rank 0 across all fp8
        # variants including float8_e8m0fnu, so safetensors copies into the
        # matching-dtype param without going through a uint8 placeholder.
        # We upcast to fp32 in process_weights_after_loading.
        on_disk_dtype = (
            torch.float8_e8m0fnu
            if self.scale_storage == "float8_e8m0fnu"
            else torch.float32
        )
        weight_scale = BlockQuantScaleParameter(
            data=torch.empty(
                (output_size_per_partition + block_n - 1) // block_n,
                (input_size_per_partition + block_k - 1) // block_k,
                dtype=on_disk_dtype,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        # ``deepseek_v4.remap_weight_name_to_dpsk_hf_format`` renames Quark's
        # on-disk ``.weight_scale`` to ``.weight_scale_inv`` for ``self_attn``
        layer.register_parameter("weight_scale_inv", weight_scale)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if self.scale_storage == "float8_e8m0fnu":
            weight_scale = layer.weight_scale_inv.data.to(torch.float32)
        else:
            weight_scale = layer.weight_scale_inv.data
        weight = layer.weight.data

        if _is_fp8_fnuz:
            weight, weight_scale, _ = normalize_e4m3fn_to_e4m3fnuz(
                weight=weight, weight_scale=weight_scale
            )

        layer.weight = Parameter(weight, requires_grad=False)
        layer.weight_scale_inv = Parameter(weight_scale, requires_grad=False)
        layer.input_scale = None

        # gfx95 + AITER bpreshuffle path requires the weight to be pre-shuffled
        # in (16, 16) tiles before the gemm.
        if (
            _use_aiter_bpreshuffle_gfx95
            and self.w8a8_block_fp8_linear is aiter_w8a8_block_fp8_linear
        ):
            n, k = layer.weight.shape
            if not use_aiter_triton_gemm_w8a8_tuned_gfx950(n, k):
                layer.weight = Parameter(
                    shuffle_weight(layer.weight, (16, 16)), requires_grad=False
                )

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # DSV4 may pass a pre-quantized activation as an (x_fp8, x_scale) tuple
        # (e.g. from the fused rmsnorm+fp8-quant path); forward the scale in that
        # case instead of re-quantizing inside the gemm.
        if isinstance(x, tuple):
            input_x, input_scale = x
        else:
            input_x, input_scale = x, None
        return self.w8a8_block_fp8_linear(
            input=input_x,
            weight=layer.weight,
            block_size=self.weight_block_size,
            weight_scale=layer.weight_scale_inv,
            input_scale=input_scale,
            bias=bias,
        )
