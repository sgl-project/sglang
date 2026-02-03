# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Any, Callable, Optional

import torch

from sglang.srt.layers.parameter import GroupQuantScaleParameter, PackedvLLMParameter
from sglang.srt.layers.quantization.quark.schemes import QuarkScheme
from sglang.srt.layers.quantization.fp8 import Fp8LinearMethod
from sglang.srt.utils import is_hip, set_weight_attrs
from sglang.srt.layers.quantization import QuantizationConfig
from sglang.srt.layers.quantization.online_quantization import CopyNumelCounter

_is_hip = is_hip()
if _is_hip:
    from aiter.ops.triton.gemm_afp4wfp4 import gemm_afp4wfp4
    from aiter.ops.triton.gemm_afp4wfp4_pre_quant_atomic import gemm_afp4wfp4_pre_quant
    from aiter.ops.triton.quant import dynamic_mxfp4_quant


__all__ = ["QuarkW4A4MXFP4"]
logger = logging.getLogger(__name__)

OCP_MX_BLOCK_SIZE = 32


def _copy_missing_attrs(old: torch.Tensor, new: torch.Tensor) -> None:
    """Copies any attrs present in `old` but not in `new` to `new`"""
    new_attrs = set(dir(new))
    attrs_to_set = {}
    for attr in dir(old):
        if attr not in new_attrs:
            attrs_to_set[attr] = getattr(old, attr)
    set_weight_attrs(new, attrs_to_set)


class QuarkW4A4MXFP4(QuarkScheme):

    def __init__(
        self,
        weight_quant_spec: dict[str, Any],
        input_quant_spec: dict[str, Any],
        is_checkpoint_mxfp4_serialized: bool = True,
        dequantization_config: QuantizationConfig | None = None,
    ):
        self.out_dtype = torch.get_default_dtype()
        self.qscheme = "per_group"
        self.weight_quant_spec = weight_quant_spec
        self.input_quant_spec = input_quant_spec
        self.is_checkpoint_mxfp4_serialized = is_checkpoint_mxfp4_serialized
        self.dequantization_config = dequantization_config

        if not self.is_checkpoint_mxfp4_serialized:
            logger.info_once(
                "Using online MXFP4 quantization from a higher precision checkpoint. Beware that this optimization may degrade prediction quality - please validate your model accuracy. More details at https://docs.sglang.io/advanced_features/quantization.html#online-quantization."
            )

    @classmethod
    def get_min_capability(cls) -> int:
        return 70

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if not self.is_checkpoint_mxfp4_serialized:
            assert layer.weight.dtype == torch.uint8
            assert layer.weight_scale.dtype == torch.uint8

    def create_weights(
        self,
        layer: torch.nn.Module,
        output_partition_sizes: list[int],
        input_size_per_partition: int,
        params_dtype: torch.dtype,
        weight_loader: Callable,
        **kwargs
    ):
        self.input_size_per_partition = input_size_per_partition

        output_size_per_partition = sum(output_partition_sizes)
        self.output_size_per_partition = output_size_per_partition

        layer.logical_widths = output_partition_sizes

        # If dequantization_config is provided, we need to create FP8 weights first
        # for dequantization from FP8 checkpoint to MXFP4
        if self.dequantization_config is not None:
            # Create FP8 weights for re-quantization from FP8 checkpoint
            # Extract necessary parameters from dequantization_config
            weight_block_size = self.dequantization_config.weight_block_size

            if self.dequantization_config.use_mxfp8:
                raise NotImplementedError("use_mxfp8=True is not supported in Quark MXFP4 requantization.")

            # Determine if block quantization is used
            block_quant = weight_block_size is not None

            layer._fp8_weight_loaded_numel = 0
            layer._fp8_scale_loaded_numel = 0
            layer._load_device = torch.get_default_device()

            # Wrap the weight loader to handle FP8->MXFP4 conversion
            fp8_to_mxfp4_weight_loader = self.get_online_fp8_to_mxfp4_weight_loader(
                layer, weight_loader
            )

            # Create FP8 weight parameters on meta device to avoid memory overhead
            # They will be materialized on first load in the weight loader
            with torch.device("meta"):
                Fp8LinearMethod.create_fp8_weight_(
                    layer=layer,
                    block_quant=block_quant,
                    quant_config=self.dequantization_config,
                    use_mxfp8=False,
                    output_size_per_partition=output_size_per_partition,
                    input_size_per_partition=input_size_per_partition,
                    output_partition_sizes=output_partition_sizes,
                    weight_loader=fp8_to_mxfp4_weight_loader,
                    is_checkpoint_fp8_serialized=True,
                    params_dtype=params_dtype,
                    skip_block_quant_check=False,
                    input_size=kwargs.get("input_size", input_size_per_partition),
                    output_size=kwargs.get("output_size", output_size_per_partition),
                )
        else:
            original_weight_loader = weight_loader
            if not self.is_checkpoint_mxfp4_serialized:
                weight_loader = self.get_online_mxfp4_weight_loader(layer, weight_loader)

            # WEIGHT
            # Both serialized and online quantization use packed uint8 format
            weight = PackedvLLMParameter(
                data=torch.empty(
                    output_size_per_partition,
                    input_size_per_partition // 2,
                    dtype=torch.uint8,
                ),
                input_dim=1,
                output_dim=0,
                packed_dim=1,
                packed_factor=2,
                weight_loader=weight_loader,
            )
            layer.register_parameter("weight", weight)

            # WEIGHT SCALE
            weight_scale = GroupQuantScaleParameter(
                data=torch.empty(
                    output_size_per_partition,
                    input_size_per_partition // OCP_MX_BLOCK_SIZE,
                    dtype=torch.uint8,
                ),
                input_dim=1,
                output_dim=0,
                weight_loader=original_weight_loader,
            )
            layer.register_parameter("weight_scale", weight_scale)

    def get_online_mxfp4_weight_loader(
        self,
        layer,
        original_weight_loader: Callable,
    ) -> Callable:
        """
        Wrap the original weight loader to perform online MXFP4 quantization.
        """

        def online_mxfp4_weight_loader(
            param: torch.nn.Parameter,
            loaded_weight: torch.Tensor,
            shard_id: int | str | None = None,
        ):
            # Materialize on device the loaded weight.
            loaded_weight = loaded_weight.to(param.device)

            # Quantize the loaded weight shard immediately. Since MXFP4 uses per-group quantization, there is no need to load all shards (e.g. q_proj, k_proj, v_proj) before doing online quantization.
            qweight, weight_scale = dynamic_mxfp4_quant(loaded_weight)

            # Required e.g. for q_proj, k_proj, v_proj.
            kwargs = {}
            if shard_id is not None:
                kwargs["loaded_shard_id"] = shard_id

            # Use the original weight loader to handle the loading logic
            # (e.g. qkv sharding, etc.)
            original_weight_loader(param, qweight, **kwargs)

            layer.weight_scale.weight_loader(layer.weight_scale, weight_scale, **kwargs)

        return online_mxfp4_weight_loader

    def get_online_fp8_to_mxfp4_weight_loader(
        self,
        layer,
        original_weight_loader: Callable,
    ) -> Callable:
        """
        Wrap the original weight loader to perform FP8 to MXFP4 requantization.

        This loader handles:
        1. Loading FP8 weights and weight_scale_inv parameters
        2. Waiting for all shards (e.g., q_proj, k_proj, v_proj) to be loaded
        3. Dequantizing FP8 -> BF16 using weight_scale_inv
        4. Requantizing BF16 -> MXFP4
        """
        def online_fp8_to_mxfp4_weight_loader(
            param: torch.nn.Parameter,
            loaded_weight: torch.Tensor,
            shard_id: int | str | None = None,
        ):
            # Determine if we're loading weight or weight_scale_inv
            param_name = None
            for name, p in layer.named_parameters():
                if p is param:
                    param_name = name
                    break

            is_weight = param_name == "weight"
            is_weight_scale_inv = param_name == "weight_scale_inv"

            if not is_weight and not is_weight_scale_inv:
                # For other parameters (e.g., bias), use original loader
                kwargs = {}
                if shard_id is not None:
                    kwargs["loaded_shard_id"] = shard_id
                original_weight_loader(param, loaded_weight, **kwargs)
                return

            # Materialize FP8 parameters on first load from meta device
            if is_weight and layer._fp8_weight_loaded_numel == 0:
                assert param.device.type == "meta", f"Expected weight on meta device, got {param.device}"

                # Materialize weight parameter on device
                from sglang.srt.layers.parameter import ModelWeightParameter
                layer.weight = ModelWeightParameter(
                    data=torch.empty_like(param.data, device=layer._load_device),
                    input_dim=1,
                    output_dim=0,
                    weight_loader=online_fp8_to_mxfp4_weight_loader,
                )
                _copy_missing_attrs(param, layer.weight)
                param = layer.weight

            if is_weight_scale_inv and layer._fp8_scale_loaded_numel == 0:
                assert param.device.type == "meta", f"Expected weight_scale_inv on meta device, got {param.device}"

                # Materialize scale parameter on device
                from sglang.srt.layers.parameter import BlockQuantScaleParameter
                layer.weight_scale_inv = BlockQuantScaleParameter(
                    data=torch.empty_like(param.data, device=layer._load_device),
                    input_dim=1,
                    output_dim=0,
                    weight_loader=online_fp8_to_mxfp4_weight_loader,
                )
                _copy_missing_attrs(param, layer.weight_scale_inv)
                param = layer.weight_scale_inv

            # Move loaded weight to device
            loaded_weight = loaded_weight.to(layer._load_device)

            kwargs = {}
            if shard_id is not None:
                kwargs["loaded_shard_id"] = shard_id

            # Track how much data we are actually loading (`narrow` used in weight loader)
            copy_numel_counter = CopyNumelCounter()
            with copy_numel_counter:
                original_weight_loader(param, loaded_weight, **kwargs)

            # Update loading progress
            if is_weight:
                layer._fp8_weight_loaded_numel += copy_numel_counter.copied_numel
            elif is_weight_scale_inv:
                layer._fp8_scale_loaded_numel += copy_numel_counter.copied_numel

            # Check if both weight and scale are fully loaded
            weight_fully_loaded = layer._fp8_weight_loaded_numel == layer.weight.numel()
            scale_fully_loaded = layer._fp8_scale_loaded_numel == layer.weight_scale_inv.numel()

            # Perform dequantization and requantization only when both are fully loaded
            if weight_fully_loaded and scale_fully_loaded:
                # Abstract function placeholders - to be implemented
                def fp8_to_bf16(weight, weight_scale):
                    """Dequantize FP8 weight to BF16 using weight_scale_inv."""
                    raise NotImplementedError("fp8_to_bf16 dequantization not yet implemented")

                def quantize_mxfp4(weight_bf16):
                    """Quantize BF16 weight to MXFP4, returning (weight_mxfp4, weight_mxfp4_scale)."""
                    raise NotImplementedError("quantize_mxfp4 not yet implemented")

                # FP8 -> BF16 dequantization
                weight_bf16 = fp8_to_bf16(layer.weight, layer.weight_scale_inv)

                # BF16 -> MXFP4 requantization
                weight_mxfp4, weight_mxfp4_scale = quantize_mxfp4(weight_bf16)

                # Replace FP8 weight parameter with MXFP4 weight
                layer.weight = torch.nn.Parameter(weight_mxfp4, requires_grad=False)

                # Create and set MXFP4 weight_scale parameter
                layer.weight_scale = torch.nn.Parameter(weight_mxfp4_scale, requires_grad=False)

                # Clean up FP8 parameters and tracking attributes
                del layer.weight_scale_inv
                del layer._fp8_weight_loaded_numel
                del layer._fp8_scale_loaded_numel
                del layer._load_device

        return online_fp8_to_mxfp4_weight_loader

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Bias will be added after the GEMM if provided
        three_d = False
        x_s = None
        y = None
        if isinstance(x, tuple):
            assert len(x) in [
                2,
                3,
            ], "For tuple input, only (x, x_s) or (x, x_s, y) formats are accepted"
            if len(x) == 2:
                x, x_s = x
            elif len(x) == 3:
                x, x_s, y = x

        use_fused_quant_gemm = (
            x_s is None and y is not None and layer.weight.shape[0] == y.shape[1]
        )

        if x.dim() == 3:
            three_d = True
            x = x.view(-1, x.shape[-1])
            output_shape = [*x.shape[:-1], layer.weight.shape[0]]

        # use_fused_quant_gemm = true, x_q is a bf16/fp16 num
        # x_s is not None = true, x_q is uint8 num
        if use_fused_quant_gemm or x_s is not None:
            x_q = x
        else:
            x_q, x_s = dynamic_mxfp4_quant(x)

        if y is None:
            y = torch.empty(
                x_q.shape[0],
                layer.weight.shape[0],
                device=x_q.device,
                dtype=self.out_dtype,
            )

        if use_fused_quant_gemm:
            gemm_afp4wfp4_pre_quant(x_q, layer.weight, layer.weight_scale, y.dtype, y)
            y = y.to(x.dtype)
        else:
            gemm_afp4wfp4(x_q, layer.weight, x_s, layer.weight_scale, self.out_dtype, y)

        if bias is not None:
            y = y + bias

        if three_d:
            return y.view(*output_shape)

        return y
