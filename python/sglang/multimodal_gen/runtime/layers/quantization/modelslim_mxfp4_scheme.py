"""ModelSlim MXFP4 scheme for pre-quantized weight inference on NPU.

Loads repacked msmodelslim weights and selects single/dual-level MXFP4 matmul.

Checkpoint tensor formats (verified from msmodelslim export):
  weight:           [out, in/2]         uint8          (packed FP4 E2M1)
  weight_scale:     [out, in/32]        uint8          (L1 block scales, e8m0+127)
  weight_dual_scale:[out, in/512, 1]    float32        (L0 coarse scales)
  mul_scale:        [in]                float32        (smooth quant activation scale)
"""

from typing import List, Optional

import torch

from sglang.multimodal_gen.runtime.layers.quantization.modelslim_mxfp_utils import (
    mxfp4_quant_kwargs,
    resolve_precision,
)
from sglang.multimodal_gen.runtime.models.parameter import (
    GroupQuantScaleParameter,
    ModelWeightParameter,
    RowvLLMParameter,
)
from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.srt.layers.quantization.modelslim.schemes import ModelSlimLinearScheme

if current_platform.is_npu():
    import torch_npu  # noqa: E402

MXFP4_BLOCK_SIZE = 32
# L1 (dual) scale groups this many L0 blocks together.
# L1 block covers 16 * 32 = 512 elements.
MXFP4_DUAL_LEVEL_RATIO = 16
MXFP4_PACK_FACTOR = 2


class ModelSlimMXFP4Scheme(ModelSlimLinearScheme):
    def __init__(
        self,
        quant_config: dict,
        prefix: str,
        quant_type: str,
    ):
        self.quant_config = quant_config
        self.prefix = prefix
        self.quant_type = quant_type

        self.is_dual_scale = quant_type == "W4A4_MXFP4_DUALSCALE"
        self.dual_scale_key = prefix + ".weight_dual_scale"
        self.mul_scale_key = prefix + ".mul_scale"
        self.legacy_mul_scale_key = prefix + ".div.mul_scale"
        self.has_mul_scale = (
            self.legacy_mul_scale_key in quant_config
            or self.mul_scale_key in quant_config
        )
        self.single_level_kernel = None
        self.w4a8_kernel = None
        if not self.is_dual_scale:
            from sglang.srt.hardware_backend.npu.quantization.linear_method_npu import (
                NPUMXFP4W4A8OfflineLinearMethod,
                NPUSingleLevelMXFP4OfflineLinearMethod,
            )

            self.single_level_kernel = NPUSingleLevelMXFP4OfflineLinearMethod()
            self.w4a8_kernel = NPUMXFP4W4A8OfflineLinearMethod()
        else:
            if self.dual_scale_key not in self.quant_config:
                raise ValueError(
                    f"Dual-level MXFP4 quantization requires missing '{self.dual_scale_key}' in quant_config."
                    "Check that the model was exported with dual-level quantization."
                )

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        weight_loader = extra_weight_attrs.get("weight_loader")
        output_size_per_partition = sum(output_partition_sizes)
        alignment = 512 if self.is_dual_scale else 64
        if input_size_per_partition % alignment:
            raise ValueError(
                f"{self.prefix}: MXFP4 input partition must be divisible by {alignment}"
            )

        # wan_repack converts numeric FP8 containers to packed FP4 bytes.
        weight = ModelWeightParameter(
            data=torch.empty(
                (
                    output_size_per_partition,
                    input_size_per_partition // MXFP4_PACK_FACTOR,
                ),
                dtype=torch.uint8,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        weight.missing_param_init = "error"
        layer.register_parameter("weight", weight)

        # L1 block scale: uint8 [out, in/32], e8m0 scale with +127 offset.
        scale_dim = input_size_per_partition // MXFP4_BLOCK_SIZE
        weight_scale = GroupQuantScaleParameter(
            data=torch.empty(
                (output_size_per_partition, scale_dim),
                dtype=torch.uint8,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        weight_scale.missing_param_init = "error"
        layer.register_parameter("weight_scale", weight_scale)
        if self.is_dual_scale:
            # L0 (coarse) scale for dual-level quantization matmul.
            # Each L0 block covers MXFP4_DUAL_LEVEL_RATIO L1 blocks = 16 * 32 = 512 elements.
            dual_scale_dim = scale_dim // MXFP4_DUAL_LEVEL_RATIO  # in/32 / 16 = in/512
            weight_dual_scale = GroupQuantScaleParameter(
                data=torch.empty(
                    (output_size_per_partition, dual_scale_dim, 1),
                    dtype=torch.float32,
                ),
                input_dim=1,
                output_dim=0,
                weight_loader=weight_loader,
            )
            weight_dual_scale.missing_param_init = "error"
            layer.register_parameter("weight_dual_scale", weight_dual_scale)

        if self.has_mul_scale:
            # Smooth quant activation scale (mul_scale) from NonFusionSmoothQuantWrapper.
            # msmodelslim exports this as `<prefix>.div.mul_scale` with shape [in].
            # After repack, it becomes `<prefix>.mul_scale`.
            # This is CRITICAL: the offline-quantized weights were calibrated with
            # x * mul_scale applied to the activation. Omitting it causes mosaic output.
            mul_scale = RowvLLMParameter(
                data=torch.empty(
                    (input_size_per_partition,),
                    dtype=torch.float32,
                ),
                input_dim=0,
                weight_loader=weight_loader,
            )
            mul_scale.missing_param_init = "error"
            layer.register_parameter("mul_scale", mul_scale)

    def process_weights_after_loading(self, layer: torch.nn.Module):
        if not self.is_dual_scale:
            policy = self.quant_config.get("timestep_policy", {}).get("w4a4_linear", {})
            uses_w4a8 = "W4A8" in policy.values()
            # resolve_precision falls back to W4A4 for steps the policy does not
            # cover, so a policy without a W4A8 "default" still executes W4A4.
            uses_w4a4 = "W4A4" in policy.values() or "default" not in policy
            if uses_w4a8 and uses_w4a4:
                raise NotImplementedError(
                    f"{self.prefix}: mixed W4A4/W4A8 linear timestep policies are "
                    "unsupported for single-level MXFP4. Both load paths now lay "
                    "the weight out as FRACTAL_NZ, but the two applies quantize "
                    "activations differently (FP8 vs FP4) and the per-timestep "
                    "switch has not been verified on hardware; use a uniform "
                    "policy."
                )
            kernel = self.w4a8_kernel if uses_w4a8 else self.single_level_kernel
            kernel.process_weights_after_loading(layer)
            layer.mxfp4_quant_kwargs = mxfp4_quant_kwargs(self.quant_config)
            if self.has_mul_scale:
                mul_scale = layer.mul_scale.data
                if not mul_scale.is_npu:
                    mul_scale = mul_scale.to(f"npu:{torch.npu.current_device()}")
                layer.mul_scale = torch.nn.Parameter(mul_scale, requires_grad=False)
                layer.use_mul_scale = not torch.all(mul_scale == 1.0).item()
            else:
                layer.use_mul_scale = False
            return

        # Preserve packed FP4 bytes when changing the storage format.
        weight = layer.weight.data
        if not weight.is_npu:
            weight = weight.to(f"npu:{torch.npu.current_device()}")
        # npu_dual_level_quant_matmul requires x2 in FRACTAL_NZ format (format 29).
        weight = torch_npu.npu_format_cast(
            weight.view(torch.int8), 29, customize_dtype=torch.int8
        )
        layer.weight = torch.nn.Parameter(weight, requires_grad=False)

        # Reshape weight_scale: [out, in/32] -> [out, in/64, 2]
        # The dual-level matmul API expects L1 scales in this 3D format
        weight_scale = layer.weight_scale.data
        if not weight_scale.is_npu:
            weight_scale = weight_scale.to(f"npu:{torch.npu.current_device()}")
        weight_scale = weight_scale.reshape(weight_scale.shape[0], -1, 2)
        layer.weight_scale = torch.nn.Parameter(weight_scale, requires_grad=False)

        if self.is_dual_scale:
            # Transform weight_dual_scale: [out, in/512, 1] -> [in/512, out]
            weight_dual_scale = layer.weight_dual_scale.data
            if not weight_dual_scale.is_npu:
                weight_dual_scale = weight_dual_scale.to(
                    f"npu:{torch.npu.current_device()}"
                )
            weight_dual_scale = (
                weight_dual_scale.squeeze(-1).transpose(0, 1).contiguous()
            )
            layer.weight_dual_scale = torch.nn.Parameter(
                weight_dual_scale, requires_grad=False
            )

        if self.has_mul_scale:
            # Move mul_scale to NPU if present and not already there
            mul_scale = layer.mul_scale.data
            if not mul_scale.is_npu:
                mul_scale = mul_scale.to(f"npu:{torch.npu.current_device()}")
            layer.mul_scale = torch.nn.Parameter(mul_scale, requires_grad=False)
            layer.use_mul_scale = not torch.all(mul_scale == 1.0).item()
        else:
            layer.use_mul_scale = False

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        precision = resolve_precision(self.quant_config, "w4a4_linear", "W4A4")
        if not self.is_dual_scale:
            if getattr(layer, "use_mul_scale", False):
                x = x * layer.mul_scale.to(x.dtype)
            kernel = (
                self.w4a8_kernel if precision == "W4A8" else self.single_level_kernel
            )
            return kernel.apply(layer, x, bias)

        if precision != "W4A4":
            raise NotImplementedError(
                "Dual-scale MXFP4 does not support W4A8 timestep switching"
            )

        original_dtype = x.dtype
        if original_dtype not in (torch.float16, torch.bfloat16):
            x = x.to(torch.bfloat16)
            original_dtype = torch.bfloat16

        # Flatten to 2D for npu_dynamic_dual_level_mx_quant
        input_shape = x.shape
        x_2d = x.reshape(-1, x.shape[-1])

        # Apply smooth quant scale before activation quantization.
        # The offline-quantized weights were calibrated under x * mul_scale,
        # so we MUST apply it here for scale alignment.
        if getattr(layer, "use_mul_scale", False):
            x_2d = x_2d * layer.mul_scale.to(x_2d.dtype)

        # Dual-level MXFP4 activation quantization
        x1, l0_scale, l1_scale = torch_npu.npu_dynamic_dual_level_mx_quant(
            x_2d, smooth_scale=None
        )

        # Dual-level MXFP4 matmul
        output = torch_npu.npu_dual_level_quant_matmul(
            x1,
            layer.weight,
            l0_scale,
            layer.weight_dual_scale,
            l1_scale,
            layer.weight_scale,
            bias=bias.to(torch.float32) if bias is not None else None,
            output_dtype=original_dtype,
        )

        # Restore original shape
        output_shape = list(input_shape[:-1]) + [output.shape[-1]]
        return output.reshape(output_shape)
