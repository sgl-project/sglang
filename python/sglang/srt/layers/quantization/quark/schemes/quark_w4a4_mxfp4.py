# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Any, Callable, Optional

import torch

from sglang.srt.layers.parameter import GroupQuantScaleParameter, PackedvLLMParameter
from sglang.srt.layers.quantization.quark.schemes import QuarkLinearScheme
from sglang.srt.utils import get_bool_env_var, is_hip
from sglang.srt.utils.common import direct_register_custom_op, mxfp_supported

_is_hip = is_hip()

# On GPUs that lack the fp4-activation WMMA scale instruction
# (V_WMMA_SCALE_F32_32X16X128_F4, e.g. gfx1250) the a4w4 (fp4 x fp4) linear GEMM
# cannot run. The MoE path is switched to a8w4 via AITER_FORCE_A8W4=1 (handled
# inside aiter.fused_moe); there is currently no working dense a8w4 GEMM for
# plain nn.Linear on this arch, so under the same flag the (few) MXFP4-quantized
# linear layers dequantize their FP4 weights to bf16 once at load and run a
# plain bf16 GEMM. This trades a little memory for correctness on hardware that
# cannot execute the fp4 kernel at all.
_dequant_linear_to_bf16 = _is_hip and get_bool_env_var("AITER_FORCE_A8W4", "false")

# MXFP4 (OCP MX FP4 / e2m1) decode table, indexed by the 4-bit code.
_MXFP4_VALUES = [
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
]


def _dequant_mxfp4_to_bf16(
    weight: torch.Tensor, weight_scale: torch.Tensor
) -> torch.Tensor:
    """Dequantize a packed MXFP4 weight ``(N, K//2)`` uint8 + e8m0 group scale
    ``(N, K//32)`` uint8 into a dense bf16 weight ``(N, K)``."""
    N, k_packed = weight.shape
    K = k_packed * 2
    lut = torch.tensor(_MXFP4_VALUES, device=weight.device, dtype=torch.float32)
    lo = (weight & 0xF).long()
    hi = (weight >> 4).long()
    vals = torch.empty(N, K, device=weight.device, dtype=torch.float32)
    vals[:, 0::2] = lut[lo]
    vals[:, 1::2] = lut[hi]
    # e8m0 byte b decodes to 2^(b-127); 255 is the NaN/Inf sentinel (unused by
    # real weights) -> map to 0 so it can't poison the matmul.
    scale = torch.exp2(weight_scale.to(torch.float32) - 127.0)
    scale = torch.where(weight_scale == 255, torch.zeros_like(scale), scale)
    scale = scale.view(N, K // 32, 1)
    w = (vals.view(N, K // 32, 32) * scale).view(N, K)
    return w.to(torch.bfloat16)


if _is_hip:
    from aiter.ops.triton.gemm.fused.fused_gemm_afp4wfp4_split_cat import (
        fused_gemm_afp4wfp4_split_cat as _fused_gemm_afp4wfp4_split_cat_orig,
    )
    from aiter.ops.triton.gemm_afp4wfp4 import gemm_afp4wfp4 as _gemm_afp4wfp4_orig
    from aiter.ops.triton.gemm_afp4wfp4_pre_quant_atomic import (
        gemm_afp4wfp4_pre_quant as _gemm_afp4wfp4_pre_quant_orig,
    )
    from aiter.ops.triton.quant import dynamic_mxfp4_quant as _dynamic_mxfp4_quant_orig

    def _aiter_gemm_afp4wfp4(
        x: torch.Tensor,
        w: torch.Tensor,
        x_scales: torch.Tensor,
        w_scales: torch.Tensor,
        y: torch.Tensor,
    ) -> None:
        _gemm_afp4wfp4_orig(x, w, x_scales, w_scales, y.dtype, y)

    def _aiter_gemm_afp4wfp4_fake(
        x: torch.Tensor,
        w: torch.Tensor,
        x_scales: torch.Tensor,
        w_scales: torch.Tensor,
        y: torch.Tensor,
    ) -> None:
        return None

    direct_register_custom_op(
        op_name="aiter_gemm_afp4wfp4",
        op_func=_aiter_gemm_afp4wfp4,
        mutates_args=["y"],
        fake_impl=_aiter_gemm_afp4wfp4_fake,
    )

    def gemm_afp4wfp4(x, w, x_scales, w_scales, dtype, y):
        torch.ops.sglang.aiter_gemm_afp4wfp4(x, w, x_scales, w_scales, y)

    def _aiter_gemm_afp4wfp4_pre_quant(
        x: torch.Tensor,
        w: torch.Tensor,
        w_scales: torch.Tensor,
        y: torch.Tensor,
    ) -> None:
        _gemm_afp4wfp4_pre_quant_orig(x, w, w_scales, y.dtype, y)

    def _aiter_gemm_afp4wfp4_pre_quant_fake(
        x: torch.Tensor,
        w: torch.Tensor,
        w_scales: torch.Tensor,
        y: torch.Tensor,
    ) -> None:
        return None

    direct_register_custom_op(
        op_name="aiter_gemm_afp4wfp4_pre_quant",
        op_func=_aiter_gemm_afp4wfp4_pre_quant,
        mutates_args=["y"],
        fake_impl=_aiter_gemm_afp4wfp4_pre_quant_fake,
    )

    def gemm_afp4wfp4_pre_quant(x, w, w_scales, dtype, y):
        torch.ops.sglang.aiter_gemm_afp4wfp4_pre_quant(x, w, w_scales, y)

    def _aiter_dynamic_mxfp4_quant(
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return _dynamic_mxfp4_quant_orig(x)

    def _aiter_dynamic_mxfp4_quant_fake(
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        M, N = x.shape
        x_fp4 = torch.empty((M, N // 2), dtype=torch.uint8, device=x.device)
        blockscale = torch.empty(
            (M, (N + 31) // 32), dtype=torch.uint8, device=x.device
        )
        return x_fp4, blockscale

    direct_register_custom_op(
        op_name="aiter_dynamic_mxfp4_quant",
        op_func=_aiter_dynamic_mxfp4_quant,
        mutates_args=[],
        fake_impl=_aiter_dynamic_mxfp4_quant_fake,
    )

    def dynamic_mxfp4_quant(x):
        return torch.ops.sglang.aiter_dynamic_mxfp4_quant(x)

    def _aiter_fused_gemm_split_cat(
        x: torch.Tensor,
        w: torch.Tensor,
        y: torch.Tensor,
        x_scale: torch.Tensor,
        w_scale: torch.Tensor,
        S1: int,
        S2: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return _fused_gemm_afp4wfp4_split_cat_orig(
            x=x,
            w=w,
            y=y,
            x_scale=x_scale,
            w_scale=w_scale,
            S1=S1,
            S2=S2,
            dtype=y.dtype,
        )

    def _aiter_fused_gemm_split_cat_fake(
        x: torch.Tensor,
        w: torch.Tensor,
        y: torch.Tensor,
        x_scale: torch.Tensor,
        w_scale: torch.Tensor,
        S1: int,
        S2: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        M = x.shape[0]
        D = y.shape[1]
        S3 = y.shape[2]
        c1 = torch.empty((M, D, S1 + S3), dtype=y.dtype, device=x.device)
        c2 = torch.empty((M, D, S2), dtype=y.dtype, device=x.device)
        return c1, c2

    direct_register_custom_op(
        op_name="aiter_fused_gemm_split_cat",
        op_func=_aiter_fused_gemm_split_cat,
        mutates_args=[],
        fake_impl=_aiter_fused_gemm_split_cat_fake,
    )

    def fused_gemm_afp4wfp4_split_cat(x, w, y, x_scale, w_scale, S1, S2, dtype):
        return torch.ops.sglang.aiter_fused_gemm_split_cat(
            x, w, y, x_scale, w_scale, S1, S2
        )


__all__ = ["QuarkW4A4MXFP4"]
logger = logging.getLogger(__name__)

OCP_MX_BLOCK_SIZE = 32


class QuarkW4A4MXFP4(QuarkLinearScheme):

    def __init__(
        self,
        weight_quant_spec: dict[str, Any],
        input_quant_spec: dict[str, Any],
        is_checkpoint_mxfp4_serialized: bool = True,
    ):
        self.out_dtype = torch.get_default_dtype()
        self.qscheme = "per_group"
        self.weight_quant_spec = weight_quant_spec
        self.input_quant_spec = input_quant_spec
        self.is_checkpoint_mxfp4_serialized = is_checkpoint_mxfp4_serialized

        if not self.is_checkpoint_mxfp4_serialized:
            if not mxfp_supported():
                raise NotImplementedError(
                    "Online MXFP4 quantization requires an AMD ROCm device with "
                    "FP4 hardware support (gfx95x, e.g. MI355x)."
                )
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

        if _dequant_linear_to_bf16:
            w_bf16 = _dequant_mxfp4_to_bf16(layer.weight.data, layer.weight_scale.data)
            layer.weight = torch.nn.Parameter(w_bf16, requires_grad=False)
            # FP4 block scales are folded into the bf16 weight; drop them.
            layer.weight_scale = None
            layer.dequantized_bf16 = True

    def create_weights(
        self,
        layer: torch.nn.Module,
        output_partition_sizes: list[int],
        input_size_per_partition: int,
        params_dtype: torch.dtype,
        weight_loader: Callable,
        **kwargs,
    ):
        self.input_size_per_partition = input_size_per_partition

        output_size_per_partition = sum(output_partition_sizes)
        self.output_size_per_partition = output_size_per_partition

        layer.logical_widths = output_partition_sizes

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

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # bf16 fallback: FP4 weights were dequantized to bf16 at load time
        # because this HW cannot run the fp4 GEMM. Run a plain bf16 linear.
        # (The fused tuple-input paths below are only used by MLA attention
        # projections, which are excluded from quantization for this checkpoint,
        # so a plain-tensor activation is what reaches here.)
        if getattr(layer, "dequantized_bf16", False):
            if isinstance(x, tuple):
                x = x[0]
            return torch.nn.functional.linear(x, layer.weight, bias)

        # Bias will be added after the GEMM if provided
        three_d = False
        fused_gemm_split_cat = False
        x_s = None
        y = None

        if isinstance(x, tuple):
            assert len(x) in [
                2,
                3,
                5,
            ], "For tuple input, only (x, x_s), (x, x_s, y), or (x, y, S1, S2, out_dtype) formats are accepted"
            if len(x) == 2:
                x, x_s = x
            elif len(x) == 3:
                x, x_s, y = x
            elif len(x) == 5:
                x, y, S1, S2, out_dtype = x
                fused_gemm_split_cat = True

        use_fused_quant_gemm = (
            not fused_gemm_split_cat
            and x_s is None
            and y is not None
            and layer.weight.shape[0] == y.shape[1]
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
        elif fused_gemm_split_cat:
            k, v = fused_gemm_afp4wfp4_split_cat(
                x=x_q,
                w=layer.weight,
                y=y,
                x_scale=x_s,
                w_scale=layer.weight_scale,
                S1=S1,
                S2=S2,
                dtype=out_dtype,
            )
        else:
            gemm_afp4wfp4(x_q, layer.weight, x_s, layer.weight_scale, self.out_dtype, y)

        if bias is not None:
            y = y + bias

        if fused_gemm_split_cat:
            return k, v
        elif three_d:
            return y.view(*output_shape)
        else:
            return y
