# SPDX-License-Identifier: Apache-2.0

import logging
import threading
from typing import Any, Callable, Optional

import torch

from sglang.srt.layers.parameter import (
    GroupQuantScaleParameter,
    ModelWeightParameter,
    PackedvLLMParameter,
    PerTensorScaleParameter,
)
from sglang.srt.layers.quantization import QuantizationConfig
from sglang.srt.layers.quantization.dequantization import dequantize_nvfp4
from sglang.srt.layers.quantization.online_quantization import CopyNumelCounter
from sglang.srt.layers.quantization.quark.schemes import QuarkLinearScheme
from sglang.srt.layers.quantization.quark.utils import Nvfp4SourceConfig
from sglang.srt.utils import is_hip
from sglang.srt.utils.common import direct_register_custom_op, mxfp_supported

NVFP4_BLOCK_SIZE = 16

_is_hip = is_hip()
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

    # PackedvLLMParameter / ModelWeightParameter (online and NVFP4->MXFP4
    # paths) only implement the v2 loader API.
    requires_weight_loader_v2 = True

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

        # If dequantization_config is provided, we dequantize the source
        # checkpoint and re-quantize to MXFP4 at load time.
        if self.dequantization_config is not None:
            if isinstance(self.dequantization_config, Nvfp4SourceConfig):
                self._create_weights_from_nvfp4(
                    layer=layer,
                    output_size_per_partition=output_size_per_partition,
                    input_size_per_partition=input_size_per_partition,
                    output_partition_sizes=output_partition_sizes,
                    weight_loader=weight_loader,
                )
            else:
                raise NotImplementedError(
                    f"Requantization in QuarkW4A4MXFP4 from {self.dequantization_config.__class__.__name__} is not supported."
                )
        else:
            original_weight_loader = weight_loader
            if not self.is_checkpoint_mxfp4_serialized:
                weight_loader = self.get_online_mxfp4_weight_loader(
                    layer, weight_loader
                )

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

    def _create_weights_from_nvfp4(
        self,
        layer,
        output_size_per_partition,
        input_size_per_partition,
        output_partition_sizes,
        weight_loader,
    ):
        layer._nvfp4_loaded_numel = 0
        # torch.get_default_device() may return `cuda` (no index), which breaks
        # the `current_device() == idx` assert in the loader
        layer._load_device = torch.device(f"cuda:{torch.cuda.current_device()}")
        layer._nvfp4_loading_lock = threading.Lock()

        nvfp4_loader = self.get_online_nvfp4_to_mxfp4_weight_loader(
            layer, weight_loader
        )

        layer.register_parameter(
            "weight",
            ModelWeightParameter(
                data=torch.empty(
                    output_size_per_partition,
                    input_size_per_partition // 2,
                    dtype=torch.uint8,
                    device=layer._load_device,
                ),
                input_dim=1,
                output_dim=0,
                weight_loader=nvfp4_loader,
            ),
        )
        layer.register_parameter(
            "weight_scale",
            ModelWeightParameter(
                data=torch.empty(
                    output_size_per_partition,
                    input_size_per_partition // NVFP4_BLOCK_SIZE,
                    dtype=torch.float8_e4m3fn,
                    device=layer._load_device,
                ),
                input_dim=1,
                output_dim=0,
                weight_loader=nvfp4_loader,
            ),
        )
        layer.register_parameter(
            "weight_scale_2",
            PerTensorScaleParameter(
                data=torch.empty(
                    len(output_partition_sizes),
                    dtype=torch.float32,
                    device=layer._load_device,
                ),
                weight_loader=nvfp4_loader,
            ),
        )

        # NVFP4 checkpoints carry per-tensor `input_scale` (activation scale).
        # MXFP4 uses dynamic activation quantization, so we discard it, but
        # we still register the param so upstream model loaders that rename
        # `gate_proj.input_scale` -> `gate_up_proj.input_scale` find a slot
        # to write into
        def _discard_loader(param, loaded_weight, shard_id=None):
            pass

        layer.register_parameter(
            "input_scale",
            PerTensorScaleParameter(
                data=torch.empty(
                    len(output_partition_sizes),
                    dtype=torch.float32,
                    device=layer._load_device,
                ),
                weight_loader=_discard_loader,
            ),
        )

        layer.weight._param_name = "weight"
        layer.weight_scale._param_name = "weight_scale"
        layer.weight_scale_2._param_name = "weight_scale_2"

    def get_online_nvfp4_to_mxfp4_weight_loader(
        self,
        layer,
        original_weight_loader: Callable,
    ) -> Callable:
        """NVFP4 -> MXFP4 loader: dequantize+requantize once all source bytes
        are in place."""

        def loader(param, loaded_weight, shard_id=None):
            param_name = getattr(param, "_param_name", None)
            assert torch.cuda.current_device() == layer._load_device.index

            with layer._nvfp4_loading_lock:
                param = getattr(layer, param_name)

            kwargs = {"loaded_shard_id": shard_id} if shard_id is not None else {}
            counter = CopyNumelCounter()
            with counter:
                original_weight_loader(param, loaded_weight, **kwargs)

            with layer._nvfp4_loading_lock:
                layer._nvfp4_loaded_numel += counter.copied_numel
                target = (
                    layer.weight.numel()
                    + layer.weight_scale.numel()
                    + layer.weight_scale_2.numel()
                )
                if layer._nvfp4_loaded_numel == target:
                    # weight_scale_2 is one fp32 per output partition (e.g. 2
                    # for gate_up_proj, 3 for qkv_proj). Expand to a per-row
                    # scalar matching layer.weight's output dim so it
                    # broadcasts against the per-block scale.
                    per_row_scale_2 = layer.weight_scale_2.repeat_interleave(
                        torch.tensor(
                            layer.logical_widths, device=layer.weight_scale_2.device
                        )
                    ).view(-1, 1)
                    # Dequantize to fp32: the intermediate feeds straight into the
                    # MXFP4 requant
                    dequantized_weight = dequantize_nvfp4(
                        layer.weight,
                        layer.weight_scale,
                        per_row_scale_2,
                        out_dtype=torch.float32,
                    )
                    mxfp4_weight, mxfp4_scale = dynamic_mxfp4_quant(dequantized_weight)
                    layer.weight = torch.nn.Parameter(mxfp4_weight, requires_grad=False)
                    layer.weight_scale = torch.nn.Parameter(
                        mxfp4_scale, requires_grad=False
                    )
                    del layer.weight_scale_2
                    del layer._load_device

        return loader

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
