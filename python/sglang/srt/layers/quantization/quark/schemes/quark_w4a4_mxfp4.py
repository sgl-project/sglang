# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Any, Callable, Optional

import torch

from sglang.srt.layers.parameter import GroupQuantScaleParameter, PackedvLLMParameter
from sglang.srt.layers.quantization.quark.schemes import QuarkScheme
from sglang.srt.utils import is_hip

_is_hip = is_hip()
if _is_hip:
    from aiter.ops.triton.gemm_afp4wfp4 import gemm_afp4wfp4
    from aiter.ops.triton.gemm_afp4wfp4_pre_quant_atomic import gemm_afp4wfp4_pre_quant
    from aiter.ops.triton.quant import dynamic_mxfp4_quant


__all__ = ["QuarkW4A4MXFP4"]
logger = logging.getLogger(__name__)

OCP_MX_BLOCK_SIZE = 32


class QuarkW4A4MXFP4(QuarkScheme):

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
            logger.info_once(
                "Using online MXFP4 quantization from a higher precision checkpoint. Beware that this optimization may degrade prediction quality - please validate your model accuracy. More details at https://docs.sglang.io/advanced_features/quantization.html#online-quantization."
            )

    @classmethod
    def get_min_capability(cls) -> int:
        return 70

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """
        Process weights after loading. If checkpoint is not serialized in MXFP4,
        perform online quantization.
        """
        if not self.is_checkpoint_mxfp4_serialized:
            assert layer.weight.dtype == torch.uint8
            assert layer.weight_scale.dtype == torch.uint8

        # Checkpoint already serialized in MXFP4, just freeze parameters
        layer.weight.requires_grad_(False)
        layer.weight_scale.requires_grad_(False)

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
