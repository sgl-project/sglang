# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import torch

from sglang.srt.layers.moe import MoeRunner, MoeRunnerBackend, MoeRunnerConfig
from sglang.srt.layers.moe.utils import get_moe_weight_sizes
from sglang.srt.layers.quantization.quark.schemes import QuarkMoEScheme
from sglang.srt.utils import (
    get_bool_env_var,
    is_gfx95_supported,
    is_hip,
    set_weight_attrs,
)
from sglang.srt.utils.common import mxfp_supported

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher import (
        CombineInput,
        StandardDispatchOutput,
    )

logger = logging.getLogger(__name__)

_is_shuffle_moe_mxfp4 = is_gfx95_supported()

__all__ = ["QuarkW4A4MXFp4MoE"]

_is_hip = is_hip()
_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip
if _use_aiter:
    from aiter.ops.shuffle import shuffle_weight
    from aiter.utility.fp4_utils import e8m0_shuffle

if _is_hip:
    from aiter.ops.triton.quant import dynamic_mxfp4_quant
else:
    dynamic_mxfp4_quant = None

OCP_MX_BLOCK_SIZE = 32


class QuarkW4A4MXFp4MoE(QuarkMoEScheme):

    def __init__(
        self,
        weight_config: dict[str, Any],
        input_config: dict[str, Any],
        is_checkpoint_mxfp4_serialized: bool = True,
    ):
        self.weight_quant = weight_config
        self.input_quant = input_config
        self.is_checkpoint_mxfp4_serialized = is_checkpoint_mxfp4_serialized

        weight_qscheme = self.weight_quant.get("qscheme")
        input_qscheme = self.input_quant.get("qscheme")
        if not (weight_qscheme == "per_group" and input_qscheme == "per_group"):
            raise ValueError(
                "For MX(FP4) Fused MoE layers, only per-group scales "
                "for weights and activations are supported. Found "
                f"{weight_qscheme}, {input_qscheme}"
            )  # noqa E501

        self.static_input_scales = not self.input_quant.get("is_dynamic")
        self.with_bias = False

        if not self.is_checkpoint_mxfp4_serialized:
            if not mxfp_supported():
                raise NotImplementedError(
                    "Online MXFP4 quantization for MoE layers requires an AMD ROCm "
                    "device with FP4 hardware support (gfx95x, e.g. MI355x)."
                )
            logger.info_once(
                "Using online MXFP4 quantization for MoE layers from a higher precision checkpoint. "
                "Beware that this optimization may degrade prediction quality - please validate your model accuracy. "
                "More details at https://docs.sglang.io/advanced_features/quantization.html#online-quantization."
            )

    @classmethod
    def get_min_capability(cls) -> int:
        return 70

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):

        from sglang.srt.layers.moe.fused_moe_triton import FusedMoeWeightScaleSupported

        w13_up_dim, w2_down_dim, weight_padded = get_moe_weight_sizes(
            intermediate_size_per_partition,
            is_aiter_moe=_use_aiter,
            is_concat=True,
            is_packed=True,
        )

        # Add the quantization method used (per tensor/grouped/channel)
        # to ensure the weight scales are loaded in properly
        extra_weight_attrs.update(
            {
                "quant_method": FusedMoeWeightScaleSupported.BLOCK.value,
                "weight_padded": weight_padded,
            },
        )

        params_dtype = torch.uint8

        original_weight_loader = extra_weight_attrs.get("weight_loader")

        if self.is_checkpoint_mxfp4_serialized:
            weight_loader = original_weight_loader
        else:
            weight_loader = self.get_online_weight_loader(layer, original_weight_loader)

        extra_weight_attrs["weight_loader"] = weight_loader

        # WEIGHTS — always uint8 (packed mxfp4), always on device
        w13_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                w13_up_dim,
                hidden_size // 2,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w2_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                w2_down_dim,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        # WEIGHT_SCALES
        extra_weight_attrs["weight_loader"] = original_weight_loader

        w13_weight_scale = torch.nn.Parameter(
            torch.ones(
                num_experts,
                w13_up_dim,
                hidden_size // OCP_MX_BLOCK_SIZE,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )

        # 1. w2 scale is floor division of inter_dim by blockscale.
        # 2. w2 scale needs to scale up just as w2.
        # We combine 1. and 2. to keep the integer precision.
        w2_weight_scale = torch.nn.Parameter(
            torch.ones(
                num_experts,
                hidden_size,
                (w2_down_dim * 2) // OCP_MX_BLOCK_SIZE,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        set_weight_attrs(w13_weight_scale, extra_weight_attrs)
        set_weight_attrs(w2_weight_scale, extra_weight_attrs)
        layer.register_parameter("w13_weight_scale", w13_weight_scale)
        layer.register_parameter("w2_weight_scale", w2_weight_scale)

    def get_online_weight_loader(self, layer, original_weight_loader):
        """
        Wrap the original weight loader to perform online MXFP4 quantization.
        """

        def online_mxfp4_moe_weight_loader(
            param: torch.nn.Parameter,
            loaded_weight: torch.Tensor,
            weight_name: str,
            shard_id: str,
            expert_id: int,
        ):
            if dynamic_mxfp4_quant is None:
                raise NotImplementedError(
                    "Online MXFP4 quantization for MoE is only supported on AMD GPUs."
                )

            # Materialize on device the loaded weight.
            loaded_weight = loaded_weight.to(param.device)

            # Quantize the high-precision shard loaded_weight to MXFP4.
            qweight, weight_scale = dynamic_mxfp4_quant(loaded_weight)

            original_weight_loader(param, qweight, weight_name, shard_id, expert_id)

            if "w13" in weight_name:
                scale_param = layer.w13_weight_scale
                scale_weight_name = "w13_weight_scale"
            else:
                # w2.
                scale_param = layer.w2_weight_scale
                scale_weight_name = "w2_weight_scale"

            scale_param.weight_loader(
                scale_param, weight_scale, scale_weight_name, shard_id, expert_id
            )

        return online_mxfp4_moe_weight_loader

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # FlyDSL MegaMoE (AMD): consume RAW fp4 weights with its own
        # shuffle_weight(16,16) + e8m0_shuffle, so build BEFORE the aiter shuffle
        # below overwrites them. a4w4 = fp4 activations + fp4 weights.
        from sglang.srt.environ import envs
        from sglang.srt.layers.moe.utils import get_moe_a2a_backend

        if get_moe_a2a_backend().is_megamoe() and envs.SGLANG_AMD_USE_FLYDSL_MEGA_MOE.get():
            from sglang.srt.layers.moe.mega_moe import build_mega_moe_experts_weights

            layer._mega_quant = "a4w4"
            build_mega_moe_experts_weights(layer)
            return

        # Pre-shuffle weight scales
        s0, s1, _ = layer.w13_weight_scale.shape
        w13_weight_scale = layer.w13_weight_scale.view(s0 * s1, -1)
        w13_weight_scale = e8m0_shuffle(w13_weight_scale)
        layer.w13_weight_scale.data = w13_weight_scale.view(s0, s1, -1)

        s0, s1, _ = layer.w2_weight_scale.shape
        w2_weight_scale = layer.w2_weight_scale.view(s0 * s1, -1)
        w2_weight_scale = e8m0_shuffle(w2_weight_scale)
        layer.w2_weight_scale.data = w2_weight_scale.view(s0, s1, -1)

        # Pre-shuffle weight
        if _is_shuffle_moe_mxfp4:
            layer.w13_weight.data = shuffle_weight(
                layer.w13_weight.contiguous(), (16, 16)
            )
            layer.w2_weight.data = shuffle_weight(
                layer.w2_weight.contiguous(), (16, 16)
            )
            layer.w13_weight.is_shuffled = True
            layer.w2_weight.is_shuffled = True

        if hasattr(layer, "dispatcher"):
            # Weights are stored as torch.uint8 but semantically MXFP4
            layer.dispatcher.set_quant_config({"weight_dtype": torch.float4_e2m1fn_x2})

    def create_moe_runner(
        self, layer: torch.nn.Module, moe_runner_config: MoeRunnerConfig
    ):
        from sglang.srt.layers.moe.utils import (
            get_moe_a2a_backend,
            get_moe_runner_backend,
        )

        self.moe_runner_config = moe_runner_config
        moe_runner_backend = get_moe_runner_backend()
        if moe_runner_backend.is_auto() and get_moe_a2a_backend().supports_aiter():
            moe_runner_backend = MoeRunnerBackend.AITER

        if moe_runner_backend.is_aiter():
            self.runner = MoeRunner(moe_runner_backend, moe_runner_config)
        else:
            # TODO(cwan): refactor other backends
            pass

    def apply_weights(
        self,
        layer: torch.nn.Module,
        dispatch_output: StandardDispatchOutput,
    ) -> CombineInput:
        from sglang.srt.layers.moe.moe_runner.aiter import (
            AiterMoeQuantInfo,
            AiterQuantType,
        )

        if hasattr(torch, "float4_e2m1fn_x2"):
            w13_weight = layer.w13_weight.view(torch.float4_e2m1fn_x2)
            w2_weight = layer.w2_weight.view(torch.float4_e2m1fn_x2)
        else:
            w13_weight = layer.w13_weight
            w2_weight = layer.w2_weight

        if hasattr(layer.w13_weight, "is_shuffled"):
            w13_weight.is_shuffled = True
            w2_weight.is_shuffled = True

        quant_info = AiterMoeQuantInfo(
            w13_weight=w13_weight,
            w2_weight=w2_weight,
            quant_type=AiterQuantType.PER_1X32,
            w13_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            expert_mask=layer.dispatcher.expert_mask_gpu,
        )
        return self.runner.run(dispatch_output, quant_info)
