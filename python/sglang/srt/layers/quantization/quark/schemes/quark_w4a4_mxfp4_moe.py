# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING, Any

import torch

from sglang.srt.layers.moe import MoeRunner, MoeRunnerBackend, MoeRunnerConfig
from sglang.srt.layers.moe.utils import get_moe_weight_sizes
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.quantization.dequantization import dequantize_nvfp4
from sglang.srt.layers.quantization.online_quantization import CopyNumelCounter
from sglang.srt.layers.quantization.quark.schemes import QuarkMoEScheme
from sglang.srt.layers.quantization.quark.utils import Nvfp4SourceConfig
from sglang.srt.utils import (
    get_bool_env_var,
    is_gfx95_supported,
    is_hip,
    set_weight_attrs,
)
from sglang.srt.utils.common import mxfp_supported

NVFP4_BLOCK_SIZE = 16

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
        dequantization_config: QuantizationConfig | None = None,
    ):
        self.weight_quant = weight_config
        self.input_quant = input_config
        self.is_checkpoint_mxfp4_serialized = is_checkpoint_mxfp4_serialized
        self.dequantization_config = dequantization_config

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

        original_weight_loader = extra_weight_attrs.get("weight_loader")

        # Handle source-checkpoint -> MXFP4 requantization at load time.
        if self.dequantization_config is not None:
            if isinstance(self.dequantization_config, Nvfp4SourceConfig):
                self._create_weights_from_nvfp4_moe(
                    layer=layer,
                    num_experts=num_experts,
                    hidden_size=hidden_size,
                    intermediate_size_per_partition=intermediate_size_per_partition,
                    original_weight_loader=original_weight_loader,
                    extra_weight_attrs=extra_weight_attrs,
                )
            else:
                raise NotImplementedError(
                    f"Requantization in QuarkW4A4MXFp4MoE from {self.dequantization_config.__class__.__name__} is not supported."
                )
            return

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

    def _create_weights_from_nvfp4_moe(
        self,
        *,
        layer,
        num_experts,
        hidden_size,
        intermediate_size_per_partition,
        original_weight_loader,
        extra_weight_attrs,
    ):
        layer._nvfp4_loaded_numel = 0
        layer._load_device = torch.device(f"cuda:{torch.cuda.current_device()}")
        layer._nvfp4_loading_lock = threading.Lock()

        nvfp4_loader = self.get_online_nvfp4_to_mxfp4_weight_loader(
            layer, original_weight_loader
        )
        extra_weight_attrs["weight_loader"] = nvfp4_loader

        def _param(shape, dtype):
            return torch.nn.Parameter(
                torch.empty(*shape, dtype=dtype, device=layer._load_device),
                requires_grad=False,
            )

        params = {
            "w13_weight": _param(
                (num_experts, 2 * intermediate_size_per_partition, hidden_size // 2),
                torch.uint8,
            ),
            "w2_weight": _param(
                (num_experts, hidden_size, intermediate_size_per_partition // 2),
                torch.uint8,
            ),
            "w13_weight_scale": _param(
                (
                    num_experts,
                    2 * intermediate_size_per_partition,
                    hidden_size // NVFP4_BLOCK_SIZE,
                ),
                torch.float8_e4m3fn,
            ),
            "w2_weight_scale": _param(
                (
                    num_experts,
                    hidden_size,
                    intermediate_size_per_partition // NVFP4_BLOCK_SIZE,
                ),
                torch.float8_e4m3fn,
            ),
        }
        # w13 fuses gate(w1)+up(w3): FusedMoE stores a per-tensor scale for
        # each at param[expert][0|1], so shape is [E, 2]. w2 (down) is single.
        params["w13_weight_scale_2"] = _param((num_experts, 2), torch.float32)
        params["w2_weight_scale_2"] = _param((num_experts,), torch.float32)

        from sglang.srt.layers.moe.fused_moe_triton import FusedMoeWeightScaleSupported

        # FusedMoE's scale loader dispatches on param.quant_method. NVFP4
        # per-block weight_scale -> GROUP; per-tensor weight_scale_2 -> TENSOR.
        # (The packed weight tensors skip that branch, name has no "scale".)
        for name, param in params.items():
            layer.register_parameter(name, param)
            attrs = dict(extra_weight_attrs)
            if name.endswith("weight_scale_2"):
                attrs["quant_method"] = FusedMoeWeightScaleSupported.TENSOR.value
            elif name.endswith("weight_scale"):
                attrs["quant_method"] = FusedMoeWeightScaleSupported.GROUP.value
            set_weight_attrs(param, attrs)

        # NVFP4 checkpoints carry per-expert `input_scale` (activation scale)
        # per projection. MXFP4 uses dynamic activation quant; discard them but
        # register slots so upstream MoE loaders that route w1/w3.input_scale ->
        # w13_input_scale find a target. No-op loader absorbs any call shape.
        def _discard_loader(param, loaded_weight, weight_name, shard_id, expert_id):
            pass

        w13_input_scale = torch.nn.Parameter(
            torch.empty(num_experts, dtype=torch.float32, device=layer._load_device),
            requires_grad=False,
        )
        w2_input_scale = torch.nn.Parameter(
            torch.empty(num_experts, dtype=torch.float32, device=layer._load_device),
            requires_grad=False,
        )
        layer.register_parameter("w13_input_scale", w13_input_scale)
        layer.register_parameter("w2_input_scale", w2_input_scale)
        set_weight_attrs(
            w13_input_scale, {**extra_weight_attrs, "weight_loader": _discard_loader}
        )
        set_weight_attrs(
            w2_input_scale, {**extra_weight_attrs, "weight_loader": _discard_loader}
        )

    def get_online_nvfp4_to_mxfp4_weight_loader(self, layer, original_weight_loader):
        """NVFP4 MoE loader: expert-wise dequant+requant once all source bytes
        are in place."""
        bulk_names = ["w13_weight", "w2_weight", "w13_weight_scale", "w2_weight_scale"]
        scale2_names = ["w13_weight_scale_2", "w2_weight_scale_2"]

        def loader(param, loaded_weight, weight_name, shard_id, expert_id):
            is_scale_2 = "weight_scale_2" in weight_name
            is_scale = ("weight_scale" in weight_name) and not is_scale_2
            is_w13 = "w13" in weight_name
            assert torch.cuda.current_device() == layer._load_device.index

            with layer._nvfp4_loading_lock:
                if is_scale_2:
                    name = "w13_weight_scale_2" if is_w13 else "w2_weight_scale_2"
                elif is_scale:
                    name = "w13_weight_scale" if is_w13 else "w2_weight_scale"
                else:
                    name = "w13_weight" if is_w13 else "w2_weight"
                param = getattr(layer, name)

            counter = CopyNumelCounter()
            with counter:
                original_weight_loader(
                    param, loaded_weight, weight_name, shard_id, expert_id
                )

            with layer._nvfp4_loading_lock:
                layer._nvfp4_loaded_numel += counter.copied_numel
                total = sum(
                    getattr(layer, name).numel() for name in bulk_names + scale2_names
                )
                if layer._nvfp4_loaded_numel == total:
                    self._requantize_nvfp4_to_mxfp4(layer, "w13")
                    self._requantize_nvfp4_to_mxfp4(layer, "w2")
                    for name in scale2_names:
                        delattr(layer, name)
                    del layer._load_device

        return loader

    def _requantize_nvfp4_to_mxfp4(self, layer, prefix):
        # dynamic_mxfp4_quant is 2-D only; loop over experts.
        packed_weight = getattr(layer, f"{prefix}_weight")
        weight_scale = getattr(layer, f"{prefix}_weight_scale")
        weight_scale_2 = getattr(layer, f"{prefix}_weight_scale_2")

        # Zero-pad the intermediate dim up to the AITER MoE alignment before the
        # MXFP4 requant. (process_weights_after_loading's e8m0_shuffle pads column
        # count up to a multiple of 8 which could cause weight K-blocks to be
        # miscalculated, leading to scale misalignment and garbage output
        inter_pad = 0
        if _use_aiter:
            if prefix == "w2":  # [E, hidden, inter // 2]
                real_inter = packed_weight.shape[-1] * 2
            else:  # w13
                real_inter = packed_weight.shape[1] // 2
            _, w2_down_dim, _ = get_moe_weight_sizes(
                real_inter, is_concat=True, is_packed=True, is_aiter_moe=True
            )
            inter_pad = max(0, w2_down_dim * 2 - real_inter)

        quantized_weights, quantized_scales = [], []
        for expert_idx in range(packed_weight.shape[0]):
            if prefix == "w13":
                # weight_scale_2[expert_idx] = [gate_scale, up_scale]; the fused
                # weight is [gate_rows; up_rows] so expand each scalar over its
                # half as a per-row [2I, 1] multiplier.
                half = packed_weight[expert_idx].shape[0] // 2
                expert_scale_2 = torch.cat(
                    [
                        weight_scale_2[expert_idx, 0].repeat(half),
                        weight_scale_2[expert_idx, 1].repeat(half),
                    ]
                ).view(-1, 1)
            else:  # w2: single per-expert per-tensor scalar
                expert_scale_2 = weight_scale_2[expert_idx]
            dequantized_weight = dequantize_nvfp4(
                packed_weight[expert_idx],
                weight_scale[expert_idx],
                expert_scale_2,
                out_dtype=torch.float32,
            )
            if inter_pad:
                if prefix == "w2":
                    # Pad the trailing K dim with zeros.
                    dequantized_weight = torch.nn.functional.pad(
                        dequantized_weight, (0, inter_pad)
                    )
                else:
                    # w13: pad each of the gate/up halves' rows so the [gate; up]
                    # split properly
                    half_rows = dequantized_weight.shape[0] // 2
                    gate = torch.nn.functional.pad(
                        dequantized_weight[:half_rows], (0, 0, 0, inter_pad)
                    )
                    up = torch.nn.functional.pad(
                        dequantized_weight[half_rows:], (0, 0, 0, inter_pad)
                    )
                    dequantized_weight = torch.cat([gate, up], dim=0)
            requantized_weight, requantized_scale = dynamic_mxfp4_quant(
                dequantized_weight
            )
            quantized_weights.append(requantized_weight)
            quantized_scales.append(requantized_scale)
        setattr(
            layer,
            f"{prefix}_weight",
            torch.nn.Parameter(torch.stack(quantized_weights), requires_grad=False),
        )
        setattr(
            layer,
            f"{prefix}_weight_scale",
            torch.nn.Parameter(torch.stack(quantized_scales), requires_grad=False),
        )

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
