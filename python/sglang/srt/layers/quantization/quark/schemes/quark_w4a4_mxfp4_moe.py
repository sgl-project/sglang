# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING, Any

import torch

from sglang.srt.layers.moe import MoeRunner, MoeRunnerBackend, MoeRunnerConfig
from sglang.srt.layers.moe.utils import get_moe_weight_sizes
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.quantization.dequantization import (
    copy_missing_attrs,
    dequantize_fp8,
    dequantize_nvfp4,
)
from sglang.srt.layers.quantization.fp8 import Fp8Config, Fp8MoEMethod
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

        # Handle source-checkpoint -> MXFP4 requantization at load time. The
        # source may be NVFP4 (ModelOpt/Quark) or FP8 (block-quantized).
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
            elif isinstance(self.dequantization_config, Fp8Config):
                with_bias = extra_weight_attrs.pop("with_bias", False)
                self.with_bias = with_bias

                if self.dequantization_config.use_mxfp8:
                    raise NotImplementedError(
                        "use_mxfp8=True is not supported in Quark MXFP4 requantization."
                    )

                block_quant = self.dequantization_config.weight_block_size is not None

                if not block_quant:
                    raise NotImplementedError(
                        "Only block_quant=True is supported in Quark MXFP4 requantization, got block_quant=False."
                    )

                # `_fp8_loaded_numel` is used to trigger FP8 -> MXFP4 requantization once all weights are loaded.
                # `_fp8_materialized` is used to ensure only one thread materializes weights from meta device.
                layer._fp8_loaded_numel = 0
                layer._fp8_materialized = False
                layer._load_device = torch.get_default_device()
                layer._fp8_loading_lock = threading.Lock()

                # Custom weight loader handling FP8->MXFP4 conversion.
                fp8_to_mxfp4_weight_loader = self.get_online_fp8_to_mxfp4_weight_loader(
                    layer, original_weight_loader
                )

                extra_weight_attrs["weight_loader"] = fp8_to_mxfp4_weight_loader
                # Create FP8 MoE weight parameters on meta device to avoid device memory overhead during weight loading, as the resulting model uses MXFP4 using less device memory.
                # The weight loader handles progressive FP8 weight materialization on device.
                with torch.device("meta"):
                    Fp8MoEMethod.create_fp8_moe_weight_(
                        layer=layer,
                        num_experts=num_experts,
                        hidden_size=hidden_size,
                        intermediate_size_per_partition=intermediate_size_per_partition,
                        block_quant=block_quant,
                        quant_config=self.dequantization_config,
                        use_mxfp8=False,
                        is_checkpoint_fp8_serialized=True,
                        is_fp4_expert=False,
                        params_dtype=params_dtype,
                        with_bias=with_bias,
                        **extra_weight_attrs,
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

        if self.is_checkpoint_mxfp4_serialized:
            weight_loader = original_weight_loader
            weight_device = torch.get_default_device()
            weight_dtype = torch.uint8
        else:
            # Online quantization: use original dtype and meta device
            weight_loader = self.get_online_weight_loader(layer, original_weight_loader)
            weight_device = torch.device("meta")
            weight_dtype = params_dtype

        params_dtype = torch.uint8

        layer._load_device = torch.get_default_device()
        layer._w13_loaded_numel = 0
        layer._w2_loaded_numel = 0

        extra_weight_attrs["weight_loader"] = weight_loader

        # WEIGHTS
        w13_shape = (
            num_experts,
            2 * intermediate_size_per_partition,
            hidden_size // 2 if self.is_checkpoint_mxfp4_serialized else hidden_size,
        )
        w13_weight = torch.nn.Parameter(
            torch.empty(
                w13_shape,
                dtype=weight_dtype,
                device=weight_device,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)

        set_weight_attrs(w13_weight, extra_weight_attrs)

        w2_shape = (
            num_experts,
            hidden_size,
            (
                intermediate_size_per_partition // 2
                if self.is_checkpoint_mxfp4_serialized
                else intermediate_size_per_partition
            ),
        )
        w2_weight = torch.nn.Parameter(
            torch.empty(
                w2_shape,
                dtype=weight_dtype,
                device=weight_device,
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
        Wrap the original weight loader to perform online MXFP4 quantization for MoE layers.
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

            # Determine which weight parameter we're loading (w13 or w2)
            is_w13 = "w13" in weight_name
            is_w2 = "w2" in weight_name

            # Initialize weight on device if first load
            if is_w13 and layer._w13_loaded_numel == 0:
                layer.w13_weight = torch.nn.Parameter(
                    torch.empty_like(param.data, device=layer._load_device),
                    requires_grad=False,
                )
                param = layer.w13_weight
            elif is_w2 and layer._w2_loaded_numel == 0:
                layer.w2_weight = torch.nn.Parameter(
                    torch.empty_like(param.data, device=layer._load_device),
                    requires_grad=False,
                )
                param = layer.w2_weight

            # Move to device for faster quantization
            loaded_weight = loaded_weight.to(layer._load_device)

            if is_w13:
                param = layer.w13_weight
            elif is_w2:
                param = layer.w2_weight

            # In case TP>1, the weight loader logic uses narrow so we cannot directly rely on `param.shape` or `loaded_weight.shape`.
            copy_numel_counter = CopyNumelCounter()
            with copy_numel_counter:
                original_weight_loader(
                    param, loaded_weight, weight_name, shard_id, expert_id
                )

            if is_w13:
                layer._w13_loaded_numel += copy_numel_counter.copied_numel
                target_loaded_numel = layer.w13_weight.numel()
                current_loaded = layer._w13_loaded_numel
            elif is_w2:
                layer._w2_loaded_numel += copy_numel_counter.copied_numel
                target_loaded_numel = layer.w2_weight.numel()
                current_loaded = layer._w2_loaded_numel
            else:
                raise ValueError("Expected w13 or w2.")

            assert (
                current_loaded <= target_loaded_numel
            ), f"target_loaded_numel={target_loaded_numel}, current_loaded={current_loaded}"

            # Delay online quantization until all tensor shards (e.g. w1 and w3) are loaded, to avoid having to re-quantize later on.
            if is_w13 and layer._w13_loaded_numel == target_loaded_numel:
                self._quantize_w13_online(layer, dynamic_mxfp4_quant)
            elif is_w2 and layer._w2_loaded_numel == target_loaded_numel:
                self._quantize_w2_online(layer, dynamic_mxfp4_quant)

        return online_mxfp4_moe_weight_loader

    def get_online_fp8_to_mxfp4_weight_loader(self, layer, original_weight_loader):
        """
        Wrap the original weight loader to perform FP8 to MXFP4 requantization for MoE layers.

        This loader handles:
        1. Loading FP8 weights (w13_weight, w2_weight) and weight_scale_inv parameters
        2. Waiting for all experts to be loaded
        3. Dequantizing FP8 -> BF16 using weight_scale_inv
        4. Requantizing BF16 -> MXFP4
        """

        def online_fp8_to_mxfp4_moe_weight_loader(
            param: torch.nn.Parameter,
            loaded_weight: torch.Tensor,
            weight_name: str,
            shard_id: str,
            expert_id: int,
        ):
            is_w13_weight = "w13_weight" in weight_name and "scale" not in weight_name
            is_w2_weight = "w2_weight" in weight_name and "scale" not in weight_name
            is_w13_scale = "w13_weight_scale_inv" in weight_name
            is_w2_scale = "w2_weight_scale_inv" in weight_name

            # Sanity multi-threaded load check.
            assert torch.cuda.current_device() == layer._load_device.index

            # Materialize FP8 parameters on first load from meta device. Adds a small but manageable overhead compared to materializing one by one - but weights are loaded in order layer by layer so it is fine.
            with layer._fp8_loading_lock:

                if not layer._fp8_materialized:
                    # w13_weight
                    assert layer.w13_weight.device.type == "meta"
                    materialized = torch.nn.Parameter(
                        torch.empty_like(
                            layer.w13_weight.data, device=layer._load_device
                        ),
                        requires_grad=False,
                    )
                    copy_missing_attrs(layer.w13_weight, materialized)
                    layer.w13_weight = materialized

                    # w13_weight_scale_inv
                    materialized = torch.nn.Parameter(
                        torch.empty_like(
                            layer.w13_weight_scale_inv.data, device=layer._load_device
                        ),
                        requires_grad=False,
                    )
                    copy_missing_attrs(layer.w13_weight_scale_inv, materialized)
                    layer.w13_weight_scale_inv = materialized

                    # w2_weight
                    assert layer.w2_weight.device.type == "meta"
                    materialized = torch.nn.Parameter(
                        torch.empty_like(
                            layer.w2_weight.data, device=layer._load_device
                        ),
                        requires_grad=False,
                    )
                    copy_missing_attrs(layer.w2_weight, materialized)
                    layer.w2_weight = materialized

                    # w2_weight_scale_inv
                    assert layer.w2_weight_scale_inv.device.type == "meta"
                    materialized = torch.nn.Parameter(
                        torch.empty_like(
                            layer.w2_weight_scale_inv.data, device=layer._load_device
                        ),
                        requires_grad=False,
                    )
                    copy_missing_attrs(layer.w2_weight_scale_inv, materialized)
                    layer.w2_weight_scale_inv = materialized

                    # Mark as materialized to prevent other threads from doing it again.
                    layer._fp8_materialized = True

                if is_w13_weight:
                    param = layer.w13_weight
                elif is_w2_weight:
                    param = layer.w2_weight
                elif is_w13_scale:
                    param = layer.w13_weight_scale_inv
                elif is_w2_scale:
                    param = layer.w2_weight_scale_inv

            # Track how much data we are actually loading (`narrow` used in weight loader)
            copy_numel_counter = CopyNumelCounter()
            with copy_numel_counter:
                original_weight_loader(
                    param, loaded_weight, weight_name, shard_id, expert_id
                )

            with layer._fp8_loading_lock:
                layer._fp8_loaded_numel += copy_numel_counter.copied_numel

                total_target_numel = (
                    layer.w13_weight.numel()
                    + layer.w2_weight.numel()
                    + layer.w13_weight_scale_inv.numel()
                    + layer.w2_weight_scale_inv.numel()
                )

                # Sanity check
                assert layer._fp8_loaded_numel <= total_target_numel

                # Perform dequantization and requantization only when all data is loaded AND no other threads are still loading.
                if layer._fp8_loaded_numel == total_target_numel:
                    if dynamic_mxfp4_quant is None:
                        raise NotImplementedError(
                            "MXFP4 quantization for MoE is only supported on AMD GPUs."
                        )

                    assert layer.w13_weight.device.type == "cuda"
                    assert layer.w13_weight_scale_inv.device.type == "cuda"
                    assert layer.w13_weight.dtype != torch.uint8

                    # Dequantize and requantize w13
                    w13_bf16 = dequantize_fp8(
                        layer.w13_weight,
                        layer.w13_weight_scale_inv,
                        block_size=self.dequantization_config.weight_block_size,
                    )

                    qw13_weight_list = []
                    w13_weight_scale_list = []
                    for expert_idx in range(w13_bf16.shape[0]):
                        # NOTE: dynamic_mxfp4_quant does not accept 3D inputs.
                        qweight, weight_scale = dynamic_mxfp4_quant(
                            w13_bf16[expert_idx]
                        )
                        qw13_weight_list.append(qweight)
                        w13_weight_scale_list.append(weight_scale)

                    qw13_weight = torch.stack(qw13_weight_list)
                    w13_weight_scale = torch.stack(w13_weight_scale_list)

                    # Dequantize and requantize w2
                    w2_bf16 = dequantize_fp8(
                        layer.w2_weight,
                        layer.w2_weight_scale_inv,
                        block_size=self.dequantization_config.weight_block_size,
                    )

                    qw2_weight_list = []
                    w2_weight_scale_list = []
                    for expert_idx in range(w2_bf16.shape[0]):
                        # NOTE: dynamic_mxfp4_quant does not accept 3D inputs.
                        qweight, weight_scale = dynamic_mxfp4_quant(w2_bf16[expert_idx])
                        qw2_weight_list.append(qweight)
                        w2_weight_scale_list.append(weight_scale)

                    qw2_weight = torch.stack(qw2_weight_list)
                    w2_weight_scale = torch.stack(w2_weight_scale_list)

                    # Replace FP8 parameters with MXFP4 parameters
                    layer.w13_weight = torch.nn.Parameter(
                        qw13_weight, requires_grad=False
                    )
                    layer.w13_weight_scale = torch.nn.Parameter(
                        w13_weight_scale, requires_grad=False
                    )
                    layer.w2_weight = torch.nn.Parameter(
                        qw2_weight, requires_grad=False
                    )
                    layer.w2_weight_scale = torch.nn.Parameter(
                        w2_weight_scale, requires_grad=False
                    )

                    # Clean up FP8 parameters and tracking attributes
                    del layer.w13_weight_scale_inv
                    del layer.w2_weight_scale_inv
                    del layer._fp8_materialized
                    del layer._load_device
                    del layer._fp8_loading_lock

        return online_fp8_to_mxfp4_moe_weight_loader

    def _quantize_w13_online(self, layer, dynamic_mxfp4_quant):
        qw13_weight = torch.empty(
            layer.w13_weight.shape[0],
            layer.w13_weight.shape[1],
            layer.w13_weight.shape[2] // 2,
            dtype=torch.uint8,
            device=layer._load_device,
        )

        for expert in range(layer.w13_weight.shape[0]):
            qweight, weight_scale = dynamic_mxfp4_quant(layer.w13_weight.data[expert])
            assert qw13_weight[expert].shape == qweight.shape
            assert qw13_weight[expert].dtype == qweight.dtype
            qw13_weight[expert] = qweight

            assert layer.w13_weight_scale[expert].shape == weight_scale.shape
            assert layer.w13_weight_scale[expert].dtype == weight_scale.dtype
            layer.w13_weight_scale[expert] = weight_scale

        layer.w13_weight = torch.nn.Parameter(qw13_weight, requires_grad=False)

    def _quantize_w2_online(self, layer, dynamic_mxfp4_quant):
        qw2_weight = torch.empty(
            layer.w2_weight.shape[0],
            layer.w2_weight.shape[1],
            layer.w2_weight.shape[2] // 2,
            dtype=torch.uint8,
            device=layer._load_device,
        )

        for expert in range(layer.w2_weight.shape[0]):
            qweight, weight_scale = dynamic_mxfp4_quant(layer.w2_weight.data[expert])
            qw2_weight[expert] = qweight
            layer.w2_weight_scale[expert] = weight_scale

        layer.w2_weight = torch.nn.Parameter(qw2_weight, requires_grad=False)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if (
            not self.is_checkpoint_mxfp4_serialized
            or self.dequantization_config is not None
        ):
            # Quantization already happened during weight loading.
            # This covers both:
            # - Online quantization from BF16/FP16 -> MXFP4
            # - Requantization from FP8 -> MXFP4
            assert layer.w13_weight.dtype == torch.uint8
            assert layer.w2_weight.dtype == torch.uint8
            assert layer.w13_weight_scale.dtype == torch.uint8
            assert layer.w2_weight_scale.dtype == torch.uint8

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
