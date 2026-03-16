import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import torch
from tqdm import tqdm
from tqdm.std import EMA

from sglang.srt.distributed import get_tensor_model_parallel_rank
from sglang.srt.layers.int4fp8_utils import (
    pack_int4_to_int32,
    quantize_fp8_scale_tensorwise,
    quantize_int4_scale_columnwise,
)
from sglang.srt.layers.moe import MoeRunnerConfig
from sglang.srt.layers.quantization.base_config import (
    FusedMoEMethodBase,
    QuantizationConfig,
    QuantizeMethodBase,
)
from sglang.srt.layers.quantization.fp8 import Fp8LinearMethod
from sglang.srt.utils import BAR_FORMAT, is_hip, set_weight_attrs

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher import DispatchOutput

_is_hip = is_hip()


if _is_hip:
    from aiter import ActivationType, QuantType
    from aiter.fused_moe import fused_moe
    from aiter.ops.shuffle import shuffle_weight

    ON_GFX950 = "gfx950" in torch.cuda.get_device_properties("cuda").gcnArchName

logger = logging.getLogger(__name__)


def tqdm_reset_no_print(tqdm_bar: tqdm, total=None):
    tqdm_bar.n = 0
    if total is not None:
        tqdm_bar.total = total
    if tqdm_bar.disable:
        return
    tqdm_bar.last_print_n = 0
    tqdm_bar.last_print_t = tqdm_bar.start_t = tqdm_bar._time()
    tqdm_bar._ema_dn = EMA(tqdm_bar.smoothing)
    tqdm_bar._ema_dt = EMA(tqdm_bar.smoothing)
    tqdm_bar._ema_miniters = EMA(tqdm_bar.smoothing)


class QuarkInt4Fp8Config(QuantizationConfig):
    """Config class for Quark Quantization.

    - Weight: static, per-channel, symmetric
    - Activation: dynamic, per-token, symmetric
    """

    def __init__(
        self,
        is_checkpoint_fp8_serialized: bool = False,
        activation_scheme: str = "dynamic",
    ):
        self.is_checkpoint_fp8_serialized = is_checkpoint_fp8_serialized
        self.activation_scheme = activation_scheme

        if activation_scheme != "dynamic":
            raise NotImplementedError(
                "QuarkInt4Fp8Config only supports activation_scheme='dynamic'."
            )

        self.weight_block_size = None

        self.num_quant_layers = 0

        tp_rank = get_tensor_model_parallel_rank()

        # The weight iterator already has a progress bar on rank=0, account for that.
        position = 1 + tqdm._get_free_pos()
        self.online_quant_progress_bar = tqdm(
            total=0,
            desc=f"Online quark_int4fp8_moe quantization on rank={tp_rank}",
            position=position,
            bar_format=BAR_FORMAT,
            mininterval=2.0,
        )

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.float16, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 70

    @classmethod
    def get_name(self) -> str:
        return "quark_int4fp8_moe"

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        return []

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "QuarkInt4Fp8Config":
        return cls()

    def get_quant_method(
        self,
        layer: torch.nn.Module,
        prefix: str,
    ) -> Optional["QuantizeMethodBase"]:
        # TODO: fix circular imports issues in sglang forcing us to import here instead of at
        # the top of file.
        from sglang.srt.layers.linear import LinearBase
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoE

        if isinstance(layer, LinearBase):
            return Fp8LinearMethod(self)
        elif isinstance(layer, FusedMoE):
            return QuarkInt4Fp8MoEMethod(self)

        return None

    def get_scaled_act_names(self) -> List[str]:
        return []


class QuarkInt4Fp8MoEMethod(FusedMoEMethodBase):
    """MoE method for INT4FP8.

    Supports loading BF16/FP16 checkpoints, quantizing down to INT4, and dequantizing to FP8 during inference.

    Args:
        quant_config: The quantization config.
    """

    def __init__(self, quant_config):
        self.quant_config = quant_config

        self.online_quant_progress_bar = self.quant_config.online_quant_progress_bar

        self.tp_rank = get_tensor_model_parallel_rank()

        if not _is_hip:
            raise NotImplementedError(
                "The quark_int4fp8_moe online quantization scheme is only supported on AMD GPUs."
            )

    def get_weight_loader(self, layer, original_weight_loader):
        def online_int4_fp8_weight_loader(
            param: torch.nn.Parameter,
            loaded_weight: torch.Tensor,
            weight_name: str,
            shard_id: str,
            expert_id: int,
        ):
            if shard_id in ["w1", "w3"]:
                shard_size = self.w13_shard_size
            else:
                shard_size = self.w2_shard_size

            original_use_presharded_weights = layer.use_presharded_weights

            if not layer.use_presharded_weights:
                # In case the model is not pre-sharded (most checkpoints on HF Hub),
                # we shard the model here in order to run online quantization on
                # already sharded weights.
                # Some models as `lmzheng/grok-1` are already be sharded.
                layer.use_presharded_weights = True

                if shard_id in ["w1", "w3"]:
                    shard_dim = 0
                    loaded_weight = loaded_weight.narrow(
                        shard_dim, shard_size * self.tp_rank, shard_size
                    )
                else:
                    shard_dim = 1
                    loaded_weight = loaded_weight.narrow(
                        shard_dim, shard_size * self.tp_rank, shard_size
                    )

            # We want to run online quantization on-device for speed purposes.
            loaded_weight = loaded_weight.to(param.device)

            _, fp8_scale = quantize_fp8_scale_tensorwise(loaded_weight)

            int4_w, int4_scale = quantize_int4_scale_columnwise(loaded_weight)

            int4_w = pack_int4_to_int32(int4_w)
            int4_scale /= fp8_scale

            if shard_id in ["w1", "w3"]:
                if shard_id == "w1":
                    shard_slice = slice(0, shard_size)
                    idx = 0
                else:
                    shard_slice = slice(shard_size, 2 * shard_size)
                    idx = 1

                assert param[expert_id][shard_slice].dtype == int4_w.dtype

                assert (
                    layer.w13_int4_scale[expert_id][shard_slice].shape
                    == int4_scale.shape
                )
                assert (
                    layer.w13_int4_scale[expert_id][shard_slice].dtype
                    == int4_scale.dtype
                )

                layer.w13_int4_scale[expert_id][shard_slice].copy_(int4_scale)

                assert layer.w13_fp8_scale[expert_id][idx].shape == fp8_scale.shape
                assert layer.w13_fp8_scale[expert_id][idx].dtype == fp8_scale.dtype

                layer.w13_fp8_scale[expert_id][idx].copy_(fp8_scale)
            else:
                assert param[expert_id].dtype == int4_w.dtype
                assert param[expert_id].shape == int4_w.shape

                assert layer.w2_int4_scale[expert_id].shape == int4_scale.shape
                assert layer.w2_int4_scale[expert_id].dtype == int4_scale.dtype

                layer.w2_int4_scale[expert_id].copy_(int4_scale)

                assert layer.w2_fp8_scale[expert_id].shape == fp8_scale.shape
                assert layer.w2_fp8_scale[expert_id].dtype == fp8_scale.dtype

                layer.w2_fp8_scale[expert_id].copy_(fp8_scale)

            original_weight_loader(
                param,
                int4_w,
                shard_id=shard_id,
                weight_name=weight_name,
                expert_id=expert_id,
            )

            # Reset `use_presharded_weights` as the same layer may load several different weights.
            layer.use_presharded_weights = original_use_presharded_weights

            self.online_quant_progress_bar.update(1)

        return online_int4_fp8_weight_loader

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        # TODO: fix circular imports issues in sglang forcing us to import here instead of at
        # the top of file.
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoeWeightScaleSupported

        # print("intermediate_size_per_partition", intermediate_size_per_partition)
        # fused moe logic already hands TP logic.
        self.w13_shard_size = intermediate_size_per_partition
        self.w2_shard_size = intermediate_size_per_partition

        assert "weight_loader" in extra_weight_attrs
        original_weight_loader = extra_weight_attrs.get("weight_loader")

        online_int4fp8_weight_loader = self.get_weight_loader(
            layer, original_weight_loader
        )
        extra_weight_attrs["weight_loader"] = online_int4fp8_weight_loader

        params_dtype = torch.uint32
        # WEIGHTS
        # INT4 MoE weight - INT32 packed
        w13_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size // 8,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        w2_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                intermediate_size_per_partition // 8,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        # Allocate 2 scales for w1 and w3 respectively.
        # They will be combined to a single scale after weight loading.
        w13_fp8_scale = torch.nn.Parameter(
            torch.ones(num_experts, 2, dtype=torch.float32), requires_grad=False
        )
        w2_fp8_scale = torch.nn.Parameter(
            torch.ones(num_experts, dtype=torch.float32), requires_grad=False
        )
        layer.register_parameter("w13_fp8_scale", w13_fp8_scale)
        layer.register_parameter("w2_fp8_scale", w2_fp8_scale)

        if _is_hip:
            w13_int4_scale = torch.nn.Parameter(
                torch.ones(
                    num_experts,
                    2 * intermediate_size_per_partition,
                    dtype=torch.float32,
                ),
                requires_grad=False,
            )
            w2_int4_scale = torch.nn.Parameter(
                torch.ones(num_experts, hidden_size, dtype=torch.float32),
                requires_grad=False,
            )
            layer.register_parameter("w13_int4_scale", w13_int4_scale)
            layer.register_parameter("w2_int4_scale", w2_int4_scale)

        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.TENSOR.value}
        )

        set_weight_attrs(w13_fp8_scale, extra_weight_attrs)
        set_weight_attrs(w2_fp8_scale, extra_weight_attrs)

        # Add the quantization method used (per tensor/grouped/channel)
        # to ensure the weight scales are loaded in properly
        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.CHANNEL.value}
        )

        set_weight_attrs(w13_int4_scale, extra_weight_attrs)
        set_weight_attrs(w2_int4_scale, extra_weight_attrs)

        w13_input_scale = None
        layer.register_parameter("w13_input_scale", w13_input_scale)

        w2_input_scale = None
        layer.register_parameter("w2_input_scale", w2_input_scale)

        # Loading from the checkpoint w1, w2, w3 times the number of experts.
        total = self.online_quant_progress_bar.total + num_experts * 3
        tqdm_reset_no_print(self.online_quant_progress_bar, total=total)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if _is_hip and not ON_GFX950:
            # CDNA3 does not support OCP FP8E4M3FN, but uses FP8E4M3FNUZ.
            # CDNA4 supports OCP FP8E4M3FN.
            layer.w13_int4_scale *= 0.5
            layer.w2_int4_scale *= 0.5

            layer.w13_fp8_scale *= 2.0
            layer.w2_fp8_scale *= 2.0

        # TODO: and use_aiter_moe: add after triton kernel added
        # INT4-FP8 (INT4 MoE Weight, FP8 Compute)
        # Weight Permutation
        layer.w13_weight = torch.nn.Parameter(
            shuffle_weight(layer.w13_weight.data, (16, 16)),
            requires_grad=False,
        )
        torch.cuda.empty_cache()
        layer.w2_weight = torch.nn.Parameter(
            shuffle_weight(layer.w2_weight.data, (16, 16)),
            requires_grad=False,
        )
        torch.cuda.empty_cache()

        # INT4-FP8 : offset INT4 w13_int4_scale to single w13_fp8_scale
        # Fp8 moe kernel needs single fp8 w13_fp8_scale for w13 per expert.
        # We won't do requant each expert's fp8 weight (not direct available),
        # instead we adjust half of INT4 w13_int4_scale numbers
        assert layer.w13_fp8_scale is not None
        shard_size = layer.intermediate_size_per_partition
        max_w13_scales = layer.w13_fp8_scale.max(dim=1).values
        for expert_id in range(layer.num_experts):
            start = 0
            max_w13_scale_fp8 = max_w13_scales[expert_id]
            for shard_id in range(2):
                if layer.w13_fp8_scale[expert_id][shard_id] != max_w13_scale_fp8:
                    int4_rescale = (
                        layer.w13_fp8_scale[expert_id][shard_id] / max_w13_scale_fp8
                    )
                    layer.w13_int4_scale[expert_id][
                        start : start + shard_size
                    ] *= int4_rescale
                start += shard_size

        layer.w13_fp8_scale = torch.nn.Parameter(max_w13_scales, requires_grad=False)

        # special hack to asm_moe, which takes (weight_int4_scale * weight_scale) as post GEMM scaling
        # optimal design - shall apply per-column weight_int4_scale before GEMM, and weight_scale post
        for expert_id in range(layer.num_experts):
            layer.w13_int4_scale[expert_id] *= max_w13_scales[expert_id]
            layer.w2_int4_scale[expert_id] *= layer.w2_fp8_scale[expert_id]

    def create_moe_runner(
        self, layer: torch.nn.Module, moe_runner_config: MoeRunnerConfig
    ):
        self.moe_runner_config = moe_runner_config

    def apply(
        self,
        layer: torch.nn.Module,
        dispatch_output: "DispatchOutput",
    ) -> torch.Tensor:
        # TODO: fix circular imports issues in sglang forcing us to import here instead of at
        # the top of file.
        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

        topk_output = dispatch_output.topk_output
        moe_runner_config = self.moe_runner_config

        # TODO: add triton kernel and add check get_bool_env_var("CK_MOE")
        assert (
            not moe_runner_config.no_combine
        ), f"no_combine={moe_runner_config.no_combine} is not supported."

        output = fused_moe(
            dispatch_output.hidden_states,
            layer.w13_weight,
            layer.w2_weight,
            topk_output.topk_weights,
            topk_output.topk_ids,
            quant_type=QuantType.per_Token,
            w1_scale=layer.w13_int4_scale,
            w2_scale=layer.w2_int4_scale,
            activation=(
                ActivationType.Silu
                if moe_runner_config.activation == "silu"
                else ActivationType.Gelu
            ),
        )

        return StandardCombineInput(hidden_states=output)
