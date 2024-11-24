# Adapted from
# https://github.com/vllm-project/vllm/tree/v0.5.4/vllm/model_executor/layers/fused_moe
import os
from abc import abstractmethod
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from vllm.logger import init_logger
from vllm.model_executor.custom_op import CustomOp
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.layers.quantization.fp8 import Fp8Config
from vllm.model_executor.utils import set_weight_attrs

from sglang.srt.layers.fused_moe_grok.fused_moe import padding_size
from sglang.srt.utils import is_hip

logger = init_logger(__name__)


class FusedMoEMethodBase(QuantizeMethodBase):

    @abstractmethod
    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        raise NotImplementedError

    @abstractmethod
    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        renormalize: bool = True,
        use_grouped_topk: bool = False,
        num_expert_group: Optional[int] = None,
        topk_group: Optional[int] = None,
    ) -> torch.Tensor:
        raise NotImplementedError


class UnquantizedFusedMoEMethod(FusedMoEMethodBase, CustomOp):
    """MoE method without quantization."""

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):

        # Fused gate_up_proj (column parallel)
        w13_weight = torch.nn.Parameter(
            torch.empty(
                num_experts, 2 * intermediate_size, hidden_size, dtype=params_dtype
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        # down_proj (row parallel)
        w2_weight = torch.nn.Parameter(
            torch.empty(
                num_experts, hidden_size, intermediate_size, dtype=params_dtype
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        renormalize: bool = True,
        use_grouped_topk: bool = False,
        num_expert_group: Optional[int] = None,
        topk_group: Optional[int] = None,
    ) -> torch.Tensor:
        return self.forward(
            x,
            layer.w13_weight,
            layer.w2_weight,
            router_logits,
            top_k,
            renormalize,
            use_grouped_topk,
            num_expert_group,
            topk_group,
        )

    def forward_cuda(
        self,
        x: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        renormalize: bool,
        use_grouped_topk: bool,
        num_expert_group: Optional[int],
        topk_group: Optional[int],
    ) -> torch.Tensor:
        from sglang.srt.layers.fused_moe_grok.fused_moe import fused_moe

        return fused_moe(
            x,
            w1,
            w2,
            router_logits,
            top_k,
            renormalize=renormalize,
            inplace=True,
            use_grouped_topk=use_grouped_topk,
            num_expert_group=num_expert_group,
            topk_group=topk_group,
        )

    def forward_cpu(self, *args, **kwargs):
        raise NotImplementedError("The CPU backend currently does not support MoE.")

    def forward_tpu(
        self,
        x: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        renormalize: bool,
        use_grouped_topk: bool,
        num_expert_group: Optional[int],
        topk_group: Optional[int],
    ) -> torch.Tensor:
        raise NotImplementedError("The TPU backend currently does not support MoE.")


class FusedMoE(torch.nn.Module):
    """FusedMoE layer for MoE models.

    This layer contains both MergedColumnParallel weights (gate_up_proj /
    w13) and RowParallelLinear weights (down_proj/ w2).

    Note: Mixtral uses w1, w2, and w3 for gate, up, and down_proj. We
    copy that naming convention here and handle any remapping in the
    load_weights function in each model implementation.

    Args:
        num_experts: Number of experts in the model
        top_k: Number of experts selected for each token
        hidden_size: Input hidden state size of the transformer
        intermediate_size: Intermediate size of the experts
        params_dtype: Data type for the parameters.
        reduce_results: Whether to all all_reduce on the output of the layer
        renomalize: Whether to renormalize the logits in the fused_moe kernel
        quant_config: Quantization configure.
    """

    def __init__(
        self,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        params_dtype: Optional[torch.dtype] = None,
        reduce_results: bool = False,
        renormalize: bool = True,
        use_grouped_topk: bool = False,
        num_expert_group: Optional[int] = None,
        topk_group: Optional[int] = None,
        quant_config: Optional[QuantizationConfig] = None,
        tp_size: Optional[int] = None,
        prefix: str = "",
    ):
        super().__init__()

        if params_dtype is None:
            params_dtype = torch.get_default_dtype()

        self.tp_size = (
            tp_size if tp_size is not None else get_tensor_model_parallel_world_size()
        )
        self.top_k = top_k
        self.num_experts = num_experts
        self.intermediate_size_per_partition = intermediate_size // self.tp_size
        self.reduce_results = reduce_results
        self.renormalize = renormalize
        self.use_grouped_topk = use_grouped_topk
        if self.use_grouped_topk:
            assert num_expert_group is not None and topk_group is not None
        self.num_expert_group = num_expert_group
        self.topk_group = topk_group

        if quant_config is None:
            self.quant_method: Optional[QuantizeMethodBase] = (
                UnquantizedFusedMoEMethod()
            )
        else:
            if isinstance(quant_config, Fp8Config):
                self.quant_method = Fp8MoEMethod(quant_config)
            else:
                self.quant_method = quant_config.get_quant_method(self, prefix)
        assert self.quant_method is not None

        self.quant_method.create_weights(
            layer=self,
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=self.intermediate_size_per_partition,
            params_dtype=params_dtype,
            weight_loader=self.weight_loader,
        )

    def weight_loader(
        self,
        param: torch.nn.Parameter,
        loaded_weight: torch.Tensor,
        weight_name: str,
        shard_id: int,
        expert_id: int,
        use_presharded_weights: bool = False,
    ):
        param_data = param.data

        # Input scales can be loaded directly and should be equal.
        if "input_scale" in weight_name:
            if (
                param_data[expert_id] != 1
                and (param_data[expert_id] - loaded_weight).abs() > 1e-5
            ):
                raise ValueError(
                    "input_scales of w1 and w3 of a layer "
                    f"must be equal. But got {param_data[expert_id]} "
                    f"vs. {loaded_weight}"
                )
            param_data[expert_id] = loaded_weight
        # Weight scales
        elif "weight_scale" in weight_name:
            # If we are in merged column case (gate_up_proj)
            #   shard_id 0 == gate_proj / w1
            #   shard_id 2 == up_proj / w3
            if shard_id == 0 or shard_id == 2:
                # We have to keep the weight scales of w1 and w3 because
                # we need to re-quantize w1/w3 weights after weight loading.
                idx = 0 if shard_id == 0 else 1
                param_data[expert_id][idx] = loaded_weight
            # If we are in the row parallel case (down_proj)
            #   shard_id 1 == down_proj / w2
            else:
                param_data[expert_id] = loaded_weight
        # Weights
        else:
            tp_rank = get_tensor_model_parallel_rank()
            shard_size = self.intermediate_size_per_partition
            if use_presharded_weights:
                shard = slice(None)
            else:
                shard = slice(tp_rank * shard_size, (tp_rank + 1) * shard_size)

            # w1, gate_proj case: Load into first shard of w13.
            if shard_id == 0:
                param_data[expert_id, 0:shard_size, :] = loaded_weight[shard, :]
            # w3, up_proj case: Load into second shard of w13.
            elif shard_id == 2:
                param_data[expert_id, shard_size : 2 * shard_size, :] = loaded_weight[
                    shard, :
                ]
            # w2, down_proj case: Load into only shard of w2.
            elif shard_id == 1:
                param_data[expert_id, :, :] = loaded_weight[:, shard]
            else:
                raise ValueError(f"Shard id must be in [0,1,2] but got {shard_id}")

    def forward(self, hidden_states: torch.Tensor, router_logits: torch.Tensor):
        assert self.quant_method is not None

        # Matrix multiply.
        final_hidden_states = self.quant_method.apply(
            self,
            x=hidden_states,
            router_logits=router_logits,
            top_k=self.top_k,
            renormalize=self.renormalize,
            use_grouped_topk=self.use_grouped_topk,
            num_expert_group=self.num_expert_group,
            topk_group=self.topk_group,
        )

        if self.reduce_results and self.tp_size > 1:
            final_hidden_states = tensor_model_parallel_all_reduce(final_hidden_states)

        return final_hidden_states

    @classmethod
    def make_expert_params_mapping(
        cls,
        ckpt_gate_proj_name: str,
        ckpt_down_proj_name: str,
        ckpt_up_proj_name: str,
        num_experts: int,
    ) -> List[Tuple[str, str, int, int]]:

        gate_up = [ckpt_gate_proj_name, ckpt_up_proj_name]
        gate_down_up = [ckpt_gate_proj_name, ckpt_down_proj_name, ckpt_up_proj_name]

        return (
            [
                # These are the weight scales for the experts
                # (param_name, weight_name, expert_id, shard_id)
                (
                    (
                        "experts.w13_scale"
                        if weight_name in gate_up
                        else "experts.w2_scale"
                    ),
                    f"experts.{expert_id}.{weight_name}.weight_scale",
                    expert_id,
                    shard_id,
                )
                for expert_id in range(num_experts)
                for shard_id, weight_name in enumerate(gate_down_up)
            ]
            + [
                # These are the weights for the experts
                # (param_name, weight_name, expert_id, shard_id)
                (
                    (
                        "experts.w13_weight"
                        if weight_name in gate_up
                        else "experts.w2_weight"
                    ),
                    f"experts.{expert_id}.{weight_name}.weight",
                    expert_id,
                    shard_id,
                )
                for expert_id in range(num_experts)
                for shard_id, weight_name in enumerate(gate_down_up)
            ]
            + [
                # These are the weight scales for the experts
                # (param_name, weight_name, expert_id, shard_id)
                (
                    (
                        "experts.a13_scale"
                        if weight_name in gate_up
                        else "experts.a2_scale"
                    ),
                    f"experts.{expert_id}.{weight_name}.input_scale",
                    expert_id,
                    shard_id,
                )
                for expert_id in range(num_experts)
                for shard_id, weight_name in enumerate(gate_down_up)
            ]
        )


import torch
from torch.nn import Module
from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
    all_close_1d,
    normalize_e4m3fn_to_e4m3fnuz,
    per_tensor_dequantize,
)
from vllm.utils import print_warning_once


class Fp8MoEMethod(FusedMoEMethodBase):
    """MoE method for FP8.
    Supports loading FP8 checkpoints with static weight scale and
    dynamic/static activation scale.

    Also supports loading quantized FP16/BF16 model checkpoints with dynamic
    activation scaling. The weight scaling factor will be initialized after
    the model weights are loaded.

    Args:
        quant_config: The quantization config.
    """

    def __init__(self, quant_config: Fp8Config):
        self.quant_config = quant_config

    def create_weights(
        self,
        layer: Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):

        if self.quant_config.is_checkpoint_fp8_serialized:
            params_dtype = torch.float8_e4m3fn

        # WEIGHTS
        w13_weight = torch.nn.Parameter(
            torch.empty(
                num_experts, 2 * intermediate_size, hidden_size, dtype=params_dtype
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w2_weight = torch.nn.Parameter(
            torch.empty(
                num_experts, hidden_size, intermediate_size, dtype=params_dtype
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        # WEIGHT_SCALES
        # Allocate 2 scales for w1 and w3 respectively.
        # They will be combined to a single scale after weight loading.
        w13_scale = torch.nn.Parameter(
            torch.ones(num_experts, 2, dtype=torch.float32), requires_grad=False
        )
        layer.register_parameter("w13_scale", w13_scale)

        w2_scale = torch.nn.Parameter(
            torch.ones(num_experts, dtype=torch.float32), requires_grad=False
        )
        layer.register_parameter("w2_scale", w2_scale)

        # If loading fp8 checkpoint, pass the weight loaders.
        # If loading an fp16 checkpoint, do not (we will quantize in
        #   process_weights_after_loading()
        if self.quant_config.is_checkpoint_fp8_serialized:
            set_weight_attrs(w13_scale, extra_weight_attrs)
            set_weight_attrs(w2_scale, extra_weight_attrs)

        # INPUT_SCALES
        if self.quant_config.activation_scheme == "static":
            if not self.quant_config.is_checkpoint_fp8_serialized:
                raise ValueError(
                    "Found static activation scheme for checkpoint that "
                    "was not serialized fp8."
                )

            a13_scale = torch.nn.Parameter(
                torch.ones(num_experts, dtype=torch.float32), requires_grad=False
            )
            layer.register_parameter("a13_scale", a13_scale)
            set_weight_attrs(a13_scale, extra_weight_attrs)

            a2_scale = torch.nn.Parameter(
                torch.ones(num_experts, dtype=torch.float32), requires_grad=False
            )
            layer.register_parameter("a2_scale", a2_scale)
            set_weight_attrs(a2_scale, extra_weight_attrs)
        else:
            layer.a13_scale = None
            layer.a2_scale = None

    def process_weights_after_loading(self, layer: Module) -> None:

        # If checkpoint is fp16 or bfloat16, quantize in place.
        if not self.quant_config.is_checkpoint_fp8_serialized:
            # If ROCm, use float8_e4m3fnuz instead (MI300x HW)
            fp8_dtype = torch.float8_e4m3fnuz if is_hip() else torch.float8_e4m3fn
            w13_weight = torch.empty_like(layer.w13_weight.data, dtype=fp8_dtype)
            w2_weight = torch.empty_like(layer.w2_weight.data, dtype=fp8_dtype)

            # Re-initialize w13_scale because we directly quantize
            # merged w13 weights and generate a single scaling factor.
            layer.w13_scale = torch.nn.Parameter(
                torch.ones(
                    layer.num_experts, dtype=torch.float32, device=w13_weight.device
                ),
                requires_grad=False,
            )
            for expert in range(layer.num_experts):
                w13_weight[expert, :, :], layer.w13_scale[expert] = (
                    ops.scaled_fp8_quant(layer.w13_weight.data[expert, :, :])
                )
                w2_weight[expert, :, :], layer.w2_scale[expert] = ops.scaled_fp8_quant(
                    layer.w2_weight.data[expert, :, :]
                )
            layer.w13_weight = torch.nn.Parameter(w13_weight, requires_grad=False)
            layer.w2_weight = torch.nn.Parameter(w2_weight, requires_grad=False)

            # If ROCm, apply weight padding (min. Mem channel contention) only if set
            if is_hip() and bool(int(os.getenv("MOE_PADDING", "0"))):
                layer.w13_weight = torch.nn.Parameter(
                    F.pad(layer.w13_weight.data, (0, padding_size), "constant", 0),
                    requires_grad=False,
                )
                torch.cuda.empty_cache()
                layer.w2_weight = torch.nn.Parameter(
                    F.pad(layer.w2_weight.data, (0, padding_size), "constant", 0),
                    requires_grad=False,
                )
                torch.cuda.empty_cache()
            return

        # If checkpoint is fp8, we need to handle that the
        # MoE kernels require single activation scale and single weight
        # scale for w13 per expert.
        else:
            # Fp8 moe kernels require a single activation scale.
            # We take the max of all the scales in case they differ.
            if self.quant_config.activation_scheme == "static":
                if layer.a13_scale is None or layer.a2_scale is None:
                    raise ValueError(
                        "QuantConfig has static quantization, but found "
                        "activation scales are None."
                    )
                if not all_close_1d(layer.a13_scale) or not all_close_1d(
                    layer.a2_scale
                ):
                    print_warning_once(
                        "Found input_scales that are not equal for "
                        "fp8 MoE layer. Using the maximum across experts "
                        "for each layer. "
                    )
                layer.a13_scale = torch.nn.Parameter(
                    layer.a13_scale.max(), requires_grad=False
                )
                layer.a2_scale = torch.nn.Parameter(
                    layer.a2_scale.max(), requires_grad=False
                )

            # If ROCm, normalize the weights and scales to e4m3fnuz
            if is_hip():
                # Normalize the weights and scales
                w13_weight, w13_scale, a13_scale = normalize_e4m3fn_to_e4m3fnuz(
                    layer.w13_weight, layer.w13_scale, layer.a13_scale
                )
                w2_weight, w2_scale, a2_scale = normalize_e4m3fn_to_e4m3fnuz(
                    layer.w2_weight, layer.w2_scale, layer.a2_scale
                )
                # Reset the parameters
                layer.w13_weight = torch.nn.Parameter(w13_weight, requires_grad=False)
                layer.w13_scale = torch.nn.Parameter(w13_scale, requires_grad=False)
                if a13_scale is not None:
                    layer.a13_scale = torch.nn.Parameter(a13_scale, requires_grad=False)
                layer.w2_weight = torch.nn.Parameter(w2_weight, requires_grad=False)
                layer.w2_scale = torch.nn.Parameter(w2_scale, requires_grad=False)
                if a2_scale is not None:
                    layer.a2_scale = torch.nn.Parameter(a2_scale, requires_grad=False)

            # Fp8 moe kernel needs single weight scale for w13 per expert.
            # We take the max then dequant and requant each expert.
            assert layer.w13_scale is not None
            shard_size = layer.intermediate_size_per_partition
            max_w13_scales = layer.w13_scale.max(dim=1).values
            for expert_id in range(layer.num_experts):
                start = 0
                for shard_id in range(2):
                    dq_weight = per_tensor_dequantize(
                        layer.w13_weight[expert_id][start : start + shard_size, :],
                        layer.w13_scale[expert_id][shard_id],
                    )
                    layer.w13_weight[expert_id][start : start + shard_size, :], _ = (
                        ops.scaled_fp8_quant(dq_weight, max_w13_scales[expert_id])
                    )
                    start += shard_size

            layer.w13_scale = torch.nn.Parameter(max_w13_scales, requires_grad=False)
            # If ROCm, apply weight padding (min. Mem channel contention) only if set
            if is_hip() and bool(int(os.getenv("MOE_PADDING", "0"))):
                layer.w13_weight = torch.nn.Parameter(
                    F.pad(layer.w13_weight.data, (0, padding_size), "constant", 0),
                    requires_grad=False,
                )
                torch.cuda.empty_cache()
                layer.w2_weight = torch.nn.Parameter(
                    F.pad(layer.w2_weight.data, (0, padding_size), "constant", 0),
                    requires_grad=False,
                )
                torch.cuda.empty_cache()
            return

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        renormalize: bool = True,
        use_grouped_topk: bool = False,
        num_expert_group: Optional[int] = None,
        topk_group: Optional[int] = None,
    ) -> torch.Tensor:

        from sglang.srt.layers.fused_moe_grok.fused_moe import fused_moe

        return fused_moe(
            x,
            layer.w13_weight,
            layer.w2_weight,
            router_logits,
            top_k,
            renormalize=renormalize,
            inplace=True,
            use_fp8=True,
            w1_scale=layer.w13_scale,
            w2_scale=layer.w2_scale,
            a1_scale=layer.a13_scale,
            a2_scale=layer.a2_scale,
            use_grouped_topk=use_grouped_topk,
            num_expert_group=num_expert_group,
            topk_group=topk_group,
        )
