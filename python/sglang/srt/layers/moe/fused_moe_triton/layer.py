# Adapted from https://github.com/vllm-project/vllm/blob/a6221a144af772fd1a68fe7e627935dc53e81738/vllm/model_executor/layers/fused_moe/layer.py

<<<<<<< HEAD
from abc import abstractmethod
from enum import Enum
from typing import Callable, List, Optional, Tuple

import torch

from sglang.srt.custom_op import CustomOp
from sglang.srt.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from sglang.srt.layers.amx_utils import _amx_process_weight_after_loading
from sglang.srt.layers.moe.fused_moe_native import moe_forward_native
from sglang.srt.layers.moe.topk import select_experts
=======
import logging
from enum import Enum
from typing import List, Optional, Tuple

import torch

from sglang.srt.distributed import (
    get_moe_expert_parallel_rank,
    get_moe_expert_parallel_world_size,
    get_moe_tensor_parallel_rank,
    get_moe_tensor_parallel_world_size,
    get_tp_group,
    tensor_model_parallel_all_reduce,
)
from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    use_symmetric_memory,
)
from sglang.srt.eplb.expert_location import get_global_expert_location_metadata
from sglang.srt.layers.moe import (
    MoeRunnerConfig,
    get_moe_runner_backend,
    should_use_flashinfer_trtllm_moe,
)
from sglang.srt.layers.moe.topk import TopKOutput, TopKOutputChecker
>>>>>>> origin/main
from sglang.srt.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
<<<<<<< HEAD
=======
from sglang.srt.layers.quantization.fp8 import Fp8MoEMethod
from sglang.srt.layers.quantization.modelopt_quant import ModelOptNvFp4FusedMoEMethod
from sglang.srt.layers.quantization.unquant import UnquantizedFusedMoEMethod
from sglang.srt.managers.schedule_batch import global_server_args_dict
>>>>>>> origin/main
from sglang.srt.model_loader.weight_utils import narrow_padded_param_and_loaded_weight
from sglang.srt.utils import (
    cpu_has_amx_support,
    get_bool_env_var,
    is_cpu,
<<<<<<< HEAD
    is_hip,
    set_weight_attrs,
    use_intel_amx_backend,
)

if torch.cuda.is_available():
    from sglang.srt.layers.moe.fused_moe_triton.fused_moe import fused_experts
else:
    fused_experts = None  # type: ignore

import logging
=======
    is_flashinfer_available,
    is_hip,
    next_power_of_2,
    round_up,
)

if is_flashinfer_available():
    from flashinfer import (
        RoutingMethodType,
        fp4_quantize,
        reorder_rows_for_gated_act_gemm,
        shuffle_matrix_a,
        shuffle_matrix_sf_a,
    )
>>>>>>> origin/main

_is_hip = is_hip()
_is_cpu_amx_available = cpu_has_amx_support()
_is_cpu = is_cpu()
<<<<<<< HEAD
_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip

if _use_aiter:
    from aiter import ActivationType
    from aiter.fused_moe import fused_moe
    from aiter.fused_moe_bf16_asm import ck_moe_2stages
    from aiter.ops.shuffle import shuffle_weight
=======


# Try to import FP4 TRTLLM function if flashinfer is available
trtllm_fp4_block_scale_moe = None
if should_use_flashinfer_trtllm_moe():
    try:
        from flashinfer.fused_moe import trtllm_fp4_block_scale_moe
    except ImportError:
        trtllm_fp4_block_scale_moe = None
>>>>>>> origin/main

logger = logging.getLogger(__name__)


<<<<<<< HEAD
=======
def _is_fp4_quantization_enabled():
    """Check if ModelOpt FP4 quantization is enabled."""
    try:
        # Use the same simple check that works for class selection
        quantization = global_server_args_dict.get("quantization")
        return quantization == "modelopt_fp4"
    except:
        return False


def _get_tile_tokens_dim(num_tokens, top_k, num_experts):
    # Guess tokens per expert assuming perfect expert distribution first.
    num_tokens_per_expert = (num_tokens * top_k) // num_experts
    # And pad the number to the next power of 2.
    tile_tokens_dim = next_power_of_2(num_tokens_per_expert)
    # Cap to 8-64 tokens per CTA tile as it's the range supported by the kernel.
    tile_tokens_dim = min(max(tile_tokens_dim, 8), 64)
    return tile_tokens_dim


>>>>>>> origin/main
class FusedMoeWeightScaleSupported(Enum):
    TENSOR = "tensor"
    CHANNEL = "channel"
    GROUP = "group"
    BLOCK = "block"


<<<<<<< HEAD
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
        renormalize: bool,
        use_grouped_topk: bool,
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

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if _use_aiter:
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

        # Pack weight for get better performance on CPU
        if _is_cpu and _is_cpu_amx_available:
            _amx_process_weight_after_loading(layer, ["w13_weight", "w2_weight"])

        return

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        renormalize: bool,
        use_grouped_topk: bool,
        topk_group: Optional[int] = None,
        num_expert_group: Optional[int] = None,
        num_fused_shared_experts: int = 0,
        custom_routing_function: Optional[Callable] = None,
        correction_bias: Optional[torch.Tensor] = None,
        activation: str = "silu",
        apply_router_weight_on_input: bool = False,
        inplace: bool = True,
        no_combine: bool = False,
        routed_scaling_factor: Optional[float] = None,
    ) -> torch.Tensor:
        return self.forward(
            x=x,
            layer=layer,
            router_logits=router_logits,
            top_k=top_k,
            renormalize=renormalize,
            use_grouped_topk=use_grouped_topk,
            topk_group=topk_group,
            num_expert_group=num_expert_group,
            num_fused_shared_experts=num_fused_shared_experts,
            custom_routing_function=custom_routing_function,
            correction_bias=correction_bias,
            activation=activation,
            apply_router_weight_on_input=apply_router_weight_on_input,
            inplace=inplace,
            no_combine=no_combine,
            routed_scaling_factor=routed_scaling_factor,
        )

    def forward_cuda(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        use_grouped_topk: bool,
        top_k: int,
        router_logits: torch.Tensor,
        renormalize: bool,
        topk_group: Optional[int] = None,
        num_expert_group: Optional[int] = None,
        num_fused_shared_experts: int = 0,
        custom_routing_function: Optional[Callable] = None,
        correction_bias: Optional[torch.Tensor] = None,
        activation: str = "silu",
        apply_router_weight_on_input: bool = False,
        inplace: bool = True,
        no_combine: bool = False,
        routed_scaling_factor: Optional[float] = None,
    ) -> torch.Tensor:
        topk_weights, topk_ids = select_experts(
            hidden_states=x,
            router_logits=router_logits,
            use_grouped_topk=use_grouped_topk,
            top_k=top_k,
            renormalize=renormalize,
            topk_group=topk_group,
            num_expert_group=num_expert_group,
            num_fused_shared_experts=num_fused_shared_experts,
            custom_routing_function=custom_routing_function,
            correction_bias=correction_bias,
            routed_scaling_factor=routed_scaling_factor,
        )

        if _use_aiter:
            assert not no_combine, "unsupported"
            if apply_router_weight_on_input:
                assert (
                    topk_weights.dim() == 2
                ), "`topk_weights` should be in shape (num_tokens, topk)"
                _, topk = topk_weights.shape
                assert (
                    topk == 1
                ), "Only support topk=1 when `apply_router_weight_on_input` is True"
                x = x * topk_weights.to(x.dtype)
                topk_weights = torch.ones_like(
                    topk_weights, dtype=torch.float32
                )  # topk_weights must be FP32 (float32)

            return fused_moe(
                x,
                layer.w13_weight,
                layer.w2_weight,
                topk_weights,
                topk_ids,
                activation=(
                    ActivationType.Silu if activation == "silu" else ActivationType.Gelu
                ),
            )
        else:
            return fused_experts(
                hidden_states=x,
                w1=layer.w13_weight,
                w2=layer.w2_weight,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                inplace=inplace and not no_combine,
                activation=activation,
                apply_router_weight_on_input=apply_router_weight_on_input,
                no_combine=no_combine,
                routed_scaling_factor=routed_scaling_factor,
            )

    def forward_cpu(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        use_grouped_topk: bool,
        top_k: int,
        router_logits: torch.Tensor,
        renormalize: bool,
        topk_group: Optional[int] = None,
        num_expert_group: Optional[int] = None,
        num_fused_shared_experts: int = 0,
        custom_routing_function: Optional[Callable] = None,
        correction_bias: Optional[torch.Tensor] = None,
        activation: str = "silu",
        apply_router_weight_on_input: bool = False,
        inplace: bool = True,
        no_combine: bool = False,
        routed_scaling_factor: Optional[float] = None,
    ) -> torch.Tensor:
        assert activation == "silu", f"activation = {activation} is not supported."

        if use_intel_amx_backend(layer) and not apply_router_weight_on_input:
            topk_weights, topk_ids = select_experts(
                hidden_states=x,
                router_logits=router_logits,
                use_grouped_topk=use_grouped_topk,
                top_k=top_k,
                renormalize=renormalize,
                topk_group=topk_group,
                num_expert_group=num_expert_group,
                num_fused_shared_experts=num_fused_shared_experts,
                custom_routing_function=custom_routing_function,
                correction_bias=correction_bias,
                routed_scaling_factor=routed_scaling_factor,
            )

            # TODO: support apply_router_weight_on_input in the fused_experts_cpu kernel
            return torch.ops.sgl_kernel.fused_experts_cpu(
                x,
                layer.w13_weight,
                layer.w2_weight,
                topk_weights.to(
                    torch.float
                ),  # TODO: the topk_weights of llama4 is computed via Llama4MoE:custom_routing_function and is bfloat16 while the kernel requires it to be float32
                topk_ids,
                False,  # inplace # See [Note] inplace should be False in fused_experts.
                False,  # use_int8_w8a8
                False,  # use_fp8_w8a16
                None,  # w1_scale
                None,  # w2_scale
                None,  # block_size
                None,  # a1_scale
                None,  # a2_scale
                True,  # is_vnni
            )
        else:
            return moe_forward_native(
                layer,
                x,
                use_grouped_topk,
                top_k,
                router_logits,
                renormalize,
                topk_group,
                num_expert_group,
                num_fused_shared_experts,
                custom_routing_function,
                correction_bias,
                activation,
                apply_router_weight_on_input,
                inplace,
                no_combine,
                routed_scaling_factor,
            )

    def forward_npu(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        use_grouped_topk: bool,
        top_k: int,
        router_logits: torch.Tensor,
        renormalize: bool,
        topk_group: Optional[int] = None,
        num_expert_group: Optional[int] = None,
        num_fused_shared_experts: int = 0,
        custom_routing_function: Optional[Callable] = None,
        correction_bias: Optional[torch.Tensor] = None,
        activation: str = "silu",
        apply_router_weight_on_input: bool = False,
        inplace: bool = True,
        no_combine: bool = False,
        routed_scaling_factor: Optional[float] = None,
    ) -> torch.Tensor:
        return moe_forward_native(
            layer,
            x,
            use_grouped_topk,
            top_k,
            router_logits,
            renormalize,
            topk_group,
            num_expert_group,
            num_fused_shared_experts,
            custom_routing_function,
            correction_bias,
            activation,
            apply_router_weight_on_input,
            inplace,
            no_combine,
            routed_scaling_factor,
        )

    def forward_tpu(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError("The TPU backend currently does not support MoE.")

    forward_native = forward_cpu


=======
>>>>>>> origin/main
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
<<<<<<< HEAD
        reduce_results: Whether to all all_reduce on the output of the layer
        renomalize: Whether to renormalize the logits in the fused_moe kernel
        quant_config: Quantization configure.
=======
        reduce_results: Whether to apply all_reduce on the output of the layer
        quant_config: Quantization configuration.
>>>>>>> origin/main
        inplace: suggestion to compute inplace (modify input activation).
    """

    def __init__(
        self,
        num_experts: int,
<<<<<<< HEAD
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        layer_id: Optional[int] = None,
        params_dtype: Optional[torch.dtype] = None,
        reduce_results: bool = False,
        renormalize: bool = True,
        use_grouped_topk: bool = False,
        num_expert_group: Optional[int] = None,
        num_fused_shared_experts: int = 0,
        topk_group: Optional[int] = None,
        quant_config: Optional[QuantizationConfig] = None,
        tp_size: Optional[int] = None,
        prefix: str = "",
        custom_routing_function: Optional[Callable] = None,
        correction_bias: Optional[torch.Tensor] = None,
=======
        hidden_size: int,
        intermediate_size: int,
        layer_id: int,
        top_k: Optional[int] = None,
        num_fused_shared_experts: int = 0,
        params_dtype: Optional[torch.dtype] = None,
        reduce_results: bool = False,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
>>>>>>> origin/main
        activation: str = "silu",
        apply_router_weight_on_input: bool = False,
        use_presharded_weights: bool = False,
        inplace: bool = True,
        no_combine: bool = False,
        routed_scaling_factor: Optional[float] = None,
<<<<<<< HEAD
        enable_flashinfer_moe: Optional[bool] = False,
        enable_ep_moe: Optional[bool] = False,
=======
        gemm1_alpha: Optional[float] = None,
        gemm1_clamp_limit: Optional[float] = None,
        use_weight_loader_fused: bool = False,
        with_bias=False,
>>>>>>> origin/main
    ):
        super().__init__()

        if params_dtype is None:
            params_dtype = torch.get_default_dtype()

<<<<<<< HEAD
        self.hidden_size = hidden_size
        self.tp_size = (
            tp_size if tp_size is not None else get_tensor_model_parallel_world_size()
        )
        self.tp_rank = get_tensor_model_parallel_rank()
        self.num_experts = num_experts
        self.expert_map = None

        if enable_flashinfer_moe and quant_config is None:
            logger.warning("Disable flashinfer MoE when quantization config is None.")
            enable_flashinfer_moe = False
            enable_ep_moe = False

        self.enable_flashinfer_moe = enable_flashinfer_moe
        if enable_ep_moe:
            assert (
                self.enable_flashinfer_moe
            ), "FusedMoE only supports EP with --enable-flashinfer-moe"
            self.ep_size = self.tp_size
            self.ep_rank = self.tp_rank
            self.tp_size = 1
            self.tp_rank = 0
            # Create a tensor of size num_experts filled with -1
            self.expert_map = torch.full((self.num_experts,), -1, dtype=torch.int32)
            # Create a expert map for the local experts
            assert num_experts % self.ep_size == 0
            self.local_num_experts = num_experts // self.ep_size
            self.expert_map[
                self.ep_rank
                * self.local_num_experts : (self.ep_rank + 1)
                * self.local_num_experts
            ] = torch.arange(0, self.local_num_experts, dtype=torch.int32, device="cpu")
        else:
            self.ep_size = 1
            self.ep_rank = 0
            self.local_num_experts = num_experts
        self.routed_scaling_factor = routed_scaling_factor
        self.top_k = top_k
        assert intermediate_size % self.tp_size == 0
        self.intermediate_size_per_partition = intermediate_size // self.tp_size
        self.reduce_results = reduce_results
        self.renormalize = renormalize
        self.use_grouped_topk = use_grouped_topk
        if self.use_grouped_topk:
            assert num_expert_group is not None and topk_group is not None
        self.num_expert_group = num_expert_group
        self.num_fused_shared_experts = num_fused_shared_experts
        self.topk_group = topk_group
        self.custom_routing_function = custom_routing_function
        self.correction_bias = correction_bias
        self.activation = activation
        self.apply_router_weight_on_input = apply_router_weight_on_input
        self.use_presharded_weights = use_presharded_weights
        self.inplace = inplace
        self.no_combine = no_combine

        if quant_config is None:
            self.quant_method: Optional[QuantizeMethodBase] = (
                UnquantizedFusedMoEMethod()
            )
        else:
            self.quant_method = quant_config.get_quant_method(self, prefix)
            if self.quant_method.__class__.__name__ == "ModelOptNvFp4FusedMoEMethod":
                self.quant_method.enable_flashinfer_moe = self.enable_flashinfer_moe
        assert self.quant_method is not None

        self.quant_method.create_weights(
            layer=self,
            num_experts=self.local_num_experts,
=======
        self.layer_id = layer_id
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.num_fused_shared_experts = num_fused_shared_experts
        self.expert_map_cpu = None
        self.expert_map_gpu = None

        self.moe_runner_config = MoeRunnerConfig(
            activation=activation,
            apply_router_weight_on_input=apply_router_weight_on_input,
            inplace=inplace,
            no_combine=no_combine,
            routed_scaling_factor=routed_scaling_factor,
            gemm1_alpha=gemm1_alpha,
            gemm1_clamp_limit=gemm1_clamp_limit,
        )

        enable_flashinfer_cutlass_moe = get_moe_runner_backend().is_flashinfer_cutlass()

        if enable_flashinfer_cutlass_moe and quant_config is None:
            logger.warning("Disable flashinfer MoE when quantization config is None.")
            enable_flashinfer_cutlass_moe = False

        self.enable_flashinfer_cutlass_moe = enable_flashinfer_cutlass_moe
        self.moe_ep_size = get_moe_expert_parallel_world_size()
        self.moe_ep_rank = get_moe_expert_parallel_rank()
        self.moe_tp_size = get_moe_tensor_parallel_world_size()
        self.moe_tp_rank = get_moe_tensor_parallel_rank()
        assert num_experts % self.moe_ep_size == 0
        self.num_local_experts = num_experts // self.moe_ep_size
        if self.moe_ep_size > 1:
            # TODO(ch-wan): support shared experts fusion
            # Create a tensor of size num_experts filled with -1
            self.expert_map_cpu = torch.full(
                (self.num_experts,), -1, dtype=torch.int32, device="cpu"
            )
            # Create a expert map for the local experts
            self.expert_map_cpu[
                self.moe_ep_rank
                * self.num_local_experts : (self.moe_ep_rank + 1)
                * self.num_local_experts
            ] = torch.arange(0, self.num_local_experts, dtype=torch.int32, device="cpu")

        assert intermediate_size % self.moe_tp_size == 0
        self.intermediate_size_per_partition = intermediate_size // self.moe_tp_size
        self.reduce_results = reduce_results
        self.use_presharded_weights = use_presharded_weights

        self.use_triton_kernels = get_moe_runner_backend().is_triton_kernel()
        if quant_config is None:
            self.quant_method: Optional[QuantizeMethodBase] = UnquantizedFusedMoEMethod(
                self.use_triton_kernels
            )
        else:
            self.quant_method = quant_config.get_quant_method(self, prefix)
        assert self.quant_method is not None

        self.quant_config = quant_config
        self.use_flashinfer_mxfp4_moe = get_moe_runner_backend().is_flashinfer_mxfp4()
        # TODO maybe we should remove this `if`, since `Mxfp4MoEMethod` does another round-up logic
        if (
            self.quant_config is not None
            and self.quant_config.get_name() == "mxfp4"
            and self.use_flashinfer_mxfp4_moe
        ):
            hidden_size = round_up(hidden_size, 256)
        self.quant_method.create_weights(
            layer=self,
            num_experts=self.num_local_experts,
>>>>>>> origin/main
            hidden_size=hidden_size,
            # FIXME: figure out which intermediate_size to use
            intermediate_size=self.intermediate_size_per_partition,
            intermediate_size_per_partition=self.intermediate_size_per_partition,
            params_dtype=params_dtype,
<<<<<<< HEAD
            weight_loader=self.weight_loader,
=======
            weight_loader=(
                self.weight_loader
                if not use_weight_loader_fused
                else self.weight_loader_fused
            ),
            with_bias=with_bias,
>>>>>>> origin/main
        )

    def _load_per_tensor_weight_scale(
        self,
        shard_id: str,
        param: torch.nn.Parameter,
        loaded_weight: torch.Tensor,
        expert_id: int,
    ):
        param_data = param.data
        # for per tensor weight quantization
        if shard_id in ("w1", "w3"):
            # We have to keep the weight scales of w1 and w3 because
            # we need to re-quantize w1/w3 weights after weight loading.
            idx = 0 if shard_id == "w1" else 1
            param_data[expert_id][idx] = loaded_weight
        # If we are in the row parallel case (down_proj)
        elif shard_id == "w2":
            param_data[expert_id] = loaded_weight

    def _load_model_weight_or_group_weight_scale(
        self,
        shard_dim: int,
        expert_data: torch.Tensor,
        shard_id: str,
<<<<<<< HEAD
        loaded_weight: torch.tensor,
        tp_rank: int,
=======
        loaded_weight: torch.Tensor,
        tp_rank: int,
        is_bias: bool = False,
>>>>>>> origin/main
    ):
        # Load grouped weight scales for group quantization
        # or model weights
        if shard_id == "w2":
            self._load_w2(
                shard_id=shard_id,
                shard_dim=shard_dim,
                loaded_weight=loaded_weight,
                expert_data=expert_data,
                tp_rank=tp_rank,
<<<<<<< HEAD
            )
        elif shard_id in ("w1", "w3"):
=======
                is_bias=is_bias,
            )
        elif shard_id in ("w1", "w3", "w13"):
>>>>>>> origin/main
            self._load_w13(
                shard_id=shard_id,
                shard_dim=shard_dim,
                loaded_weight=loaded_weight,
                expert_data=expert_data,
                tp_rank=tp_rank,
<<<<<<< HEAD
=======
                is_bias=is_bias,
>>>>>>> origin/main
            )

    def _load_per_channel_weight_scale(
        self,
        expert_data: torch.Tensor,
        shard_dim: int,
        shard_id: str,
<<<<<<< HEAD
        loaded_weight: torch.tensor,
=======
        loaded_weight: torch.Tensor,
>>>>>>> origin/main
        tp_rank: int,
    ):
        # for per channel weight quantization
        if shard_id == "w2":
            expert_data.copy_(loaded_weight)
        elif shard_id in ("w1", "w3"):
            self._load_w13(
                shard_id=shard_id,
                shard_dim=shard_dim,
                loaded_weight=loaded_weight,
                expert_data=expert_data,
                tp_rank=tp_rank,
            )

    def _load_w13(
        self,
        expert_data: torch.Tensor,
        shard_dim: int,
        shard_id: str,
<<<<<<< HEAD
        loaded_weight: torch.tensor,
        tp_rank: int,
=======
        loaded_weight: torch.Tensor,
        tp_rank: int,
        is_bias: bool = False,
>>>>>>> origin/main
    ):

        # Index the loaded weight for tp sharding.
        # gate_up_proj: "MergedColumnParallel", so tp sharding on output_dim
<<<<<<< HEAD
        shard_size = expert_data.shape[shard_dim] // 2
=======
        assert shard_id in {"w1", "w3", "w13"}

        if is_bias:
            # if this weight is a bias, the last dimension must be the sharded dimension
            shard_dim = -1

        if shard_id in {"w1", "w3"}:
            # non-fused version
            shard_size = expert_data.shape[shard_dim] // 2
        elif shard_id in {"w13"}:
            # fused version
            shard_size = expert_data.shape[shard_dim]
        else:
            raise NotImplementedError
>>>>>>> origin/main

        # Narrow parameter and load.
        # w1, gate_proj: Load into first logical weight of w13.
        # w3, up_proj: Load into second logical weight of w13.
        # trtllm cutlass kernel assumes differently
<<<<<<< HEAD
        assert shard_id in ("w1", "w3")
=======
>>>>>>> origin/main
        switch_w13 = getattr(self.quant_method, "load_up_proj_weight_first", False)
        if (switch_w13 and shard_id == "w1") or (not switch_w13 and shard_id == "w3"):
            start = shard_size
        else:
            start = 0

        if _is_cpu:
            expert_data, loaded_weight = narrow_padded_param_and_loaded_weight(
                expert_data,
                loaded_weight,
                start,
                shard_size * tp_rank,
                shard_dim,
                shard_size,
                not self.use_presharded_weights,
            )
        else:
            if not self.use_presharded_weights:
<<<<<<< HEAD
=======
                if not is_bias and self.use_triton_kernels:
                    # do not transpose for bias
                    loaded_weight = loaded_weight.transpose(-2, -1)
>>>>>>> origin/main
                loaded_weight = loaded_weight.narrow(
                    shard_dim, shard_size * tp_rank, shard_size
                )

            expert_data = expert_data.narrow(shard_dim, start, shard_size)
        expert_data.copy_(loaded_weight)

    def _load_w2(
        self,
        expert_data: torch.Tensor,
        shard_dim: int,
        shard_id: str,
<<<<<<< HEAD
        loaded_weight: torch.tensor,
        tp_rank: int,
    ):
=======
        loaded_weight: torch.Tensor,
        tp_rank: int,
        is_bias: bool = False,
    ):
        """Load w2 weights for down projection.

        Args:
            expert_data: The expert data tensor to load into
            shard_dim: The dimension to shard along
            shard_id: The shard ID (must be "w2")
            loaded_weight: The weight tensor to load from
            tp_rank: The tensor parallel rank
        """
        if not isinstance(expert_data, torch.Tensor) or not isinstance(
            loaded_weight, torch.Tensor
        ):
            raise ValueError("expert_data and loaded_weight must be torch.Tensor")

        if (
            self.quant_config is not None
            and "modelopt" in self.quant_config.get_name()
            and (expert_data.dim() != 2 or loaded_weight.dim() != 2)
        ):
            raise ValueError(
                f"Expected 2D tensors, got expert_data shape {expert_data.shape} and loaded_weight shape {loaded_weight.shape}"
            )

        if shard_id != "w2":
            raise ValueError(f"shard_id must be 'w2', got {shard_id}")
>>>>>>> origin/main

        # Index the loaded weight for tp sharding.
        # down_proj: "RowParallel" so tp sharding on input_dim
        # Narrow parameter and load.
<<<<<<< HEAD
        shard_size = expert_data.shape[shard_dim]
=======
        if is_bias:
            # this expert_data is a bias, not weight,
            # for w2_weight_bias in TP, it does not need to be sharded
            shard_size = expert_data.shape[-1]
        else:
            # this parameter is a weight matrix
            # for w2 in TP, it shards the input_features, i.e., shard_dim=2
            shard_size = expert_data.shape[shard_dim]
>>>>>>> origin/main

        if _is_cpu:
            expert_data, loaded_weight = narrow_padded_param_and_loaded_weight(
                expert_data,
                loaded_weight,
                0,  # param_data_start
                shard_size * tp_rank,
                shard_dim,
                shard_size,
                not self.use_presharded_weights,
            )
        else:
<<<<<<< HEAD
            if not self.use_presharded_weights:
=======
            if not is_bias and not self.use_presharded_weights:
                if self.use_triton_kernels:
                    loaded_weight = loaded_weight.transpose(-2, -1)
>>>>>>> origin/main
                loaded_weight = loaded_weight.narrow(
                    shard_dim, shard_size * tp_rank, shard_size
                )

        # w2, down_proj: Load into only logical weight of w2.
        expert_data.copy_(loaded_weight)

    def _load_single_value(
        self, param: torch.nn.Parameter, loaded_weight: torch.Tensor, expert_id: int
    ):
        param_data = param.data

        # Input scales can be loaded directly and should be equal.
        param_data[expert_id] = loaded_weight

    def _load_g_idx(
        self,
        shard_id: str,
        expert_data: torch.Tensor,
        shard_dim: int,
<<<<<<< HEAD
        loaded_weight: torch.tensor,
=======
        loaded_weight: torch.Tensor,
>>>>>>> origin/main
        tp_rank: int,
    ):

        if shard_id == "w2":
            self._load_w2(
                shard_id=shard_id,
                shard_dim=shard_dim,
                loaded_weight=loaded_weight,
                expert_data=expert_data,
                tp_rank=tp_rank,
            )
        else:
            assert shard_id in ("w1", "w3")
            expert_data.copy_(loaded_weight)

    def _map_global_expert_id_to_local_expert_id(self, expert_id: int) -> int:
<<<<<<< HEAD
        if self.expert_map is None:
            return expert_id
        return self.expert_map[expert_id].item()
=======
        if self.expert_map_cpu is None:
            return expert_id
        return self.expert_map_cpu[expert_id].item()
>>>>>>> origin/main

    def weight_loader(
        self,
        param: torch.nn.Parameter,
        loaded_weight: torch.Tensor,
        weight_name: str,
        shard_id: str,
<<<<<<< HEAD
        expert_id: int,
    ) -> None:
        expert_id = self._map_global_expert_id_to_local_expert_id(expert_id)
        if expert_id == -1:
            return

        # TP rank is set to 0 if EP is enabled
        tp_rank = 0 if self.ep_size > 1 else get_tensor_model_parallel_rank()
=======
        expert_id: Optional[int],
    ) -> None:

        # if expert_id is None, then
        # all the experts are loaded at the same time
        if (
            not expert_id
            and self.quant_config is not None
            and self.quant_config.get_name() == "mxfp4"
            and self.quant_config.is_static_cfg()
        ):
            if "bias" in weight_name:
                dim1 = loaded_weight.shape[1]
                param.data[:, :dim1].copy_(loaded_weight)
            else:
                dim1 = loaded_weight.shape[1]
                dim2 = loaded_weight.shape[2]
                param.data[:, :dim1, :dim2].copy_(loaded_weight)
            return

        global_expert_location_metadata = get_global_expert_location_metadata()
        if global_expert_location_metadata is None:
            self._weight_loader_impl(
                param=param,
                loaded_weight=loaded_weight,
                weight_name=weight_name,
                shard_id=shard_id,
                expert_id=expert_id,
            )
            return

        if expert_id >= self.num_experts - self.num_fused_shared_experts:
            # This is a shared expert.
            physical_expert_ids = [expert_id]
        else:
            physical_expert_ids = (
                global_expert_location_metadata.logical_to_all_physical(
                    self.layer_id, expert_id
                )
            )

        for physical_expert_id in physical_expert_ids:
            self._weight_loader_physical(
                param=param,
                loaded_weight=loaded_weight,
                weight_name=weight_name,
                shard_id=shard_id,
                expert_id=physical_expert_id,
            )

    def _weight_loader_physical(
        self,
        param: torch.nn.Parameter,
        loaded_weight: torch.Tensor,
        weight_name: str,
        shard_id: str,
        expert_id: int,
    ) -> None:

        expert_id = self._map_global_expert_id_to_local_expert_id(expert_id)
        if expert_id == -1:
            return
        self._weight_loader_impl(
            param=param,
            loaded_weight=loaded_weight,
            weight_name=weight_name,
            shard_id=shard_id,
            expert_id=expert_id,
        )

    def _weight_loader_impl(
        self,
        param: torch.nn.Parameter,
        loaded_weight: torch.Tensor,
        weight_name: str,
        shard_id: str,
        expert_id: int,
    ) -> None:

        tp_rank = self.moe_tp_rank
>>>>>>> origin/main

        # compressed-tensors checkpoints with packed weights are stored flipped
        # TODO (mgoin): check self.quant_method.quant_config.quant_format
        # against known CompressionFormat enum values that have this quality
        loaded_weight = (
            loaded_weight.t().contiguous()
            if (
                self.quant_method.__class__.__name__
                == "CompressedTensorsWNA16MoEMethod"
            )
            else loaded_weight
        )

        if shard_id not in ("w1", "w2", "w3"):
            raise ValueError(
                f"shard_id must be ['w1','w2','w3'] but " f"got {shard_id}."
            )

<<<<<<< HEAD
=======
        # Flashinfer assumes w31 format for w13_weight. Same for the scales.
        if should_use_flashinfer_trtllm_moe():
            shard_id = {"w1": "w3", "w3": "w1", "w2": "w2"}[shard_id]

>>>>>>> origin/main
        WEIGHT_SCALE_SUPPORTED = [e.value for e in FusedMoeWeightScaleSupported]
        # Fetch the dim to shard the parameter/loaded weight
        # based on the shard id. This will be whatever
        # dimension intermediate_size is used.
        SHARD_ID_TO_SHARDED_DIM = {"w1": 0, "w2": 1, "w3": 0}

        expert_data = param.data[expert_id]

        # is_transposed: if the dim to shard the weight
        # should be flipped. Required by GPTQ, compressed-tensors
        # should be whatever dimension intermediate_size is
        is_transposed = getattr(param, "is_transposed", False)
        shard_dim = SHARD_ID_TO_SHARDED_DIM[shard_id]
<<<<<<< HEAD
=======
        if self.use_triton_kernels:
            is_transposed = True
>>>>>>> origin/main
        if is_transposed:
            shard_dim = int(not shard_dim)

        # Case input scale: input_scale loading is only supported for fp8
        if "input_scale" in weight_name:
            # INT4-FP8 (INT4 MoE Weight, FP8 Compute): Adjust input_scale for e4m3fnuz (AMD)
            if _is_hip and get_bool_env_var("SGLANG_INT4_WEIGHT"):
                loaded_weight = loaded_weight * 2.0

            # this is needed for compressed-tensors only
            loaded_weight = loaded_weight.to(param.data.device)

            if (
                "compressed" in self.quant_method.__class__.__name__.lower()
                and param.data[expert_id] != 1
                and (param.data[expert_id] - loaded_weight).abs() > 1e-5
            ):
                raise ValueError(
                    "input_scales of w1 and w3 of a layer "
                    f"must be equal. But got {param.data[expert_id]} "
                    f"vs. {loaded_weight}"
                )

            self._load_single_value(
                param=param, loaded_weight=loaded_weight, expert_id=expert_id
            )
            return

        # Case g_idx
        if "g_idx" in weight_name:
            self._load_g_idx(
                shard_dim=0,
                shard_id=shard_id,
                loaded_weight=loaded_weight,
                expert_data=expert_data,
                tp_rank=tp_rank,
            )
            return
<<<<<<< HEAD
        if "ModelOpt" in self.quant_method.__class__.__name__:
            if "weight_scale_2" in weight_name or "input_scale" in weight_name:
=======

        if "ModelOpt" in self.quant_method.__class__.__name__:
            # Determine per-tensor weight scale patterns based on variant
            is_fp4_variant = isinstance(self.quant_method, ModelOptNvFp4FusedMoEMethod)

            # FP4 uses "weight_scale_2" for per-tensor, FP8 uses "weight_scale" for per-tensor
            per_tensor_conditions = (
                "weight_scale_2" in weight_name
                if is_fp4_variant
                else "weight_scale" in weight_name
            ) or "input_scale" in weight_name

            if per_tensor_conditions:
>>>>>>> origin/main
                self._load_per_tensor_weight_scale(
                    shard_id=shard_id,
                    param=param,
                    loaded_weight=loaded_weight,
                    expert_id=expert_id,
                )
            elif "weight" in weight_name:
                self._load_model_weight_or_group_weight_scale(
                    shard_id=shard_id,
                    shard_dim=shard_dim,
                    loaded_weight=loaded_weight,
                    expert_data=expert_data,
                    tp_rank=tp_rank,
                )
            return

        # Case weight scales and zero_points
<<<<<<< HEAD
        if "scale" in weight_name or "zero" in weight_name:
=======
        if "scale" in weight_name or "zero" in weight_name or "offset" in weight_name:
>>>>>>> origin/main
            # load the weight scales and zp based on the quantization scheme
            # supported weight scales/zp can be found in
            # FusedMoeWeightScaleSupported
            # TODO @dsikka: once hardened, refactor to use vLLM Parameters
            # specific to each case
            quant_method = getattr(param, "quant_method", None)
            if quant_method == FusedMoeWeightScaleSupported.CHANNEL.value:
                # INT4-FP8 (INT4 MoE Weight, FP8 Compute): Adjust INT4 column-wise scaling number to e4m3fnuz (AMD)
                if _is_hip and get_bool_env_var("SGLANG_INT4_WEIGHT"):
                    loaded_weight = loaded_weight * 0.5

                self._load_per_channel_weight_scale(
                    shard_id=shard_id,
                    shard_dim=shard_dim,
                    loaded_weight=loaded_weight,
                    expert_data=expert_data,
                    tp_rank=tp_rank,
                )
            elif quant_method in [
                FusedMoeWeightScaleSupported.GROUP.value,
                FusedMoeWeightScaleSupported.BLOCK.value,
            ]:
                self._load_model_weight_or_group_weight_scale(
                    shard_id=shard_id,
                    shard_dim=shard_dim,
                    loaded_weight=loaded_weight,
                    expert_data=expert_data,
                    tp_rank=tp_rank,
                )
            elif quant_method == FusedMoeWeightScaleSupported.TENSOR.value:
                # INT4-FP8 (INT4 MoE Weight, FP8 Compute): Adjust FP8 per-tensor scaling number for e4m3fnuz (AMD)
                if _is_hip and get_bool_env_var("SGLANG_INT4_WEIGHT"):
                    loaded_weight = loaded_weight * 2.0

                self._load_per_tensor_weight_scale(
                    shard_id=shard_id,
                    param=param,
                    loaded_weight=loaded_weight,
                    expert_id=expert_id,
                )
            else:
                raise ValueError(
                    f"quant method must be one of {WEIGHT_SCALE_SUPPORTED}"
                )
            return

        # Case weight_shape
        if "weight_shape" in weight_name:
            # only required by compressed-tensors
            self._load_single_value(
                param=param, loaded_weight=loaded_weight, expert_id=expert_id
            )
            return

        # Case model weights
        if "weight" in weight_name:
            self._load_model_weight_or_group_weight_scale(
                shard_id=shard_id,
                shard_dim=shard_dim,
                loaded_weight=loaded_weight,
                expert_data=expert_data,
                tp_rank=tp_rank,
            )
            return

<<<<<<< HEAD
    def forward(self, hidden_states: torch.Tensor, router_logits: torch.Tensor):
        assert self.quant_method is not None

        # Matrix multiply.
        final_hidden_states = self.quant_method.apply(
            layer=self,
            x=hidden_states,
            router_logits=router_logits,
            top_k=self.top_k,
            renormalize=self.renormalize,
            use_grouped_topk=self.use_grouped_topk,
            topk_group=self.topk_group,
            num_expert_group=self.num_expert_group,
            num_fused_shared_experts=self.num_fused_shared_experts,
            custom_routing_function=self.custom_routing_function,
            correction_bias=self.correction_bias,
            activation=self.activation,
            apply_router_weight_on_input=self.apply_router_weight_on_input,
            routed_scaling_factor=self.routed_scaling_factor,
            **(
                dict(
                    tp_rank=self.tp_rank,
                    tp_size=self.tp_size,
                    ep_rank=self.ep_rank,
                    ep_size=self.ep_size,
                )
                if self.quant_method.__class__.__name__ == "ModelOptNvFp4FusedMoEMethod"
                else {}
            ),
        )

        if self.reduce_results and (self.tp_size > 1 or self.ep_size > 1):
=======
    def weight_loader_fused(
        self,
        param: torch.nn.Parameter,
        loaded_weight: torch.Tensor,
        weight_name: str,
        shard_id: str,
    ) -> None:
        tp_rank = self.moe_tp_rank

        if (
            self.quant_config is not None
            and self.quant_config.get_name() == "mxfp4"
            and self.quant_config.is_static_cfg()
        ):
            if "bias" in weight_name:
                dim1 = loaded_weight.shape[1]
                param.data[:, :dim1].copy_(loaded_weight)
            elif "scale" in weight_name:
                param.data.copy_(loaded_weight)
            else:
                dim1 = loaded_weight.shape[1]
                dim2 = loaded_weight.shape[2]
                param.data[:, :dim1, :dim2].copy_(loaded_weight)
            return

        # compressed-tensors checkpoints with packed weights are stored flipped
        # TODO: check self.quant_method.quant_config.quant_format
        # against known CompressionFormat enum values that have this quality
        loaded_weight = (
            loaded_weight.t().contiguous()
            if (
                self.quant_method.__class__.__name__
                == "CompressedTensorsWNA16MoEMethod"
            )
            else loaded_weight
        )

        if shard_id not in ("w13", "w2"):
            raise ValueError(f"shard_id must be ['w13','w2'] but " f"got {shard_id}.")

        # Fetch the dim to shard the parameter/loaded weight
        # based on the shard id. This will be whatever
        # dimension intermediate_size is used.
        SHARD_ID_TO_SHARDED_DIM = {"w13": 1, "w2": 2}
        SHARD_ID_TO_SHARDED_DIM_TRANSPOSE = {"w13": 2, "w2": 1}

        expert_data = param.data
        is_bias = expert_data.dim() == 2

        # is_transposed: if the dim to shard the weight
        # should be flipped. Required by GPTQ, compressed-tensors
        # should be whatever dimension intermediate_size is
        is_transposed = getattr(param, "is_transposed", False)

        if self.use_triton_kernels:
            is_transposed = True
        shard_dim = (
            SHARD_ID_TO_SHARDED_DIM[shard_id]
            if not is_transposed
            else SHARD_ID_TO_SHARDED_DIM_TRANSPOSE[shard_id]
        )

        # Case model weights
        if "weight" in weight_name:
            self._load_model_weight_or_group_weight_scale(
                shard_id=shard_id,
                shard_dim=shard_dim,
                loaded_weight=loaded_weight,
                expert_data=expert_data,
                tp_rank=tp_rank,
                is_bias=is_bias,
            )
            return
        else:
            logging.warning(
                f"Unsupported weight_name {weight_name} for FusedMoE weight_loader_fused. Nothing is loaded."
            )

    def forward(self, hidden_states: torch.Tensor, topk_output: TopKOutput):
        origin_hidden_states_dim = hidden_states.shape[-1]
        assert self.quant_method is not None

        if self.moe_ep_size > 1 and not self.enable_flashinfer_cutlass_moe:
            if self.expert_map_cpu is not None and self.expert_map_gpu is None:
                # If we are in EP mode, we need to move the expert map to GPU.
                self.expert_map_gpu = self.expert_map_cpu.to(device="cuda")

        if self.expert_map_gpu is not None:
            if TopKOutputChecker.format_is_standard(topk_output):
                topk_output = topk_output._replace(
                    topk_ids=self.expert_map_gpu[topk_output.topk_ids]
                )
            elif TopKOutputChecker.format_is_triton_kernel(topk_output):
                raise NotImplementedError()

        # Matrix multiply.
        with use_symmetric_memory(get_tp_group()) as sm:

            final_hidden_states = self.quant_method.apply(
                layer=self,
                x=hidden_states,
                topk_output=topk_output,
                moe_runner_config=self.moe_runner_config,
            )
            sm.tag(final_hidden_states)

        final_hidden_states = final_hidden_states[
            ..., :origin_hidden_states_dim
        ].contiguous()

        if self.reduce_results and (self.moe_tp_size > 1 or self.moe_ep_size > 1):
>>>>>>> origin/main
            final_hidden_states = tensor_model_parallel_all_reduce(final_hidden_states)

        return final_hidden_states

    @classmethod
    def make_expert_params_mapping(
        cls,
        ckpt_gate_proj_name: str,
        ckpt_down_proj_name: str,
        ckpt_up_proj_name: str,
        num_experts: int,
    ) -> List[Tuple[str, str, int, str]]:

        return [
            # (param_name, weight_name, expert_id, shard_id)
            (
                (
                    "experts.w13_"
                    if weight_name in [ckpt_gate_proj_name, ckpt_up_proj_name]
                    else "experts.w2_"
                ),
                f"experts.{expert_id}.{weight_name}.",
                expert_id,
                shard_id,
            )
            for expert_id in range(num_experts)
            for shard_id, weight_name in [
                ("w1", ckpt_gate_proj_name),
                ("w2", ckpt_down_proj_name),
                ("w3", ckpt_up_proj_name),
            ]
        ]

<<<<<<< HEAD
    def _load_fp8_scale(
        self,
        param: torch.nn.Parameter,
        loaded_weight: torch.Tensor,
        weight_name: str,
        shard_id: str,
        expert_id: int,
    ) -> None:
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
            if shard_id in ("w1", "w3"):
                # We have to keep the weight scales of w1 and w3 because
                # we need to re-quantize w1/w3 weights after weight loading.
                idx = 0 if shard_id == "w1" else 1
                param_data[expert_id][idx] = loaded_weight
            # If we are in the row parallel case (down_proj)
            else:
                param_data[expert_id] = loaded_weight
=======
    @classmethod
    def make_expert_params_mapping_fused(
        cls,
        ckpt_gate_up_proj_name: str,
        ckpt_down_proj_name: str,
        ckpt_gate_up_proj_bias_name: str,
        ckpt_down_proj_bias_name: str,
    ):
        return [
            ("experts.w13_weight", f"experts.{ckpt_gate_up_proj_name}", "w13"),
            (
                "experts.w13_weight_bias",
                f"experts.{ckpt_gate_up_proj_bias_name}",
                "w13",
            ),
            ("experts.w2_weight", f"experts.{ckpt_down_proj_name}", "w2"),
            ("experts.w2_weight_bias", f"experts.{ckpt_down_proj_bias_name}", "w2"),
        ]

    @classmethod
    def make_expert_params_mapping_fused_mxfp4(
        cls,
        ckpt_gate_up_proj_name: str,
        ckpt_down_proj_name: str,
        ckpt_gate_up_proj_bias_name: str,
        ckpt_down_proj_bias_name: str,
        ckpt_gate_up_proj_scale_name: str,
        ckpt_down_proj_scale_name: str,
    ):
        return [
            ("experts.w13_weight", f"experts.{ckpt_gate_up_proj_name}", "w13"),
            (
                "experts.w13_weight_bias",
                f"experts.{ckpt_gate_up_proj_bias_name}",
                "w13",
            ),
            ("experts.w2_weight", f"experts.{ckpt_down_proj_name}", "w2"),
            ("experts.w2_weight_bias", f"experts.{ckpt_down_proj_bias_name}", "w2"),
            (
                "experts.w13_weight_scale",
                f"experts.{ckpt_gate_up_proj_scale_name}",
                "w13",
            ),
            ("experts.w2_weight_scale", f"experts.{ckpt_down_proj_scale_name}", "w2"),
        ]

    @classmethod
    def make_expert_input_scale_params_mapping(
        cls,
        num_experts: int,
    ) -> List[Tuple[str, str, int, str]]:
        # (param_name, weight_name, expert_id, shard_id)
        return [
            (
                "experts.w13_" if shard_id in ["w1", "w3"] else "experts.w2_",
                f"experts.{expert_id}.{shard_id}.",
                expert_id,
                shard_id,
            )
            for expert_id in range(num_experts)
            for shard_id in ["w1", "w2", "w3"]
        ]

    def should_fuse_routed_scaling_factor_in_topk(self):
        return isinstance(self.quant_method, ModelOptNvFp4FusedMoEMethod) or (
            isinstance(self.quant_method, Fp8MoEMethod)
            and self.quant_method.use_cutlass_fused_experts_fp8
        )


class FlashInferFusedMoE(FusedMoE):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_flashinfer_trtllm_moe = should_use_flashinfer_trtllm_moe()

    def forward(self, hidden_states: torch.Tensor, topk_output: TopKOutput):
        assert self.use_flashinfer_trtllm_moe
        assert (
            self.moe_runner_config.activation == "silu"
        ), "Only silu is supported for flashinfer blockscale fp8 moe"
        assert self.quant_method is not None
        assert (
            topk_output.topk_config.renormalize
        ), "Renormalize is required for flashinfer blockscale fp8 moe"
        assert (
            self.num_fused_shared_experts == 0
        ), "Fused shared experts are not supported for flashinfer blockscale fp8 moe"

        assert TopKOutputChecker.format_is_bypassed(topk_output)

        # Matrix multiply.
        final_hidden_states = self.quant_method.apply_with_router_logits(
            layer=self,
            x=hidden_states,
            topk_output=topk_output,
            moe_runner_config=self.moe_runner_config,
        )

        if self.reduce_results and (self.moe_tp_size > 1 or self.moe_ep_size > 1):
            final_hidden_states = tensor_model_parallel_all_reduce(final_hidden_states)

        return final_hidden_states


class FlashInferFP4MoE(FusedMoE):
    """FP4 TRTLLM MoE implementation using FlashInfer."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # ---------------------------------------------------------------------
    # Helper: quantize hidden states to FP4 each forward pass
    # ---------------------------------------------------------------------
    def _quantize_hidden_states_fp4(self, hidden_states: torch.Tensor):
        """
        Quantize hidden states using global scale factor from quantization method.

        Global scale factor is set by ModelOptNvFp4FusedMoEMethod during weight loading.
        Only block scales are computed at runtime for efficiency.

        Returns (packed_fp4_uint8, scale_float8_e4m3fn_runtime, global_scale_float32)
        """

        # flashinfer.fp4_quantize returns (packed_uint8, scale_fp8)
        # Only the block scales are computed at runtime
        hs_fp4_bytes, hs_sf_bytes = fp4_quantize(
            hidden_states,
            self.w13_input_scale_quant,
            16,  # sf_vec_size
            False,  # use_ue8m0
            False,  # is_sf_swizzled_layout
        )

        hs_fp4 = hs_fp4_bytes.reshape(
            hidden_states.shape[0], hidden_states.shape[1] // 2
        )
        hs_sf = hs_sf_bytes.view(torch.float8_e4m3fn).reshape(-1)

        return hs_fp4, hs_sf

    def forward(self, hidden_states: torch.Tensor, topk_output: TopKOutput):
        """Forward pass using FP4 TRTLLM kernel.

        Args:
            hidden_states: Input tensor
            topk_output: TopKOutput object with Bypassed format
        """
        assert isinstance(self.quant_method, ModelOptNvFp4FusedMoEMethod)

        assert TopKOutputChecker.format_is_bypassed(topk_output)

        router_logits = topk_output.router_logits
        topk_config = topk_output.topk_config

        hs_fp4, hs_scale_linear = self._quantize_hidden_states_fp4(hidden_states)

        router_logits = router_logits.to(torch.float32)

        result = trtllm_fp4_block_scale_moe(
            routing_logits=router_logits,
            routing_bias=topk_config.correction_bias.to(hidden_states.dtype),
            hidden_states=hs_fp4,
            hidden_states_scale=hs_scale_linear.view(torch.float8_e4m3fn).flatten(),
            gemm1_weights=self.gemm1_weights_fp4_shuffled.data,
            gemm1_weights_scale=self.gemm1_scales_fp4_shuffled.data.view(
                torch.float8_e4m3fn
            ),
            gemm1_bias=None,
            gemm1_alpha=None,
            gemm1_beta=None,
            gemm1_clamp_limit=None,
            gemm2_weights=self.gemm2_weights_fp4_shuffled.data,
            gemm2_weights_scale=self.gemm2_scales_fp4_shuffled.data.view(
                torch.float8_e4m3fn
            ),
            gemm2_bias=None,
            output1_scale_scalar=self.g1_scale_c.data,
            output1_scale_gate_scalar=self.g1_alphas.data,
            output2_scale_scalar=self.g2_alphas.data,
            num_experts=self.num_experts,
            top_k=topk_config.top_k,
            n_group=topk_config.num_expert_group,
            topk_group=topk_config.topk_group,
            intermediate_size=self.intermediate_size_per_partition,
            local_expert_offset=self.moe_ep_rank * self.num_local_experts,
            local_num_experts=self.num_local_experts,
            routed_scaling_factor=self.moe_runner_config.routed_scaling_factor,
            tile_tokens_dim=_get_tile_tokens_dim(
                hidden_states.shape[0], topk_config.top_k, self.num_local_experts
            ),
            routing_method_type=RoutingMethodType.DeepSeekV3,
            do_finalize=True,
        )[0]

        return result


def get_fused_moe_impl_class():
    """Factory function to get the appropriate FusedMoE implementation class."""
    if should_use_flashinfer_trtllm_moe() and _is_fp4_quantization_enabled():
        # Use FP4 variant when FP4 quantization is enabled
        return FlashInferFP4MoE
    elif should_use_flashinfer_trtllm_moe():
        # Use regular FlashInfer variant for non-FP4 FlashInfer cases
        return FlashInferFusedMoE
    else:
        # Default case
        return FusedMoE
>>>>>>> origin/main
