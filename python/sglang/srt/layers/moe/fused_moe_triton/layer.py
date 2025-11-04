# Adapted from https://github.com/vllm-project/vllm/blob/a6221a144af772fd1a68fe7e627935dc53e81738/vllm/model_executor/layers/fused_moe/layer.py

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
from sglang.srt.layers.dp_attention import is_allocation_symmetric
from sglang.srt.layers.moe import (
    MoeRunnerConfig,
    get_deepep_mode,
    get_moe_a2a_backend,
    get_moe_runner_backend,
)
from sglang.srt.layers.moe.token_dispatcher import CombineInput, DispatchOutput
from sglang.srt.layers.moe.token_dispatcher.base import BaseDispatcher
from sglang.srt.layers.moe.token_dispatcher.standard import (
    StandardDispatcher,
    StandardDispatchOutput,
)
from sglang.srt.layers.moe.topk import TopKOutput, TopKOutputChecker
from sglang.srt.layers.quantization.base_config import (
    FusedMoEMethodBase,
    QuantizationConfig,
)
from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors_moe import (
    CompressedTensorsWNA16AMXEPMoEMethod,
    CompressedTensorsWNA16AMXMoEMethod,
    CompressedTensorsWNA16MoEMethod,
)
from sglang.srt.layers.quantization.fp8 import Fp8MoEMethod
from sglang.srt.layers.quantization.modelopt_quant import ModelOptNvFp4FusedMoEMethod
from sglang.srt.layers.quantization.unquant import UnquantizedFusedMoEMethod
from sglang.srt.model_loader.weight_utils import narrow_padded_param_and_loaded_weight
from sglang.srt.two_batch_overlap import MaybeTboDeepEPDispatcher
from sglang.srt.utils import (
    cpu_has_amx_support,
    get_bool_env_var,
    is_cpu,
    is_flashinfer_available,
    is_hip,
    round_up,
)

if is_flashinfer_available():
    from flashinfer import RoutingMethodType, fp4_quantize

# Try to import FP4 TRTLLM function if flashinfer is available
trtllm_fp4_block_scale_moe = None
if get_moe_runner_backend().is_flashinfer_trtllm():
    try:
        from flashinfer.fused_moe import trtllm_fp4_block_scale_moe
    except ImportError:
        trtllm_fp4_block_scale_moe = None

_is_hip = is_hip()
_is_cpu_amx_available = cpu_has_amx_support()
_is_cpu = is_cpu()

logger = logging.getLogger(__name__)


def create_moe_dispatcher(moe_runner_config: MoeRunnerConfig) -> BaseDispatcher:
    a2a_backend = get_moe_a2a_backend()
    if a2a_backend.is_none():
        return StandardDispatcher(moe_runner_config)
    elif a2a_backend.is_deepep() or a2a_backend.is_mooncake():
        return MaybeTboDeepEPDispatcher(
            group=get_tp_group().device_group,
            router_topk=moe_runner_config.top_k,
            permute_fusion=True,
            num_experts=moe_runner_config.num_experts,
            num_local_experts=moe_runner_config.num_local_experts,
            hidden_size=moe_runner_config.hidden_size,
            params_dtype=moe_runner_config.params_dtype,
            deepep_mode=get_deepep_mode(),
            async_finish=True,
            return_recv_hook=True,
        )
    else:
        raise NotImplementedError(f"Unsupported a2a backend: {a2a_backend}")


class FusedMoeWeightScaleSupported(Enum):
    TENSOR = "tensor"
    CHANNEL = "channel"
    GROUP = "group"
    BLOCK = "block"


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
        reduce_results: Whether to apply all_reduce on the output of the layer
        quant_config: Quantization configuration.
        inplace: suggestion to compute inplace (modify input activation).
    """

    def __init__(
        self,
        num_experts: int,
        hidden_size: int,
        intermediate_size: int,
        layer_id: int,
        top_k: Optional[int] = None,
        num_fused_shared_experts: int = 0,
        params_dtype: Optional[torch.dtype] = None,
        reduce_results: bool = False,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        activation: str = "silu",
        apply_router_weight_on_input: bool = False,
        use_presharded_weights: bool = False,
        inplace: bool = True,
        no_combine: bool = False,
        routed_scaling_factor: Optional[float] = None,
        gemm1_alpha: Optional[float] = None,
        gemm1_clamp_limit: Optional[float] = None,
        use_weight_loader_fused: bool = False,
        with_bias=False,
    ):
        super().__init__()
        if params_dtype is None:
            params_dtype = torch.get_default_dtype()

        self.layer_id = layer_id
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.num_fused_shared_experts = num_fused_shared_experts

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

        assert intermediate_size % self.moe_tp_size == 0
        self.intermediate_size_per_partition = intermediate_size // self.moe_tp_size
        self.reduce_results = reduce_results
        self.use_presharded_weights = use_presharded_weights

        self.use_triton_kernels = get_moe_runner_backend().is_triton_kernels()

        self.quant_config = quant_config
        self.use_flashinfer_mxfp4_moe = get_moe_runner_backend().is_flashinfer_mxfp4()
        # TODO maybe we should remove this `if`, since `Mxfp4MoEMethod` does another round-up logic
        if (
            self.quant_config is not None
            and self.quant_config.get_name() == "mxfp4"
            and self.use_flashinfer_mxfp4_moe
        ):
            hidden_size = round_up(hidden_size, 256)
        self.hidden_size = hidden_size

        self.moe_runner_config = MoeRunnerConfig(
            num_experts=num_experts,
            num_local_experts=self.num_local_experts,
            hidden_size=hidden_size,
            intermediate_size_per_partition=self.intermediate_size_per_partition,
            layer_id=layer_id,
            top_k=top_k,
            num_fused_shared_experts=num_fused_shared_experts,
            params_dtype=params_dtype,
            activation=activation,
            apply_router_weight_on_input=apply_router_weight_on_input,
            inplace=inplace,
            no_combine=no_combine,
            routed_scaling_factor=routed_scaling_factor,
            gemm1_alpha=gemm1_alpha,
            gemm1_clamp_limit=gemm1_clamp_limit,
        )

        self.quant_method: Optional[FusedMoEMethodBase] = None
        if quant_config is not None:
            self.quant_method = quant_config.get_quant_method(self, prefix)
        if self.quant_method is None:
            self.quant_method = UnquantizedFusedMoEMethod(self.use_triton_kernels)

        self.quant_method.create_weights(
            layer=self,
            num_experts=self.num_local_experts,
            hidden_size=hidden_size,
            intermediate_size_per_partition=self.intermediate_size_per_partition,
            params_dtype=params_dtype,
            weight_loader=(
                self.weight_loader
                if not use_weight_loader_fused
                else self.weight_loader_fused
            ),
            intermediate_size_full=intermediate_size,
            top_k=top_k,
            with_bias=with_bias,
        )

        self.quant_method.create_moe_runner(self, self.moe_runner_config)
        self.dispatcher = create_moe_dispatcher(self.moe_runner_config)

        self.should_fuse_routed_scaling_factor_in_topk = isinstance(
            self.quant_method, ModelOptNvFp4FusedMoEMethod
        ) or (
            isinstance(self.quant_method, Fp8MoEMethod)
            and get_moe_runner_backend().is_cutlass()
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
        loaded_weight: torch.Tensor,
        tp_rank: int,
        is_bias: bool = False,
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
                is_bias=is_bias,
            )
        elif shard_id in ("w1", "w3", "w13"):
            self._load_w13(
                shard_id=shard_id,
                shard_dim=shard_dim,
                loaded_weight=loaded_weight,
                expert_data=expert_data,
                tp_rank=tp_rank,
                is_bias=is_bias,
            )

    def _load_per_channel_weight_scale(
        self,
        expert_data: torch.Tensor,
        shard_dim: int,
        shard_id: str,
        loaded_weight: torch.Tensor,
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
        loaded_weight: torch.Tensor,
        tp_rank: int,
        is_bias: bool = False,
    ):

        # Index the loaded weight for tp sharding.
        # gate_up_proj: "MergedColumnParallel", so tp sharding on output_dim
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

        # Narrow parameter and load.
        # w1, gate_proj: Load into first logical weight of w13.
        # w3, up_proj: Load into second logical weight of w13.
        # trtllm cutlass kernel assumes differently
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
                if not is_bias and self.use_triton_kernels:
                    # do not transpose for bias
                    loaded_weight = loaded_weight.transpose(-2, -1)
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

        # Index the loaded weight for tp sharding.
        # down_proj: "RowParallel" so tp sharding on input_dim
        # Narrow parameter and load.
        if is_bias:
            # this expert_data is a bias, not weight,
            # for w2_weight_bias in TP, it does not need to be sharded
            shard_size = expert_data.shape[-1]
        else:
            # this parameter is a weight matrix
            # for w2 in TP, it shards the input_features, i.e., shard_dim=2
            shard_size = expert_data.shape[shard_dim]

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
            if not is_bias and not self.use_presharded_weights:
                if self.use_triton_kernels:
                    loaded_weight = loaded_weight.transpose(-2, -1)
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
        loaded_weight: torch.Tensor,
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
        start_idx = self.moe_ep_rank * self.num_local_experts
        end_idx = (self.moe_ep_rank + 1) * self.num_local_experts
        if start_idx <= expert_id < end_idx:
            return expert_id - start_idx
        else:
            return -1

    def weight_loader(
        self,
        param: torch.nn.Parameter,
        loaded_weight: torch.Tensor,
        weight_name: str,
        shard_id: str,
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
        # WARN: This makes the `expert_id` mean "local" and "global" in different cases
        if not getattr(param, "_sglang_require_global_experts", False):
            expert_id = self._map_global_expert_id_to_local_expert_id(expert_id)
            if expert_id == -1:
                return

        if isinstance(
            self.quant_method,
            (
                CompressedTensorsWNA16MoEMethod,
                CompressedTensorsWNA16AMXMoEMethod,
                CompressedTensorsWNA16AMXEPMoEMethod,
            ),
        ):
            if self.quant_method.num_gpu_experts != -1:
                if expert_id >= self.quant_method.num_gpu_experts:
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

        # compressed-tensors checkpoints with packed weights are stored flipped
        # TODO (mgoin): check self.quant_method.quant_config.quant_format
        # against known CompressionFormat enum values that have this quality
        loaded_weight = (
            loaded_weight.t().contiguous()
            if (
                self.quant_method.__class__.__name__
                in [
                    "CompressedTensorsWNA16MarlinMoEMethod",
                    "CompressedTensorsWNA16MoEMethod",
                    "CompressedTensorsWNA16AMXMoEMethod",
                    "CompressedTensorsWNA16AMXEPMoEMethod",
                ]
            )
            else loaded_weight
        )

        if shard_id not in ("w1", "w2", "w3"):
            raise ValueError(
                f"shard_id must be ['w1','w2','w3'] but " f"got {shard_id}."
            )

        # Flashinfer assumes w31 format for w13_weight. Same for the scales.
        if get_moe_runner_backend().is_flashinfer_trtllm() and (
            isinstance(self.quant_method, ModelOptNvFp4FusedMoEMethod)
            or isinstance(self.quant_method, Fp8MoEMethod)
        ):
            shard_id = {"w1": "w3", "w3": "w1", "w2": "w2"}[shard_id]

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
        if self.use_triton_kernels:
            is_transposed = True
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
                (
                    "compressed" in self.quant_method.__class__.__name__.lower()
                    or "w4afp8" in self.quant_config.get_name()
                )
                and (param.data[expert_id] != 1).any()
                and ((param.data[expert_id] - loaded_weight).abs() > 1e-5).any()
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
        if "scale" in weight_name or "zero" in weight_name or "offset" in weight_name:
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

    def forward(self, hidden_states: torch.Tensor, topk_output: TopKOutput, **kwargs):
        origin_hidden_states_dim = hidden_states.shape[-1]
        assert self.quant_method is not None

        dispatch_output = self.dispatcher.dispatch(
            hidden_states=hidden_states, topk_output=topk_output
        )

        combine_input = self.run_moe_core(
            dispatch_output=dispatch_output,
            **kwargs,
        )

        with use_symmetric_memory(
            get_tp_group(), disabled=not is_allocation_symmetric()
        ):
            final_hidden_states = self.dispatcher.combine(combine_input=combine_input)

            # TODO: should we add some conditions here?
            final_hidden_states = final_hidden_states[
                ..., :origin_hidden_states_dim
            ].contiguous()

        if self.reduce_results and (self.moe_tp_size > 1 or self.moe_ep_size > 1):
            final_hidden_states = tensor_model_parallel_all_reduce(final_hidden_states)

        return final_hidden_states

    def run_moe_core(self, dispatch_output: DispatchOutput, **kwargs) -> CombineInput:
        # TODO: consider using symmetric memory
        return self.quant_method.apply(
            layer=self,
            dispatch_output=dispatch_output,
            **kwargs,
        )

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


class FlashInferFusedMoE(FusedMoE):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, hidden_states: torch.Tensor, topk_output: TopKOutput):
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
            dispatch_output=StandardDispatchOutput(
                hidden_states=hidden_states, topk_output=topk_output
            ),
        )

        # NOTE for symmetric memory tagging:
        # We do not create the context in this function.
        # Instead, we create the context and tagging inside each FusedMoEMethodBase
        # This can allow fine-grained tagging.

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

        with use_symmetric_memory(
            get_tp_group(), disabled=not is_allocation_symmetric()
        ):
            symm_output = torch.empty_like(hidden_states)
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
            tile_tokens_dim=None,
            routing_method_type=RoutingMethodType.DeepSeekV3,
            do_finalize=True,
            output=symm_output,
        )[0]

        return result
