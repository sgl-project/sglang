import logging
from typing import Callable, List, Optional, Tuple

import torch

# TODO: use deep_gemm masked kernel after low latency dispatch
# import deep_gemm
# from deep_gemm import (
#     get_col_major_tma_aligned_tensor,
#     m_grouped_gemm_fp8_fp8_bf16_nt_masked,
# )
from torch.nn import Module

from sglang.srt.custom_op import CustomOp
from sglang.srt.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from sglang.srt.layers.moe.ep_moe.kernels import (
    gelu_and_mul_triton_kernel,
    grouped_gemm_triton,
    post_reorder_triton_kernel,
    pre_reorder_triton_kernel,
    run_moe_ep_preproess,
    silu_and_mul_triton_kernel,
)
from sglang.srt.layers.moe.fused_moe_triton import FusedMoeWeightScaleSupported
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoEMethodBase
from sglang.srt.layers.moe.topk import select_experts
from sglang.srt.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from sglang.srt.layers.quantization.fp8 import Fp8Config, Fp8MoEMethod
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.utils import is_cuda, is_hip, set_weight_attrs

_is_cuda = is_cuda()

if _is_cuda:
    from sglang.srt.custom_op import scaled_fp8_quant as sgl_scaled_fp8_quant
else:
    from vllm import _custom_ops as vllm_ops


logger = logging.getLogger(__name__)

_is_hip = is_hip()

_buffer = None


class GroupedGemmRunner(torch.nn.Module):
    flashinfer_gemm_warpper = None

    def __init__(self, device, use_flashinfer: bool = False):
        super().__init__()
        self.device = device
        self.use_flashinfer = use_flashinfer
        if self.use_flashinfer and GroupedGemmRunner.flashinfer_gemm_warpper is None:
            GroupedGemmRunner._init_flashinfer_wrapper(device)

    @classmethod
    def _init_flashinfer_wrapper(cls, device):
        from flashinfer import SegmentGEMMWrapper

        workspace_buffer = torch.empty(
            128 * 1024 * 1024, dtype=torch.int8, device=device
        )
        cls.flashinfer_gemm_warpper = SegmentGEMMWrapper(workspace_buffer)

    # c = a * b
    def forward(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        c: torch.Tensor,
        batch_size: int,
        weight_column_major: bool,
        seg_indptr: Optional[torch.Tensor] = None,
        weight_indices: Optional[torch.Tensor] = None,
        use_fp8_w8a8: bool = False,
        scale_a: torch.Tensor = None,
        scale_b: torch.Tensor = None,
        block_shape: Optional[List[int]] = None,
    ):
        if self.use_flashinfer:
            # TODO: flashinfer
            assert False
            assert GroupedGemmRunner.flashinfer_gemm_warpper is not None
            c = GroupedGemmRunner.flashinfer_gemm_warpper.run(
                x=a,
                weights=b,
                batch_size=batch_size,
                weight_column_major=weight_column_major,
                seg_indptr=seg_indptr,
                weight_indices=weight_indices,
            )
        else:
            assert weight_column_major == True
            c = grouped_gemm_triton(
                a,
                b,
                c,
                batch_size,
                weight_column_major,
                seg_indptr,
                weight_indices,
                use_fp8_w8a8,
                scale_a,
                scale_b,
                block_shape=block_shape,
            )
        return c


class EPMoE(torch.nn.Module):
    """
    MoE Expert Parallel Impl


    """

    def __init__(
        self,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        params_dtype: Optional[torch.dtype] = None,
        renormalize: bool = True,
        use_grouped_topk: bool = False,
        num_expert_group: Optional[int] = None,
        topk_group: Optional[int] = None,
        quant_config: Optional[QuantizationConfig] = None,
        tp_size: Optional[int] = None,
        prefix: str = "",
        correction_bias: Optional[torch.Tensor] = None,
        custom_routing_function: Optional[Callable] = None,
        activation: str = "silu",
    ):
        super().__init__()

        if params_dtype is None:
            params_dtype = torch.get_default_dtype()

        self.tp_size = (
            tp_size if tp_size is not None else get_tensor_model_parallel_world_size()
        )
        self.tp_rank = get_tensor_model_parallel_rank()

        self.num_experts = num_experts
        assert self.num_experts % self.tp_size == 0
        self.num_experts_per_partition = self.num_experts // self.tp_size
        self.start_expert_id = self.tp_rank * self.num_experts_per_partition
        self.end_expert_id = self.start_expert_id + self.num_experts_per_partition - 1

        self.top_k = top_k
        self.intermediate_size = intermediate_size
        self.renormalize = renormalize
        self.use_grouped_topk = use_grouped_topk
        if self.use_grouped_topk:
            assert num_expert_group is not None and topk_group is not None
        self.num_expert_group = num_expert_group
        self.topk_group = topk_group
        self.correction_bias = correction_bias
        self.custom_routing_function = custom_routing_function
        self.activation = activation

        if quant_config is None:
            self.quant_method: Optional[QuantizeMethodBase] = UnquantizedEPMoEMethod()
            self.use_fp8_w8a8 = False
            self.use_block_quant = False
            self.block_shape = None
            self.activation_scheme = None
        else:
            self.quant_method: Optional[QuantizeMethodBase] = Fp8EPMoEMethod(
                quant_config
            )
            self.use_fp8_w8a8 = True
            self.use_block_quant = getattr(self.quant_method, "block_quant", False)
            self.block_shape = (
                self.quant_method.quant_config.weight_block_size
                if self.use_block_quant
                else None
            )
            self.fp8_dtype = torch.float8_e4m3fn
            self.activation_scheme = quant_config.activation_scheme

        self.quant_method.create_weights(
            layer=self,
            num_experts_per_partition=self.num_experts_per_partition,
            hidden_size=hidden_size,
            intermediate_size=self.intermediate_size,
            params_dtype=params_dtype,
            weight_loader=self.weight_loader,
        )

        self.grouped_gemm_runner = None

    def forward(self, hidden_states: torch.Tensor, router_logits: torch.Tensor):
        assert self.quant_method is not None

        if self.grouped_gemm_runner is None:
            self.grouped_gemm_runner = GroupedGemmRunner(
                hidden_states.device,
                use_flashinfer=False,  # TODO: use flashinfer
            )

        topk_weights, topk_ids = select_experts(
            hidden_states=hidden_states,
            router_logits=router_logits,
            top_k=self.top_k,
            use_grouped_topk=self.use_grouped_topk,
            renormalize=self.renormalize,
            topk_group=self.topk_group,
            num_expert_group=self.num_expert_group,
            correction_bias=self.correction_bias,
            custom_routing_function=self.custom_routing_function,
        )

        reorder_topk_ids, src2dst, seg_indptr = run_moe_ep_preproess(
            topk_ids, self.num_experts
        )

        gateup_input = torch.empty(
            (int(hidden_states.shape[0] * self.top_k), hidden_states.shape[1]),
            device=hidden_states.device,
            dtype=(
                self.fp8_dtype
                if (self.use_fp8_w8a8 and not self.use_block_quant)
                else hidden_states.dtype
            ),
        )
        if self.activation_scheme == "dynamic" and not self.use_block_quant:
            max_value = (
                torch.max(hidden_states)
                .repeat(self.num_experts_per_partition)
                .to(torch.float32)
            )
            self.w13_input_scale = max_value / torch.finfo(self.fp8_dtype).max

        # PreReorder
        pre_reorder_triton_kernel[(hidden_states.shape[0],)](
            hidden_states,
            gateup_input,
            src2dst,
            topk_ids,
            self.w13_input_scale,
            self.start_expert_id,
            self.end_expert_id,
            self.top_k,
            hidden_states.shape[1],
            BLOCK_SIZE=512,
        )

        seg_indptr_cur_rank = seg_indptr[self.start_expert_id : self.end_expert_id + 2]
        weight_indices_cur_rank = torch.arange(
            0,
            self.num_experts_per_partition,
            device=hidden_states.device,
            dtype=torch.int64,
        )
        # GroupGemm-0
        gateup_output = torch.empty(
            gateup_input.shape[0],
            self.w13_weight.shape[1],
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        gateup_output = self.grouped_gemm_runner(
            a=gateup_input,
            b=self.w13_weight,
            c=gateup_output,
            batch_size=self.num_experts_per_partition,
            weight_column_major=True,
            seg_indptr=seg_indptr_cur_rank,
            weight_indices=weight_indices_cur_rank,
            use_fp8_w8a8=self.use_fp8_w8a8,
            scale_a=self.w13_input_scale,
            scale_b=(
                self.w13_weight_scale_inv
                if self.use_block_quant
                else self.w13_weight_scale
            ),
            block_shape=self.block_shape,
        )

        # Act
        down_input = torch.empty(
            gateup_output.shape[0],
            gateup_output.shape[1] // 2,
            device=gateup_output.device,
            dtype=(
                self.fp8_dtype
                if (self.use_fp8_w8a8 and not self.use_block_quant)
                else hidden_states.dtype
            ),
        )
        if self.w2_input_scale is None and not self.use_block_quant:
            self.w2_input_scale = torch.ones(
                self.num_experts_per_partition,
                dtype=torch.float32,
                device=hidden_states.device,
            )

        if self.activation == "silu":
            silu_and_mul_triton_kernel[(gateup_output.shape[0],)](
                gateup_output,
                down_input,
                gateup_output.shape[1],
                reorder_topk_ids,
                self.w2_input_scale,
                self.start_expert_id,
                self.end_expert_id,
                BLOCK_SIZE=512,
            )
        elif self.activation == "gelu":
            gelu_and_mul_triton_kernel[(gateup_output.shape[0],)](
                gateup_output,
                down_input,
                gateup_output.shape[1],
                reorder_topk_ids,
                self.w2_input_scale,
                self.start_expert_id,
                self.end_expert_id,
                BLOCK_SIZE=512,
            )
        else:
            raise ValueError(f"Unsupported activation: {self.activation=}")

        # GroupGemm-1
        down_output = torch.empty(
            down_input.shape[0],
            self.w2_weight.shape[1],
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        down_output = self.grouped_gemm_runner(
            a=down_input,
            b=self.w2_weight,
            c=down_output,
            batch_size=self.num_experts_per_partition,
            weight_column_major=True,
            seg_indptr=seg_indptr_cur_rank,
            weight_indices=weight_indices_cur_rank,
            use_fp8_w8a8=self.use_fp8_w8a8,
            scale_a=self.w2_input_scale,
            scale_b=(
                self.w2_weight_scale_inv
                if self.use_block_quant
                else self.w2_weight_scale
            ),
            block_shape=self.block_shape,
        )

        # PostReorder
        output = torch.empty_like(hidden_states)
        post_reorder_triton_kernel[(hidden_states.size(0),)](
            down_output,
            output,
            src2dst,
            topk_ids,
            topk_weights,
            self.start_expert_id,
            self.end_expert_id,
            self.top_k,
            hidden_states.size(1),
            BLOCK_SIZE=512,
        )
        return output

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

    def weight_loader(
        self,
        param: torch.nn.Parameter,
        loaded_weight: torch.Tensor,
        weight_name: str,
        shard_id: str,
        expert_id: int,
    ) -> None:
        if expert_id < self.start_expert_id or expert_id > self.end_expert_id:
            return
        expert_id = expert_id - self.start_expert_id

        if shard_id not in ("w1", "w2", "w3"):
            raise ValueError(
                f"shard_id must be ['w1','w2','w3'] but " f"got {shard_id}."
            )

        # Special case for fp8 scales.
        if "scale" in weight_name:
            self._load_fp8_scale(
                param.data,
                loaded_weight,
                weight_name,
                shard_id,
                expert_id,
            )
            return

        if shard_id == "w2":
            param.data[expert_id] = loaded_weight
        elif shard_id == "w1":
            param.data[expert_id][: self.intermediate_size, :] = loaded_weight
        elif shard_id == "w3":
            param.data[expert_id][self.intermediate_size :, :] = loaded_weight
        else:
            raise ValueError(f"Expected shard_id w1,w2 or w3 but got {shard_id}")

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
            if self.use_block_quant:
                block_n, block_k = self.block_shape[0], self.block_shape[1]
                if shard_id == "w1":
                    param_data[expert_id][
                        : (self.intermediate_size + block_n - 1) // block_n, :
                    ] = loaded_weight
                elif shard_id == "w3":
                    param_data[expert_id][
                        (self.intermediate_size + block_n - 1) // block_n :, :
                    ] = loaded_weight
                else:  # w2
                    param_data[expert_id] = loaded_weight
            # If we are in merged column case (gate_up_proj)
            else:
                if shard_id in ("w1", "w3"):
                    # We have to keep the weight scales of w1 and w3 because
                    # we need to re-quantize w1/w3 weights after weight loading.
                    idx = 0 if shard_id == "w1" else 1
                    param_data[expert_id][idx] = loaded_weight

                # If we are in the row parallel case (down_proj)
                else:
                    param_data[expert_id] = loaded_weight


class UnquantizedEPMoEMethod(FusedMoEMethodBase, CustomOp):

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts_per_partition: int,
        hidden_size: int,
        intermediate_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        # Fused gate_up_proj (column parallel)
        w13_weight = torch.nn.Parameter(
            torch.empty(
                num_experts_per_partition,
                2 * intermediate_size,
                hidden_size,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        # down_proj (row parallel)
        w2_weight = torch.nn.Parameter(
            torch.empty(
                num_experts_per_partition,
                hidden_size,
                intermediate_size,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        # scale
        ones_tensor = torch.ones(num_experts_per_partition, dtype=torch.float32)
        w13_input_scale = torch.nn.Parameter(
            ones_tensor,
            requires_grad=False,
        )
        layer.register_parameter("w13_input_scale", w13_input_scale)
        set_weight_attrs(w13_input_scale, extra_weight_attrs)

        w2_input_scale = torch.nn.Parameter(
            ones_tensor,
            requires_grad=False,
        )
        layer.register_parameter("w2_input_scale", w2_input_scale)
        set_weight_attrs(w2_input_scale, extra_weight_attrs)

        w13_weight_scale = torch.nn.Parameter(
            ones_tensor,
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_scale", w13_weight_scale)
        set_weight_attrs(w13_weight_scale, extra_weight_attrs)

        w2_weight_scale = torch.nn.Parameter(
            ones_tensor,
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_scale", w2_weight_scale)
        set_weight_attrs(w2_weight_scale, extra_weight_attrs)

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
        custom_routing_function: Optional[Callable] = None,
    ) -> torch.Tensor:
        raise NotImplementedError


class Fp8EPMoEMethod(Fp8MoEMethod):
    """MoE method for FP8.
    Supports loading FP8 checkpoints with static weight scale and
    dynamic/static activation scale.

    Args:
        quant_config: The quantization config.
    """

    def __init__(self, quant_config: Fp8Config):
        self.quant_config = quant_config
        self.block_quant = self.quant_config.weight_block_size is not None

    def create_weights(
        self,
        layer: Module,
        num_experts_per_partition: int,
        hidden_size: int,
        intermediate_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):

        if self.quant_config.is_checkpoint_fp8_serialized:
            params_dtype = torch.float8_e4m3fn

        tp_size = get_tensor_model_parallel_world_size()
        if self.block_quant:
            block_n, block_k = (
                self.quant_config.weight_block_size[0],
                self.quant_config.weight_block_size[1],
            )
            # NOTE(HandH1998): To ensure proper alignment of the block-wise quantization scales, the output_size of the weights for both the gate and up layers must be divisible by block_n.
            # Required by collum parallel or enabling merged weights
            if intermediate_size % block_n != 0:
                raise ValueError(
                    f"The output_size of gate's and up's weight = "
                    f"{intermediate_size} is not divisible by "
                    f"weight quantization block_n = {block_n}."
                )
            if tp_size > 1:
                # Required by row parallel
                if intermediate_size % block_k != 0:
                    raise ValueError(
                        f"The input_size of down's weight = "
                        f"{intermediate_size} is not divisible by "
                        f"weight quantization block_k = {block_k}."
                    )

        # WEIGHTS
        w13_weight = torch.nn.Parameter(
            torch.empty(
                num_experts_per_partition,
                2 * intermediate_size,
                hidden_size,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w2_weight = torch.nn.Parameter(
            torch.empty(
                num_experts_per_partition,
                hidden_size,
                intermediate_size,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        # WEIGHT_SCALES
        if self.block_quant:
            w13_weight_scale = torch.nn.Parameter(
                torch.ones(
                    num_experts_per_partition,
                    2 * ((intermediate_size + block_n - 1) // block_n),
                    (hidden_size + block_k - 1) // block_k,
                    dtype=torch.float32,
                ),
                requires_grad=False,
            )
            w2_weight_scale = torch.nn.Parameter(
                torch.ones(
                    num_experts_per_partition,
                    (hidden_size + block_n - 1) // block_n,
                    (intermediate_size + block_k - 1) // block_k,
                    dtype=torch.float32,
                ),
                requires_grad=False,
            )
            layer.register_parameter("w13_weight_scale_inv", w13_weight_scale)
            layer.register_parameter("w2_weight_scale_inv", w2_weight_scale)
            assert self.quant_config.activation_scheme == "dynamic"
        else:
            # WEIGHT_SCALES
            # Allocate 2 scales for w1 and w3 respectively.
            w13_weight_scale = torch.nn.Parameter(
                torch.ones(num_experts_per_partition, 2, dtype=torch.float32),
                requires_grad=False,
            )
            layer.register_parameter("w13_weight_scale", w13_weight_scale)

            w2_weight_scale = torch.nn.Parameter(
                torch.ones(num_experts_per_partition, dtype=torch.float32),
                requires_grad=False,
            )
            layer.register_parameter("w2_weight_scale", w2_weight_scale)
        # Add the quantization method used (per tensor/grouped/channel)
        # to ensure the weight scales are loaded in properly
        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.BLOCK.value}
            if self.block_quant
            else {"quant_method": FusedMoeWeightScaleSupported.TENSOR.value}
        )
        # If loading fp8 checkpoint, pass the weight loaders.
        # If loading an fp16 checkpoint, do not (we will quantize in
        #   process_weights_after_loading()
        if self.quant_config.is_checkpoint_fp8_serialized:
            set_weight_attrs(w13_weight_scale, extra_weight_attrs)
            set_weight_attrs(w2_weight_scale, extra_weight_attrs)

        # INPUT_SCALES
        if self.quant_config.activation_scheme == "static":
            if not self.quant_config.is_checkpoint_fp8_serialized:
                raise ValueError(
                    "Found static activation scheme for checkpoint that "
                    "was not serialized fp8."
                )

            w13_input_scale = torch.nn.Parameter(
                torch.ones(num_experts_per_partition, dtype=torch.float32),
                requires_grad=False,
            )
            layer.register_parameter("w13_input_scale", w13_input_scale)
            set_weight_attrs(w13_input_scale, extra_weight_attrs)

            w2_input_scale = torch.nn.Parameter(
                torch.ones(num_experts_per_partition, dtype=torch.float32),
                requires_grad=False,
            )
            layer.register_parameter("w2_input_scale", w2_input_scale)
            set_weight_attrs(w2_input_scale, extra_weight_attrs)

        else:
            layer.w13_input_scale = None
            layer.w2_input_scale = None

    def process_weights_after_loading(self, layer: Module) -> None:

        # If checkpoint is fp16, quantize in place.
        if not self.quant_config.is_checkpoint_fp8_serialized:
            # If rocm, use float8_e4m3fnuz as dtype
            fp8_dtype = torch.float8_e4m3fnuz if _is_hip else torch.float8_e4m3fn
            w13_weight = torch.empty_like(layer.w13_weight.data, dtype=fp8_dtype)
            w2_weight = torch.empty_like(layer.w2_weight.data, dtype=fp8_dtype)

            layer.w13_weight_scale = torch.nn.Parameter(
                torch.ones(
                    layer.num_experts_per_partition,
                    dtype=torch.float32,
                    device=w13_weight.device,
                ),
                requires_grad=False,
            )

            for expert in range(layer.num_experts_per_partition):
                if _is_cuda:
                    w13_weight[expert, :, :], layer.w13_weight_scale[expert] = (
                        sgl_scaled_fp8_quant(layer.w13_weight.data[expert, :, :])
                    )
                    w2_weight[expert, :, :], layer.w2_weight_scale[expert] = (
                        sgl_scaled_fp8_quant(layer.w2_weight.data[expert, :, :])
                    )
                else:
                    w13_weight[expert, :, :], layer.w13_weight_scale[expert] = (
                        vllm_ops.scaled_fp8_quant(layer.w13_weight.data[expert, :, :])
                    )
                    w2_weight[expert, :, :], layer.w2_weight_scale[expert] = (
                        vllm_ops.scaled_fp8_quant(layer.w2_weight.data[expert, :, :])
                    )
            layer.w13_weight = torch.nn.Parameter(w13_weight, requires_grad=False)
            layer.w2_weight = torch.nn.Parameter(w2_weight, requires_grad=False)
            return

        # If checkpoint is fp8, we need to handle that the
        # MoE kernels require single activation scale and single weight
        # scale for w13 per expert.
        else:
            if self.quant_config.activation_scheme == "static":
                if layer.w13_input_scale is None or layer.w2_input_scale is None:
                    raise ValueError(
                        "QuantConfig has static quantization, but found "
                        "activation scales are None."
                    )
                layer.w13_weight_scale = torch.nn.Parameter(
                    torch.max(layer.w13_weight_scale, dim=1).values,
                    requires_grad=False,
                )
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
        custom_routing_function: Optional[Callable] = None,
    ) -> torch.Tensor:
        raise NotImplementedError


class DeepEPMoE(EPMoE):
    """
    MoE Expert Parallel Impl based on DeepEP (https://github.com/deepseek-ai/DeepEP/tree/main)
    """

    _has_printed = False

    def __init__(
        self,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        params_dtype: Optional[torch.dtype] = None,
        renormalize: bool = True,
        use_grouped_topk: bool = False,
        num_expert_group: Optional[int] = None,
        topk_group: Optional[int] = None,
        quant_config: Optional[QuantizationConfig] = None,
        tp_size: Optional[int] = None,
        prefix: str = "",
        correction_bias: Optional[torch.Tensor] = None,
        custom_routing_function: Optional[Callable] = None,
        activation: str = "silu",
    ):
        super().__init__(
            num_experts,
            top_k,
            hidden_size,
            intermediate_size,
            params_dtype,
            renormalize,
            use_grouped_topk,
            num_expert_group,
            topk_group,
            quant_config,
            tp_size,
            prefix,
            correction_bias,
            custom_routing_function,
            activation,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        reorder_topk_ids: torch.Tensor,
        seg_indptr: torch.Tensor,
        forward_mode: ForwardMode,
    ):
        # Todo: use m_grouped_gemm_fp8_fp8_bf16_nt_masked after low_latency dispatch (decode)
        if True:  # not forward_mode.is_decode():
            return self.forward_normal(hidden_states, reorder_topk_ids, seg_indptr)
        else:
            return self.forward_deepgemm_masked(
                hidden_states, reorder_topk_ids, seg_indptr
            )

    def forward_normal(
        self,
        hidden_states: torch.Tensor,
        reorder_topk_ids: torch.Tensor,
        seg_indptr: torch.Tensor,
    ):
        assert self.quant_method is not None
        assert self.activation == "silu"
        if self.grouped_gemm_runner is None:
            self.grouped_gemm_runner = GroupedGemmRunner(
                hidden_states.device, use_flashinfer=False  # TODO: use flashinfer
            )

        if self.activation_scheme == "dynamic" and not self.use_block_quant:
            max_value = (
                torch.max(hidden_states)
                .repeat(self.num_experts_per_partition)
                .to(torch.float32)
            )
            self.w13_input_scale = max_value / torch.finfo(self.fp8_dtype).max
        weight_indices_cur_rank = torch.arange(
            0,
            self.num_experts_per_partition,
            device=hidden_states.device,
            dtype=torch.int64,
        )

        # GroupGemm-0
        gateup_output = torch.empty(
            hidden_states.shape[0],
            self.w13_weight.shape[1],
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )

        if hidden_states.shape[0] > 0:
            gateup_output = self.grouped_gemm_runner(
                a=hidden_states,
                b=self.w13_weight,
                c=gateup_output,
                batch_size=self.num_experts_per_partition,
                weight_column_major=True,
                seg_indptr=seg_indptr,
                weight_indices=weight_indices_cur_rank,
                use_fp8_w8a8=self.use_fp8_w8a8,
                scale_a=self.w13_input_scale,
                scale_b=(
                    self.w13_weight_scale_inv
                    if self.use_block_quant
                    else self.w13_weight_scale
                ),
                block_shape=self.block_shape,
            )

        # Act
        down_input = torch.empty(
            gateup_output.shape[0],
            gateup_output.shape[1] // 2,
            device=gateup_output.device,
            dtype=(
                self.fp8_dtype
                if (self.use_fp8_w8a8 and not self.use_block_quant)
                else hidden_states.dtype
            ),
        )
        if self.w2_input_scale is None and not self.use_block_quant:
            self.w2_input_scale = torch.ones(
                self.num_experts_per_partition,
                dtype=torch.float32,
                device=hidden_states.device,
            )

        if self.activation == "silu":
            silu_and_mul_triton_kernel[(gateup_output.shape[0],)](
                gateup_output,
                down_input,
                gateup_output.shape[1],
                reorder_topk_ids,
                self.w2_input_scale,
                0,
                self.num_experts_per_partition - 1,
                BLOCK_SIZE=512,
            )
        else:
            raise ValueError(f"Unsupported activation: {self.activation=}")

        # GroupGemm-1
        down_output = torch.empty(
            down_input.shape[0],
            self.w2_weight.shape[1],
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        if down_input.shape[0] > 0:
            down_output = self.grouped_gemm_runner(
                a=down_input,
                b=self.w2_weight,
                c=down_output,
                batch_size=self.num_experts_per_partition,
                weight_column_major=True,
                seg_indptr=seg_indptr,
                weight_indices=weight_indices_cur_rank,
                use_fp8_w8a8=self.use_fp8_w8a8,
                scale_a=self.w2_input_scale,
                scale_b=(
                    self.w2_weight_scale_inv
                    if self.use_block_quant
                    else self.w2_weight_scale
                ),
                block_shape=self.block_shape,
            )
        return down_output

    def forward_deepgemm_masked(
        self,
        hidden_states: torch.Tensor,
        reorder_topk_ids: torch.Tensor,
        seg_indptr: torch.Tensor,
    ):
        assert self.quant_method is not None
        assert self.activation == "silu"

        if self.activation_scheme == "dynamic" and not self.use_block_quant:
            max_value = (
                torch.max(hidden_states)
                .repeat(self.num_experts_per_partition)
                .to(torch.float32)
            )
            self.w13_input_scale = max_value / torch.finfo(self.fp8_dtype).max

        # GroupGemm-0
        gateup_output = torch.empty(
            hidden_states.shape[0],
            self.w13_weight.shape[1],
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        if hidden_states.shape[0] > 0:
            # Transpose earlier so that the testing will not trigger transposing kernels
            hidden_states = (
                hidden_states[0],
                get_col_major_tma_aligned_tensor(hidden_states[1]),
            )
            """
            gateup_output = deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_masked(
                hidden_states, self.w13_weight, out, masked_m, expected_m
            )
            """

        # Act
        down_input = torch.empty(
            gateup_output.shape[0],
            gateup_output.shape[1] // 2,
            device=gateup_output.device,
            dtype=(
                self.fp8_dtype
                if (self.use_fp8_w8a8 and not self.use_block_quant)
                else hidden_states.dtype
            ),
        )
        if self.w2_input_scale is None and not self.use_block_quant:
            self.w2_input_scale = torch.ones(
                self.num_experts_per_partition,
                dtype=torch.float32,
                device=hidden_states.device,
            )

        if self.activation == "silu":
            silu_and_mul_triton_kernel[(gateup_output.shape[0],)](
                gateup_output,
                down_input,
                gateup_output.shape[1],
                reorder_topk_ids,
                self.w2_input_scale,
                0,
                self.num_experts_per_partition - 1,
                BLOCK_SIZE=512,
            )
        else:
            raise ValueError(f"Unsupported activation: {self.activation=}")

        # GroupGemm-1
        down_output = torch.empty(
            down_input.shape[0],
            self.w2_weight.shape[1],
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        if down_input.shape[0] > 0:
            # Transpose earlier so that the testing will not trigger transposing kernels
            down_input = (
                down_input[0],
                get_col_major_tma_aligned_tensor(down_input[1]),
            )
            """
            down_output = deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_masked(
                down_input, self.w2_weight, out, masked_m, expected_m
            )
            """

        return down_output
