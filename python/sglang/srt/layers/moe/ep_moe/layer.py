import logging
from typing import List, Optional, Tuple

import torch

from sglang.srt.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from sglang.srt.eplb.expert_location import get_global_expert_location_metadata
from sglang.srt.layers.moe.ep_moe.kernels import (
    ep_gather,
    ep_scatter,
    gelu_and_mul_triton_kernel,
    grouped_gemm_triton,
    moe_ep_deepgemm_preprocess,
    post_reorder_triton_kernel,
    pre_reorder_triton_kernel,
    pre_reorder_triton_kernel_for_cutlass_moe,
    run_cutlass_moe_ep_preproess,
    run_moe_ep_preproess,
    silu_and_mul_masked_post_quant_fwd,
    silu_and_mul_triton_kernel,
    tma_align_input_scale,
)
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.moe.topk import TopKOutput
from sglang.srt.layers.quantization import deep_gemm_wrapper
from sglang.srt.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from sglang.srt.layers.quantization.fp8 import Fp8Config, Fp8MoEMethod
from sglang.srt.layers.quantization.fp8_kernel import (
    is_fp8_fnuz,
    sglang_per_token_group_quant_fp8,
    sglang_per_token_quant_fp8,
)
from sglang.srt.layers.quantization.unquant import UnquantizedFusedMoEMethod
from sglang.srt.layers.quantization.w4afp8 import W4AFp8Config, W4AFp8MoEMethod
from sglang.srt.managers.schedule_batch import global_server_args_dict
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.utils import (
    DeepEPMode,
    ceil_div,
    dispose_tensor,
    get_bool_env_var,
    is_hip,
    is_npu,
    next_power_of_2,
)

_is_hip = is_hip()
_is_npu = is_npu()
_is_fp8_fnuz = is_fp8_fnuz()
_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip
use_flashinfer_trtllm_moe = (
    global_server_args_dict["enable_flashinfer_trtllm_moe"]
    and global_server_args_dict["enable_ep_moe"]
)

if not (_is_npu or _is_hip):
    from sgl_kernel import silu_and_mul

if _use_aiter:
    from aiter import ActivationType, QuantType
    from aiter.fused_moe import fused_moe
    from aiter.ops.shuffle import shuffle_weight

if use_flashinfer_trtllm_moe:
    try:
        import flashinfer.fused_moe as fi_fused_moe
    except ImportError:
        fi_fused_moe = None
        use_flashinfer_trtllm_moe = False

logger = logging.getLogger(__name__)


class GroupedGemmRunner(torch.nn.Module):
    flashinfer_gemm_warpper = None

    def __init__(
        self,
        device,
        use_flashinfer: bool = False,
        use_per_token_if_dynamic: bool = True,
    ):
        super().__init__()
        self.device = device
        self.use_flashinfer = use_flashinfer
        self.use_per_token_if_dynamic = use_per_token_if_dynamic
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
        c_dtype=None,
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
                c_dtype=c_dtype,
                use_per_token_if_dynamic=self.use_per_token_if_dynamic,
            )
        return c


def _get_tile_tokens_dim(num_tokens, top_k, num_experts):
    # Guess tokens per expert assuming perfect expert distribution first.
    num_tokens_per_expert = (num_tokens * top_k) // num_experts
    # And pad the number to the next power of 2.
    tile_tokens_dim = next_power_of_2(num_tokens_per_expert)
    # Cap to 8-64 tokens per CTA tile as it's the range supported by the kernel.
    tile_tokens_dim = min(max(tile_tokens_dim, 8), 64)
    return tile_tokens_dim


class EPMoE(FusedMoE):
    """
    MoE Expert Parallel Impl


    """

    def __init__(
        self,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        layer_id: int,
        params_dtype: Optional[torch.dtype] = None,
        quant_config: Optional[QuantizationConfig] = None,
        tp_size: Optional[int] = None,
        prefix: str = "",
        activation: str = "silu",
        routed_scaling_factor: Optional[float] = None,
        use_per_token_if_dynamic: bool = True,
    ):
        super().__init__(
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            top_k=top_k,
            layer_id=layer_id,
            params_dtype=params_dtype,
            quant_config=quant_config,
            tp_size=tp_size,
            prefix=prefix,
            activation=activation,
            routed_scaling_factor=routed_scaling_factor,
            enable_ep_moe=True,
            skip_quant=True,
        )

        if params_dtype is None:
            params_dtype = torch.get_default_dtype()

        self.layer_id = layer_id
        self.num_local_experts, self.expert_map = self.determine_expert_map()
        self.start_expert_id = self.ep_rank * self.num_local_experts
        self.end_expert_id = self.start_expert_id + self.num_local_experts - 1

        self.intermediate_size = intermediate_size
        self.use_per_token_if_dynamic = use_per_token_if_dynamic

        # TODO(ch-wan): move quant preparation to FusedMoE
        if quant_config is None:
            self.quant_method: Optional[QuantizeMethodBase] = (
                UnquantizedFusedMoEMethod()
            )
            self.use_fp8_w8a8 = False
            self.use_block_quant = False
            self.block_shape = None
            self.activation_scheme = None
            self.w13_input_scale = None
            self.w2_input_scale = None
            self.w13_weight_scale = None
            self.w2_weight_scale = None
        elif isinstance(quant_config, W4AFp8Config):
            self.quant_method: Optional[QuantizeMethodBase] = W4AFp8MoEMethod(
                quant_config
            )
            self.use_fp8_w8a8 = False
            self.use_block_quant = False
            self.fp8_dtype = torch.float8_e4m3fn
            self.w13_input_scale = None
            self.w2_input_scale = None
            self.w13_weight_scale = None
            self.w2_weight_scale = None
            self.activation_scheme = quant_config.moe_activation_scheme
        elif isinstance(quant_config, Fp8Config):
            self.quant_method: Optional[QuantizeMethodBase] = Fp8MoEMethod(quant_config)
            self.use_fp8_w8a8 = True
            self.use_block_quant = getattr(self.quant_method, "block_quant", False)
            self.block_shape = (
                self.quant_method.quant_config.weight_block_size
                if self.use_block_quant
                else None
            )
            self.fp8_dtype = torch.float8_e4m3fn
            self.activation_scheme = quant_config.activation_scheme
        else:
            raise ValueError(f"Unsupported quant_config: {quant_config}")

        self.quant_config = quant_config
        self.quant_method.create_weights(
            layer=self,
            num_experts=self.num_local_experts,
            hidden_size=hidden_size,
            intermediate_size=self.intermediate_size,
            params_dtype=params_dtype,
            weight_loader=self.weight_loader,
        )

        self.grouped_gemm_runner = None

    # Adapted from https://github.com/vllm-project/vllm/blob/9fb52e523abf7bdaf7e60cf2971edb5a1b13dc08/vllm/model_executor/layers/fused_moe/layer.py#L544C1-L586C43
    # Modifications: use determine_expert_map as a class internal function, set 'global_num_experts' rather than '-1' for experts not assigned to the current rank.
    def determine_expert_map(self) -> Tuple[int, Optional[torch.Tensor]]:
        """
        Calculates how many experts should be assigned to each rank for EP and
        creates a mapping from global to local expert index. Experts are
        distributed evenly across ranks. Any remaining are assigned to the
        last rank.

        Returns:
            Tuple[int, Optional[torch.Tensor]]: A tuple containing:
                - local_num_experts (int): The number of experts assigned
                    to the current rank.
                - expert_map (Optional[torch.Tensor]): A tensor of shape
                    (global_num_experts,) mapping from global to local index.
                    Contains global_num_experts for experts not assigned to the current rank.
                    Returns None if ep_size is 1.
        """
        ep_size = self.ep_size
        ep_rank = self.ep_rank
        global_num_experts = self.num_experts

        assert ep_size > 0
        if ep_size == 1:
            return (global_num_experts, None)

        local_num_experts = global_num_experts // ep_size

        expert_map = torch.full(
            (global_num_experts,), global_num_experts, dtype=torch.int32
        )
        if ep_rank < (ep_size - 1):
            expert_map[
                ep_rank * local_num_experts : (ep_rank + 1) * local_num_experts
            ] = torch.arange(0, local_num_experts, dtype=torch.int32)
        else:
            local_num_experts = global_num_experts - ep_rank * local_num_experts

            expert_map[-local_num_experts:] = torch.arange(
                0, local_num_experts, dtype=torch.int32
            )
        return (local_num_experts, expert_map)

    def forward(self, hidden_states: torch.Tensor, topk_output: TopKOutput):
        if deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM and self.use_fp8_w8a8:
            return self.forward_deepgemm(hidden_states, topk_output)
        else:
            return self.forward_normal(hidden_states, topk_output)

    def forward_deepgemm(
        self,
        hidden_states: torch.Tensor,
        topk_output: TopKOutput,
    ):

        self.w13_weight_fp8 = (
            self.w13_weight,
            (
                self.w13_weight_scale_inv
                if self.use_block_quant
                else self.w13_weight_scale
            ),
        )
        self.w2_weight_fp8 = (
            self.w2_weight,
            self.w2_weight_scale_inv if self.use_block_quant else self.w2_weight_scale,
        )

        assert self.quant_method is not None
        assert self.activation == "silu"
        hidden_states_shape = hidden_states.shape
        hidden_states_dtype = hidden_states.dtype
        hidden_states_device = hidden_states.device

        topk_weights, topk_ids, _ = topk_output

        if not self.use_block_quant:
            # Convert per-tensor quant to per-block quant by repeating scales for forward_deepgemm
            scale_block_size = 128
            w13_weight_scale_n = 2 * (
                (self.intermediate_size + scale_block_size - 1) // scale_block_size
            )
            w13_weight_scale_k = (
                hidden_states_shape[-1] + scale_block_size - 1
            ) // scale_block_size
            w13_weight_scale = (
                self.w13_weight_scale.unsqueeze(1)
                .repeat_interleave(w13_weight_scale_n, dim=1)
                .unsqueeze(2)
                .repeat_interleave(w13_weight_scale_k, dim=2)
            )
            self.w13_weight_fp8 = (
                self.w13_weight,
                w13_weight_scale,
            )
            w2_weight_scale_n = (
                hidden_states_shape[-1] + scale_block_size - 1
            ) // scale_block_size
            w2_weight_scale_k = (
                self.intermediate_size + scale_block_size - 1
            ) // scale_block_size
            w2_weight_scale = (
                self.w2_weight_scale.unsqueeze(1)
                .repeat_interleave(w2_weight_scale_n, dim=1)
                .unsqueeze(2)
                .repeat_interleave(w2_weight_scale_k, dim=2)
            )
            self.w2_weight_fp8 = (
                self.w2_weight,
                w2_weight_scale,
            )

        # PreReorder
        m_max, masked_m, expected_m, src2dst, gateup_input, gateup_input_scale = (
            moe_ep_deepgemm_preprocess(
                topk_ids,
                self.num_experts,
                hidden_states,
                self.top_k,
                self.start_expert_id,
                self.end_expert_id,
                self.block_shape,
            )
        )

        dispose_tensor(hidden_states)

        # GroupGemm-0
        gateup_input_fp8 = (
            gateup_input,
            deep_gemm_wrapper.get_col_major_tma_aligned_tensor(gateup_input_scale),
        )
        num_groups, m, k = gateup_input_fp8[0].size()
        n = self.w13_weight.size(1)
        gateup_output = torch.empty(
            (num_groups, m, n), device=hidden_states_device, dtype=torch.bfloat16
        )
        deep_gemm_wrapper.grouped_gemm_nt_f8f8bf16_masked(
            gateup_input_fp8, self.w13_weight_fp8, gateup_output, masked_m, expected_m
        )
        del gateup_input
        del gateup_input_fp8

        # Act
        down_input = torch.empty(
            (
                gateup_output.shape[0],
                gateup_output.shape[1],
                gateup_output.shape[2] // 2,
            ),
            device=hidden_states_device,
            dtype=self.fp8_dtype,
        )
        scale_block_size = 128
        down_input_scale = torch.empty(
            (
                gateup_output.shape[0],
                gateup_output.shape[1],
                gateup_output.shape[2] // 2 // scale_block_size,
            ),
            device=hidden_states_device,
            dtype=torch.float32,
        )
        silu_and_mul_masked_post_quant_fwd(
            gateup_output,
            down_input,
            down_input_scale,
            scale_block_size,
            masked_m,
        )
        del gateup_output

        # GroupGemm-1
        n = self.w2_weight.size(1)
        down_input_fp8 = (
            down_input,
            deep_gemm_wrapper.get_col_major_tma_aligned_tensor(down_input_scale),
        )
        down_output = torch.empty(
            (num_groups, m, n), device=hidden_states_device, dtype=torch.bfloat16
        )
        deep_gemm_wrapper.grouped_gemm_nt_f8f8bf16_masked(
            down_input_fp8, self.w2_weight_fp8, down_output, masked_m, expected_m
        )
        del down_input
        del down_input_fp8

        # PostReorder
        output = torch.empty(
            hidden_states_shape, dtype=hidden_states_dtype, device=hidden_states_device
        )
        post_reorder_triton_kernel[(hidden_states_shape[0],)](
            down_output,
            output,
            src2dst,
            topk_ids,
            topk_weights,
            self.start_expert_id,
            self.end_expert_id,
            self.top_k,
            hidden_states_shape[1],
            m_max * self.start_expert_id,
            BLOCK_SIZE=512,
        )
        return output

    def forward_normal(self, hidden_states: torch.Tensor, topk_output: TopKOutput):
        return self.quant_method.apply(self, hidden_states, topk_output)

    def run_moe(self, hidden_states: torch.Tensor, topk_output: TopKOutput):

        topk_weights, topk_ids, _ = topk_output

        hidden_states_shape = hidden_states.shape
        hidden_states_dtype = hidden_states.dtype
        hidden_states_device = hidden_states.device
        if self.grouped_gemm_runner is None:
            self.grouped_gemm_runner = GroupedGemmRunner(
                hidden_states.device,
                use_flashinfer=False,  # TODO: use flashinfer
                use_per_token_if_dynamic=self.use_per_token_if_dynamic,
            )

        num_experts = self.num_experts

        reorder_topk_ids, src2dst, seg_indptr = run_moe_ep_preproess(
            topk_ids,
            num_experts,
        )

        gateup_input = torch.empty(
            (int(hidden_states.shape[0] * self.top_k), hidden_states.shape[1]),
            device=hidden_states.device,
            dtype=(
                self.fp8_dtype
                if self.use_fp8_w8a8 and not self.use_block_quant
                else hidden_states.dtype
            ),
        )
        if self.activation_scheme == "dynamic" and not self.use_block_quant:
            if self.use_per_token_if_dynamic:
                max_value = torch.max(hidden_states, dim=1).values.to(torch.float32)
                self.w13_input_scale = max_value / torch.finfo(self.fp8_dtype).max
            else:
                max_value = (
                    torch.max(hidden_states)
                    .repeat(self.num_local_experts)
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
            use_per_token_if_dynamic=self.use_per_token_if_dynamic,
        )
        dispose_tensor(hidden_states)

        if (
            self.activation_scheme == "dynamic"
            and not self.use_block_quant
            and self.use_per_token_if_dynamic
        ):
            scale = torch.empty(
                hidden_states_shape[0] * self.top_k,
                device=hidden_states_device,
                dtype=torch.float32,
            )
            scale[src2dst] = (
                self.w13_input_scale.unsqueeze(1)
                .expand(hidden_states_shape[0], self.top_k)
                .reshape(-1)
            )
            self.w13_input_scale = scale

        seg_indptr_cur_rank = seg_indptr[self.start_expert_id : self.end_expert_id + 2]
        weight_indices_cur_rank = torch.arange(
            0,
            self.num_local_experts,
            device=hidden_states_device,
            dtype=torch.int64,
        )
        # GroupGemm-0
        gateup_output = self.grouped_gemm_runner(
            a=gateup_input,
            b=self.w13_weight,
            c=None,
            c_dtype=hidden_states_dtype,
            batch_size=self.num_local_experts,
            weight_column_major=True,
            seg_indptr=seg_indptr_cur_rank,
            weight_indices=weight_indices_cur_rank,
            use_fp8_w8a8=self.use_fp8_w8a8,
            scale_a=self.w13_input_scale,
            scale_b=self.w13_weight_scale,
            block_shape=self.block_shape,
        )
        del gateup_input

        # Act
        if self.activation_scheme == "dynamic" and not self.use_block_quant:
            self.w2_input_scale = None
            down_input = torch.empty(
                gateup_output.shape[0],
                gateup_output.shape[1] // 2,
                device=gateup_output.device,
                dtype=hidden_states_dtype,
            )
        else:
            down_input = torch.empty(
                gateup_output.shape[0],
                gateup_output.shape[1] // 2,
                device=gateup_output.device,
                dtype=(
                    self.fp8_dtype
                    if (self.use_fp8_w8a8 and not self.use_block_quant)
                    else hidden_states_dtype
                ),
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
        del gateup_output

        if self.activation_scheme == "dynamic" and not self.use_block_quant:
            if self.use_per_token_if_dynamic:
                down_input, self.w2_input_scale = sglang_per_token_quant_fp8(down_input)
            else:
                self.w2_input_scale = torch.ones(
                    self.num_local_experts,
                    dtype=torch.float32,
                    device=hidden_states_device,
                )

        # GroupGemm-1
        down_output = torch.empty(
            down_input.shape[0],
            self.w2_weight.shape[1],
            device=hidden_states_device,
            dtype=hidden_states_dtype,
        )
        down_output = self.grouped_gemm_runner(
            a=down_input,
            b=self.w2_weight,
            c=down_output,
            batch_size=self.num_local_experts,
            weight_column_major=True,
            seg_indptr=seg_indptr_cur_rank,
            weight_indices=weight_indices_cur_rank,
            use_fp8_w8a8=self.use_fp8_w8a8,
            scale_a=self.w2_input_scale,
            scale_b=self.w2_weight_scale,
            block_shape=self.block_shape,
        )
        del down_input

        # PostReorder
        output = torch.empty(
            hidden_states_shape, dtype=hidden_states_dtype, device=hidden_states_device
        )
        post_reorder_triton_kernel[(hidden_states_shape[0],)](
            down_output,
            output,
            src2dst,
            topk_ids,
            topk_weights,
            self.start_expert_id,
            self.end_expert_id,
            self.top_k,
            hidden_states_shape[1],
            0,
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

    def weight_loader(
        self,
        param: torch.nn.Parameter,
        loaded_weight: torch.Tensor,
        weight_name: str,
        shard_id: str,
        expert_id: int,
    ) -> None:
        physical_expert_ids = (
            get_global_expert_location_metadata().logical_to_all_physical(
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
        if expert_id < self.start_expert_id or expert_id > self.end_expert_id:
            return
        expert_id = expert_id - self.start_expert_id

        self._weight_loader_impl(
            param=param,
            loaded_weight=loaded_weight,
            weight_name=weight_name,
            shard_id=shard_id,
            expert_id=expert_id,
        )
        return


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
        layer_id: int,
        params_dtype: Optional[torch.dtype] = None,
        quant_config: Optional[QuantizationConfig] = None,
        tp_size: Optional[int] = None,
        prefix: str = "",
        activation: str = "silu",
        routed_scaling_factor: Optional[float] = None,
        deepep_mode: DeepEPMode = DeepEPMode.auto,
    ):
        super().__init__(
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            layer_id=layer_id,
            params_dtype=params_dtype,
            quant_config=quant_config,
            tp_size=tp_size,
            prefix=prefix,
            activation=activation,
            routed_scaling_factor=routed_scaling_factor,
        )
        self.deepep_mode = deepep_mode
        if deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM:
            assert self.use_fp8_w8a8, (
                "DeepGEMM requires an fp8_w8a8 model; "
                "alternatively, you can disable DeepGEMM by turning off the ENABLE_JIT_DEEPGEMM environment variable."
            )

        if self.deepep_mode.enable_low_latency():
            assert (
                deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM
            ), f"DeepEP {self.deepep_mode} mode requires deep_gemm"
        if _use_aiter:
            # expert_mask is of size (self.num_local_experts + 1),
            # the extra 1 is for invalid rank_id (in original deepep, the invalid rank_id is -1, but aiter does not allow -1, we use a mask to make those ids invalid)
            # for instance, if we have 4 experts on this rank, we would have a expert_mask like:
            #     self.expert_mask = [1, 1, 1, 1, 0]
            # idx from 0-3 is valid and will be processed, while idx == 4 will be masked out
            self.expert_mask = torch.zeros(
                (self.num_local_experts + 1),
                device=torch.cuda.current_device(),
                dtype=torch.int,
            )
            # the last one is invalid rank_id
            self.expert_mask[:-1] = 1
        else:
            self.w13_weight_fp8 = (
                self.w13_weight,
                (
                    self.w13_weight_scale_inv
                    if self.use_block_quant
                    else self.w13_weight_scale
                ),
            )
            self.w2_weight_fp8 = (
                self.w2_weight,
                (
                    self.w2_weight_scale_inv
                    if self.use_block_quant
                    else self.w2_weight_scale
                ),
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        reorder_topk_ids: torch.Tensor,
        seg_indptr: torch.Tensor,
        masked_m: torch.Tensor,
        expected_m: int,
        num_recv_tokens_per_expert: List[int],
        forward_batch: ForwardBatch,
    ):
        if _use_aiter:
            # in forward_aiter, we skip token permutation and unpermutation, which have been fused inside aiter kernel
            return self.forward_aiter(hidden_states, topk_idx, topk_weights)
        resolved_deepep_mode = self.deepep_mode.resolve(
            forward_batch.is_extend_in_batch
        )
        if resolved_deepep_mode == DeepEPMode.normal:
            if deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM:
                return self.forward_deepgemm_contiguous(
                    hidden_states, topk_idx, topk_weights, num_recv_tokens_per_expert
                )
            else:
                return self.forward_normal(hidden_states, reorder_topk_ids, seg_indptr)
        elif resolved_deepep_mode == DeepEPMode.low_latency:
            return self.forward_deepgemm_masked(hidden_states, masked_m, expected_m)
        else:
            raise ValueError(f"Invalid deepep_mode: {self.deepep_mode}")

    def forward_normal(
        self,
        hidden_states: torch.Tensor,
        reorder_topk_ids: torch.Tensor,
        seg_indptr: torch.Tensor,
    ):
        hidden_states_dtype = hidden_states.dtype
        hidden_states_device = hidden_states.device

        assert self.quant_method is not None
        assert self.activation == "silu"
        if self.grouped_gemm_runner is None:
            self.grouped_gemm_runner = GroupedGemmRunner(
                hidden_states.device, use_flashinfer=False  # TODO: use flashinfer
            )

        if self.activation_scheme == "dynamic" and not self.use_block_quant:
            max_value = (
                torch.max(hidden_states)
                .repeat(self.num_local_experts)
                .to(torch.float32)
            )
            self.w13_input_scale = max_value / torch.finfo(self.fp8_dtype).max
        weight_indices_cur_rank = torch.arange(
            0,
            self.num_local_experts,
            device=hidden_states.device,
            dtype=torch.int64,
        )

        # GroupGemm-0
        if hidden_states.shape[0] > 0:
            gateup_output = self.grouped_gemm_runner(
                a=hidden_states,
                b=self.w13_weight,
                c=None,
                c_dtype=hidden_states.dtype,
                batch_size=self.num_local_experts,
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
        else:
            gateup_output = torch.empty(
                hidden_states.shape[0],
                self.w13_weight.shape[1],
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )

        # Act
        down_input = torch.empty(
            gateup_output.shape[0],
            gateup_output.shape[1] // 2,
            device=gateup_output.device,
            dtype=(
                self.fp8_dtype
                if (self.use_fp8_w8a8 and not self.use_block_quant)
                else hidden_states_dtype
            ),
        )
        if self.w2_input_scale is None and not self.use_block_quant:
            self.w2_input_scale = torch.ones(
                self.num_local_experts,
                dtype=torch.float32,
                device=hidden_states_device,
            )

        if self.activation == "silu":
            silu_and_mul_triton_kernel[(gateup_output.shape[0],)](
                gateup_output,
                down_input,
                gateup_output.shape[1],
                reorder_topk_ids,
                self.w2_input_scale,
                0,
                self.num_local_experts - 1,
                BLOCK_SIZE=512,
            )
        else:
            raise ValueError(f"Unsupported activation: {self.activation=}")

        del gateup_output

        # GroupGemm-1
        down_output = torch.empty(
            down_input.shape[0],
            self.w2_weight.shape[1],
            device=hidden_states_device,
            dtype=hidden_states_dtype,
        )
        if down_input.shape[0] > 0:
            down_output = self.grouped_gemm_runner(
                a=down_input,
                b=self.w2_weight,
                c=down_output,
                batch_size=self.num_local_experts,
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

    def forward_aiter(
        self,
        hidden_states: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
    ):
        if hidden_states.shape[0] == 0:
            return hidden_states
        # in original deepep, idx == -1 meaning invalid and will not be processed.
        # aiter does not accept -1, we use a expert mask to make these idx invalid
        # (idx == num_local_experts) meaning not used in aiter fused_moe
        topk_idx_copy = topk_idx.to(torch.int32)
        topk_idx_copy[topk_idx_copy == -1] = self.num_local_experts

        return fused_moe(
            hidden_states,
            self.w13_weight,
            self.w2_weight,
            topk_weights,
            topk_idx_copy,
            w1_scale=self.w13_weight_scale_inv,
            w2_scale=self.w2_weight_scale_inv,
            quant_type=QuantType.per_128x128,
            activation=(
                ActivationType.Silu
                if self.activation == "silu"
                else ActivationType.Gelu
            ),
            expert_mask=self.expert_mask,
        )

    def forward_deepgemm_contiguous(
        self,
        hidden_states_fp8: Tuple[torch.Tensor, torch.Tensor],
        topk_idx,
        topk_weights,
        num_recv_tokens_per_expert: List[int],
    ):
        hidden_states_fp8, hidden_states_scale = hidden_states_fp8
        assert self.quant_method is not None
        assert self.activation == "silu"
        if num_recv_tokens_per_expert is None:
            return hidden_states_fp8.bfloat16()
        all_tokens = sum(num_recv_tokens_per_expert)
        if all_tokens <= 0:
            return hidden_states_fp8.bfloat16()
        M, K = hidden_states_fp8.size()
        N = self.w13_weight.size(1)
        scale_block_size = 128

        hidden_states_fp8_shape = hidden_states_fp8.shape
        hidden_states_fp8_device = hidden_states_fp8.device
        hidden_states_fp8_dtype = hidden_states_fp8.dtype

        input_tensor = [
            torch.empty(
                (all_tokens, K),
                device=hidden_states_fp8.device,
                dtype=hidden_states_fp8.dtype,
            ),
            (
                # TODO check whether need `zeros`
                torch.zeros(
                    (ceil_div(K // 128, 4), all_tokens),
                    device=hidden_states_fp8.device,
                    dtype=torch.int,
                ).transpose(0, 1)
                if deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0
                else torch.empty(
                    (all_tokens, K // 128),
                    device=hidden_states_fp8.device,
                    dtype=torch.float32,
                )
            ),
        ]
        m_indices = torch.empty(
            all_tokens, device=hidden_states_fp8.device, dtype=torch.int32
        )
        output_index = torch.empty_like(topk_idx)

        num_recv_tokens_per_expert_gpu = torch.tensor(
            num_recv_tokens_per_expert,
            dtype=torch.int32,
            pin_memory=True,
            device="cpu",
        ).cuda(non_blocking=True)
        expert_start_loc = torch.empty_like(num_recv_tokens_per_expert_gpu)

        ep_scatter(
            hidden_states_fp8,
            hidden_states_scale,
            topk_idx,
            num_recv_tokens_per_expert_gpu,
            expert_start_loc,
            input_tensor[0],
            input_tensor[1],
            m_indices,
            output_index,
            scale_ue8m0=deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,
        )
        dispose_tensor(hidden_states_fp8)

        gateup_output = torch.empty(
            (all_tokens, N),
            device=hidden_states_fp8_device,
            dtype=torch.bfloat16,
        )
        if not deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0:
            input_tensor[1] = tma_align_input_scale(input_tensor[1])
        deep_gemm_wrapper.grouped_gemm_nt_f8f8bf16_contig(
            input_tensor, self.w13_weight_fp8, gateup_output, m_indices
        )
        del input_tensor
        down_input = torch.empty(
            (
                all_tokens,
                N // 2,
            ),
            device=gateup_output.device,
            dtype=torch.bfloat16,
        )
        silu_and_mul(gateup_output.view(-1, N), down_input)
        del gateup_output
        down_output = torch.empty(
            (all_tokens, K),
            device=hidden_states_fp8_device,
            dtype=torch.bfloat16,
        )
        down_input_fp8, down_input_scale = sglang_per_token_group_quant_fp8(
            down_input,
            scale_block_size,
            column_major_scales=deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,
            scale_tma_aligned=deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,
            scale_ue8m0=deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,
        )
        del down_input
        if not deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0:
            down_input_scale = tma_align_input_scale(down_input_scale)
        deep_gemm_wrapper.grouped_gemm_nt_f8f8bf16_contig(
            (down_input_fp8, down_input_scale),
            self.w2_weight_fp8,
            down_output,
            m_indices,
        )
        del down_input_fp8, down_input_scale

        gather_out = torch.empty(
            hidden_states_fp8_shape,
            device=hidden_states_fp8_device,
            dtype=torch.bfloat16,
        )
        ep_gather(down_output, topk_idx, topk_weights, output_index, gather_out)

        return gather_out

    def forward_deepgemm_masked(
        self,
        hidden_states_fp8: Tuple[torch.Tensor, torch.Tensor],
        masked_m: torch.Tensor,
        expected_m: int,
    ):
        assert self.quant_method is not None
        assert self.activation == "silu"

        # GroupGemm-0
        num_groups, m, k = hidden_states_fp8[0].size()
        n = self.w13_weight.size(1)
        expected_m = min(expected_m, m)
        gateup_output = torch.empty(
            (num_groups, m, n), device=hidden_states_fp8[0].device, dtype=torch.bfloat16
        )
        deep_gemm_wrapper.grouped_gemm_nt_f8f8bf16_masked(
            hidden_states_fp8,
            self.w13_weight_fp8,
            gateup_output,
            masked_m,
            expected_m,
            recipe=(1, 128, 128) if deep_gemm_wrapper.DEEPGEMM_BLACKWELL else None,
        )
        dispose_tensor(hidden_states_fp8[0])

        # Act
        down_input = torch.empty(
            (
                gateup_output.shape[0],
                gateup_output.shape[1],
                gateup_output.shape[2] // 2,
            ),
            device=gateup_output.device,
            dtype=self.fp8_dtype,
        )
        scale_block_size = 128
        down_input_scale = torch.empty(
            (
                gateup_output.shape[0],
                gateup_output.shape[1],
                gateup_output.shape[2] // 2 // scale_block_size,
            ),
            device=gateup_output.device,
            dtype=torch.float32,
        )
        silu_and_mul_masked_post_quant_fwd(
            gateup_output,
            down_input,
            down_input_scale,
            scale_block_size,
            masked_m,
            scale_ue8m0=deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,
        )
        del gateup_output

        # GroupGemm-1
        n = self.w2_weight.size(1)
        down_input_fp8 = (
            down_input,
            (
                down_input_scale
                if deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0
                else deep_gemm_wrapper.get_col_major_tma_aligned_tensor(
                    down_input_scale
                )
            ),
        )
        down_output = torch.empty(
            (num_groups, m, n), device=down_input.device, dtype=torch.bfloat16
        )
        deep_gemm_wrapper.grouped_gemm_nt_f8f8bf16_masked(
            down_input_fp8,
            self.w2_weight_fp8,
            down_output,
            masked_m,
            expected_m,
            recipe=(1, 128, 128) if deep_gemm_wrapper.DEEPGEMM_BLACKWELL else None,
        )

        return down_output


class FlashInferEPMoE(EPMoE):
    def __init__(self, *args, **kwargs):
        renormalize = kwargs.pop("renormalize", True)
        num_fused_shared_experts = kwargs.pop("num_fused_shared_experts", 0)
        use_grouped_topk = kwargs.pop("use_grouped_topk", False)
        num_expert_group = kwargs.pop("num_expert_group", None)
        topk_group = kwargs.pop("topk_group", None)
        correction_bias = kwargs.pop("correction_bias", None)
        super().__init__(*args, **kwargs)
        self.renormalize = renormalize
        self.num_fused_shared_experts = num_fused_shared_experts
        self.use_grouped_topk = use_grouped_topk
        if self.use_grouped_topk:
            assert num_expert_group is not None and topk_group is not None
        self.num_expert_group = num_expert_group
        self.topk_group = topk_group
        self.correction_bias = correction_bias

    def forward(self, hidden_states: torch.Tensor, router_logits: torch.Tensor):
        assert use_flashinfer_trtllm_moe
        assert (
            self.activation == "silu"
        ), "Only silu is supported for flashinfer blockscale fp8 moe"
        assert (
            self.renormalize
        ), "Renormalize is required for flashinfer blockscale fp8 moe"
        assert (
            self.num_fused_shared_experts == 0
        ), "Fused shared experts are not supported for flashinfer blockscale fp8 moe"
        a_q, a_sf = sglang_per_token_group_quant_fp8(hidden_states, self.block_shape[1])
        # NOTE: scales of hidden states have to be transposed!
        a_sf_t = a_sf.t().contiguous()
        assert fi_fused_moe is not None
        return fi_fused_moe.trtllm_fp8_block_scale_moe(
            routing_logits=router_logits.to(torch.float32),
            routing_bias=self.correction_bias.to(hidden_states.dtype),
            hidden_states=a_q,
            hidden_states_scale=a_sf_t,
            gemm1_weights=self.w13_weight,
            gemm1_weights_scale=self.w13_weight_scale_inv,
            gemm2_weights=self.w2_weight,
            gemm2_weights_scale=self.w2_weight_scale_inv,
            num_experts=self.num_experts,
            top_k=self.top_k,
            n_group=self.num_expert_group,
            topk_group=self.topk_group,
            intermediate_size=self.w2_weight.shape[2],
            local_expert_offset=self.start_expert_id,
            local_num_experts=self.num_experts_per_partition,
            routed_scaling_factor=self.routed_scaling_factor,
            tile_tokens_dim=_get_tile_tokens_dim(
                hidden_states.shape[0], self.top_k, self.num_experts
            ),
            routing_method_type=2,  # DeepSeek-styled routing method
            use_shuffled_weight=False,
        )


def get_moe_impl_class():
    if global_server_args_dict["enable_deepep_moe"]:
        return DeepEPMoE
    if global_server_args_dict["enable_flashinfer_cutlass_moe"]:
        # Must come before EPMoE because FusedMoE also supports enable_ep_moe
        return FusedMoE
    if use_flashinfer_trtllm_moe:
        # Must come before EPMoE because FusedMoE also supports enable_ep_moe
        return FlashInferEPMoE
    if global_server_args_dict["enable_ep_moe"]:
        return EPMoE
    return FusedMoE
