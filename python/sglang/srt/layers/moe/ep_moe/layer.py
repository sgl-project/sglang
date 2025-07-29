from __future__ import annotations

import logging
from typing import TYPE_CHECKING, List, Optional, Tuple

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

if TYPE_CHECKING:
    from sglang.srt.layers.moe.ep_moe.token_dispatcher import (
        DeepEPLLOutput,
        DeepEPNormalOutput,
        DispatchOutput,
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
        num_fused_shared_experts: int = 0,
        params_dtype: Optional[torch.dtype] = None,
        quant_config: Optional[QuantizationConfig] = None,
        tp_size: Optional[int] = None,
        prefix: str = "",
        activation: str = "silu",
        routed_scaling_factor: Optional[float] = None,
    ):
        super().__init__(
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_fused_shared_experts=num_fused_shared_experts,
            layer_id=layer_id,
            top_k=top_k,
            params_dtype=params_dtype,
            quant_config=quant_config,
            tp_size=tp_size,
            prefix=prefix,
            activation=activation,
            # apply_router_weight_on_input=apply_router_weight_on_input,
            routed_scaling_factor=routed_scaling_factor,
            enable_ep_moe=True,
        )

        self.start_expert_id = self.ep_rank * self.num_local_experts
        self.end_expert_id = self.start_expert_id + self.num_local_experts - 1

        self.intermediate_size = intermediate_size

        if isinstance(quant_config, Fp8Config):
            self.use_block_quant = getattr(self.quant_method, "block_quant", False)
            self.block_shape = (
                self.quant_method.quant_config.weight_block_size
                if self.use_block_quant
                else None
            )
            self.fp8_dtype = torch.float8_e4m3fn
            self.activation_scheme = quant_config.activation_scheme
        else:
            self.use_fp8_w8a8 = False
            self.use_block_quant = False
            self.block_shape = None
            self.activation_scheme = None


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

    def weight_loader(
        self,
        param: torch.nn.Parameter,
        loaded_weight: torch.Tensor,
        weight_name: str,
        shard_id: str,
        expert_id: int,
    ) -> None:
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

        physical_expert_ids = global_expert_location_metadata.logical_to_all_physical(
            self.layer_id, expert_id
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
        num_fused_shared_experts: int = 0,
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
            num_fused_shared_experts=num_fused_shared_experts,
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

        # TODO: move to the beginning of the file
        from sglang.srt.distributed.parallel_state import get_tp_group
        from sglang.srt.managers.schedule_batch import global_server_args_dict
        from sglang.srt.two_batch_overlap import MaybeTboDeepEPDispatcher

        self.deepep_dispatcher = MaybeTboDeepEPDispatcher(
            group=get_tp_group().device_group,
            router_topk=self.top_k,
            permute_fusion=True,
            num_experts=self.num_experts,
            num_local_experts=self.num_local_experts,
            hidden_size=hidden_size,
            params_dtype=params_dtype,
            deepep_mode=deepep_mode,
            async_finish=True,  # TODO
            return_recv_hook=True,
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
        forward_batch: ForwardBatch,
    ):
        dispatch_output = self.dispatch(
            hidden_states, topk_idx, topk_weights, forward_batch
        )
        hidden_states = self.moe_impl(dispatch_output)
        hidden_states = self.combine(
            hidden_states,
            dispatch_output.topk_idx,
            dispatch_output.topk_weights,
            forward_batch,
        )
        return hidden_states

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        forward_batch: ForwardBatch,
    ):
        return self.deepep_dispatcher.dispatch(
            hidden_states=hidden_states,
            topk_idx=topk_idx,
            topk_weights=topk_weights,
            forward_batch=forward_batch,
        )

    def moe_impl(self, dispatch_output: DispatchOutput):
        if _use_aiter:
            # in forward_aiter, we skip token permutation and unpermutation, which have been fused inside aiter kernel
            return self.forward_aiter(dispatch_output)
        if dispatch_output.format.is_deepep_normal():
            if deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM:
                return self.forward_deepgemm_contiguous(dispatch_output)
            else:
                return self.forward_normal(dispatch_output)
        elif dispatch_output.format.is_deepep_ll():
            return self.forward_deepgemm_masked(dispatch_output)
        else:
            raise ValueError(f"Invalid deepep_mode: {self.deepep_mode}")

    def combine(
        self,
        hidden_states: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        forward_batch: ForwardBatch,
    ):
        return self.deepep_dispatcher.combine(
            hidden_states=hidden_states,
            topk_idx=topk_idx,
            topk_weights=topk_weights,
            forward_batch=forward_batch,
        )

    def _prepare_for_normal(
        self,
        hidden_states: torch.Tensor,
        topk_idx: torch.Tensor,
    ):
        from sglang.srt.layers.moe.ep_moe.kernels import (
            deepep_permute_triton_kernel,
            deepep_run_moe_deep_preprocess,
        )

        if hidden_states.shape[0] == 0:
            reorder_topk_ids = torch.empty(
                (0,), device=hidden_states.device, dtype=torch.int64
            )
            seg_indptr = torch.zeros(
                (self.num_experts + 1,),
                device=hidden_states.device,
                dtype=torch.int64,
            )
            return reorder_topk_ids, seg_indptr, hidden_states
        else:
            if _use_aiter:
                # skip permutation here as aiter fused_moe has fused inside
                reorder_topk_ids = torch.empty(
                    (0,), device=hidden_states.device, dtype=torch.int64
                )
                seg_indptr = torch.zeros(
                    (self.num_experts + 1,),
                    device=hidden_states.device,
                    dtype=torch.int64,
                )
                return reorder_topk_ids, seg_indptr, hidden_states

            reorder_topk_ids, self.src2dst, seg_indptr = deepep_run_moe_deep_preprocess(
                topk_idx, self.num_experts
            )
            num_total_tokens = reorder_topk_ids.numel()
            gateup_input = torch.empty(
                (int(num_total_tokens), hidden_states.shape[1]),
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )
            # PreReorder
            deepep_permute_triton_kernel[(hidden_states.shape[0],)](
                hidden_states,
                gateup_input,
                self.src2dst,
                topk_idx,
                None,
                self.router_topk,
                hidden_states.shape[1],
                BLOCK_SIZE=512,
            )
            return reorder_topk_ids, seg_indptr, gateup_input

    def forward_normal(
        self,
        dispatch_output: DeepEPNormalOutput,
    ):
        hidden_states, topk_idx = (
            dispatch_output.hidden_states,
            dispatch_output.topk_idx,
        )
        reorder_topk_ids, seg_indptr, hidden_states = self._prepare_for_normal(
            hidden_states, topk_idx
        )
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
        dispatch_output: DeepEPNormalOutput,
    ):
        hidden_states, topk_idx, topk_weights = (
            dispatch_output.hidden_states,
            dispatch_output.topk_idx,
            dispatch_output.topk_weights,
        )
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
        dispatch_output: DeepEPNormalOutput,
    ):
        hidden_states_fp8, topk_idx, topk_weights, num_recv_tokens_per_expert = (
            dispatch_output
        )
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
        dispatch_output: DeepEPLLOutput,
    ):
        hidden_states_fp8, _, _, masked_m, expected_m = dispatch_output
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
        self.use_flashinfer_trtllm_moe = use_flashinfer_trtllm_moe

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
            local_num_experts=self.num_local_experts,
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
