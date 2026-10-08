"""Direct A-to-expert DeepEP V2 data plane for AFD."""

from __future__ import annotations

import bisect
import importlib
import logging
import os
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from sglang.srt.afd.afd_config import AFDRole
from sglang.srt.afd.comm.base import AFDCommBackend

if TYPE_CHECKING:
    from sglang.srt.afd.layers import AFDMoELayer

logger = logging.getLogger(__name__)

_COMPONENT_PACKAGE = "sglang_afd_deepep_v2"
_COMPONENT_DISTRIBUTION = "sglang-afd-deepep-v2"
EXPERT_ALIGNMENT = 128


def _import_component() -> Any:
    try:
        module = importlib.import_module(_COMPONENT_PACKAGE)
    except ModuleNotFoundError as exc:
        if exc.name != _COMPONENT_PACKAGE:
            raise
        raise RuntimeError(
            "--afd-backend direct_deepep requires the optional package "
            f"{_COMPONENT_DISTRIBUTION!r}"
        ) from exc
    return module


def get_direct_deepep_version() -> str:
    return _import_component().__version__


def build_direct_deepep_lane_capacities(
    token_buckets: Sequence[int],
    num_lanes: int,
    tokens_per_batch: int,
) -> tuple[int, ...]:
    buckets = sorted(set(token_buckets))
    max_tokens = buckets[-1]
    if num_lanes == 1:
        return (max_tokens,)

    max_batch_size = max_tokens // tokens_per_batch
    child_size, remainder = divmod(max_batch_size, num_lanes)
    child_sizes = tuple(
        child_size + (lane_id < remainder) for lane_id in range(num_lanes)
    )
    # Lane 0 also serves the N=1 fallback; other lanes only serve split children.
    lane_capacities = [max_tokens]
    for child_batch_size in child_sizes[1:]:
        required_tokens = child_batch_size * tokens_per_batch
        lane_capacities.append(buckets[bisect.bisect_left(buckets, required_tokens)])
    return tuple(lane_capacities)


@dataclass(frozen=True)
class _AFDElasticHandle:
    num_max_tokens_per_rank: int
    num_sms: int
    topk_ids: torch.Tensor
    psum_num_recv_tokens_per_scaleup_rank: torch.Tensor
    recv_src_metadata: torch.Tensor
    token_metadata_at_forward: torch.Tensor | None
    channel_linked_list: torch.Tensor | None


@dataclass(frozen=True)
class AFDDirectDeepEPDispatchOutput:
    lane_id: int
    input_tokens: int
    hidden_states: torch.Tensor
    hidden_states_scale: torch.Tensor
    topk_ids: torch.Tensor
    topk_weights: torch.Tensor
    expert_psum: torch.Tensor
    masked_m: torch.Tensor | None
    expert_capacity: int
    handle: _AFDElasticHandle
    completion_event: Any | None

    def wait(self) -> None:
        if self.completion_event is not None:
            self.completion_event.current_stream_wait()


@dataclass(frozen=True)
class AFDDirectDeepEPCombineOutput:
    hidden_states: torch.Tensor
    completion_event: Any | None

    def wait(self) -> torch.Tensor:
        if self.completion_event is not None:
            self.completion_event.current_stream_wait()
        return self.hidden_states


class AFDDirectDeepEPAdapter:
    def __init__(
        self,
        *,
        comm: AFDCommBackend,
        hidden_size: int,
        top_k: int,
        num_real_experts: int,
        num_max_dispatch_tokens_per_rank: int,
        lane_id: int,
        num_lanes: int,
        comm_sm_budget: int,
        component: Any,
        nvfp4_input_scales: dict[int, torch.Tensor],
        fp8_scale_ue8m0: bool | None,
        expert_alignment: int = EXPERT_ALIGNMENT,
        allow_hybrid_mode: bool = True,
    ) -> None:
        nccl_comm_ptr = comm.nccl_comm_ptr
        if nccl_comm_ptr is None:
            raise RuntimeError("Direct DeepEP requires a joint NCCL communicator")

        num_ffn_ranks = comm.config.num_ffn_ranks
        if num_real_experts % num_ffn_ranks != 0:
            raise RuntimeError(
                f"Direct DeepEP cannot shard {num_real_experts} experts over "
                f"{num_ffn_ranks} ffn ranks"
            )
        experts_per_rank = num_real_experts // num_ffn_ranks
        self._num_attn_ranks = comm.config.num_attn_ranks
        self._num_virtual_experts = comm.config.world_size * experts_per_rank
        module = component
        self._role = comm.config.role
        self._lane_id = lane_id
        rank = comm.config.global_rank
        self._max_tokens = num_max_dispatch_tokens_per_rank
        self._topk_dtype = module.topk_idx_t
        self._attn_combine_input = torch.empty(
            (0, hidden_size), dtype=torch.bfloat16, device=comm.config.device
        )
        self._capture_start_barrier_pending = False
        self._comm = comm
        self._nvfp4_input_scales = nvfp4_input_scales
        self._use_nvfp4 = bool(nvfp4_input_scales)
        self._fp8_scale_ue8m0 = fp8_scale_ue8m0
        self._expert_alignment = expert_alignment

        allow_multiple_reduction = True
        # Only lane 0 serves serial fallback. Other lanes need at most one data
        # QP per channel (the component caps channels at 8/SM), plus one notify QP.
        # Use the F-authoritative budget so every rank allocates the same count.
        num_allocated_qps = (
            129 if lane_id == 0 else min(128, (comm_sm_budget // num_lanes) * 8) + 1
        )
        if not allow_hybrid_mode:
            # The native direct path needs no forwarding QPs. Lane zero
            # also serves serial fallback and retains its larger budget.
            lane_budget = (
                comm_sm_budget if lane_id == 0 else comm_sm_budget // num_lanes
            )
            num_allocated_qps = min(lane_budget, 9)
        total_sms = torch.cuda.get_device_properties(
            torch.cuda.current_device()
        ).multi_processor_count
        self._serial_communication_num_sms = min(total_sms, comm_sm_budget)
        self._overlap_communication_num_sms = (
            self._serial_communication_num_sms // num_lanes
        )
        if self._overlap_communication_num_sms < 4:
            raise RuntimeError(
                "AFD Direct DeepEP overlap requires at least 4 SMs per lane: "
                f"budget={comm_sm_budget}, lanes={num_lanes}"
            )
        self._num_qps = num_allocated_qps
        num_bytes = int(
            module.calculate_elastic_buffer_size(
                nccl_comm_ptr,
                self._max_tokens + 1,
                hidden_size,
                top_k,
                False,
                allow_hybrid_mode,
                allow_multiple_reduction,
            )
        )
        self._runtime = module.ElasticBuffer(
            rank,
            comm.config.world_size,
            nccl_comm_ptr,
            num_bytes,
            False,
            allow_hybrid_mode,
            allow_multiple_reduction,
            True,
            int(os.environ.get("EP_OVERRIDE_RDMA_SL", "3")),
            num_allocated_qps,
            120,
            120,
            True,
            self._role is AFDRole.ATTN,
        )

        torch.cuda.synchronize()
        comm.startup_barrier()
        self._runtime.barrier(True, False)
        torch.cuda.synchronize()

        logger.info(
            "AFD Direct DeepEP ready: role=%s lane=%d rank=%d/%d ffn_ranks=%d "
            "attn_ranks=%d real_experts=%d virtual_experts=%d max_tokens=%d "
            "serial_communication_sms=%d overlap_communication_sms=%d qps=%d "
            "expert_alignment=%d hybrid=%s",
            self._role.value,
            self._lane_id,
            rank,
            comm.config.world_size,
            num_ffn_ranks,
            comm.config.num_attn_ranks,
            num_real_experts,
            self._num_virtual_experts,
            self._max_tokens,
            self._serial_communication_num_sms,
            self._overlap_communication_num_sms,
            self._num_qps,
            self._expert_alignment,
            allow_hybrid_mode,
        )

    @property
    def overlap_communication_num_sms(self) -> int:
        return self._overlap_communication_num_sms

    def arm_capture_start_barrier(self) -> None:
        self._capture_start_barrier_pending = True

    def dispatch(
        self,
        layer_id: int,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        *,
        num_max_dispatch_tokens_per_rank: int,
        do_cpu_sync: bool = True,
        async_with_compute_stream: bool = False,
    ) -> AFDDirectDeepEPDispatchOutput:
        bucket = int(num_max_dispatch_tokens_per_rank)
        if not hidden_states.shape[0] <= bucket <= self._max_tokens:
            raise RuntimeError(
                "Direct DeepEP dispatch bucket is outside the allocated range: "
                f"tokens={hidden_states.shape[0]}, bucket={bucket}, "
                f"allocated={self._max_tokens}"
            )
        if self._role is AFDRole.FFN and hidden_states.shape[0] != 0:
            raise RuntimeError("Direct DeepEP ffn ranks must not source tokens")
        input_scale = None
        if self._use_nvfp4:
            input_scale = self._nvfp4_input_scales[layer_id]
            wire_hidden_states = hidden_states.contiguous()
            wire_scale = None
        else:
            from sglang.kernels.ops.quantization.fp8_kernel import (
                sglang_per_token_group_quant_fp8,
            )

            # Follow the FFN GEMM encoding, not the local GPU preference.
            scale_ue8m0 = self._fp8_scale_ue8m0
            wire_hidden_states, wire_scale = sglang_per_token_group_quant_fp8(
                hidden_states.contiguous(),
                128,
                column_major_scales=scale_ue8m0,
                scale_tma_aligned=scale_ue8m0,
                scale_ue8m0=scale_ue8m0,
            )
        expert_capacity = 0
        if self._use_nvfp4:
            total_attn_tokens = self._num_attn_ranks * bucket
            expert_capacity = (
                (total_attn_tokens + EXPERT_ALIGNMENT - 1)
                // EXPERT_ALIGNMENT
                * EXPERT_ALIGNMENT
            )
        if self._capture_start_barrier_pending:
            self._comm.startup_barrier()
            self._capture_start_barrier_pending = False

        num_sms = (
            self._overlap_communication_num_sms
            if async_with_compute_stream
            else self._serial_communication_num_sms
        )

        (
            recv_x,
            recv_scale,
            _recv_topk_ids,
            recv_topk_weights,
            copied_topk_ids,
            _num_recv_tokens_per_expert,
            psum_num_recv_tokens_per_scaleup_rank,
            psum_num_recv_tokens_per_expert,
            masked_m,
            recv_src_metadata,
            _dst_buffer_slot_idx,
            token_metadata_at_forward,
            channel_linked_list,
            completion_event,
        ) = self._runtime.dispatch(
            wire_hidden_states,
            wire_scale,
            topk_ids.to(dtype=self._topk_dtype).contiguous(),
            topk_weights.float().contiguous(),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            bucket,
            self._num_virtual_experts,
            self._expert_alignment,
            num_sms,
            self._num_qps,
            None,
            None,
            async_with_compute_stream,
            False,
            True,
            do_cpu_sync,
            True,
            True,
            self._use_nvfp4,
            input_scale,
            expert_capacity,
        )

        return AFDDirectDeepEPDispatchOutput(
            lane_id=self._lane_id,
            input_tokens=hidden_states.shape[0],
            hidden_states=recv_x,
            hidden_states_scale=recv_scale,
            topk_ids=copied_topk_ids,
            topk_weights=recv_topk_weights,
            expert_psum=psum_num_recv_tokens_per_expert,
            masked_m=masked_m,
            expert_capacity=expert_capacity,
            handle=_AFDElasticHandle(
                num_max_tokens_per_rank=bucket,
                num_sms=num_sms,
                topk_ids=copied_topk_ids,
                psum_num_recv_tokens_per_scaleup_rank=(
                    psum_num_recv_tokens_per_scaleup_rank
                ),
                recv_src_metadata=recv_src_metadata,
                token_metadata_at_forward=token_metadata_at_forward,
                channel_linked_list=channel_linked_list,
            ),
            completion_event=completion_event,
        )

    def combine(
        self,
        expert_output: torch.Tensor,
        dispatch_output: AFDDirectDeepEPDispatchOutput,
        *,
        routed_scaling_factor: float = 1.0,
        async_with_compute_stream: bool = False,
    ) -> AFDDirectDeepEPCombineOutput:
        handle = dispatch_output.handle
        combine_input = expert_output
        expert_weights = dispatch_output.topk_weights
        if self._role is AFDRole.ATTN:
            expert_weights = None
            combine_input = self._attn_combine_input
        combined, _combined_topk_weights, completion_event = self._runtime.combine(
            combine_input.contiguous(),
            expert_weights,
            None,
            None,
            handle.recv_src_metadata,
            handle.topk_ids,
            handle.psum_num_recv_tokens_per_scaleup_rank,
            handle.token_metadata_at_forward,
            handle.channel_linked_list,
            self._num_virtual_experts,
            handle.num_max_tokens_per_rank,
            handle.num_sms,
            self._num_qps,
            None,
            None,
            async_with_compute_stream,
            False,
            True,
            routed_scaling_factor,
        )
        return AFDDirectDeepEPCombineOutput(
            hidden_states=combined,
            completion_event=completion_event,
        )

    def combine_attn(
        self,
        dispatch_output: AFDDirectDeepEPDispatchOutput,
        *,
        async_with_compute_stream: bool = False,
    ) -> AFDDirectDeepEPCombineOutput:
        return self.combine(
            dispatch_output.hidden_states,
            dispatch_output,
            async_with_compute_stream=async_with_compute_stream,
        )


def build_afd_direct_deepep_adapters(
    *,
    comm: AFDCommBackend,
    layers: Sequence[AFDMoELayer],
    token_buckets: Sequence[int],
    num_lanes: int,
    tokens_per_batch: int,
    dtype: torch.dtype,
    comm_sm_budget: int | None,
    nvfp4_input_scales: Sequence[float],
    fp8_scale_ue8m0: bool | None,
    expert_alignment: int = EXPERT_ALIGNMENT,
    allow_hybrid_mode: bool = True,
) -> tuple[AFDDirectDeepEPAdapter, ...]:
    if dtype != torch.bfloat16:
        raise RuntimeError(f"AFD Direct DeepEP requires BF16 activations, got {dtype}")

    component = _import_component()
    component.validate_runtime()
    if comm_sm_budget is None:
        raise RuntimeError("AFD Direct DeepEP requires an FFN communication SM budget")

    first = layers[0]
    schema = (first.hidden_size, first.top_k, first.num_experts)
    if any(
        (layer.hidden_size, layer.top_k, layer.num_experts) != schema
        for layer in layers[1:]
    ):
        raise RuntimeError("AFD Direct DeepEP requires one shared MoE schema")

    use_nvfp4 = bool(nvfp4_input_scales)
    if expert_alignment not in (16, 32, 64, 128):
        raise RuntimeError(
            "AFD Direct DeepEP expert alignment must be 16, 32, 64, or 128"
        )
    if use_nvfp4 and expert_alignment != 128:
        raise RuntimeError("AFD Direct DeepEP NVFP4 requires expert alignment 128")
    if comm.config.role is AFDRole.FFN and expert_alignment < 128:
        capability = torch.cuda.get_device_capability(torch.device(comm.config.device))
        if capability not in ((10, 0), (10, 3)):
            raise RuntimeError(
                "AFD Direct DeepEP alignment below 128 requires SM100 or SM103 FFN"
            )
    if use_nvfp4 and len(nvfp4_input_scales) != len(layers):
        raise RuntimeError(
            "AFD Direct DeepEP NVFP4 scale count does not match MoE layers"
        )

    if comm.config.role is AFDRole.FFN:
        from sglang.srt.layers.moe.ep_moe.layer import DeepEPMoE
        from sglang.srt.layers.quantization.fp8 import Fp8MoEMethod
        from sglang.srt.layers.quantization.modelopt_quant import (
            ModelOptNvFp4FusedMoEMethod,
        )

        for layer in layers:
            experts = layer.experts
            quant_method = getattr(experts, "quant_method", None)
            runner = getattr(quant_method, "runner", None)
            runner_backend = getattr(runner, "runner_backend", None)
            if use_nvfp4:
                valid_quant_core = (
                    isinstance(quant_method, ModelOptNvFp4FusedMoEMethod)
                    and runner_backend is not None
                    and runner_backend.is_flashinfer_cutedsl()
                )
            else:
                valid_quant_core = (
                    isinstance(quant_method, Fp8MoEMethod)
                    and runner_backend is not None
                    and runner_backend.is_deep_gemm()
                    and experts.w13_weight.dtype != torch.bfloat16
                )
            if expert_alignment < 64 and not getattr(
                quant_method, "is_fp4_expert", False
            ):
                raise RuntimeError(
                    "AFD Direct DeepEP alignment below 64 requires MXFP4 weights"
                )
            if (
                not isinstance(experts, DeepEPMoE)
                or not experts.deprecate_flag
                or not valid_quant_core
            ):
                raise RuntimeError(
                    "AFD Direct DeepEP requires either the FP8 DeepGEMM core "
                    "or ModelOpt NVFP4 with flashinfer_cutedsl"
                )

        if not use_nvfp4:
            from deep_gemm.utils.layout import (
                set_mk_alignment_for_contiguous_layout,
            )

            set_mk_alignment_for_contiguous_layout(expert_alignment)

    nvfp4_scale_by_layer = {
        layer.layer_id: torch.tensor(
            scale,
            dtype=torch.float32,
            device=comm.config.device,
        )
        for layer, scale in zip(layers, nvfp4_input_scales)
    }

    lane_capacities = build_direct_deepep_lane_capacities(
        token_buckets,
        num_lanes,
        tokens_per_batch,
    )
    return tuple(
        AFDDirectDeepEPAdapter(
            comm=comm,
            hidden_size=first.hidden_size,
            top_k=first.top_k,
            num_real_experts=first.num_experts,
            num_max_dispatch_tokens_per_rank=int(capacity),
            lane_id=lane_id,
            num_lanes=num_lanes,
            comm_sm_budget=comm_sm_budget,
            component=component,
            nvfp4_input_scales=nvfp4_scale_by_layer,
            fp8_scale_ue8m0=fp8_scale_ue8m0,
            expert_alignment=expert_alignment,
            allow_hybrid_mode=allow_hybrid_mode,
        )
        for lane_id, capacity in enumerate(lane_capacities)
    )
