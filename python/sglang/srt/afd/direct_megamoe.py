"""Direct MegaMoE M2N decode data plane for AFD."""

from __future__ import annotations

import importlib
import importlib.metadata
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from sglang.srt.afd.afd_config import AFDRole, resolve_overlap_sm_partition
from sglang.srt.afd.comm.base import AFDCommBackend

if TYPE_CHECKING:
    from sglang.srt.afd.layers import AFDMoELayer

logger = logging.getLogger(__name__)

_COMPONENT_DISTRIBUTION = "sgl-deep-gemm"
_COMPONENT_MODULE = "deep_gemm.mega.m2n"
_PROCESS_GROUP_NAME = "afd_direct_megamoe_m2n"


def _import_component() -> Any:
    try:
        return importlib.import_module(_COMPONENT_MODULE)
    except ModuleNotFoundError as exc:
        if exc.name not in (_COMPONENT_MODULE, "deep_gemm.mega", "deep_gemm"):
            raise
        raise RuntimeError(
            "--afd-backend direct_megamoe requires an sgl-deep-gemm build "
            "with MegaMoE M2N support"
        ) from exc


def get_direct_megamoe_version() -> str:
    _import_component()
    try:
        return importlib.metadata.version(_COMPONENT_DISTRIBUTION)
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError(
            "--afd-backend direct_megamoe requires the optional package "
            f"{_COMPONENT_DISTRIBUTION!r}"
        ) from exc


def prepare_nvfp4_megamoe_weights(experts: torch.nn.Module) -> None:
    from deep_gemm.mega import transform_weights_for_mega_moe
    from deep_gemm.utils import transform_ue4m3_sf_into_required_layout

    if not experts.quant_method.runner.runner_backend.is_flashinfer_cutedsl():
        raise RuntimeError(
            "AFD Direct MegaMoE NVFP4 requires flashinfer_cutedsl for staged prefill"
        )
    if not torch.equal(experts.g1_alphas, experts.g1_alphas_up):
        raise RuntimeError(
            "AFD NVFP4 staged prefill requires shared gate/up global scales"
        )
    w13_sf = transform_ue4m3_sf_into_required_layout(
        experts.w13_weight_scale.data, experts.w13_weight.shape[1]
    )
    w2_sf = transform_ue4m3_sf_into_required_layout(
        experts.w2_weight_scale.data, experts.w2_weight.shape[1]
    )
    # Prefill needs contiguous gate/up halves. Only W13 needs a second layout;
    # W2 is shared, and the original scales remain available to CuteDSL.
    experts.mega_l1_weights, experts.mega_l2_weights = transform_weights_for_mega_moe(
        (experts.w13_weight.data.view(torch.int8), w13_sf),
        (experts.w2_weight.data.view(torch.int8), w2_sf),
        mma_type="nvfp4xnvfp4",
    )
    experts._mega_moe_weights_built = True


@dataclass(frozen=True)
class AFDDirectMegaMoEDispatchOutput:
    lane_id: int
    input_tokens: int
    hidden_states: torch.Tensor


class AFDDirectMegaMoEAdapter:
    def __init__(
        self,
        *,
        comm: AFDCommBackend,
        layers: Sequence[AFDMoELayer],
        num_max_tokens_per_rank: int,
        num_lanes: int,
        dtype: torch.dtype,
        comm_sm_budget: int,
        nvfp4_input_scales: Sequence[float],
        component: Any,
    ) -> None:
        from sglang.srt.afd.comm.nccl import NCCLCommBackend

        if not isinstance(comm, NCCLCommBackend):
            raise TypeError("AFD Direct MegaMoE requires the NCCL backend")
        if not layers:
            raise RuntimeError("AFD Direct MegaMoE requires routed MoE layers")
        if dtype != torch.bfloat16:
            raise RuntimeError(
                f"AFD Direct MegaMoE requires BF16 activations, got {dtype}"
            )
        capability = torch.cuda.get_device_capability(torch.device(comm.config.device))
        if capability not in ((10, 0), (10, 3)):
            raise RuntimeError(
                "AFD Direct MegaMoE requires SM100 or SM103, "
                f"got SM{capability[0]}{capability[1]}"
            )
        first = layers[0]
        schema = (
            first.hidden_size,
            first.intermediate_size,
            first.top_k,
            first.num_experts,
        )
        if any(
            (
                layer.hidden_size,
                layer.intermediate_size,
                layer.top_k,
                layer.num_experts,
            )
            != schema
            for layer in layers[1:]
        ):
            raise RuntimeError("AFD Direct MegaMoE requires one shared MoE schema")
        if any(layer.num_fused_shared_experts for layer in layers):
            raise RuntimeError(
                "AFD Direct MegaMoE requires shared experts to remain on attn ranks"
            )
        if any(
            layer.experts.moe_runner_config.activation != "silu"
            or not layer.experts.moe_runner_config.is_gated
            for layer in layers
        ):
            raise RuntimeError("AFD Direct MegaMoE requires SwiGLU experts")

        self._role = comm.config.role
        self._layers = tuple(layers)
        self._component = component
        self._comm = comm
        self._max_tokens = int(num_max_tokens_per_rank)
        self._num_lanes = int(num_lanes)
        self._use_nvfp4 = bool(nvfp4_input_scales)
        self._mma_type = "nvfp4xnvfp4" if self._use_nvfp4 else "fp8xfp4"
        self._nvfp4_input_scales = (
            {
                layer.layer_id: torch.tensor(
                    scale, dtype=torch.float32, device=comm.config.device
                )
                for layer, scale in zip(layers, nvfp4_input_scales, strict=True)
            }
            if self._use_nvfp4
            else {}
        )
        total_num_sms = torch.cuda.get_device_properties(
            comm.config.device
        ).multi_processor_count
        self._ag_num_sms = min(total_num_sms, comm_sm_budget) // self._num_lanes
        if self._ag_num_sms <= 0:
            raise RuntimeError(
                "AFD Direct MegaMoE requires a positive AG SM budget per lane: "
                f"budget={comm_sm_budget}, lanes={self._num_lanes}"
            )
        self.compute_num_sms_by_ag_lanes: tuple[int | None, ...] = (None,)
        if self._role is AFDRole.ATTN:
            # AG remains resident through EG and return reduction. Shared
            # experts can overlap their own AG as well as the other lanes.
            self.compute_num_sms_by_ag_lanes += tuple(
                resolve_overlap_sm_partition(
                    total_num_sms=total_num_sms,
                    num_microbatches=self._num_lanes,
                    overlap_sm_budget=ag_lanes * self._ag_num_sms,
                )[0]
                for ag_lanes in range(1, self._num_lanes + 1)
            )
        self._capture_start_barrier_pending = False
        self._num_experts_per_rank = first.num_experts // comm.config.num_ffn_ranks
        if self._num_experts_per_rank * comm.config.num_ffn_ranks != first.num_experts:
            raise RuntimeError(
                f"AFD Direct MegaMoE cannot shard {first.num_experts} experts over "
                f"{comm.config.num_ffn_ranks} ffn ranks"
            )

        group_rank = (
            comm.config.attn_instance_rank_offset + comm.config.instance_rank
            if self._role is AFDRole.ATTN
            else comm.config.num_attn_ranks + comm.config.instance_rank
        )
        self._group = comm.create_process_group(
            rank=group_rank,
            group_name=_PROCESS_GROUP_NAME,
        )
        self._buffers = tuple(
            component.get_symm_buffer_for_mega_moe_m2n(
                self._group,
                ag_size=comm.config.num_attn_ranks,
                eg_size=comm.config.num_ffn_ranks,
                num_experts=first.num_experts,
                num_max_tokens_per_rank=self._max_tokens,
                num_topk=first.top_k,
                hidden=first.hidden_size,
                intermediate_hidden=first.intermediate_size,
                mma_type=self._mma_type,
            )
            for _ in range(self._num_lanes)
        )

        self._communication_streams: tuple[torch.cuda.Stream, ...] = ()
        self._output_storage: tuple[torch.Tensor, ...] = ()
        self._empty_hidden_states: torch.Tensor | None = None
        self._empty_topk_ids: torch.Tensor | None = None
        self._empty_topk_weights: torch.Tensor | None = None
        self._output_scales = {
            layer.layer_id: float(
                1.0
                if layer.routed_scaling_factor is None
                else layer.routed_scaling_factor
            )
            for layer in layers
        }
        self._weight_descs: torch.Tensor | None = None
        self._activation_clamp: float | None = None
        self._l1_alphas: torch.Tensor | None = None
        self._l2_alphas: torch.Tensor | None = None
        self._l2_act_scales: torch.Tensor | None = None

        if self._role is AFDRole.ATTN:
            self._communication_streams = tuple(
                torch.cuda.Stream(device=torch.device(comm.config.device))
                for _ in range(self._num_lanes)
            )
            self._output_storage = tuple(
                torch.empty(
                    (max(self._max_tokens, 1), first.hidden_size),
                    dtype=torch.bfloat16,
                    device=comm.config.device,
                )
                for _ in range(self._num_lanes)
            )
            self._empty_hidden_states = torch.zeros(
                (1, first.hidden_size), dtype=dtype, device=comm.config.device
            )
            self._empty_topk_ids = torch.full(
                (1, first.top_k),
                -1,
                dtype=torch.int32,
                device=comm.config.device,
            )
            self._empty_topk_weights = torch.zeros(
                (1, first.top_k), dtype=torch.float32, device=comm.config.device
            )
        else:
            self._build_ffn_state()

        logger.info(
            "AFD Direct MegaMoE ready: role=%s rank=%d/%d ag_ranks=%d "
            "eg_ranks=%d layers=%d lanes=%d max_tokens=%d ag_sms_per_lane=%d",
            self._role.value,
            self._buffers[0].rank,
            self._group.size(),
            comm.config.num_attn_ranks,
            comm.config.num_ffn_ranks,
            len(self._layers),
            self._num_lanes,
            self._max_tokens,
            self._ag_num_sms,
        )

    def arm_capture_start_barrier(self) -> None:
        self._capture_start_barrier_pending = True

    def _build_ffn_state(self) -> None:
        from sglang.srt.layers.quantization.fp8 import Fp8MoEMethod
        from sglang.srt.layers.quantization.modelopt_quant import (
            ModelOptNvFp4FusedMoEMethod,
        )

        descriptors = []
        activation_clamps = set()
        weight_dtypes = set()
        quant_method_type = (
            ModelOptNvFp4FusedMoEMethod if self._use_nvfp4 else Fp8MoEMethod
        )
        for layer in self._layers:
            experts = layer.experts
            quant_method = getattr(experts, "quant_method", None)
            if not isinstance(quant_method, quant_method_type) or not getattr(
                experts, "_mega_moe_weights_built", False
            ):
                raise RuntimeError(
                    "AFD Direct MegaMoE requires prepared NVFP4, MXFP4 or "
                    "128x128 block-wise FP8 weights"
                )
            if experts.num_local_experts != self._num_experts_per_rank:
                raise RuntimeError(
                    "AFD Direct MegaMoE requires contiguous expert sharding: "
                    f"local={experts.num_local_experts}, "
                    f"expected={self._num_experts_per_rank}"
                )
            descriptors.append(
                self._component.make_m2n_weight_descs(
                    experts.mega_l1_weights,
                    experts.mega_l2_weights,
                    num_experts_per_rank=self._num_experts_per_rank,
                    hidden=layer.hidden_size,
                    intermediate_hidden=layer.intermediate_size,
                    mma_type=self._mma_type,
                )
            )
            activation_clamps.add(experts.moe_runner_config.swiglu_limit)
            weight_dtypes.add(experts.w13_weight.dtype)

        if len(weight_dtypes) != 1:
            raise RuntimeError("AFD Direct MegaMoE requires one expert weight dtype")
        self._use_fp8_weights = weight_dtypes.pop() == torch.float8_e4m3fn
        if len(activation_clamps) != 1:
            raise RuntimeError(
                "AFD Direct MegaMoE requires one SwiGLU clamp across all layers"
            )
        self._activation_clamp = activation_clamps.pop()
        self._weight_descs = torch.stack(descriptors).to(
            device=self._buffers[0].buffer.device
        )
        if self._use_nvfp4:
            self._l1_alphas = torch.stack(
                [
                    torch.stack(
                        (layer.experts.g1_alphas, layer.experts.g1_alphas_up), dim=-1
                    )
                    for layer in self._layers
                ]
            ).contiguous()
            self._l2_alphas = torch.stack(
                [layer.experts.g2_alphas for layer in self._layers]
            ).contiguous()
            self._l2_act_scales = torch.stack(
                [layer.experts.w2_input_scale_quant for layer in self._layers]
            ).contiguous()

    def dispatch(
        self,
        layer_id: int,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        *,
        lane_id: int,
        num_max_dispatch_tokens_per_rank: int,
    ) -> AFDDirectMegaMoEDispatchOutput:
        if self._role is not AFDRole.ATTN:
            raise RuntimeError("AFD Direct MegaMoE dispatch requires an attn rank")
        num_tokens = hidden_states.shape[0]
        if not num_tokens <= num_max_dispatch_tokens_per_rank <= self._max_tokens:
            raise RuntimeError(
                "AFD Direct MegaMoE dispatch bucket is outside the allocated range: "
                f"tokens={num_tokens}, bucket={num_max_dispatch_tokens_per_rank}, "
                f"allocated={self._max_tokens}"
            )
        communication_stream = self._communication_streams[lane_id]
        if num_tokens == 0:
            assert self._empty_hidden_states is not None
            assert self._empty_topk_ids is not None
            assert self._empty_topk_weights is not None
            # Send one unrouted sentinel row for collective participation
            # while preserving an empty model output.
            hidden_states = self._empty_hidden_states
            topk_ids = self._empty_topk_ids
            topk_weights = self._empty_topk_weights

        hidden_states = hidden_states.contiguous()
        topk_ids = topk_ids.contiguous()
        topk_weights = topk_weights.float().contiguous()
        output = self._output_storage[lane_id][: hidden_states.shape[0]]
        current_stream = torch.cuda.current_stream(hidden_states.device)
        communication_stream.wait_stream(current_stream)
        hidden_states.record_stream(communication_stream)
        topk_ids.record_stream(communication_stream)
        topk_weights.record_stream(communication_stream)
        if self._capture_start_barrier_pending:
            self._comm.startup_barrier()
            self._capture_start_barrier_pending = False
        with torch.cuda.stream(communication_stream):
            self._component.mega_moe_m2n_ag(
                output,
                hidden_states,
                topk_ids,
                topk_weights,
                self._buffers[lane_id],
                output_scale=self._output_scales[layer_id],
                num_sms=self._ag_num_sms,
                input_scale=self._nvfp4_input_scales.get(layer_id),
            )
        return AFDDirectMegaMoEDispatchOutput(
            lane_id=lane_id,
            input_tokens=num_tokens,
            hidden_states=output[:num_tokens],
        )

    def combine(self, dispatch_output: AFDDirectMegaMoEDispatchOutput) -> torch.Tensor:
        if self._role is not AFDRole.ATTN:
            raise RuntimeError("AFD Direct MegaMoE combine requires an attn rank")
        communication_stream = self._communication_streams[dispatch_output.lane_id]
        torch.cuda.current_stream(dispatch_output.hidden_states.device).wait_stream(
            communication_stream
        )
        return dispatch_output.hidden_states

    def run_eg(self, wire_capacities: Sequence[int]) -> None:
        if self._role is not AFDRole.FFN:
            raise RuntimeError("AFD Direct MegaMoE EG requires an ffn rank")
        assert self._weight_descs is not None
        self._component.mega_moe_m2n_eg(
            self._weight_descs,
            self._buffers[: len(wire_capacities)],
            expected_num_tokens_per_rank=max(wire_capacities),
            num_experts_per_rank=self._num_experts_per_rank,
            use_fp8_weights=self._use_fp8_weights,
            activation_clamp=self._activation_clamp,
            fast_math=True,
            num_sms=0,
            l1_alphas=self._l1_alphas,
            l2_alphas=self._l2_alphas,
            l2_act_scales=self._l2_act_scales,
        )


def build_afd_direct_megamoe_adapter(
    *,
    comm: AFDCommBackend,
    layers: Sequence[AFDMoELayer],
    token_buckets: Sequence[int],
    num_lanes: int,
    dtype: torch.dtype,
    comm_sm_budget: int | None,
    nvfp4_input_scales: Sequence[float],
) -> AFDDirectMegaMoEAdapter:
    if not token_buckets:
        raise RuntimeError("AFD Direct MegaMoE requires decode token buckets")
    if comm_sm_budget is None:
        raise RuntimeError("AFD Direct MegaMoE requires a communication SM budget")
    return AFDDirectMegaMoEAdapter(
        comm=comm,
        layers=layers,
        num_max_tokens_per_rank=max(token_buckets),
        num_lanes=num_lanes,
        dtype=dtype,
        comm_sm_budget=comm_sm_budget,
        nvfp4_input_scales=nvfp4_input_scales,
        component=_import_component(),
    )
