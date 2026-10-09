"""FFN-side expert compute and decode graph capture."""

from __future__ import annotations

import bisect
import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Callable

import torch
import tqdm

from sglang.srt.afd.afd_config import (
    AFDRole,
    is_direct_afd_backend,
)
from sglang.srt.afd.afd_handshake import (
    AFDFFNCapacityPlan,
)
from sglang.srt.afd.layers import AFDMoELayer, collect_afd_moe_layers
from sglang.srt.distributed.parallel_state import graph_capture
from sglang.srt.layers.compute_sm import compute_sm_budget
from sglang.srt.layers.dp_attention import set_is_extend_in_batch
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.srt.model_executor.runner.base_cuda_graph_runner import freeze_gc
from sglang.srt.model_executor.runner_backend.full_cuda_graph_backend import (
    FullCudaGraphBackend,
)
from sglang.srt.model_executor.runner_utils.capture_mode import model_capture_mode
from sglang.srt.model_executor.runner_utils.deepep_adapter import (
    DeepEPCudaGraphRunnerAdapter,
)
from sglang.srt.runtime_context import get_disagg, get_exec
from sglang.srt.utils import get_available_gpu_memory, get_bool_env_var, is_hip

logger = logging.getLogger(__name__)
_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and is_hip()

if TYPE_CHECKING:
    from sglang.srt.afd.direct_deepep import AFDDirectDeepEPDispatchOutput
    from sglang.srt.managers.scheduler import Scheduler


class _RoutedScalingMode(Enum):
    TOPK_WEIGHTS = "topk_weights"
    RUNNER = "runner"
    OUTPUT = "output"


def _resolve_routed_scaling_mode(layer: AFDMoELayer) -> _RoutedScalingMode:
    if layer.routed_scaling_factor in (None, 1.0):
        return _RoutedScalingMode.RUNNER

    if layer.experts.should_fuse_routed_scaling_factor_in_topk:
        return _RoutedScalingMode.TOPK_WEIGHTS

    from sglang.srt.layers.moe.ep_moe.layer import DeepEPMoE

    if isinstance(layer.experts, DeepEPMoE):
        return _RoutedScalingMode.RUNNER if _use_aiter else _RoutedScalingMode.OUTPUT

    from sglang.srt.layers.quantization.mxfp4_flashinfer_trtllm_moe import (
        requires_post_expert_routed_scaling,
    )

    if requires_post_expert_routed_scaling(layer.experts):
        return _RoutedScalingMode.OUTPUT
    return _RoutedScalingMode.RUNNER


@dataclass
class _GraphBuffers:
    hidden_states: torch.Tensor
    topk_ids: torch.Tensor
    topk_weights: torch.Tensor
    router_logits: torch.Tensor


class AFDFFNCompute:
    """Expert compute with an independent decode CUDA Graph policy."""

    def __init__(self, scheduler: Scheduler):
        model_runner = scheduler.tp_worker.model_runner
        # FullCudaGraphBackend binds these two attributes from its runner.
        self.model_runner = model_runner
        self.device_module = scheduler.device_module
        self._dtype = model_runner.dtype
        self._layers = collect_afd_moe_layers(model_runner.model, AFDRole.FFN)
        self._scaling_modes = [
            _resolve_routed_scaling_mode(layer) for layer in self._layers
        ]
        self._device = next(self._layers[0].experts.parameters()).device
        self._device_name = model_runner.device
        self._gpu_id = scheduler.ps.gpu_id
        self._tp_group = scheduler.tp_group
        self._disable_padding = get_exec().graph.disable_cuda_graph_padding
        self._enable_cudagraph_gc = get_exec().graph.enable_cudagraph_gc
        self._enable_memory_saver = get_exec().features.enable_memory_saver
        direct_backend = is_direct_afd_backend(get_disagg().afd_backend)
        self._scattered = get_disagg().afd_ffn_moe_backend == "deepep"
        self._deepep_adapter = (
            DeepEPCudaGraphRunnerAdapter()
            if self._scattered and not direct_backend
            else None
        )
        self._uses_deepep_low_latency = (
            self._scattered and get_exec().moe.deepep_mode != "normal"
        )
        self._deepep_dispatch_limit: int | None = None
        self._max_graph_capacity = 0
        self._graph_buckets: list[int] = []
        self._graph_buffers: dict[tuple[int, int, int], _GraphBuffers] = {}
        self._graph_buffer_capacity = 0
        self._graph_backend: FullCudaGraphBackend | None = None
        self._graph_compute_num_sms: int | None = None

    @property
    def layers(self) -> list[AFDMoELayer]:
        return self._layers

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def enable_memory_saver(self) -> bool:
        return self._enable_memory_saver

    @property
    def enable_cudagraph_gc(self) -> bool:
        return self._enable_cudagraph_gc

    def configure_native_deepep_dispatch_limit(
        self,
        dispatch_limit: int | None,
    ) -> None:
        if not self._uses_deepep_low_latency:
            return
        if dispatch_limit is None:
            raise RuntimeError("AFD DeepEP low-latency capacity was not derived")

        for layer in self._layers:
            for dispatcher in layer.experts.dispatcher._inners:
                if dispatcher.deepep_mode.enable_low_latency():
                    dispatcher._low_latency_dispatcher.num_max_dispatch_tokens_per_rank = dispatch_limit
                if dispatcher.deepep_mode.enable_normal():
                    dispatcher._normal_dispatcher.num_max_dispatch_tokens_per_rank = (
                        dispatch_limit
                    )
        self._deepep_dispatch_limit = dispatch_limit

    def configure_staged_capacity(self, plan: AFDFFNCapacityPlan) -> None:
        graph_buckets = list(plan.graph_token_buckets)
        self._max_graph_capacity = graph_buckets[-1] if graph_buckets else 0
        self._graph_buckets = (
            graph_buckets
            if not self._scattered or get_disagg().afd_attn_instances > 1
            else []
        )

        if self._uses_deepep_low_latency:
            self.configure_native_deepep_dispatch_limit(plan.deepep_dispatch_limit)

        if self._tp_group.rank_in_group == 0:
            logger.info(
                "AFD ffn capacity derived from attn handshakes: "
                "graph_buckets=%s deepep_dispatch_limit=%s",
                graph_buckets,
                self._deepep_dispatch_limit,
            )

    def _prepare_graph_capture(self, max_tokens: int) -> None:
        self._pre_initialize_flashinfer_allreduce_workspace(self.model_runner)
        self._allocate_graph_buffers(max_tokens)

    def capture_compute_graphs(self) -> None:
        if not self._graph_buckets or self._graph_backend is not None:
            return
        self._prepare_graph_capture(self._graph_buckets[-1])
        self._graph_backend = FullCudaGraphBackend(
            self,
            enable_memory_saver=self._enable_memory_saver,
        )
        with compute_sm_budget(self._graph_compute_num_sms):
            self._capture(self._enable_cudagraph_gc)

    def run(
        self,
        layer_index: int,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        router_logits: torch.Tensor,
        allow_graph: bool,
        *,
        defer_output_scaling: bool = False,
    ) -> torch.Tensor:
        num_tokens = hidden_states.shape[0]
        graph_capacity = (
            self.graph_capacity_for(num_tokens)
            if allow_graph and not self._scattered and not defer_output_scaling
            else None
        )
        if graph_capacity is not None:
            return self.run_graph(
                layer_index,
                graph_capacity,
                hidden_states,
                topk_ids,
                topk_weights,
            )

        output = self._run_experts(
            layer_index,
            hidden_states,
            topk_ids,
            topk_weights,
            router_logits,
            defer_output_scaling=defer_output_scaling,
        )
        self._validate_output(self._layers[layer_index], output, num_tokens)
        return output

    def run_graph(
        self,
        layer_index: int,
        capacity: int,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> torch.Tensor:
        if self._graph_backend is None:
            raise RuntimeError("AFD compute-only graph is not initialized")
        layer = self._layers[layer_index]
        buffers = self._buffers_for(layer)
        num_tokens = hidden_states.shape[0]
        if num_tokens > capacity:
            raise RuntimeError(
                f"AFD compute graph capacity {capacity} cannot hold {num_tokens} tokens"
            )
        buffers.hidden_states[:num_tokens].copy_(hidden_states)
        buffers.topk_ids[:num_tokens].copy_(topk_ids)
        buffers.topk_weights[:num_tokens].copy_(topk_weights)
        if num_tokens < capacity:
            buffers.hidden_states[num_tokens:capacity].zero_()
            buffers.topk_ids[num_tokens:capacity].fill_(-1 if self._scattered else 0)
            buffers.topk_weights[num_tokens:capacity].zero_()

        if self._deepep_adapter is not None:
            self._deepep_adapter.replay()
        with self._graph_backend.replay_session():
            output = self._graph_backend.replay(
                (layer.layer_id, capacity), static_forward_batch=None
            )
        return output[:num_tokens]

    def run_direct_deepep(
        self,
        layer_index: int,
        dispatch_output: AFDDirectDeepEPDispatchOutput,
    ) -> torch.Tensor:
        from sglang.srt.layers.moe.token_dispatcher.deepep import (
            DeepEPLLDispatchOutput,
            DeepEPNormalDispatchOutput,
        )

        layer = self._layers[layer_index]
        if dispatch_output.masked_m is not None:
            expert_capacity = dispatch_output.expert_capacity
            combine_input = layer.experts.run_moe_core(
                DeepEPLLDispatchOutput(
                    hidden_states=dispatch_output.hidden_states,
                    hidden_states_scale=dispatch_output.hidden_states_scale,
                    topk_ids=dispatch_output.topk_ids,
                    topk_weights=dispatch_output.topk_weights,
                    masked_m=dispatch_output.masked_m,
                    expected_m=expert_capacity,
                )
            )
            output = combine_input.hidden_states
            num_local_experts = dispatch_output.masked_m.shape[0]
            expected_shape = (
                num_local_experts,
                expert_capacity,
                layer.hidden_size,
            )
            if tuple(output.shape) != expected_shape:
                raise RuntimeError(
                    f"AFD layer {layer.layer_id} NVFP4 output mismatch: "
                    f"expected shape={expected_shape}, got {tuple(output.shape)}"
                )
            output = output.reshape(-1, layer.hidden_size)
            self._validate_output(layer, output, output.shape[0])
            return output

        num_recv_tokens = dispatch_output.hidden_states.shape[0]
        if num_recv_tokens == 0:
            return dispatch_output.hidden_states.new_empty(
                dispatch_output.hidden_states.shape,
                dtype=self._dtype,
            )

        combine_input = layer.experts.run_moe_core(
            DeepEPNormalDispatchOutput(
                hidden_states=dispatch_output.hidden_states,
                hidden_states_scale=dispatch_output.hidden_states_scale,
                topk_ids=dispatch_output.topk_ids,
                topk_weights=dispatch_output.topk_weights,
                num_recv_tokens_per_expert=dispatch_output.expert_psum,
            )
        )
        output = combine_input.hidden_states
        self._validate_output(
            layer,
            output,
            num_recv_tokens,
        )
        return output

    def graph_capacity_for(self, num_tokens: int) -> int | None:
        if not self._graph_buckets or num_tokens > self._graph_buckets[-1]:
            return None
        index = bisect.bisect_left(self._graph_buckets, num_tokens)
        capacity = self._graph_buckets[index]
        if self._disable_padding and capacity != num_tokens:
            return None
        return capacity

    def _run_experts(
        self,
        layer_index: int,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        router_logits: torch.Tensor,
        *,
        defer_output_scaling: bool = False,
    ) -> torch.Tensor:
        layer = self._layers[layer_index]
        scaling_mode = self._scaling_modes[layer_index]
        if scaling_mode is _RoutedScalingMode.TOPK_WEIGHTS:
            topk_weights.mul_(layer.routed_scaling_factor)

        output = layer.experts(
            hidden_states,
            StandardTopKOutput(
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                router_logits=router_logits,
            ),
        )
        if scaling_mode is _RoutedScalingMode.OUTPUT and not defer_output_scaling:
            output.mul_(layer.routed_scaling_factor)
        return output

    def stage_deferred_output(
        self,
        layer_index: int,
        destination: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        layer = self._layers[layer_index]
        if self._scaling_modes[layer_index] is _RoutedScalingMode.OUTPUT:
            torch.mul(
                output,
                layer.routed_scaling_factor,
                out=destination,
            )
            return
        destination.copy_(output)

    def finalize_deferred_output_in_place(
        self,
        layer_index: int,
        output: torch.Tensor,
    ) -> None:
        if self._scaling_modes[layer_index] is _RoutedScalingMode.OUTPUT:
            output.mul_(self._layers[layer_index].routed_scaling_factor)

    def _capture(self, enable_cudagraph_gc: bool) -> None:
        assert self._graph_backend is not None
        rank = self._tp_group.rank_in_group
        graph_count = len(self._layers) * len(self._graph_buckets)
        tic = time.perf_counter()
        before_mem = get_available_gpu_memory(
            self._device_name, self._gpu_id, empty_cache=False
        )
        if rank == 0:
            logger.info(
                "Capture AFD ffn decode CUDA graph begin. bs=%s graphs=%d "
                "avail mem=%.2f GB",
                self._graph_buckets,
                graph_count,
                before_mem,
            )

        capture_items = [
            (capacity, layer_index)
            for capacity in reversed(self._graph_buckets)
            for layer_index in range(len(self._layers))
        ]
        capture_range = tqdm.tqdm(capture_items) if rank == 0 else iter(capture_items)
        with (
            torch.inference_mode(),
            model_capture_mode(),
            freeze_gc(enable_cudagraph_gc),
            graph_capture() as ctx,
        ):
            with self._graph_backend.capture_session(ctx.stream):
                for capacity, layer_index in capture_range:
                    layer = self._layers[layer_index]
                    if rank == 0:
                        capture_range.set_description(
                            "Capturing AFD expert graphs "
                            f"(bs={capacity} layer={layer.layer_id})"
                        )
                    self._capture_one(layer_index, capacity)

        self.device_module.synchronize()
        self._tp_group.barrier()
        if rank == 0:
            after_mem = get_available_gpu_memory(
                self._device_name, self._gpu_id, empty_cache=False
            )
            logger.info(
                "Capture AFD ffn decode CUDA graph end. elapsed=%.2f s, "
                "graphs=%d, mem usage=%.2f GB, avail mem=%.2f GB",
                time.perf_counter() - tic,
                graph_count,
                before_mem - after_mem,
                after_mem,
            )

    @staticmethod
    def _pre_initialize_flashinfer_allreduce_workspace(model_runner) -> None:
        if get_exec().comm.flashinfer_allreduce_fusion_backend is None:
            return

        from sglang.srt.layers.communicator import FUSE_ALLREDUCE_MAX_BATCH_SIZE
        from sglang.srt.layers.flashinfer_comm_fusion import (
            pre_initialize_workspaces,
        )

        pre_initialize_workspaces(
            max_token_num=FUSE_ALLREDUCE_MAX_BATCH_SIZE,
            hidden_dim=model_runner.model_config.hidden_size,
            dtype=model_runner.dtype,
        )

    def _allocate_graph_buffers(self, max_tokens: int) -> None:
        # Captured graphs keep references into these buffers, so they are
        # allocated once (sized for the largest capacity any session can
        # request) and shared by every later capture.
        if self._graph_buffer_capacity:
            if max_tokens > self._graph_buffer_capacity:
                raise RuntimeError(
                    "AFD ffn graph buffers were sized for "
                    f"{self._graph_buffer_capacity} tokens but a later "
                    f"capture needs {max_tokens}"
                )
            return
        max_tokens = max(max_tokens, self._max_graph_capacity)
        self._graph_buffer_capacity = max_tokens
        schemas = {
            (layer.hidden_size, layer.top_k, layer.num_experts)
            for layer in self._layers
        }
        for hidden_size, top_k, num_experts in sorted(schemas):
            self._graph_buffers[(hidden_size, top_k, num_experts)] = _GraphBuffers(
                hidden_states=torch.zeros(
                    (max_tokens, hidden_size), dtype=self._dtype, device=self._device
                ),
                topk_ids=torch.zeros(
                    (max_tokens, top_k), dtype=torch.int32, device=self._device
                ),
                topk_weights=torch.empty(
                    (max_tokens, top_k), dtype=torch.float32, device=self._device
                ),
                router_logits=torch.zeros(
                    (max_tokens, num_experts),
                    dtype=torch.float32,
                    device=self._device,
                ),
            )

    def _capture_one(
        self,
        layer_index: int,
        capacity: int,
    ) -> None:
        assert self._graph_backend is not None
        layer = self._layers[layer_index]
        buffers = self._buffers_for(layer)
        if self._deepep_adapter is not None:
            set_is_extend_in_batch(False)
            self._deepep_adapter.capture(is_extend_in_batch=False)

        def reset_inputs() -> None:
            self._prepare_capture_inputs(buffers, layer, capacity)

        reset_inputs()
        self._graph_backend.capture_one(
            (layer.layer_id, capacity),
            lambda: self._run_graph_forward(layer_index, buffers, capacity),
            post_warmup_hook=reset_inputs,
        )

    def _prepare_capture_inputs(
        self, buffers: _GraphBuffers, layer: AFDMoELayer, capacity: int
    ) -> None:
        buffers.hidden_states[:capacity].zero_()
        buffers.topk_ids[:capacity].zero_()
        buffers.topk_weights[:capacity].fill_(1.0 / layer.top_k)
        buffers.router_logits[:capacity].zero_()

    def prepare_integrated_capture_inputs(self, capacity: int) -> None:
        prepared = set()
        for layer in self._layers:
            key = (layer.hidden_size, layer.top_k, layer.num_experts)
            if key in prepared:
                continue
            prepared.add(key)
            self._prepare_capture_inputs(
                self._graph_buffers[key],
                layer,
                capacity,
            )

    def run_integrated_layer(
        self,
        layer_index: int,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        wire_capacity: int,
        compute_capacity: int,
        *,
        defer_output_scaling: bool = False,
        stage_inputs: bool = True,
    ) -> torch.Tensor:
        layer = self._layers[layer_index]
        buffers = self._buffers_for(layer)
        if stage_inputs:
            buffers.hidden_states[:wire_capacity].copy_(hidden_states)
            buffers.topk_ids[:wire_capacity].copy_(topk_ids)
            buffers.topk_weights[:wire_capacity].copy_(topk_weights)
            if wire_capacity < compute_capacity:
                buffers.hidden_states[wire_capacity:compute_capacity].zero_()
                buffers.topk_ids[wire_capacity:compute_capacity].fill_(
                    -1 if self._scattered else 0
                )
                buffers.topk_weights[wire_capacity:compute_capacity].zero_()
            expert_hidden_states = buffers.hidden_states[:compute_capacity]
            expert_topk_ids = buffers.topk_ids[:compute_capacity]
            expert_topk_weights = buffers.topk_weights[:compute_capacity]
        else:
            if wire_capacity > compute_capacity:
                raise ValueError(
                    "AFD direct FFN inputs exceed compute capacity: "
                    f"wire={wire_capacity}, "
                    f"compute={compute_capacity}"
                )
            direct_inputs = (hidden_states, topk_ids, topk_weights)
            if any(tensor.shape[0] != compute_capacity for tensor in direct_inputs):
                raise ValueError(
                    "AFD direct FFN inputs must expose the full compute capacity: "
                    f"wire={wire_capacity}, compute={compute_capacity}, "
                    f"shapes={[tuple(tensor.shape) for tensor in direct_inputs]}"
                )
            if wire_capacity < compute_capacity:
                hidden_states[wire_capacity:compute_capacity].zero_()
                topk_ids[wire_capacity:compute_capacity].fill_(
                    -1 if self._scattered else 0
                )
                topk_weights[wire_capacity:compute_capacity].zero_()
            expert_hidden_states = hidden_states
            expert_topk_ids = topk_ids
            expert_topk_weights = topk_weights

        output = self._run_experts(
            layer_index,
            expert_hidden_states,
            expert_topk_ids,
            expert_topk_weights,
            buffers.router_logits[:compute_capacity],
            defer_output_scaling=defer_output_scaling,
        )
        self._validate_output(layer, output, compute_capacity)
        return output

    def _run_graph_forward(
        self, layer_index: int, buffers: _GraphBuffers, capacity: int
    ) -> torch.Tensor:
        output = self._run_experts(
            layer_index,
            buffers.hidden_states[:capacity],
            buffers.topk_ids[:capacity],
            buffers.topk_weights[:capacity],
            buffers.router_logits[:capacity],
        )
        self._validate_output(self._layers[layer_index], output, capacity)
        return output

    def _buffers_for(self, layer: AFDMoELayer) -> _GraphBuffers:
        return self._graph_buffers[(layer.hidden_size, layer.top_k, layer.num_experts)]

    def _validate_output(
        self, layer: AFDMoELayer, output: torch.Tensor, num_tokens: int
    ) -> None:
        expected_shape = (num_tokens, layer.hidden_size)
        if (
            tuple(output.shape) != expected_shape
            or output.dtype != self._dtype
            or output.device != self._device
        ):
            raise RuntimeError(
                f"AFD layer {layer.layer_id} output mismatch: "
                f"expected shape={expected_shape}, dtype={self._dtype}, "
                f"device={self._device}; got shape={tuple(output.shape)}, "
                f"dtype={output.dtype}, device={output.device}"
            )


class _AFDFFNDecodeStepGraph:
    def __init__(
        self,
        compute: AFDFFNCompute,
        run_step: Callable[[tuple[int, ...], tuple[int, ...]], None],
        *,
        direct_data_plane: bool = False,
    ):
        self._compute = compute
        self._run_step = run_step
        self._backend = FullCudaGraphBackend(
            compute,
            enable_memory_saver=compute.enable_memory_saver,
        )
        self._deepep_adapter = (
            None if direct_data_plane else DeepEPCudaGraphRunnerAdapter()
        )
        self._keys: dict[
            tuple[int, ...],
            tuple[str, tuple[int, ...], tuple[int, ...]],
        ] = {}

    def capture_session(self, stream: torch.cuda.Stream):
        return self._backend.capture_session(stream)

    def capture_variant(
        self,
        wire_capacities: tuple[int, ...],
        compute_capacities: tuple[int, ...],
    ) -> None:
        key = (
            "afd_ffn_decode_step",
            wire_capacities,
            compute_capacities,
        )
        set_is_extend_in_batch(False)
        reset_inputs = None
        if self._deepep_adapter is not None:
            max_compute_capacity = max(compute_capacities)
            self._compute.prepare_integrated_capture_inputs(max_compute_capacity)
            self._deepep_adapter.capture(is_extend_in_batch=False)

            def reset_inputs() -> None:
                self._compute.prepare_integrated_capture_inputs(max_compute_capacity)

        self._backend.capture_one(
            key,
            lambda: self._run_step(wire_capacities, compute_capacities),
            post_warmup_hook=reset_inputs,
        )
        self._keys[wire_capacities] = key

    def replay(self, wire_capacities: tuple[int, ...]) -> bool:
        key = self._keys.get(wire_capacities)
        if key is None:
            return False
        if self._deepep_adapter is not None:
            self._deepep_adapter.replay()
        with self._backend.replay_session():
            self._backend.replay(key, static_forward_batch=None)
        return True
