"""Two-microbatch role operations executed by the native TBO scheduler."""

from __future__ import annotations

import logging
from typing import Any

from .config import AFDConfig
from .connector import AFDConnector
from .contracts import (
    AFDError,
    AFDModelAdapter,
    AFDRole,
    AFDShapeFactory,
    AFDStepDescriptor,
)
from .ffn_backend import direct_step_reason, select_direct_step, select_m2n_step
from .layer_executor import execute_layer_operations
from .model_adapters.base import stage_forward_context

logger = logging.getLogger(__name__)

_RELEASE_BACKING_REASONS = frozenset(
    (
        "AFD_TYPED_EAGER_HBM_LIMIT",
        "AFD_TYPED_EAGER_CAPTURE_FAILED",
        "AFD_TYPED_EAGER_PARTIAL_CAPTURE_ROLLBACK",
        "AFD_TYPED_EAGER_METADATA_DRIFT",
        "AFD_TYPED_EAGER_REPLAY_FAILED",
    )
)


def _dtype_name(tensor: Any) -> str:
    value = str(tensor.dtype)
    return value[6:] if value.startswith("torch.") else value


def _stage_participation(
    lane_stage_rows: tuple[tuple[int, ...], ...],
) -> tuple[bool, ...]:
    """Decide per stage whether every lane runs it.

    The FFN role merges rows across lanes, so this cannot be lane-local: an idle
    lane skipping a stage a peer still runs would leave that collective
    unmatched. Every lane sees the same matrix, so every lane decides the same.
    """

    return tuple(
        any(vector[stage] for vector in lane_stage_rows)
        for stage in range(len(lane_stage_rows[0]))
    )


def _finish_backend_warmup(connector: AFDConnector) -> None:
    backend = connector.transport.ffn_backend
    if not backend.needs_warmup_barrier:
        return
    import torch

    # Full backend runs its warmups through this closure, then captures it.
    # Complete the two-sided work before capture; never record a CPU barrier.
    if (
        connector.graph_strategy.capturing
        and not torch.cuda.is_current_stream_capturing()
    ):
        backend.finish_warmup()


class AFDAttentionPipeline:
    """Attention role: local attention, A2E, E2A, then layer postprocess."""

    def __init__(
        self,
        *,
        adapter: AFDModelAdapter,
        connector: AFDConnector,
        config: AFDConfig,
        shape_factory: AFDShapeFactory,
    ) -> None:
        if adapter.role != AFDRole.ATTENTION:
            raise AFDError("AFD_ATTENTION_PIPELINE_ROLE_INVALID")
        self._adapter = adapter
        self._connector = connector
        self._config = config
        self._shape_factory = shape_factory
        self._step_id = 0
        self._startup_capture = False
        # Not True or False, so the first step's verdict is always logged.
        self._last_eligible: bool | None = None
        self._last_backend_selection = None

    def capture_startup(self, runner: Any) -> None:
        """Reuse native dummy decode inputs; only coordinate the role captures."""
        import torch

        from sglang.srt.model_executor.forward_batch_info import ForwardMode

        graph = self._connector.graph_strategy
        sizes = graph.capture_sizes
        self._startup_capture = True
        try:
            if sizes:
                buffers = runner._alloc_dummy_decode_buffers(
                    max_bs=2 * sizes[-1], allocate_logits_buffer=False
                )
                with torch.no_grad():
                    for width in reversed(sizes):
                        runner._dummy_run(
                            2 * width,
                            forward_mode_override=ForwardMode.DECODE,
                            buffers=buffers,
                        )
                torch.cuda.synchronize()
            graph.finish_capture()
            self._connector.transport.capture_ready()
            from sglang.srt.runtime_context import get_parallel

            get_parallel().tp_group.barrier()
            logger.info("AFD_STARTUP_CAPTURE_READY stage_sizes=%s", sizes)
        finally:
            self._startup_capture = False

    def execute(
        self,
        *,
        hidden_states: Any,
        residual: Any,
        positions: Any,
        forward_batch: Any,
    ) -> tuple[Any, Any]:
        from sglang.srt.layers.logits_processor import get_in_autotune_dummy_run

        from .contracts import validate_non_speculative_batch

        mode = forward_batch.forward_mode
        validate_non_speculative_batch(forward_batch)
        decode_or_idle = mode.is_decode_or_idle()
        global_extend = forward_batch.is_extend_in_batch
        if type(decode_or_idle) is not bool or type(global_extend) is not bool:
            raise AFDError("AFD_ATTENTION_FORWARD_MODE_INVALID")
        # Batch mode is independent of graph eligibility: an observing decode
        # remains decode, and extend must be propagated even without DP gather.
        local_extend = False if decode_or_idle else mode.is_extend()
        if type(local_extend) is not bool or not (decode_or_idle or local_extend):
            raise AFDError("AFD_ATTENTION_FORWARD_MODE_UNSUPPORTED", f"mode={mode!r}")
        is_extend_in_batch = local_extend or global_extend
        stages = self._adapter.split_step(
            hidden_states=hidden_states,
            residual=residual,
            positions=positions,
            forward_batch=forward_batch,
            stages=self._config.stages,
        )
        lane = self._connector.transport.lane
        rows = tuple(stage.rows for stage in stages)
        lane_rows = self._adapter.lane_stage_rows(
            forward_batch=forward_batch,
            local_rows=rows,
            lanes=self._config.attention_lane_count,
            lane=lane,
        )
        # The shared row matrix and DP extend flag keep every A/F rank on
        # the same communication sequence. A zero stage makes the whole wave
        # eager. Autotune dummy forwards must also finish outside capture.
        eligible = (
            decode_or_idle
            and not get_in_autotune_dummy_run()
            and not is_extend_in_batch
            and all(all(stage_rows) for stage_rows in lane_rows)
        )
        uses_m2n = select_m2n_step(self._config, lane_rows, is_extend_in_batch)
        backend = self._connector.transport.ffn_backend
        reason = direct_step_reason(
            self._config, lane_rows, is_extend_in_batch, backend=backend
        )
        uses_direct = reason == "direct"
        if reason == "backend_not_qualified":
            reason = getattr(
                self._connector.transport, "direct_selection_reason", reason
            )
        if backend is not None and not (uses_direct or uses_m2n):
            eligible = False
        if reason != self._last_backend_selection:
            self._last_backend_selection = reason
            logger.info(
                "AFD_FFN_SELECTION step_id=%d backend=%s reason=%s lane_rows=%r",
                self._step_id,
                "m2n" if uses_m2n else "nccl_direct" if uses_direct else "fixed",
                reason,
                lane_rows,
            )
        # The usage snapshot only carries a cumulative eligible_steps, which cannot
        # say whether the lanes disagreed on one particular step -- exactly what a
        # deadlock post-mortem needs. Log transitions only, so this is a handful of
        # lines per run rather than one per step.
        if eligible is not self._last_eligible:
            self._last_eligible = eligible
            logger.info(
                "AFD_ELIGIBILITY_FLIP step_id=%d eligible=%s mode=%s "
                "local_rows=%r lane_rows=%r",
                self._step_id,
                eligible,
                forward_batch.forward_mode,
                rows,
                lane_rows,
            )
        shape = self._shape_factory(
            lane=lane,
            lane_rows=lane_rows,
            hidden_size=self._adapter.hidden_size,
            dtype=_dtype_name(hidden_states),
            config=self._config,
        )
        descriptor = AFDStepDescriptor(
            kind="CAPTURE" if self._startup_capture else "STEP",
            step_id=self._step_id,
            lane_stage_rows=lane_rows,
            hidden_size=self._adapter.hidden_size,
            dtype=_dtype_name(hidden_states),
            num_layers=self._adapter.num_layers,
            graph_eligible=eligible,
            is_extend_in_batch=is_extend_in_batch,
            uses_direct=uses_direct,
            uses_m2n=uses_m2n,
        )
        if uses_direct:
            backend.prepare_step(shape=shape)
        self._connector.transport.begin_step(descriptor)
        backing_hbm_bytes = (
            sum(shape.bucket_rows)
            * self._adapter.hidden_size
            * hidden_states.element_size()
        )
        self._connector.graph_strategy.begin_step(
            capture=self._startup_capture,
            step_id=self._step_id,
            shape=shape,
            eligible=eligible,
            backing_hbm_bytes=backing_hbm_bytes if eligible else 0,
        )
        self._step_id += 1
        retain = self._connector.graph_strategy.retains_backing
        buffers = self._connector.transport.acquire_buffers(
            key=shape.digest,
            capacities=shape.bucket_rows,
            hidden_size=self._adapter.hidden_size,
            dtype=hidden_states.dtype,
            retain=retain,
        )
        if retain and not self._connector.graph_strategy.ensure_backing_hbm(
            retained_hbm_bytes=self._connector.transport.buffer_hbm_bytes(
                key=shape.digest
            )
        ):
            self._connector.transport.release_buffers(key=shape.digest)
            retain = False
            buffers = self._connector.transport.acquire_buffers(
                key=shape.digest,
                capacities=shape.bucket_rows,
                hidden_size=self._adapter.hidden_size,
                dtype=hidden_states.dtype,
                retain=False,
            )
        # A whole captured step is one fixed-shape program, so every tensor inside
        # it is the bucket's padded width -- not only the entry tensors. Row slices
        # make the first layer's output narrow to real rows while the residual it
        # is added to stays padded, which the fused rmsnorm rejects.
        # retain is what says this step is being graphed. The two roles cannot
        # disagree here: the strategy only runs eagerly for reasons both reach
        # from the shared descriptor, and refuses the rest rather than quietly
        # exchanging a different number of rows than its peer.
        recv_buffers = (
            list(buffers)
            if retain
            else [buffer[: stage.rows] for stage, buffer in zip(stages, buffers)]
        )
        end_reason = None
        try:
            participates = _stage_participation(lane_rows)
            self._run_step_as_unit(
                stages=stages,
                recv_buffers=recv_buffers,
                bucket_rows=shape.bucket_rows,
                participates=participates,
                uses_backend=uses_direct or uses_m2n,
            )
        finally:
            end_reason = self._connector.graph_strategy.end_step()
            if (
                retain
                and end_reason is not None
                and end_reason.value in _RELEASE_BACKING_REASONS
            ):
                self._connector.transport.release_buffers(key=shape.digest)
            if self._connector.shutdown_requested:
                self.close()
        return self._adapter.join_step(stages=stages)

    def _run_layers(
        self,
        *,
        stages: list[Any],
        recv_buffers: list[Any],
        participates: tuple[bool, ...],
        uses_backend: bool = False,
    ) -> None:
        recv_events: list[Any] = [None] * len(stages)
        shared_outputs: list[Any] = [None] * len(stages)
        first_backend_layer = (
            self._connector.transport.ffn_backend.first_layer
            if uses_backend
            else self._adapter.num_layers
        )

        def run_layer(*, state, layer, index):
            stage, recv_buffer = stages[index], recv_buffers[index]

            if participates[stage.index]:
                if layer:
                    # Preserve the whole-role stream dependency in both
                    # eager execution and capture; do not serialize the host
                    # at every A/F layer boundary.
                    self._connector.transport.wait(recv_events[index])
                    if stage.rows:
                        stage.hidden_states, stage.residual = (
                            self._adapter.finish_layer(
                                layer=layer - 1,
                                stage=stage,
                                ffn_output=(
                                    recv_buffer
                                    if shared_outputs[index] is None
                                    else recv_buffer + shared_outputs[index]
                                ),
                                residual=stage.residual,
                            )
                        )
                if stage.rows:
                    with stage_forward_context(stage.forward_batch):
                        self._adapter.prepare_stage(stage=stage)
                        stage.hidden_states, stage.residual = (
                            self._adapter.local_compute(
                                layer=layer,
                                stage=stage,
                                positions=stage.positions,
                                hidden_states=stage.hidden_states,
                                residual=stage.residual,
                            )
                        )
                # Dispatch before the next stage's local compute, so the FFN
                # role works on this stage while this role runs the next one.
                if layer >= first_backend_layer:
                    if stage.rows:
                        with stage_forward_context(stage.forward_batch):
                            topk = self._adapter.inner.layers[layer].mlp.route(
                                stage.hidden_states
                            )
                        ids, weights = topk.topk_ids, topk.topk_weights
                    else:
                        import torch

                        k = self._connector.transport.ffn_backend.top_k
                        ids = torch.empty(
                            (0, k), dtype=torch.int64, device=stage.hidden_states.device
                        )
                        weights = torch.empty(
                            (0, k),
                            dtype=torch.float32,
                            device=stage.hidden_states.device,
                        )
                    recv_events[index] = self._connector.transport.ffn_backend.dispatch(
                        hidden=stage.hidden_states,
                        ids=ids,
                        weights=weights,
                        stage=index,
                        output=recv_buffer,
                    )
                    # Shared expert stays local to each A lane. Both branches
                    # only read hidden; join once after the routed return event.
                    shared = getattr(
                        self._adapter.inner.layers[layer].mlp, "shared_experts", None
                    )
                    with stage_forward_context(stage.forward_batch):
                        shared_outputs[index] = (
                            shared(stage.hidden_states) if shared is not None else None
                        )
                    return
                shared_outputs[index] = None
                if layer == 0 and self._adapter.requires_token_ids:
                    self._connector.transport.dispatch(
                        stage.forward_batch.input_ids.reshape(-1, 1)
                    )
                self._connector.transport.dispatch(stage.hidden_states)
                recv_events[index] = self._connector.transport.receive_return(
                    recv_buffer
                )

        execute_layer_operations(self._adapter.num_layers, run_layer)
        for index, (stage, ffn_output) in enumerate(zip(stages, recv_buffers)):
            if participates[stage.index]:
                self._connector.transport.wait(recv_events[index])
                if stage.rows:
                    stage.hidden_states, stage.residual = self._adapter.finish_layer(
                        layer=self._adapter.num_layers - 1,
                        stage=stage,
                        ffn_output=(
                            ffn_output
                            if shared_outputs[index] is None
                            else ffn_output + shared_outputs[index]
                        ),
                        residual=stage.residual,
                    )

    def _run_step_as_unit(
        self,
        *,
        stages: list[Any],
        recv_buffers: list[Any],
        bucket_rows: tuple[int, ...],
        participates: tuple[bool, ...],
        uses_backend: bool = False,
    ) -> None:
        """Hand the whole layer loop to a strategy that captures it as one graph.

        Only the step's entry tensors are staged. Intermediate tensors live in
        the graph's pool until the bucket closes.
        """

        residual_present = tuple(stage.residual is not None for stage in stages)
        stage_args = tuple(
            (
                (stage.positions, stage.hidden_states)
                if stage.residual is None
                else (stage.positions, stage.hidden_states, stage.residual)
            )
            for stage in stages
        )
        guards = (
            tuple(
                self._adapter.metadata_guard(
                    stage=stage,
                    bucket_rows=bucket_rows[stage.index],
                )
                for stage in stages
            )
            if self._connector.graph_strategy.capturing
            else ()
        )

        def compute(
            staged: tuple[tuple[Any, ...], ...],
        ) -> tuple[tuple[Any, ...], ...]:
            for stage, values, has_residual in zip(stages, staged, residual_present):
                stage.positions = values[0]
                stage.hidden_states = values[1]
                stage.residual = values[2] if has_residual else None
            self._run_layers(
                stages=stages,
                recv_buffers=recv_buffers,
                participates=participates,
                uses_backend=uses_backend,
            )
            # Last thing in the region: the send side never rejoins on its own,
            # and an unjoined stream leaves its work outside the graph's order.
            self._connector.transport.rejoin_streams()
            if uses_backend:
                _finish_backend_warmup(self._connector)
            return tuple((stage.hidden_states, stage.residual) for stage in stages)

        result = self._connector.graph_strategy.execute_step(
            stage_args=stage_args,
            stage_rows=tuple(stage.rows for stage in stages),
            forward_batches=tuple(stage.forward_batch for stage in stages),
            compute=compute,
            metadata_guards=guards,
        )
        for stage, outputs in zip(stages, result.value):
            hidden, residual = outputs
            # Back to real rows: the region ran at the bucket's padded width, and
            # join_step writes these into the model's own tensors.
            stage.hidden_states = hidden[: stage.rows]
            stage.residual = None if residual is None else residual[: stage.rows]

    def close(self) -> dict[str, Any]:
        return self._connector.close()

    def request_shutdown(self) -> None:
        self._connector.request_shutdown()


class AFDFFNPipeline:
    """FFN role service loop for the matching layer-major/stage-major order."""

    def __init__(
        self,
        *,
        adapter: AFDModelAdapter,
        connector: AFDConnector,
        config: AFDConfig,
        device: Any,
        dtype: Any,
        shape_factory: AFDShapeFactory,
    ) -> None:
        if adapter.role != AFDRole.FFN:
            raise AFDError("AFD_FFN_PIPELINE_ROLE_INVALID")
        self._adapter = adapter
        self._connector = connector
        self._config = config
        self._device = device
        self._dtype = dtype
        self._shape_factory = shape_factory
        adapter.validate_merge_reduce_scatter()

    def run_once(self) -> bool:
        descriptor = self._connector.transport.begin_step(None)
        if descriptor.kind == "CLOSE":
            return False
        if descriptor.kind == "READY":
            self._connector.graph_strategy.finish_capture()
            self._connector.transport.capture_ready()
            logger.info("AFD_STARTUP_CAPTURE_READY role=ffn")
            return True
        self._validate_descriptor(descriptor=descriptor)
        return self._run_descriptor(descriptor)

    def _run_descriptor(self, descriptor: AFDStepDescriptor) -> bool:
        lane = self._connector.transport.lane
        shape = self._shape_factory(
            lane=lane,
            lane_rows=descriptor.lane_stage_rows,
            hidden_size=descriptor.hidden_size,
            dtype=descriptor.dtype,
            config=self._config,
            tokens_per_request=descriptor.tokens_per_request,
        )
        if descriptor.uses_direct:
            self._connector.transport.ffn_backend.prepare_step(shape=shape)
        # One block per (stage, lane) this rank serves, stage-major, which at k == 1
        # is the per-stage vector the symmetric path always allocated.
        capacities = shape.group_bucket_rows(ffn_ordinal=lane)
        group = len(shape.group_lanes(ffn_ordinal=lane))
        input_ids_options = (
            {"include_input_ids": True} if self._adapter.requires_token_ids else {}
        )
        self._connector.graph_strategy.begin_step(
            capture=descriptor.kind == "CAPTURE",
            step_id=descriptor.step_id,
            shape=shape,
            eligible=descriptor.graph_eligible,
            backing_hbm_bytes=(
                sum(capacities)
                * descriptor.hidden_size
                * (2 if descriptor.dtype in ("bfloat16", "float16") else 0)
                + (sum(capacities) * 8 if input_ids_options else 0)
            ),
        )
        retain = self._connector.graph_strategy.retains_backing
        buffers = self._connector.transport.acquire_buffers(
            key=shape.digest,
            capacities=capacities,
            hidden_size=descriptor.hidden_size,
            dtype=self._dtype,
            retain=retain,
            **input_ids_options,
        )
        if retain and not self._connector.graph_strategy.ensure_backing_hbm(
            retained_hbm_bytes=self._connector.transport.buffer_hbm_bytes(
                key=shape.digest
            )
        ):
            self._connector.transport.release_buffers(key=shape.digest)
            retain = False
            buffers = self._connector.transport.acquire_buffers(
                key=shape.digest,
                capacities=capacities,
                hidden_size=descriptor.hidden_size,
                dtype=self._dtype,
                retain=False,
                **input_ids_options,
            )
        stages, recv_buffers = self._make_stages(
            descriptor=descriptor,
            buffers=buffers,
            lane=lane,
            shape=shape,
        )
        # Capture receives the padded whole-role widths; eager uses live rows.
        if retain:
            recv_buffers = [
                buffers[index * group : (index + 1) * group]
                for index in range(len(stages))
            ]
        end_reason = None
        participates = _stage_participation(descriptor.lane_stage_rows)
        try:
            self._run_step_as_unit(
                stages=stages,
                recv_buffers=recv_buffers,
                participates=participates,
                uses_backend=descriptor.uses_direct or descriptor.uses_m2n,
            )
        finally:
            end_reason = self._connector.graph_strategy.end_step()
            if (
                retain
                and end_reason is not None
                and end_reason.value in _RELEASE_BACKING_REASONS
            ):
                self._connector.transport.release_buffers(key=shape.digest)
        return True

    def _run_layers(
        self,
        *,
        stages: list[Any],
        recv_buffers: list[Any],
        participates: tuple[bool, ...],
        num_layers: int | None = None,
    ) -> tuple[tuple[Any, ...], ...]:
        num_layers = self._adapter.num_layers if num_layers is None else num_layers
        last: list[tuple[Any, ...]] = [() for _ in stages]
        active = tuple(index for index, present in enumerate(participates) if present)
        if not active or not num_layers:
            return tuple(last)
        # Only capture supplies the compute-to-receive edge protecting reused
        # stage buffers. Eager/warmup receives remain strictly one at a time.
        prefetch = len(active) == 2 and bool(self._connector.transport.capturing)

        def receive(index: int, layer: int) -> Any:
            if layer == 0 and self._adapter.requires_token_ids:
                stage = stages[index]
                # Warmup and capture call this same Python closure. Re-record
                # the ID gather each invocation; otherwise capture would reuse
                # warmup's tensor without recording its producer.
                stage.merged_input_ids = None
                self._connector.transport.receive_dispatch(
                    tuple(
                        ids[: hidden.shape[0]]
                        for ids, hidden in zip(stage.input_ids, recv_buffers[index])
                    )
                )
            return self._connector.transport.receive_dispatch(recv_buffers[index])

        events: list[Any] = [None] * len(stages)
        events[active[0]] = receive(active[0], 0)

        def run_boundary(*, state, layer, index):
            if not participates[index]:
                return
            stage = stages[index]
            next_index = 1 - index if len(active) == 2 else index
            next_layer = layer + (next_index <= index)
            has_next = next_layer < num_layers
            # One-boundary lookahead includes the layer transition while keeping
            # each communicator's receive/return order identical to its peer.
            if prefetch and has_next:
                events[next_index] = receive(next_index, next_layer)
            self._connector.transport.wait(events[index])
            output, _ = self._adapter.local_compute(
                layer=layer,
                stage=stage,
                hidden_states=recv_buffers[index],
                residual=None,
            )
            self._connector.transport.return_result(output)
            last[index] = output or (stage.graph_output,)
            if not prefetch and has_next:
                events[next_index] = receive(next_index, next_layer)

        execute_layer_operations(num_layers, run_boundary)
        return tuple(last)

    def _run_step_as_unit(
        self,
        *,
        stages: list[Any],
        recv_buffers: list[Any],
        participates: tuple[bool, ...],
        uses_backend: bool = False,
    ) -> None:
        """Hand the whole layer loop to a strategy that captures it as one graph.

        This role stages nothing and extracts nothing: its inputs arrive over the
        wire into the retained receive buffers and its results leave over the
        wire, both from inside the captured region.
        """

        def compute(
            staged: tuple[tuple[Any, ...], ...],
        ) -> tuple[tuple[Any, ...], ...]:
            del staged
            # The last output per stage is returned, not used: this role's results
            # leave over the wire, but a captured region still has to expose one
            # tensor it wrote, or nothing can distinguish it from a graph that
            # captured no work at all.
            if uses_backend:
                first_layer = self._connector.transport.ffn_backend.first_layer
                if first_layer:
                    self._run_layers(
                        stages=stages,
                        recv_buffers=recv_buffers,
                        participates=participates,
                        num_layers=first_layer,
                    )
                    self._connector.transport.rejoin_streams()
                outputs = self._connector.transport.ffn_backend.run_ffn_step()
            else:
                outputs = self._run_layers(
                    stages=stages,
                    recv_buffers=recv_buffers,
                    participates=participates,
                )
            # Last thing in the region: the send side never rejoins on its own,
            # and an unjoined stream leaves its work outside the graph's order.
            self._connector.transport.rejoin_streams()
            if uses_backend:
                _finish_backend_warmup(self._connector)
            return outputs

        self._connector.graph_strategy.execute_step(
            stage_args=tuple(() for _ in stages),
            stage_rows=tuple(stage.rows for stage in stages),
            forward_batches=tuple(None for _ in stages),
            compute=compute,
            metadata_guards=tuple(None for _ in stages),
        )

    def _validate_descriptor(self, *, descriptor: AFDStepDescriptor) -> None:
        descriptor.validate_mode()
        if descriptor.uses_direct != select_direct_step(
            self._config,
            descriptor.lane_stage_rows,
            descriptor.is_extend_in_batch,
            backend=self._connector.transport.ffn_backend,
        ):
            raise AFDError("AFD_DIRECT_MODE_MISMATCH")
        if (
            self._connector.transport.ffn_backend is not None
            and not descriptor.uses_direct
            and not descriptor.uses_m2n
            and descriptor.graph_eligible
        ):
            raise AFDError("AFD_DIRECT_FALLBACK_MUST_BE_EAGER")
        if descriptor.uses_m2n != select_m2n_step(
            self._config, descriptor.lane_stage_rows, descriptor.is_extend_in_batch
        ):
            raise AFDError("AFD_M2N_STEP_MODE_MISMATCH")
        if (
            self._config.experimental_m2n
            and not descriptor.uses_m2n
            and descriptor.graph_eligible
        ):
            raise AFDError("AFD_M2N_FALLBACK_GRAPH_UNSUPPORTED")
        expected = (
            descriptor.kind in ("STEP", "CAPTURE")
            and len(descriptor.lane_stage_rows) == self._config.attention_lane_count
            and all(
                len(vector) == self._config.stages
                for vector in descriptor.lane_stage_rows
            )
            and descriptor.num_layers == self._adapter.num_layers
            and descriptor.hidden_size == self._adapter.hidden_size
            and descriptor.dtype in ("bfloat16", "float16")
        )
        if not expected:
            raise AFDError(
                "AFD_FFN_STEP_DESCRIPTOR_MISMATCH",
                f"descriptor={descriptor!r}",
            )

    def _make_stages(
        self,
        *,
        descriptor: AFDStepDescriptor,
        buffers: tuple[Any, ...],
        lane: int,
        shape: Any,
    ) -> tuple[list[Any], list[Any]]:
        return self._adapter.make_ffn_stages(
            descriptor=descriptor,
            buffers=buffers,
            lane=lane,
            shape=shape,
        )

    def close(self) -> dict[str, Any]:
        return self._connector.close()
