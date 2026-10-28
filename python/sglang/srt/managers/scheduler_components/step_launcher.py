"""Launches a scheduled step: runs its forward and starts its result's copy to
host."""

from __future__ import annotations

import logging
import time
from contextlib import nullcontext
from functools import partial
from typing import TYPE_CHECKING, Optional, Union

from sglang.srt.managers.io_struct import ActiveRanksOutput
from sglang.srt.managers.overlap_utils import resolve_forward_inputs
from sglang.srt.managers.utils import EmbeddingBatchResult, GenerationBatchResult
from sglang.srt.observability.forward_pass_metrics import wrap_forward_with_fpm
from sglang.srt.observability.scheduler_stage_metrics import (
    SCHEDULER_STAGE_RUN_BATCH,
    scheduler_stage_method,
)
from sglang.srt.runtime_context import (
    get_exec,
    get_memory,
    get_parallel,
    get_schedule,
    get_spec,
)
from sglang.srt.speculative.forward_batch_view import worker_forward_batch
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.utils import is_hip

if TYPE_CHECKING:
    from sglang.srt.beam_search.coordinator import BeamCoordinator
    from sglang.srt.managers.overlap_utils import FutureMap
    from sglang.srt.managers.request_lists import CoreLists
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.managers.scheduler_components.bundles import (
        KvResources,
        SchedulerServices,
        Streams,
        Workers,
    )
    from sglang.srt.managers.scheduler_components.engine_state import (
        EngineFlags,
        LaunchCounters,
    )
    from sglang.srt.managers.scheduler_components.ipc_channels import (
        SchedulerIpcChannels,
    )
    from sglang.srt.managers.scheduler_components.mode_stages import ModeStages
    from sglang.srt.managers.scheduler_components.pending_steps import PendingSteps
    from sglang.srt.managers.scheduler_components.pp_spec_forward import (
        PPSpecForward,
    )
    from sglang.srt.managers.scheduler_components.profiler_manager import (
        SchedulerProfilerManager,
    )
    from sglang.srt.model_executor.forward_batch_info import PPProxyTensors
    from sglang.test.scripted_runtime.scheduler_hook import ScriptedSchedulerHook
_is_hip = is_hip()
logger = logging.getLogger(__name__)


class StepLauncher:
    """The launch of a step. It holds the mode's stages, so a PD role switch
    rebuilds it."""

    def __init__(
        self,
        *,
        kv: KvResources,
        streams: Streams,
        future_map: FutureMap,
        counters: LaunchCounters,
        flags: EngineFlags,
        pending: PendingSteps,
        core_lists: CoreLists,
        ipc_channels: SchedulerIpcChannels,
        services: SchedulerServices,
        profiler_manager: SchedulerProfilerManager,
        beam_coordinator: BeamCoordinator,
        pp_spec_forward: Optional[PPSpecForward],
        stages: ModeStages,
        scripted_scheduler_hook: Optional[ScriptedSchedulerHook],
        workers: Workers,
        is_generation: bool,
    ):
        spec_algorithm = SpeculativeAlgorithm.from_string(
            get_spec().speculative_algorithm
        )
        self.token_to_kv_pool_allocator = kv.token_to_kv_pool_allocator
        self.tree_cache = kv.tree_cache
        device_module = streams.device_module
        self.schedule_stream = streams.schedule_stream
        self.forward_stream = streams.forward_stream
        self.copy_stream = streams.copy_stream
        self.forward_stream_ctx = device_module.stream(self.forward_stream)
        self.copy_stream_ctx = device_module.stream(self.copy_stream)
        self.async_d2h_worker = streams.async_d2h_worker
        self.future_map = future_map
        self.counters = counters
        self.flags = flags
        self.pending = pending
        self.lists = core_lists
        self.ipc_channels = ipc_channels
        self.metrics_sink = services.metrics_sink
        self.scheduler_stage_metrics = services.scheduler_stage_metrics
        self.profiler_manager = profiler_manager
        self.beam_coordinator = beam_coordinator
        self.pp_spec_forward = pp_spec_forward
        self.stages = stages
        self.scripted_scheduler_hook = scripted_scheduler_hook
        self.tp_worker = workers.tp_worker
        self.model_worker = workers.model_worker
        self.spec_algorithm = spec_algorithm
        self.is_generation = is_generation
        self.device_module = device_module

        # Pick how a step launches, once: what runs its forward and how its
        # result starts toward the host differ by mode, not by step.
        self.launch_stream_ctx = (
            self.forward_stream_ctx
            if (not get_schedule().disable_overlap_schedule)
            else nullcontext()
        )
        # The beam coordinator relays tokens of non-speculative steps only.
        self.relays_beam_tokens = self.spec_algorithm.is_none()
        if not self.is_generation:
            self.forward_step = self._forward_embedding
            self.finish_launch = self._finish_embedding_launch
        elif not get_schedule().disable_overlap_schedule:
            # FIXME: pp is not compatible with overlap
            self.forward_step = self.model_worker.forward_batch_generation
            self.finish_launch = self._finish_overlap_launch
        elif get_parallel().pp_size > 1 and not self.spec_algorithm.is_none():
            self.forward_step = self.pp_spec_forward.spec_forward
            # The ring brings the result back from the last stage.
            self.finish_launch = lambda step, result: None
        elif self.spec_algorithm.is_none():
            self.forward_step = self.model_worker.forward_batch_generation
            self.finish_launch = self._finish_sync_launch
        else:
            # Non-overlap: drive the V2 worker synchronously; it keys its next
            # draft input into FutureMap.
            self.forward_step = self.model_worker.forward_batch_generation
            self.finish_launch = self._finish_sync_spec_launch
        if self.metrics_sink.enable_fpm:
            # Each result carries its forward's GPU span for the forward pass
            # metrics; the result processing publishes it once ready.
            self.forward_step = wrap_forward_with_fpm(
                self.forward_step, self.metrics_sink.forward_pass_device_timer
            )

    @scheduler_stage_method(SCHEDULER_STAGE_RUN_BATCH)
    def run_batch(
        self,
        batch: ScheduleBatch,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> Union[GenerationBatchResult, EmbeddingBatchResult]:
        """Run a batch."""
        self.metrics_sink.record_scheduler_active(time.monotonic_ns())
        self.counters.forward_ct += 1

        # Accumulate the prefill-token counter used by the HRRN scheduling policy. Decode / prebuilt batches contribute 0.
        if batch.extend_num_tokens:
            self.counters.processed_tokens_counter += batch.extend_num_tokens

        if self.scripted_scheduler_hook is not None:
            self.scripted_scheduler_hook.on_run_batch(batch)

        # Whether to run the profiler
        self.profiler_manager._profile_batch_predicate(batch)
        if self.flags.forward_sleep_time is not None:
            logger.info(f"Scheduler.run_batch sleep {self.flags.forward_sleep_time}s")
            time.sleep(self.flags.forward_sleep_time)

        if (before_forward := self.stages.before_forward) is not None:
            before_forward(batch)

        with self.launch_stream_ctx:
            if not get_schedule().disable_overlap_schedule:
                self.forward_stream.wait_stream(self.schedule_stream)
            # The worker resolves the step's inputs on a view of it and writes
            # its device commit; the launch then starts the result's copy to
            # host.
            ret = self.forward_step(
                batch, pp_proxy_tensors=pp_proxy_tensors, pending_grammar=self.pending
            )
            self.finish_launch(batch, ret)

        if get_memory().enable_unified_cache_external_linker:
            failed = self.tree_cache.finish_external_linker_loads(batch.reqs)
            if batch.chunked_req in failed:
                # Result processing finishes the others; a chunked one is
                # released like a client abort so its next chunk never runs.
                self.lists._pending_chunked_abort_req = batch.chunked_req

        self._maybe_report_active_ranks()

        return ret

    def _forward_embedding(
        self,
        step: ScheduleBatch,
        *,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
        pending_grammar: Optional[PendingSteps] = None,
    ) -> EmbeddingBatchResult:
        """An embedding or reward step: no relay, no grammar."""
        # The forward fills a view of the step; the step stays as built.
        forward_batch = worker_forward_batch(step)
        resolve_forward_inputs(forward_batch, self.future_map)
        pooler_output, can_run_cuda_graph = self.tp_worker.forward_batch_embedding(
            forward_batch
        )
        return EmbeddingBatchResult(
            embeddings=pooler_output.embeddings,
            pooled_hidden_states=pooler_output.pooled_hidden_states,
            can_run_cuda_graph=can_run_cuda_graph,
        )

    def _finish_embedding_launch(
        self, batch: ScheduleBatch, result: EmbeddingBatchResult
    ) -> None:
        if not get_schedule().disable_overlap_schedule:
            result.copy_to_cpu()

    def _finish_overlap_launch(
        self, batch: ScheduleBatch, batch_result: GenerationBatchResult
    ) -> None:
        """Start the result's copy to host on the copy stream; a delayed sample
        starts it once it runs (launch_batch_sample_if_needed)."""
        if get_memory().enable_unified_memory:
            # Record a `forward_done` event after the forward (before
            # copy_to_cpu); lazy-compaction `_flush` gates src reuse on
            # it. Only the unified pool's allocator exposes these hooks.
            allocator = self.token_to_kv_pool_allocator
            forward_done = self.device_module.Event()
            forward_done.record(stream=self.forward_stream)
            allocator.set_latest_forward_done_event(forward_done)
            # Write-set classification: hand the allocator this
            # forward's virtual out_cache_loc as a tensor ref (no GPU work).
            allocator.set_inflight_forward(
                forward_done,
                batch.out_cache_loc,
            )
        # FIXME(lsyin): maybe move this to forward_batch_generation
        batch_result.copy_done = self.device_module.Event()
        if batch_result.delay_sample_func is None:
            if self.relays_beam_tokens:
                self._relay_beam_tokens(batch, batch_result)
            if _is_hip:
                # Cross-stream sync costs more than the tiny D2H it
                # overlaps.
                batch_result.copy_to_cpu(
                    return_logprob=batch.return_logprob,
                    return_hidden_states=batch.return_hidden_states,
                )
            else:
                self._copy_result_to_cpu(batch_result, batch)

    def _finish_sync_launch(
        self, batch: ScheduleBatch, batch_result: GenerationBatchResult
    ) -> None:
        self._relay_beam_tokens(batch, batch_result)
        self.update_cache_from_scheduler(batch, batch_result)
        self._copy_auxiliary_output_to_cpu(batch, batch_result)

    def _finish_sync_spec_launch(
        self, batch: ScheduleBatch, batch_result: GenerationBatchResult
    ) -> None:
        self.update_cache_from_scheduler(batch, batch_result)
        batch_result.copy_done = self.device_module.Event()
        if batch_result.has_sampled_token_ids:
            batch_result.copy_to_cpu(
                return_logprob=batch.return_logprob,
                return_hidden_states=batch.return_hidden_states,
            )

    def _maybe_report_active_ranks(self) -> None:
        if not (
            get_parallel().attn_dp_enabled
            and get_exec().moe.elastic_ep_backend is not None
        ):
            return
        from sglang.srt.elastic_ep.elastic_ep import ElasticEPStateManager
        from sglang.srt.elastic_ep.topology import (
            attn_replica_size,
            collapse_physical_rank_status,
        )

        inst = ElasticEPStateManager.instance()
        if inst is not None and inst.active_ranks_cpu is not None:
            dp_status = collapse_physical_rank_status(
                inst.active_ranks_cpu.tolist(),
                attn_replica_size(),
            )
            self.ipc_channels.send_to_tokenizer.send_output(
                ActiveRanksOutput(status=dp_status)
            )
        else:
            logger.debug("[Elastic EP] active rank state is unavailable")
            return

        model_runner = self.tp_worker.model_runner
        pending = model_runner._pending_elastic_scale_update
        if pending is not None:
            self.ipc_channels.send_to_tokenizer.send_output(pending)
            model_runner._pending_elastic_scale_update = None

    def _relay_beam_tokens(
        self, batch: ScheduleBatch, batch_result: GenerationBatchResult
    ) -> None:
        """Overwrite the beam rows' relayed tokens with the joint-selected
        ones, after the worker stashed the step's sampled tokens."""
        if not batch_result.has_sampled_token_ids:
            return
        self.beam_coordinator.maybe_select_and_relay(
            batch, batch_result, chunked_req=batch.chunked_req
        )

    def _copy_auxiliary_output_to_cpu(
        self,
        batch: ScheduleBatch,
        result: GenerationBatchResult,
    ) -> None:
        logits_output = result.logits_output
        if (
            logits_output is None
            or (
                logits_output.auxiliary_device_output is None
                and result.forward_auxiliary_output is None
            )
            or result.auxiliary_host_output is not None
        ):
            return
        # PP transports the device output to the first rank before copying it.
        if get_parallel().pp_size > 1:
            return
        if result.copy_done is not None:
            raise RuntimeError(
                "generation result has an uncopied auxiliary output after its "
                "device-to-host copy was scheduled"
            )
        result.copy_done = self.device_module.Event()
        result.copy_to_cpu(
            return_logprob=batch.return_logprob,
            return_hidden_states=batch.return_hidden_states,
        )

    def _copy_result_to_cpu(
        self, batch_result: GenerationBatchResult, batch: ScheduleBatch
    ) -> None:
        """Issue the result readback on copy_stream so it overlaps the next forward.

        copy_stream is a leaf gated by copy_done that forward_stream never waits
        for. Under Confidential Computing the copy would block this thread until
        the forward is done, so the worker issues it from its own thread instead.
        """
        copy_fn = partial(
            batch_result.copy_to_cpu,
            return_logprob=batch.return_logprob,
            return_hidden_states=batch.return_hidden_states,
        )
        self.copy_stream.wait_stream(self.forward_stream)
        with self.copy_stream_ctx:
            if self.async_d2h_worker is None:
                copy_fn()
            elif batch.return_hidden_states or batch.spec_algorithm.is_dspark():
                # Hidden states and the DSpark accept lengths are views of CUDA
                # graph buffers that the next replay rewrites; only a copy queued
                # right behind this forward reads them in time.
                copy_fn()
            else:
                batch_result.copy_done = self.async_d2h_worker.submit(copy_fn)

    def launch_batch_sample_if_needed(
        self, batch_result: GenerationBatchResult, cur_batch: ScheduleBatch
    ) -> Union[GenerationBatchResult]:
        # TODO(lsyin): make the delayed sample a default behavior after
        # unifying the forward_batch_generation interface (related to spec V2).
        if batch_result is None or batch_result.delay_sample_func is None:
            return

        with self.forward_stream_ctx:
            self.forward_stream.wait_stream(self.schedule_stream)
            # Delay-sample is non-spec only; the worker's sample stashes the
            # sampled bonus tokens.
            _batch_result = batch_result.delay_sample_func()
            assert _batch_result is batch_result
            self._relay_beam_tokens(cur_batch, batch_result)

        self._copy_result_to_cpu(batch_result, cur_batch)

        # Release the closure and large GPU tensors that are no longer needed.
        # The delay_sample_func closure captures forward_batch (which holds
        # sampling_info with vocab_mask) and logits_output (which holds
        # next_token_logits). Without clearing these, they stay alive via
        # the pending result until its commit in the next iteration,
        # causing a steady VRAM leak with structured output.
        batch_result.delay_sample_func = None
        if batch_result.logits_output is not None:
            batch_result.logits_output.next_token_logits = None

    # placeholder for override
    def update_cache_from_scheduler(
        self, schedule_batch: ScheduleBatch, batch_result: GenerationBatchResult
    ):
        pass
