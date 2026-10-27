# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""A scheduler that manages a tensor parallel GPU worker."""

import faulthandler
import logging
import os
import signal
import time
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)

from sglang.srt.runtime_context import (
    SpawnRanks,
    get_context,
    get_disagg,
    get_memory,
    get_mm,
    get_observability,
    get_parallel,
    get_schedule,
    get_serving,
    get_spec,
    hicache_storage_enabled,
    publish,
    spawn_world_rank,
)

from sglang.srt.utils.common import suppress_noisy_warnings  # isort: skip

suppress_noisy_warnings()

import psutil  # isort: skip
import setproctitle
import torch
import torch.distributed
from torch.distributed import barrier

from sglang.srt.disaggregation import role_switch
from sglang.srt.disaggregation.decode import DisaggDecode
from sglang.srt.disaggregation.prefill import (
    PREFILL_INPUT_POLL_INTERVAL_S,
    DisaggPrefill,
)
from sglang.srt.disaggregation.utils import (
    DisaggregationMode,
    prepare_abort,
)
from sglang.srt.distributed.parallel_state import (
    abort_distributed_environment,
)
from sglang.srt.dllm.manager import DllmManager
from sglang.srt.environ import envs
from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
from sglang.srt.managers.hisparse_coordinator import HiSparseCoordinator
from sglang.srt.managers.io_struct import (
    AbortReq,
    AddExternalCorpusReqInput,
    AddExternalCorpusReqOutput,
    AttachHiCacheStorageReqInput,
    AttachHiCacheStorageReqOutput,
    BatchTokenizedEmbeddingReqInput,
    BatchTokenizedGenerateReqInput,
    BeginWeightUpdateReqInput,
    CheckWeightsReqInput,
    ClearHiCacheReqInput,
    ClearHiCacheReqOutput,
    CloseSessionReqInput,
    ConfigureLoggingReq,
    ContinueGenerationReqInput,
    DestroyWeightsUpdateGroupReqInput,
    DetachHiCacheStorageReqInput,
    DetachHiCacheStorageReqOutput,
    DumperControlReqInput,
    DumperControlReqOutput,
    EndWeightUpdateReqInput,
    ExpertDistributionReq,
    ExpertDistributionReqOutput,
    ExpertDistributionReqType,
    FlushCacheReqInput,
    FreezeGCReq,
    GetInternalStateReq,
    GetWeightsByNameReqInput,
    InitWeightsSendGroupForRemoteInstanceReqInput,
    InitWeightsSendGroupForRemoteInstanceReqOutput,
    InitWeightsUpdateGroupReqInput,
    ListExternalCorporaReqInput,
    ListExternalCorporaReqOutput,
    LoadLoRAAdapterFromTensorsReqInput,
    LoadLoRAAdapterFromTensorsReqOutput,
    LoadLoRAAdapterReqInput,
    LoadLoRAAdapterReqOutput,
    OpenSessionReqInput,
    PauseGenerationReqInput,
    PdRoleSwitchReqInput,
    ProfileReq,
    ReleaseMemoryOccupationReqInput,
    RemoveExternalCorpusReqInput,
    RemoveExternalCorpusReqOutput,
    ResumeMemoryOccupationReqInput,
    RpcReqInput,
    RpcReqOutput,
    ScaleElasticEPReqInput,
    ScaleElasticEPReqOutput,
    SendWeightsToRemoteInstanceReqInput,
    SendWeightsToRemoteInstanceReqOutput,
    SetInternalStateReq,
    ShutdownReq,
    SlowDownReqInput,
    SlowDownReqOutput,
    TokenizedEmbeddingReqInput,
    TokenizedGenerateReqInput,
    UnloadLoRAAdapterReqInput,
    UnloadLoRAAdapterReqOutput,
    UpdateWeightFromDiskReqInput,
    UpdateWeightsFromDistributedReqInput,
    UpdateWeightsFromIPCReqInput,
    UpdateWeightsFromTensorReqInput,
    UpdateWeightVersionReqInput,
    UpdateWeightVersionReqOutput,
    sock_send,
)
from sglang.srt.managers.schedule_batch import (
    FINISH_ABORT,
    Req,
    ScheduleBatch,
    commit_step_counters,
)
from sglang.srt.managers.scheduler_assembly import (
    assemble_scheduler,
    build_pp_ring,
    default_component_classes,
)
from sglang.srt.managers.scheduler_components.bundles import ComponentClasses
from sglang.srt.managers.scheduler_components.invariant_checker import (
    CHECK_REQUEST_LISTS,
)
from sglang.srt.managers.scheduler_components.mode_stages import (
    DriverOps,
    LoopSchedules,
    ScheduleSlot,
)
from sglang.srt.managers.scheduler_components.pending_steps import (
    PendingStep,
)
from sglang.srt.managers.scheduler_components.request_retirer import (
    make_abort_req,
)
from sglang.srt.managers.utils import (
    EmbeddingBatchResult,
    GenerationBatchResult,
    is_health_check_generate_req,
)
from sglang.srt.mem_cache.common import checkpoint_kv_cache
from sglang.srt.model_executor.forward_batch_info import PPProxyTensors
from sglang.srt.observability.req_time_stats import (
    flush_trace_batch,
    set_schedule_time_batch,
)
from sglang.srt.observability.scheduler_stage_metrics import (
    SCHEDULER_STAGE_GET_NEXT_BATCH,
    SCHEDULER_STAGE_IDLE,
    SCHEDULER_STAGE_PROCESS_BATCH_RESULT,
    SCHEDULER_STAGE_PROCESS_REQUESTS,
    SCHEDULER_STAGE_SANITY_CHECK_CACHE,
    scheduler_stage_method,
)
from sglang.srt.observability.trace import process_tracing_init, trace_set_thread_info
from sglang.srt.plugins import load_plugins
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.state_capturer.indexer_topk import destroy_global_indexer_capturer
from sglang.srt.state_capturer.routed_experts import destroy_global_experts_capturer
from sglang.srt.utils import (
    DynamicGradMode,
    configure_logger,
    freeze_gc,
    is_cuda,
    kill_itself_when_parent_died,
    rank_consensus_checker,
    set_gpu_proc_affinity,
    suppress_other_loggers,
    triton_load_watch,
)
from sglang.srt.utils.numa_utils import get_numa_node_if_available, numa_bind_to_node
from sglang.utils import TypeBasedDispatcher, get_exception_traceback

logger = logging.getLogger(__name__)


# Min wall-clock between load publishes on the stalled no-batch path, which
# spins on_idle without sleeping. Bounds the O(queue) get_loads for both the
# DP-balancing writer and the router-facing socket.
LOAD_STALL_REFRESH_S = 0.05


class Scheduler:
    """A scheduler that manages a tensor parallel GPU worker: the event loop
    and the control handlers, over the parts `assemble_scheduler` builds."""

    def __init__(
        self,
        server_args: ServerArgs,
        port_args: PortArgs,
    ):
        self.parts, self.mode, self.hub = assemble_scheduler(
            server_args, port_args, self.loop_schedules(), self.component_classes()
        )
        self.init_loop_state()
        self.init_request_dispatcher()
        if (hook := self.parts.scripted_scheduler_hook) is not None:
            hook.attach(self)

    def loop_schedules(self) -> LoopSchedules:
        """The loop's own schedule stages, which each mode's stages take."""
        return LoopSchedules(
            admit_prefill=self._admit_prefill,
            admit_prefill_after_interval=self._admit_prefill_after_interval,
            admit_dllm=self._admit_dllm_step,
            select_decode=self._select_decode,
            ring_intake_disagg_prefill=self.process_bootstrapped_queue,
            ring_release_disagg_prefill=self.process_prefill_transfer_queue,
        )

    def component_classes(self) -> ComponentClasses:
        """The component classes the assembly builds; a subclass swaps one
        with `dataclasses.replace(super().component_classes(), ...)`."""
        return default_component_classes()

    def init_loop_state(self) -> None:
        # The stage recorder the scheduler_stage_method decorator reads.
        self.scheduler_stage_metrics = self.parts.metrics_sink.scheduler_stage_metrics
        # The flat driver's schedule slot: every decoding request (the PP ring
        # keeps one per microbatch slot).
        self.flat_slot = ScheduleSlot()
        self._deferred_input_requests: List = []
        self._sched_idled = False
        self._last_stall_publish_ts = float("-inf")
        # Set by a runtime PD role switch to break out of the current event loop.
        self._event_loop_should_restart = False
        # Guards against concurrent/re-entrant PD role switches.
        self._pd_role_switch_in_progress = False
        # Set if a role switch tore down the old role but failed to rebuild
        # either the new or the old role; the instance can no longer serve.
        self._pd_role_switch_unhealthy = False

    def publish_load_snapshot(self, force: bool = False):
        """Returns the LoadSnapshot it published, or None when disabled,
        throttled, or failed — so co-located sinks (the router-facing load
        publisher) can reuse it instead of walking the queues again."""
        writer = self.parts.load_snapshot_writer
        if writer is None:
            return None
        if not force:
            writer.publish_counter += 1
            if writer.publish_counter < writer.publish_interval:
                return None
        writer.publish_counter = 0
        try:
            load = self.hub.load_inquirer.get_loads()
            writer.write(load)
            return load
        except Exception as e:
            logger.warning("load snapshot publish failed: %s", e)
            return None

    # The mode owners, read through the holder a PD role switch updates.
    @property
    def disagg_prefill(self) -> Optional[DisaggPrefill]:
        return self.parts.mode_owners.disagg_prefill

    @property
    def disagg_decode(self) -> Optional[DisaggDecode]:
        return self.parts.mode_owners.disagg_decode

    @property
    def dllm(self) -> Optional[DllmManager]:
        return self.parts.mode_owners.dllm

    @property
    def hisparse_coordinator(self) -> Optional[HiSparseCoordinator]:
        return self.parts.mode_owners.hisparse_coordinator

    def init_request_dispatcher(self):
        self._request_dispatcher = TypeBasedDispatcher(
            [
                (TokenizedGenerateReqInput, self.handle_generate_request),
                (TokenizedEmbeddingReqInput, self.handle_embedding_request),
                (BatchTokenizedGenerateReqInput, self.handle_batch_generate_request),
                (BatchTokenizedEmbeddingReqInput, self.handle_batch_embedding_request),
                (FlushCacheReqInput, self.hub.flush_wrapper.handle),
                (ClearHiCacheReqInput, self.clear_hicache_storage_wrapped),
                (AttachHiCacheStorageReqInput, self.attach_hicache_storage_wrapped),
                (DetachHiCacheStorageReqInput, self.detach_hicache_storage_wrapped),
                (AbortReq, self.abort_request),
                (OpenSessionReqInput, self.open_session),
                (CloseSessionReqInput, self.close_session),
                (
                    UpdateWeightFromDiskReqInput,
                    self.hub.weight_updater.update_weights_from_disk,
                ),
                (
                    InitWeightsUpdateGroupReqInput,
                    self.hub.weight_updater.init_weights_update_group,
                ),
                (
                    DestroyWeightsUpdateGroupReqInput,
                    self.hub.weight_updater.destroy_weights_update_group,
                ),
                (
                    InitWeightsSendGroupForRemoteInstanceReqInput,
                    self.init_weights_send_group_for_remote_instance,
                ),
                (
                    SendWeightsToRemoteInstanceReqInput,
                    self.send_weights_to_remote_instance,
                ),
                (
                    BeginWeightUpdateReqInput,
                    self.hub.weight_updater.begin_weight_update,
                ),
                (
                    EndWeightUpdateReqInput,
                    self.hub.weight_updater.end_weight_update,
                ),
                (
                    UpdateWeightsFromDistributedReqInput,
                    self.hub.weight_updater.update_weights_from_distributed,
                ),
                (
                    UpdateWeightsFromTensorReqInput,
                    self.hub.weight_updater.update_weights_from_tensor,
                ),
                (
                    UpdateWeightsFromIPCReqInput,
                    self.hub.weight_updater.update_weights_from_ipc,
                ),
                (
                    UpdateWeightVersionReqInput,
                    self.handle_update_weight_version,
                ),
                (
                    GetWeightsByNameReqInput,
                    self.hub.weight_updater.get_weights_by_name,
                ),
                (
                    ReleaseMemoryOccupationReqInput,
                    self.hub.weight_updater.release_memory_occupation,
                ),
                (
                    ResumeMemoryOccupationReqInput,
                    self.hub.weight_updater.resume_memory_occupation,
                ),
                (
                    CheckWeightsReqInput,
                    self.hub.weight_updater.check_weights,
                ),
                (SlowDownReqInput, self.slow_down),
                (PdRoleSwitchReqInput, self.handle_pd_role_switch),
                (
                    ProfileReq,
                    lambda req: self.parts.profiler_manager._profile(req),
                ),
                (FreezeGCReq, self.handle_freeze_gc),
                (ShutdownReq, self.handle_shutdown),
                (GetInternalStateReq, self.get_internal_state),
                (SetInternalStateReq, self.set_internal_state),
                (RpcReqInput, self.handle_rpc_request),
                (ExpertDistributionReq, self.expert_distribution_handle),
                (LoadLoRAAdapterReqInput, self.load_lora_adapter),
                (
                    LoadLoRAAdapterFromTensorsReqInput,
                    self.load_lora_adapter_from_tensors,
                ),
                (UnloadLoRAAdapterReqInput, self.unload_lora_adapter),
                (PauseGenerationReqInput, self.pause_generation),
                (ContinueGenerationReqInput, self.continue_generation),
                (ConfigureLoggingReq, self.configure_logging),
                (ScaleElasticEPReqInput, self.handle_scale_elastic_ep),
                (DumperControlReqInput, self.handle_dumper_control),
                (AddExternalCorpusReqInput, self.add_external_corpus),
                (
                    RemoveExternalCorpusReqInput,
                    self.remove_external_corpus,
                ),
                (
                    ListExternalCorporaReqInput,
                    self.list_external_corpora,
                ),
            ]
        )

    def get_init_info(self) -> Dict[str, Any]:
        """Return scheduler initialization info for handshake.

        This method provides the initialization info needed by the tokenizer manager
        and other components to verify the scheduler is ready.
        """
        result_dict = {
            "status": "ready",
            "max_total_num_tokens": self.parts.capacity.max_total_num_tokens,
            "max_req_input_len": self.parts.limits.max_req_input_len,
            "startup_time": self.hub.internal_state.startup_time,
        }

        if get_serving().grpc_port is not None and not (
            get_serving().smg_grpc_mode or get_serving().grpc_mode
        ):
            result_dict["kv_event_sources"] = (
                self.parts.kv_events_publisher.local_kv_event_sources(
                    get_schedule().page_size * get_parallel().dcp_size
                )
            )
        return result_dict

    def release_host_resources(self) -> None:
        # Release pinned host buffers in userspace on graceful shutdown; see
        # HostKVCache.destroy. Called from run_scheduler_process's finally.
        if self.hisparse_coordinator is not None:
            self.hisparse_coordinator.destroy()
        self.parts.kv.tree_cache.release_host_resources()
        if self.parts.decode_offload_manager is not None:
            self.parts.decode_offload_manager.release_host_resources()
        destroy_global_experts_capturer()
        destroy_global_indexer_capturer()

        rank_consensus_checker.shutdown()

    def run_event_loop(self) -> None:
        """Run the scheduler's event loop.

        The event loop blocks until shutdown.
        """
        # Engine init (graph capture, warmups) is done; from here on any
        # Triton kernel device-load is a lazy first-use at serving time.
        triton_load_watch.install()
        triton_load_watch.mark_serving_started()

        # The global WAR barrier fences the scheduler's next shared-buffer write
        # on the previous forward's read of the unified memory pool.
        self._war_barrier_enabled = is_cuda() or envs.SGLANG_ENABLE_WAR_BARRIER.get()
        with self.parts.streams.device_module.StreamContext(
            self.parts.streams.schedule_stream
        ):
            self.parts.metrics_sink.start_scheduler_time_accounting()
            dispatch_event_loop(self)

    def _apply_war_barrier(self):
        # WAR: keep later schedule_stream writes behind this forward's shared reads.
        # Clearing matters: a phase that skips the publish then falls back to coarse.
        if not self._war_barrier_enabled:
            return
        runner = self.parts.workers.model_worker.last_shared_read_runner
        ev = runner.shared_read_done_event
        runner.shared_read_done_event = None
        if ev is not None and not envs.SGLANG_FORCE_COARSE_WAR_BARRIER.get():
            self.parts.streams.schedule_stream.wait_event(ev)
        else:
            self.parts.streams.schedule_stream.wait_stream(
                self.parts.streams.forward_stream
            )

    @DynamicGradMode()
    def event_loop_normal(self):
        """A normal scheduler loop: a step commits in the iteration launching it."""
        self.run_flat_loop(overlap=False)

    @DynamicGradMode()
    def event_loop_overlap(self):
        """A scheduler loop that overlaps the CPU processing and GPU computation:
        a step commits in the next iteration, after that iteration's launch."""
        self.parts.flags.enable_skip_finishing_decode = True
        self.run_flat_loop(overlap=True)

    @torch.no_grad()
    def event_loop_normal_disagg_prefill(self) -> None:
        """A normal scheduler loop for prefill worker in disaggregation mode."""
        self.run_flat_loop(overlap=False)

    def _yield_gil_if_needed(self) -> None:
        if get_parallel().tp_size == 1:
            # No collective parks this thread at TP=1. A bounded sleep lets
            # bootstrap, transfer and cache workers reacquire the GIL.
            time.sleep(PREFILL_INPUT_POLL_INTERVAL_S)

    @torch.no_grad()
    def event_loop_overlap_disagg_prefill(self) -> None:
        """An overlap scheduler loop for prefill worker in disaggregation mode;
        with continuous input polling, intake runs while a result is pending."""
        self.run_flat_loop(overlap=True)

    @torch.no_grad()
    def event_loop_normal_disagg_decode(self):
        """A normal scheduler loop for decode worker in disaggregation mode."""
        self.run_flat_loop(overlap=False)

    @torch.no_grad()
    def event_loop_overlap_disagg_decode(self):
        """An overlap scheduler loop for decode worker in disaggregation mode."""
        self.run_flat_loop(overlap=True)

    @DynamicGradMode()
    def event_loop_pp(self):
        """A scheduler loop for pipeline parallelism."""
        self.init_pp_ring()
        self.pp_ring.run_ring_loop()

    def init_pp_ring(self):
        """Build the ring driver over the loop's core stages, once per ring loop."""
        self.pp_ring = build_pp_ring(
            self.parts,
            self.mode.stages,
            DriverOps(
                ingest_requests=self.ingest_requests,
                launch_step=self.launch_step,
                commit_step=self.commit_step,
                run_batch=self.mode.step_launcher.run_batch,
                on_idle=self.on_idle,
                mark_idle=self.mark_sched_idled,
                paused_pass=self.paused_pass,
                schedule_step=self.schedule_step,
            ),
        )

    def mark_sched_idled(self):
        self._sched_idled = True

    def run_flat_loop(self, overlap: bool):
        """The loop of every mode without PP. Each pass takes intake, schedules
        and launches a step, then commits: without overlap every pending step,
        with overlap the oldest step launched before this pass. With a commit
        readiness gate (PD prefill continuous polling), a pass whose oldest
        result is not ready on every rank returns to intake instead."""
        stages = self.mode.stages
        prefetch, intake = stages.prefetch, stages.intake
        after_commit = stages.after_commit
        is_result_ready = stages.is_result_ready if overlap else None
        slot = self.flat_slot
        # Fixed for the loop, so read once: an env read costs more than the
        # check it gates.
        may_sync_overlap = overlap and self.may_disable_overlap()
        busy_self_check = envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY.get()
        # True while passes repeat intake for the oldest pending result.
        waiting_for_result = False
        # A schedule that found nothing is not retried until new input.
        skip_schedule = False
        # The newest step built and the launched one not sampled yet; under
        # the gate they carry across the passes that wait.
        step = launched = None
        # The step the last full pass built, and whether the chunk it
        # launched was processed: once per launched step, not per pass.
        last_step, chunk_processed = None, False
        while True:
            if self.parts.flags.gracefully_exit:
                break

            # Receive requests
            if prefetch is not None and not self.parts.flags._engine_paused:
                # Pending rooms from the prior cycle can overlap request intake
                # and the tail of the in-flight decode graph.
                prefetch()
            received = self.ingest_requests(waiting_for_result)
            if self.paused_pass():
                continue
            joined = intake() if intake is not None else None
            if joined:
                self.parts.lists.waiting_queue.extend(joined)

            # Schedule the step
            scheduled = False
            if is_result_ready is None:
                if not self.parts.flags.gracefully_exit:
                    step = self.schedule_step(slot)
                    scheduled = True
            else:
                if not waiting_for_result or received or joined:
                    skip_schedule = False
                if (
                    len(self.parts.pending) < 2
                    and not self.parts.flags.gracefully_exit
                    and not self._deferred_input_requests
                    and not skip_schedule
                ):
                    step = self.schedule_step(
                        slot,
                        # Storage retry deadlines count schedules, not polls.
                        should_retry_storage_prefetch=not waiting_for_result,
                        process_chunk=last_step is None or not chunk_processed,
                    )
                    chunk_processed = last_step is not None
                    skip_schedule = step is None
                    scheduled = True

            # Launch it; a step that must not overlap the pending ones commits
            # them first.
            record = None
            if scheduled:
                self.parts.counters.cur_batch_for_debug = step
                if may_sync_overlap and self.is_disable_overlap_for_batch(step):
                    self.commit_pending_steps()
                    # Opportunistic flush at the disable_overlap sync boundary:
                    # forward_stream is idle (prev forward drained, next not launched),
                    # so `_flush`'s non-urgent guard compacts freely. Sync-free, best-effort.
                    if get_memory().enable_unified_memory:
                        try:
                            self.parts.kv.token_to_kv_pool_allocator.flush_opportunistic()
                        except Exception:
                            pass
                if step:
                    launched = record = self.launch_step(step, slot, overlap=overlap)

            # Commit: without overlap every pending step; with overlap the
            # oldest launched before this pass, once its result is ready.
            if not overlap:
                num_committed = self.commit_pending_steps()
            elif len(self.parts.pending) > (1 if record is not None else 0):
                if (
                    is_result_ready is not None
                    and not self.parts.flags.gracefully_exit
                    and not is_result_ready(next(iter(self.parts.pending)).result)
                ):
                    waiting_for_result = True
                    self._yield_gil_if_needed()
                    continue
                self.commit_step(self.parts.pending.popleft())
                num_committed = 1
            else:
                num_committed = 0
            if step is None and num_committed == 0:
                # When the server is idle, do self-check and re-init some states.
                self.on_idle()
            if after_commit is not None:
                # Poll the KV transfers of the requests the commits handed over.
                self.parts.lists.waiting_queue = after_commit(
                    self.parts.lists.waiting_queue
                )

            # Run sample of the launched step. It depends on the result of the
            # pending one (e.g., grammar), so it runs after that commit.
            if launched is not None:
                if overlap and self.parts.is_generation:
                    self.mode.step_launcher.launch_batch_sample_if_needed(
                        launched.result, launched.step
                    )
                launched = None

            if step is None:
                # A launch clears it.
                self._sched_idled = True
            if self.parts.recv_skipper is not None:
                self.parts.recv_skipper.on_launch(step)
            if is_result_ready is not None:
                if step is not last_step:
                    last_step, chunk_processed = step, False
                waiting_for_result = False

            if busy_self_check:
                self.hub.invariant_checker.self_check_during_busy()

    def paused_pass(self) -> bool:
        """Whether the engine is paused; a paused pass records the
        scheduler's state and yields the GIL. The flat driver then runs
        nothing after intake; the PP ring launches nothing but keeps its
        exchanges and commits."""
        if not self.parts.flags._engine_paused:
            return False
        self._record_scheduler_state_for_paused_engine()
        self._yield_gil_if_needed()
        return True

    def launch_step(
        self,
        step: ScheduleBatch,
        slot: ScheduleSlot,
        overlap: bool = False,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> PendingStep:
        """Run a step, move the requests it launches, and keep it pending. A
        ring slot's step runs behind the PP fences, until the ring brings its
        outputs back."""
        launch_ts = time.monotonic()
        after_idle_gap = self._sched_idled
        self._sched_idled = False
        if slot.index is not None:
            result = self.pp_ring.run_slot_batch(slot, step, pp_proxy_tensors)
        else:
            result = self.mode.step_launcher.run_batch(step)
            if overlap:
                # Fence result processing behind this forward's shared reads.
                self._apply_war_barrier()
        self.join_launched_step(step, slot)
        record = PendingStep(
            step=step,
            result=result,
            launch_ts=launch_ts,
            after_idle_gap=after_idle_gap,
        )
        self.parts.pending.push(record)
        return record

    def join_launched_step(self, step: ScheduleBatch, slot: ScheduleSlot) -> None:
        """Move the requests a launched step prefills: out of the waiting queue,
        into the chunked holder for a middle chunk, into decoding once their
        last prefill chunk runs (on a PD prefill node, into prefilling until
        the commit hands their KV over; with HiSparse, into staging until their
        KV is copied to host; in dLLM mode, into dLLM staging until the
        commit)."""
        stages = self.mode.stages
        if stages.on_launch is not None:
            stages.on_launch(step)
            return
        prefill_reqs = step.prefill_reqs()
        if not prefill_reqs:
            return
        admitted = set(prefill_reqs)
        self.parts.lists.waiting_queue = [
            r for r in self.parts.lists.waiting_queue if r not in admitted
        ]
        self.parts.lists.chunked_req = step.chunked_req
        launched = [r for r in prefill_reqs if r is not step.chunked_req]
        if stages.on_prefilled is not None:
            stages.on_prefilled(launched)
        else:
            self.parts.lists.join_decoding(launched, slot)

    def commit_pending_steps(self, keep: int = 0) -> int:
        """Commit the oldest pending steps until keep remain; return how many."""
        num_committed = 0
        while len(self.parts.pending) > keep:
            self.commit_step(self.parts.pending.popleft())
            num_committed += 1
        return num_committed

    def commit_step(self, record: PendingStep) -> None:
        """The host commit of one launched step, popped from pending."""
        self.process_batch_result(record.step, record.result)
        # Its forward has finished: what released requests left to it is free.
        self.parts.request_retirer.release_held(record.step)
        self.parts.step_counters.record(record)
        if CHECK_REQUEST_LISTS:
            self.hub.invariant_checker.check_no_finished_reqs(
                self.hub.request_registry.request_lists()
            )

    def leave_finished_reqs(self) -> None:
        """Drop the requests a commit finished from decoding (PD decode holds
        those still releasing KV), and from the mode's own lists (HiSparse
        staging)."""
        if (hold_finished := self.mode.stages.hold_finished) is not None:
            hold_finished([r for r in self.parts.lists.decoding if r.finished()])
        self.parts.lists.decoding = [
            r for r in self.parts.lists.decoding if not r.finished()
        ]
        if (leave_finished := self.mode.stages.leave_finished) is not None:
            leave_finished()

    def may_disable_overlap(self) -> bool:
        """Whether is_disable_overlap_for_batch can hold for any step: only
        with consecutive prefill overlap disabled or a speculative algorithm."""
        return (
            envs.SGLANG_DISABLE_CONSECUTIVE_PREFILL_OVERLAP.get()
            or not self.parts.spec_algorithm.is_none()
        )

    def is_disable_overlap_for_batch(self, batch: Optional[ScheduleBatch]) -> bool:
        # For two consecutive prefill batches, we disable overlap to improve the TTFT of the first batch.
        # This might slightly hurt the throughput, so we use an environment variable to control it.
        # In DP attention mode, use the globally synchronized is_extend_in_batch
        # so all DP ranks make the same overlap decision (avoiding deadlock).
        # In non-DP mode, use the local forward_mode directly.
        if self.parts.require_mlp_sync:
            is_extend = lambda b: b and b.is_extend_in_batch
        else:
            is_extend = lambda b: b and b.forward_mode.is_extend()

        # The step launched last, still pending under overlap.
        last = self.parts.pending.newest()
        batch_is_extend = is_extend(batch)
        last_batch_is_extend = is_extend(last.step if last is not None else None)

        disable_overlap_for_batch = (
            envs.SGLANG_DISABLE_CONSECUTIVE_PREFILL_OVERLAP.get()
            and batch_is_extend
            and last_batch_is_extend
        )

        # Sync so the FSM advance lands before the next batch's bitmask, for
        # algorithms whose verify does not run the pending grammar commit
        # (the forward's pending_grammar). Permanent path for host-draft
        # algorithms.
        need_grammar_sync = (
            batch
            and not batch.spec_algorithm.is_none()
            and batch.grammar_needs_sync()
            and batch.forward_mode.is_decode()
            and len(self.parts.pending) > 0
        )

        return disable_overlap_for_batch or need_grammar_sync

    def ingest_requests(self, stop_at_pause: bool = False) -> List:
        """Receive and broadcast input, then dispatch queued requests in order.

        With stop_at_pause, retain the first pause and all following inputs
        until the caller reaches a scheduling boundary. Receiving and broadcasting
        continue each pass; all ranks retain the same suffix.

        The one place a new per-iteration input source belongs; the return
        value exists for the pipeline stages that relay requests onward.
        """
        local_reqs = []
        # Avoid re-enqueuing timeout aborts while dispatch is stopped at a pause.
        if (
            not self._deferred_input_requests
            and get_parallel().pp_rank == 0
            and get_parallel().attn_tp_rank == 0
            and get_parallel().attn_cp_rank == 0
        ):
            local_reqs = self.hub.request_registry.poll_timeout_aborts()
        recv_start_ns = time.monotonic_ns()
        self._deferred_input_requests.extend(
            self.parts.request_receiver.recv_requests(local_reqs=local_reqs)
        )

        num_requests_to_process = len(self._deferred_input_requests)
        if stop_at_pause:
            for index, req in enumerate(self._deferred_input_requests):
                if isinstance(req, PauseGenerationReqInput):
                    num_requests_to_process = index
                    break
        recv_reqs = self._deferred_input_requests[:num_requests_to_process]
        del self._deferred_input_requests[:num_requests_to_process]

        if recv_reqs:
            # Count successful receive/broadcast time as active; empty polls stay idle.
            self.parts.metrics_sink.record_scheduler_active(recv_start_ns)
        self.process_input_requests(recv_reqs)
        return recv_reqs

    @scheduler_stage_method(SCHEDULER_STAGE_PROCESS_REQUESTS)
    def process_input_requests(self, recv_reqs: List):
        now = time.monotonic()
        self.parts.session_controller.maybe_reap(now)

        for recv_req in recv_reqs:
            # Before any early exit, so every request releases its CUDA IPC pool slot.
            self.hub.request_intake.reconstruct_positional_embed_overrides(recv_req)

            vmm_errors = None
            if get_mm().mm_feature_transport == "cuda_vmm":
                vmm_errors = self.hub.request_intake.materialize_cuda_vmm_inputs(
                    recv_req
                )

            # Skip health check when server is busy — ongoing requests already carry health info.
            if is_health_check_generate_req(
                recv_req
            ) and not self.hub.request_registry.is_fully_idle(for_health_check=True):
                self.parts.output_streamer.defer_health_check(
                    getattr(recv_req, "http_worker_ipc", None)
                )
                continue

            if vmm_errors is not None and any(vmm_errors):
                self.hub.request_intake.dispatch_tokenized_mm_requests(
                    recv_req, vmm_errors
                )
                continue

            output = self._request_dispatcher(recv_req)
            if output is not None:
                if self.parts.rust_server is not None:
                    # Embedded Rust server: every control-request response goes
                    # back through the egress ring (the zmq tokenizer socket is
                    # not consumed); the Rust api_server shapes it per-endpoint.
                    self.parts.rust_server.push_control_output(recv_req, output)
                elif isinstance(output, RpcReqOutput):
                    if self.parts.ipc_channels.recv_from_rpc is not None:
                        sock_send(self.parts.ipc_channels.recv_from_rpc, output)
                else:
                    self.parts.ipc_channels.send_to_tokenizer.send_output(
                        output, recv_req
                    )

        self.hub.flush_wrapper.check_pending()
        if self.parts.external_corpus_manager is not None:
            self.parts.external_corpus_manager.check_pending_load()

        # A runtime PD role switch rebuilt the disaggregation structures for a new
        # role. The response has already been sent above; now break out of the
        # current (old-role) event loop so the supervisor can re-dispatch.
        if get_disagg().enable_pd_role_switch and self._event_loop_should_restart:
            self._event_loop_should_restart = False
            raise role_switch.PdRoleSwitchRestart()

    def handle_generate_request(self, recv_req: TokenizedGenerateReqInput):
        self.hub.request_intake.handle_generate_request(recv_req)

    def handle_batch_generate_request(self, recv_req: BatchTokenizedGenerateReqInput):
        self.hub.request_intake.handle_batch_generate_request(recv_req)

    def handle_embedding_request(self, recv_req: TokenizedEmbeddingReqInput):
        self.hub.request_intake.handle_embedding_request(recv_req)

    def handle_batch_embedding_request(self, recv_req: BatchTokenizedEmbeddingReqInput):
        self.hub.request_intake.handle_batch_embedding_request(recv_req)

    def stash_chunked_request(self) -> None:
        """Publish the KV the chunked request's last launched chunk computed;
        a later schedule before its next chunk finds nothing new."""
        req = self.parts.lists.chunked_req
        # A parked chunk (add_chunked_req hybrid-SWA early-return) leaves
        # extend_end at prefix_len: it computed no new KV, so nothing to stash.
        if req is None or req.extend_end <= req.prefix_len:
            return
        if self.disagg_prefill is not None:
            self.disagg_prefill.checkpoint(req)
        else:
            checkpoint_kv_cache(req, self.parts.kv.tree_cache)

    def process_pending_chunked_abort(self) -> None:
        """Abort an in-flight chunked-prefill request once it is safe to do so.

        ``abort_request`` only records the target in ``_pending_chunked_abort_req``
        (tearing it down mid-iteration is unsafe). Clearing ``chunked_req`` here at
        the top of the scheduling step stops the next chunk from launching; the
        chunk already launched is drained when its result is resolved. Under overlap
        the result lands a step later; that step still names the request as its
        ``chunked_req``, so the batch-result processors skip the aborted chunk:
        ``DisaggPrefill.commit_extend`` via its ``is_aborted`` drop, and
        ``process_batch_result_prefill`` via its chunked branch (the finished req
        is excluded from streaming and its logprob offset is still accounted).
        Mirrors ``handle_bootstrap_failure``.
        """
        req = self.parts.lists._pending_chunked_abort_req
        if req is None:
            return
        if self.parts.lists.chunked_req is not req:
            # Already past chunked prefill; the running-batch abort path handles
            # it. Drop the marker once the request is actually gone.
            if req.finished() or not req.kv.holds_kv:
                self.parts.lists._pending_chunked_abort_req = None
                return
            # The request moved to another scheduler queue after abort_request
            # deferred it, so retry against its current location.
            self.parts.lists._pending_chunked_abort_req = None
            self.hub.request_registry.abort_request(AbortReq(rid=req.rid))
            return

        reason = req.discard_output_reason or FINISH_ABORT("Aborted")
        prepare_abort(req, reason.message, reason.status_code)
        req.time_stats.trace_ctx.abort(abort_info={"reason": reason.message})
        req.to_finish = None
        if self.disagg_prefill is not None:
            self.disagg_prefill.release_aborted_chunked_req(req)
        self.parts.request_retirer.retire_aborted(req)

        self.parts.lists.chunked_req = None
        self.parts.lists._pending_chunked_abort_req = None
        # A client abort leaves the message to the tokenizer.
        finished_reason = req.discard_output_reason and reason.to_json()
        self.parts.ipc_channels.send_to_tokenizer.send_output(
            make_abort_req(req, finished_reason=finished_reason), req
        )
        logger.debug(f"Abort chunked prefill request. {req.rid=}")

    def _process_hicache_events(
        self, should_retry_storage_prefetch: bool = True
    ) -> None:
        # The HiCache drain is TP-wide consensus; run it before rank-local
        # decisions (_should_defer_prefill) or ranks enter different collectives.
        if (
            get_memory().enable_hierarchical_cache
            or get_memory().enable_flexkv
            or get_memory().enable_unified_cache_external_linker
            or get_memory().enable_lmcache
        ):
            self.parts.kv.tree_cache.check_hicache_events()
            if hicache_storage_enabled() and should_retry_storage_prefetch:
                self.hub.request_intake.process_storage_prefetch_retries()

    @scheduler_stage_method(SCHEDULER_STAGE_GET_NEXT_BATCH)
    def schedule_step(
        self,
        slot: ScheduleSlot,
        should_retry_storage_prefetch: bool = True,
        process_chunk: bool = True,
    ) -> Optional[ScheduleBatch]:
        """Schedule this iteration and build its step for slot from the
        request lists: every mode runs these steps in this order, and its
        stages supply only what differs. The flat PD prefill driver repeats a
        schedule while it waits for a result: those repeats retry no storage
        prefetch and do not advance the chunked request again.

        The step is built fresh from the lists and the requests admitted now;
        launching it moves them (see ``join_launched_step``)."""
        stages = self.mode.stages
        lists = self.parts.lists
        self.process_pending_chunked_abort()
        # The HiCache drain is TP-wide consensus; run it before rank-local
        # decisions (should_defer_prefill) or ranks enter different collectives.
        if stages.drain_hicache is not None:
            stages.drain_hicache()
        else:
            self._process_hicache_events(should_retry_storage_prefetch)
        self._start_fpm_clock()
        self.stash_chunked_request()

        if stages.before_admission is not None:
            stages.before_admission()
        if stages.resolve_waiting is not None:
            lists.waiting_queue = stages.resolve_waiting(lists.waiting_queue)
        if stages.advance_chunked is not None and process_chunk:
            lists.chunked_req, lists.waiting_queue = stages.advance_chunked(
                lists.chunked_req, lists.waiting_queue, self.parts.pending
            )
        self.hub.request_intake.requeue_ready_grammars()
        if stages.join_ready is not None:
            if ready := stages.join_ready(slot):
                lists.join_decoding(ready, slot)

        new_batch = stages.admit(slot) if stages.admit is not None else None
        need_mlp_sync = None
        if (
            stages.admit is not None
            and stages.select_decode is not None
            and self.parts.require_mlp_sync
            and not self.parts.spec_algorithm.is_none()
            and not get_spec().speculative_skip_dp_mlp_sync
            and not envs.SGLANG_ENABLE_DP_SPEC_PREFILL_COORDINATION.get()
        ):
            # NOTE: This branch makes sure prefill and decode batches will not be mixed when spec and dp-attn is enabled.
            # Before building the decode step:
            # 1. All new batches are none -> need_mlp_sync remains true (sync is needed for decode batch).
            # 2. All new batches are some (prefill / idle) -> we do not need prepare mlp sync one more time.
            new_batch = self.parts.dp_attn_adapter.maybe_prepare_mlp_sync_batch(
                new_batch
            )
            need_mlp_sync = new_batch is None

        # Run prefill first if possible
        step = new_batch
        if step is None and stages.select_decode is not None:
            step = stages.select_decode(slot)
        return self.finalize_step(step, slot, need_mlp_sync)

    def _admit_prefill(self, slot: ScheduleSlot) -> Optional[ScheduleBatch]:
        return self.hub.step_planner.get_new_batch_prefill(slot)

    def _admit_prefill_after_interval(
        self, slot: ScheduleSlot
    ) -> Optional[ScheduleBatch]:
        """Prefill admission, unless the prefill/decode interval defers it."""
        if self.hub.step_planner.should_defer_prefill():
            return None
        return self.hub.step_planner.get_new_batch_prefill(slot)

    def _admit_dllm_step(self, slot: ScheduleSlot) -> Optional[ScheduleBatch]:
        """The dLLM owner takes the requests it runs; the rest stay waiting."""
        step, self.parts.lists.waiting_queue = self.dllm.schedule(
            self.parts.lists.waiting_queue
        )
        return step

    def _select_decode(self, slot: ScheduleSlot) -> Optional[ScheduleBatch]:
        return self.hub.step_planner.get_decode_step(slot)

    def process_bootstrapped_queue(self):
        self.parts.lists.waiting_queue.extend(self.disagg_prefill.intake())

    def process_prefill_transfer_queue(self, release_rids: Optional[List[str]]) -> None:
        self.parts.lists.waiting_queue = self.disagg_prefill.after_commit(
            self.parts.lists.waiting_queue, release_rids
        )

    def finalize_step(
        self,
        step: Optional[ScheduleBatch],
        slot: ScheduleSlot,
        need_mlp_sync: Optional[bool] = None,
    ) -> Optional[ScheduleBatch]:
        """The tail of every schedule, on the step it built: the DP attention
        sync and the step's forward inputs; nothing writes the step after it
        until its commit. need_mlp_sync overrides whether the sync runs."""
        step = self.parts.dp_attn_adapter.maybe_prepare_mlp_sync_batch(
            step, need_sync=need_mlp_sync
        )
        # The extend view keeps a heterogeneous dp step replayable.
        self.parts.dp_attn_adapter.maybe_set_extend_view(step)
        self.hub.step_planner.arm_prefill_decode_interval(step)

        # Handle ngram embedding
        step = self.parts.ngram_embedding_manager.prepare_for_forward(
            step, chunked_req=step.chunked_req if step is not None else None
        )

        if step:
            set_schedule_time_batch(step)
            if self.parts.metrics_sink.enable_fpm:
                step.fpm_start_time = self._fpm_batch_t0

        if self.parts.scripted_scheduler_hook is not None:
            self.parts.scripted_scheduler_hook.on_schedule(step, slot)
        return step

    def _start_fpm_clock(self) -> None:
        """Start the forward pass metrics clock of the step being scheduled."""
        if self.parts.metrics_sink.enable_fpm:
            self._fpm_batch_t0 = time.monotonic()

    @scheduler_stage_method(SCHEDULER_STAGE_PROCESS_BATCH_RESULT)
    def process_batch_result(
        self,
        batch: ScheduleBatch,
        result: Union[GenerationBatchResult, EmbeddingBatchResult],
    ):
        # Flush async trace ops here: in overlap mode this CPU work runs while
        # the next batch's GPU forward is in flight, giving free overlap.
        flush_trace_batch(batch.reqs)
        snapshot = self.publish_load_snapshot(force=batch.forward_mode.is_extend())
        # Router-facing gauge on the dedicated PUB socket, reusing the
        # snapshot above rather than walking the queues again.
        self.parts.load_publisher.publish_load_stat(
            self.hub.load_inquirer.get_loads,
            force=batch.forward_mode.is_extend(),
            snapshot=snapshot,
        )

        commit_step_counters(batch)
        if batch.forward_mode.is_decode():
            num_generated_tokens = (
                self.parts.batch_result_processor.process_batch_result_decode(
                    batch, result
                )
            )
            self.hub.metrics_reporter.report_decode_stats(
                result.can_run_cuda_graph,
                batch=batch,
                num_generated_tokens=num_generated_tokens,
            )
            if (
                after_decode_commit := self.mode.stages.after_decode_commit
            ) is not None:
                after_decode_commit()
        elif batch.forward_mode.is_extend():
            if requeued := self.mode.stages.commit_extend(batch, result):
                self.parts.lists.waiting_queue = [
                    *requeued,
                    *self.parts.lists.waiting_queue,
                ]
            # None on decode->extend converted batches; they are decode work and
            # have no prefill stats to report.
            if batch.prefill_stats is not None:
                self.hub.metrics_reporter.report_prefill_stats(
                    batch=batch,
                    prefill_stats=batch.prefill_stats,
                    can_run_cuda_graph=result.can_run_cuda_graph,
                    dp_cooperation_info=batch.dp_cooperation_info,
                )
        elif batch.forward_mode.is_idle():
            self.parts.batch_result_processor.process_batch_result_idle(batch, result)
        self.leave_finished_reqs()

        # Submit this batch's queued host backups before the next scheduler step.
        self.parts.kv.tree_cache.flush_pending_backups()

        self.parts.metrics_sink.log_batch_result_stats(batch, result)

        # Emit forward pass metrics (every iteration when enabled)
        if self.parts.metrics_sink.enable_fpm:
            self.hub.metrics_reporter._emit_forward_pass_metrics(batch, result)

        self.parts.request_retirer.clear_finished_media(batch.reqs)
        self.parts.output_streamer.maybe_reply_health_check()
        self.parts.metrics_sink.update_device_timer()

        if self.parts.scripted_scheduler_hook is not None:
            self.parts.scripted_scheduler_hook.on_process_result(batch)

    def add_external_corpus(
        self, recv_req: AddExternalCorpusReqInput
    ) -> Optional[AddExternalCorpusReqOutput]:
        if self.parts.external_corpus_manager is None:
            return AddExternalCorpusReqOutput(
                success=False,
                message="Ngram speculative decoding is not enabled.",
            )
        return self.parts.external_corpus_manager.add(recv_req)

    def remove_external_corpus(
        self, recv_req: RemoveExternalCorpusReqInput
    ) -> RemoveExternalCorpusReqOutput:
        if self.parts.external_corpus_manager is None:
            return RemoveExternalCorpusReqOutput(
                success=False,
                message="Ngram speculative decoding is not enabled.",
            )
        return self.parts.external_corpus_manager.remove(recv_req)

    def list_external_corpora(
        self, recv_req: ListExternalCorporaReqInput
    ) -> ListExternalCorporaReqOutput:
        if self.parts.external_corpus_manager is None:
            return ListExternalCorporaReqOutput(
                success=False,
                message="Ngram speculative decoding is not enabled.",
            )
        return self.parts.external_corpus_manager.list(recv_req)

    def clear_hicache_storage_wrapped(self, recv_req: ClearHiCacheReqInput):
        if get_memory().enable_hierarchical_cache or get_memory().enable_lmcache:
            self.parts.kv.tree_cache.clear_storage_backend()
            logger.info("Hierarchical cache or LMCache cleared successfully!")
            if_success = True
        else:
            logging.warning("Hierarchical cache or LMCache is not enabled.")
            if_success = False
        return ClearHiCacheReqOutput(success=if_success)

    @scheduler_stage_method(SCHEDULER_STAGE_IDLE)
    def on_idle(self):
        """Idle housekeeping: guard, check, metrics, reset, sleep."""
        self.parts.dp_attn_adapter.drop_sync_wait_carry()
        # Flush any health-check signal deferred while the engine was busy.
        self.parts.output_streamer.maybe_reply_health_check()

        # Publish before the fully-idle gate: a no-batch-but-not-idle stall
        # (queues parked under KV pressure / disagg transfer) has no
        # process_batch_result to publish the growing gauge, and gating here
        # froze /get_loads, DP balancing, and the LoadStat for the stall. This
        # path is polled repeatedly, so a wall-clock floor bounds the
        # O(queue) get_loads for both sinks; the fully-idle publish runs
        # post-flush below.
        fully_idle = self.hub.request_registry.is_fully_idle()
        if not fully_idle:
            self.parts.metrics_sink.record_scheduler_active(time.monotonic_ns())
            now = time.monotonic()
            if now - self._last_stall_publish_ts >= LOAD_STALL_REFRESH_S:
                self._last_stall_publish_ts = now
                snapshot = self.publish_load_snapshot(force=True)
                self.parts.load_publisher.publish_load_stat(
                    self.hub.load_inquirer.get_loads, force=True, snapshot=snapshot
                )
            if (
                hicache_storage_enabled()
                or self.mode.disaggregation_mode != DisaggregationMode.NULL
                or get_memory().enable_lmcache
            ):
                # Storage and transfer workers need the GIL between I/O calls.
                # Singleton PD polls no longer yield through a collective.
                time.sleep(0)
            return
        self.parts.metrics_sink.record_scheduler_idle()

        if get_memory().enable_unified_memory:
            try:
                self.parts.kv.token_to_kv_pool_allocator.flush_opportunistic()
            except Exception:
                pass

        # memory leak check (skipped for hisparse — pool counters intentionally
        # diverge during host-backup, see _get_swa_token_info clamp). Requests
        # holding KV until an ack lands keep the scheduler from fully idle.
        with self.scheduler_stage_metrics.record(SCHEDULER_STAGE_SANITY_CHECK_CACHE):
            if not get_memory().enable_hisparse:
                has_leak, messages = self.hub.invariant_checker._check_all_pools(
                    self.parts.pool_stats_observer.get_pool_stats(),
                )
                if has_leak:
                    self.hub.invariant_checker._report_leak("pool", "\n".join(messages))
                self.hub.invariant_checker._check_req_pool()
                # Byte-conservation diagnostic (allocator-owned; static pools
                # return [] — the token identity above can't see byte leaks).
                byte_violations = (
                    self.parts.kv.token_to_kv_pool_allocator.verify_byte_accounting()
                )
                if byte_violations:
                    self.hub.invariant_checker._report_leak(
                        "pool-bytes", "\n".join(byte_violations)
                    )

            # tree cache sanity check
            self.hub.invariant_checker._check_tree_cache()

        # metrics every 30s
        self.hub.metrics_reporter._maybe_log_idle_metrics()

        # kv event publishing
        self.parts.kv_events_publisher.publish_kv_events()

        # reset token ratio
        self.parts.new_token_ratio_tracker.reset()

        # Nothing decodes: release the requests the last decode step's inputs
        # still reference.
        self.parts.step_builder.clear_decode_rows()

        # Fully-idle publish, post-flush so the gauge reflects compacted KV.
        # Forced (immediate) so the busy->idle transition is never delayed.
        snapshot = self.publish_load_snapshot(force=True)
        self.parts.load_publisher.publish_load_stat(
            self.hub.load_inquirer.get_loads, force=True, snapshot=snapshot
        )

        # sleep until next event
        self.maybe_sleep_on_idle()
        self.parts.metrics_sink.record_scheduler_idle()

    def _record_scheduler_state_for_paused_engine(self) -> None:
        if self.hub.request_registry.is_fully_idle():
            self.parts.metrics_sink.record_scheduler_idle()
        else:
            self.parts.metrics_sink.record_scheduler_active(time.monotonic_ns())

    def attach_hicache_storage_wrapped(
        self, recv_req: AttachHiCacheStorageReqInput
    ) -> AttachHiCacheStorageReqOutput:
        if not get_memory().enable_hierarchical_cache:
            return AttachHiCacheStorageReqOutput(
                success=False, message="Hierarchical cache is not enabled."
            )

        if not self.hub.request_registry.is_fully_idle():
            return AttachHiCacheStorageReqOutput(
                success=False,
                message=(
                    "Reject attach: scheduler is not idle. "
                    f"#queue-req={len(self.hub.request_registry.queued_reqs())} "
                    f"#running-req={len(self.hub.request_registry.running_reqs())}"
                ),
            )

        if not hasattr(self.parts.kv.tree_cache, "attach_storage_backend"):
            return AttachHiCacheStorageReqOutput(
                success=False,
                message="Current tree_cache implementation does not support dynamic attach.",
            )

        try:
            ok, msg = self.parts.kv.tree_cache.attach_storage_backend(
                storage_backend=recv_req.hicache_storage_backend,
                storage_backend_extra_config_json=recv_req.hicache_storage_backend_extra_config_json,
                served_model_name=get_serving().served_model_name,
                hicache_storage_prefetch_policy=recv_req.hicache_storage_prefetch_policy,
                hicache_write_policy=recv_req.hicache_write_policy,
            )
        except Exception as e:
            logger.exception("Attach HiCache storage backend failed with exception.")
            return AttachHiCacheStorageReqOutput(success=False, message=str(e))
        if ok:
            hicache_fields = {
                "hicache_storage_backend": recv_req.hicache_storage_backend
            }
            if recv_req.hicache_storage_backend_extra_config_json is not None:
                hicache_fields["hicache_storage_backend_extra_config"] = (
                    recv_req.hicache_storage_backend_extra_config_json
                )
            if recv_req.hicache_storage_prefetch_policy is not None:
                hicache_fields["hicache_storage_prefetch_policy"] = (
                    recv_req.hicache_storage_prefetch_policy
                )
            if recv_req.hicache_write_policy is not None:
                hicache_fields["hicache_write_policy"] = recv_req.hicache_write_policy
            get_context().override("scheduler.attach_hicache", **hicache_fields)
            logger.info(
                f"Attached HiCache storage backend: {recv_req.hicache_storage_backend}"
            )
        return AttachHiCacheStorageReqOutput(success=ok, message=msg)

    def detach_hicache_storage_wrapped(
        self, recv_req: DetachHiCacheStorageReqInput
    ) -> DetachHiCacheStorageReqOutput:
        if not get_memory().enable_hierarchical_cache:
            return DetachHiCacheStorageReqOutput(
                success=False, message="Hierarchical cache is not enabled."
            )

        if not self.hub.request_registry.is_fully_idle():
            return DetachHiCacheStorageReqOutput(
                success=False,
                message=(
                    "Reject detach: scheduler is not idle. "
                    f"#queue-req={len(self.hub.request_registry.queued_reqs())} "
                    f"#running-req={len(self.hub.request_registry.running_reqs())}"
                ),
            )

        if not hasattr(self.parts.kv.tree_cache, "detach_storage_backend"):
            return DetachHiCacheStorageReqOutput(
                success=False,
                message="Current tree_cache implementation does not support dynamic detach.",
            )

        # Idempotent detach: even if scheduler thinks storage is disabled, we still
        # attempt best-effort cleanup in tree_cache (it may have leftover state).
        try:
            ok, msg = self.parts.kv.tree_cache.detach_storage_backend()
        except Exception as e:
            logger.exception("Detach HiCache storage backend failed with exception.")
            return DetachHiCacheStorageReqOutput(success=False, message=str(e))

        if ok or (not hicache_storage_enabled()):
            # Treat "already disabled / nothing to do" as success for idempotence.
            get_context().override(
                "scheduler.detach_hicache",
                hicache_storage_backend=None,
                hicache_storage_backend_extra_config=None,
            )
            logger.info("Detached HiCache storage backend.")
            return DetachHiCacheStorageReqOutput(
                success=True, message=msg or "HiCache storage backend is detached."
            )

        return DetachHiCacheStorageReqOutput(success=False, message=msg)

    def flush_cache(self, empty_cache: bool = True):
        """Flush memory pools (e.g., KV cache, Mamba cache) and optionally empty device allocator cache."""
        return self.hub.step_planner.flush_cache(empty_cache)

    def get_internal_state(self, recv_req: GetInternalStateReq):
        return self.hub.internal_state.get_internal_state(recv_req)

    def set_internal_state(self, recv_req: SetInternalStateReq):
        return self.hub.internal_state.set_internal_state(recv_req)

    def save_remote_model(self, **kwargs):
        self.hub.weight_updater.save_remote_model(kwargs)

    def save_sharded_model(self, **kwargs):
        self.hub.weight_updater.save_sharded_model(kwargs)

    def handle_rpc_request(self, recv_req: RpcReqInput):
        # Handle RPC requests
        logger.info(
            f"handle_rpc_request: {recv_req.method}, param: {recv_req.parameters}"
        )

        success = True
        exec = None
        try:
            func = getattr(self, recv_req.method)
            if recv_req.parameters is not None:
                func(**recv_req.parameters)
            else:
                func()
        except Exception as e:
            success = False
            exec = e
            logger.error(f"Failed to call rpc {recv_req.method}: {str(e)}")

        barrier(group=get_parallel().tp_group.cpu_group)
        return RpcReqOutput(success=success, message="" if not exec else str(exec))

    def handle_update_weight_version(
        self, recv_req: UpdateWeightVersionReqInput
    ) -> UpdateWeightVersionReqOutput:
        self.hub.weight_updater.record_weight_version_change(
            new_version=recv_req.new_version
        )
        return UpdateWeightVersionReqOutput()

    # The abort RPC handler.
    def abort_request(self, recv_req: AbortReq):
        self.hub.request_registry.abort_request(recv_req)

    def _pause_engine(self) -> Tuple[List[Req], int]:
        raise NotImplementedError()

    def pause_generation(self, recv_req: PauseGenerationReqInput):
        assert recv_req.mode in ("in_place", "retract")
        self.parts.flags._engine_paused = True

        if recv_req.mode == "in_place":
            # In-place pause: just set the flag and return immediately.
            # The request lists, chunked_req and the pending steps are left
            # untouched. On resume, the event loop commits the pending steps
            # and schedules from the lists through the standard code paths.
            return

        # Under PP the ring commits its pending steps once their outputs come
        # back over the relay; there is nothing to commit here.
        if self.parts.pending and get_parallel().pp_size == 1:
            self.commit_pending_steps()

        self.hub.step_planner.retract_for_pause()

        # Surface the paused state to dashboards immediately. The scheduler
        # event loop short-circuits before reaching ``on_idle`` while paused,
        # so without this hop ``gen_throughput`` retains its last non-zero
        # value and KV events are not flushed for the entire pause window
        # (e.g. across a weight update). Zero the gauge, force a one-shot
        # idle log by resetting the rate-limit timestamp, and flush pending
        # KV events.
        self.parts.metrics_sink.last_gen_throughput = 0.0
        if self.parts.metrics_sink.current_scheduler_metrics_enabled:
            self.parts.metrics_sink.metrics_collector.last_log_time = 0.0
            self.hub.metrics_reporter._maybe_log_idle_metrics()
        self.parts.kv_events_publisher.publish_kv_events()

    def continue_generation(self, recv_req: ContinueGenerationReqInput):
        if recv_req.torch_empty_cache:
            before_mb = torch.cuda.memory_reserved() / (1024 * 1024)
            torch.cuda.empty_cache()
            after_mb = torch.cuda.memory_reserved() / (1024 * 1024)
            logger.info(
                f"[continue_generation] torch.cuda.empty_cache() called: "
                f"reserved {before_mb:.1f} MB -> {after_mb:.1f} MB "
                f"(freed {before_mb - after_mb:.1f} MB)"
            )
        # Enqueue any rebootstrap requests that were staged during a
        # retract-mode pause. Deferring until resume keeps the preallocation
        # queue empty during the pause window (so an intervening weight update
        # can flush the cache) and recomputes the prefix KV under the updated
        # weights.
        if self.disagg_decode is not None:
            self.disagg_decode.prealloc_queue.enqueue_held_rebootstrap()
        self.parts.flags._engine_paused = False

    def handle_scale_elastic_ep(
        self, recv_req: ScaleElasticEPReqInput
    ) -> ScaleElasticEPReqOutput:
        """Begin a pending elastic EP scale-up request."""
        from sglang.srt.elastic_ep.elastic_ep import ElasticEPStateManager

        old_ep_size = ElasticEPStateManager.get_effective_ep_size()
        new_ep_size = recv_req.new_ep_size
        max_ep_size = get_parallel().max_world_size

        logger.debug(
            "[Elastic EP][scale] request received: new_ep_size=%d "
            "old_ep_size=%d max_ep_size=%d",
            new_ep_size,
            old_ep_size,
            max_ep_size,
        )

        if new_ep_size <= old_ep_size:
            return ScaleElasticEPReqOutput(
                success=False,
                message=(
                    f"new_ep_size ({new_ep_size}) must be greater than current "
                    f"effective_ep_size ({old_ep_size})."
                ),
                old_ep_size=old_ep_size,
                new_ep_size=new_ep_size,
            )
        if new_ep_size > max_ep_size:
            return ScaleElasticEPReqOutput(
                success=False,
                message=(
                    f"new_ep_size ({new_ep_size}) exceeds --max-ep-size "
                    f"({max_ep_size}). Restart with a larger --max-ep-size."
                ),
                old_ep_size=old_ep_size,
                new_ep_size=new_ep_size,
            )
        from sglang.srt.elastic_ep.topology import attn_replica_size

        replica_size = attn_replica_size()
        if new_ep_size % replica_size != 0:
            return ScaleElasticEPReqOutput(
                success=False,
                message=(
                    f"new_ep_size ({new_ep_size}) would admit an incomplete "
                    f"attention replica with size {replica_size}."
                ),
                old_ep_size=old_ep_size,
                new_ep_size=new_ep_size,
            )
        if ElasticEPStateManager.is_scaling():
            return ScaleElasticEPReqOutput(
                success=False,
                message=(
                    "A previous scale operation has not completed yet. Wait until "
                    "all pending ranks have joined before issuing another scale."
                ),
                old_ep_size=old_ep_size,
                new_ep_size=new_ep_size,
                pending_ep_size=ElasticEPStateManager.get_pending_ep_size(),
                scale_phase=ElasticEPStateManager.get_scale_phase(),
            )

        if not ElasticEPStateManager.request_scale(new_ep_size):
            return ScaleElasticEPReqOutput(
                success=False,
                message=(
                    "Failed to queue elastic EP scale: no elastic state or "
                    "scale already pending."
                ),
                old_ep_size=old_ep_size,
                new_ep_size=new_ep_size,
                pending_ep_size=ElasticEPStateManager.get_pending_ep_size(),
                scale_phase=ElasticEPStateManager.get_scale_phase(),
            )
        if (
            eplb_manager := self.parts.workers.tp_worker.model_runner.eplb_manager
        ) is not None:
            eplb_manager.disable_rebalance("elastic EP scale-up is pending")
        logger.debug(
            "[Elastic EP][scale] scale requested: target_ep_size=%d; "
            "waiting for a joining cohort",
            new_ep_size,
        )

        return ScaleElasticEPReqOutput(
            success=True,
            message=f"Scaling initiated from {old_ep_size} to {new_ep_size}",
            old_ep_size=old_ep_size,
            new_ep_size=new_ep_size,
            pending_ep_size=ElasticEPStateManager.get_pending_ep_size(),
            scale_phase=ElasticEPStateManager.get_scale_phase(),
        )

    def load_lora_adapter(
        self, recv_req: LoadLoRAAdapterReqInput
    ) -> LoadLoRAAdapterReqOutput:
        """In-place loading a new lora adapter from disk or huggingface."""

        result = self.parts.workers.tp_worker.load_lora_adapter(recv_req)
        if result.success:
            self.parts.kv_event_lora_names.register(
                lora_id=recv_req.lora_id, lora_name=recv_req.lora_name
            )
        return result

    def load_lora_adapter_from_tensors(
        self, recv_req: LoadLoRAAdapterFromTensorsReqInput
    ) -> LoadLoRAAdapterFromTensorsReqOutput:
        """In-place loading a new lora adapter from serialized tensors."""

        result = self.parts.workers.tp_worker.load_lora_adapter_from_tensors(recv_req)
        if result.success:
            self.parts.kv_event_lora_names.register(
                lora_id=recv_req.lora_id, lora_name=recv_req.lora_name
            )
        return result

    def unload_lora_adapter(
        self, recv_req: UnloadLoRAAdapterReqInput
    ) -> UnloadLoRAAdapterReqOutput:
        """Unload the lora adapter."""

        result = self.parts.workers.tp_worker.unload_lora_adapter(recv_req)
        return result

    def init_weights_send_group_for_remote_instance(
        self, recv_req: InitWeightsSendGroupForRemoteInstanceReqInput
    ):
        """Init the seed and client instance communication group."""
        success, message = (
            self.parts.workers.tp_worker.init_weights_send_group_for_remote_instance(
                recv_req
            )
        )
        return InitWeightsSendGroupForRemoteInstanceReqOutput(
            success=success, message=message
        )

    def send_weights_to_remote_instance(
        self, recv_req: SendWeightsToRemoteInstanceReqInput
    ):
        """Send the seed instance weights to the destination instance."""
        success, message = self.parts.workers.tp_worker.send_weights_to_remote_instance(
            recv_req
        )
        return SendWeightsToRemoteInstanceReqOutput(success=success, message=message)

    def slow_down(self, recv_req: SlowDownReqInput):
        t = recv_req.forward_sleep_time
        if t is not None and t <= 0:
            t = None
        self.parts.flags.forward_sleep_time = t
        return SlowDownReqOutput()

    def handle_pd_role_switch(self, recv_req: PdRoleSwitchReqInput):
        return role_switch.handle_pd_role_switch(self, recv_req)

    def expert_distribution_handle(self, recv_req: ExpertDistributionReq):
        action = recv_req.action
        if action == ExpertDistributionReqType.START_RECORD:
            get_global_expert_distribution_recorder().start_record()
        elif action == ExpertDistributionReqType.STOP_RECORD:
            get_global_expert_distribution_recorder().stop_record()
        elif action == ExpertDistributionReqType.DUMP_RECORD:
            get_global_expert_distribution_recorder().dump_record()
        else:
            raise ValueError(f"Unrecognized ExpertDistributionReq value: {recv_req=}")
        return ExpertDistributionReqOutput()

    def open_session(self, recv_req: OpenSessionReqInput):
        output = self.parts.session_controller.open(recv_req)
        if output.success and get_memory().enable_session_radix_cache:
            self.parts.kv.tree_cache.open_radix_session(recv_req.session_id)
        if (
            get_parallel().pp_rank == 0
            and get_parallel().tp_rank == 0
            and get_parallel().attn_cp_rank == 0
        ):
            return output
        return None

    def close_session(self, recv_req: CloseSessionReqInput):
        if get_memory().enable_session_radix_cache:
            self.parts.kv.tree_cache.release_radix_session(recv_req.session_id)
        if (
            recv_req.session_id in self.parts.session_controller
            or not get_memory().enable_session_radix_cache
        ):
            self.parts.session_controller.close(recv_req)

    def maybe_sleep_on_idle(self):
        if self.parts.idle_sleeper is not None:
            self.parts.idle_sleeper.maybe_sleep()

    def handle_freeze_gc(self, recv_req: FreezeGCReq):
        """Handle freeze_gc request: freeze scheduler's GC and forward to detokenizer."""
        freeze_gc("Scheduler")
        self.parts.ipc_channels.send_to_detokenizer.send_output(recv_req, recv_req)
        return None

    def handle_shutdown(self, recv_req: ShutdownReq):
        # Break the event loop; the finally in run_scheduler_process releases resources.
        self.parts.flags.gracefully_exit = True
        return None

    def configure_logging(self, recv_req: ConfigureLoggingReq):
        if recv_req.log_level is not None:
            logging.getLogger().setLevel(recv_req.log_level.upper())
        self.parts.ipc_channels.send_to_detokenizer.send_output(recv_req, recv_req)

    def handle_dumper_control(self, recv_req: DumperControlReqInput):
        from sglang.srt.debug_utils.dumper import dumper

        try:
            response: list = []
            if (
                not torch.distributed.is_initialized()
                or torch.distributed.get_rank() == 0
            ):
                response = dumper._http_manager.handle_request(
                    method=recv_req.method, body=recv_req.body
                )
            self.parts.ipc_channels.send_to_tokenizer.send_output(
                DumperControlReqOutput(success=True, response=response), recv_req
            )
        except Exception as e:
            print(f"[Scheduler] handle_dumper_control error: {e}", flush=True)
            self.parts.ipc_channels.send_to_tokenizer.send_output(
                DumperControlReqOutput(success=False, response=[], error=str(e)),
                recv_req,
            )


def dispatch_event_loop(scheduler: Scheduler):
    if scheduler.parts.server_args.enable_pd_role_switch:
        return role_switch.run_event_loop_supervisor(
            scheduler,
            _dispatch_event_loop_once,
        )
    return _dispatch_event_loop_once(scheduler)


def _dispatch_event_loop_once(scheduler: Scheduler):
    # A PD role switch can select a different loop on the same scheduler.
    scheduler.parts.flags.enable_skip_finishing_decode = False
    disaggregation_mode: DisaggregationMode = scheduler.mode.disaggregation_mode
    if get_parallel().pp_size > 1:
        # The ring takes each mode's stages, PD consensus included.
        scheduler.event_loop_pp()
    elif disaggregation_mode == DisaggregationMode.NULL:
        if not get_schedule().disable_overlap_schedule:
            scheduler.event_loop_overlap()
        else:
            scheduler.event_loop_normal()
    elif disaggregation_mode == DisaggregationMode.PREFILL:
        if not get_schedule().disable_overlap_schedule:
            scheduler.event_loop_overlap_disagg_prefill()
        else:
            scheduler.event_loop_normal_disagg_prefill()
    elif disaggregation_mode == DisaggregationMode.DECODE:
        if not get_schedule().disable_overlap_schedule:
            scheduler.event_loop_overlap_disagg_decode()
        else:
            scheduler.event_loop_normal_disagg_decode()


def resolve_spawn_dp_rank(dp_rank: Optional[int]) -> Optional[int]:
    """Resolve the launcher DP rank, falling back to ``SGLANG_DP_RANK``."""
    if dp_rank is None and "SGLANG_DP_RANK" in os.environ:
        return int(os.environ["SGLANG_DP_RANK"])
    return dp_rank


def configure_scheduler_process(
    server_args: ServerArgs,
    gpu_id: int,
    display_tp_rank: Optional[int] = None,
    display_dp_rank: Optional[int] = None,
    display_moe_ep_rank: Optional[int] = None,
) -> None:
    """Configure scheduler worker logging and process title.

    Runs after `publish`, so every rank it labels the process with comes from
    the context. display_* ranks are cosmetic; runtime ranks stay local.
    """
    kill_itself_when_parent_died()

    # Generate the logger prefix
    parallel = get_parallel()
    shown_dp = display_dp_rank if display_dp_rank is not None else parallel.dp_rank
    shown_tp = display_tp_rank if display_tp_rank is not None else parallel.tp_rank
    shown_moe_ep = (
        display_moe_ep_rank if display_moe_ep_rank is not None else parallel.moe_ep_rank
    )

    prefix = ""
    if shown_dp is not None:
        prefix += f" DP{shown_dp}"
    if parallel.pp_size > 1:
        prefix += f" PP{parallel.pp_rank}"
    if parallel.attn_cp_size > 1:
        prefix += f" ATTN_CP{parallel.attn_cp_rank}"
    if parallel.moe_dp_size > 1:
        prefix += f" MOE_DP{parallel.moe_dp_rank}"
    if parallel.tp_size > 1:
        prefix += f" TP{shown_tp}"
    if parallel.ep_size > 1:
        prefix += f" EP{shown_moe_ep}"

    # Config the process
    setproctitle.setproctitle(f"sglang::scheduler{prefix.replace(' ', '_')}")
    faulthandler.enable()

    # Configure the logger
    configure_logger(server_args, prefix=prefix)
    suppress_other_loggers()

    # Set cpu affinity to this gpu process
    if envs.SGLANG_SET_CPU_AFFINITY.get():
        set_gpu_proc_affinity(gpu_id)
    if not envs.SGLANG_NUMA_BIND_V2.get():
        numa_node = get_numa_node_if_available(server_args, gpu_id)
        if numa_node is not None:
            numa_bind_to_node(numa_node)


def run_scheduler_process(
    server_args: ServerArgs,
    port_args: PortArgs,
    gpu_id: int,
    tp_rank: int,
    pp_rank: int,
    dp_rank: Optional[int],
    pipe_writer,
    display_tp_rank: Optional[int] = None,
    display_dp_rank: Optional[int] = None,
    display_moe_ep_rank: Optional[int] = None,
):
    # Load plugins so hooks can override Scheduler and its dependencies.
    load_plugins()
    dp_rank = resolve_spawn_dp_rank(dp_rank)
    # Placement is needed before publish(): TP/PP select a WORLD rank;
    # ordinary DP replicas have separate WORLDs and need an explicit DP rank.
    publish(
        server_args,
        role="scheduler",
        ranks=SpawnRanks(
            world_rank=spawn_world_rank(server_args, tp_rank=tp_rank, pp_rank=pp_rank),
            dp_rank=dp_rank,
            gpu_id=gpu_id,
        ),
    )
    configure_scheduler_process(
        server_args,
        gpu_id,
        display_tp_rank=display_tp_rank,
        display_dp_rank=display_dp_rank,
        display_moe_ep_rank=display_moe_ep_rank,
    )
    parent_process = psutil.Process().parent()

    # Set up tracing
    if get_observability().enable_trace:
        process_tracing_init(
            get_observability().otlp_traces_endpoint,
            get_observability().otlp_service_name,
            trace_modules=get_observability().trace_modules,
        )
        thread_label = "Scheduler"
        if get_disagg().disaggregation_mode == "prefill":
            thread_label = "Prefill Scheduler"
        elif get_disagg().disaggregation_mode == "decode":
            thread_label = "Decode Scheduler"
        trace_set_thread_info(thread_label, tp_rank, dp_rank, pp_rank)

    # Create a scheduler and run the event loop
    scheduler = None
    try:
        scheduler = Scheduler(server_args, port_args)

        # Send initialization info back to the parent process
        pipe_writer.send(scheduler.get_init_info())

        # Run the event loop (blocks until a ShutdownReq sets gracefully_exit)
        scheduler.run_event_loop()

    except Exception:
        traceback = get_exception_traceback()
        logger.error(f"Scheduler hit an exception: {traceback}")
        parent_process.send_signal(signal.SIGQUIT)
        # Opt-in: SIGKILL the pgroup so sibling ranks don't spew thousands
        # of NCCL/TCPStore tracebacks before they finally die.
        if envs.SGLANG_KILLPG_ON_SCHEDULER_EXCEPTION.get():
            try:
                os.killpg(os.getpgrp(), signal.SIGKILL)
            except Exception:
                pass
    finally:
        if scheduler is not None:
            # FPM has a background ZMQ publisher thread that needs explicit
            # teardown to flush queued metrics and close the socket cleanly.
            scheduler.parts.metrics_sink._shutdown_fpm()
            # Graceful path only: on the exception path the GPU may be wedged
            # and the synchronize() in destroy() could itself hang.
            if scheduler.parts.flags.gracefully_exit:
                scheduler.release_host_resources()
                # Last: anything above may still need a working communicator.
                abort_distributed_environment()
