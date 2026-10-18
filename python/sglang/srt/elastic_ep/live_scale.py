"""Live scale coordinator. All WORLD operations run on the scheduler.

SGLANG_ELASTIC_LIVE_SCALE=prepare overlaps inactive NIXL connection only;
``split`` reuses graphs with --elastic-ep-reuse-cuda-graph; otherwise it
yields to eager serving between target CUDA graph captures.
The first implementation deliberately requires the non-overlap scheduler.
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid

logger = logging.getLogger(__name__)


def live_scale_mode():
    mode = os.getenv("SGLANG_ELASTIC_LIVE_SCALE", "")
    if mode not in ("", "prepare", "split"):
        raise ValueError("SGLANG_ELASTIC_LIVE_SCALE must be prepare or split")
    return mode


class CaptureProgress:
    """A deterministic service quantum shared by every DP scheduler rank."""

    def __init__(self, sizes, service_steps):
        if service_steps < 1:
            raise ValueError("Capture service quantum must be positive")
        self.sizes = tuple(reversed(sizes))
        if not self.sizes:
            raise ValueError("The graph capture plan must contain at least one shape")
        self.service_steps = service_steps
        self.served = 0
        self.index = 0

    def next_step(self):
        self.served += 1
        if self.served < self.service_steps:
            return None
        self.served = 0
        return self.sizes[self.index]

    def complete_step(self):
        self.index += 1
        return self.index == len(self.sizes)


class LiveScaleCoordinator:
    def __init__(self, runner):
        from sglang.srt.environ import envs
        from sglang.srt.runtime_context import get_exec, get_parallel, get_schedule

        self.runner = runner
        self.mode = live_scale_mode()
        self.reuse_graphs = (
            self.mode == "split"
            and get_exec().moe.elastic_ep_reuse_cuda_graph
            and runner._elastic_cuda_graph_enabled()
        )
        self.preparation = None
        self.candidate = None
        self.progress = None
        self.agreed_target = 0
        self.service_steps = int(
            os.getenv("SGLANG_ELASTIC_CAPTURE_SERVICE_STEPS", "32")
        )
        if self.service_steps < 1:
            raise ValueError("SGLANG_ELASTIC_CAPTURE_SERVICE_STEPS must be positive")
        if not get_schedule().disable_overlap_schedule:
            raise ValueError("Live scale requires --disable-overlap-schedule")
        if (
            get_exec().moe.elastic_ep_backend != "mooncake"
            or get_exec().moe.moe_a2a_backend != "nixl"
        ):
            raise ValueError("Live scale requires Mooncake PG and NIXL EP")
        if (
            get_parallel().attn_tp_size != 1
            or get_parallel().attn_cp_size != 1
            or get_parallel().dcp_size != 1
            or get_parallel().pp_size != 1
        ):
            raise ValueError("Live scale requires attention TP=1, CP=1, DCP=1 and PP=1")
        if runner.spec_algorithm.is_speculative():
            raise ValueError("Live scale does not support speculative decoding")
        if self.reuse_graphs and (
            get_parallel().moe_dense_tp_size != 1
            or not get_parallel().enable_dp_lm_head
        ):
            raise ValueError(
                "Live graph reuse requires dense TP=1 and DP LM head; "
                "joiner capture must not record cross-cohort collectives"
            )
        if self.reuse_graphs and get_exec().moe.ep_dispatch_algorithm == "lp":
            raise ValueError(
                "Live graph reuse does not yet support LP dispatch solvers"
            )
        if envs.SGLANG_ENABLE_GRAPH_POOL_BORROW.get():
            raise ValueError("Live scale does not support borrowing graph pool storage")
        if get_exec().overlap.enable_two_batch_overlap:
            raise ValueError("Live scale does not support two-batch overlap")

    def event(self, name, **fields):
        logger.info(
            "[Live scale event] %s",
            json.dumps(
                dict(
                    event=name,
                    time=time.time(),
                    rank=self.runner._elastic_global_rank(),
                    operation_id=(
                        self.preparation.operation_id if self.preparation else None
                    ),
                    **fields,
                )
            ),
        )

    def initialize(self):
        import torch.distributed as dist

        from sglang.srt.elastic_ep.elastic_ep import (
            clear_expert_map_inbox,
            register_scale_cohort,
            seed_barrier_epochs,
        )
        from sglang.srt.layers.moe.token_dispatcher.nixl import NixlEPBuffer
        from sglang.srt.runtime_context import get_exec, get_parallel

        NixlEPBuffer.initialize_for_live_prepare()
        if get_exec().moe.is_ep_scale_joiner:
            old = get_parallel().ep_join_rank_offset
            clear_expert_map_inbox(self.runner._elastic_global_rank())
            seed_barrier_epochs()
            dist.barrier(group=get_parallel().tp_group.cpu_group)
            if get_parallel().tp_rank == 0:
                register_scale_cohort(
                    old,
                    old + get_parallel().tp_size,
                    self.runner._elastic_cuda_graph_enabled(),
                )

    def start_preparation(self, old, target):
        from sglang.srt.distributed.utils import get_global_tcp_store
        from sglang.srt.elastic_ep.live_prepare import LiveScalePreparation
        from sglang.srt.layers.moe.token_dispatcher.nixl import NixlEPBuffer
        from sglang.srt.model_executor.model_runner_components.cuda_graph_setup import (
            resolve_elastic_cuda_graph_config,
        )
        from sglang.srt.runtime_context import get_exec

        if self.preparation is not None and self.preparation.target_size == target:
            return
        if self.preparation is not None and self.preparation.poll() != 1:
            raise RuntimeError(
                "Previous preparation is not safely complete; restart required"
            )
        store = get_global_tcp_store()
        key = f"elastic_ep/live_operation/{old}/{target}"
        if self.runner._elastic_global_rank() == 0:
            store.set(key, uuid.uuid4().hex.encode())
        operation_id = store.get(key).decode()
        manifest = repr(
            (
                self.mode,
                old,
                target,
                self.service_steps,
                get_exec().moe.elastic_ep_reuse_cuda_graph,
                NixlEPBuffer.live_layout_signature(),
                resolve_elastic_cuda_graph_config(self.runner).signature(),
            )
        )
        self.preparation = LiveScalePreparation(
            store,
            self.runner._elastic_global_rank(),
            old,
            target,
            get_exec().moe.elastic_ep_scale_timeout,
            operation_id,
            manifest,
        )
        if self.reuse_graphs and self.runner._elastic_global_rank() < old:
            self.reuse_snapshot = self.graph_storage_snapshot()
        self.event("prepare_start", old=old, target=target)
        NixlEPBuffer.start_live_prepare(self.preparation, self.runner.gpu_id)

    def prepare(self, old, target):
        import torch.distributed as dist

        from sglang.srt.elastic_ep.elastic_ep import ElasticEPStateManager
        from sglang.srt.runtime_context import get_exec

        state = ElasticEPStateManager.instance()
        error = None
        try:
            self.start_preparation(old, target)
            vote = self.preparation.poll()
            error = self.preparation.error
            if (
                time.monotonic() - state.pending_since
                > get_exec().moe.elastic_ep_scale_timeout
            ):
                vote, error = -1, "Live preparation timed out"
        except Exception as exc:
            vote, error = -1, str(exc)
        consensus = state.active_ranks.new_tensor(vote)
        dist.all_reduce(consensus, op=dist.ReduceOp.MIN, group=dist.group.WORLD)
        vote = consensus.item()
        if vote < 0:
            # A failed connection attempt cannot currently be cancelled reliably.
            # Fail closed, rather than allowing a retry to race a connection worker.
            raise RuntimeError(
                error or "Peer failed live preparation; restart required"
            )
        if vote == 0:
            return False
        self.event("prepare_complete", target=target)
        return True

    def finish_joiner(self):
        from sglang.srt.elastic_ep.elastic_ep import ElasticEPStateManager
        from sglang.srt.model_executor.model_runner_components.cuda_graph_setup import (
            finalize_cuda_graph_capture,
        )
        from sglang.srt.platforms import current_platform
        from sglang.srt.runtime_context import get_exec, get_parallel

        if not get_exec().moe.is_ep_scale_joiner:
            return
        old = get_parallel().ep_join_rank_offset
        target = old + get_parallel().tp_size
        self.event("joiner_local_ready", old=old, target=target)
        self.start_preparation(old, target)
        self.preparation.wait_connected()
        if self.reuse_graphs:
            try:
                self.capture_joiner_graphs(target)
            except Exception as exc:
                self.preparation.store.set(
                    f"error/{self.preparation.rank}", str(exc).encode()
                )
                raise
        current_platform.synchronize()
        self.preparation.mark_ready()
        self.runner._initialize_elastic_ep_joiner()
        ElasticEPStateManager.on_scale(old, target)
        if self.reuse_graphs:
            self.finish_reused_graphs(target, old, joiner=True)
        elif self.mode == "split" and self.runner._elastic_cuda_graph_enabled():
            self.activate(target, old)
        else:
            if self.runner._elastic_cuda_graph_enabled():
                self.runner._recapture_elastic_cuda_graphs()
            finalize_cuda_graph_capture(self.runner)
            self.runner._finish_joiner_ready(target)
            self.preparation = None

    def capture_joiner_graphs(self, target):
        """Capture on joiner GPUs while the old cohort keeps replaying its graphs.

        The joiner's scheduler has not entered serving yet. Its TP barriers are
        cohort-local, DP scheduling has not started, and remote NIXL peers stay
        masked until the whole cohort commits. No expanded WORLD operation is
        allowed here. The same buffers and graph entries survive admission.
        """
        from sglang.srt.layers.moe.token_dispatcher.nixl import NixlEPBuffer
        from sglang.srt.model_executor.model_runner_components.cuda_graph_setup import (
            capture_decode_graph,
            refresh_deep_gemm_layout_memory_budget,
        )
        from sglang.srt.platforms import current_platform

        mr = self.runner
        self.event("joiner_capture_start", target=target)
        started = time.monotonic()
        NixlEPBuffer.prepare_local_capture(target, mr._elastic_global_rank())
        refresh_deep_gemm_layout_memory_budget(mr)
        capture = capture_decode_graph(model_runner=mr)
        if capture.runner is None:
            raise RuntimeError("Joiner preparation did not produce decode graphs")
        current_platform.synchronize()
        mr.decode_cuda_graph_runner = capture.runner
        mr.graph_memory_usage["decode"] = capture.memory_usage_gb
        mr.graph_time_usage["decode"] = capture.capture_time
        self.reuse_snapshot = self.graph_storage_snapshot()
        self.event(
            "joiner_capture_end", target=target, duration=time.monotonic() - started
        )

    def graph_storage_snapshot(self):
        from sglang.srt.eplb.expert_location import get_global_expert_location_metadata
        from sglang.srt.layers.moe.token_dispatcher.nixl import NixlEPBuffer

        runner = self.runner.decode_cuda_graph_runner
        if runner is None or runner is self.runner.eager_runner:
            raise RuntimeError("Live graph reuse requires captured decode graphs")
        metadata = get_global_expert_location_metadata()
        pointers = tuple(
            tensor.data_ptr() if tensor is not None else None
            for tensor in (
                metadata.logical_to_all_physical_map,
                metadata.logical_to_all_physical_map_num_valid,
                metadata.logical_to_rank_dispatch_physical_map,
            )
        )
        return (
            id(runner),
            tuple(id(g) for g in runner.backend._graphs.values()),
            id(NixlEPBuffer._state().buffer),
            pointers,
            # Admission may reload relabelled joiner slots. Such reloads must
            # copy into captured parameter/scale storage, never replace it.
            tuple(
                (name, tensor.data_ptr())
                for name, tensor in (
                    *self.runner.model.named_parameters(),
                    *self.runner.model.named_buffers(),
                )
            ),
        )

    def finish_reused_graphs(self, target, old, *, joiner=False):
        from sglang.srt.elastic_ep.elastic_ep import ElasticEPStateManager
        from sglang.srt.model_executor.model_runner_components.cuda_graph_setup import (
            finalize_cuda_graph_capture,
        )

        mr = self.runner
        if (
            mr.decode_cuda_graph_runner is None
            or mr.decode_cuda_graph_runner is mr.eager_runner
        ):
            raise RuntimeError(
                "Cannot commit graph reuse without prepared decode graphs"
            )
        if self.graph_storage_snapshot() != self.reuse_snapshot:
            raise RuntimeError(
                "Graph-facing storage changed during live scale preparation"
            )
        if joiner:
            finalize_cuda_graph_capture(mr)
            self.joiner_finalized = True
        # All graph-facing storage was updated in place. Only host bookkeeping
        # follows the new width; there is no eager interval or graph recapture.
        mr.decode_cuda_graph_runner.num_dp_ranks = mr.eager_runner.num_dp_ranks
        # Joiners did not receive the HTTP scale request. Give commit_scale the
        # same pending target so they enter warming_up/serving_expanded too;
        # otherwise they skip EPLB's WORLD collectives while survivors enter them.
        if joiner:
            ElasticEPStateManager.instance().pending_ep_size = target
        ElasticEPStateManager.mark_syncing_new_world()
        ElasticEPStateManager.commit_scale()
        mr._elastic_scale_ready_barrier(target, "LIVE_GRAPH_REUSE")
        mr._rearm_eplb_after_elastic_scale()
        self.old, self.target = old, target
        self.event(
            "graphs_reused",
            target=target,
            joiner=joiner,
            graph_count=len(self.reuse_snapshot[1]),
            storage_preserved=True,
        )
        self.event("serving_ready", target=target)
        self.event("graph_ready", target=target)
        self.report(complete=True)

    def activate(self, target, old):
        import torch.distributed as dist

        from sglang.srt.elastic_ep.elastic_ep import ElasticEPStateManager
        from sglang.srt.model_executor.model_runner_components.cuda_graph_setup import (
            drop_elastic_cuda_graph_state,
            refresh_deep_gemm_layout_memory_budget,
            resolve_elastic_cuda_graph_config,
            sync_elastic_cuda_graph_config,
        )
        from sglang.srt.platforms import current_platform

        mr = self.runner
        self.event("activation_graph_setup_start", target=target)
        config = resolve_elastic_cuda_graph_config(mr)
        sync_elastic_cuda_graph_config(
            config=config, device=mr.device, world_group=dist.group.WORLD
        )
        drop_elastic_cuda_graph_state(
            decode_runner=mr.decode_cuda_graph_runner, eager_runner=mr.eager_runner
        )
        mr.decode_cuda_graph_runner = None
        refresh_deep_gemm_layout_memory_budget(mr)
        mr._kernel_warmed_up = False
        self.candidate = mr._decode_cuda_graph_runner_cls()(mr, defer_capture=True)
        from sglang.srt.model_executor.runner_utils.capture_mode import (
            model_capture_mode,
        )

        self.candidate.buffers.reset_index_buffers()
        self.candidate.buffers.seq_lens.fill_(self.candidate.seq_len_fill_value)
        self.candidate.buffers.seq_lens_cpu.fill_(self.candidate.seq_len_fill_value)
        with model_capture_mode():
            self.candidate.warmup()
        current_platform.synchronize()
        mr._elastic_scale_ready_barrier(target, "LIVE_EAGER")
        state = ElasticEPStateManager.instance()
        state.effective_ep_size = target
        state.pending_ep_size = target
        state.has_scaled = True
        state.scale_phase = "graph_warming"
        state.reset()
        self.progress = CaptureProgress(self.candidate.capture_bs, self.service_steps)
        self.old, self.target = old, target
        self.capture_seconds = 0.0
        self.event("serving_ready", target=target)
        self.report(complete=False)

    def report(self, *, complete):
        from sglang.srt.managers.io_struct import ElasticScaleUpdateReq

        if self.runner._elastic_global_rank() == 0:
            self.runner._pending_elastic_scale_update = ElasticScaleUpdateReq(
                success=True,
                effective_ep_size=self.target,
                slot_offset=self.old,
                slot_count=self.target - self.old,
                graph_ready=complete,
            )
        if complete:
            self.preparation = None

    def local_control_target(self):
        from sglang.srt.elastic_ep.elastic_ep import ElasticEPStateManager
        from sglang.srt.runtime_context import get_exec

        state = ElasticEPStateManager.instance()
        if state.pending_ep_size is None:
            return 0
        if self.candidate is None and state.pending_since is not None:
            if (
                time.monotonic() - state.pending_since
                > get_exec().moe.elastic_ep_scale_timeout
            ):
                # Once preparation mutates NIXL, cancellation is unsafe.
                # Otherwise abort the request together at the scheduler boundary.
                if self.preparation is not None and not state.pending_recover_ranks:
                    return -1
                return -2
        return state.pending_ep_size

    def observe_control_targets(self, targets):
        """Consume the existing DP batch gather; never add an idle WORLD op."""
        if -1 in targets:
            raise RuntimeError("Timed out during live preparation; restart required")
        if -2 in targets:
            self.agreed_target = -2
            return
        nonzero = set(targets) - {0}
        if len(nonzero) > 1:
            raise RuntimeError("Conflicting live scale targets across DP ranks")
        self.agreed_target = targets[0] if targets and len(set(targets)) == 1 else 0

    def tick(self):
        import torch.distributed as dist

        from sglang.srt.elastic_ep.elastic_ep import ElasticEPStateManager
        from sglang.srt.platforms import current_platform

        mr = self.runner
        if self.agreed_target == -2:
            mr._fail_scale(
                "Timed out agreeing on an elastic scale target",
                ElasticEPStateManager.get_effective_ep_size(),
            )
            self.agreed_target = 0
            return
        if self.candidate is None:
            # The existing DP batch gather observes staggered HTTP delivery.
            # A separate WORLD operation during idle joiner startup can race
            # the launch-time TP gather. Only enter WORLD after agreed demand.
            if self.agreed_target > 0:
                mr.maybe_join_ep_ranks()
            return
        size = self.progress.next_step()
        if size is None:
            return
        # This hook runs after process_batch_result in the non-overlap loop.
        current_platform.synchronize()
        self.event("capture_start", batch_size=size)
        started = time.monotonic()
        self.candidate.capture_step(size)
        current_platform.synchronize()
        dist.barrier(group=dist.group.WORLD)
        elapsed = time.monotonic() - started
        self.capture_seconds += elapsed
        self.event("capture_end", batch_size=size, duration=elapsed)
        if not self.progress.complete_step():
            return
        mr.decode_cuda_graph_runner = self.candidate
        self.candidate = None
        from sglang.srt.model_executor.model_runner_components.cuda_graph_setup import (
            finalize_cuda_graph_capture,
        )
        from sglang.srt.runtime_context import get_exec

        if get_exec().moe.is_ep_scale_joiner and not getattr(
            self, "joiner_finalized", False
        ):
            finalize_cuda_graph_capture(mr)
            self.joiner_finalized = True
        mr.graph_time_usage["decode"] = self.capture_seconds
        ElasticEPStateManager.commit_scale()
        mr._rearm_eplb_after_elastic_scale()
        self.event("graph_ready", target=self.target)
        self.report(complete=True)
