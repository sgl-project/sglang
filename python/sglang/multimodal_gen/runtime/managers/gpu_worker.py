# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
import gc
import logging
import multiprocessing as mp
import os
import tempfile
import time
from contextlib import ExitStack
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Iterator, List, Union

import numpy as np
import torch
from setproctitle import setproctitle

from sglang.multimodal_gen.runtime.warmup_request_builder import lighten_warmup_req

from sglang.multimodal_gen.runtime.utils.logging_utils import (  # isort: skip
    globally_suppress_loggers,
)

# spawned workers import model dependencies before entering run_scheduler_process
globally_suppress_loggers()

from sglang.multimodal_gen import envs
from sglang.multimodal_gen.runtime.distributed import (
    get_replica_group,
    get_sp_group,
    get_tp_rank,
    get_tp_world_size,
    maybe_init_distributed_environment_and_model_parallel,
    model_parallel_is_initialized,
)
from sglang.multimodal_gen.runtime.distributed.device_communicators.ipc_a2a import (
    IPC_A2A,
)
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    get_cfg_group,
    get_classifier_free_guidance_rank,
    get_classifier_free_guidance_world_size,
    get_ring_parallel_rank,
    get_ring_parallel_world_size,
    get_tp_group,
    get_ulysses_parallel_rank,
    get_ulysses_parallel_world_size,
)
from sglang.multimodal_gen.runtime.entrypoints.utils import (
    materialize_output_sample,
    post_process_sample,
    save_outputs,
)
from sglang.multimodal_gen.runtime.managers.memory_managers.auto_residency import (
    GIB_BYTES,
    MIN_POST_ADJUSTMENT_REGRESSION_NS,
    PLACEMENT_STATUS_ADJUSTED,
    PLACEMENT_STATUS_ROLLBACK_FAILED,
    PLACEMENT_STATUS_ROLLED_BACK,
    PLACEMENT_STATUS_SKIPPED,
    PLACEMENT_STATUS_VALIDATED,
    POST_ADJUSTMENT_REGRESSION_FRACTION,
    AppliedResidencyChange,
    AutoResidencyPlan,
    AutoResidencyRollbackError,
    DefaultWorkload,
    RankResidencyReport,
    ResidencyTarget,
    WarmupMemoryRecord,
    apply_residency_changes,
    collect_residency_targets,
    commit_residency_changes,
    component_current_device_weight_bytes,
    component_runtime_weight_bytes,
    current_placement_reserve_shortfall_bytes,
    describe_error,
    estimate_allocator_headroom_bytes,
    estimate_candidate_latency_savings_ns,
    estimate_default_workload_peak_bytes,
    estimate_default_workload_timing,
    estimate_layerwise_layer_uses,
    estimate_workload_phase_peaks,
    format_applied_changes,
    format_plan_summary,
    layerwise_host_pin_capacity_bytes,
    layerwise_mapped_bytes,
    layerwise_pinned_host_bytes,
    layerwise_streamed_mapped_bytes,
    measured_failed_workload_phase_peaks,
    plan_auto_residency,
    plan_summary_payload,
    pre_warmup_residency_targets,
    rank_candidates_by_h2d_savings,
    resolve_default_workload,
    resolve_measured_default_workload,
    rollback_residency_changes,
)
from sglang.multimodal_gen.runtime.managers.memory_managers.component_manager import (
    get_global_component_residency_manager,
    peek_global_component_residency_manager,
)
from sglang.multimodal_gen.runtime.managers.memory_managers.component_residency import (
    RESIDENT,
)
from sglang.multimodal_gen.runtime.managers.memory_managers.host_memory_budget import (
    HOST_COPY_RESERVE_BYTES,
    host_memory_available_bytes,
    shared_pool_available_bytes,
)
from sglang.multimodal_gen.runtime.managers.memory_managers.layerwise_offload import (
    LayerwiseUsageTracker,
    configure_layerwise_offload_modules,
)
from sglang.multimodal_gen.runtime.managers.memory_managers.layerwise_offload_components import (
    is_dit_component_name,
)
from sglang.multimodal_gen.runtime.managers.memory_managers.memory_occupation_controller import (
    MemoryOccupationController,
)
from sglang.multimodal_gen.runtime.pipelines_core import (
    ComposedPipelineBase,
    LoRAPipeline,
    Req,
    build_pipeline,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch
from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.multimodal_gen.runtime.post_training.gpu_worker_post_training_mixin import (
    GPUWorkerPostTrainingMixin,
)
from sglang.multimodal_gen.runtime.realtime.session import RealtimeSessionCache
from sglang.multimodal_gen.runtime.server_args import PortArgs, ServerArgs
from sglang.multimodal_gen.runtime.server_args.auto_tune import (
    fixed_loading_residency_components,
)
from sglang.multimodal_gen.runtime.utils.common import set_cuda_arch, set_musa_arch
from sglang.multimodal_gen.runtime.utils.logging_utils import (
    configure_logger,
    init_logger,
)
from sglang.multimodal_gen.runtime.utils.perf_logger import (
    PerformanceLogger,
    capture_memory_snapshot,
)
from sglang.multimodal_gen.runtime.utils.profiler import maybe_record_function
from sglang.multimodal_gen.runtime.utils.realtime_video import (
    RAW_RGB_CONTENT_TYPE,
    build_raw_rgb_frame_batches,
)
from sglang.multimodal_gen.runtime.utils.trace_wrapper import (
    DiffStage,
    init_diffusion_tracing,
    trace_slice,
)
from sglang.multimodal_gen.utils import kill_itself_when_parent_died
from sglang.srt.environ import third_party_cache_defaults
from sglang.srt.utils.network import NetworkAddress

logger = init_logger(__name__)


@dataclass
class _ExpandedOutputParts:
    tensor_outputs: list[torch.Tensor] = field(default_factory=list)
    list_outputs: list[Any] = field(default_factory=list)
    tensor_audio: list[torch.Tensor] = field(default_factory=list)
    trajectory_latents: list[torch.Tensor] = field(default_factory=list)
    noise_preds: list[torch.Tensor] = field(default_factory=list)
    output_file_paths: list[str] = field(default_factory=list)
    metrics_list: list[Any] = field(default_factory=list)
    trajectory_decoded_parts: list[list[torch.Tensor]] | None = None


def _worker_cpu_intra_op_threads(num_gpus: int) -> int | None:
    """CPU intra-op thread budget for one of `num_gpus` co-located workers.

    torch defaults the intra-op pool to every host core in every worker, so
    co-located workers oversubscribe the host num_gpus-fold and any CPU op
    past the ~32k-element parallel grain pays pool wakeup contention instead
    of microseconds (measured 500x on request-static packed layouts). An
    explicit OMP_NUM_THREADS keeps deployer intent (returns None).
    """
    if "OMP_NUM_THREADS" in os.environ:
        return None
    cpu_count = os.cpu_count() or 1
    return max(1, min(16, cpu_count // max(1, num_gpus)))


def _format_calibration_timing(records) -> str:
    """One line of what the planner's duration model was fed by the last probe."""
    successful = [record for record in records if record.succeeded]
    if not successful:
        return "calibration timing: no successful probe"
    record = max(successful, key=lambda r: (r.workload_units(), r.total_duration_ms))
    stages = ", ".join(
        f"{name.replace('MiniMaxH3', '').replace('Stage', '')}={ms / 1000:.1f}s"
        for name, ms in record.stage_duration_ms.items()
        if ms >= 100
    )
    steps = ", ".join(f"{ms / 1000:.2f}" for ms in record.step_duration_ms)
    iterations = ", ".join(
        f"{name.replace('MiniMaxH3', '').replace('Stage', '')}={measured}->{target}"
        for name, (measured, target) in record.stage_iterations.items()
    )
    return (
        f"calibration timing ({record.width}x{record.height}x{record.num_frames}f, "
        f"{record.num_inference_steps} steps): total={record.total_duration_ms / 1000:.1f}s; "
        f"stages: {stages}; steps: [{steps}]; iterations: {iterations}"
    )


def _probe_total_duration_ns(records, target_units) -> int:
    """Wall time of the representative full-shape probe (0 without one)."""
    successful = [record for record in records if record.succeeded]
    if target_units is not None:
        at_target = [r for r in successful if r.workload_units() >= target_units]
        if at_target:
            successful = at_target
    if not successful:
        return 0
    record = max(successful, key=lambda r: (r.workload_units(), r.total_duration_ms))
    return max(0, int(record.total_duration_ms * 1_000_000))


PROBE_FIT_MIN_MARGIN_BYTES = 1 << 30


def _shape_label(req: Req) -> str:
    return f"{req.width}x{req.height}x{req.num_frames or 1}f"


def fit_auto_residency_probe(
    req: Req,
    *,
    records: list[WarmupMemoryRecord],
    free_bytes: int,
    total_bytes: int,
    server_args: ServerArgs,
) -> tuple[Req, int | None, int]:
    """Shrink a full-shape probe until its extrapolated peak fits the memory left.

    The probe measures the default workload under the load-safe placement, so
    a probe the card cannot hold would only be found out by running out of
    memory. The bounded warmup that runs before it gives one measurement to
    extrapolate from; while that extrapolation exceeds free memory minus the
    reserve, frames go first and then area, the ladder the OOM retry walks.
    Returns the fitted request, its estimate and the number of shrink steps.
    """
    # Only the probe has to fit, so the margin is allocator slack, not the
    # planner's placement reserve (which held back 4 GiB of a 32 GiB card and
    # shrank a probe that had 10 GiB to spare).
    budget = free_bytes - max(PROBE_FIT_MIN_MARGIN_BYTES, total_bytes // 50)
    fitted, steps = req, 0
    while True:
        units = (
            max(1, int(fitted.width or 1))
            * max(1, int(fitted.height or 1))
            * max(1, int(fitted.num_frames or 1))
        )
        estimate = estimate_default_workload_peak_bytes(
            records=records, target_units=units
        )
        if estimate is None or estimate <= budget:
            return fitted, estimate, steps
        lighter = lighten_warmup_req(server_args, fitted)
        if lighter is None:
            return fitted, estimate, steps
        fitted, steps = lighter, steps + 1


class GPUWorker(GPUWorkerPostTrainingMixin):
    """
    A worker that executes the model on a single GPU.
    """

    def __init__(
        self,
        local_rank: int,
        rank: int,
        master_port: int,
        server_args: ServerArgs,
    ):
        self.local_rank = local_rank
        self.rank = rank
        # the rank that materializes output and replies to the client: the
        # first rank of this DP replica, which is global rank 0 only at dp=1
        gpus_per_replica = max(1, server_args.num_gpus // (server_args.dp_size or 1))
        self.is_output_rank = rank % gpus_per_replica == 0
        self.master_port = master_port
        # FIXME: should we use tcp as distribute init method?
        self.server_args = server_args
        self.pipeline: ComposedPipelineBase = None

        self.init_device_and_model()
        self._load_peak_reserved_mb = (
            0.0
            if current_platform.is_cpu()
            else capture_memory_snapshot().peak_reserved_mb
        )
        self._runtime_peak_reserved_mb = 0.0
        # Warmup probes run the default workload's full shape and may exceed any
        # serving request; keep their peak out of the runtime figure.
        self._warmup_peak_reserved_mb = 0.0
        self._release_warmup_pool_before_serving = False
        self.sp_group = get_sp_group()
        self.sp_cpu_group = self.sp_group.cpu_group
        self.tp_group = get_tp_group()
        self.tp_cpu_group = self.tp_group.cpu_group

        self.cfg_group = get_cfg_group()
        self.cfg_cpu_group = self.cfg_group.cpu_group
        self._realtime_sessions = RealtimeSessionCache(max_sessions=1)
        self.memory_occupation: MemoryOccupationController | None = None
        # per-rank memory measurements of server warmup forwards; consumed by
        # the auto-residency placement decision before the server turns ready
        self._auto_residency_warmup_records: list[WarmupMemoryRecord] = []
        self._auto_residency_applied: list[AppliedResidencyChange] = []
        self._auto_residency_round_sizes: list[int] = []
        self._auto_residency_last_applied_plan: AutoResidencyPlan | None = None
        # Keep the serving solve and its validation on one objective. The
        # measured candidate layout may be accepted or rolled back, but does
        # not rewrite utility and start another search.
        self._auto_residency_reference_request_duration_ns: int | None = None
        self._auto_residency_reference_stage_duration_ns: dict[str, int] = {}
        self._auto_residency_reference_component_stages: dict[str, tuple[str, ...]] = {}
        self._auto_residency_repeated_components: set[str] = set()
        # default workload resolved once for the per-request residency hint
        self._cached_default_workload: DefaultWorkload | None = None
        self._cached_default_workload_failed = False

    def _default_workload_for_hint(self) -> DefaultWorkload | None:
        if (
            self._cached_default_workload is None
            and not self._cached_default_workload_failed
        ):
            try:
                self._cached_default_workload = resolve_default_workload(
                    self.server_args
                )
            except Exception:
                logger.debug("Default workload unresolvable", exc_info=True)
                self._cached_default_workload_failed = True
        return self._cached_default_workload

    def release_realtime_session(self, session_id: str) -> OutputBatch:
        """release the session of a realtime connection"""
        if not session_id:
            return OutputBatch(
                output={
                    "released": False,
                    "session_id": session_id,
                    "reason": "empty_session_id",
                }
            )

        released = self._realtime_sessions.release(session_id)
        if released:
            if torch.cuda.is_initialized():
                torch.cuda.empty_cache()
        return OutputBatch(output={"released": released, "session_id": session_id})

    def _configure_persistent_torch_compile_cache(self) -> None:
        """Persist torch.compile's Inductor/Triton cache across restarts"""
        compile_cache_root = os.path.join(
            envs.SGLANG_DIFFUSION_CACHE_ROOT, "torch_compile_cache"
        )
        tmp_root = tempfile.gettempdir()
        sglang_defaults = third_party_cache_defaults()
        for env_name, sub in (
            ("TORCHINDUCTOR_CACHE_DIR", "inductor"),
            ("TRITON_CACHE_DIR", "triton"),
        ):
            current = os.environ.get(env_name)
            if (
                current
                and current != sglang_defaults.get(env_name)
                and not current.startswith(tmp_root)
            ):
                # Respect an explicit, non-ephemeral user-provided cache dir.
                continue
            cache_path = os.path.join(compile_cache_root, sub)
            try:
                os.makedirs(cache_path, exist_ok=True)
            except OSError as e:
                logger.warning(
                    "Could not create torch.compile cache dir %s: %s", cache_path, e
                )
                continue
            os.environ[env_name] = cache_path
        logger.info(
            "torch.compile cache: TORCHINDUCTOR_CACHE_DIR=%s TRITON_CACHE_DIR=%s",
            os.environ.get("TORCHINDUCTOR_CACHE_DIR"),
            os.environ.get("TRITON_CACHE_DIR"),
        )

    def is_sleeping(self) -> bool:
        return self.memory_occupation.is_sleeping() if self.memory_occupation else False

    def _get_memory_occupation(self) -> MemoryOccupationController:
        if self.memory_occupation is None:
            self.memory_occupation = MemoryOccupationController(
                pipeline=self.pipeline,
                rank=self.rank,
                use_fsdp_inference=self.server_args.use_fsdp_inference,
            )
        return self.memory_occupation

    def _cap_device_memory_for_tests(self) -> None:
        """Make a large CI card behave like the consumer card a case targets.

        The caching allocator otherwise reserves past the pretended budget
        whenever the physical card has room, and a peak-VRAM baseline stops
        meaning "fits the card". OOM inside the cap is the intended signal.
        """
        cap_gib = envs.SGLANG_DIFFUSION_TEST_CAP_DEVICE_MEMORY_GIB
        if cap_gib is None or not current_platform.is_cuda():
            return
        device = torch.cuda.current_device()
        total = torch.cuda.get_device_properties(device).total_memory
        fraction = min(1.0, cap_gib * 1024**3 / total)
        torch.cuda.set_per_process_memory_fraction(fraction, device)
        logger.info(
            "Test hook: CUDA allocator capped at %.1f GiB (fraction %.4f)",
            cap_gib,
            fraction,
        )

    def init_device_and_model(self) -> None:
        """Initialize the device and load the model."""
        if not current_platform.is_mps():
            current_platform.set_device(current_platform.get_device(self.local_rank))
        self._cap_device_memory_for_tests()
        # num_gpus is the total world size across every node; the co-located,
        # CPU-contending worker count on THIS host is num_gpus // nnodes.
        local_num_gpus = self.server_args.num_gpus // self.server_args.nnodes
        intra_op_threads = _worker_cpu_intra_op_threads(local_num_gpus)
        if intra_op_threads is not None:
            torch.set_num_threads(intra_op_threads)
        # Set environment variables for distributed initialization. Single
        # node rendezvous stays on loopback; cross-node rendezvous must use
        # an address every node can reach, so --dist-init-addr takes over.
        if self.server_args.nnodes > 1:
            rendezvous_addr = NetworkAddress.parse(self.server_args.dist_init_addr)
        else:
            rendezvous_addr = NetworkAddress("127.0.0.1", self.master_port)
        os.environ["MASTER_ADDR"] = rendezvous_addr.host
        os.environ["MASTER_PORT"] = str(rendezvous_addr.port)
        os.environ["LOCAL_RANK"] = str(self.local_rank)
        os.environ["RANK"] = str(self.rank)
        os.environ["WORLD_SIZE"] = str(self.server_args.num_gpus)
        self._configure_persistent_torch_compile_cache()
        # initialize the distributed environment
        maybe_init_distributed_environment_and_model_parallel(
            tp_size=self.server_args.tp_size,
            cfg_degree=self.server_args.cfg_parallel_degree or 1,
            ulysses_degree=self.server_args.ulysses_degree,
            ring_degree=self.server_args.ring_degree,
            sp_size=self.server_args.sp_degree,
            dp_size=self.server_args.dp_size,
            distributed_init_method=rendezvous_addr.to_tcp(),
            dist_timeout=self.server_args.dist_timeout,
        )

        from sglang.srt.runtime_context import get_context, publish
        from sglang.srt.server_args import ServerArgs as SrtServerArgs

        if get_context()._server_args is None:
            # srt reads the size from the configuration and the rank from the
            # live group, so the dummy carries the width just installed.
            publish(
                SrtServerArgs(model_path="dummy", tp_size=self.server_args.tp_size),
                role="diffusion_gpu_worker",
            )

        # set proc title
        if model_parallel_is_initialized():
            suffix = ""
            if get_tp_world_size() != 1:
                tp_rank = get_tp_rank()
                suffix += f"_TP{tp_rank}"
            if get_ulysses_parallel_world_size() != 1:
                u_rank = get_ulysses_parallel_rank()
                suffix += f"_U{u_rank}"
            if get_ring_parallel_world_size() != 1:
                r_rank = get_ring_parallel_rank()
                suffix += f"_R{r_rank}"
            if get_classifier_free_guidance_world_size() != 1:
                c_rank = get_classifier_free_guidance_rank()
                suffix += f"_C{c_rank}"
            setproctitle(f"sgl_diffusion::scheduler{suffix}")
        else:
            setproctitle(f"sgl_diffusion::scheduler_{self.local_rank}")

        self.pipeline = build_pipeline(self.server_args)

        # apply layerwise offload after lora is applied while building LoRAPipeline
        # otherwise empty offloaded weights could fail lora converting
        if self.server_args.has_layerwise_offload_components():
            configure_layerwise_offload_modules(
                self.pipeline.modules,
                self.server_args,
                component_names=(
                    None
                    if self.server_args.component_residency is not None
                    else self.server_args.layerwise_offload_components
                ),
                warn_missing=(
                    self.server_args.component_residency is not None
                    or self.server_args.is_arg_explicitly_set(
                        "layerwise_offload_components"
                    )
                    or self.server_args.is_arg_explicitly_set("dit_layerwise_offload")
                ),
            )

        logger.info(
            f"Worker {self.rank}: Initialized device, model, and distributed environment."
        )

    def do_mem_analysis(self, output_batch: OutputBatch):
        final_snapshot = capture_memory_snapshot()
        if output_batch.metrics:
            output_batch.metrics.record_memory_snapshot("mem_analysis", final_snapshot)

        peak_reserved_bytes = final_snapshot.peak_reserved_mb * (1024**2)
        peak_allocated_bytes = final_snapshot.peak_allocated_mb * (1024**2)

        output_batch.peak_memory_mb = peak_reserved_bytes / (1024**2)
        peak_reserved_gb = peak_reserved_bytes / (1024**3)
        peak_allocated_gb = peak_allocated_bytes / (1024**3)

        # Use the same process-visible capacity as auto residency. Physical
        # card memory can be larger than this worker's allocator cap or can be
        # partly occupied by another process; either case would make a hint
        # based on total device memory unsafe.
        remaining_gpu_mem_gb = max(
            0.0,
            self._auto_residency_budget_bytes() / GIB_BYTES - peak_reserved_gb,
        )
        try:
            can_stay_resident = self.get_can_stay_resident_components(
                remaining_gpu_mem_gb
            )
        except Exception:
            # a debug-only hint must never fail a completed request
            logger.debug("Residency hint unavailable", exc_info=True)
            can_stay_resident = []

        pool_overhead_gb = peak_reserved_gb - peak_allocated_gb
        pool_overhead_pct = (
            pool_overhead_gb / peak_reserved_gb * 100 if peak_reserved_gb else 0.0
        )

        residency_hint = (
            f" Components that can remain on GPU: {can_stay_resident}. "
            "Make it explicit with --component-residency <name>=resident; "
            "--performance-mode auto with server warmup applies safe "
            "adjustments automatically."
            if can_stay_resident
            else ""
        )
        logger.debug(
            "GPU memory: peak=%.2f GB, allocated=%.2f GB, pool=%.2f GB (%.1f%%), "
            "headroom=%.2f GB.%s",
            peak_reserved_gb,
            peak_allocated_gb,
            pool_overhead_gb,
            pool_overhead_pct,
            remaining_gpu_mem_gb,
            residency_hint,
        )

    def execute_forward(
        self, batch: List[Req], return_req: bool = False
    ) -> OutputBatch | Req:
        """
        Execute a forward pass.

        Args:
            batch: List of requests to process.
            return_req: If True, return the raw Req instead of OutputBatch.
                Used by disaggregated pipelines to access intermediate tensors.
        """
        assert self.pipeline is not None
        # request boundary: the IPC watchdog flag is a device read, illegal
        # inside a graph capture and too costly per exchange
        IPC_A2A.check_timeout()
        if len(batch) > 1:
            if return_req:
                raise ValueError(
                    "Grouped execute_forward does not support return_req=True"
                )
            # grouped reqs currently come only from expanded num_outputs_per_prompt
            self._validate_group_forward_reqs(batch)
            return self._execute_forward_batch(batch)

        req = batch[0]
        if req.is_warmup and req.extra.get("auto_residency_full_shape_probe"):
            self._fit_auto_residency_probe(req)
        return self._execute_forward_common(
            req,
            forward_fn=lambda: self.pipeline.forward(req, self.server_args),
            log_reqs=[req],
            return_req=return_req,
            save_output_paths=lambda output_batch: self._save_output_paths(
                req, output_batch
            ),
            error_context=f"request {req.request_id}",
        )

    def execute_forward_sequentially(self, batch: list[Req]) -> Iterator[OutputBatch]:
        """Yield grouped results after each request finishes its terminal stage."""
        assert self.pipeline is not None
        results = self.pipeline.forward_batch_sequentially(batch, self.server_args)
        group_start_time = time.monotonic()

        try:
            for req in batch:
                output_count = (
                    max(1, int(req.num_outputs_per_prompt or 1))
                    if self.server_args.pipeline_config.supports_sequential_multi_output_inference()
                    else 1
                )
                output_batch = self._execute_forward_common(
                    req,
                    forward_fn=lambda results=results, output_count=output_count: (
                        self._collect_sequential_outputs(results, output_count)
                    ),
                    log_reqs=[req],
                    return_req=False,
                    save_output_paths=lambda output_batch, req=req: (
                        self._save_output_paths(req, output_batch)
                    ),
                    error_context=f"grouped request {req.request_id}",
                    execution_start_time=group_start_time,
                    propagate_forward_errors=True,
                )
                assert isinstance(output_batch, OutputBatch)
                yield output_batch
                del output_batch
        finally:
            close = getattr(results, "close", None)
            if close is not None:
                close()

    def _collect_sequential_outputs(
        self,
        results: Iterator[OutputBatch | Req],
        output_count: int,
    ) -> OutputBatch | Req:
        if output_count == 1:
            return next(results)

        output_batches = [
            self._to_output_batch(next(results)) for _ in range(output_count)
        ]
        return self._merge_expanded_output_batches(output_batches)

    def _execute_forward_batch(self, batch: list[Req]) -> OutputBatch | Req:
        """Execute expanded multi-output requests as one grouped forward."""
        # TODO: support early return or mix-stage execution for reqs in a group
        assert self.pipeline is not None
        req = batch[0]
        return self._execute_forward_common(
            req,
            forward_fn=lambda: self._forward_group(batch),
            log_reqs=batch,
            return_req=False,
            save_output_paths=lambda output_batch: self._save_group_output_paths(
                batch, output_batch
            ),
            error_context=f"grouped request {req.request_id}",
        )

    def _execute_forward_common(
        self,
        req: Req,
        *,
        forward_fn: Callable[[], Req | OutputBatch],
        log_reqs: list[Req],
        return_req: bool,
        save_output_paths: Callable[[OutputBatch], None],
        error_context: str,
        execution_start_time: float | None = None,
        propagate_forward_errors: bool = False,
    ) -> OutputBatch | Req:
        """
        Args:
            forward_fn: the actual forward function for reqs
        """
        output_batch = None
        forward_failed = False
        # Prewarm reqs (is_warmup=False) run a different offload layout and
        # must not contaminate the calibration records. Pipelines that cannot
        # apply a residency plan also skip the temporary per-layer hooks.
        measure_server_warmup = (
            req.is_warmup
            and bool(req.extra.get("server_based_warmup"))
            and self.server_args.pipeline_config.supports_auto_residency
            and current_platform.is_cuda()
        )
        warmup_workload = (
            (
                int(req.width or 0),
                int(req.height or 0),
                int(req.num_frames or 1),
                max(1, int(req.num_inference_steps or 1)),
            )
            if measure_server_warmup
            else None
        )
        warmup_baseline_allocated_bytes = 0
        warmup_baseline_reserved_bytes = 0
        layerwise_usage_tracker: LayerwiseUsageTracker | None = None
        layerwise_layer_uses_by_stage: dict[
            str, dict[str, dict[str, tuple[int, ...]]]
        ] = {}
        try:
            if measure_server_warmup:
                # Drop the previous request's allocator pool so each probe
                # starts from the same placement and can return released
                # component storage before its allocated peak is measured.
                torch.get_device_module().empty_cache()
            self._release_warmup_pool(req)
            if not current_platform.is_cpu() and not current_platform.is_mps():
                torch.get_device_module().reset_peak_memory_stats()
            if measure_server_warmup:
                warmup_baseline_allocated_bytes = (
                    torch.get_device_module().memory_allocated()
                )
                warmup_baseline_reserved_bytes = (
                    torch.get_device_module().memory_reserved()
                )
                if (
                    self.server_args.performance_mode == "auto"
                    and self.pipeline is not None
                ):
                    layerwise_usage_tracker = LayerwiseUsageTracker(
                        self.pipeline.modules,
                        stage_name_provider=(
                            lambda: (
                                req.metrics.active_stage_name
                                if req.metrics is not None
                                else None
                            )
                        ),
                    )

            start_time = (
                execution_start_time
                if execution_start_time is not None
                else time.monotonic()
            )
            self._realtime_sessions.attach(req)

            # capture memory baseline for each req in grouped forward on rank-0
            request_metrics = [
                item.metrics for item in log_reqs if item.metrics is not None
            ]
            if (
                self.is_output_rank
                and request_metrics
                and not current_platform.is_cpu()
            ):
                baseline_snapshot = capture_memory_snapshot()
                for metrics in request_metrics:
                    metrics.record_memory_snapshot("before_forward", baseline_snapshot)

            for item in log_reqs:
                item.log(server_args=self.server_args)
            with ExitStack() as stack:
                for item in log_reqs:
                    stack.enter_context(
                        trace_slice(item.trace_ctx, DiffStage.GPU_FORWARD)
                    )
                try:
                    result = forward_fn()
                except Exception:
                    forward_failed = True
                    raise

            # disagg roles return raw Req so callers can keep and transfer intermediate tensors
            # before converting it to OutputBatch
            if return_req and isinstance(result, Req):
                return result

            output_batch = self._to_output_batch(result)

            output_metrics = self._iter_output_metrics(output_batch)
            if self.is_output_rank and output_metrics and not current_platform.is_cpu():
                peak_snapshot = capture_memory_snapshot()
                for metrics in output_metrics:
                    metrics.record_memory_snapshot("after_forward", peak_snapshot)

            duration_ms = (time.monotonic() - start_time) * 1000
            for metrics in output_metrics:
                metrics.total_duration_ms = duration_ms

            req_label = req.request_id[:8] if req.request_id else "unnamed"
            with maybe_record_function(f"SAVE_OUTPUTS {req_label}"):
                self._materialize_output_transport(output_batch, req, save_output_paths)
            self._record_output_peak_memory(output_batch, is_warmup=req.is_warmup)

            collect_perf = (
                req.perf_dump_path is not None or envs.SGLANG_DIFFUSION_STAGE_LOGGING
            )
            if collect_perf and not req.is_warmup:
                self._record_replica_peak_memory(output_metrics)

            if (
                self.is_output_rank
                and not req.suppress_logs
                and not current_platform.is_cpu()
                and logger.isEnabledFor(logging.DEBUG)
            ):
                self.do_mem_analysis(output_batch)

            if (
                not current_platform.is_cpu()
                and output_batch.output is None
                and not req.return_raw_frames
            ):
                with maybe_record_function("EMPTY_CACHE"):
                    torch.get_device_module().empty_cache()

            if req.perf_dump_path is not None or envs.SGLANG_DIFFUSION_STAGE_LOGGING:
                if not req.is_warmup:
                    PerformanceLogger.log_request_summary(metrics=output_batch.metrics)

            # dump per-request perf report to the server-mode file path.
            if (
                req.perf_dump_path is not None
                and not req.is_warmup
                and output_batch.metrics is not None
            ):
                PerformanceLogger.dump_benchmark_report(
                    file_path=req.perf_dump_path,
                    metrics=output_batch.metrics,
                    meta={"model": self.server_args.model_path},
                    tag="server_perf_dump",
                )
        except Exception as e:
            if propagate_forward_errors and forward_failed:
                if isinstance(e, StopIteration):
                    raise RuntimeError(
                        "Grouped pipeline returned fewer outputs than requests."
                    ) from e
                raise
            logger.error(
                f"Error executing {error_context}: {e}",
                exc_info=True,
            )
            if isinstance(e, _oom_exceptions()):
                logger.warning(OOM_MSG)
            if output_batch is None:
                output_batch = OutputBatch()
            output_batch.error = f"Error executing {error_context}: {e}"
            self._record_output_peak_memory(output_batch, is_warmup=req.is_warmup)
            # clean cache if OOM
            if not current_platform.is_cpu():
                torch.get_device_module().empty_cache()
        finally:
            # also runs on the propagate_forward_errors re-raise: a warmup
            # forward that never completed must still leave a failed record,
            # or the estimator would plan from the remaining partial data
            if measure_server_warmup:
                assert warmup_workload is not None
                if layerwise_usage_tracker is not None:
                    (
                        layerwise_layer_uses,
                        layerwise_layer_uses_by_stage,
                    ) = layerwise_usage_tracker.finish_with_stages()
                else:
                    layerwise_layer_uses = {}
                self._record_server_warmup_memory(
                    req=req,
                    workload=warmup_workload,
                    baseline_allocated_bytes=warmup_baseline_allocated_bytes,
                    baseline_reserved_bytes=warmup_baseline_reserved_bytes,
                    succeeded=output_batch is not None and output_batch.error is None,
                    layerwise_layer_uses=layerwise_layer_uses,
                    layerwise_layer_uses_by_stage=layerwise_layer_uses_by_stage,
                )
        return output_batch

    def _record_server_warmup_memory(
        self,
        *,
        req: Req,
        workload: tuple[int, int, int, int],
        baseline_allocated_bytes: int,
        baseline_reserved_bytes: int,
        succeeded: bool,
        layerwise_layer_uses: dict[str, dict[str, tuple[int, ...]]] | None = None,
        layerwise_layer_uses_by_stage: (
            dict[str, dict[str, dict[str, tuple[int, ...]]]] | None
        ) = None,
    ) -> None:
        phase_allocated_peaks: dict[str, int] = {}
        phase_components: dict[str, tuple[str, ...]] = {}
        phase_used_components: dict[str, tuple[str, ...]] = {}
        phase_prefetched_components: dict[str, tuple[str, ...]] = {}
        phase_full_weight_transition_components: dict[str, tuple[str, ...]] = {}
        untracked_active_components: tuple[str, ...] = ()
        residency_manager = peek_global_component_residency_manager()
        if residency_manager is not None:
            for phase_name, peak in residency_manager.take_warmup_phase_peaks().items():
                phase_allocated_peaks[phase_name] = peak.allocated_bytes
                phase_components[phase_name] = peak.active_components
                phase_used_components[phase_name] = peak.used_components
                phase_prefetched_components[phase_name] = peak.prefetched_components
                phase_full_weight_transition_components[phase_name] = (
                    peak.full_weight_transition_components
                )
            untracked_active_components = residency_manager.current_device_components()
        request_allocated_peak = max(
            int(torch.get_device_module().max_memory_allocated()),
            max(phase_allocated_peaks.values(), default=0),
        )
        if request_allocated_peak > max(phase_allocated_peaks.values(), default=0):
            # Work after the residency-managed stage timeline (for example,
            # output materialization) must remain a placement constraint.
            # A reserved-only increase is not a second live placement. Record
            # it separately so post-placement validation can still require
            # allocator headroom without charging cache to candidate deltas.
            phase_allocated_peaks["request:untracked"] = request_allocated_peak
            phase_components["request:untracked"] = untracked_active_components
            phase_used_components["request:untracked"] = ()
            phase_prefetched_components["request:untracked"] = ()
            phase_full_weight_transition_components["request:untracked"] = ()
        metrics = req.metrics
        stage_duration_ms = {}
        if metrics is not None:
            profile_to_component_stage = {
                stage._active_profile_stage_name(): stage._component_stage_name()
                for stage in self.pipeline.stages
            }
            stage_duration_ms = {
                profile_to_component_stage.get(stage_name, stage_name): duration_ms
                for stage_name, duration_ms in metrics.stages.items()
            }
        width, height, num_frames, num_inference_steps = workload
        self._auto_residency_warmup_records.append(
            WarmupMemoryRecord(
                width=width,
                height=height,
                num_frames=num_frames,
                baseline_allocated_bytes=int(baseline_allocated_bytes),
                peak_allocated_bytes=request_allocated_peak,
                succeeded=succeeded,
                baseline_reserved_bytes=int(baseline_reserved_bytes),
                peak_reserved_bytes=int(
                    torch.get_device_module().max_memory_reserved()
                ),
                phase_peak_allocated_bytes=phase_allocated_peaks,
                phase_active_components=phase_components,
                phase_used_components=phase_used_components,
                phase_prefetched_components=phase_prefetched_components,
                phase_full_weight_transition_components=(
                    phase_full_weight_transition_components
                ),
                layerwise_layer_uses=layerwise_layer_uses or {},
                layerwise_layer_uses_by_stage=layerwise_layer_uses_by_stage or {},
                num_inference_steps=num_inference_steps,
                total_duration_ms=(
                    float(metrics.total_duration_ms) if metrics is not None else 0.0
                ),
                stage_duration_ms=stage_duration_ms,
                step_duration_ms=(tuple(metrics.steps) if metrics is not None else ()),
                step_duration_ms_by_stage=(
                    {
                        stage_name: tuple(durations)
                        for stage_name, durations in metrics.steps_by_stage.items()
                    }
                    if metrics is not None
                    else {}
                ),
                stage_iterations=(
                    dict(metrics.stage_iterations) if metrics is not None else {}
                ),
            )
        )

    def _materialize_output_transport(
        self,
        output_batch: OutputBatch,
        req: Req,
        save_output_paths: Callable[[OutputBatch], None],
    ) -> None:
        if req.return_raw_frames:
            self._materialize_raw_frame_transport(output_batch, req)
        elif req.save_output and req.return_file_paths_only:
            self._materialize_file_path_transport(output_batch, save_output_paths)
        elif getattr(req, "return_frames", False):
            self._materialize_frame_outputs_for_return(output_batch, req)

    def _materialize_raw_frame_transport(
        self, output_batch: OutputBatch, req: Req
    ) -> None:
        if not self.is_output_rank:
            return
        if output_batch.output is not None:
            output_batch.raw_frame_content_type = RAW_RGB_CONTENT_TYPE
            (
                output_batch.raw_frame_batches,
                output_batch.raw_frame_metadata,
            ) = build_raw_rgb_frame_batches(
                output_batch.output,
                req,
                output_batch,
                post_process_sample,
            )
            output_batch.output = None
        output_batch.audio = None
        output_batch.audio_sample_rate = None

    def _materialize_file_path_transport(
        self,
        output_batch: OutputBatch,
        save_output_paths: Callable[[OutputBatch], None],
    ) -> None:
        if self.is_output_rank:
            save_output_paths(output_batch)
        output_batch.output = None
        output_batch.audio = None
        output_batch.audio_sample_rate = None

    def _materialize_frame_outputs_for_return(
        self, output_batch: OutputBatch, req: Req
    ) -> None:
        """materialize the output from tensor to numpy frames for faster serialization"""
        if (
            not self.is_output_rank
            or output_batch.output is None
            or not getattr(req, "return_frames", False)
        ):
            return

        if (
            os.environ.get("SGLANG_DIFFUSION_SYNC_STAGE_PROFILING", "0") == "1"
            and torch.cuda.is_initialized()
        ):
            torch.cuda.synchronize()
        start_time = time.perf_counter()
        output_batch.output = [
            self._materialize_frame_output(output, output_batch, req)
            for output in output_batch.output
        ]
        if output_batch.metrics is not None:
            if (
                os.environ.get("SGLANG_DIFFUSION_SYNC_STAGE_PROFILING", "0") == "1"
                and torch.cuda.is_initialized()
            ):
                torch.cuda.synchronize()
            output_batch.metrics.record_stage(
                "GPUWorker.frame_materialize_for_return",
                time.perf_counter() - start_time,
            )

    @staticmethod
    def _materialize_frame_output(
        output: Any, output_batch: OutputBatch, req: Req
    ) -> np.ndarray:
        if (
            isinstance(output, torch.Tensor)
            and not req.enable_frame_interpolation
            and not req.enable_upscaling
        ):
            if output.dim() == 3:
                output = output.unsqueeze(1)
            output = (output * 255).clamp(0, 255).to(torch.uint8)
            return output.permute(1, 2, 3, 0).cpu().numpy()

        if (
            isinstance(output, np.ndarray)
            and output.dtype == np.uint8
            and output.ndim == 4
            and output.shape[-1] in (1, 3, 4)
        ):
            return output

        materialized = materialize_output_sample(
            output,
            req.data_type,
            req.fps,
            enable_frame_interpolation=req.enable_frame_interpolation,
            frame_interpolation_exp=req.frame_interpolation_exp,
            frame_interpolation_scale=req.frame_interpolation_scale,
            frame_interpolation_model_path=req.frame_interpolation_model_path,
            enable_upscaling=req.enable_upscaling,
            upscaling_model_path=req.upscaling_model_path,
            upscaling_scale=req.upscaling_scale,
        )
        return np.asarray(materialized.frames)

    def _fit_auto_residency_probe(self, req: Req) -> None:
        """Size the full-shape probe to what the card has left, on every rank alike."""
        records = [r for r in self._auto_residency_warmup_records if r.succeeded]
        if not records or not current_platform.is_cuda():
            return
        device = current_platform.get_device(self.local_rank)
        free_bytes = int(
            current_platform.get_available_gpu_memory(empty_cache=True) * (1 << 30)
        )
        total_bytes = int(torch.cuda.get_device_properties(device).total_memory)
        _, _, steps = fit_auto_residency_probe(
            req,
            records=records,
            free_bytes=free_bytes,
            total_bytes=total_bytes,
            server_args=self.server_args,
        )
        requested_units = (
            max(1, int(req.width or 1))
            * max(1, int(req.height or 1))
            * max(1, int(req.num_frames or 1))
        )
        estimate = estimate_default_workload_peak_bytes(
            records=records, target_units=requested_units
        )
        # Ranks see different free memory and hold different records; the
        # forward must run one shape everywhere, so the most cautious rank wins.
        agreed = torch.tensor([steps], dtype=torch.int64, device=device)
        agreed = get_replica_group().all_reduce(
            agreed, op=torch.distributed.ReduceOp.MAX
        )
        steps = int(agreed.item())
        if steps == 0:
            return
        fitted = req
        for _ in range(steps):
            lighter = lighten_warmup_req(self.server_args, fitted)
            if lighter is None:
                break
            fitted = lighter
        if self.is_output_rank:
            logger.warning(
                "Auto residency probe %s would not fit: extrapolated peak %.1f GiB "
                "against %.1f GiB free; probing at %s instead",
                _shape_label(req),
                (estimate or 0) / (1 << 30),
                free_bytes / (1 << 30),
                _shape_label(fitted),
            )
        req.sampling_params = fitted.sampling_params

    def _release_warmup_pool(self, req: Req) -> None:
        """Drop the allocator pool the full-shape probe left behind.

        The probe runs a shape serving never sees; its cached blocks would
        otherwise become the floor of every runtime peak measurement. The
        request after the probe (the bounded re-warm) regrows the pool at a
        serving-sized shape.
        """
        if req.is_warmup and req.extra.get("auto_residency_full_shape_probe"):
            self._release_warmup_pool_before_serving = True
            return
        if not self._release_warmup_pool_before_serving:
            return
        self._release_warmup_pool_before_serving = False
        if current_platform.is_cpu() or current_platform.is_mps():
            return
        torch.get_device_module().empty_cache()

    def _record_output_peak_memory(
        self, output_batch: OutputBatch, *, is_warmup: bool = False
    ) -> None:
        if current_platform.is_cpu():
            return
        peak_reserved_mb = capture_memory_snapshot().peak_reserved_mb
        if is_warmup:
            self._warmup_peak_reserved_mb = max(
                self._warmup_peak_reserved_mb, peak_reserved_mb
            )
        else:
            self._runtime_peak_reserved_mb = max(
                self._runtime_peak_reserved_mb, peak_reserved_mb
            )
        if self.is_output_rank:
            output_batch.peak_memory_mb = peak_reserved_mb

    def _record_replica_peak_memory(self, output_metrics: list[Any]) -> None:
        """Record replica-wide loading and runtime allocator peaks."""
        if not current_platform.is_cuda():
            return

        peaks = torch.tensor(
            [
                self._load_peak_reserved_mb,
                self._runtime_peak_reserved_mb,
                self._warmup_peak_reserved_mb,
            ],
            dtype=torch.float64,
            device=current_platform.get_device(self.local_rank),
        )
        peaks = get_replica_group().all_reduce(peaks, op=torch.distributed.ReduceOp.MAX)
        if not self.is_output_rank:
            return

        snapshot = capture_memory_snapshot()
        load_peak_mb, runtime_peak_mb, warmup_peak_mb = peaks.tolist()
        for metrics in output_metrics:
            metrics.record_memory_snapshot(
                "load_peak", replace(snapshot, peak_reserved_mb=load_peak_mb)
            )
            metrics.record_memory_snapshot(
                "runtime_peak", replace(snapshot, peak_reserved_mb=runtime_peak_mb)
            )
            metrics.record_memory_snapshot(
                "warmup_peak", replace(snapshot, peak_reserved_mb=warmup_peak_mb)
            )

    def _forward_group(self, batch: list[Req]) -> OutputBatch:
        assert self.pipeline is not None
        results = self.pipeline.forward_batch(batch, self.server_args)
        output_batches = [self._to_output_batch(result) for result in results]
        return self._merge_expanded_output_batches(output_batches)

    def _save_output_paths(self, req: Req, output_batch: OutputBatch) -> None:
        """save outputs to files"""
        if not self.is_output_rank or output_batch.output is None:
            return

        dynamic_output_paths = None
        if req.extra:
            dynamic_output_paths = req.extra.get("dynamic_batch_output_paths")
        if dynamic_output_paths is not None and (
            len(dynamic_output_paths) != len(output_batch.output)
        ):
            logger.warning(
                "dynamic_batch_output_paths length mismatch (got=%d, expected=%d). "
                "Falling back to merged request output file naming.",
                len(dynamic_output_paths),
                len(output_batch.output),
            )
            dynamic_output_paths = None

        if dynamic_output_paths is not None:

            def build_output_path(idx: int) -> str:
                return dynamic_output_paths[idx]

        else:
            num_outputs = len(output_batch.output)

            def build_output_path(idx: int) -> str:
                return req.output_file_path(num_outputs, idx)

        output_batch.output_file_paths = save_outputs(
            output_batch.output,
            req.data_type,
            req.fps,
            True,
            build_output_path,
            audio=output_batch.audio,
            audio_sample_rate=output_batch.audio_sample_rate,
            output_compression=req.output_compression,
            enable_frame_interpolation=req.enable_frame_interpolation,
            frame_interpolation_exp=req.frame_interpolation_exp,
            frame_interpolation_scale=req.frame_interpolation_scale,
            frame_interpolation_model_path=req.frame_interpolation_model_path,
            enable_upscaling=req.enable_upscaling,
            upscaling_model_path=req.upscaling_model_path,
            upscaling_scale=req.upscaling_scale,
        )

    def _save_group_output_paths(
        self,
        reqs: list[Req],
        output_batch: OutputBatch,
    ) -> None:
        if not self.is_output_rank or output_batch.output is None:
            return
        if len(output_batch.output) != len(reqs):
            raise RuntimeError(
                f"Expected {len(reqs)} grouped outputs, got {len(output_batch.output)}"
            )

        first_req = reqs[0]
        output_batch.output_file_paths = save_outputs(
            output_batch.output,
            first_req.data_type,
            first_req.fps,
            True,
            lambda idx: reqs[idx].output_file_path(1, 0),
            audio=output_batch.audio,
            audio_sample_rate=output_batch.audio_sample_rate,
            output_compression=first_req.output_compression,
            enable_frame_interpolation=first_req.enable_frame_interpolation,
            frame_interpolation_exp=first_req.frame_interpolation_exp,
            frame_interpolation_scale=first_req.frame_interpolation_scale,
            frame_interpolation_model_path=first_req.frame_interpolation_model_path,
            enable_upscaling=first_req.enable_upscaling,
            upscaling_model_path=first_req.upscaling_model_path,
            upscaling_scale=first_req.upscaling_scale,
        )

    @staticmethod
    def _validate_group_forward_reqs(reqs: list[Req]) -> None:
        """Validate fields that the grouped output/save path treats as shared."""
        first_req = reqs[0]
        shared_output_fields = (
            "save_output",
            "return_frames",
            "return_file_paths_only",
            "data_type",
            "fps",
            "output_compression",
            "enable_frame_interpolation",
            "frame_interpolation_exp",
            "frame_interpolation_scale",
            "frame_interpolation_model_path",
            "enable_upscaling",
            "upscaling_model_path",
            "upscaling_scale",
        )
        for req in reqs[1:]:
            mismatched = [
                field
                for field in shared_output_fields
                if getattr(req, field, None) != getattr(first_req, field, None)
            ]
            if mismatched:
                raise ValueError(
                    "Grouped execute_forward requires matching output settings; "
                    f"mismatched fields: {mismatched}"
                )

    @staticmethod
    def _iter_output_metrics(output_batch: OutputBatch):
        """Return all metrics objects carried by an output batch."""
        if output_batch.metrics_list is not None:
            return [
                metrics for metrics in output_batch.metrics_list if metrics is not None
            ]
        if output_batch.metrics is not None:
            return [output_batch.metrics]
        return []

    @staticmethod
    def _to_output_batch(result: Req | OutputBatch) -> OutputBatch:
        if isinstance(result, Req):
            return GPUWorker._req_to_output_batch(result)
        return result

    @staticmethod
    def _req_to_output_batch(result: Req) -> OutputBatch:
        return OutputBatch(
            output=result.output,
            audio=getattr(result, "audio", None),
            audio_sample_rate=getattr(result, "audio_sample_rate", None),
            metrics=result.metrics,
            usage=getattr(result, "usage", None),
            trajectory_timesteps=getattr(result, "trajectory_timesteps", None),
            trajectory_latents=getattr(result, "trajectory_latents", None),
            rollout_trajectory_data=getattr(result, "rollout_trajectory_data", None),
            noise_pred=getattr(result, "noise_pred", None),
            trajectory_decoded=getattr(result, "trajectory_decoded", None),
        )

    @staticmethod
    def _merge_expanded_output_batches(
        output_batches: list[OutputBatch],
    ) -> OutputBatch:
        """Merge per-output batches produced by grouped execution."""
        merged = OutputBatch()
        parts = _ExpandedOutputParts()

        for output_batch in output_batches:
            GPUWorker._merge_expanded_singletons(merged, output_batch)
            GPUWorker._collect_expanded_parts(parts, output_batch)

        GPUWorker._finalize_expanded_parts(
            merged,
            parts,
            audio_sample_rate=output_batches[0].audio_sample_rate,
        )

        return merged

    @staticmethod
    def _merge_expanded_singletons(
        merged: OutputBatch, output_batch: OutputBatch
    ) -> None:
        if output_batch.error is not None and merged.error is None:
            merged.error = output_batch.error
        merged.peak_memory_mb = max(merged.peak_memory_mb, output_batch.peak_memory_mb)
        if output_batch.usage is not None:
            if merged.usage is None:
                merged.usage = {}
            for key, value in output_batch.usage.items():
                if isinstance(value, int):
                    merged.usage[key] = int(merged.usage.get(key, 0)) + value
                else:
                    merged.usage[key] = value
        if (
            merged.trajectory_timesteps is None
            and output_batch.trajectory_timesteps is not None
        ):
            merged.trajectory_timesteps = output_batch.trajectory_timesteps
        if (
            merged.rollout_trajectory_data is None
            and output_batch.rollout_trajectory_data is not None
        ):
            merged.rollout_trajectory_data = output_batch.rollout_trajectory_data

    @staticmethod
    def _collect_expanded_parts(
        parts: _ExpandedOutputParts, output_batch: OutputBatch
    ) -> None:
        """Collect expanded outputs"""
        parts.metrics_list.append(output_batch.metrics)
        if output_batch.output_file_paths:
            parts.output_file_paths.extend(output_batch.output_file_paths)
        if isinstance(output_batch.output, torch.Tensor):
            parts.tensor_outputs.append(output_batch.output)
        elif output_batch.output is not None:
            parts.list_outputs.extend(output_batch.output)
        if isinstance(output_batch.audio, torch.Tensor):
            parts.tensor_audio.append(output_batch.audio)
        if isinstance(output_batch.trajectory_latents, torch.Tensor):
            parts.trajectory_latents.append(output_batch.trajectory_latents)
        if isinstance(output_batch.noise_pred, torch.Tensor):
            parts.noise_preds.append(output_batch.noise_pred)
        if output_batch.trajectory_decoded:
            GPUWorker._collect_trajectory_decoded(
                parts, output_batch.trajectory_decoded
            )

    @staticmethod
    def _collect_trajectory_decoded(
        parts: _ExpandedOutputParts, trajectory_decoded: list[torch.Tensor]
    ) -> None:
        if parts.trajectory_decoded_parts is None:
            parts.trajectory_decoded_parts = [[] for _ in trajectory_decoded]
        for index, decoded in enumerate(trajectory_decoded):
            parts.trajectory_decoded_parts[index].append(decoded)

    @staticmethod
    def _finalize_expanded_parts(
        merged: OutputBatch,
        parts: _ExpandedOutputParts,
        *,
        audio_sample_rate: int | None,
    ) -> None:
        """
        merge batched output
        """
        if parts.output_file_paths:
            merged.output_file_paths = parts.output_file_paths
        if any(metrics is not None for metrics in parts.metrics_list):
            merged.metrics_list = parts.metrics_list
            merged.metrics = next(
                metrics for metrics in parts.metrics_list if metrics is not None
            )
        if parts.tensor_outputs:
            merged.output = torch.cat(parts.tensor_outputs, dim=0)
        elif parts.list_outputs:
            merged.output = parts.list_outputs
        if parts.tensor_audio:
            merged.audio = torch.cat(parts.tensor_audio, dim=0)
            merged.audio_sample_rate = audio_sample_rate
        if parts.trajectory_latents:
            merged.trajectory_latents = torch.cat(parts.trajectory_latents, dim=0)
        if parts.noise_preds:
            merged.noise_pred = torch.cat(parts.noise_preds, dim=0)
        if parts.trajectory_decoded_parts:
            merged.trajectory_decoded = [
                torch.cat(decoded_step, dim=0)
                for decoded_step in parts.trajectory_decoded_parts
            ]

    def get_can_stay_resident_components(
        self, remaining_gpu_mem_gb: float
    ) -> List[str]:
        """Which currently offloaded components would fit in the headroom.

        Reuses the auto-residency candidate frontier and benefit ranking, but
        remains a raw-capacity hint rather than the phase-constrained joint
        plan. It omits components driven by fixed pipeline-custom residency
        strategies and applies no reserve or activation margin.
        """
        if not self.pipeline:
            return []

        modules = self._auto_residency_modules()
        if not modules:
            return []
        workload = self._default_workload_for_hint()
        candidates = collect_residency_targets(
            modules=modules,
            residency_mode_of=self.server_args.residency_mode,
            # The hint also covers explicitly offloaded components: the user
            # chose offload and should learn when the headroom no longer
            # requires it (automatic adjustment never touches explicit ones).
            explicit_residency_mode_of=lambda _name: None,
            custom_strategy_names=self._fixed_custom_residency_strategy_names(),
            num_inference_steps=(
                workload.num_inference_steps if workload is not None else 1
            ),
            # this hint only consumes permanent-residency targets; HostPin
            # alternatives cannot change its answer
            allow_host_pin_reallocation=False,
            mixed_dtype_components=self._mixed_dtype_residency_components(),
        )

        can_stay_resident = []
        seen_components: set[str] = set()
        for candidate in rank_candidates_by_h2d_savings(candidates):
            if (
                not candidate.permanent_residency
                or candidate.component_name in seen_components
            ):
                continue
            seen_components.add(candidate.component_name)
            usage_gb = candidate.target_resident_weight_bytes / GIB_BYTES
            if usage_gb <= remaining_gpu_mem_gb:
                can_stay_resident.append(candidate.component_name)
                remaining_gpu_mem_gb -= usage_gb
        return can_stay_resident

    def apply_auto_residency(
        self, *, pre_warmup: bool = False, validate_only: bool = False
    ) -> OutputBatch:
        """Apply one warmup-calibrated residency adjustment round.

        Every rank executes this handler at the same queue position; the plan
        is computed from all-gathered rank reports so each rank reaches the
        same decision. After applying a plan, the server calls this once in
        validation-only mode with measurements from the new layout. Validation
        may keep or roll back that plan, but never starts another placement
        search. A failure on any rank rolls every rank back to the previously
        calibrated placement.

        Everything before the first all-gather is fenced into a skip report:
        an uncaught raise there would leave the peer ranks parked in the
        collective until the group timeout.
        """
        if pre_warmup and validate_only:
            raise ValueError("static placement cannot be validation-only")
        records = list(self._auto_residency_warmup_records)
        if self.is_output_rank:
            # The regression check compares request durations extrapolated from
            # these few steps, so a rollback is only explainable with them.
            for record in records:
                logger.info(
                    "Auto residency calibration record: %dx%dx%df steps=%d ok=%s "
                    "total=%.1fs stages=%s steps_ms=%s iterations=%s",
                    record.width,
                    record.height,
                    record.num_frames,
                    record.num_inference_steps,
                    record.succeeded,
                    record.total_duration_ms / 1000.0,
                    {
                        name: round(ms / 1000.0, 2)
                        for name, ms in record.stage_duration_ms.items()
                        if ms >= 100.0
                    },
                    [round(ms) for ms in record.step_duration_ms],
                    record.stage_iterations,
                )
        # Each round must describe one placement. Intersecting phase ownership
        # across old and newly adjusted layouts would double-count weights that
        # are already resident in the new layout.
        self._auto_residency_warmup_records.clear()

        try:
            workload = resolve_default_workload(self.server_args)
            workload = resolve_measured_default_workload(workload, records)
            local_report = (
                self._build_pre_warmup_auto_residency_report(workload=workload)
                if pre_warmup
                else self._build_auto_residency_report(
                    workload=workload,
                    records=records,
                    include_candidates=not validate_only,
                )
            )
        except Exception as e:
            logger.warning(
                "Auto residency: rank %d could not build its report: %s",
                self.rank,
                e,
                exc_info=True,
            )
            workload = DefaultWorkload(
                width=None, height=None, num_frames=1, num_inference_steps=1
            )
            local_report = RankResidencyReport(
                rank=self.rank,
                budget_bytes=0,
                estimated_peak_bytes=None,
                candidates=[],
                skip_reason=describe_error(e),
            )
        reports = self._auto_residency_all_gather(local_report)
        recovering_from_oom = any(report.warmup_oom for report in reports)
        if validate_only:
            invalid_report = next(
                (
                    report
                    for report in reports
                    if report.skip_reason is not None
                    or report.estimated_peak_bytes is None
                ),
                None,
            )
            if invalid_report is not None:
                reason = invalid_report.skip_reason or "no usable warmup measurement"
                return self._rollback_everywhere(
                    cause=(
                        "post-adjustment calibration could not validate rank "
                        f"{invalid_report.rank}: {reason}"
                    ),
                    already_failed=False,
                    latest_round_only=True,
                )
        if (
            not pre_warmup
            and self._auto_residency_round_sizes
            and not recovering_from_oom
            and not self._latest_auto_residency_round_is_resident_only()
        ):
            regressions = []
            for report in reports:
                reference_ns = report.reference_probe_duration_ns
                measured_ns = report.probe_duration_ns
                tolerance_ns = max(
                    MIN_POST_ADJUSTMENT_REGRESSION_NS,
                    int(reference_ns * POST_ADJUSTMENT_REGRESSION_FRACTION),
                )
                if reference_ns > 0 and measured_ns > reference_ns + tolerance_ns:
                    regressions.append((measured_ns - reference_ns, report))
            if regressions:
                _, regressed = max(regressions, key=lambda item: item[0])
                cause = (
                    "post-adjustment calibration regressed the probe's duration "
                    f"from {regressed.reference_probe_duration_ns / 1e9:.2f}s "
                    f"to {regressed.probe_duration_ns / 1e9:.2f}s"
                )
                if self.is_output_rank:
                    logger.warning("Auto residency: %s; rolling back", cause)
                    logger.warning(
                        "Auto residency: %s", _format_calibration_timing(records)
                    )
                return self._rollback_everywhere(
                    cause=cause,
                    already_failed=False,
                    latest_round_only=True,
                )
        if validate_only:
            shortfall_bytes = current_placement_reserve_shortfall_bytes(reports)
            if shortfall_bytes > 0:
                shortfall_gib = shortfall_bytes / GIB_BYTES
                if self.is_output_rank:
                    logger.warning(
                        "Auto residency calibration exceeded the VRAM reserve by "
                        "%.1f GiB; rolling back the latest adjustment round.",
                        shortfall_gib,
                    )
                return self._rollback_everywhere(
                    cause=f"VRAM reserve exceeded by {shortfall_gib:.1f} GiB",
                    already_failed=False,
                    latest_round_only=True,
                )
            plan = self._auto_residency_last_applied_plan
            if plan is None:
                return self._rollback_everywhere(
                    cause="post-adjustment validation lost the selected plan",
                    already_failed=False,
                    latest_round_only=True,
                )
            latest_round = self._latest_auto_residency_round()
            short_validation = (
                self._latest_auto_residency_round_supports_short_validation()
            )
            self._commit_auto_residency_changes(latest_round)
            self._auto_residency_applied = []
            self._auto_residency_round_sizes = []
            self._auto_residency_last_applied_plan = None
            if self.is_output_rank:
                reference_ns = local_report.estimated_request_duration_ns
                measured_ns = local_report.measured_request_duration_ns
                if short_validation:
                    result = "full-shape one-step memory check passed"
                elif reference_ns > 0 and measured_ns > 0:
                    change = (measured_ns - reference_ns) / reference_ns
                    result = (
                        "calibrated request estimate "
                        f"{reference_ns / 1e9:.2f}s -> {measured_ns / 1e9:.2f}s "
                        f"({change:+.1%})"
                    )
                else:
                    result = "memory and execution checks passed"
                logger.info(
                    "Auto residency: post-adjustment validation passed (%s); "
                    "keeping the selected placement.",
                    result,
                )
                if envs.SGLANG_DIFFUSION_DEBUG_HOST_MEMORY:
                    from sglang.multimodal_gen.runtime.managers.memory_managers.host_memory_breakdown import (
                        log_host_memory_breakdown,
                    )

                    log_host_memory_breakdown(
                        self._auto_residency_modules(), label="after auto residency"
                    )
            return OutputBatch(
                output=plan_summary_payload(
                    plan=plan, status=PLACEMENT_STATUS_VALIDATED
                )
            )
        plan = plan_auto_residency(reports=reports)
        summary = format_plan_summary(plan=plan, workload=workload, records=records)
        if self.is_output_rank and records:
            logger.info("Auto residency: %s", _format_calibration_timing(records))
        if plan.skip_reason is not None or not plan.changes:
            if self.is_output_rank:
                logger.info("%s", summary)
            return OutputBatch(
                output=plan_summary_payload(plan=plan, status=PLACEMENT_STATUS_SKIPPED)
            )

        apply_error: str | None = None
        local_rollback_failed = False
        # Keep round history aligned across ranks. A zero entry means this
        # rank's transactional apply failed before committing the round, so a
        # peer-triggered rollback must leave earlier calibrated rounds alone.
        self._auto_residency_round_sizes.append(0)
        try:
            newly_applied = apply_residency_changes(
                plan=plan,
                modules=self._auto_residency_modules(),
                server_args=self.server_args,
                rank=self.rank,
            )
            if not newly_applied:
                raise RuntimeError("placement plan applied no residency changes")
            self._auto_residency_applied.extend(newly_applied)
            self._auto_residency_round_sizes[-1] = len(newly_applied)
        except AutoResidencyRollbackError as e:
            logger.error(
                "Auto residency adjustment failed on rank %d and the rank "
                "could not roll itself back: %s",
                self.rank,
                e,
                exc_info=True,
            )
            apply_error = describe_error(e)
            local_rollback_failed = True
        except Exception as e:  # this rank already rolled itself back
            logger.error(
                "Auto residency adjustment failed on rank %d: %s",
                self.rank,
                e,
                exc_info=True,
            )
            apply_error = describe_error(e)

        gathered = self._auto_residency_all_gather((apply_error, local_rollback_failed))
        rank_errors = [error for error, _ in gathered if error is not None]
        any_rollback_failed = any(failed for _, failed in gathered)
        if rank_errors:
            return self._rollback_everywhere(
                cause=rank_errors[0],
                already_failed=any_rollback_failed,
                latest_round_only=True,
            )

        self._invalidate_component_strategies(
            [candidate.component_name for candidate in plan.changes]
        )
        if pre_warmup:
            self._commit_auto_residency_changes(self._latest_auto_residency_round())
            self._auto_residency_applied = []
            self._auto_residency_round_sizes = []
            self._auto_residency_last_applied_plan = None
        else:
            self._auto_residency_last_applied_plan = plan
        if self.is_output_rank:
            logger.info("%s", summary)
            logger.info("%s", format_applied_changes(plan=plan))
        return OutputBatch(
            output=plan_summary_payload(
                plan=plan,
                status=PLACEMENT_STATUS_ADJUSTED,
                short_validation=(
                    self._latest_auto_residency_round_supports_short_validation()
                ),
            )
        )

    def rollback_auto_residency(self) -> OutputBatch:
        """Undo the latest adjustment round after its calibration failed."""
        return self._rollback_everywhere(
            cause=None, already_failed=False, latest_round_only=True
        )

    def _commit_auto_residency_changes(
        self, changes: list[AppliedResidencyChange]
    ) -> None:
        modules = (
            self._auto_residency_modules()
            if any(
                change.previous_layerwise_resident_layers is not None
                for change in changes
            )
            else {}
        )
        commit_error = None
        try:
            commit_residency_changes(
                applied=changes,
                modules=modules,
                server_args=self.server_args,
            )
        except Exception as error:
            commit_error = describe_error(error)
        commit_errors = self._auto_residency_all_gather(commit_error)
        if any(error is not None for error in commit_errors):
            raise RuntimeError(
                "residency commit failed: "
                + next(error for error in commit_errors if error is not None)
            )

    def _rollback_everywhere(
        self,
        *,
        cause: str | None,
        already_failed: bool,
        latest_round_only: bool = False,
    ) -> OutputBatch:
        """Roll this rank back and gather the replica-wide outcome.

        ``cause`` is the adjustment failure that triggered the rollback (None
        when the rollback was requested after a failed re-warm);
        ``already_failed`` marks that some rank already failed its in-apply
        rollback, which is fatal regardless of what the remaining ranks do.
        Collective-symmetric: every rank gathers exactly once.
        """
        rollback_error = self._rollback_applied_residency_changes(
            latest_round_only=latest_round_only
        )
        gathered = self._auto_residency_all_gather(rollback_error)
        rank_errors = [error for error in gathered if error is not None]
        if already_failed and not rank_errors:
            rank_errors = ["a rank could not undo its own residency changes"]
        if rank_errors:
            prefix = (
                f"auto residency adjustment failed ({cause}) and rollback failed"
                if cause is not None
                else "auto residency rollback failed"
            )
            return OutputBatch(
                error=f"{prefix}: {rank_errors[0]}",
                output={"status": PLACEMENT_STATUS_ROLLBACK_FAILED},
            )
        if cause is not None:
            restored = (
                "previously calibrated placement"
                if self._auto_residency_applied
                else "original strategy"
            )
            return OutputBatch(
                error=(
                    f"auto residency adjustment failed; {restored} restored: {cause}"
                ),
                output={"status": PLACEMENT_STATUS_ROLLED_BACK},
            )
        if self.is_output_rank:
            if self._auto_residency_applied:
                logger.info(
                    "Auto residency: rolled back the latest adjustment round; "
                    "the previously calibrated placement remains active."
                )
            else:
                logger.info(
                    "Auto residency: rolled back; the residency configured at "
                    "startup is in effect again (no equivalent server-arg "
                    "changes remain)."
                )
        return OutputBatch(output={"status": PLACEMENT_STATUS_ROLLED_BACK})

    def _auto_residency_budget_bytes(self, *, streamed_mapped_bytes: int = 0) -> int:
        # Free VRAM plus this process's reserved allocator pool excludes memory
        # held by unrelated processes while preserving reusable local capacity.
        reserved_bytes = int(torch.get_device_module().memory_reserved())
        if current_platform.device_shares_host_memory():
            # One pool: the driver's free figure is the kernel's MemFree, which
            # leaves out the page cache -- memory the kernel hands back on
            # demand, so the plan may spend it -- except the part holding the
            # layers the current layout streams: evicting those turns every
            # denoise step into a disk read, so they stay committed.
            budget_bytes = (
                shared_pool_available_bytes() + reserved_bytes - streamed_mapped_bytes
            )
        else:
            free_bytes, _ = torch.get_device_module().mem_get_info()
            budget_bytes = int(free_bytes) + reserved_bytes
        test_cap_gib = envs.SGLANG_DIFFUSION_TEST_CAP_DEVICE_MEMORY_GIB
        if test_cap_gib is not None:
            budget_bytes = min(budget_bytes, int(test_cap_gib * GIB_BYTES))
        return budget_bytes

    def _collect_auto_residency_targets(
        self,
        workload: DefaultWorkload,
        *,
        allow_host_pin_reallocation: bool = True,
        used_components: set[str] | None = None,
        layerwise_layer_uses: dict[str, dict[str, tuple[int, ...]]] | None = None,
        host_pin_headroom_bytes: int | None = None,
        request_duration_ns: int = 0,
        latency_upper_bound_ns_by_component: dict[str, int] | None = None,
        estimated_peak_bytes: int | None = None,
    ) -> list[ResidencyTarget]:
        assert self.pipeline is not None
        modules = self._auto_residency_modules()
        mapped_stream_cost_multiplier = 0
        if current_platform.device_shares_host_memory():
            # The page cache the request cycle needs is every mapped byte the
            # layerwise components stream; what the pool can give it is what
            # is available minus the device growth still ahead of us. Once the
            # cycle does not fit, streaming a mapped layer costs a disk read
            # on every pass and is priced as such.
            device_growth = max(
                0,
                (estimated_peak_bytes or 0)
                - int(torch.get_device_module().memory_allocated()),
            )
            cache_capacity = shared_pool_available_bytes() - device_growth
            mapped_total = layerwise_mapped_bytes(modules)
            if mapped_total > cache_capacity:
                from sglang.multimodal_gen.runtime.managers.memory_managers.auto_residency import (
                    DISK_MISS_COST_MULTIPLIER,
                )

                mapped_stream_cost_multiplier = DISK_MISS_COST_MULTIPLIER
            logger.info(
                "Shared pool: layerwise components map %.1f GiB, the page cache "
                "can hold %.1f GiB (%.1f GiB available, %.1f GiB of device growth "
                "ahead); mapped streaming is priced at %dx.",
                mapped_total / GIB_BYTES,
                cache_capacity / GIB_BYTES,
                (cache_capacity + device_growth) / GIB_BYTES,
                device_growth / GIB_BYTES,
                mapped_stream_cost_multiplier,
            )
        local_worker_count = max(
            1, self.server_args.num_gpus // self.server_args.nnodes
        )
        return collect_residency_targets(
            modules=modules,
            residency_mode_of=self.server_args.residency_mode,
            baseline_residency_mode_of=(self.server_args.configured_residency_mode),
            explicit_residency_mode_of=self.server_args.explicit_residency_mode,
            custom_strategy_names=self._fixed_custom_residency_strategy_names(),
            num_inference_steps=workload.num_inference_steps,
            allow_host_pin_reallocation=allow_host_pin_reallocation,
            mixed_dtype_components=self._mixed_dtype_residency_components(),
            required_resident_components={
                name
                for name in modules
                if self.server_args.component_residency_requirement(name) is not None
            },
            layerwise_tuning_of=lambda name, dit_group: (
                self.server_args.layerwise_tuning_for(name, dit_group=dit_group)
            ),
            layerwise_policy_is_explicit=lambda name, dit_group: (
                self.server_args.is_layerwise_residency_policy_explicit(
                    name, dit_group=dit_group
                )
            ),
            pin_cpu_memory=self.server_args.pin_cpu_memory,
            used_components=used_components,
            layerwise_layer_uses=layerwise_layer_uses,
            host_transition_headroom_bytes=(
                max(0, host_memory_available_bytes() - HOST_COPY_RESERVE_BYTES)
                // local_worker_count
            ),
            host_pin_headroom_bytes=host_pin_headroom_bytes,
            request_duration_ns=request_duration_ns,
            latency_upper_bound_ns_by_component=(latency_upper_bound_ns_by_component),
            shared_memory_pool=current_platform.device_shares_host_memory(),
            mapped_stream_cost_multiplier=mapped_stream_cost_multiplier,
        )

    def _build_pre_warmup_auto_residency_report(
        self, *, workload: DefaultWorkload
    ) -> RankResidencyReport:
        """Build exact weight lower bounds before the first serving forward."""
        if envs.SGLANG_DIFFUSION_DEBUG_HOST_MEMORY:
            from sglang.multimodal_gen.runtime.managers.memory_managers.host_memory_breakdown import (
                log_anon_vmas,
            )

            log_anon_vmas("before the pre-warmup plan (model loaded)")
        if self.pipeline is None:
            return RankResidencyReport(
                rank=self.rank,
                budget_bytes=0,
                estimated_peak_bytes=None,
                candidates=[],
                skip_reason="pipeline not initialized",
            )

        budget_bytes = self._auto_residency_budget_bytes(
            streamed_mapped_bytes=layerwise_streamed_mapped_bytes(
                self._auto_residency_modules()
            )
        )
        all_candidates = self._collect_auto_residency_targets(
            workload,
            # static weights cannot score pin placement; keep the current layout
            allow_host_pin_reallocation=False,
        )
        candidates = pre_warmup_residency_targets(
            all_candidates,
            excluded_components=(self.pipeline.preload_residency_excluded_components),
        )
        if not candidates:
            return RankResidencyReport(
                rank=self.rank,
                budget_bytes=budget_bytes,
                estimated_peak_bytes=None,
                candidates=[],
                skip_reason="no eligible residency alternatives",
            )
        modules = self._auto_residency_modules()
        active_weights = component_runtime_weight_bytes(modules)
        current_device_weights = component_current_device_weight_bytes(modules)
        permanent_components = {
            candidate.component_name
            for candidate in all_candidates
            if candidate.current_placement and candidate.permanent_residency
        }
        baseline_allocated = int(torch.get_device_module().memory_allocated())
        phase_peaks = {}
        phase_active = {}
        phase_used = {}
        for component_name in sorted(active_weights):
            phase_name = f"static:{component_name}"
            phase_peaks[phase_name] = baseline_allocated + (
                0
                if component_name in permanent_components
                else active_weights.get(component_name, 0)
            )
            phase_active[phase_name] = tuple(
                sorted(permanent_components | {component_name})
            )
            phase_used[phase_name] = (component_name,)

        local_worker_count = max(
            1, self.server_args.num_gpus // self.server_args.nnodes
        )
        pin_budget = self.server_args.host_pin_budget()
        return RankResidencyReport(
            rank=self.rank,
            budget_bytes=budget_bytes,
            host_shares_device_pool=current_platform.device_shares_host_memory(),
            estimated_peak_bytes=max(phase_peaks.values()),
            estimated_peak_bytes_by_phase=phase_peaks,
            active_components_by_phase=phase_active,
            used_components_by_phase=phase_used,
            prefetched_components_by_phase={},
            current_device_weight_bytes_by_component=current_device_weights,
            current_active_weight_bytes_by_component=active_weights,
            node_rank=self.server_args.node_rank,
            pinned_host_bytes=layerwise_pinned_host_bytes(
                modules, pin_budget=pin_budget
            ),
            host_pin_capacity_bytes=layerwise_host_pin_capacity_bytes(
                modules, pin_budget=pin_budget
            ),
            host_transition_headroom_bytes=max(
                0, host_memory_available_bytes() - HOST_COPY_RESERVE_BYTES
            )
            // local_worker_count,
            device_transition_allocated_bytes=baseline_allocated,
            candidates=candidates,
            require_feasible_placement=True,
            # Static weight accounting establishes feasibility, but contains no
            # latency observation. Give each exact current state one synthetic
            # unit so the solver preserves the maximum number of current
            # placements and only demotes what is necessary before the first
            # target-shape probe.
            estimated_request_duration_ns=1,
            candidate_latency_savings_ns={
                candidate.option_key(): int(candidate.current_placement)
                for candidate in candidates
            },
        )

    def _build_auto_residency_report(
        self,
        *,
        workload: DefaultWorkload,
        records: List[WarmupMemoryRecord],
        include_candidates: bool = True,
    ) -> RankResidencyReport:
        skip_reason = None
        if self.pipeline is None:
            skip_reason = "pipeline not initialized"
        elif not records:
            skip_reason = "no server warmup measurements"
        elif workload.workload_units() is None:
            skip_reason = "default workload resolution unknown"
        if skip_reason is not None:
            return RankResidencyReport(
                rank=self.rank,
                budget_bytes=0,
                estimated_peak_bytes=None,
                candidates=[],
                skip_reason=skip_reason,
            )
        modules = self._auto_residency_modules()
        target_units = workload.workload_units()
        assert target_units is not None
        target_records = [
            record
            for record in records
            if record.succeeded and record.workload_units() >= target_units
        ]
        warmup_oom = any(
            not record.succeeded and record.workload_units() <= target_units
            for record in records
        )
        runtime_weights_by_component = component_runtime_weight_bytes(modules)
        allocator_headroom_bytes = estimate_allocator_headroom_bytes(
            records=records,
            target_units=target_units,
        )
        if warmup_oom:
            (
                estimated_peak_bytes,
                estimated_phase_peaks,
                active_components_by_phase,
                used_components_by_phase,
                prefetched_components_by_phase,
                full_weight_transition_components_by_phase,
            ) = measured_failed_workload_phase_peaks(
                records=records,
                target_units=target_units,
            )
        else:
            estimated_peak_bytes = estimate_default_workload_peak_bytes(
                records=records,
                target_units=target_units,
                constant_weight_bytes=max(
                    runtime_weights_by_component.values(), default=0
                ),
            )
            (
                estimated_phase_peaks,
                active_components_by_phase,
                used_components_by_phase,
                prefetched_components_by_phase,
                full_weight_transition_components_by_phase,
            ) = estimate_workload_phase_peaks(
                records=records,
                target_units=target_units,
                component_weight_bytes=runtime_weights_by_component,
            )
        measured_used_components = {
            component_name
            for component_names in used_components_by_phase.values()
            for component_name in component_names
        }
        has_component_use_measurement = any(
            record.phase_used_components for record in records if record.succeeded
        )
        layerwise_layer_uses = estimate_layerwise_layer_uses(
            records=records,
            target_units=target_units,
            target_num_inference_steps=workload.num_inference_steps,
        )
        (
            measured_request_duration_ns,
            measured_stage_duration_ns,
            measured_component_stages,
        ) = estimate_default_workload_timing(
            records=records,
            target_units=target_units,
            target_num_inference_steps=workload.num_inference_steps,
        )
        explicitly_repeated_stages = {
            stage_name
            for record in records
            if record.succeeded
            for stage_name in record.stage_iterations
        }
        repeated_components = {
            component_name
            for component_name in layerwise_layer_uses
            if is_dit_component_name(component_name)
        }
        repeated_components.update(
            component_name
            for component_name, stage_names in measured_component_stages.items()
            if any(
                stage_name in explicitly_repeated_stages
                or (
                    stage_name.endswith("DenoisingStage")
                    and not stage_name.endswith("BeforeDenoisingStage")
                )
                for stage_name in stage_names
            )
        )
        if include_candidates:
            self._auto_residency_repeated_components = repeated_components
        probe_duration_ns = _probe_total_duration_ns(records, target_units)
        reference_probe_duration_ns = getattr(
            self, "_auto_residency_reference_probe_duration_ns", None
        )
        if (
            not warmup_oom
            and reference_probe_duration_ns is None
            and probe_duration_ns > 0
        ):
            reference_probe_duration_ns = probe_duration_ns
            self._auto_residency_reference_probe_duration_ns = probe_duration_ns
        reference_request_duration_ns = (
            self._auto_residency_reference_request_duration_ns
        )
        if (
            not warmup_oom
            and reference_request_duration_ns is None
            and measured_request_duration_ns > 0
        ):
            reference_request_duration_ns = measured_request_duration_ns
            self._auto_residency_reference_request_duration_ns = (
                reference_request_duration_ns
            )
            self._auto_residency_reference_stage_duration_ns = (
                measured_stage_duration_ns
            )
            self._auto_residency_reference_component_stages = measured_component_stages
        estimated_request_duration_ns = (
            reference_request_duration_ns or measured_request_duration_ns
        )
        reference_stage_duration_ns = (
            self._auto_residency_reference_stage_duration_ns
            or measured_stage_duration_ns
        )
        reference_component_stages = (
            self._auto_residency_reference_component_stages or measured_component_stages
        )
        latency_upper_bound_ns_by_component = {}
        for component_name in modules:
            component_stage_duration_ns = sum(
                reference_stage_duration_ns.get(stage_name, 0)
                for stage_name in reference_component_stages.get(component_name, ())
            )
            latency_upper_bound_ns_by_component[component_name] = (
                estimated_request_duration_ns
                if is_dit_component_name(component_name)
                or component_name in repeated_components
                else component_stage_duration_ns or estimated_request_duration_ns
            )
        budget_bytes = self._auto_residency_budget_bytes(
            streamed_mapped_bytes=layerwise_streamed_mapped_bytes(modules)
        )
        local_worker_count = max(
            1, self.server_args.num_gpus // self.server_args.nnodes
        )
        pin_budget = self.server_args.host_pin_budget()
        host_transition_headroom_bytes = (
            max(0, host_memory_available_bytes() - HOST_COPY_RESERVE_BYTES)
            // local_worker_count
        )
        pinned_host_bytes = (
            layerwise_pinned_host_bytes(modules, pin_budget=pin_budget)
            if include_candidates
            else 0
        )
        host_pin_capacity_bytes = (
            layerwise_host_pin_capacity_bytes(modules, pin_budget=pin_budget)
            if include_candidates
            else 0
        )
        candidates = []
        if include_candidates:
            candidate_started = time.perf_counter()
            candidates = self._collect_auto_residency_targets(
                workload,
                estimated_peak_bytes=estimated_peak_bytes,
                used_components=(
                    measured_used_components if has_component_use_measurement else None
                ),
                layerwise_layer_uses=layerwise_layer_uses,
                host_pin_headroom_bytes=max(
                    0, host_pin_capacity_bytes - pinned_host_bytes
                ),
                request_duration_ns=estimated_request_duration_ns,
                latency_upper_bound_ns_by_component=(
                    latency_upper_bound_ns_by_component
                ),
            )
            candidates_by_component: dict[str, int] = {}
            for candidate in candidates:
                candidates_by_component[candidate.component_name] = (
                    candidates_by_component.get(candidate.component_name, 0) + 1
                )
            logger.debug(
                "Auto residency candidate frontier built in %.3fs: %s",
                time.perf_counter() - candidate_started,
                candidates_by_component,
            )
        if warmup_oom:
            dit_candidate_deltas: dict[str, list[tuple[int, int, int]]] = {}
            for candidate in candidates:
                if not is_dit_component_name(candidate.component_name):
                    continue
                dit_candidate_deltas.setdefault(candidate.component_name, []).append(
                    (
                        round(candidate.active_device_delta_bytes / 1024**2),
                        round(candidate.inactive_device_delta_bytes / 1024**2),
                        round(candidate.device_transition_delta_bytes / 1024**2),
                    )
                )
            candidate_summary = {
                name: {
                    "count": len(deltas),
                    "active_mib": (
                        min(d[0] for d in deltas),
                        max(d[0] for d in deltas),
                    ),
                    "inactive_mib": (
                        min(d[1] for d in deltas),
                        max(d[1] for d in deltas),
                    ),
                    "transition_mib": (
                        min(d[2] for d in deltas),
                        max(d[2] for d in deltas),
                    ),
                }
                for name, deltas in dit_candidate_deltas.items()
            }
            logger.debug(
                "Auto residency OOM report on rank %d: phase_peaks_gib=%s, "
                "active=%s, used=%s, prefetched=%s, candidate_deltas_mib=%s",
                self.rank,
                {
                    phase: round(peak / GIB_BYTES, 3)
                    for phase, peak in estimated_phase_peaks.items()
                },
                active_components_by_phase,
                used_components_by_phase,
                prefetched_components_by_phase,
                candidate_summary,
            )
        candidate_latency_savings_ns = (
            estimate_candidate_latency_savings_ns(
                candidates=candidates,
                request_duration_ns=estimated_request_duration_ns,
                stage_duration_ns=reference_stage_duration_ns,
                component_stages=reference_component_stages,
                repeated_components=repeated_components,
            )
            if include_candidates
            else {}
        )
        return RankResidencyReport(
            rank=self.rank,
            budget_bytes=budget_bytes,
            estimated_peak_bytes=estimated_peak_bytes,
            planning_headroom_correction_bytes=allocator_headroom_bytes,
            target_workload_measured=bool(target_records),
            observed_reserved_bytes=max(
                (record.peak_reserved_bytes for record in target_records), default=0
            ),
            estimated_peak_bytes_by_phase=estimated_phase_peaks,
            active_components_by_phase=active_components_by_phase,
            used_components_by_phase=used_components_by_phase,
            prefetched_components_by_phase=prefetched_components_by_phase,
            full_weight_transition_components_by_phase=(
                full_weight_transition_components_by_phase
            ),
            current_device_weight_bytes_by_component=(
                component_current_device_weight_bytes(modules)
                if include_candidates
                else {}
            ),
            current_active_weight_bytes_by_component=runtime_weights_by_component,
            node_rank=self.server_args.node_rank,
            host_shares_device_pool=current_platform.device_shares_host_memory(),
            pinned_host_bytes=pinned_host_bytes,
            host_pin_capacity_bytes=host_pin_capacity_bytes,
            host_transition_headroom_bytes=(
                host_transition_headroom_bytes if include_candidates else 0
            ),
            device_transition_allocated_bytes=(
                int(torch.get_device_module().memory_allocated())
                if include_candidates
                else 0
            ),
            estimated_request_duration_ns=estimated_request_duration_ns,
            measured_request_duration_ns=measured_request_duration_ns,
            probe_duration_ns=probe_duration_ns,
            reference_probe_duration_ns=reference_probe_duration_ns or 0,
            candidate_latency_savings_ns=candidate_latency_savings_ns,
            candidates=candidates,
            warmup_oom=warmup_oom,
            skip_reason=skip_reason,
        )

    def _auto_residency_all_gather(self, obj: Any) -> list[Any]:
        replica_group = get_replica_group()
        if replica_group.world_size == 1:
            return [obj]
        gathered: list[Any] = [None] * replica_group.world_size
        torch.distributed.all_gather_object(
            gathered, obj, group=replica_group.cpu_group
        )
        return gathered

    def _mixed_dtype_residency_components(self) -> set[str]:
        if self.pipeline is None:
            return set()
        manager = get_global_component_residency_manager(
            self.pipeline, self.server_args
        )
        return manager.components_with_mixed_use_dtypes(
            self.pipeline.stages, self.server_args
        )

    def _auto_residency_modules(self) -> dict[str, object]:
        assert self.pipeline is not None
        manager = peek_global_component_residency_manager()
        if manager is None:
            return dict(self.pipeline.modules)
        return manager.placement_modules()

    def _fixed_custom_residency_strategy_names(self) -> set[str]:
        if self.pipeline is None:
            return set()
        fixed = {
            name
            for name, strategy in self.pipeline.component_residency_strategies.items()
            if not strategy.supports_auto_residency()
        }
        fixed.update(
            fixed_loading_residency_components(self.server_args, self.pipeline.modules)
        )
        return fixed

    def _latest_auto_residency_round(self) -> list[AppliedResidencyChange]:
        if not self._auto_residency_round_sizes:
            return []
        round_size = self._auto_residency_round_sizes[-1]
        if round_size <= 0 or round_size > len(self._auto_residency_applied):
            return []
        return self._auto_residency_applied[-round_size:]

    def _latest_auto_residency_round_is_resident_only(self) -> bool:
        """Whether validation follows only execution-monotonic promotions.

        Moving an offloaded component to resident removes transfer/hooks from
        its existing compute path. Its post-adjustment warmup must still prove
        VRAM safety, but one noisy duration sample must not undo that promotion.
        Layerwise layout and lower-memory transitions retain latency rollback.
        """
        changes = self._latest_auto_residency_round()
        return all(
            adjustment.residency_mode != RESIDENT
            and self.server_args.residency_mode(adjustment.component_name) == RESIDENT
            for adjustment in changes
        ) and bool(changes)

    def _latest_auto_residency_round_supports_short_validation(self) -> bool:
        """Whether one full-shape step covers every changed execution path."""
        return self._latest_auto_residency_round_is_resident_only()

    def _rollback_applied_residency_changes(
        self, *, latest_round_only: bool = False
    ) -> str | None:
        if latest_round_only:
            if not self._auto_residency_round_sizes:
                return None
            round_size = self._auto_residency_round_sizes[-1]
            if round_size == 0:
                self._auto_residency_round_sizes.pop()
                return None
            if round_size > len(self._auto_residency_applied):
                return "auto residency round history is inconsistent"
            applied = self._auto_residency_applied[-round_size:]
        else:
            if not self._auto_residency_applied:
                self._auto_residency_round_sizes = []
                return None
            applied = self._auto_residency_applied
        try:
            rollback_residency_changes(
                applied=applied,
                modules=self._auto_residency_modules(),
                server_args=self.server_args,
            )
        except Exception as e:
            logger.error(
                "Auto residency rollback failed on rank %d: %s",
                self.rank,
                e,
                exc_info=True,
            )
            return describe_error(e)
        self._invalidate_component_strategies(
            [adjustment.component_name for adjustment in applied]
        )
        if latest_round_only and self._auto_residency_round_sizes:
            del self._auto_residency_applied[-len(applied) :]
            self._auto_residency_round_sizes.pop()
        else:
            self._auto_residency_applied = []
            self._auto_residency_round_sizes = []
        self._auto_residency_last_applied_plan = None
        return None

    @staticmethod
    def _invalidate_component_strategies(component_names: List[str]) -> None:
        manager = peek_global_component_residency_manager()
        if manager is not None:
            manager.invalidate_component_strategies(component_names)

    def set_lora(
        self,
        lora_nickname: Union[str, List[str]],
        lora_path: Union[str, None, List[Union[str, None]]] = None,
        target: Union[str, List[str]] = "all",
        strength: Union[float, List[float]] = 1.0,
        merge_mode: str | None = None,
        lora_alpha: int | None | list[int | None] = None,
    ) -> OutputBatch:
        """
        Set the LoRA adapter(s) for the pipeline.
        Supports both single LoRA (backward compatible) and multiple LoRA adapters.

        Args:
            lora_nickname: The nickname(s) of the adapter(s). Can be a string or a list of strings.
            lora_path: Path(s) to the LoRA adapter(s). Can be a string, None, or a list of strings/None.
            target: Which transformer(s) to apply the LoRA to. Can be a string or a list of strings.
            strength: LoRA strength(s) for merge, default 1.0. Can be a float or a list of floats.
            merge_mode: Optional per-request LoRA merge mode.
        """
        if not isinstance(self.pipeline, LoRAPipeline):
            return OutputBatch(error="Lora is not enabled")
        self.pipeline.set_lora(
            lora_nickname,
            lora_path,
            target,
            strength,
            merge_mode=merge_mode,
            lora_alpha=lora_alpha,
        )
        return OutputBatch()

    def merge_lora_weights(
        self, target: str = "all", strength: float = 1.0
    ) -> OutputBatch:
        """
        Merge LoRA weights.

        Args:
            target: Which transformer(s) to merge.
            strength: LoRA strength for merge, default 1.0.
        """
        if not isinstance(self.pipeline, LoRAPipeline):
            return OutputBatch(error="Lora is not enabled")
        self.pipeline.merge_lora_weights(target, strength)
        return OutputBatch()

    def unmerge_lora_weights(self, target: str = "all") -> OutputBatch:
        """
        Unmerge LoRA weights.

        Args:
            target: Which transformer(s) to unmerge.
        """
        if not isinstance(self.pipeline, LoRAPipeline):
            return OutputBatch(error="Lora is not enabled")
        self.pipeline.unmerge_lora_weights(target)
        return OutputBatch()

    def list_loras(self) -> OutputBatch:
        """
        List loaded LoRA adapters and current application status per module.
        """
        from sglang.multimodal_gen.runtime.pipelines_core.lora.pipeline import (
            LoRAPipeline,
        )

        if not isinstance(self.pipeline, LoRAPipeline):
            return OutputBatch(error="Lora is not enabled")
        status = self.pipeline.get_lora_status()
        return OutputBatch(output=status)

    def release_memory_occupation(self) -> dict:
        return self._get_memory_occupation().release_memory_occupation()

    def resume_memory_occupation(self) -> dict:
        if self.memory_occupation is None:
            return {
                "success": True,
                "sleeping": False,
                "message": "already awake",
            }
        return self.memory_occupation.resume_memory_occupation()


OOM_MSG = """
OOM detected. Possible solutions:
  - If the OOM occurs during loading:
    1. Check available memory on every selected GPU, not only total capacity.
       In multi-GPU runs, the least-free selected GPU is the bottleneck.
    2. For single-GPU deployment, use `--performance-mode memory`, component CPU offload,
       or `--dit-layerwise-offload` for supported Wan/MOVA DiTs.
    3. For multi-GPU deployment, keep the default `--performance-mode auto` or set
       `--use-fsdp-inference true` to shard DiT weights with FSDP. FSDP is not a
       single-GPU substitute for CPU offload.
  - If the OOM occurs during runtime:
    1. Reduce resolution, `--num-frames`, or batch size.
    2. Use `--performance-mode memory` for lower memory usage.
    3. Enable SP/Ulysses/Ring for sequence-heavy workloads in multi-GPU setups.
    4. Use FSDP, with CFG parallelism when supported, for validated multi-GPU workloads.
    5. Use a lower-memory attention backend or quantization when available.
  Or, open an issue on GitHub https://github.com/sgl-project/sglang/issues/new/choose
"""


def _oom_exceptions():
    # torch.OutOfMemoryError exists only in some PyTorch builds
    types = [torch.cuda.OutOfMemoryError]
    if hasattr(torch, "OutOfMemoryError"):
        types.append(torch.OutOfMemoryError)
    return tuple(types)


def run_scheduler_process(
    local_rank: int,
    rank: int,
    master_port: int,
    server_args: ServerArgs,
    pipe_writer: mp.connection.Connection,
    # For all workers: pipe to receive tasks from rank 0
    task_pipe_r: mp.connection.Connection,
    # For slave workers: pipe to send results back to rank 0
    result_pipe_w: mp.connection.Connection | None,
    # For rank 0 worker only: pipes to send tasks to slaves
    task_pipes_to_slaves: list[mp.connection.Connection] | None = None,
    # For rank 0 worker only: pipes to receive results from slaves
    result_pipes_from_slaves: list[mp.connection.Connection] | None = None,
) -> None:
    """
    The entry point for the worker process.
    Rank 0 acts as the master, handling ZMQ requests and coordinating slaves.
    Ranks > 0 act as slaves, waiting for tasks from the master.
    """
    kill_itself_when_parent_died()
    configure_logger(server_args)
    globally_suppress_loggers()
    if current_platform.is_cuda():
        set_cuda_arch()
    elif current_platform.is_musa():
        set_musa_arch()

    init_diffusion_tracing(server_args, f"DiffWorker_rank{rank}")

    port_args = PortArgs.from_server_args(server_args)

    # start the scheduler event loop
    assert task_pipes_to_slaves is not None
    assert result_pipes_from_slaves is not None
    from sglang.multimodal_gen.runtime.managers.scheduler import Scheduler

    try:
        scheduler = Scheduler(
            server_args,
            gpu_id=rank,
            port_args=port_args,
            task_pipes_to_slaves=task_pipes_to_slaves,
            result_pipes_from_slaves=result_pipes_from_slaves,
            local_rank=local_rank,
        )
        logger.info(f"Worker {rank}: Scheduler loop started.")
        pipe_writer.send(
            {
                "status": "ready",
            }
        )
        scheduler.event_loop()
    except _oom_exceptions() as _e:
        logger.warning(OOM_MSG)
        raise
    finally:
        # Clean up resources to speed up shutdown
        if "scheduler" in locals():
            del scheduler
        gc.collect()
        if torch.cuda.is_initialized():
            torch.cuda.empty_cache()
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        logger.info(f"Worker {rank}: Shutdown complete.")
