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
from sglang.multimodal_gen.runtime.managers.memory_managers.layerwise_offload import (
    configure_layerwise_offload_modules,
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

OFFLOAD_DISABLE_RECOMMENDATION_ORDER = (
    "vae",
    "image_encoder",
    "text_encoder",
    "text_encoder_2",
    "transformer",
)


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
        self.sp_group = get_sp_group()
        self.sp_cpu_group = self.sp_group.cpu_group
        self.tp_group = get_tp_group()
        self.tp_cpu_group = self.tp_group.cpu_group

        self.cfg_group = get_cfg_group()
        self.cfg_cpu_group = self.cfg_group.cpu_group
        self._realtime_sessions = RealtimeSessionCache(max_sessions=1)
        self.memory_occupation: MemoryOccupationController | None = None

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

        remaining_gpu_mem_gb = (
            current_platform.get_device_total_memory() / (1024**3) - peak_reserved_gb
        )
        can_stay_resident = self.get_can_stay_resident_components(remaining_gpu_mem_gb)

        pool_overhead_gb = peak_reserved_gb - peak_allocated_gb
        pool_overhead_pct = (
            pool_overhead_gb / peak_reserved_gb * 100 if peak_reserved_gb else 0.0
        )

        logger.debug(
            "GPU memory: peak=%.2f GB, allocated=%.2f GB, pool=%.2f GB (%.1f%%), "
            "headroom=%.2f GB. Components that can remain on GPU: %s. "
            "Adjust --cpu-offload-components or --layerwise-offload-components "
            "to change residency.",
            peak_reserved_gb,
            peak_allocated_gb,
            pool_overhead_gb,
            pool_overhead_pct,
            remaining_gpu_mem_gb,
            can_stay_resident,
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
        try:
            if not current_platform.is_cpu() and not current_platform.is_mps():
                torch.get_device_module().reset_peak_memory_stats()

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
            self._record_output_peak_memory(output_batch)

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
            self._record_output_peak_memory(output_batch)
            # clean cache if OOM
            if not current_platform.is_cpu():
                torch.get_device_module().empty_cache()
        return output_batch

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

    def _record_output_peak_memory(self, output_batch: OutputBatch) -> None:
        if current_platform.is_cpu():
            return
        peak_reserved_mb = capture_memory_snapshot().peak_reserved_mb
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
            [self._load_peak_reserved_mb, self._runtime_peak_reserved_mb],
            dtype=torch.float64,
            device=current_platform.get_device(self.local_rank),
        )
        peaks = get_replica_group().all_reduce(peaks, op=torch.distributed.ReduceOp.MAX)
        if not self.is_output_rank:
            return

        snapshot = capture_memory_snapshot()
        load_peak_mb, runtime_peak_mb = peaks.tolist()
        for metrics in output_metrics:
            metrics.record_memory_snapshot(
                "load_peak", replace(snapshot, peak_reserved_mb=load_peak_mb)
            )
            metrics.record_memory_snapshot(
                "runtime_peak", replace(snapshot, peak_reserved_mb=runtime_peak_mb)
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
        """
        Calculate which components can stay resident on GPU without being offloaded.
        """
        can_stay_resident = []
        if not self.pipeline:
            return can_stay_resident

        memory_usages = self.pipeline.memory_usages
        ordered_names = [
            name
            for name in OFFLOAD_DISABLE_RECOMMENDATION_ORDER
            if name in memory_usages
        ]
        ordered_names.extend(
            name
            for name in memory_usages
            if name not in OFFLOAD_DISABLE_RECOMMENDATION_ORDER
        )
        for name in ordered_names:
            usage = memory_usages[name]
            if not (
                self.server_args.should_cpu_offload_component(name)
                or self.server_args.should_configure_layerwise_offload_for_lazy_component(
                    name
                )
            ):
                continue
            if usage is None:
                continue
            if usage <= remaining_gpu_mem_gb:
                can_stay_resident.append(name)
                remaining_gpu_mem_gb -= usage

        return can_stay_resident

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
