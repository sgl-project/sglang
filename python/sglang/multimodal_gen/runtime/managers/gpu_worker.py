# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
import gc
import multiprocessing as mp
import os
import time
from typing import List, Union

import torch
from setproctitle import setproctitle

from sglang.multimodal_gen import envs
from sglang.multimodal_gen.runtime.distributed import (
    get_sp_group,
    get_tp_rank,
    get_tp_world_size,
    maybe_init_distributed_environment_and_model_parallel,
    model_parallel_is_initialized,
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
from sglang.multimodal_gen.runtime.entrypoints.utils import save_outputs
from sglang.multimodal_gen.runtime.loader.weight_utils import compute_weights_checksum
from sglang.multimodal_gen.runtime.loader.weights_updater import (
    WeightsUpdater,
    get_updatable_modules,
)
from sglang.multimodal_gen.runtime.pipelines_core import (
    ComposedPipelineBase,
    LoRAPipeline,
    Req,
    build_pipeline,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch
from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.multimodal_gen.runtime.server_args import PortArgs, ServerArgs
from sglang.multimodal_gen.runtime.utils.common import set_cuda_arch
from sglang.multimodal_gen.runtime.utils.layerwise_offload import (
    OffloadableDiTMixin,
    iter_materialized_weights,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import (
    configure_logger,
    globally_suppress_loggers,
    init_logger,
)
from sglang.multimodal_gen.runtime.utils.perf_logger import (
    PerformanceLogger,
    capture_memory_snapshot,
)

logger = init_logger(__name__)


class GPUWorker:
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
        self.master_port = master_port
        # FIXME: should we use tcp as distribute init method?
        self.server_args = server_args
        self.pipeline: ComposedPipelineBase = None

        self.init_device_and_model()
        self.sp_group = get_sp_group()
        self.sp_cpu_group = self.sp_group.cpu_group
        self.tp_group = get_tp_group()
        self.tp_cpu_group = self.tp_group.cpu_group

        self.cfg_group = get_cfg_group()
        self.cfg_cpu_group = self.cfg_group.cpu_group

    def init_device_and_model(self) -> None:
        """Initialize the device and load the model."""
        torch.get_device_module().set_device(self.local_rank)
        # Set environment variables for distributed initialization
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(self.master_port)
        os.environ["LOCAL_RANK"] = str(self.local_rank)
        os.environ["RANK"] = str(self.rank)
        os.environ["WORLD_SIZE"] = str(self.server_args.num_gpus)
        # initialize the distributed environment
        maybe_init_distributed_environment_and_model_parallel(
            tp_size=self.server_args.tp_size,
            enable_cfg_parallel=self.server_args.enable_cfg_parallel,
            ulysses_degree=self.server_args.ulysses_degree,
            ring_degree=self.server_args.ring_degree,
            sp_size=self.server_args.sp_degree,
            dp_size=self.server_args.dp_size,
            distributed_init_method=f"tcp://127.0.0.1:{self.master_port}",
            dist_timeout=self.server_args.dist_timeout,
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
        if self.server_args.dit_layerwise_offload:
            # enable layerwise offload if possible
            for module_name in [
                "transformer",
                "transformer_2",
                "video_dit",
                "video_dit_2",
                "audio_dit",
            ]:
                dit = self.pipeline.get_module(module_name)
                if dit:
                    if isinstance(dit, OffloadableDiTMixin):
                        dit.configure_layerwise_offload(self.server_args)
                    else:
                        logger.info(
                            f"Module {type(dit).__name__} does not support layerwise offload. Skipping."
                        )

        logger.info(
            f"Worker {self.rank}: Initialized device, model, and distributed environment."
        )

    def do_mem_analysis(self, output_batch: OutputBatch):
        final_snapshot = capture_memory_snapshot()
        if output_batch.metrics:
            output_batch.metrics.record_memory_snapshot("mem_analysis", final_snapshot)

        # for details on max_memory_reserved: https://docs.pytorch.org/docs/stable/generated/torch.cuda.memory.max_memory_reserved.html
        peak_reserved_bytes = torch.get_device_module().max_memory_reserved()
        peak_allocated_bytes = torch.get_device_module().max_memory_allocated()

        output_batch.peak_memory_mb = peak_reserved_bytes / (1024**2)
        peak_reserved_gb = peak_reserved_bytes / (1024**3)
        peak_allocated_gb = peak_allocated_bytes / (1024**3)

        remaining_gpu_mem_gb = (
            current_platform.get_device_total_memory() / (1024**3) - peak_reserved_gb
        )
        can_stay_resident = self.get_can_stay_resident_components(remaining_gpu_mem_gb)
        suggested_args = set()
        component_to_arg = {
            "vae": "--vae-cpu-offload",
            "text_encoder": "--text-encoder-cpu-offload",
            "text_encoder_2": "--text-encoder-cpu-offload",
            "image_encoder": "--image-encoder-cpu-offload",
        }

        for component in can_stay_resident:
            if component == "transformer":
                if self.server_args.dit_layerwise_offload:
                    suggested_args.add("--dit-layerwise-offload")
                elif self.server_args.dit_cpu_offload:
                    suggested_args.add("--dit-cpu-offload")
            elif component in component_to_arg:
                suggested_args.add(component_to_arg[component])

        suggested_args_str = (
            ", ".join(sorted(suggested_args)) if suggested_args else "None"
        )

        pool_overhead_gb = peak_reserved_gb - peak_allocated_gb

        logger.info(
            f"Peak GPU memory: {peak_reserved_gb:.2f} GB, "
            f"Peak allocated: {peak_allocated_gb:.2f} GB, "
            f"Memory pool overhead: {pool_overhead_gb:.2f} GB ({pool_overhead_gb / peak_reserved_gb * 100:.1f}%), "
            f"Remaining GPU memory at peak: {remaining_gpu_mem_gb:.2f} GB. "
            f"Components that could stay resident (based on the last request workload): {can_stay_resident}. "
            f"Related offload server args to disable: {suggested_args_str}"
        )

    def execute_forward(self, batch: List[Req]) -> OutputBatch:
        """
        Execute a forward pass.
        """
        assert self.pipeline is not None
        req = batch[0]
        output_batch = None
        try:
            if self.rank == 0:
                torch.get_device_module().reset_peak_memory_stats()

            start_time = time.monotonic()

            # capture memory baseline before forward
            if self.rank == 0 and req.metrics:
                baseline_snapshot = capture_memory_snapshot()
                req.metrics.record_memory_snapshot("before_forward", baseline_snapshot)

            req.log(server_args=self.server_args)
            result = self.pipeline.forward(req, self.server_args)

            if isinstance(result, Req):
                output_batch = OutputBatch(
                    output=result.output,
                    audio=getattr(result, "audio", None),
                    audio_sample_rate=getattr(result, "audio_sample_rate", None),
                    metrics=result.metrics,
                    trajectory_timesteps=getattr(result, "trajectory_timesteps", None),
                    trajectory_latents=getattr(result, "trajectory_latents", None),
                    noise_pred=getattr(result, "noise_pred", None),
                    trajectory_decoded=getattr(result, "trajectory_decoded", None),
                )
            else:
                output_batch = result

            # capture memory after forward (peak)
            if self.rank == 0 and output_batch.metrics:
                peak_snapshot = capture_memory_snapshot()
                output_batch.metrics.record_memory_snapshot(
                    "after_forward", peak_snapshot
                )

            if self.rank == 0 and not req.suppress_logs:
                self.do_mem_analysis(output_batch)

            duration_ms = (time.monotonic() - start_time) * 1000
            output_batch.metrics.total_duration_ms = duration_ms

            # Save output to file and return file path only if requested. Avoid the serialization
            # and deserialization overhead between scheduler_client and gpu_worker.
            if req.save_output and req.return_file_paths_only and self.rank == 0:
                output_paths = save_outputs(
                    output_batch.output,
                    req.data_type,
                    req.fps,
                    True,
                    lambda idx: req.output_file_path(len(output_batch.output), idx),
                    audio=output_batch.audio,
                    audio_sample_rate=output_batch.audio_sample_rate,
                    output_compression=req.output_compression,
                    enable_frame_interpolation=req.enable_frame_interpolation,
                    frame_interpolation_exp=req.frame_interpolation_exp,
                    frame_interpolation_scale=req.frame_interpolation_scale,
                    frame_interpolation_model_path=req.frame_interpolation_model_path,
                )
                output_batch.output_file_paths = output_paths
                output_batch.output = None

            # TODO: extract to avoid duplication
            if req.perf_dump_path is not None or envs.SGLANG_DIFFUSION_STAGE_LOGGING:
                # Avoid logging warmup perf records that share the same request_id.
                if not req.is_warmup:
                    PerformanceLogger.log_request_summary(metrics=output_batch.metrics)
        except Exception as e:
            logger.error(
                f"Error executing request {req.request_id}: {e}", exc_info=True
            )
            if isinstance(e, _oom_exceptions()):
                logger.warning(OOM_MSG)
            if output_batch is None:
                output_batch = OutputBatch()
            output_batch.error = f"Error executing request {req.request_id}: {e}"
        return output_batch

    def get_can_stay_resident_components(
        self, remaining_gpu_mem_gb: float
    ) -> List[str]:
        """
        Calculate which components can stay resident on GPU without being offloaded.
        """
        can_stay_resident = []
        if not self.pipeline:
            return can_stay_resident

        # Map memory_usage keys to server_args offload flags
        # If the flag is False, the component is ALREADY resident, so we don't suggest it.
        # If the flag is True, it is currently offloaded, so it's a candidate to "stay resident".
        offload_flags = {
            "transformer": self.server_args.dit_cpu_offload
            or self.server_args.dit_layerwise_offload,
            "vae": self.server_args.vae_cpu_offload,
            "text_encoder": self.server_args.text_encoder_cpu_offload,
            "text_encoder_2": self.server_args.text_encoder_cpu_offload,
            "image_encoder": self.server_args.image_encoder_cpu_offload,
        }

        for name, usage in self.pipeline.memory_usages.items():
            # Only consider components that are currently configured to be offloaded
            is_offload_configured = offload_flags.get(name, False)
            if not is_offload_configured:
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
    ) -> OutputBatch:
        """
        Set the LoRA adapter(s) for the pipeline.
        Supports both single LoRA (backward compatible) and multiple LoRA adapters.

        Args:
            lora_nickname: The nickname(s) of the adapter(s). Can be a string or a list of strings.
            lora_path: Path(s) to the LoRA adapter(s). Can be a string, None, or a list of strings/None.
            target: Which transformer(s) to apply the LoRA to. Can be a string or a list of strings.
            strength: LoRA strength(s) for merge, default 1.0. Can be a float or a list of floats.
        """
        if not isinstance(self.pipeline, LoRAPipeline):
            return OutputBatch(error="Lora is not enabled")
        self.pipeline.set_lora(lora_nickname, lora_path, target, strength)
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
        from sglang.multimodal_gen.runtime.pipelines_core.lora_pipeline import (
            LoRAPipeline,
        )

        if not isinstance(self.pipeline, LoRAPipeline):
            return OutputBatch(error="Lora is not enabled")
        status = self.pipeline.get_lora_status()
        return OutputBatch(output=status)

    def update_weights_from_disk(
        self,
        model_path: str,
        flush_cache: bool = True,
        target_modules: list[str] | None = None,
    ) -> tuple[bool, str]:
        """Update model weights from disk inplace without restarting the server."""
        if not self.pipeline:
            return False, "Pipeline is not initialized"

        updater = WeightsUpdater(self.pipeline)
        success, message = updater.update_weights_from_disk(
            model_path,
            flush_cache=flush_cache,
            target_modules=target_modules,
        )
        if success:
            self.server_args.model_path = model_path
            self.pipeline.model_path = model_path
        return success, message

    def get_weights_checksum(
        self, module_names: list[str] | None = None
    ) -> dict[str, str]:
        """Compute SHA-256 checksum of each module's weights."""
        if not self.pipeline:
            return {"error": "Pipeline is not initialized"}

        all_modules = get_updatable_modules(self.pipeline)
        names = module_names if module_names is not None else list(all_modules.keys())

        checksums: dict[str, str] = {}
        for name in names:
            module = all_modules.get(name)
            if module is None:
                checksums[name] = "not_found"
                continue
            checksums[name] = compute_weights_checksum(
                iter_materialized_weights(module)
            )
        return checksums


OOM_MSG = f"""
OOM detected. Possible solutions:
  - If the OOM occurs during loading:
    1. Enable CPU offload for memory-intensive components, or use `--dit-layerwise-offload` for DiT
  - If the OOM occurs during runtime:
    1. Enable SP and/or TP (in a multi-GPU setup)
    2. Reduce the number of output tokens by lowering resolution or decreasing `--num-frames`
    3. Opt for a sparse-attention backend
    4. Enable FSDP by `--use-fsdp-inference` (in a multi-GPU setup)
    5. Enable quantization (e.g. nunchaku)
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
    configure_logger(server_args)
    globally_suppress_loggers()
    if current_platform.is_cuda():
        set_cuda_arch()

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
