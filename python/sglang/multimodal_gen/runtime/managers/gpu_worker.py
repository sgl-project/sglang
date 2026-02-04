# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
import multiprocessing as mp
import os
import time
from typing import List, Union

import torch
from setproctitle import setproctitle

from sglang.multimodal_gen import envs
from sglang.multimodal_gen.runtime.distributed import (
    get_sp_group,
    maybe_init_distributed_environment_and_model_parallel,
)
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    get_cfg_group,
    get_tp_group,
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
from sglang.multimodal_gen.runtime.utils.layerwise_offload import OffloadableDiTMixin
from sglang.multimodal_gen.runtime.utils.logging_utils import (
    configure_logger,
    globally_suppress_loggers,
    init_logger,
)
from sglang.multimodal_gen.runtime.utils.perf_logger import PerformanceLogger

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
        setproctitle(f"sgl_diffusion::scheduler_TP{self.local_rank}")
        torch.cuda.set_device(self.local_rank)
        # Set environment variables for distributed initialization
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(self.master_port)
        os.environ["LOCAL_RANK"] = str(self.local_rank)
        os.environ["RANK"] = str(self.rank)
        os.environ["WORLD_SIZE"] = str(self.server_args.num_gpus)
        # Initialize the distributed environment
        maybe_init_distributed_environment_and_model_parallel(
            tp_size=self.server_args.tp_size,
            enable_cfg_parallel=self.server_args.enable_cfg_parallel,
            ulysses_degree=self.server_args.ulysses_degree,
            ring_degree=self.server_args.ring_degree,
            sp_size=self.server_args.sp_degree,
            dp_size=self.server_args.dp_size,
        )

        self.pipeline = build_pipeline(self.server_args)

        # apply layerwise offload after lora is applied while building LoRAPipeline
        # otherwise empty offloaded weights could fail lora converting
        if self.server_args.dit_layerwise_offload:
            # enable layerwise offload if possible
            for dit in filter(
                None,
                [
                    self.pipeline.get_module("transformer"),
                    self.pipeline.get_module("transformer_2"),
                    self.pipeline.get_module("video_dit"),
                    self.pipeline.get_module("video_dit_2"),
                    self.pipeline.get_module("audio_dit"),
                ],
            ):
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
        peak_memory_bytes = torch.cuda.max_memory_allocated()
        output_batch.peak_memory_mb = peak_memory_bytes / (1024**2)
        peak_memory_gb = peak_memory_bytes / (1024**3)
        remaining_gpu_mem_gb = (
            current_platform.get_device_total_memory() / (1024**3) - peak_memory_gb
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
        logger.info(
            f"Peak GPU memory: {peak_memory_gb:.2f} GB, "
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
                torch.cuda.reset_peak_memory_stats()

            start_time = time.monotonic()

            req.log(server_args=self.server_args)
            result = self.pipeline.forward(req, self.server_args)

            if isinstance(result, Req):
                output_batch = OutputBatch(
                    output=result.output,
                    audio=getattr(result, "audio", None),
                    audio_sample_rate=getattr(result, "audio_sample_rate", None),
                    timings=result.timings,
                    trajectory_timesteps=getattr(result, "trajectory_timesteps", None),
                    trajectory_latents=getattr(result, "trajectory_latents", None),
                    noise_pred=getattr(result, "noise_pred", None),
                    trajectory_decoded=getattr(result, "trajectory_decoded", None),
                )
            else:
                output_batch = result

            if self.rank == 0 and not req.suppress_logs:
                self.do_mem_analysis(output_batch)

            duration_ms = (time.monotonic() - start_time) * 1000
            output_batch.timings.total_duration_ms = duration_ms

            # TODO: extract to avoid duplication
            if req.perf_dump_path is not None or envs.SGLANG_DIFFUSION_STAGE_LOGGING:
                # Avoid logging warmup perf records that share the same request_id.
                if not req.is_warmup:
                    PerformanceLogger.log_request_summary(timings=output_batch.timings)
        except Exception as e:
            logger.error(
                f"Error executing request {req.request_id}: {e}", exc_info=True
            )
            if output_batch is None:
                output_batch = OutputBatch()
            output_batch.error = f"Error executing request {req.request_id}: {e}"
        finally:
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


OOM_MSG = f"""
OOM detected. Possible solutions:
  - If the OOM occurs during loading:
    1. Enable CPU offload for memory-intensive components, or use `--dit-layerwise-offload` for DiT
  - If the OOM occurs during runtime:
    1. Reduce the number of output tokens by lowering resolution or decreasing `--num-frames`
    2. Enable SP and/or TP
    3. Enable a sparse-attention backend
  Or, open an issue on GitHub https://github.com/sgl-project/sglang/issues/new/choose
"""


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
    except torch.OutOfMemoryError as _e:
        print(OOM_MSG)
        raise
    finally:
        logger.info(f"Worker {rank}: Shutdown complete.")
