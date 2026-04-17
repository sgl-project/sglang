# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
import os
import time
from typing import List

import torch

from sglang.multimodal_gen import envs
from sglang.multimodal_gen.runtime.distributed import (
    get_sp_group,
)
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    get_cfg_group,
    get_tp_group,
)
from sglang.multimodal_gen.runtime.entrypoints.utils import save_outputs
from sglang.multimodal_gen.runtime.pipelines_core import (
    ComposedPipelineBase,
    Req,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import (
    init_logger,
)
from sglang.multimodal_gen.runtime.utils.perf_logger import (
    PerformanceLogger,
)
from sglang.srt.utils import cpu_has_amx_support, get_cpu_ids_by_node

from .gpu_worker import OOM_MSG, GPUWorker, _oom_exceptions

_is_cpu_amx_available = cpu_has_amx_support()

logger = init_logger(__name__)


class CPUWorker(GPUWorker):
    """
    A worker that executes the model on pure CPU platforms
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
        if _is_cpu_amx_available:
            self.init_cpu_threads_binding()

        self.init_device_and_model()
        self.sp_group = get_sp_group()
        self.sp_cpu_group = self.sp_group.cpu_group
        self.tp_group = get_tp_group()
        self.tp_cpu_group = self.tp_group.cpu_group

        self.cfg_group = get_cfg_group()
        self.cfg_cpu_group = self.cfg_group.cpu_group

    def execute_forward(self, batch: List[Req]) -> OutputBatch:
        """
        Execute a forward pass.
        """
        assert self.pipeline is not None
        req = batch[0]
        output_batch = None
        try:
            start_time = time.monotonic()

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
                    rollout_trajectory_data=getattr(
                        result, "rollout_trajectory_data", None
                    ),
                    noise_pred=getattr(result, "noise_pred", None),
                    trajectory_decoded=getattr(result, "trajectory_decoded", None),
                )
            else:
                output_batch = result

            duration_ms = (time.monotonic() - start_time) * 1000
            output_batch.metrics.total_duration_ms = duration_ms

            # Save output to file and return file path only if requested. Avoid the serialization
            # and deserialization overhead between scheduler_client and cpu_worker.
            if req.save_output and req.return_file_paths_only:
                if self.rank == 0 and output_batch.output is not None:
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
                        enable_upscaling=req.enable_upscaling,
                        upscaling_model_path=req.upscaling_model_path,
                        upscaling_scale=req.upscaling_scale,
                    )
                    output_batch.output_file_paths = output_paths

                # No rank needs to hold on to generated tensors once the file-path
                # response has been materialized on rank 0
                output_batch.output = None
                output_batch.audio = None
                output_batch.audio_sample_rate = None

            # TODO: extract to avoid duplication
            if req.perf_dump_path is not None or envs.SGLANG_DIFFUSION_STAGE_LOGGING:
                # Avoid logging warmup perf records that share the same request_id.
                if not req.is_warmup:
                    PerformanceLogger.log_request_summary(metrics=output_batch.metrics)

            # dump per-request perf report to specified file (server mode)
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
            logger.error(
                f"Error executing request {req.request_id}: {e}", exc_info=True
            )
            if isinstance(e, _oom_exceptions()):
                logger.warning(OOM_MSG)
            if output_batch is None:
                output_batch = OutputBatch()
            output_batch.error = f"Error executing request {req.request_id}: {e}"
        return output_batch

    def init_cpu_threads_binding(self):
        omp_cpuids = os.environ.get("SGLANG_CPU_OMP_THREADS_BIND", "all")
        cpu_ids_by_node = get_cpu_ids_by_node()
        n_numa_node = len(cpu_ids_by_node)
        if omp_cpuids == "all":
            assert self.server_args.tp_size <= n_numa_node, (
                f"SGLANG_CPU_OMP_THREADS_BIND is not set, in this case, "
                f"tp_size {self.server_args.tp_size} should be smaller than or equal to number of numa node on the machine {n_numa_node}. "
                f"If you need tp_size to be larger than number of numa node, please set the CPU cores for each tp rank via SGLANG_CPU_OMP_THREADS_BIND explicitly. "
                f"For example, on a machine with 2 numa nodes, where core 0-31 are on numa node 0 and core 32-63 are on numa node 1, "
                f"it is suggested to use -tp 2 and bind tp rank 0 to core 0-31 and tp rank 1 to core 32-63. "
                f"This is the default behavior if SGLANG_CPU_OMP_THREADS_BIND is not set and it is the same as setting SGLANG_CPU_OMP_THREADS_BIND=0-31|32-63. "
                f"If you do need tp_size to be larger than the number of numa nodes, you could set SGLANG_CPU_OMP_THREADS_BIND explicitly for example SGLANG_CPU_OMP_THREADS_BIND=0-15|16-31|32-47|48-63 and run with -tp 4. "
                f"If you don't want each tp rank to use all the cores on one numa node, you could set for example SGLANG_CPU_OMP_THREADS_BIND=0-15|32-47 and run with -tp 2."
            )
            if self.server_args.tp_size < n_numa_node:
                logger.warning(
                    f"Detected the current machine has {n_numa_node} numa nodes available, but tp_size is set to {self.server_args.tp_size}, so only {self.server_args.tp_size} numa nodes are used."
                )
            self.local_omp_cpuid = cpu_ids_by_node[self.rank]
        else:
            threads_bind_list = omp_cpuids.split("|")
            assert self.server_args.tp_size == len(threads_bind_list), (
                f"SGLANG_CPU_OMP_THREADS_BIND setting must be aligned with TP size parameter ({self.server_args.tp_size}). "
                f"Please double check your settings."
            )
            self.local_omp_cpuid = threads_bind_list[self.rank]
            if self.server_args.tp_size > n_numa_node:
                logger.warning(
                    f"TP size ({self.server_args.tp_size})is larger than numa node number ({n_numa_node}), "
                    f"in this case the available memory amount of each rank cannot be determined in prior. "
                    f"Please set proper `--max-total-tokens` to avoid the out-of-memory error."
                )

        # Bind OpenMP threads to CPU cores
        torch.ops.sgl_kernel.init_cpu_threads_env(self.local_omp_cpuid)

        # Set local size to hint SGLang to use shared memory based AllReduce
        os.environ["LOCAL_SIZE"] = str(self.server_args.tp_size)
        torch.ops.sgl_kernel.initialize(self.server_args.tp_size, self.rank)

        @torch.library.register_fake("sgl_kernel::shm_allgather")
        def _(data, dim):
            return torch.cat([data] * self.server_args.tp_size, dim=dim)
