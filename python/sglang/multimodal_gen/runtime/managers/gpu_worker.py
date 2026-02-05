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
                torch.get_device_module().reset_peak_memory_stats()

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
                )
                output_batch.output_file_paths = output_paths
                output_batch.output = None

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

    # Module name to weight directory mapping for different model architectures
    _MODULE_WEIGHT_DIR_MAPPING = {
        "transformer": ["transformer", "dit", "model"],
        "transformer_2": ["transformer_2"],
        "video_dit": ["video_dit", "transformer", "dit", "model"],
        "video_dit_2": ["video_dit_2"],
        "audio_dit": ["audio_dit"],
    }

    # Default modules to update for RL workflows (typically only transformer is trained)
    _DEFAULT_TARGET_MODULES = [
        "transformer",
        "transformer_2",
        "video_dit",
        "video_dit_2",
        "audio_dit",
    ]

    def update_weights_from_disk(
        self,
        model_path: str,
        load_format: str = "auto",
        flush_cache: bool = True,
        target_modules: list[str] | None = None,
    ) -> tuple[bool, str]:
        """
        Update model weights from disk in-place without restarting the server.

        This method enables dynamic weight updates for RL workflows and iterative
        model fine-tuning scenarios. Includes rollback mechanism to restore original
        weights if loading fails.

        By default, updates ALL nn.Module components in the pipeline (transformer, vae,
        text_encoder, etc.). Use target_modules to specify a subset if needed.

        Args:
            model_path: Path to the new model weights (HuggingFace model path or local directory).
            load_format: Format of the weights to load (default: "auto").
            flush_cache: Whether to reset cache state after updating weights (default: True).
            target_modules: List of module names to update. If None or ["all"], updates all
                           nn.Module components. Specify a list like ["transformer"] to update
                           only specific modules.

        Returns:
            Tuple of (success: bool, message: str).
        """
        import gc
        import os

        from sglang.multimodal_gen.runtime.loader.utils import _list_safetensors_files
        from sglang.multimodal_gen.runtime.loader.weight_utils import (
            safetensors_weights_iterator,
        )
        from sglang.multimodal_gen.runtime.utils.hf_diffusers_utils import (
            maybe_download_model,
        )

        logger.info(f"Updating weights from disk: {model_path}")

        # Store original model path for potential rollback
        original_model_path = self.server_args.model_path

        if not self.pipeline:
            return False, "Pipeline is not initialized"

        available_modules: list[str] = []
        if hasattr(self.pipeline, "modules"):
            available_modules = list(self.pipeline.modules.keys())

        # Determine which modules to update
        if target_modules is None or target_modules == ["all"]:
            # Default: update all nn.Module components in the pipeline
            module_names = [
                name
                for name in available_modules
                if isinstance(self.pipeline.get_module(name), torch.nn.Module)
            ]
        else:
            module_names = target_modules

        # Collect all modules that need to be updated
        modules_to_update: list[tuple[str, torch.nn.Module]] = []

        for name in module_names:
            module = self.pipeline.get_module(name)
            if module is not None and isinstance(module, torch.nn.Module):
                modules_to_update.append((name, module))

        # For DiffusersPipeline, also check diffusers_pipe attributes
        diffusers_pipe = self.pipeline.get_module("diffusers_pipeline")
        if diffusers_pipe is not None and not modules_to_update:
            for name in module_names:
                if hasattr(diffusers_pipe, name):
                    module = getattr(diffusers_pipe, name)
                    if module is not None and isinstance(module, torch.nn.Module):
                        modules_to_update.append((name, module))

        if not modules_to_update:
            # Provide detailed error message
            error_msg = (
                f"No matching modules found for update. "
                f"Requested: {module_names}. "
                f"Available in pipeline: {available_modules}"
            )
            logger.error(error_msg)
            return False, error_msg

        # Helper function to find weights directory for a module
        def find_weights_dir(local_path: str, module_name: str) -> str | None:
            possible_dirs = self._MODULE_WEIGHT_DIR_MAPPING.get(
                module_name, [module_name]
            )
            for dir_name in possible_dirs:
                dir_path = os.path.join(local_path, dir_name)
                if os.path.exists(dir_path):
                    return dir_path
            # Fallback: check if weights are in root directory (for single-module models)
            if _list_safetensors_files(local_path):
                return local_path
            return None

        # Helper function to get weights iterator from a directory
        def get_weights_iter(weights_dir: str):
            safetensors_files = _list_safetensors_files(weights_dir)
            if not safetensors_files:
                raise FileNotFoundError(f"No safetensors files found in {weights_dir}")
            return safetensors_weights_iterator(safetensors_files), len(
                safetensors_files
            )

        # Helper function to load weights into model
        def load_weights_into_model(
            weights_iter, model_params: dict
        ) -> tuple[int, int]:
            try:
                from torch.distributed.tensor import DTensor, distribute_tensor
            except ImportError:
                DTensor = None
                distribute_tensor = None

            updated = 0
            skipped = 0
            for name, loaded_weight in weights_iter:
                if name in model_params:
                    param = model_params[name]
                    if param.shape == loaded_weight.shape:
                        if DTensor is not None and isinstance(param, DTensor):
                            # For DTensor, distribute the loaded weight first then copy
                            distributed_weight = distribute_tensor(
                                loaded_weight.to(param.device, param.dtype),
                                param.device_mesh,
                                param.placements,
                            )
                            param._local_tensor.copy_(distributed_weight._local_tensor)
                        else:
                            param.data.copy_(
                                loaded_weight.to(param.device, param.dtype)
                            )
                        updated += 1
                    else:
                        logger.warning(
                            f"Shape mismatch for {name}: model={param.shape}, loaded={loaded_weight.shape}"
                        )
                        skipped += 1
                else:
                    skipped += 1
            return updated, skipped

        # Download model if it's a HuggingFace path
        try:
            local_model_path = maybe_download_model(model_path)
        except Exception as e:
            return False, f"Failed to download model: {e}"

        # Phase 1: Validate ALL modules have their weight directories before any update
        # This ensures we don't do partial updates
        module_weights_map: dict[str, str] = {}  # module_name -> weights_dir
        missing_modules: list[str] = []

        for module_name, module in modules_to_update:
            weights_dir = find_weights_dir(local_model_path, module_name)
            if weights_dir is None:
                missing_modules.append(module_name)
            else:
                # Also validate that we can get weights iterator
                try:
                    safetensors_files = _list_safetensors_files(weights_dir)
                    if not safetensors_files:
                        missing_modules.append(module_name)
                    else:
                        module_weights_map[module_name] = weights_dir
                except Exception:
                    missing_modules.append(module_name)

        # Fail if any module is missing weights - no partial updates allowed
        if missing_modules:
            error_message = (
                f"Cannot update weights: missing weight files for modules: {missing_modules}. "
                f"All modules must have corresponding weights. No partial updates allowed."
            )
            logger.error(error_message)
            return False, error_message

        # Log which modules will be updated from which directories
        logger.info(
            f"Updating {len(module_weights_map)} modules: "
            + ", ".join(
                f"{name} <- {path}" for name, path in module_weights_map.items()
            )
        )

        # Phase 2: Update all modules
        # First, disable layerwise offload for all modules (load weights from CPU to GPU)
        offload_disabled_modules: list[torch.nn.Module] = []
        for module_name, module in modules_to_update:
            if (
                hasattr(module, "layerwise_offload_managers")
                and module.layerwise_offload_managers
            ):
                module.disable_offload()
                offload_disabled_modules.append(module)

        total_updated = 0
        total_skipped = 0
        updated_modules: list[str] = []

        for module_name, module in modules_to_update:
            weights_dir = module_weights_map[module_name]
            model_state_dict = dict(module.named_parameters())

            try:
                weights_iter, _ = get_weights_iter(weights_dir)
                updated, skipped = load_weights_into_model(
                    weights_iter, model_state_dict
                )
                total_updated += updated
                total_skipped += skipped
                updated_modules.append(module_name)
            except Exception as e:
                # Rollback ALL modules (including the ones already updated)
                error_message = (
                    f"Failed to update {module_name}: {e}. Rolling back all modules."
                )
                logger.error(error_message, exc_info=True)

                if updated_modules:
                    try:
                        original_local_path = maybe_download_model(original_model_path)
                        for rollback_name in updated_modules:
                            rollback_module = self.pipeline.get_module(rollback_name)
                            if rollback_module is None:
                                continue
                            rollback_weights_dir = find_weights_dir(
                                original_local_path, rollback_name
                            )
                            if rollback_weights_dir is None:
                                continue
                            rollback_iter, _ = get_weights_iter(rollback_weights_dir)
                            rollback_params = dict(rollback_module.named_parameters())
                            load_weights_into_model(rollback_iter, rollback_params)
                    except Exception as rollback_error:
                        logger.error(f"Rollback failed: {rollback_error}")
                        # Re-enable offload before returning
                        for m in offload_disabled_modules:
                            m.enable_offload()
                        return (
                            False,
                            f"{error_message} Rollback also failed: {rollback_error}",
                        )

                gc.collect()
                torch.cuda.empty_cache()
                # Re-enable offload before returning
                for m in offload_disabled_modules:
                    m.enable_offload()
                return False, error_message

        # Clean up GPU memory
        gc.collect()
        torch.cuda.empty_cache()

        # Reset cache state for all updated modules
        if flush_cache:
            for module_name, module in modules_to_update:
                if module_name in updated_modules:
                    self._reset_cache_state_after_weight_update(module)

        # Re-enable layerwise offload (sync new weights to CPU)
        for module in offload_disabled_modules:
            module.enable_offload()

        # Update the model path in server_args
        self.server_args.model_path = model_path

        message = f"Successfully updated {len(updated_modules)} modules ({', '.join(updated_modules)}): {total_updated} params updated"
        logger.info(message)
        return True, message

    def _reset_cache_state_after_weight_update(self, module: torch.nn.Module) -> None:
        """
        Reset cache state for a single module after weight updates.

        This resets TeaCache state. Cache-DiT context is automatically refreshed
        at the start of each inference request with the correct num_inference_steps,
        so we don't need to manually reset it here.

        Args:
            module: The module whose cache state should be reset.
        """
        # Reset TeaCache state if the module has it
        if hasattr(module, "reset_teacache_state"):
            module.reset_teacache_state()


OOM_MSG = f"""
OOM detected. Possible solutions:
  - If the OOM occurs during loading:
    1. Enable CPU offload for memory-intensive components, or use `--dit-layerwise-offload` for DiT
  - If the OOM occurs during runtime:
    1. Reduce the number of output tokens by lowering resolution or decreasing `--num-frames`
    2. Enable SP and/or TP
    3. Opt for a sparse-attention backend
    4. Enable FSDP by `--use-fsdp-inference` (in a multi-GPU setup)
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
    except torch.OutOfMemoryError as _e:
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
