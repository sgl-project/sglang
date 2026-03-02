# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
"""
DiffGenerator module for sglang-diffusion.

This module provides a consolidated interface for generating images/videos using
diffusion models.
"""

import multiprocessing as mp
import os
import time
from typing import Any, List, Union

from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams
from sglang.multimodal_gen.runtime.entrypoints.utils import (
    GenerationResult,
    ListLorasReq,
    MergeLoraWeightsReq,
    SetLoraReq,
    ShutdownReq,
    UnmergeLoraWeightsReq,
    format_lora_message,
    prepare_request,
    save_outputs,
)
from sglang.multimodal_gen.runtime.launch_server import launch_server
from sglang.multimodal_gen.runtime.pipelines_core import Req
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch
from sglang.multimodal_gen.runtime.scheduler_client import sync_scheduler_client
from sglang.multimodal_gen.runtime.server_args import PortArgs, ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import (
    GREEN,
    RESET,
    init_logger,
    log_batch_completion,
    log_generation_timer,
)

logger = init_logger(__name__)

# TODO: move to somewhere appropriate
try:
    # Set the start method to 'spawn' to avoid CUDA errors in forked processes.
    # This must be done at the top level of the module, before any CUDA context
    # or other processes are initialized.
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    # The start method can only be set once per program execution.
    pass


class DiffGenerator:
    """
    A unified class for generating images/videos using diffusion models.

    This class provides a simple interface for image/video generation with rich
    customization options, similar to popular frameworks like HF Diffusers.
    """

    def __init__(
        self,
        server_args: ServerArgs,
    ):
        """
        Initialize the generator.

        Args:
            server_args: The inference arguments
        """
        self.server_args = server_args
        self.port_args = PortArgs.from_server_args(server_args)

        # The executor is now a client to the Scheduler service
        self.local_scheduler_process: list[mp.Process] | None = None
        self.owns_scheduler_client: bool = False

    @classmethod
    def from_pretrained(
        cls,
        local_mode: bool = True,
        **kwargs,
    ) -> "DiffGenerator":
        """
        Create a DiffGenerator from a pretrained model.

        Priority level: Default pipeline config < User's pipeline config < User's kwargs
        """
        # If users also provide some kwargs, it will override the ServerArgs and PipelineConfig.

        if (server_args := kwargs.get("server_args", None)) is not None:
            if isinstance(server_args, ServerArgs):
                pass
            elif isinstance(server_args, dict):
                server_args = ServerArgs.from_kwargs(**server_args)
        else:
            server_args = ServerArgs.from_kwargs(**kwargs)

        return cls.from_server_args(server_args, local_mode=local_mode)

    @classmethod
    def from_server_args(
        cls, server_args: ServerArgs, local_mode: bool = True
    ) -> "DiffGenerator":
        """
        Create a DiffGenerator with the specified arguments.

        Args:
            server_args: The inference arguments

        Returns:
            The created DiffGenerator
        """
        instance = cls(
            server_args=server_args,
        )
        logger.info(f"Local mode: {local_mode}")
        if local_mode:
            instance.local_scheduler_process = instance._start_local_server_if_needed()
        else:
            # In remote mode, we just need to connect and check.
            sync_scheduler_client.initialize(server_args)
            instance._check_remote_scheduler()

        # In both modes, this DiffGenerator instance is responsible for the client's lifecycle.
        instance.owns_scheduler_client = True
        return instance

    def _start_local_server_if_needed(
        self,
    ) -> list[mp.Process]:
        """Check if a local server is running; if not, start it and return the process handles."""
        # First, we need a client to test the server. Initialize it temporarily.
        sync_scheduler_client.initialize(self.server_args)

        processes = launch_server(self.server_args, launch_http_server=False)

        return processes

    def _check_remote_scheduler(self):
        """Check if the remote scheduler is accessible."""
        if not sync_scheduler_client.ping():
            raise ConnectionError(
                f"Could not connect to remote scheduler at "
                f"{self.server_args.scheduler_endpoint} with `local mode` as False. "
                "Please ensure the server is running."
            )
        logger.info(
            f"Successfully connected to remote scheduler at "
            f"{self.server_args.scheduler_endpoint}."
        )

    def generate(
        self,
        sampling_params_kwargs: dict | None = None,
    ) -> GenerationResult | list[GenerationResult] | None:
        """Generate image(s)/video(s) based on the given prompt(s).

        Returns a single GenerationResult for a single prompt, a list for
        multiple prompts, or None when every request failed.
        """
        # 1. prepare requests
        prompts = self._resolve_prompts(sampling_params_kwargs.get("prompt"))
        sampling_params = SamplingParams.from_user_sampling_params_args(
            self.server_args.model_path,
            server_args=self.server_args,
            **sampling_params_kwargs,
        )

        requests: list[Req] = []
        for p in prompts:
            sampling_params.prompt = p
            req = prepare_request(
                server_args=self.server_args,
                sampling_params=sampling_params,
            )
            requests.append(req)

        results: list[GenerationResult] = []
        total_start_time = time.perf_counter()

        # 2. send requests to scheduler one at a time
        # TODO: send batch when supported
        for request_idx, req in enumerate(requests):
            try:
                with log_generation_timer(
                    logger, req.prompt, request_idx + 1, len(requests)
                ) as timer:
                    output_batch = self._send_to_scheduler_and_wait_for_response([req])
                    if output_batch.error:
                        raise Exception(f"{output_batch.error}")

                    if (
                        output_batch.output is None
                        and output_batch.output_file_paths is None
                    ):
                        logger.error(
                            "Received empty output from scheduler for prompt %d",
                            request_idx + 1,
                        )
                        continue

                    common = dict(
                        prompt=req.prompt,
                        size=(req.height, req.width, req.num_frames),
                        generation_time=timer.duration,
                        peak_memory_mb=output_batch.peak_memory_mb,
                        metrics=(
                            output_batch.metrics.to_dict()
                            if output_batch.metrics
                            else {}
                        ),
                        trajectory_latents=output_batch.trajectory_latents,
                        trajectory_timesteps=output_batch.trajectory_timesteps,
                        trajectory_decoded=output_batch.trajectory_decoded,
                    )

                    if req.save_output and req.return_file_paths_only:
                        for idx, path in enumerate(output_batch.output_file_paths):
                            results.append(
                                GenerationResult(
                                    **common,
                                    prompt_index=idx,
                                    output_file_path=path,
                                )
                            )
                        continue

                    samples_out: list[Any] = []
                    audios_out: list[Any] = []
                    frames_out: list[Any] = []
                    num_outputs = len(output_batch.output)
                    save_outputs(
                        output_batch.output,
                        req.data_type,
                        req.fps,
                        req.save_output,
                        lambda idx: req.output_file_path(num_outputs, idx),
                        audio=output_batch.audio,
                        audio_sample_rate=output_batch.audio_sample_rate,
                        samples_out=samples_out,
                        audios_out=audios_out,
                        frames_out=frames_out,
                        output_compression=req.output_compression,
                        enable_frame_interpolation=req.enable_frame_interpolation,
                        frame_interpolation_exp=req.frame_interpolation_exp,
                        frame_interpolation_scale=req.frame_interpolation_scale,
                        frame_interpolation_model_path=req.frame_interpolation_model_path,
                    )

                    for idx in range(len(samples_out)):
                        results.append(
                            GenerationResult(
                                **common,
                                samples=samples_out[idx],
                                frames=frames_out[idx],
                                audio=audios_out[idx],
                                prompt_index=idx,
                                output_file_path=req.output_file_path(num_outputs, idx),
                            )
                        )
            except Exception as e:
                logger.error(
                    "Generation failed for prompt %d/%d: %s",
                    request_idx + 1,
                    len(requests),
                    e,
                    exc_info=True,
                )
                continue

        total_gen_time = time.perf_counter() - total_start_time
        log_batch_completion(logger, len(results), total_gen_time)
        self._log_summary(results)

        if not results:
            return None
        return results[0] if len(results) == 1 else results

    def _resolve_prompts(self, prompt: str | list[str] | None) -> list[str]:
        """Collect prompts from the argument or from a prompt file."""
        if self.server_args.prompt_file_path is not None:
            path = self.server_args.prompt_file_path
            if not os.path.exists(path):
                raise FileNotFoundError(f"Prompt text file not found: {path}")
            with open(path, encoding="utf-8") as f:
                prompts = [line.strip() for line in f if line.strip()]
            if not prompts:
                raise ValueError(f"No prompts found in file: {path}")
            logger.info("Found %d prompts in %s", len(prompts), path)
            return prompts

        if prompt is None:
            return [" "]
        if isinstance(prompt, str):
            return [prompt]
        return list(prompt)

    def _log_summary(self, results: list[GenerationResult]) -> None:
        if not results:
            return
        if self.server_args.warmup:
            total_duration_ms = results[0].metrics.get("total_duration_ms", 0)
            logger.info(
                f"Warmed-up request processed in {GREEN}%.2f{RESET} seconds (with warmup excluded)",
                total_duration_ms / 1000.0,
            )

        peak_memories = [r.peak_memory_mb for r in results if r.peak_memory_mb]
        if peak_memories:
            logger.info(
                f"Memory usage - Max peak: {max(peak_memories):.2f} MB, "
                f"Avg peak: {sum(peak_memories) / len(peak_memories):.2f} MB"
            )

    def _send_to_scheduler_and_wait_for_response(self, batch: list[Req]) -> OutputBatch:
        """
        Sends a request to the scheduler and waits for a response.
        """
        return sync_scheduler_client.forward(batch)

    # LoRA
    def _send_lora_request(self, req: Any, success_msg: str, failure_msg: str):
        response = sync_scheduler_client.forward(req)
        if response.error is None:
            logger.info(success_msg)
            return response
        else:
            error_msg = response.error
            raise RuntimeError(f"{failure_msg}: {error_msg}")

    def set_lora(
        self,
        lora_nickname: Union[str, List[str]],
        lora_path: Union[str, None, List[Union[str, None]]] = None,
        target: Union[str, List[str]] = "all",
        strength: Union[float, List[float]] = 1.0,
    ) -> None:
        """
        Set LoRA adapter(s) for the specified transformer(s).
        Supports both single LoRA (backward compatible) and multiple LoRA adapters.

        Args:
            lora_nickname: The nickname(s) of the adapter(s). Can be a string or a list of strings.
            lora_path: Path(s) to the LoRA adapter(s). Can be a string, None, or a list of strings/None.
            target: Which transformer(s) to apply the LoRA to. Can be a string or a list of strings.
                Valid values:
                - "all": Apply to all transformers (default)
                - "transformer": Apply only to the primary transformer (high noise for Wan2.2)
                - "transformer_2": Apply only to transformer_2 (low noise for Wan2.2)
                - "critic": Apply only to the critic model
            strength: LoRA strength(s) for merge, default 1.0. Can be a float or a list of floats.
        """
        req = SetLoraReq(
            lora_nickname=lora_nickname,
            lora_path=lora_path,
            target=target,
            strength=strength,
        )
        nickname_str, target_str, strength_str = format_lora_message(
            lora_nickname, target, strength
        )

        self._send_lora_request(
            req,
            f"Successfully set LoRA adapter(s): {nickname_str} (target: {target_str}, strength: {strength_str})",
            "Failed to set LoRA adapter",
        )

    def unmerge_lora_weights(self, target: str = "all") -> None:
        """
        Unmerge LoRA weights from the base model.

        Args:
            target: Which transformer(s) to unmerge.
        """
        req = UnmergeLoraWeightsReq(target=target)
        self._send_lora_request(
            req,
            f"Successfully unmerged LoRA weights (target: {target})",
            "Failed to unmerge LoRA weights",
        )

    def merge_lora_weights(self, target: str = "all", strength: float = 1.0) -> None:
        """
        Merge LoRA weights into the base model.

        Args:
            target: Which transformer(s) to merge.
            strength: LoRA strength for merge, default 1.0.
        """
        req = MergeLoraWeightsReq(target=target, strength=strength)
        self._send_lora_request(
            req,
            f"Successfully merged LoRA weights (target: {target}, strength: {strength})",
            "Failed to merge LoRA weights",
        )

    def list_loras(self) -> dict:
        """List loaded LoRA adapters and current application status per module."""
        output = self._send_lora_request(
            req=ListLorasReq(),
            success_msg="Successfully listed LoRA adapters",
            failure_msg="Failed to list LoRA adapters",
        )
        # _send_lora_request already raises on error, so output.error is always None here
        return output.output or {}

    def _ensure_lora_state(
        self,
        lora_path: str | None,
        lora_nickname: str | None = None,
        merge_lora: bool = True,
    ) -> None:
        """
        Ensure the LoRA state matches the desired configuration.

        Note: This method does not cache client-side state. The server handles
        idempotent operations, so redundant calls are safe but may have minor overhead.
        """
        if lora_path is None:
            # Unmerge all LoRA weights when no lora_path is provided
            self.unmerge_lora_weights()
            return

        lora_nickname = lora_nickname or self.server_args.lora_nickname

        # Set the LoRA adapter (server handles idempotent logic)
        self.set_lora(lora_nickname, lora_path)

        # Merge or unmerge based on the merge_lora flag
        if merge_lora:
            self.merge_lora_weights()
        else:
            self.unmerge_lora_weights()

    def generate_with_lora(
        self,
        prompt: str | list[str] | None = None,
        sampling_params: SamplingParams | None = None,
        *,
        lora_path: str | None = None,
        lora_nickname: str | None = None,
        merge_lora: bool = True,
        **kwargs,
    ):
        self._ensure_lora_state(
            lora_path=lora_path, lora_nickname=lora_nickname, merge_lora=merge_lora
        )
        return self.generate(
            sampling_params_kwargs=dict(
                prompt=prompt,
                sampling_params=sampling_params,
                **kwargs,
            )
        )

    def shutdown(self):
        """
        Shutdown the generator.
        If in local mode, it also shuts down the scheduler server.
        """
        # sends the shutdown command to the server
        if self.local_scheduler_process and self.owns_scheduler_client:
            try:
                sync_scheduler_client.forward(ShutdownReq())
            except Exception:
                pass

        if self.local_scheduler_process:
            for process in self.local_scheduler_process:
                process.join(timeout=10)
                if process.is_alive():
                    logger.warning(
                        f"Local worker {process.name} did not terminate gracefully, forcing."
                    )
                    process.terminate()
            self.local_scheduler_process = None

        if self.owns_scheduler_client:
            sync_scheduler_client.close()
            self.owns_scheduler_client = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()

    def __del__(self):
        if self.owns_scheduler_client:
            logger.warning(
                "Generator was garbage collected without being shut down. "
                "Attempting to shut down the local server and client."
            )
            self.shutdown()
        elif self.local_scheduler_process:
            logger.warning(
                "Generator was garbage collected without being shut down. "
                "Attempting to shut down the local server."
            )
            self.shutdown()
