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

import numpy as np
import torch

from sglang.multimodal_gen.configs.sample.sampling_params import (
    DataType,
    SamplingParams,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.utils import (
    ListLorasReq,
    MergeLoraWeightsReq,
    SetLoraReq,
    UnmergeLoraWeightsReq,
    format_lora_message,
)
from sglang.multimodal_gen.runtime.entrypoints.utils import (
    post_process_sample,
    prepare_request,
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
    ) -> dict[str, Any] | list[np.ndarray] | list[dict[str, Any]] | None:
        """
        Generate a image/video based on the given prompt.

        Args:

        Returns:
            Either the output dictionary, list of frames, or list of results for batch processing
        """
        # 1. prepare requests
        prompt = sampling_params_kwargs.get("prompt", None)
        prompts: list[str] = []
        # Handle batch processing from text file
        if self.server_args.prompt_file_path is not None:
            prompt_txt_path = self.server_args.prompt_file_path
            if not os.path.exists(prompt_txt_path):
                raise FileNotFoundError(
                    f"Prompt text file not found: {prompt_txt_path}"
                )
            # Read prompts from file
            with open(prompt_txt_path, encoding="utf-8") as f:
                prompts.extend(line.strip() for line in f if line.strip())

            if not prompts:
                raise ValueError(f"No prompts found in file: {prompt_txt_path}")

            logger.info("Found %d prompts in %s", len(prompts), prompt_txt_path)
        else:
            if prompt is None:
                prompt = " "
            if isinstance(prompt, str):
                prompts.append(prompt)
            elif isinstance(prompt, list):
                prompts.extend(prompt)
        sampling_params = SamplingParams.from_user_sampling_params_args(
            self.server_args.model_path,
            server_args=self.server_args,
            **sampling_params_kwargs,
        )

        # Extract diffusers_kwargs if passed
        diffusers_kwargs = sampling_params_kwargs.pop("diffusers_kwargs", None)

        requests: list[Req] = []
        for output_idx, p in enumerate(prompts):
            sampling_params.prompt = p
            req = prepare_request(
                server_args=self.server_args,
                sampling_params=sampling_params,
            )
            # Add diffusers_kwargs to request's extra dict
            if diffusers_kwargs:
                req.extra["diffusers_kwargs"] = diffusers_kwargs
            requests.append(req)

        results = []
        total_start_time = time.perf_counter()

        # 2. send requests to scheduler, one at a time
        # TODO: send batch when supported
        for request_idx, req in enumerate(requests):
            try:
                with log_generation_timer(
                    logger, req.prompt, request_idx + 1, len(requests)
                ) as timer:
                    output_batch = self._send_to_scheduler_and_wait_for_response([req])
                    if output_batch.error:
                        raise Exception(f"{output_batch.error}")

                    if output_batch.output is None:
                        logger.error(
                            "Received empty output from scheduler for prompt %d",
                            request_idx + 1,
                        )
                        continue
                    audio_sample_rate = output_batch.audio_sample_rate
                    for output_idx, sample in enumerate(output_batch.output):
                        num_outputs = len(output_batch.output)
                        audio = output_batch.audio
                        if req.data_type == DataType.VIDEO:
                            if isinstance(audio, torch.Tensor) and audio.ndim >= 2:
                                audio = (
                                    audio[output_idx]
                                    if audio.shape[0] > output_idx
                                    else None
                                )
                            elif isinstance(audio, np.ndarray) and audio.ndim >= 2:
                                audio = (
                                    audio[output_idx]
                                    if audio.shape[0] > output_idx
                                    else None
                                )
                            if audio is not None and not (
                                isinstance(sample, (tuple, list)) and len(sample) == 2
                            ):
                                sample = (sample, audio)
                        frames = post_process_sample(
                            sample,
                            fps=req.fps,
                            save_output=req.save_output,
                            # TODO: output file path for req should be determined
                            save_file_path=req.output_file_path(
                                num_outputs, output_idx
                            ),
                            data_type=req.data_type,
                            audio_sample_rate=audio_sample_rate,
                        )

                        result_item: dict[str, Any] = {
                            "samples": sample,
                            "frames": frames,
                            "audio": audio,
                            "prompts": req.prompt,
                            "size": (req.height, req.width, req.num_frames),
                            "generation_time": timer.duration,
                            "peak_memory_mb": output_batch.peak_memory_mb,
                            "timings": (
                                output_batch.timings.to_dict()
                                if output_batch.timings
                                else {}
                            ),
                            "trajectory": output_batch.trajectory_latents,
                            "trajectory_timesteps": output_batch.trajectory_timesteps,
                            "trajectory_decoded": output_batch.trajectory_decoded,
                            "prompt_index": output_idx,
                        }
                        results.append(result_item)
            except Exception:
                continue

        total_gen_time = time.perf_counter() - total_start_time
        log_batch_completion(logger, len(results), total_gen_time)

        if results:
            if self.server_args.warmup:
                total_duration_ms = results[0]["timings"]["total_duration_ms"]
                logger.info(
                    f"Warmed-up request processed in {GREEN}%.2f{RESET} seconds (with warmup excluded)",
                    total_duration_ms / 1000.0,
                )

            peak_memories = [r.get("peak_memory_mb", 0) for r in results]
            if peak_memories:
                max_peak_memory = max(peak_memories)
                avg_peak_memory = sum(peak_memories) / len(peak_memories)
                logger.info(
                    f"Memory usage - Max peak: {max_peak_memory:.2f} MB, "
                    f"Avg peak: {avg_peak_memory:.2f} MB"
                )

        if len(results) == 0:
            return None
        else:
            if requests[0].return_frames:
                results = [r["frames"] for r in results]
            if len(results) == 1:
                return results[0]
            return results

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

    def list_loras(self) -> OutputBatch:
        """
        List loaded LoRA adapters and current application status per module.
        """

        output = self._send_lora_request(
            req=ListLorasReq(),
            success_msg="Successfully listed LoRA adapters",
            failure_msg="Failed to list LoRA adapters",
        )
        if output.error is None:
            return output.output or {}
        else:
            raise RuntimeError(f"Failed to list LoRA adapters: {output.error}")

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
        if self.local_scheduler_process:
            logger.info("Waiting for local worker processes to terminate...")
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
