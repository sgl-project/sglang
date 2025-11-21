# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
"""
DiffGenerator module for sgl-diffusion.

This module provides a consolidated interface for generating videos using
diffusion models.
"""

import logging
import multiprocessing as mp
import os
import time
from copy import deepcopy
from typing import Any

import imageio
import numpy as np
import torch
import torchvision
from einops import rearrange

from sglang.multimodal_gen.runtime.pipelines_core import Req
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch

# Suppress verbose logging from imageio, which is triggered when saving images.
logging.getLogger("imageio").setLevel(logging.WARNING)
logging.getLogger("imageio_ffmpeg").setLevel(logging.WARNING)
# Suppress Pillow plugin import logs when app log level is DEBUG
logging.getLogger("PIL").setLevel(logging.WARNING)
logging.getLogger("PIL.Image").setLevel(logging.WARNING)

from sglang.multimodal_gen.configs.sample.base import DataType, SamplingParams
from sglang.multimodal_gen.runtime.entrypoints.utils import prepare_request
from sglang.multimodal_gen.runtime.launch_server import launch_server
from sglang.multimodal_gen.runtime.managers.schedulerbase import SchedulerBase
from sglang.multimodal_gen.runtime.server_args import PortArgs, ServerArgs
from sglang.multimodal_gen.runtime.sync_scheduler_client import sync_scheduler_client
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

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


# TODO: rename
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
        **kwargs,
    ) -> "DiffGenerator":
        """
        Create a DiffGenerator from a pretrained model.

        Args:
            **kwargs: Additional arguments to customize model loading, set any ServerArgs or PipelineConfig attributes here.

        Returns:
            The created DiffGenerator

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

        return cls.from_server_args(server_args)

    @classmethod
    def from_server_args(cls, server_args: ServerArgs) -> "DiffGenerator":
        """
        Create a DiffGenerator with the specified arguments.

        Args:
            server_args: The inference arguments

        Returns:
            The created DiffGenerator
        """
        executor_class = SchedulerBase.get_class(server_args)
        instance = cls(
            server_args=server_args,
        )
        is_local_mode = server_args.is_local_mode
        logger.info(f"Local mode: {is_local_mode}")
        if is_local_mode:
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
                f"{self.server_args.scheduler_endpoint()} with `local mode` as False. "
                "Please ensure the server is running."
            )
        logger.info(
            f"Successfully connected to remote scheduler at "
            f"{self.server_args.scheduler_endpoint()}."
        )

    def post_process_sample(
        self,
        sample: torch.Tensor,
        data_type: DataType,
        fps: int,
        save_output: bool = True,
        save_file_path: str = None,
    ):
        """
        Process a single sample output and save output if necessary
        """
        # Process outputs
        if sample.dim() == 3:
            # for images, dim t is missing
            sample = sample.unsqueeze(1)
        sample = rearrange(sample, "c t h w -> t c h w")
        frames = []
        # TODO: this can be batched
        for x in sample:
            x = torchvision.utils.make_grid(x, nrow=6)
            x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
            frames.append((x * 255).numpy().astype(np.uint8))

        # Save outputs if requested
        if save_output:
            if save_file_path:
                os.makedirs(os.path.dirname(save_file_path), exist_ok=True)
                if data_type == DataType.VIDEO:
                    imageio.mimsave(
                        save_file_path,
                        frames,
                        fps=fps,
                        format=data_type.get_default_extension(),
                    )
                else:
                    imageio.imwrite(save_file_path, frames[0])
                logger.info("Saved output to %s", save_file_path)
            else:
                logger.warning("No output path provided, output not saved")

        return frames

    def generate(
        self,
        prompt: str | list[str] | None = None,
        sampling_params: SamplingParams | None = None,
        **kwargs,
    ) -> dict[str, Any] | list[np.ndarray] | list[dict[str, Any]] | None:
        """
        Generate a image/video based on the given prompt.

        Args:
            prompt: The prompt to use for generation (optional if prompt_txt is provided)
            output_file_name: Name of the file to save. Default is the first 100 characters of the prompt.
            save_output: Whether to save the output to disk
            return_frames: Whether to return the raw frames
            num_inference_steps: Number of denoising steps (overrides server_args)
            guidance_scale: Classifier-free guidance scale (overrides server_args)
            num_frames: Number of frames to generate (overrides server_args)
            height: Height of generated file (overrides server_args)
            width: Width of generated file (overrides server_args)
            fps: Frames per second for saved file (overrides server_args)
            seed: Random seed for generation (overrides server_args)
            callback: Callback function called after each step
            callback_steps: Number of steps between each callback

        Returns:
            Either the output dictionary, list of frames, or list of results for batch processing
        """
        # 1. prepare requests
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
        elif prompt is not None:
            if isinstance(prompt, str):
                prompts.append(prompt)
            elif isinstance(prompt, list):
                prompts.extend(prompt)
        else:
            raise ValueError("Either prompt or prompt_txt must be provided")

        pretrained_sampling_params = SamplingParams.from_pretrained(
            self.server_args.model_path, **kwargs
        )
        pretrained_sampling_params._merge_with_user_params(sampling_params)
        # TODO: simplify
        data_type = (
            DataType.IMAGE
            if self.server_args.pipeline_config.task_type.is_image_gen()
            or pretrained_sampling_params.num_frames == 1
            else DataType.VIDEO
        )
        pretrained_sampling_params.data_type = data_type
        pretrained_sampling_params.set_output_file_name()

        requests: list[Req] = []
        for output_idx, p in enumerate(prompts):
            current_sampling_params = deepcopy(pretrained_sampling_params)
            current_sampling_params.prompt = p
            requests.append(
                prepare_request(
                    p,
                    server_args=self.server_args,
                    sampling_params=current_sampling_params,
                )
            )

        results = []
        total_start_time = time.perf_counter()
        # 2. send requests to scheduler, one at a time
        # TODO: send batch when supported
        for request_idx, req in enumerate(requests):
            logger.info(
                "Processing prompt %d/%d: %s...",
                request_idx + 1,
                len(requests),
                req.prompt[:100],
            )
            try:
                start_time = time.perf_counter()
                output_batch = self._send_to_scheduler_and_wait_for_response([req])
                gen_time = time.perf_counter() - start_time
                if output_batch.error:
                    raise Exception(f"{output_batch.error}")

                # FIXME: in generate mode, an internal assertion error won't raise an error
                logger.info(
                    "Pixel data generated successfully in %.2f seconds",
                    gen_time,
                )

                if output_batch.output is None:
                    logger.error(
                        "Received empty output from scheduler for prompt %d",
                        request_idx + 1,
                    )
                    continue
                for output_idx, sample in enumerate(output_batch.output):
                    num_outputs = len(output_batch.output)
                    output_file_name = req.output_file_name
                    if num_outputs > 1 and output_file_name:
                        base, ext = os.path.splitext(output_file_name)
                        output_file_name = f"{base}_{output_idx}{ext}"

                    save_path = (
                        os.path.join(req.output_path, output_file_name)
                        if output_file_name
                        else None
                    )
                    frames = self.post_process_sample(
                        sample,
                        fps=req.fps,
                        save_output=req.save_output,
                        save_file_path=save_path,
                        data_type=req.data_type,
                    )

                    result_item: dict[str, Any] = {
                        "samples": sample,
                        "frames": frames,
                        "prompts": req.prompt,
                        "size": (req.height, req.width, req.num_frames),
                        "generation_time": gen_time,
                        "logging_info": output_batch.logging_info,
                        "trajectory": output_batch.trajectory_latents,
                        "trajectory_timesteps": output_batch.trajectory_timesteps,
                        "trajectory_decoded": output_batch.trajectory_decoded,
                        "prompt_index": output_idx,
                    }
                    results.append(result_item)
            except Exception as e:
                logger.error(
                    "Failed to generate output for prompt %d: %s", request_idx + 1, e
                )
                continue

        total_gen_time = time.perf_counter() - total_start_time
        logger.info(
            "Completed batch processing. Generated %d outputs in %.2f seconds.",
            len(results),
            total_gen_time,
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

    def set_lora_adapter(
        self, lora_nickname: str, lora_path: str | None = None
    ) -> None:
        # self.scheduler.set_lora_adapter(lora_nickname, lora_path)
        pass  # Removed as per edit hint

    def unmerge_lora_weights(self) -> None:
        """
        Use unmerged weights for inference to produce outputs that align with
        validation outputs generated during training.
        """
        # self.scheduler.unmerge_lora_weights()
        pass  # Removed as per edit hint

    def merge_lora_weights(self) -> None:
        # self.scheduler.merge_lora_weights()
        pass  # Removed as per edit hint

    def shutdown(self):
        """
        Shutdown the generator.
        If in local mode, it also shuts down the scheduler server.
        """
        # This sends the shutdown command to the server
        # self.scheduler.shutdown()

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
