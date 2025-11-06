# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
"""
ODE Trajectory Data Preprocessing pipeline implementation.

This module contains an implementation of the ODE Trajectory Data Preprocessing pipeline
using the modular pipeline architecture.

Sec 4.3 of CausVid paper: https://arxiv.org/pdf/2412.07772
"""

import os
from collections.abc import Iterator
from typing import Any

import pyarrow as pa
import torch
from torch.utils.data import DataLoader
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from sglang.multimodal_gen.configs.sample import SamplingParams
from sglang.multimodal_gen.dataset import gettextdataset
from sglang.multimodal_gen.dataset.dataloader.parquet_io import (
    ParquetDatasetWriter,
    records_to_table,
)
from sglang.multimodal_gen.dataset.dataloader.record_schema import (
    ode_text_only_record_creator,
)
from sglang.multimodal_gen.dataset.dataloader.schema import (
    pyarrow_schema_ode_trajectory_text_only,
)
from sglang.multimodal_gen.runtime.models.schedulers.scheduling_self_forcing_flow_match import (
    SelfForcingFlowMatchScheduler,
)
from sglang.multimodal_gen.runtime.pipelines.pipeline_batch_info import Req
from sglang.multimodal_gen.runtime.pipelines.preprocess.preprocess_pipeline_base import (
    BasePreprocessPipeline,
)
from sglang.multimodal_gen.runtime.pipelines.stages import (
    DecodingStage,
    DenoisingStage,
    InputValidationStage,
    LatentPreparationStage,
    TextEncodingStage,
    TimestepPreparationStage,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.utils import save_decoded_latents_as_video, shallow_asdict

logger = init_logger(__name__)


class PreprocessPipeline_ODE_Trajectory(BasePreprocessPipeline):
    """ODE Trajectory preprocessing pipeline implementation."""

    _required_config_modules = [
        "text_encoder",
        "tokenizer",
        "vae",
        "transformer",
        "scheduler",
    ]
    preprocess_dataloader: StatefulDataLoader
    preprocess_loader_iter: Iterator[dict[str, Any]]
    pbar: Any
    num_processed_samples: int

    def get_pyarrow_schema(self) -> pa.Schema:
        """Return the PyArrow schema for ODE Trajectory pipeline."""
        return pyarrow_schema_ode_trajectory_text_only

    def create_pipeline_stages(self, server_args: ServerArgs):
        """Set up pipeline stages with proper dependency injection."""
        assert server_args.pipeline_config.flow_shift == 5
        self.modules["scheduler"] = SelfForcingFlowMatchScheduler(
            shift=server_args.pipeline_config.flow_shift,
            sigma_min=0.0,
            extra_one_step=True,
        )
        self.modules["scheduler"].set_timesteps(
            num_inference_steps=48, denoising_strength=1.0
        )

        self.add_stage(
            stage_name="input_validation_stage", stage=InputValidationStage()
        )
        self.add_stage(
            stage_name="prompt_encoding_stage",
            stage=TextEncodingStage(
                text_encoders=[self.get_module("text_encoder")],
                tokenizers=[self.get_module("tokenizer")],
            ),
        )
        self.add_stage(
            stage_name="timestep_preparation_stage",
            stage=TimestepPreparationStage(scheduler=self.get_module("scheduler")),
        )
        self.add_stage(
            stage_name="latent_preparation_stage",
            stage=LatentPreparationStage(
                scheduler=self.get_module("scheduler"),
                transformer=self.get_module("transformer", None),
            ),
        )
        self.add_stage(
            stage_name="denoising_stage",
            stage=DenoisingStage(
                transformer=self.get_module("transformer"),
                scheduler=self.get_module("scheduler"),
                pipeline=self,
            ),
        )
        self.add_stage(
            stage_name="decoding_stage", stage=DecodingStage(vae=self.get_module("vae"))
        )

    def preprocess_text_and_trajectory(self, server_args: ServerArgs, args):
        """Preprocess text-only data and generate trajectory information."""

        for batch_idx, data in enumerate(self.pbar):
            if data is None:
                continue

            with torch.inference_mode():
                # For text-only processing, we only need text data
                # Filter out samples without text
                valid_indices = []
                for i, text in enumerate(data["text"]):
                    if text and text.strip():  # Check if text is not empty
                        valid_indices.append(i)
                self.num_processed_samples += len(valid_indices)

                if not valid_indices:
                    continue

                # Create new batch with only valid samples (text-only)
                valid_data = {
                    "text": [data["text"][i] for i in valid_indices],
                    "path": [data["path"][i] for i in valid_indices],
                }

                # Add fps and duration if available in data
                if "fps" in data:
                    valid_data["fps"] = [data["fps"][i] for i in valid_indices]
                if "duration" in data:
                    valid_data["duration"] = [
                        data["duration"][i] for i in valid_indices
                    ]

                batch_captions = valid_data["text"]
                # Encode text using the standalone TextEncodingStage API
                prompt_embeds_list, prompt_masks_list = (
                    self.prompt_encoding_stage.encode_text(
                        batch_captions,
                        server_args,
                        encoder_index=[0],
                        return_attention_mask=True,
                    )
                )
                prompt_embeds = prompt_embeds_list[0]
                prompt_attention_masks = prompt_masks_list[0]
                assert prompt_embeds.shape[0] == prompt_attention_masks.shape[0]

                sampling_params = SamplingParams.from_pretrained(args.model_path)

                # encode negative prompt for trajectory collection
                if (
                    sampling_params.guidance_scale > 1
                    and sampling_params.negative_prompt is not None
                ):
                    negative_prompt_embeds_list, negative_prompt_masks_list = (
                        self.prompt_encoding_stage.encode_text(
                            sampling_params.negative_prompt,
                            server_args,
                            encoder_index=[0],
                            return_attention_mask=True,
                        )
                    )
                    negative_prompt_embed = negative_prompt_embeds_list[0][0]
                    negative_prompt_attention_mask = negative_prompt_masks_list[0][0]
                else:
                    negative_prompt_embed = None
                    negative_prompt_attention_mask = None

                trajectory_latents = []
                trajectory_timesteps = []
                trajectory_decoded = []

                for i, (prompt_embed, prompt_attention_mask) in enumerate(
                    zip(prompt_embeds, prompt_attention_masks, strict=False)
                ):
                    prompt_embed = prompt_embed.unsqueeze(0)
                    prompt_attention_mask = prompt_attention_mask.unsqueeze(0)

                    # Collect the trajectory data (text-to-video generation)
                    batch = Req(
                        **shallow_asdict(sampling_params),
                    )
                    batch.prompt_embeds = [prompt_embed]
                    batch.prompt_attention_mask = [prompt_attention_mask]
                    batch.negative_prompt_embeds = [negative_prompt_embed]
                    batch.negative_attention_mask = [negative_prompt_attention_mask]
                    batch.num_inference_steps = 48
                    batch.return_trajectory_latents = True
                    # Enabling this will save the decoded trajectory videos.
                    # Used for debugging.
                    batch.return_trajectory_decoded = False
                    batch.height = args.max_height
                    batch.width = args.max_width
                    batch.fps = args.train_fps
                    batch.guidance_scale = 6.0
                    batch.do_classifier_free_guidance = True

                    result_batch = self.input_validation_stage(batch, server_args)
                    result_batch = self.timestep_preparation_stage(batch, server_args)
                    result_batch = self.latent_preparation_stage(
                        result_batch, server_args
                    )
                    result_batch = self.denoising_stage(result_batch, server_args)
                    result_batch = self.decoding_stage(result_batch, server_args)

                    trajectory_latents.append(result_batch.trajectory_latents.cpu())
                    trajectory_timesteps.append(result_batch.trajectory_timesteps.cpu())
                    trajectory_decoded.append(result_batch.trajectory_decoded)

                # Prepare extra features for text-only processing
                extra_features = {
                    "trajectory_latents": trajectory_latents,
                    "trajectory_timesteps": trajectory_timesteps,
                }

                if batch.return_trajectory_decoded:
                    for i, decoded_frames in enumerate(trajectory_decoded):
                        for j, decoded_frame in enumerate(decoded_frames):
                            save_decoded_latents_as_video(
                                decoded_frame,
                                f"decoded_videos/trajectory_decoded_{i}_{j}.mp4",
                                args.train_fps,
                            )

                # Prepare batch data for Parquet dataset
                batch_data: list[dict[str, Any]] = []

                # Add progress bar for saving outputs
                save_pbar = tqdm(
                    enumerate(valid_data["path"]),
                    desc="Saving outputs",
                    unit="item",
                    leave=False,
                )

                for idx, video_path in save_pbar:
                    video_name = os.path.basename(video_path).split(".")[0]

                    # Convert tensors to numpy arrays
                    text_embedding = prompt_embeds[idx].cpu().numpy()

                    # Get extra features for this sample
                    sample_extra_features = {}
                    if extra_features:
                        for key, value in extra_features.items():
                            if isinstance(value, torch.Tensor):
                                sample_extra_features[key] = value[idx].cpu().numpy()
                            else:
                                assert isinstance(value, list)
                                if isinstance(value[idx], torch.Tensor):
                                    sample_extra_features[key] = (
                                        value[idx].cpu().float().numpy()
                                    )
                                else:
                                    sample_extra_features[key] = value[idx]

                    # Create record for Parquet dataset (text-only ODE schema)
                    record: dict[str, Any] = ode_text_only_record_creator(
                        video_name=video_name,
                        text_embedding=text_embedding,
                        caption=valid_data["text"][idx],
                        trajectory_latents=sample_extra_features["trajectory_latents"],
                        trajectory_timesteps=sample_extra_features[
                            "trajectory_timesteps"
                        ],
                    )
                    batch_data.append(record)

                if batch_data:
                    write_pbar = tqdm(
                        total=1, desc="Writing to Parquet dataset", unit="batch"
                    )
                    table = records_to_table(batch_data, self.get_pyarrow_schema())
                    write_pbar.update(1)
                    write_pbar.close()

                    if not hasattr(self, "dataset_writer"):
                        self.dataset_writer = ParquetDatasetWriter(
                            out_dir=self.combined_parquet_dir,
                            samples_per_file=args.samples_per_file,
                        )
                    self.dataset_writer.append_table(table)

                    logger.info("Collected batch with %s samples", len(table))

                if self.num_processed_samples >= args.flush_frequency:
                    written = self.dataset_writer.flush()
                    logger.info("Flushed %s samples to parquet", written)
                    self.num_processed_samples = 0

        # Final flush for any remaining samples
        if hasattr(self, "dataset_writer"):
            written = self.dataset_writer.flush(write_remainder=True)
            if written:
                logger.info("Final flush wrote %s samples", written)

    def forward(self, batch: Req, server_args: ServerArgs, args):
        if not self.post_init_called:
            self.post_init()

        self.local_rank = int(os.getenv("RANK", 0))
        os.makedirs(args.output_dir, exist_ok=True)
        # Create directory for combined data
        self.combined_parquet_dir = os.path.join(
            args.output_dir, "combined_parquet_dataset"
        )
        os.makedirs(self.combined_parquet_dir, exist_ok=True)

        # Loading dataset
        train_dataset = gettextdataset(args)

        self.preprocess_dataloader = DataLoader(
            train_dataset,
            batch_size=args.preprocess_video_batch_size,
            num_workers=args.dataloader_num_workers,
        )

        self.preprocess_loader_iter = iter(self.preprocess_dataloader)

        self.num_processed_samples = 0
        # Add progress bar for video preprocessing
        self.pbar = tqdm(
            self.preprocess_loader_iter,
            desc="Processing videos",
            unit="batch",
            disable=self.local_rank != 0,
        )

        # Initialize class variables for data sharing
        self.video_data: dict[str, Any] = {}  # Store video metadata and paths
        self.latent_data: dict[str, Any] = {}  # Store latent tensors
        self.preprocess_text_and_trajectory(server_args, args)


EntryClass = PreprocessPipeline_ODE_Trajectory
