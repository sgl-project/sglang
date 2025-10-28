# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
"""
Text-only Data Preprocessing pipeline implementation.

This module contains an implementation of the Text-only Data Preprocessing pipeline
using the modular pipeline architecture, based on the ODE Trajectory preprocessing.
"""

import os
from collections.abc import Iterator
from typing import Any

import torch
from torch.utils.data import DataLoader
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from sglang.multimodal_gen.dataset import gettextdataset
from sglang.multimodal_gen.dataset.dataloader.parquet_io import (
    ParquetDatasetWriter,
    records_to_table,
)
from sglang.multimodal_gen.dataset.dataloader.record_schema import (
    text_only_record_creator,
)
from sglang.multimodal_gen.dataset.dataloader.schema import pyarrow_schema_text_only
from sglang.multimodal_gen.runtime.pipelines.pipeline_batch_info import Req
from sglang.multimodal_gen.runtime.pipelines.preprocess.preprocess_pipeline_base import (
    BasePreprocessPipeline,
)
from sglang.multimodal_gen.runtime.pipelines.stages import TextEncodingStage
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


class PreprocessPipeline_Text(BasePreprocessPipeline):
    """Text-only preprocessing pipeline implementation."""

    _required_config_modules = ["text_encoder", "tokenizer"]
    preprocess_dataloader: StatefulDataLoader
    preprocess_loader_iter: Iterator[dict[str, Any]]
    pbar: Any
    num_processed_samples: int = 0

    def get_pyarrow_schema(self):
        """Return the PyArrow schema for text-only pipeline."""
        return pyarrow_schema_text_only

    def create_pipeline_stages(self, server_args: ServerArgs):
        """Set up pipeline stages with proper dependency injection."""
        self.add_stage(
            stage_name="prompt_encoding_stage",
            stage=TextEncodingStage(
                text_encoders=[self.get_module("text_encoder")],
                tokenizers=[self.get_module("tokenizer")],
            ),
        )

    def preprocess_text_only(self, server_args: ServerArgs, args):
        """Preprocess text-only data."""

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

                logger.info("===== prompt_embeds: %s", prompt_embeds.shape)
                logger.info(
                    "===== prompt_attention_masks: %s", prompt_attention_masks.shape
                )

                # Prepare batch data for Parquet dataset
                batch_data = []

                # Add progress bar for saving outputs
                save_pbar = tqdm(
                    enumerate(valid_data["path"]),
                    desc="Saving outputs",
                    unit="item",
                    leave=False,
                )

                for idx, text_path in save_pbar:
                    text_name = os.path.basename(text_path).split(".")[0]

                    # Convert tensors to numpy arrays
                    text_embedding = prompt_embeds[idx].cpu().numpy()

                    # Create record for Parquet dataset (text-only schema)
                    record = text_only_record_creator(
                        text_name=text_name,
                        text_embedding=text_embedding,
                        caption=valid_data["text"][idx],
                    )
                    batch_data.append(record)

                if batch_data:
                    write_pbar = tqdm(
                        total=1, desc="Writing to Parquet dataset", unit="batch"
                    )
                    table = records_to_table(batch_data, pyarrow_schema_text_only)
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

    # Text-only record creation moved to sglang.multimodal_gen.dataset.dataloader.record_schema

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

        # Loading text dataset
        train_dataset = gettextdataset(args)

        self.preprocess_dataloader = DataLoader(
            train_dataset,
            batch_size=args.preprocess_video_batch_size,
            num_workers=args.dataloader_num_workers,
        )

        self.preprocess_loader_iter = iter(self.preprocess_dataloader)

        self.num_processed_samples = 0
        # Add progress bar for text preprocessing
        self.pbar = tqdm(
            self.preprocess_loader_iter,
            desc="Processing text",
            unit="batch",
            disable=self.local_rank != 0,
        )

        # Initialize class variables for data sharing
        self.text_data: dict[str, Any] = {}  # Store text metadata and paths

        self.preprocess_text_only(server_args, args)


EntryClass = PreprocessPipeline_Text
