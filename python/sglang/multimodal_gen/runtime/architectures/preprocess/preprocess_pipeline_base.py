# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
import os
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from sglang.multimodal_gen.dataset import getdataset
from sglang.multimodal_gen.dataset.dataloader.parquet_io import (
    ParquetDatasetWriter,
    records_to_table,
)
from sglang.multimodal_gen.dataset.preprocessing_datasets import PreprocessBatch
from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.pipelines.composed_pipeline_base import (
    ComposedPipelineBase,
)
from sglang.multimodal_gen.runtime.pipelines.pipeline_batch_info import Req
from sglang.multimodal_gen.runtime.pipelines.stages import TextEncodingStage
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


class BasePreprocessPipeline(ComposedPipelineBase):
    """Base class for preprocessing pipelines that handles common functionality."""

    def create_pipeline_stages(self, server_args: ServerArgs):
        """Set up pipeline stages with proper dependency injection."""
        self.add_stage(
            stage_name="prompt_encoding_stage",
            stage=TextEncodingStage(
                text_encoders=[self.get_module("text_encoder")],
                tokenizers=[self.get_module("tokenizer")],
            ),
        )

    @torch.no_grad()
    def forward(
        self,
        batch: Req,
        server_args: ServerArgs,
        args,
    ):
        if not self.post_init_called:
            self.post_init()

        # Initialize class variables for data sharing
        self.video_data: dict[str, Any] = {}  # Store video metadata and paths
        self.latent_data: dict[str, Any] = {}  # Store latent tensors
        self.preprocess_video_and_text(server_args, args)

    def get_extra_features(
        self, valid_data: dict[str, Any], server_args: ServerArgs
    ) -> dict[str, Any]:
        """Get additional features specific to the pipeline type. Override in subclasses."""
        return {}

    def get_pyarrow_schema(self) -> pa.Schema:
        """Return the PyArrow schema for this pipeline. Must be overridden."""
        raise NotImplementedError

    def get_schema_fields(self) -> list[str]:
        """Get the schema fields for the pipeline type."""
        return [f.name for f in self.get_pyarrow_schema()]

    def create_record_for_schema(
        self, preprocess_batch: PreprocessBatch, schema: pa.Schema, strict: bool = False
    ) -> dict[str, Any]:
        """Create a record for the Parquet dataset using a generic schema-based approach.

        Args:
            preprocess_batch: The batch containing the data to extract
            schema: PyArrow schema defining the expected fields
            strict: If True, raises an exception when required fields are missing or unfilled

        Returns:
            Dictionary record matching the schema

        Raises:
            ValueError: If strict=True and required fields are missing or unfilled
        """
        record = {}
        unfilled_fields = []

        for field in schema.names:
            field_filled = False

            if field.endswith("_bytes"):
                # Handle binary tensor data - convert numpy array or tensor to bytes
                tensor_name = field.replace("_bytes", "")
                tensor_data = getattr(preprocess_batch, tensor_name, None)
                if tensor_data is not None:
                    try:
                        if hasattr(tensor_data, "numpy"):  # torch tensor
                            record[field] = tensor_data.cpu().numpy().tobytes()
                            field_filled = True
                        elif hasattr(tensor_data, "tobytes"):  # numpy array
                            record[field] = tensor_data.tobytes()
                            field_filled = True
                        else:
                            raise ValueError(
                                f"Unsupported tensor type for field {field}: {type(tensor_data)}"
                            )
                    except Exception as e:
                        if strict:
                            raise ValueError(
                                f"Failed to convert tensor {tensor_name} to bytes: {e}"
                            ) from e
                        record[field] = b""  # Empty bytes for missing data
                else:
                    record[field] = b""  # Empty bytes for missing data

            elif field.endswith("_shape"):
                # Handle tensor shape info
                tensor_name = field.replace("_shape", "")
                tensor_data = getattr(preprocess_batch, tensor_name, None)
                if tensor_data is not None and hasattr(tensor_data, "shape"):
                    record[field] = list(tensor_data.shape)
                    field_filled = True
                else:
                    record[field] = []

            elif field.endswith("_dtype"):
                # Handle tensor dtype info
                tensor_name = field.replace("_dtype", "")
                tensor_data = getattr(preprocess_batch, tensor_name, None)
                if tensor_data is not None and hasattr(tensor_data, "dtype"):
                    record[field] = str(tensor_data.dtype)
                    field_filled = True
                else:
                    record[field] = "unknown"

            elif field in ["width", "height", "num_frames"]:
                # Handle integer metadata fields
                value = getattr(preprocess_batch, field, None)
                if value is not None:
                    try:
                        record[field] = int(value)
                        field_filled = True
                    except (ValueError, TypeError) as e:
                        if strict:
                            raise ValueError(
                                f"Failed to convert field {field} to int: {e}"
                            ) from e
                        record[field] = 0
                else:
                    record[field] = 0

            elif field in ["duration_sec", "fps"]:
                # Handle float metadata fields
                # Map schema field names to batch attribute names
                attr_name = "duration" if field == "duration_sec" else field
                value = getattr(preprocess_batch, attr_name, None)
                if value is not None:
                    try:
                        record[field] = float(value)
                        field_filled = True
                    except (ValueError, TypeError) as e:
                        if strict:
                            raise ValueError(
                                f"Failed to convert field {field} to float: {e}"
                            ) from e
                        record[field] = 0.0
                else:
                    record[field] = 0.0

            else:
                # Handle string fields (id, file_name, caption, media_type, etc.)
                # Map common schema field names to batch attribute names
                attr_name = field
                if field == "caption":
                    attr_name = "text"
                elif field == "file_name":
                    attr_name = "path"
                elif field == "id":
                    # Generate ID from path if available
                    path_value = getattr(preprocess_batch, "path", None)
                    if path_value:
                        import os

                        record[field] = os.path.basename(path_value).split(".")[0]
                        field_filled = True
                    else:
                        record[field] = ""
                    continue
                elif field == "media_type":
                    # Determine media type from path
                    path_value = getattr(preprocess_batch, "path", None)
                    if path_value:
                        record[field] = (
                            "video" if path_value.endswith(".mp4") else "image"
                        )
                        field_filled = True
                    else:
                        record[field] = ""
                    continue

                value = getattr(preprocess_batch, attr_name, None)
                if value is not None:
                    record[field] = str(value)
                    field_filled = True
                else:
                    record[field] = ""

            # Track unfilled fields
            if not field_filled:
                unfilled_fields.append(field)

        # Handle strict mode
        if strict and unfilled_fields:
            raise ValueError(f"Required fields were not filled: {unfilled_fields}")

        # Log unfilled fields as warning if not in strict mode
        if unfilled_fields:
            logger.warning(
                "Some fields were not filled and got default values: %s",
                unfilled_fields,
            )

        return record

    def create_record(
        self,
        video_name: str,
        vae_latent: np.ndarray,
        text_embedding: np.ndarray,
        valid_data: dict[str, Any],
        idx: int,
        extra_features: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create a record for the Parquet dataset."""
        record = {
            "id": video_name,
            "vae_latent_bytes": vae_latent.tobytes(),
            "vae_latent_shape": list(vae_latent.shape),
            "vae_latent_dtype": str(vae_latent.dtype),
            "text_embedding_bytes": text_embedding.tobytes(),
            "text_embedding_shape": list(text_embedding.shape),
            "text_embedding_dtype": str(text_embedding.dtype),
            "file_name": video_name,
            "caption": valid_data["text"][idx] if len(valid_data["text"]) > 0 else "",
            "media_type": "video",
            "width": (
                valid_data["pixel_values"][idx].shape[-2]
                if len(valid_data["pixel_values"]) > 0
                else 0
            ),
            "height": (
                valid_data["pixel_values"][idx].shape[-1]
                if len(valid_data["pixel_values"]) > 0
                else 0
            ),
            "num_frames": vae_latent.shape[1] if len(vae_latent.shape) > 1 else 0,
            "duration_sec": (
                float(valid_data["duration"][idx])
                if len(valid_data["duration"]) > 0
                else 0.0
            ),
            "fps": float(valid_data["fps"][idx]) if len(valid_data["fps"]) > 0 else 0.0,
        }
        if extra_features:
            record.update(extra_features)
        return record

    def preprocess_video_and_text(self, server_args: ServerArgs, args):
        os.makedirs(args.output_dir, exist_ok=True)
        # Create directory for combined data
        combined_parquet_dir = os.path.join(args.output_dir, "combined_parquet_dataset")
        os.makedirs(combined_parquet_dir, exist_ok=True)
        local_rank = int(os.getenv("RANK", 0))

        # Get how many samples have already been processed
        start_idx = 0
        for root, _, files in os.walk(combined_parquet_dir):
            for file in files:
                if file.endswith(".parquet"):
                    table = pq.read_table(os.path.join(root, file))
                    start_idx += table.num_rows

        # Loading dataset
        train_dataset = getdataset(args)

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=args.preprocess_video_batch_size,
            num_workers=args.dataloader_num_workers,
        )

        num_processed_samples = 0
        # Add progress bar for video preprocessing
        pbar = tqdm(
            train_dataloader,
            desc="Processing videos",
            unit="batch",
            disable=local_rank != 0,
        )

        for batch_idx, data in enumerate(pbar):
            if data is None:
                continue

            with torch.inference_mode():
                # Filter out invalid samples (those with all zeros)
                valid_indices = []
                for i, pixel_values in enumerate(data["pixel_values"]):
                    if not torch.all(pixel_values == 0):  # Check if all values are zero
                        valid_indices.append(i)
                num_processed_samples += len(valid_indices)

                if not valid_indices:
                    continue

                # Create new batch with only valid samples
                valid_data = {
                    "pixel_values": torch.stack(
                        [data["pixel_values"][i] for i in valid_indices]
                    ),
                    "text": [data["text"][i] for i in valid_indices],
                    "path": [data["path"][i] for i in valid_indices],
                    "fps": [data["fps"][i] for i in valid_indices],
                    "duration": [data["duration"][i] for i in valid_indices],
                }

                # VAE
                with torch.autocast("cuda", dtype=torch.float32):
                    latents = (
                        self.get_module("vae")
                        .encode(valid_data["pixel_values"].to(get_local_torch_device()))
                        .mean
                    )

                # Get extra features if needed
                extra_features = self.get_extra_features(valid_data, server_args)

                batch_captions = valid_data["text"]
                batch = Req(
                    data_type="video",
                    prompt=batch_captions,
                    prompt_embeds=[],
                    prompt_attention_mask=[],
                )
                assert hasattr(self, "prompt_encoding_stage")
                result_batch = self.prompt_encoding_stage(batch, server_args)
                prompt_embeds, prompt_attention_mask = (
                    result_batch.prompt_embeds[0],
                    result_batch.prompt_attention_mask[0],
                )
                assert prompt_embeds.shape[0] == prompt_attention_mask.shape[0]

                # Get sequence lengths from attention masks (number of 1s)
                seq_lens = prompt_attention_mask.sum(dim=1)

                non_padded_embeds = []
                non_padded_masks = []

                # Process each item in the batch
                for i in range(prompt_embeds.size(0)):
                    seq_len = seq_lens[i].item()
                    # Slice the embeddings and masks to keep only non-padding parts
                    non_padded_embeds.append(prompt_embeds[i, :seq_len])
                    non_padded_masks.append(prompt_attention_mask[i, :seq_len])

                # Update the tensors with non-padded versions
                prompt_embeds = non_padded_embeds
                prompt_attention_mask = non_padded_masks

            # Prepare batch data for Parquet dataset
            batch_data = []

            # Add progress bar for saving outputs
            save_pbar = tqdm(
                enumerate(valid_data["path"]),
                desc="Saving outputs",
                unit="item",
                leave=False,
            )
            for idx, video_path in save_pbar:
                # Get the corresponding latent and info using video name
                latent = latents[idx].cpu()
                video_name = os.path.basename(video_path).split(".")[0]

                # Convert tensors to numpy arrays
                vae_latent = latent.cpu().numpy()
                text_embedding = prompt_embeds[idx].cpu().numpy()

                # Get extra features for this sample if needed
                sample_extra_features = {}
                if extra_features:
                    for key, value in extra_features.items():
                        if isinstance(value, torch.Tensor):
                            sample_extra_features[key] = value[idx].cpu().numpy()
                        else:
                            sample_extra_features[key] = value[idx]

                # Create record for Parquet dataset
                record = self.create_record(
                    video_name=video_name,
                    vae_latent=vae_latent,
                    text_embedding=text_embedding,
                    valid_data=valid_data,
                    idx=idx,
                    extra_features=sample_extra_features,
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
                        out_dir=combined_parquet_dir,
                        samples_per_file=args.samples_per_file,
                    )
                self.dataset_writer.append_table(table)
                logger.info("Collected batch with %s samples", len(table))

            if num_processed_samples >= args.flush_frequency:
                written = self.dataset_writer.flush()
                logger.info("Flushed %s samples to parquet", written)
                num_processed_samples = 0
