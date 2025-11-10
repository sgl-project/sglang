# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
"""
T2V Data Preprocessing pipeline implementation.

This module contains an implementation of the T2V Data Preprocessing pipeline
using the modular pipeline architecture.
"""
from sglang.multimodal_gen.dataset.dataloader.schema import pyarrow_schema_t2v
from sglang.multimodal_gen.runtime.pipelines.preprocess.preprocess_pipeline_base import (
    BasePreprocessPipeline,
)


class PreprocessPipeline_T2V(BasePreprocessPipeline):
    """T2V preprocessing pipeline implementation."""

    _required_config_modules = ["text_encoder", "tokenizer", "vae"]

    def get_pyarrow_schema(self):
        """Return the PyArrow schema for T2V pipeline."""
        return pyarrow_schema_t2v


EntryClass = PreprocessPipeline_T2V
