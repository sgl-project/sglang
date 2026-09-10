# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for lazily loading pipeline components onto stages."""

from __future__ import annotations

from typing import Any

from sglang.multimodal_gen.runtime.loader.component_loaders.transformer_loader import (
    TransformerLoader,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs


def load_transformer_if_needed(stage: Any, server_args: ServerArgs) -> bool:
    """Load transformer onto stage if not marked loaded. Returns True if freshly loaded."""
    if server_args.model_loaded["transformer"]:
        return False
    stage.transformer = TransformerLoader().load(
        server_args.model_paths["transformer"], server_args, "transformer"
    )
    return True


def register_loaded_transformer(
    stage: Any, server_args: ServerArgs, pipeline: Any
) -> None:
    if pipeline is not None:
        pipeline.add_module("transformer", stage.transformer)
    server_args.model_loaded["transformer"] = True
