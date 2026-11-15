# SPDX-License-Identifier: Apache-2.0
"""Offline converter: stock ComfyUI workflow -> SGLD integrated mode (see workflow.py)."""

# Quoted by the runtime errors that a converted workflow avoids.
CONVERT_COMMAND = (
    "python -m sglang.multimodal_gen.apps.ComfyUI_SGLDiffusion.convert <workflow.json>"
)

from .workflow import Report, convert, convert_api, convert_workflow  # noqa: E402

__all__ = ["CONVERT_COMMAND", "Report", "convert", "convert_api", "convert_workflow"]
