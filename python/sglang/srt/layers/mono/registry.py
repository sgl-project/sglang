"""Model architecture -> its mono spec, by import path: config-time code reads a spec without importing the model."""

from __future__ import annotations

import importlib
from typing import Optional

from sglang.srt.layers.mono.spec import MonoSpec

_SPECS = {
    "DeepseekV41ForCausalLM": "sglang.srt.models.deepseek_common.amd.mono.spec:MONO_SPEC",
    # V4 and V4.1 share the model class; the spec refuses anything but V4.1-Flash's config
    "DeepseekV4ForCausalLM": "sglang.srt.models.deepseek_common.amd.mono.spec:MONO_SPEC",
    "MiniMaxM3SparseForCausalLM": "sglang.srt.models.minimax_m3_mono.spec:MONO_SPEC",
    "MiniMaxM3SparseForConditionalGeneration": "sglang.srt.models.minimax_m3_mono.spec:MONO_SPEC",
}


def spec_path(architecture: str) -> Optional[str]:
    return _SPECS.get(architecture)


def load_spec(architecture: str) -> Optional[MonoSpec]:
    path = spec_path(architecture)
    if path is None:
        return None
    module, _, attr = path.partition(":")
    return getattr(importlib.import_module(module), attr)
