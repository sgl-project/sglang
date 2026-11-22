"""Config-time override declarations for Nemotron-3-Embed.

Architectures: EmbeddingNemotron3Model.
Rejects ROCm.
"""

from typing import Any

from sglang.srt.arg_groups.model_override_base import _register_for
from sglang.srt.runtime_context import get_platform


@_register_for("EmbeddingNemotron3Model")
def _embedding_nemotron3_overrides(server_args: Any, hf_config: Any) -> dict:
    if get_platform().is_hip:
        # aiter prefill is always causal, which would silently return causal
        # embeddings; no ROCm backend has been validated for this model.
        raise ValueError("Nemotron-3-Embed is not supported on ROCm.")
    return {}
