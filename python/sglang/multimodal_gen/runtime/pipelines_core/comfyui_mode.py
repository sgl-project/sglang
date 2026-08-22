# SPDX-License-Identifier: Apache-2.0
"""Worker-side support for ``--comfyui-mode``.

ComfyUI owns the sampler loop and calls SGLang once per DiT step. This
module covers both halves of that contract:

- pipeline assembly: drop every module except the transformer, install a
  pass-through scheduler, keep only the stages needed for one forward
- run cache: after the first step, text embeddings stay on the worker;
  later steps send latents and the timestep

Single-file DiT loading lives in
``sglang.multimodal_gen.runtime.loader.comfyui_checkpoints``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from sglang.multimodal_gen.runtime.models.schedulers.scheduling_comfyui_passthrough import (
    ComfyUIPassThroughScheduler,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

if TYPE_CHECKING:
    from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
        ComposedPipelineBase,
    )

logger = init_logger(__name__)

COMFYUI_REQUIRED_MODULES = ["transformer", "scheduler"]

_CONDITIONING_FIELDS = (
    "prompt_embeds",
    "negative_prompt_embeds",
    "prompt_seq_lens",
    "negative_prompt_seq_lens",
    "pooled_embeds",
    "neg_pooled_embeds",
    "image_latent",
    "vae_image_sizes",
    "prompt_attention_mask",
    "negative_attention_mask",
    "prompt_embeds_mask",
    "negative_prompt_embeds_mask",
)

_SESSIONS: dict[str, dict[str, Any]] = {}


def is_comfyui_mode(server_args: ServerArgs) -> bool:
    return bool(server_args.comfyui_mode)


def initialize_comfyui_pipeline(
    pipeline: ComposedPipelineBase, server_args: ServerArgs
) -> None:
    """Install the pass-through scheduler and finish deriving VAE geometry.

    The VAE model itself is never loaded, but its config still carries the
    compression ratios that RoPE frequency construction reads.
    """
    pipeline.modules["scheduler"] = ComfyUIPassThroughScheduler(
        num_train_timesteps=1000
    )

    vae_config = getattr(server_args.pipeline_config, "vae_config", None)
    if (
        vae_config is not None
        and hasattr(vae_config, "post_init")
        and not hasattr(vae_config, "_post_init_called")
    ):
        vae_config.post_init()


def create_comfyui_pipeline_stages(
    pipeline: ComposedPipelineBase, server_args: ServerArgs
) -> None:
    from sglang.multimodal_gen.runtime.pipelines_core.stages import (
        ComfyUILatentPreparationStage,
        DenoisingStage,
    )

    transformer = pipeline.get_module("transformer")
    scheduler = pipeline.get_module("scheduler")
    pipeline.add_stages(
        [
            ComfyUILatentPreparationStage(scheduler=scheduler, transformer=transformer),
            DenoisingStage(transformer=transformer, scheduler=scheduler),
        ]
    )


def session_id_from_req(req) -> str | None:
    extra = getattr(req, "extra", None) or {}
    sid = extra.get("comfyui_session_id")
    return sid if sid else None


def bind_comfyui_session(req):
    """Restore cached conditioning, then refresh the cache from whatever is set."""
    sid = session_id_from_req(req)
    if not sid:
        return req

    cached = _SESSIONS.get(sid)
    if cached:
        for name, value in cached.items():
            current = getattr(req, name, None)
            if _is_empty(current):
                setattr(req, name, value)

    snapshot = {}
    for name in _CONDITIONING_FIELDS:
        value = getattr(req, name, None)
        if not _is_empty(value):
            snapshot[name] = value
    if snapshot:
        _SESSIONS[sid] = snapshot
    return req


def release_comfyui_session(session_id: str | None) -> None:
    if session_id:
        _SESSIONS.pop(session_id, None)


def _is_empty(value) -> bool:
    return value is None or value == [] or value == ()
