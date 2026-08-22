# SPDX-License-Identifier: Apache-2.0
"""Worker-side support for ``--comfyui-mode``.

ComfyUI owns the sampler loop and calls SGLang once per DiT step. This
module covers both halves of that contract:

- pipeline assembly: drop every module except the transformer, install a
  pass-through scheduler, keep only the stages needed for one forward
- run cache: after the first step, text embeddings and packed extras stay
  on the worker; later steps send latents and the timestep

Single-file / GGUF DiT loading lives in
``sglang.multimodal_gen.runtime.loader.comfyui_checkpoints``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

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
    "sigmas",
)
_SESSION_SKIP_EXTRA = frozenset(
    {"comfyui_session_id", "comfyui_cond_key", "comfyui_cache_fp"}
)
_SESSIONS: dict[str, dict[str, Any]] = {}
_RUNS: dict[str, Any] = {}


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
        arch = getattr(vae_config, "arch_config", None)
        if arch is not None and getattr(arch, "latents_mean", None) is None:
            logger.info(
                "Skipping VAE post_init in comfyui_mode; checkpoint has no VAE stats"
            )
        else:
            vae_config.post_init()


def create_comfyui_pipeline_stages(
    pipeline: ComposedPipelineBase, server_args: ServerArgs
) -> None:
    if hasattr(pipeline, "create_comfyui_stages"):
        pipeline.create_comfyui_stages(server_args)
        return

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


def _session_family(session_id: str) -> str:
    return session_id.split(":", 1)[0]


def _evict_other_runs(session_id: str) -> None:
    """Drop older ``{uuid}:{run}`` slots for this executor."""
    prefix = _session_family(session_id) + ":"
    for store in (_SESSIONS, _RUNS):
        for key in list(store):
            if key.startswith(prefix) and key != session_id:
                store.pop(key, None)


def bind_comfyui_session(req):
    """Restore cached conditioning, then refresh the cache from whatever is set.

    Conditioning is stored per ``comfyui_cond_key`` so CFG pos/neg do not
    overwrite each other. Packed extras are restored only when
    ``comfyui_cache_fp`` matches; a mismatch must not revive a stale layout.
    """
    sid = session_id_from_req(req)
    if not sid:
        return req

    _evict_other_runs(sid)
    extra = dict(getattr(req, "extra", None) or {})
    cond_key = extra.get("comfyui_cond_key") or "_"
    cached = _SESSIONS.get(sid) or {}
    cond_snap = (cached.get("_conds") or {}).get(cond_key) or {}
    current_fp = extra.get("comfyui_cache_fp")
    cached_fp = cached.get("_fp")
    extra_ok = current_fp is None or cached_fp is None or current_fp == cached_fp

    for name, value in cond_snap.items():
        if _is_empty(getattr(req, name, None)):
            setattr(req, name, value)
    if extra_ok:
        for key, value in (cached.get("_extra") or {}).items():
            if _is_empty(extra.get(key)):
                extra[key] = value
    req.extra = extra

    new_cond = {}
    for name in _CONDITIONING_FIELDS:
        value = getattr(req, name, None)
        if not _is_empty(value):
            new_cond[name] = value
    extra = getattr(req, "extra", None) or {}
    extra_snapshot = {}
    for key, value in extra.items():
        if key in _SESSION_SKIP_EXTRA or _is_empty(value):
            continue
        extra_snapshot[key] = value
    conds = dict(cached.get("_conds") or {})
    if new_cond:
        conds[cond_key] = new_cond
    snapshot: dict[str, Any] = {"_conds": conds}
    if extra_snapshot:
        snapshot["_extra"] = extra_snapshot
    if current_fp is not None:
        snapshot["_fp"] = current_fp
    elif extra_ok and cached_fp is not None:
        snapshot["_fp"] = cached_fp
    if conds or extra_snapshot:
        _SESSIONS[sid] = snapshot
    return req


def get_run_state(req):
    sid = session_id_from_req(req)
    if sid:
        return _RUNS.get(sid)
    return None


def set_run_state(req, state: Any) -> None:
    sid = session_id_from_req(req)
    if sid:
        _RUNS[sid] = state


def pop_run_state(req):
    sid = session_id_from_req(req)
    if sid:
        return _RUNS.pop(sid, None)
    return None


def get_or_create_run_state(req, factory: Callable[[], Any]):
    """Session-scoped opaque state. Created once per run."""
    existing = get_run_state(req)
    if existing is not None:
        return existing
    state = factory()
    set_run_state(req, state)
    return state


def release_run_state(session_id: str | None) -> None:
    """Drop per-run objects; keep cached conditioning for the next step."""
    if session_id:
        _RUNS.pop(session_id, None)


def release_comfyui_session(session_id: str | None) -> None:
    if session_id:
        _SESSIONS.pop(session_id, None)
        _RUNS.pop(session_id, None)


def _is_empty(value) -> bool:
    if value is None:
        return True
    if isinstance(value, (list, tuple)):
        return len(value) == 0
    return False
