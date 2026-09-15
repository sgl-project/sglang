"""Model-agnostic draft embedding for speculative decoding under pipeline parallelism.

The draft runs only on the last stage while the target embedding lives only on the
first stage, so the draft cannot share ``target.get_embed_and_head()``. This module
reads the target embed/head PP-safely and loads the checkpoint's input embedding
into the draft's own ``embed_tokens``, touching only the shard that holds it.
"""

from __future__ import annotations

import glob
import json
import logging
import os
from typing import Iterable, List, Optional, Tuple

import torch
from torch import nn

from sglang.srt.layers.utils.common import PPMissingLayer
from sglang.srt.model_loader.weight_utils import (
    default_weight_loader,
    download_weights_from_hf,
)

logger = logging.getLogger(__name__)

SAFETENSORS_INDEX_NAME = "model.safetensors.index.json"

# Checkpoint keys that hold the input embedding, most common first. A key that
# merely ends with ``embed_tokens.weight`` is accepted as a fallback as long as
# it does not belong to a draft/MTP sub-module.
_EMBED_KEY_CANDIDATES: Tuple[str, ...] = (
    "model.embed_tokens.weight",
    "model.language_model.embed_tokens.weight",
    "language_model.model.embed_tokens.weight",
    "embed_tokens.weight",
)
_EMBED_KEY_SUFFIX = "embed_tokens.weight"
_DRAFT_SUBMODULE_MARKERS: Tuple[str, ...] = ("mtp", "nextn", "eagle", "draft")


def _weight_or_none(module: Optional[nn.Module]) -> Optional[torch.Tensor]:
    if module is None or isinstance(module, PPMissingLayer):
        return None
    from sglang.srt.lora.layers import unwrap_lora_layer

    return unwrap_lora_layer(module).weight


def resolve_target_embed_and_head(
    target_model: nn.Module, *, is_first_pp_rank: bool
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """PP-safe ``target_model.get_embed_and_head()``.

    Most targets implement the getter as ``self.model.embed_tokens.weight,
    self.lm_head.weight``. On a non-first stage ``embed_tokens`` is a
    ``PPMissingLayer`` and ``.weight`` raises ``AttributeError``. Treat that as
    "this stage does not own the embedding" and read the head directly.
    """
    try:
        return target_model.get_embed_and_head()
    except AttributeError:
        if is_first_pp_rank:
            raise
        return None, _weight_or_none(target_model.lm_head)


def find_draft_embedding_param(
    draft_model: nn.Module,
) -> Optional[Tuple[str, nn.Parameter]]:
    """Return the draft model's own input-embedding parameter, if it has one."""
    matches = [
        (name, param)
        for name, param in draft_model.named_parameters(remove_duplicate=False)
        if name.endswith(_EMBED_KEY_SUFFIX)
    ]
    if not matches:
        return None
    # Prefer the shallowest match (``model.embed_tokens.weight`` over a nested one).
    matches.sort(key=lambda kv: kv[0].count("."))
    return matches[0]


def _is_input_embedding_key(key: str) -> bool:
    if not key.endswith(_EMBED_KEY_SUFFIX):
        return False
    lowered = key.lower()
    return not any(marker in lowered for marker in _DRAFT_SUBMODULE_MARKERS)


def _pick_embedding_key(keys: Iterable[str]) -> Optional[str]:
    keys = list(keys)
    for candidate in _EMBED_KEY_CANDIDATES:
        if candidate in keys:
            return candidate
    fallback = sorted((k for k in keys if _is_input_embedding_key(k)), key=len)
    return fallback[0] if fallback else None


def _safetensors_keys(path: str) -> List[str]:
    import safetensors

    with safetensors.safe_open(path, framework="pt", device="cpu") as f:
        return list(f.keys())


def locate_embedding_tensor(
    model_path: str,
    *,
    revision: Optional[str] = None,
    download_dir: Optional[str] = None,
) -> Tuple[str, str]:
    """Return ``(shard_path, key)`` for the checkpoint's input embedding.

    Uses ``model.safetensors.index.json`` when present so only one shard is
    touched (and, for Hub repos, only one shard is downloaded). Falls back to
    scanning shard headers when there is no index.
    """
    local_dir = download_weights_from_hf(
        model_path,
        cache_dir=download_dir,
        allow_patterns=[SAFETENSORS_INDEX_NAME, "*.safetensors"]
        if os.path.isdir(model_path)
        else [SAFETENSORS_INDEX_NAME],
        revision=revision,
    )
    index_file = os.path.join(local_dir, SAFETENSORS_INDEX_NAME)
    if os.path.exists(index_file):
        with open(index_file) as f:
            weight_map = json.load(f).get("weight_map", {}) or {}
        key = _pick_embedding_key(weight_map.keys())
        if key is None:
            raise ValueError(
                f"No input embedding key found in {index_file}; looked for "
                f"{_EMBED_KEY_CANDIDATES} or '*{_EMBED_KEY_SUFFIX}'."
            )
        shard_name = weight_map[key]
        if not os.path.isdir(model_path):
            local_dir = download_weights_from_hf(
                model_path,
                cache_dir=download_dir,
                allow_patterns=[shard_name],
                revision=revision,
            )
        return os.path.join(local_dir, shard_name), key

    if not os.path.isdir(model_path):
        local_dir = download_weights_from_hf(
            model_path,
            cache_dir=download_dir,
            allow_patterns=["*.safetensors"],
            revision=revision,
        )
    shards = sorted(glob.glob(os.path.join(local_dir, "*.safetensors")))
    if not shards:
        raise FileNotFoundError(
            f"No safetensors shards under {local_dir}; cannot load the draft "
            "embedding for pipeline-parallel speculative decoding."
        )
    for shard in shards:
        key = _pick_embedding_key(_safetensors_keys(shard))
        if key is not None:
            return shard, key
    raise ValueError(
        f"No input embedding key found in any shard under {local_dir}; looked "
        f"for {_EMBED_KEY_CANDIDATES} or '*{_EMBED_KEY_SUFFIX}'."
    )


def load_draft_embedding_from_checkpoint(
    draft_model: nn.Module,
    model_path: str,
    *,
    revision: Optional[str] = None,
    download_dir: Optional[str] = None,
) -> nn.Parameter:
    """Load the target checkpoint's input embedding into the draft's own parameter.

    Returns the (now initialized) draft parameter so callers can hand it to the
    draft's ``set_embed_and_head`` exactly like a shared target embedding.
    """
    import safetensors

    found = find_draft_embedding_param(draft_model)
    if found is None:
        raise ValueError(
            f"Draft model {draft_model.__class__.__name__} has no "
            f"'*{_EMBED_KEY_SUFFIX}' parameter, and the target cannot share its "
            "embedding from this pipeline stage. The draft must own an input "
            "embedding to run under pipeline parallelism "
            "(https://github.com/sgl-project/sglang/issues/39634)."
        )
    param_name, param = found

    shard_path, key = locate_embedding_tensor(
        model_path, revision=revision, download_dir=download_dir
    )
    with safetensors.safe_open(shard_path, framework="pt", device="cpu") as f:
        loaded = f.get_tensor(key)

    weight_loader = getattr(param, "weight_loader", default_weight_loader)
    weight_loader(param, loaded)
    logger.info(
        "Loaded draft embedding %s from %s:%s (%s) for pipeline-parallel "
        "speculative decoding.",
        param_name,
        os.path.basename(shard_path),
        key,
        tuple(loaded.shape),
    )
    return param
