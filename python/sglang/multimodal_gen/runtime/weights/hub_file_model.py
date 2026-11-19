"""Serve one weight file from a Hub repo as ``--model-path``.

Community repos ship single-file transformers (fine-tunes, merges, GGUF and
quantized exports) and LoRAs without a pipeline around them. Given
``OWNER/REPO/path/FILE``, find the registered pipeline the file was made for
through the model card's ``base_model`` chain, and serve it as

    --model-path BASE --component-weights-paths.transformer OWNER/REPO/path/FILE
    --model-path BASE --lora-path OWNER/REPO/path/FILE
"""

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from huggingface_hub import HfApi

from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.runtime.weights.source import (
    WeightSource,
    is_explicit_weight_file_reference,
    parse_weight_source,
)
from sglang.srt.environ import envs

logger = init_logger(__name__)

# Card links followed from the file's repo to a registered pipeline. Community
# files often name another community repo (e.g. a ComfyUI repack) as the base.
_MAX_BASE_MODEL_HOPS = 3
_LORA_KEY_MARKERS = ("lora_", ".lora.", "lora.down", "lora.up", "lokr_", "hada_")


@dataclass(frozen=True)
class HubFileModel:
    weight: str
    base_model: str
    is_lora: bool
    model_variant: str | None


def parse_hub_file_model_path(model_path: Any) -> WeightSource | None:
    """The Hub weight file ``model_path`` names, or None for anything else."""
    if envs.SGLANG_USE_MODELSCOPE.get():
        return None
    if not isinstance(model_path, str) or os.path.exists(
        os.path.expanduser(model_path)
    ):
        return None
    if not is_explicit_weight_file_reference(model_path):
        return None
    try:
        source = parse_weight_source(model_path)
    except ValueError:
        return None
    if source.kind != "huggingface" or source.filename is None:
        return None
    return source


def _card_base_models(info: Any) -> list[str]:
    base_model = (getattr(info, "card_data", None) or {}).get("base_model")
    if isinstance(base_model, str):
        return [base_model]
    return [str(model) for model in base_model or ()]


def _is_specialization(model: str, base: str) -> bool:
    """Whether registered ``model`` is a variant of registered ``base``, such
    as a distilled release whose pipeline config subclasses the base's."""
    from sglang.multimodal_gen.registry import get_registered_pipeline_config_cls

    model_cls = get_registered_pipeline_config_cls(model)
    base_cls = get_registered_pipeline_config_cls(base)
    return (
        model_cls is not None
        and base_cls is not None
        and model_cls is not base_cls
        and issubclass(model_cls, base_cls)
    )


@lru_cache(maxsize=64)
def find_registered_base_model(
    repo_id: str, revision: str | None = None, filename: str | None = None
) -> str:
    """The registered pipeline a community repo was made for.

    Follows the model card's ``base_model`` links breadth first. The registry's
    model-name detectors, applied to the file's path, pick among several linked
    pipelines, narrow a linked pipeline to a registered variant of it (e.g. a
    FastH3 repack whose card names MiniMax-H3), and stand in for a missing card.
    Cached, as both CLI dispatch and ServerArgs resolve the same file.
    """
    from sglang.multimodal_gen.registry import (
        detect_registered_hf_model_path,
        get_registered_hf_model_path,
    )

    api = HfApi()
    found: set[str] = set()
    seen: set[str] = set()
    frontier = [repo_id]
    for _ in range(_MAX_BASE_MODEL_HOPS):
        next_frontier = []
        for current in frontier:
            if current.lower() in seen:
                continue
            seen.add(current.lower())
            info = api.model_info(
                current, revision=revision if current == repo_id else None
            )
            for base in _card_base_models(info):
                registered = get_registered_hf_model_path(base)
                if registered is not None:
                    found.add(registered)
                else:
                    next_frontier.append(base)
        if found or not next_frontier:
            break
        frontier = next_frontier

    detected = detect_registered_hf_model_path(
        f"{repo_id}/{filename}" if filename else repo_id
    )
    if detected is not None and (
        detected in found
        or (len(found) == 1 and _is_specialization(detected, next(iter(found))))
    ):
        return detected
    if len(found) == 1:
        return found.pop()
    if found:
        raise ValueError(
            f"The model card of {repo_id} links to several supported base models "
            f"({', '.join(sorted(found))}). Pass the base model as --model-path "
            "and the file as --component-weights-paths.transformer or --lora-path."
        )
    if detected is not None:
        logger.info(
            "%s has no supported base_model on its model card; using %s, "
            "which matches its name.",
            repo_id,
            detected,
        )
        return detected
    raise ValueError(
        f"Could not find the base model of {repo_id}: its model card names no "
        "supported base_model. Pass the base model as --model-path and the file "
        "as --component-weights-paths.transformer or --lora-path."
    )


# Root key prefixes of components that are not served as --model-path.
_TEXT_ENCODER_KEY_PREFIXES = (
    "lm_head.",
    "model.layers.",
    "model.embed_tokens.",
    "model.language_model.",
    "language_model.",
    "visual.",
    "model.visual.",
)


def _safetensors_file_kind(source: WeightSource) -> str:
    """ "lora", "text_encoder", "vae" or "transformer", from a Hub safetensors
    file's header only; other formats are taken as transformers."""
    filename = source.filename
    assert source.repo_id is not None and filename is not None
    if not filename.lower().endswith(".safetensors"):
        return "transformer"
    keys = list(
        HfApi()
        .parse_safetensors_file_metadata(
            source.repo_id, filename, revision=source.revision
        )
        .tensors
    )
    if any(marker in key.lower() for key in keys for marker in _LORA_KEY_MARKERS):
        return "lora"
    if any(key.startswith(_TEXT_ENCODER_KEY_PREFIXES) for key in keys):
        return "text_encoder"
    if any(key.startswith("encoder.") for key in keys) and any(
        key.startswith("decoder.") for key in keys
    ):
        return "vae"
    return "transformer"


def resolve_hub_file_model(model_path: Any) -> HubFileModel | None:
    """Resolve ``OWNER/REPO/path/FILE`` to its base pipeline, or None when
    ``model_path`` is not a Hub weight file."""
    source = parse_hub_file_model_path(model_path)
    if source is None:
        return None
    from sglang.multimodal_gen.registry import get_registered_pipeline_config_cls

    assert source.repo_id is not None and source.filename is not None
    base_model = find_registered_base_model(
        source.repo_id, source.revision, source.filename
    )
    kind = _safetensors_file_kind(source)
    if kind in ("text_encoder", "vae"):
        raise ValueError(
            f"{model_path} holds {kind} weights, not a transformer or LoRA. Serve "
            f"the base model {base_model} and pass this file with "
            f"--component-weights-paths.<{kind} component>."
        )
    pipeline_config_cls = get_registered_pipeline_config_cls(base_model)
    return HubFileModel(
        weight=model_path,
        base_model=base_model,
        is_lora=kind == "lora",
        model_variant=(
            pipeline_config_cls.model_variant_for_weight_file(source.filename)
            if pipeline_config_cls is not None
            else None
        ),
    )


def apply_hub_file_model_path(kwargs: dict[str, Any], explicit: set[str]) -> None:
    """Rewrite server kwargs whose ``model_path`` is a Hub weight file into the
    base model plus a transformer override or a LoRA."""
    resolved = resolve_hub_file_model(kwargs.get("model_path"))
    if resolved is None:
        return

    if resolved.is_lora:
        if kwargs.get("lora_path"):
            raise ValueError(
                f"--model-path {resolved.weight} is a LoRA, and --lora-path is "
                "also set; pass the base model as --model-path instead."
            )
        kwargs["lora_path"] = resolved.weight
        explicit.add("lora_path")
        flag = f"--lora-path {resolved.weight}"
    else:
        component_weights_paths = dict(kwargs.get("component_weights_paths") or {})
        if "transformer" in component_weights_paths or kwargs.get(
            "transformer_weights_path"
        ):
            raise ValueError(
                f"--model-path {resolved.weight} is a transformer file, and a "
                "transformer weight override is also set; pass the base model "
                "as --model-path instead."
            )
        component_weights_paths["transformer"] = resolved.weight
        kwargs["component_weights_paths"] = component_weights_paths
        explicit.add("component_weights_paths")
        flag = f"--component-weights-paths.transformer {resolved.weight}"

    if resolved.model_variant is not None and not kwargs.get("model_variant"):
        kwargs["model_variant"] = resolved.model_variant
        explicit.add("model_variant")
        flag += f" --model-variant {resolved.model_variant}"
    if not kwargs.get("served_model_name"):
        kwargs["served_model_name"] = resolved.weight
        explicit.add("served_model_name")
    kwargs["model_path"] = resolved.base_model
    explicit.add("model_path")
    logger.info(
        "Serving %s on its base model, as --model-path %s %s",
        resolved.weight,
        resolved.base_model,
        flag,
    )
