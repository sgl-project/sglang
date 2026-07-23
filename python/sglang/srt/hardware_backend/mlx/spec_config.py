"""Configuration and request guardrails for MLX Frozen-KV MTP.

This module intentionally has no import-time dependency on MLX.  Server
argument normalization runs before the MLX worker is constructed and must
remain safe on non-Apple hosts running generic speculative tests.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)

MLX_GEMMA4_MTP_MAX_CONTEXT = 2048
MLX_GEMMA4_MTP_VERIFY_WIDTH = 2
MLX_GEMMA4_MTP_TARGET_ARCH = "Gemma4ForConditionalGeneration"
MLX_GEMMA4_MTP_ASSISTANT_ARCH = "Gemma4AssistantForCausalLM"


def _read_json(path: Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as file:
            value = json.load(file)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read Gemma 4 assistant config {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Gemma 4 assistant config {path} must contain a JSON object")
    return value


def load_assistant_config_dict(
    path_or_repo: str,
    *,
    revision: Optional[str] = None,
    configuration_file: Optional[str] = None,
) -> dict[str, Any]:
    """Load only assistant JSON metadata, never model code or weights."""

    if not path_or_repo:
        raise ValueError("MLX Gemma 4 MTP requires --speculative-draft-model-path.")

    config_name = configuration_file or "config.json"
    local = Path(path_or_repo).expanduser()
    if local.is_file():
        return _read_json(local)
    if local.is_dir():
        return _read_json(local / config_name)

    try:
        from huggingface_hub import hf_hub_download

        config_path = hf_hub_download(
            repo_id=path_or_repo,
            filename=config_name,
            revision=revision,
        )
    except Exception as exc:
        raise ValueError(
            "Cannot resolve Gemma 4 assistant config for "
            f"{path_or_repo!r} at revision {revision!r}: {exc}"
        ) from exc
    return _read_json(Path(config_path))


def is_gemma4_assistant_config(config: dict[str, Any]) -> bool:
    architectures = config.get("architectures") or []
    return config.get("model_type") == "gemma4_assistant" or (
        MLX_GEMMA4_MTP_ASSISTANT_ARCH in architectures
    )


def _attr(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _validate_target_config(server_args: ServerArgs) -> None:
    model_config = server_args.get_model_config()
    hf_config = model_config.hf_config
    text_config = _attr(hf_config, "text_config")
    architectures = _attr(hf_config, "architectures", []) or []

    if (
        _attr(hf_config, "model_type") != "gemma4"
        or MLX_GEMMA4_MTP_TARGET_ARCH not in architectures
        or text_config is None
    ):
        raise ValueError(
            "MLX Frozen-KV MTP currently supports only the Gemma 4 E2B text "
            f"target ({MLX_GEMMA4_MTP_TARGET_ARCH})."
        )

    e2b_shape = (
        _attr(text_config, "model_type") == "gemma4_text"
        and int(_attr(text_config, "hidden_size", -1)) == 1536
        and int(_attr(text_config, "num_hidden_layers", -1)) == 35
        and int(_attr(text_config, "vocab_size", -1)) == 262144
    )
    if not e2b_shape:
        raise ValueError(
            "MLX Frozen-KV MTP MVP supports only the Gemma 4 E2B target "
            "(hidden_size=1536, num_hidden_layers=35, vocab_size=262144)."
        )

    if bool(_attr(model_config, "is_multimodal", False)) and not bool(
        getattr(server_args, "language_only", True)
    ):
        raise ValueError(
            "MLX Frozen-KV MTP is text-only and does not support multimodal "
            "Gemma 4 execution."
        )


def _validate_assistant_config(server_args: ServerArgs, config: dict[str, Any]) -> None:
    if config.get("model_type") != "gemma4_assistant":
        raise ValueError(
            "MLX Frozen-KV MTP requires assistant model_type='gemma4_assistant'."
        )
    architectures = config.get("architectures") or []
    if architectures != [MLX_GEMMA4_MTP_ASSISTANT_ARCH]:
        raise ValueError(
            "MLX Frozen-KV MTP requires architecture "
            f"{MLX_GEMMA4_MTP_ASSISTANT_ARCH!r}; got {architectures!r}."
        )
    if any(key in config for key in ("auto_map", "model_file")):
        raise ValueError(
            "MLX Gemma 4 MTP does not load remote/custom assistant model code."
        )
    if int(config.get("backbone_hidden_size", -1)) != 1536:
        raise ValueError("Gemma 4 E2B assistant backbone_hidden_size must equal 1536.")
    text = config.get("text_config") or {}
    if (
        text.get("model_type") != "gemma4_text"
        or int(text.get("hidden_size", -1)) != 256
        or int(text.get("num_hidden_layers", -1)) != 4
        or int(text.get("vocab_size", -1)) != 262144
    ):
        raise ValueError(
            "MLX Frozen-KV MTP requires the matching four-layer E2B assistant."
        )
    layer_types = list(text.get("layer_types") or [])
    if layer_types != [
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "full_attention",
    ]:
        raise ValueError(
            "Gemma 4 E2B assistant must have three sliding-attention layers "
            "followed by one full-attention layer."
        )
    if not bool(config.get("use_ordered_embeddings")):
        raise ValueError("Gemma 4 E2B assistant requires ordered embeddings.")
    if int(config.get("num_centroids", -1)) != 2048:
        raise ValueError("Gemma 4 E2B assistant num_centroids must equal 2048.")
    if int(config.get("centroid_intermediate_top_k", -1)) != 32:
        raise ValueError(
            "Gemma 4 E2B assistant centroid_intermediate_top_k must equal 32."
        )
    if 262144 % int(config["num_centroids"]) != 0:
        raise ValueError("Assistant vocabulary must be divisible by num_centroids.")


def validate_mlx_frozen_kv_mtp_args(server_args: ServerArgs) -> None:
    """Validate the exact Level-1 server shape before assistant weights load."""

    if getattr(server_args, "speculative_draft_model_path", None) is None:
        raise ValueError("MLX Frozen-KV MTP requires --speculative-draft-model-path.")
    if int(getattr(server_args, "speculative_eagle_topk", -1)) != 1:
        raise ValueError("MLX Frozen-KV MTP requires speculative_eagle_topk/topk=1.")
    if int(getattr(server_args, "speculative_num_steps", -1)) != 1:
        raise ValueError(
            "MLX Frozen-KV MTP requires speculative_num_steps/num_steps=1."
        )
    if int(getattr(server_args, "speculative_num_draft_tokens", -1)) != 2:
        raise ValueError(
            "MLX Frozen-KV MTP requires speculative_num_draft_tokens/draft_tokens=2 "
            "(root query plus one proposal)."
        )
    if bool(getattr(server_args, "speculative_use_rejection_sampling", False)):
        raise ValueError("MLX Frozen-KV MTP does not support rejection sampling.")
    if not bool(getattr(server_args, "disable_overlap_schedule", False)):
        raise ValueError("MLX Frozen-KV MTP requires synchronous scheduling.")
    if not bool(getattr(server_args, "disable_radix_cache", False)):
        raise NotImplementedError("MLX Frozen-KV MTP requires --disable-radix-cache.")
    if int(getattr(server_args, "chunked_prefill_size", 0)) != -1:
        raise NotImplementedError(
            "MLX Frozen-KV MTP requires --chunked-prefill-size -1."
        )
    if bool(getattr(server_args, "enable_mixed_chunk", False)):
        raise NotImplementedError(
            "MLX Frozen-KV MTP does not support mixed chunked prefill."
        )
    if int(getattr(server_args, "max_running_requests", 0)) != 1:
        raise NotImplementedError(
            "MLX Frozen-KV MTP requires --max-running-requests 1."
        )

    context_length = getattr(server_args, "context_length", None)
    if context_length is None:
        context_length = server_args.get_model_config().context_len
    if int(context_length) > MLX_GEMMA4_MTP_MAX_CONTEXT:
        raise NotImplementedError(
            "MLX Frozen-KV MTP supports a context length no greater than 2,048."
        )
    max_total_tokens = getattr(server_args, "max_total_tokens", None)
    if max_total_tokens is not None and int(max_total_tokens) > 2048:
        raise NotImplementedError(
            "MLX Frozen-KV MTP requires --max-total-tokens no greater than 2,048."
        )

    for field, flag in (
        ("tp_size", "tp-size"),
        ("dp_size", "dp-size"),
        ("pp_size", "pp-size"),
    ):
        if int(getattr(server_args, field, 1)) != 1:
            raise NotImplementedError(f"MLX Frozen-KV MTP requires --{flag} 1.")
    if int(getattr(server_args, "nnodes", 1)) != 1:
        raise NotImplementedError("MLX Frozen-KV MTP supports one Apple host only.")
    if bool(getattr(server_args, "enable_dp_attention", False)):
        raise NotImplementedError("MLX Frozen-KV MTP does not support DP attention.")
    disagg = str(getattr(server_args, "disaggregation_mode", "null")).lower()
    if disagg not in ("null", "none") and not disagg.endswith(".null"):
        raise NotImplementedError("MLX Frozen-KV MTP does not support disaggregation.")

    _validate_target_config(server_args)
    assistant_config = load_assistant_config_dict(
        server_args.speculative_draft_model_path,
        revision=getattr(server_args, "speculative_draft_model_revision", None),
    )
    _validate_assistant_config(server_args, assistant_config)
    setattr(server_args, "_mlx_gemma4_mtp_assistant_config", assistant_config)


def validate_mlx_frozen_kv_mtp_request(
    req: Req, *, has_multimodal: bool = False
) -> Optional[str]:
    """Return an admission error for request features outside the MVP."""

    params = req.sampling_params
    if int(params.top_k) != 1:
        return (
            "MLX Frozen-KV MTP requires greedy requests with temperature=0 "
            "(normalized top_k must equal 1)."
        )
    if (
        float(params.frequency_penalty) != 0.0
        or float(params.presence_penalty) != 0.0
        or float(params.repetition_penalty) != 1.0
        or int(params.min_new_tokens) != 0
    ):
        return "MLX Frozen-KV MTP does not support sampling penalties."
    if params.logit_bias is not None:
        return "MLX Frozen-KV MTP does not support logit bias."
    if any(
        bool(value)
        for value in (
            params.json_schema,
            params.regex,
            params.ebnf,
            params.structural_tag,
            getattr(params, "stop_regex_strs", None),
        )
    ):
        return "MLX Frozen-KV MTP does not support grammar-constrained decoding."
    if int(params.n) != 1:
        return "MLX Frozen-KV MTP supports one completion per request."
    if getattr(params, "custom_params", None):
        return "MLX Frozen-KV MTP does not support custom sampling parameters."
    if bool(getattr(req, "return_logprob", False)):
        return "MLX Frozen-KV MTP does not support logprobs."
    if bool(getattr(req, "return_hidden_states", False)):
        return "MLX Frozen-KV MTP does not return target hidden states."
    if bool(getattr(req, "return_sampling_mask", False)):
        return "MLX Frozen-KV MTP does not return sampling masks."
    if getattr(req, "custom_logit_processor", None) is not None:
        return "MLX Frozen-KV MTP does not support custom logits processors."
    if (
        getattr(req, "session", None) is not None
        or getattr(req, "session_id", None) is not None
    ):
        return "MLX Frozen-KV MTP does not support sessions."
    if getattr(req, "lora_id", None) is not None:
        return "MLX Frozen-KV MTP does not support LoRA requests."
    if has_multimodal or getattr(req, "multimodal_inputs", None) is not None:
        return "MLX Frozen-KV MTP is text-only."
    return None
