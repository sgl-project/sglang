"""Restricted, prefill-only integration of the trained Clef joint decision head."""

from __future__ import annotations

import json
import logging
import math
from functools import lru_cache
from typing import Any

import torch
from pydantic import BaseModel, ConfigDict, PositiveInt, TypeAdapter
from safetensors.torch import load_file
from transformers.utils.hub import cached_file

from sglang.srt.environ import envs
from sglang.srt.layers.clef_reference import EncodedRecord, JointSchemaHead

logger = logging.getLogger(__name__)
_RECORD_ADAPTER = TypeAdapter(EncodedRecord)


class ClefHeadConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hidden_size: PositiveInt
    width: PositiveInt
    routing_layers: PositiveInt
    layers: PositiveInt
    heads: PositiveInt
    feedforward: PositiveInt
    dropout: float = 0.0


@lru_cache
def load_clef_config(model_path: str, revision: str | None) -> dict | None:
    """Detect the data-only Clef release format without executing checkpoint code."""
    path = cached_file(
        model_path,
        "joint_head_config.json",
        revision=revision,
        _raise_exceptions_for_missing_entries=False,
    )
    if path is None:
        return None
    with open(path) as config_file:
        config = ClefHeadConfig.model_validate(json.load(config_file))
    if config.width % config.heads or config.dropout != 0:
        raise ValueError("Clef requires width divisible by heads and dropout=0")
    # Presence of the config promises a head: a missing weights file is fatal.
    cached_file(model_path, "joint_head.safetensors", revision=revision)
    return config.model_dump()


def validate_clef_settings(args: Any, dtype: torch.dtype, quant_config: Any) -> None:
    """Reject engine settings that do not preserve the complete joint prompt."""
    required = {
        "tp_size": 1,
        "pp_size": 1,
        "dp_size": 1,
        "max_running_requests": 1,
        "disable_radix_cache": True,
        "disable_cuda_graph": True,
        "disable_overlap_schedule": True,
        "allow_auto_truncate": False,
    }
    errors = [
        f"{key}={value!r}"
        for key, value in required.items()
        if getattr(args, key, None) != value
    ]
    if envs.SGLANG_RUST_SERVER.get():
        errors.append("Python HTTP server (SGLANG_RUST_SERVER=0)")
    if getattr(args, "enable_lora", None) or getattr(args, "lora_paths", None):
        errors.append("LoRA disabled with no lora_paths")
    if getattr(args, "chunked_prefill_size", None) not in (-1, None):
        errors.append("chunked_prefill_size=-1")
    for feature in ("enable_torch_compile", "enable_dynamic_chunking"):
        if getattr(args, feature, False):
            errors.append(f"{feature}=False")
    if getattr(args, "disaggregation_mode", "null") != "null":
        errors.append("disaggregation_mode='null'")
    if getattr(args, "speculative_algorithm", None) is not None:
        errors.append("speculative_algorithm=None")
    # With unquantized BF16 model weights, the engine resolves auto to BF16.
    if getattr(args, "kv_cache_dtype", "auto") not in ("auto", "bf16", "bfloat16"):
        errors.append("kv_cache_dtype='auto', 'bf16', or 'bfloat16'")
    if dtype != torch.bfloat16 or quant_config is not None:
        errors.append("unquantized bfloat16 weights")
    if errors:
        raise ValueError("Clef joint head currently requires: " + ", ".join(errors))


def load_clef_head(
    lm_head: torch.nn.Module | None, quant_config: Any
) -> JointSchemaHead | None:
    """Strictly load auxiliary trained weights on the model worker's device."""
    from sglang.srt.arg_groups.model_override_base import resolving_view
    from sglang.srt.runtime_context import get_model, get_server_args

    model = get_model()
    config = load_clef_config(model.model_path, model.revision)
    if config is None:
        return None
    weight = getattr(lm_head, "weight", None)
    if not isinstance(weight, torch.Tensor):
        raise ValueError(
            "Clef requires a decoder LM head on one GPU; encoder-only and pipeline shards are unsupported"
        )
    if weight.device.type != "cuda":
        raise ValueError("Clef joint head requires CUDA-resident backbone weights")
    validate_clef_settings(
        resolving_view(get_server_args()), weight.dtype, quant_config
    )
    if config["hidden_size"] != weight.shape[1]:
        raise ValueError("Clef head hidden_size does not match the backbone")
    head = JointSchemaHead(**config).to(device=weight.device, dtype=weight.dtype)
    path = cached_file(
        model.model_path, "joint_head.safetensors", revision=model.revision
    )
    head.load_state_dict(load_file(path), strict=True)
    head.eval()
    logger.info("Loaded Clef trained joint head strictly from %s", path)
    return head


def validate_record(data: Any, input_ids: list[int]) -> EncodedRecord:
    """Check all span metadata before it can reach the GPU model worker."""
    record = (
        _RECORD_ADAPTER.validate_json(data)
        if isinstance(data, str)
        else _RECORD_ADAPTER.validate_python(data)
    )
    if record.media is not None or tuple(input_ids) != record.input_ids:
        raise ValueError("Clef requires text-only metadata matching every input token")
    if not record.questions:
        raise ValueError("Clef requires at least one question")
    seen = set()
    last_span_end = 0
    for question in record.questions:
        if question.question_id in seen or question.question_type not in (0, 1, 2):
            raise ValueError("Invalid Clef question id or type")
        seen.add(question.question_id)
        if (
            not question.option_ids
            or len(set(question.option_ids)) != len(question.option_ids)
            or len(question.option_spans) != len(question.option_ids)
        ):
            raise ValueError("Invalid Clef option ids or spans")
        for start, end in (question.question_span, *question.option_spans):
            if not last_span_end <= start < end <= len(input_ids):
                raise ValueError(
                    "Clef span overlaps or lies outside the complete prompt"
                )
            last_span_end = end
    return record


def validate_clef_request(request: Any, model_config: Any) -> None:
    """Validate internal metadata even when supplied through public /generate."""
    params = request.sampling_params
    if isinstance(params, list):
        if any("clef_record" in (item.get("custom_params") or {}) for item in params):
            raise ValueError(
                "Clef metadata does not support batched /generate requests"
            )
        return
    if "clef_record" not in (params.get("custom_params") or {}):
        return
    # ponytail: one complete text prefill; extend this contract with batching.
    unsupported = (
        "image_data",
        "audio_data",
        "video_data",
        "input_embeds",
        "embed_overrides",
        "positional_embed_overrides",
        "return_logprob",
        "return_hidden_states",
        "return_sampling_mask",
        "return_entropy",
        "return_routed_experts",
        "return_indexer_topk",
        "stream",
        "session_params",
        "lora_path",
        "custom_logit_processor",
        "token_indices_to_pool",
        "multi_item_delimiter_indices",
    )
    if (
        not hasattr(request, "return_logprob")
        or not isinstance(params["custom_params"]["clef_record"], str)
        or getattr(model_config, "clef_config", None) is None
        or params.get("max_new_tokens") != 0
        or params.get("n", 1) != 1
        or set(params) - {"max_new_tokens", "temperature", "custom_params", "n"}
        or not isinstance(request.input_ids, list)
        or not all(
            type(token) is int and 0 <= token < model_config.vocab_size
            for token in request.input_ids
        )
        or any(getattr(request, key, None) for key in unsupported)
    ):
        raise ValueError(
            "Clef metadata requires one text-only prefill without optional generation features"
        )
    validate_record(params["custom_params"]["clef_record"], request.input_ids)


def normalized_probabilities(logits: torch.Tensor) -> list[float]:
    """Retain native FP32 softmax, then normalize its serialized double values."""
    values = logits.float().softmax(-1).tolist()
    total = math.fsum(values)
    if not math.isfinite(total) or total <= 0:
        raise ValueError("Clef head produced non-finite probabilities")
    return [value / total for value in values]


def forward_clef(
    model: Any, input_ids: torch.Tensor, hidden_states: torch.Tensor, forward_batch: Any
) -> Any:
    """Execute the full trained head over final SGLang backbone states on GPU."""
    from sglang.srt.layers.logits_processor import LogitsProcessorOutput

    records = getattr(forward_batch.sampling_info, "clef_records", None)
    if not records or not any(record is not None for record in records):
        return None  # Ordinary warmup/generation remains a backbone request.
    if (
        len(records) != 1
        or records[0] is None
        or not forward_batch.is_prefill_only
        or not forward_batch.forward_mode.is_extend()
    ):
        raise ValueError("Clef requires one complete prefill-only request")
    record = validate_record(records[0], input_ids.tolist())
    if hidden_states.shape[0] != len(record.input_ids):
        raise ValueError("Clef did not receive every final prompt hidden state")
    logits = model.clef_head(
        hidden_states.unsqueeze(0),
        input_ids.unsqueeze(0),
        torch.ones_like(input_ids).unsqueeze(0),
        [record],
        model.lm_head.weight,
    )[0]
    probabilities = {
        question.question_id: dict(
            zip(question.option_ids, normalized_probabilities(values))
        )
        for question, values in zip(record.questions, logits)
    }
    model.clef_forward_count += 1
    return LogitsProcessorOutput(
        next_token_logits=None,
        customized_info={
            "clef_probabilities": [probabilities],
            "clef_execution": [
                {
                    "path": "sglang_backbone_joint_head",
                    "forward_count": model.clef_forward_count,
                }
            ],
        },
    )
