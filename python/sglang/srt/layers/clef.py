"""Cached prompt states and asynchronous results for Clef joint decisions."""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Callable, Sequence
from functools import lru_cache
from typing import Any

import msgspec
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
    """Enforce the supported single-GPU storage and execution modes."""
    required = {
        "tp_size": 1,
        "pp_size": 1,
        "dp_size": 1,
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
    graph_config = getattr(args, "cuda_graph_config", None)
    prefill_backend = (
        graph_config.prefill.backend
        if graph_config is not None
        else getattr(args, "cuda_graph_backend_prefill", None)
    )
    if prefill_backend not in (None, "disabled"):
        errors.append("cuda_graph_backend_prefill='disabled'")
    for feature in (
        "enable_torch_compile",
        "enable_unified_memory",
        "enable_hierarchical_cache",
        "enable_unified_cache_external_linker",
        "enable_hisparse",
        "enable_lmcache",
        "enable_flexkv",
        "enable_memory_saver",
        "enable_two_batch_overlap",
        "enable_single_batch_overlap",
        "prefill_only_disable_kv_cache",
    ):
        if getattr(args, feature, False):
            errors.append(f"{feature}=False")
    if (
        getattr(args, "cpu_offload_gb", 0) > 0
        or getattr(args, "offload_group_size", -1) > 0
    ):
        errors.append("CPU weight offload disabled")
    if getattr(args, "radix_cache_backend", None) in ("flexkv", "lmcache"):
        errors.append("standard radix cache backend")
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


def normalized_probabilities(values: list[float]) -> list[float]:
    """Normalize the native FP32 softmax values after their asynchronous copy."""
    total = math.fsum(values)
    if not math.isfinite(total) or total <= 0:
        raise ValueError("Clef head produced non-finite probabilities")
    return [value / total for value in values]


class ClefResult(msgspec.Struct, frozen=True):
    batch_index: int
    request_handle: Any
    record: EncodedRecord


class ClefDeviceOutput(msgspec.Struct, frozen=True):
    probabilities: torch.Tensor
    results: tuple[ClefResult, ...]

    def copy_to_host(
        self, copy_tensor: Callable[[torch.Tensor], torch.Tensor]
    ) -> ClefHostOutput:
        return ClefHostOutput(copy_tensor(self.probabilities), self.results)


class ClefHostOutput(msgspec.Struct, frozen=True):
    probabilities: torch.Tensor
    results: tuple[ClefResult, ...]

    def consume(self, batch: Any, commits: Sequence[Any]) -> None:
        from sglang.srt.managers.schedule_batch import FINISH_ABORT

        values = self.probabilities.tolist()
        offset = 0
        for result in self.results:
            probabilities = {}
            for question in result.record.questions:
                count = len(question.option_ids)
                question_values = values[offset : offset + count]
                offset += count
                probabilities[question.question_id] = dict(
                    zip(question.option_ids, normalized_probabilities(question_values))
                )
            req = batch.reqs[result.batch_index]
            if (
                req.cache_request_handle != result.request_handle
                or req.is_retracted
                or isinstance(req.finished_reason, FINISH_ABORT)
                or isinstance(req.to_finish, FINISH_ABORT)
                or not req.finished()
            ):
                continue
            if req.customized_info is None:
                req.customized_info = {}
            req.customized_info["clef_probabilities"] = [probabilities]


def forward_clef(
    head: JointSchemaHead,
    lm_head: torch.nn.Module,
    input_ids: torch.Tensor,
    hidden_states: torch.Tensor,
    forward_batch: Any,
) -> ClefDeviceOutput | None:
    """Retain every backbone row and run the native head on complete records."""
    from sglang.srt.model_executor.forward_context import (
        get_req_to_token_pool,
        get_token_to_kv_pool,
    )

    pool = get_token_to_kv_pool()
    # Ordinary generation can seed a prefix later reused by a decision request.
    pool.store_clef_hidden(forward_batch.out_cache_loc, hidden_states)
    if not forward_batch.forward_mode.is_extend():
        return None
    records = getattr(forward_batch.sampling_info, "clef_records", None)
    if not records:
        return None
    results = []
    probabilities = []
    for index, item in enumerate(records):
        if item is None:
            continue
        serialized, handle = item
        record = _RECORD_ADAPTER.validate_json(serialized)
        sequence_length = len(record.input_ids)
        prefix_length = forward_batch.extend_prefix_lens_cpu[index]
        if prefix_length + forward_batch.extend_seq_lens_cpu[index] != sequence_length:
            continue
        locations = get_req_to_token_pool().req_to_token[
            forward_batch.req_pool_indices_cpu[index], :sequence_length
        ]
        sequence_hidden = pool.gather_clef_hidden(locations)
        token_ids = torch.tensor(
            record.input_ids, dtype=input_ids.dtype, device=input_ids.device
        )
        logits = head(
            sequence_hidden.unsqueeze(0),
            token_ids.unsqueeze(0),
            torch.ones_like(token_ids).unsqueeze(0),
            [record],
            lm_head.weight,
            sequence_lengths=[sequence_length],
        )[0]
        probabilities.extend(values.float().softmax(-1) for values in logits)
        results.append(ClefResult(index, handle, record))
    if not results:
        return None
    return ClefDeviceOutput(torch.cat(probabilities), tuple(results))
