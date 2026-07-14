# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import dataclasses
from copy import deepcopy
from enum import Enum
from typing import Any

from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req


def freeze_signature_value(value: Any):
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {
            str(key): freeze_signature_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return tuple(freeze_signature_value(item) for item in value)
    return repr(value)


def build_dynamic_batch_signature(req: Req) -> tuple[Any, ...] | None:
    sampling_params = req.sampling_params
    if sampling_params is None:
        return None
    try:
        sampling_fields = dataclasses.fields(sampling_params)
    except TypeError:
        return None

    items = [
        (
            field.name,
            freeze_signature_value(getattr(sampling_params, field.name, None)),
        )
        for field in sampling_fields
        if not field.metadata.get("batch_sig_exclude", False)
    ]
    diffusers_kwargs = (req.extra or {}).get("diffusers_kwargs")
    if diffusers_kwargs:
        items.append(
            ("diffusers_kwargs", freeze_signature_value(diffusers_kwargs))
        )
    return tuple(items)


def get_dynamic_batch_signature(req: Req) -> tuple[Any, ...] | None:
    cached = getattr(req, "_dynamic_batch_sig", None)
    if cached is not None:
        return cached
    signature = build_dynamic_batch_signature(req)
    req._dynamic_batch_sig = signature
    return signature


def can_dynamic_batch(base_req: Req, candidate_req: Req) -> bool:
    if base_req.is_warmup or candidate_req.is_warmup:
        return False
    if (
        base_req.realtime_session_id
        or base_req.session is not None
        or candidate_req.realtime_session_id
        or candidate_req.session is not None
    ):
        return False
    if not isinstance(base_req.prompt, str) or not isinstance(
        candidate_req.prompt, str
    ):
        return False
    if (
        getattr(base_req, "image_path", None) is not None
        or getattr(candidate_req, "image_path", None) is not None
    ):
        return False
    if base_req.return_file_paths_only != candidate_req.return_file_paths_only:
        return False
    base_signature = get_dynamic_batch_signature(base_req)
    candidate_signature = get_dynamic_batch_signature(candidate_req)
    return base_signature is not None and base_signature == candidate_signature


def merge_generation_reqs(reqs: list[Req]) -> Req | None:
    if not reqs:
        return None
    if len(reqs) == 1:
        return deepcopy(reqs[0])

    base_req = reqs[0]
    if any(not can_dynamic_batch(base_req, req) for req in reqs[1:]):
        return None

    merged_req = deepcopy(base_req)
    merged_req.prompt = [req.prompt for req in reqs]
    merged_req.extra = deepcopy(merged_req.extra)
    merged_req.extra["dynamic_batch_seeds"] = [req.seed for req in reqs]
    if merged_req.return_file_paths_only:
        merged_req.extra["dynamic_batch_output_paths"] = [
            req.output_file_path(req.num_outputs_per_prompt, output_index)
            for req in reqs
            for output_index in range(req.num_outputs_per_prompt)
        ]
    merged_req.request_id = f"dynamic_batch::{base_req.request_id}"
    return merged_req


def slice_generation_req(req: Req, start: int, end: int, total: int) -> Req:
    shard = deepcopy(req)
    for field in dataclasses.fields(Req):
        value = getattr(req, field.name, None)
        if isinstance(value, list) and len(value) == total:
            setattr(shard, field.name, deepcopy(value[start:end]))
        elif isinstance(value, tuple) and len(value) == total:
            setattr(shard, field.name, deepcopy(value[start:end]))
        elif (
            hasattr(value, "shape")
            and len(value.shape) > 0
            and value.shape[0] == total
        ):
            setattr(shard, field.name, value[start:end])

    shard.extra = deepcopy(req.extra)
    for key in ("dynamic_batch_seeds", "dynamic_batch_output_paths"):
        value = shard.extra.get(key)
        if isinstance(value, list) and len(value) == total:
            shard.extra[key] = value[start:end]
    for name in ("prior_token_id", "prior_token_image_ids"):
        value = getattr(req, name, None)
        if isinstance(value, list) and len(value) == total:
            setattr(shard, name, deepcopy(value[start:end]))
        elif (
            hasattr(value, "shape")
            and len(value.shape) > 0
            and value.shape[0] == total
        ):
            setattr(shard, name, value[start:end])
    shard.request_id = f"{req.request_id}::shard::{start}:{end}"
    return shard
