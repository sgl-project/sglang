# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import dataclasses
from enum import Enum
from typing import Any

from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req


def _freeze_signature_value(value: Any):
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return tuple(
            (str(key), _freeze_signature_value(item))
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_signature_value(item) for item in value)
    return repr(value)


def _build_dynamic_batch_signature(
    req: Req, *, exclude_num_outputs_per_prompt: bool
) -> tuple[Any, ...] | None:
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
            _freeze_signature_value(getattr(sampling_params, field.name, None)),
        )
        for field in sampling_fields
        if not field.metadata.get("batch_sig_exclude", False)
        and not (
            exclude_num_outputs_per_prompt and field.name == "num_outputs_per_prompt"
        )
    ]

    profile_signature = (
        (True, req.profile_all_stages, req.num_profiled_timesteps)
        if req.profile
        else (False,)
    )
    items.append(("profiling", profile_signature))

    diffusers_kwargs = (req.extra or {}).get("diffusers_kwargs")
    if diffusers_kwargs:
        items.append(("diffusers_kwargs", _freeze_signature_value(diffusers_kwargs)))
    return tuple(items)


def are_requests_batch_compatible(
    base_req: Req,
    candidate_req: Req,
    *,
    exclude_num_outputs_per_prompt: bool = False,
) -> bool:
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
    base_signature = _build_dynamic_batch_signature(
        base_req,
        exclude_num_outputs_per_prompt=exclude_num_outputs_per_prompt,
    )
    candidate_signature = _build_dynamic_batch_signature(
        candidate_req,
        exclude_num_outputs_per_prompt=exclude_num_outputs_per_prompt,
    )
    return base_signature is not None and base_signature == candidate_signature
