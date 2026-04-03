from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
from safetensors.torch import load as load_safetensors

from sglang.srt.managers.io_struct import UpdateWeightsFromTensorReqInput
from sglang.srt.utils import MultiprocessingSerializer


def load_named_tensors_from_bytes(
    payload: bytes,
    tensor_format: str = "safetensors",
) -> List[Tuple[str, torch.Tensor]]:
    if tensor_format != "safetensors":
        raise ValueError(f"Unsupported tensor_format={tensor_format!r}.")

    tensor_dict = load_safetensors(payload)
    return list(tensor_dict.items())


def build_update_weights_request_from_named_tensors(
    named_tensors: List[Tuple[str, torch.Tensor]],
    *,
    tp_size: int,
    load_format: Optional[str] = None,
    flush_cache: bool = True,
    abort_all_requests: bool = False,
    base_weight_version: Optional[str] = None,
    weight_version: Optional[str] = None,
    payload_digest: Optional[str] = None,
    loader_metadata: Optional[Dict[str, Any]] = None,
    crash_on_error: bool = False,
    disable_draft_model: Optional[bool] = None,
) -> UpdateWeightsFromTensorReqInput:
    return UpdateWeightsFromTensorReqInput(
        serialized_named_tensors=[
            MultiprocessingSerializer.serialize(named_tensors) for _ in range(tp_size)
        ],
        load_format=load_format,
        flush_cache=flush_cache,
        abort_all_requests=abort_all_requests,
        base_weight_version=base_weight_version,
        weight_version=weight_version,
        payload_digest=payload_digest,
        loader_metadata=loader_metadata,
        crash_on_error=crash_on_error,
        disable_draft_model=disable_draft_model,
    )
