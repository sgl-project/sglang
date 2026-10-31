# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SGLang Team

import hashlib
import json

import torch


def multimodal_fingerprint(namespace, modality, feature, metadata):
    digest = hashlib.sha256()

    def update(value):
        if isinstance(value, torch.Tensor):
            tensor = value.detach().cpu().contiguous()
            update((str(tensor.dtype), tuple(tensor.shape)))
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(value, dict):
            digest.update(b"dict")
            for key in sorted(value):
                update(key)
                update(value[key])
        elif isinstance(value, (list, tuple)):
            digest.update(b"sequence")
            update(len(value))
            for item in value:
                update(item)
        elif value is None or isinstance(value, (str, int, float, bool)):
            digest.update(json.dumps(value, sort_keys=True, allow_nan=False).encode())
            digest.update(b"\x00")
        else:
            raise ValueError(
                f"Unsupported multimodal cache metadata: {type(value).__name__}"
            )

    update((namespace, modality, feature, metadata))
    return int.from_bytes(digest.digest()[:8], "big")


def flatten_encoder_features(output):
    if getattr(output, "pooler_output", None) is not None:
        output = output.pooler_output
    elif hasattr(output, "last_hidden_state"):
        output = output.last_hidden_state
    elif isinstance(output, dict):
        output = output.get("pooler_output", output.get("last_hidden_state"))
    if isinstance(output, (list, tuple)):
        if not output or not all(isinstance(item, torch.Tensor) for item in output):
            raise ValueError(
                "Multimodal encoders must return tensors of projected token features"
            )
        output = torch.cat([item.reshape(-1, item.shape[-1]) for item in output], dim=0)
    if not isinstance(output, torch.Tensor) or output.ndim < 2:
        raise ValueError("Multimodal encoder output has no token and hidden dimensions")
    return output.reshape(-1, output.shape[-1])


def validate_multimodal_offsets(items, input_ids, modality_token_ids):
    occupied = set()
    for item in items:
        token_id = modality_token_ids[item.modality]
        if token_id is None or not item.offsets:
            raise ValueError(
                "Multimodal cache entries require token IDs and token spans"
            )
        for start, end in item.offsets:
            if start < 0 or end < start or end >= len(input_ids):
                raise ValueError("Multimodal span is outside the prompt")
            span = set(range(start, end + 1))
            if occupied & span:
                raise ValueError("Multimodal spans must not overlap")
            if not torch.all(input_ids[start : end + 1] == token_id):
                raise ValueError("Multimodal span contains non-placeholder tokens")
            occupied.update(span)


def placeholder_spans(input_ids, token_id):
    mask = input_ids == token_id
    padded = torch.cat((mask.new_zeros(1), mask, mask.new_zeros(1)))
    changes = padded[1:].to(torch.int8) - padded[:-1].to(torch.int8)
    starts = torch.where(changes == 1)[0].tolist()
    ends = (torch.where(changes == -1)[0] - 1).tolist()
    return list(zip(starts, ends))
