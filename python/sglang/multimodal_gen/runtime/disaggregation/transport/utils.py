# SPDX-License-Identifier: Apache-2.0
"""Small helpers shared by diffusion transfer components."""

from __future__ import annotations

import logging
from typing import TypeVar

import torch

_ExcT = TypeVar("_ExcT", bound=Exception)


def warn_or_raise(
    logger: logging.Logger,
    strict: bool,
    message: str,
    *args,
    exc_type: type[_ExcT] = RuntimeError,
) -> str:
    text = message % args if args else message
    if strict:
        raise exc_type(text)
    logger.warning(text)
    return text


def require_request_id(
    msg: dict,
    *,
    context: str,
    logger: logging.Logger,
    strict: bool = False,
) -> str | None:
    request_id = str(msg.get("request_id", "") or "")
    if request_id:
        return request_id
    warn_or_raise(
        logger,
        strict,
        "%s missing request_id; dropping message: %s",
        context,
        msg,
        exc_type=ValueError,
    )
    return None


def tensor_fields_need_device_event(
    tensor_fields: dict[str, torch.Tensor | list[torch.Tensor] | None],
) -> bool:
    for value in tensor_fields.values():
        tensors = value if isinstance(value, list) else [value]
        for tensor in tensors:
            if isinstance(tensor, torch.Tensor) and tensor.device.type != "cpu":
                return True
    return False


def record_device_event_if_needed(
    *,
    enabled: bool,
    stream: torch.Stream | None = None,
):
    if not enabled:
        return None
    device_module = torch.get_device_module()
    if stream is not None:
        event = device_module.Event()
        event.record(stream)
        return event
    if device_module.is_available():
        event = device_module.Event()
        event.record(device_module.current_stream())
        return event
    return None


def record_load_event(device: torch.device | str, stream: torch.Stream | None = None):
    device_type = (
        device.type
        if isinstance(device, torch.device)
        else str(device).split(":", 1)[0]
    )
    return record_device_event_if_needed(
        enabled=device_type != "cpu",
        stream=stream,
    )
