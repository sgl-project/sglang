# SPDX-License-Identifier: Apache-2.0
"""Helpers for cooperative cancellation across scheduler processes.

Running denoising loops observe marker files because HTTP handlers, the
scheduler, and distributed workers do not share process-local state.
"""

import json
import os
import tempfile
import uuid
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote

CLIENT_CANCELLED_REASON = "client_cancelled"
CLIENT_CANCELLED_MESSAGE = "Generation cancelled by client"


@dataclass
class CancelGenerationReq:
    request_id: str
    reason: str = CLIENT_CANCELLED_REASON


class RequestCancelledError(RuntimeError):
    def __init__(
        self,
        request_id: str,
        reason: str = CLIENT_CANCELLED_REASON,
        message: str = CLIENT_CANCELLED_MESSAGE,
    ) -> None:
        self.request_id = request_id
        self.reason = reason
        super().__init__(message)


def cancellation_dir(server_args: Any) -> str:
    configured = getattr(server_args, "cancellation_dir", None)
    if configured:
        return os.path.abspath(configured)
    scheduler_port = getattr(server_args, "scheduler_port", "default")
    return os.path.join(
        tempfile.gettempdir(),
        "sglang_diffusion_cancel",
        str(scheduler_port),
    )


def _marker_path(request_id: str, server_args: Any) -> str:
    safe_id = quote(str(request_id), safe="")
    return os.path.join(cancellation_dir(server_args), f"{safe_id}.json")


def mark_request_cancelled(
    request_id: str,
    server_args: Any,
    reason: str = CLIENT_CANCELLED_REASON,
) -> None:
    if not request_id:
        return
    cancel_dir = cancellation_dir(server_args)
    os.makedirs(cancel_dir, exist_ok=True)
    path = _marker_path(request_id, server_args)
    tmp_path = f"{path}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
    payload = {"request_id": request_id, "reason": reason}
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    os.replace(tmp_path, path)


def clear_request_cancelled(request_id: str, server_args: Any) -> None:
    if not request_id:
        return
    try:
        os.remove(_marker_path(request_id, server_args))
    except FileNotFoundError:
        pass


def is_request_cancelled(request_id: str, server_args: Any) -> bool:
    if not request_id:
        return False
    return os.path.exists(_marker_path(request_id, server_args))


def get_cancel_reason(
    request_id: str,
    server_args: Any,
    default: str = CLIENT_CANCELLED_REASON,
) -> str:
    if not request_id:
        return default
    try:
        with open(_marker_path(request_id, server_args), encoding="utf-8") as f:
            payload = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return default
    reason = payload.get("reason")
    return reason if isinstance(reason, str) and reason else default


def _aliases_for_req(req: Any) -> set[str]:
    aliases: set[str] = set()
    request_id = getattr(req, "request_id", None)
    if request_id:
        aliases.add(str(request_id))

    extra = getattr(req, "extra", None)
    if isinstance(extra, dict):
        parent_request_id = extra.get("parent_request_id")
        if parent_request_id:
            aliases.add(str(parent_request_id))

    return aliases


def request_alias_groups(payload: Any) -> list[set[str]]:
    """Return cancel-id aliases grouped by logical request.

    Dynamic batches have one synthetic execution id, while each original request
    must remain independently cancellable.
    """
    if isinstance(payload, list):
        groups = [group for item in payload for group in request_alias_groups(item)]
        return [group for group in groups if group]

    extra = getattr(payload, "extra", None)
    if isinstance(extra, dict):
        dynamic_request_ids = extra.get("dynamic_batch_request_ids")
        if dynamic_request_ids:
            return [
                {str(request_id)} for request_id in dynamic_request_ids if request_id
            ]

    aliases = _aliases_for_req(payload)
    return [aliases] if aliases else []


def request_ids_for_payload(payload: Any) -> list[str]:
    seen: set[str] = set()
    request_ids: list[str] = []
    for group in request_alias_groups(payload):
        for request_id in group:
            if request_id not in seen:
                seen.add(request_id)
                request_ids.append(request_id)
    return request_ids


def cancelled_request_groups(
    payload: Any,
    server_args: Any,
) -> list[set[str]]:
    groups = request_alias_groups(payload)
    return [
        group
        for group in groups
        if any(is_request_cancelled(request_id, server_args) for request_id in group)
    ]


def is_payload_cancelled(
    payload: Any,
    server_args: Any,
    *,
    require_all_for_batched: bool = True,
) -> bool:
    """Return whether a payload should stop because cancellation was requested.

    Dynamic batches require all members by default; pass `False` when removing
    cancelled members while keeping the rest of the batch alive.
    """
    groups = request_alias_groups(payload)
    if not groups:
        return False
    cancelled_groups = cancelled_request_groups(payload, server_args)
    if require_all_for_batched and len(groups) > 1:
        return len(cancelled_groups) == len(groups)
    return bool(cancelled_groups)


def first_cancelled_request_id(payload: Any, server_args: Any) -> str | None:
    for group in request_alias_groups(payload):
        for request_id in group:
            if is_request_cancelled(request_id, server_args):
                return request_id
    return None


def raise_if_cancelled(
    payload: Any,
    server_args: Any,
    *,
    require_all_for_batched: bool = True,
) -> None:
    if not is_payload_cancelled(
        payload,
        server_args,
        require_all_for_batched=require_all_for_batched,
    ):
        return
    request_id = first_cancelled_request_id(payload, server_args)
    if request_id is None:
        request_ids = request_ids_for_payload(payload)
        request_id = request_ids[0] if request_ids else ""
    raise RequestCancelledError(
        request_id=request_id,
        reason=get_cancel_reason(request_id, server_args),
    )
