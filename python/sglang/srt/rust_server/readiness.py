"""Publish parent warmup completion through the private scheduler channels."""

from __future__ import annotations

import time
import uuid
from typing import TYPE_CHECKING

import zmq

from sglang.srt.arg_groups.model_override_base import ep_scale_joiner_of
from sglang.srt.arg_groups.overrides import resolving_view
from sglang.srt.managers.io_struct import (
    RustFrontendReadyReqInput,
    RustFrontendReadyReqOutput,
    sock_recv,
    sock_send,
)
from sglang.srt.utils.network import get_zmq_socket

if TYPE_CHECKING:
    from sglang.srt.server_args import PortArgs, ServerArgs


def publish_frontend_ready(
    server_args: ServerArgs, port_args: PortArgs, *, timeout_seconds: float = 30
) -> None:
    """Wait for every native frontend to acknowledge the completed warmup.

    The DP controller owns the input bind when present. In a single worker
    launch, this replaces the absent Python tokenizer's input bind. The
    tokenizer output endpoint is otherwise unused by the native frontend.
    """
    cfg = resolving_view(server_args)
    context = zmq.Context(1)
    requests = replies = None
    try:
        replies = get_zmq_socket(context, zmq.PULL, port_args.tokenizer_ipc_name, True)
        requests = get_zmq_socket(
            context,
            zmq.PUSH,
            port_args.scheduler_input_ipc_name,
            bind=not (cfg.dp_size > 1 or ep_scale_joiner_of(cfg)),
        )
        requests.setsockopt(zmq.SNDTIMEO, max(1, int(timeout_seconds * 1000)))
        rid = uuid.uuid4().hex
        deadline = time.monotonic() + timeout_seconds
        sock_send(requests, RustFrontendReadyReqInput(rid=rid))
        pending = set(range(cfg.dp_size))
        while pending:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not replies.poll(max(1, int(remaining * 1000))):
                raise TimeoutError(
                    f"Rust readiness acknowledgements missing from DP ranks {sorted(pending)}"
                )
            reply = sock_recv(replies)
            if (
                not isinstance(reply, RustFrontendReadyReqOutput)
                or reply.rid != rid
                or not 0 <= reply.dp_rank < cfg.dp_size
            ):
                raise RuntimeError(f"Invalid Rust readiness acknowledgement: {reply!r}")
            pending.discard(reply.dp_rank)
    finally:
        if requests is not None:
            requests.close(linger=0)
        if replies is not None:
            replies.close(linger=0)
        context.term()
