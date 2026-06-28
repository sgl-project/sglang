"""Helpers for the public HTTP server tuning CLI flags.

Used by ``sglang.srt.entrypoints.http_server`` to assemble kwargs for
uvicorn / Granian without polluting the request-path file with config-
resolution code.

The resolution rule for ``timeout_keep_alive`` is:
    CLI flag (``--timeout-keep-alive``) overrides the env var
    (``SGLANG_TIMEOUT_KEEP_ALIVE``); the env var falls back to its
    default (65s, see environ.py for the rationale).

The tuning kwargs are passed through to uvicorn/Granian only when the
operator set them, so SGLang doesn't pin the underlying server's
defaults from its side.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from sglang.srt.environ import envs

if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs


def resolved_keep_alive_timeout(server_args: ServerArgs) -> int:
    """CLI flag wins; env var is the fallback (and its default is 65)."""
    if server_args.timeout_keep_alive is not None:
        return server_args.timeout_keep_alive
    return envs.SGLANG_TIMEOUT_KEEP_ALIVE.get()


def uvicorn_tuning_kwargs(server_args: ServerArgs) -> dict:
    """Kwargs to splat into uvicorn.run / uvicorn.Config.

    ``backlog`` is always present (its default 2048 matches uvicorn's own
    so behavior is unchanged for operators not setting the flag).
    ``limit_concurrency`` and ``timeout_graceful_shutdown`` are included
    only when the operator set them, so uvicorn falls back to its own
    None defaults otherwise.
    """
    kwargs: dict = {"backlog": server_args.http_backlog}
    if server_args.http_limit_concurrency is not None:
        kwargs["limit_concurrency"] = server_args.http_limit_concurrency
    if server_args.http_timeout_graceful_shutdown is not None:
        kwargs["timeout_graceful_shutdown"] = server_args.http_timeout_graceful_shutdown
    return kwargs


def granian_http2_settings_kwargs(server_args: ServerArgs) -> dict:
    """Kwargs for ``granian.http.HTTP2Settings``, empty when none set.

    Returned as a plain dict so the caller can construct ``HTTP2Settings``
    with its lazy granian import; this module stays import-safe when
    granian isn't installed.
    """
    kwargs: dict = {}
    if server_args.http2_max_concurrent_streams is not None:
        kwargs["max_concurrent_streams"] = server_args.http2_max_concurrent_streams
    if server_args.http2_max_frame_size is not None:
        kwargs["max_frame_size"] = server_args.http2_max_frame_size
    return kwargs
