from __future__ import annotations

import json
import logging
import socket
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Mapping

from sglang.srt.mem_cache.shared_hicache.transfer import SharedHiCacheTransferBackend
from sglang.srt.mem_cache.shared_hicache.plan import (
    SHARED_HICACHE_DIRECT_TIMEOUT_REASON,
    SharedHiCachePlan,
)
from sglang.srt.mem_cache.shared_hicache.source import ResolvedHostPage

logger = logging.getLogger(__name__)


class _ReusableThreadingHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True

    def get_request(self):
        request, client_address = super().get_request()
        try:
            request.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except OSError:
            logger.debug("Failed to set TCP_NODELAY on SharedHiCache control socket")
        return request, client_address


def is_timeout_error(err: BaseException) -> bool:
    if isinstance(err, TimeoutError):
        return True
    if isinstance(err, urllib.error.URLError):
        reason = getattr(err, "reason", None)
        if isinstance(reason, TimeoutError):
            return True
    return "timed out" in str(err).lower()


def is_indeterminate_direct_transfer_reason(reason: str) -> bool:
    reason = str(reason)
    # A timeout can leave a source-side transfer still running after the target
    # has given up, so target pages must not be reused. Synchronous source-side
    # transfer failures return only after Mooncake reports completion/failure;
    # those target pages may contain partial data, but they are safe to free
    # because they were never attached to the radix cache or request.
    return reason.startswith(SHARED_HICACHE_DIRECT_TIMEOUT_REASON)


def start_source_transfer_server(
    *,
    host: str,
    port: int,
    endpoint: str,
    worker_id: int | None,
    attn_dp_rank: int,
    max_body_bytes: Callable[[], int],
    try_enter: Callable[[], bool],
    exit_resolver: Callable[[], None],
    direct_transfer_enabled: Callable[[], bool],
    handle_source_transfer: Callable[[Mapping[str, Any]], Mapping[str, Any]],
) -> tuple[ThreadingHTTPServer, threading.Thread]:
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            request_start = time.perf_counter()
            if self.path != "/transfer_direct":
                self.send_error(404)
                return
            if not try_enter():
                self._write_json(
                    503,
                    {
                        "ok": False,
                        "reason": "source_resolver_busy",
                    },
                )
                return
            try:
                try:
                    content_len = int(self.headers.get("Content-Length", "0"))
                except ValueError:
                    self._write_json(
                        400,
                        {
                            "ok": False,
                            "reason": "malformed_content_length",
                        },
                    )
                    return
                if content_len <= 0:
                    self._write_json(
                        400,
                        {
                            "ok": False,
                            "reason": "empty_request_body",
                        },
                    )
                    return
                if content_len > max_body_bytes():
                    self._write_json(
                        413,
                        {
                            "ok": False,
                            "reason": "control_payload_too_large",
                        },
                    )
                    return

                body_read_start = time.perf_counter()
                raw_body = self.rfile.read(content_len)
                body_read_ms = (time.perf_counter() - body_read_start) * 1000
                if len(raw_body) != content_len:
                    self._write_json(
                        400,
                        {
                            "ok": False,
                            "reason": "truncated_request_body",
                        },
                    )
                    return
                decode_start = time.perf_counter()
                try:
                    payload = json.loads(raw_body)
                except (UnicodeDecodeError, json.JSONDecodeError) as err:
                    self._write_json(
                        400,
                        {
                            "ok": False,
                            "reason": f"malformed_control_payload:json:{err}",
                        },
                    )
                    return
                decode_ms = (time.perf_counter() - decode_start) * 1000
                if not isinstance(payload, Mapping):
                    self._write_json(
                        400,
                        {
                            "ok": False,
                            "reason": "malformed_control_payload:not_object",
                        },
                    )
                    return
                if not direct_transfer_enabled():
                    self._write_json(
                        501,
                        {
                            "ok": False,
                            "reason": "direct_transfer_unavailable",
                        },
                    )
                    return

                pre_handler_ms = (time.perf_counter() - request_start) * 1000
                response = dict(handle_source_transfer(payload))
                response.update(
                    {
                        "source_control_pre_handler_ms": pre_handler_ms,
                        "source_control_body_read_ms": body_read_ms,
                        "source_control_json_decode_ms": decode_ms,
                    }
                )
                write_start = time.perf_counter()
                self._write_json(200, response)
                write_ms = (time.perf_counter() - write_start) * 1000
                logger.debug(
                    "SharedHiCache source control handled path=%s pre_handler_ms=%.3f body_read_ms=%.3f json_decode_ms=%.3f response_write_ms=%.3f request_total_ms=%.3f",
                    self.path,
                    pre_handler_ms,
                    body_read_ms,
                    decode_ms,
                    write_ms,
                    (time.perf_counter() - request_start) * 1000,
                )
            except Exception as err:
                logger.exception("Shared HiCache source transfer failed")
                self._write_json(500, {"ok": False, "reason": str(err)})
            finally:
                exit_resolver()

        def log_message(self, fmt, *args):
            logger.debug("Shared HiCache source resolver: " + fmt, *args)

        def _write_json(self, status_code: int, response: Mapping[str, Any]):
            data = json.dumps(response).encode("utf-8")
            self.send_response(status_code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    server = _ReusableThreadingHTTPServer((host, port), Handler)
    thread = threading.Thread(
        target=server.serve_forever,
        name=f"shared_hicache-source-{host}:{port}",
        daemon=True,
    )
    thread.start()
    logger.info(
        "Shared HiCache source resolver listening on %s for worker_id=%s attn_dp_rank=%s",
        endpoint,
        worker_id,
        attn_dp_rank,
    )
    return server, thread


def _coerce_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be an integer, got {value!r}")
    if isinstance(value, int):
        return int(value)
    raise ValueError(f"{field_name} must be an integer, got {value!r}")


def pages_from_transfer_result(
    payload: Mapping[str, Any],
    plan: SharedHiCachePlan,
    *,
    start_block: int,
    max_blocks: int,
) -> list[ResolvedHostPage]:
    transferred_blocks = _coerce_int(
        payload.get("transferred_blocks", 0), "transferred_blocks"
    )
    if transferred_blocks < 0:
        raise ValueError("transferred_blocks must be non-negative")
    transferred_blocks = min(transferred_blocks, max_blocks)
    block_hashes = plan.planned_hashes[start_block : start_block + transferred_blocks]
    return [
        ResolvedHostPage(block_hash=block_hash, hash_value="", data=b"")
        for block_hash in block_hashes
    ]


def request_source_transfer(
    *,
    transfer_backend: SharedHiCacheTransferBackend,
    endpoints: list[str],
    plan: SharedHiCachePlan,
    start_block: int,
    max_blocks: int,
    target_page_indices: list[int],
    timeout_secs: float,
) -> tuple[list[ResolvedHostPage], str]:
    encode_start = time.perf_counter()
    body = json.dumps(
        {
            "plan": plan.to_dict(),
            "start_block": start_block,
            "max_blocks": max_blocks,
            "target_session_id": transfer_backend.target_session_id,
            "transfer_backend": transfer_backend.name,
            "target_metadata": transfer_backend.target_descriptor(),
            "target_kv_ptrs": transfer_backend.target_kv_ptrs,
            "target_kv_item_lens": transfer_backend.target_kv_item_lens,
            "target_page_indices": target_page_indices,
        }
    ).encode("utf-8")
    encode_ms = (time.perf_counter() - encode_start) * 1000

    last_reason = "missing_source_endpoint"
    for endpoint in endpoints:
        request = urllib.request.Request(
            f"{endpoint}/transfer_direct",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        request_start = time.perf_counter()
        http_read_ms = 0.0
        decode_ms = 0.0
        response_bytes = 0
        try:
            with urllib.request.urlopen(request, timeout=timeout_secs) as response:
                response_body = response.read()
            http_read_ms = (time.perf_counter() - request_start) * 1000
            response_bytes = len(response_body)
            decode_start = time.perf_counter()
            payload = json.loads(response_body.decode("utf-8"))
            decode_ms = (time.perf_counter() - decode_start) * 1000
        except (
            urllib.error.URLError,
            TimeoutError,
            json.JSONDecodeError,
            UnicodeDecodeError,
        ) as err:
            if is_timeout_error(err):
                last_reason = f"{SHARED_HICACHE_DIRECT_TIMEOUT_REASON}:{err}"
                logger.warning(
                    "Shared HiCache direct transfer timed out endpoint=%s ms=%.3f; target pages will be quarantined reason=%s",
                    endpoint,
                    (time.perf_counter() - request_start) * 1000,
                    last_reason,
                )
                return [], last_reason
            last_reason = f"source_transfer_failed:{err}"
            logger.debug(
                "Shared HiCache direct transfer request failed endpoint=%s ms=%.3f request_encode_ms=%.3f http_read_ms=%.3f response_decode_ms=%.3f request_bytes=%d response_bytes=%d reason=%s",
                endpoint,
                (time.perf_counter() - request_start) * 1000,
                encode_ms,
                http_read_ms,
                decode_ms,
                len(body),
                response_bytes,
                last_reason,
            )
            continue

        if not isinstance(payload, Mapping):
            last_reason = "malformed_source_transfer_response:not_object"
            logger.debug(
                "Shared HiCache direct transfer returned malformed response endpoint=%s ms=%.3f reason=%s",
                endpoint,
                (time.perf_counter() - request_start) * 1000,
                last_reason,
            )
            continue

        last_reason = str(payload.get("reason", "ok"))
        if not payload.get("ok"):
            if is_indeterminate_direct_transfer_reason(last_reason):
                logger.warning(
                    "Shared HiCache direct transfer rejected with indeterminate target-page state endpoint=%s ms=%.3f reason=%s",
                    endpoint,
                    (time.perf_counter() - request_start) * 1000,
                    last_reason,
                )
                return [], last_reason
            logger.debug(
                "Shared HiCache direct transfer rejected endpoint=%s ms=%.3f request_encode_ms=%.3f http_read_ms=%.3f response_decode_ms=%.3f request_bytes=%d response_bytes=%d source_resolve_ms=%s source_transfer_ms=%s source_total_ms=%s reason=%s",
                endpoint,
                (time.perf_counter() - request_start) * 1000,
                encode_ms,
                http_read_ms,
                decode_ms,
                len(body),
                response_bytes,
                payload.get("resolve_ms"),
                payload.get("transfer_ms"),
                payload.get("total_ms"),
                last_reason,
            )
            continue
        try:
            parse_start = time.perf_counter()
            pages = pages_from_transfer_result(
                payload,
                plan,
                start_block=start_block,
                max_blocks=max_blocks,
            )
            parse_ms = (time.perf_counter() - parse_start) * 1000
        except (TypeError, KeyError, ValueError) as err:
            last_reason = f"malformed_source_transfer_response:{err}"
            logger.debug(
                "Shared HiCache direct transfer returned malformed pages endpoint=%s ms=%.3f request_encode_ms=%.3f http_read_ms=%.3f response_decode_ms=%.3f request_bytes=%d response_bytes=%d reason=%s",
                endpoint,
                (time.perf_counter() - request_start) * 1000,
                encode_ms,
                http_read_ms,
                decode_ms,
                len(body),
                response_bytes,
                last_reason,
            )
            continue
        logger.debug(
            "Shared HiCache direct transfer response endpoint=%s pages=%d ms=%.3f request_encode_ms=%.3f http_read_ms=%.3f response_decode_ms=%.3f response_parse_ms=%.3f request_bytes=%d response_bytes=%d source_resolve_ms=%s source_transfer_ms=%s source_total_ms=%s source_bytes=%s reason=%s",
            endpoint,
            len(pages),
            (time.perf_counter() - request_start) * 1000,
            encode_ms,
            http_read_ms,
            decode_ms,
            parse_ms,
            len(body),
            response_bytes,
            payload.get("resolve_ms"),
            payload.get("transfer_ms"),
            payload.get("total_ms"),
            payload.get("transfer_bytes"),
            last_reason,
        )
        if pages or last_reason in {"ok", "already_local"}:
            return pages, last_reason

    return [], last_reason
