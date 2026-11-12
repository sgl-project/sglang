from __future__ import annotations

import concurrent.futures
import hashlib
import json
import logging
import queue
import secrets
import threading
import time
import uuid
from array import array
from typing import TYPE_CHECKING, Dict, Optional

import numpy as np
import requests
import torch

from sglang.srt.managers.req_types import FINISH_ABORT
from sglang.srt.managers.scheduler_components.request_list import (
    IndexedReqList,
    ListKind,
    ReqList,
    move,
)
from sglang.srt.mem_cache.base_prefix_cache import InsertParams, MatchPrefixParams
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey
from sglang.srt.runtime_context import get_disagg
from sglang.srt.utils.network import NetworkAddress

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.managers.scheduler_components.request_retirer import (
        RequestRetirer,
    )
    from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
    from sglang.srt.mem_cache.memory_pool import ReqToTokenPool

logger = logging.getLogger(__name__)

REVERSE_KV_PREPARE_TAG = b"REVERSE_KV_PREPARE_V1"
REVERSE_KV_RESPONSE_TAG = b"REVERSE_KV_RESPONSE_V1"
REVERSE_KV_PROTOCOL_VERSION = 1

_MAX_HEADER_BYTES = 16 * 1024
_MAX_NAMESPACE_BYTES = 4 * 1024
_MAX_ATTEMPTS_PER_DRAIN = 32
_MAX_CONCURRENT_IMPORTS = 4
_MAX_CONCURRENT_RETURNS = 4
_MAX_CONCURRENT_TICKET_REGISTRATIONS = 4


class ReverseKVTransferUnsettled(RuntimeError):
    """A reverse NIXL read whose destination may still be in use."""


class _Ticket:
    __slots__ = (
        "bootstrap_room",
        "nonce",
        "expires_at",
        "fingerprint",
        "prompt_tokens",
        "control_endpoint",
        "attempt_id",
    )

    def __init__(
        self,
        *,
        bootstrap_room: int,
        nonce: str,
        expires_at: float,
        fingerprint: str,
        prompt_tokens: int,
        control_endpoint: str,
    ) -> None:
        self.bootstrap_room = bootstrap_room
        self.nonce = nonce
        self.expires_at = expires_at
        self.fingerprint = fingerprint
        self.prompt_tokens = prompt_tokens
        self.control_endpoint = control_endpoint
        self.attempt_id: Optional[str] = None

    def payload(self) -> dict:
        return {
            "version": REVERSE_KV_PROTOCOL_VERSION,
            "bootstrap_room": self.bootstrap_room,
            "nonce": self.nonce,
            "expires_at": self.expires_at,
            "fingerprint": self.fingerprint,
            "prompt_tokens": self.prompt_tokens,
            "control_endpoint": self.control_endpoint,
        }


class _ImportAttempt:
    __slots__ = (
        "attempt_id",
        "ticket",
        "peer_name",
        "key",
        "initial_prefix_len",
        "prefix_lock",
        "allocated_indices",
        "future",
        "terminal_message",
        "unsafe",
    )

    def __init__(
        self,
        *,
        attempt_id: str,
        ticket: _Ticket,
        peer_name: str,
        key: RadixKey,
        initial_prefix_len: int,
        prefix_lock,
        allocated_indices: torch.Tensor,
        future: Optional[concurrent.futures.Future],
    ) -> None:
        self.attempt_id = attempt_id
        self.ticket = ticket
        self.peer_name = peer_name
        self.key = key
        self.initial_prefix_len = initial_prefix_len
        self.prefix_lock = prefix_lock
        self.allocated_indices = allocated_indices
        self.future = future
        self.terminal_message: Optional[list[bytes]] = None
        self.unsafe = False


def _fingerprint_payload(
    *,
    page_size: int,
    kv_cache_dtype: Optional[str],
    kv_layout: Optional[str],
    kv_item_lens,
    kv_layer_ids,
    num_draft_entries: int,
    state_types,
) -> str:
    payload = {
        "version": REVERSE_KV_PROTOCOL_VERSION,
        "page_size": int(page_size),
        "kv_cache_dtype": kv_cache_dtype,
        "kv_layout": kv_layout,
        "kv_item_lens": [int(x) for x in kv_item_lens],
        "kv_layer_ids": [int(x) for x in kv_layer_ids],
        "num_draft_entries": int(num_draft_entries),
        "state_types": [str(x) for x in state_types],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def reverse_kv_fingerprint_from_args(kv_args) -> str:
    return _fingerprint_payload(
        page_size=kv_args.page_size,
        kv_cache_dtype=kv_args.kv_cache_dtype_str,
        kv_layout=kv_args.kv_layout,
        kv_item_lens=kv_args.kv_item_lens,
        kv_layer_ids=kv_args.kv_layer_ids,
        num_draft_entries=kv_args.num_draft_entries,
        state_types=kv_args.state_types or [],
    )


def reverse_kv_fingerprint_from_peer(peer_info) -> str:
    return _fingerprint_payload(
        page_size=peer_info.dst_page_size,
        kv_cache_dtype=peer_info.dst_kv_cache_dtype,
        kv_layout=peer_info.dst_kv_layout,
        kv_item_lens=peer_info.dst_kv_item_lens,
        kv_layer_ids=peer_info.dst_kv_layer_ids,
        num_draft_entries=peer_info.dst_num_draft_entries,
        state_types=peer_info.dst_state_types,
    )


def build_reverse_kv_prepare_message(
    *,
    attempt_id: str,
    bootstrap_room: int,
    nonce: str,
    fingerprint: str,
    agent_name: str,
    token_ids: np.ndarray,
    source_indices: np.ndarray,
    extra_key: Optional[str],
    cache_salt: Optional[str],
) -> list[bytes]:
    token_ids = np.asarray(token_ids, dtype=np.int64)
    source_indices = np.asarray(source_indices, dtype=np.int32)
    header = {
        "version": REVERSE_KV_PROTOCOL_VERSION,
        "attempt_id": attempt_id,
        "bootstrap_room": int(bootstrap_room),
        "nonce": nonce,
        "fingerprint": fingerprint,
        "agent_name": agent_name,
        "num_tokens": int(token_ids.size),
        "num_source_indices": int(source_indices.size),
        "extra_key": extra_key,
        "cache_salt": cache_salt,
    }
    return [
        REVERSE_KV_PREPARE_TAG,
        json.dumps(header, separators=(",", ":")).encode(),
        token_ids.tobytes(),
        source_indices.tobytes(),
    ]


def parse_reverse_kv_prepare_message(
    message: list[bytes], *, max_tokens: int
) -> tuple[dict, np.ndarray, np.ndarray]:
    if len(message) != 4 or message[0] != REVERSE_KV_PREPARE_TAG:
        raise ValueError("invalid reverse-KV prepare frame layout")
    if len(message[1]) > _MAX_HEADER_BYTES:
        raise ValueError("reverse-KV prepare header is too large")
    header = json.loads(message[1])
    if not isinstance(header, dict):
        raise ValueError("reverse-KV prepare header must be an object")
    if header.get("version") != REVERSE_KV_PROTOCOL_VERSION:
        raise ValueError("unsupported reverse-KV protocol version")

    try:
        bootstrap_room = int(header["bootstrap_room"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("reverse-KV bootstrap room is invalid") from error
    if bootstrap_room < 0:
        raise ValueError("reverse-KV bootstrap room is invalid")
    header["bootstrap_room"] = bootstrap_room

    num_tokens = int(header["num_tokens"])
    num_source_indices = int(header["num_source_indices"])
    if num_tokens <= 0 or num_tokens > max_tokens:
        raise ValueError("reverse-KV token count is outside configured bounds")
    if num_source_indices != num_tokens:
        raise ValueError("reverse-KV source-index count must match token count")
    if len(message[2]) != num_tokens * np.dtype(np.int64).itemsize:
        raise ValueError("reverse-KV token frame length mismatch")
    if len(message[3]) != num_tokens * np.dtype(np.int32).itemsize:
        raise ValueError("reverse-KV source-index frame length mismatch")

    for field in ("attempt_id", "nonce", "fingerprint", "agent_name"):
        value = header.get(field)
        if not isinstance(value, str) or not value:
            raise ValueError(f"reverse-KV {field} must be a non-empty string")
        if len(value.encode()) > _MAX_NAMESPACE_BYTES:
            raise ValueError(f"reverse-KV {field} is too large")
    for field in ("extra_key", "cache_salt"):
        value = header.get(field)
        if value is not None and (
            not isinstance(value, str) or len(value.encode()) > _MAX_NAMESPACE_BYTES
        ):
            raise ValueError(f"reverse-KV {field} is invalid")

    token_ids = np.frombuffer(message[2], dtype=np.int64).copy()
    source_indices = np.frombuffer(message[3], dtype=np.int32).copy()
    return header, token_ids, source_indices


def build_reverse_kv_response_message(
    *, attempt_id: str, status: str, reason: str, imported_tokens: int = 0
) -> list[bytes]:
    payload = {
        "version": REVERSE_KV_PROTOCOL_VERSION,
        "attempt_id": attempt_id,
        "status": status,
        "reason": reason,
        "imported_tokens": int(imported_tokens),
    }
    return [
        REVERSE_KV_RESPONSE_TAG,
        json.dumps(payload, separators=(",", ":")).encode(),
    ]


def parse_reverse_kv_response_message(message: list[bytes]) -> dict:
    if len(message) != 2 or message[0] != REVERSE_KV_RESPONSE_TAG:
        raise ValueError("invalid reverse-KV response frame layout")
    if len(message[1]) > _MAX_HEADER_BYTES:
        raise ValueError("reverse-KV response is too large")
    payload = json.loads(message[1])
    if payload.get("version") != REVERSE_KV_PROTOCOL_VERSION:
        raise ValueError("unsupported reverse-KV response version")
    if payload.get("status") not in {"committed", "rejected", "failed"}:
        raise ValueError("invalid reverse-KV response status")
    if not isinstance(payload.get("attempt_id"), str):
        raise ValueError("reverse-KV response has no attempt id")
    return payload


class ReverseKVImportManager:
    def __init__(
        self,
        *,
        kv_manager,
        tree_cache,
        token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator,
    ) -> None:
        if not isinstance(tree_cache, RadixCache):
            raise ValueError("Reverse KV import requires the Python RadixCache")
        if kv_manager.kv_args.num_draft_entries:
            raise ValueError("Reverse KV import does not support draft KV entries")
        if kv_manager.kv_args.state_types:
            raise ValueError(
                "Reverse KV import does not support recurrent or SWA state"
            )

        cfg = get_disagg()
        self.kv_manager = kv_manager
        self.tree_cache = tree_cache
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        self.page_size = tree_cache.page_size
        self.min_tokens = cfg.disaggregation_reverse_kv_min_tokens
        self.max_tokens = cfg.disaggregation_reverse_kv_max_tokens
        self.control_timeout = cfg.disaggregation_reverse_kv_control_timeout
        self.ticket_ttl = cfg.disaggregation_reverse_kv_ticket_ttl
        self.fingerprint = reverse_kv_fingerprint_from_args(kv_manager.kv_args)
        self._tickets: Dict[int, _Ticket] = {}
        self._attempts: Dict[str, _ImportAttempt] = {}
        self._commands: queue.SimpleQueue[list[bytes]] = queue.SimpleQueue()
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=_MAX_CONCURRENT_IMPORTS,
            thread_name_prefix="reverse-kv-import",
        )
        self._ticket_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=_MAX_CONCURRENT_TICKET_REGISTRATIONS,
            thread_name_prefix="reverse-kv-ticket",
        )
        self._ticket_registrations: Dict[
            int, tuple[_Ticket, concurrent.futures.Future]
        ] = {}
        self._admission_disabled = False
        kv_manager.reverse_kv_import_manager = self

    def request_lists(self) -> Dict[ListKind, ReqList]:
        return {}

    def issue_ticket(
        self, *, bootstrap_addr: str, bootstrap_room: int, prompt_tokens: int
    ) -> None:
        self._finish_ticket_registrations()
        if len(self._ticket_registrations) >= _MAX_CONCURRENT_TICKET_REGISTRATIONS:
            logger.warning(
                "Reverse KV ticket registration capacity exhausted for room %s",
                bootstrap_room,
            )
            return

        ticket = _Ticket(
            bootstrap_room=bootstrap_room,
            nonce=secrets.token_urlsafe(24),
            expires_at=time.time() + self.ticket_ttl,
            fingerprint=self.fingerprint,
            prompt_tokens=prompt_tokens,
            control_endpoint=NetworkAddress(
                self.kv_manager.local_ip, self.kv_manager.rank_port
            ).to_tcp(),
        )
        self._tickets[bootstrap_room] = ticket
        future = self._ticket_executor.submit(
            self._register_ticket,
            bootstrap_addr=bootstrap_addr,
            ticket=ticket,
        )
        self._ticket_registrations[bootstrap_room] = (ticket, future)

    def _register_ticket(self, *, bootstrap_addr: str, ticket: _Ticket) -> None:
        response = requests.post(
            f"http://{bootstrap_addr}/register_reverse_kv_ticket",
            json=ticket.payload(),
            timeout=min(self.control_timeout, 1.0),
        )
        if response.status_code != 200:
            raise RuntimeError(
                f"bootstrap registry returned {response.status_code}: {response.text}"
            )

    def _finish_ticket_registrations(self) -> None:
        for room, (ticket, future) in list(self._ticket_registrations.items()):
            if not future.done():
                continue
            try:
                future.result()
            except Exception as error:
                if self._tickets.get(room) is ticket:
                    del self._tickets[room]
                logger.warning(
                    "Reverse KV disabled for bootstrap room %s: ticket registration "
                    "failed: %s",
                    room,
                    error,
                )
            del self._ticket_registrations[room]

    def enqueue_prepare(self, message: list[bytes]) -> None:
        self._commands.put(message)

    def process_pending(self) -> None:
        self._finish_ticket_registrations()
        self._finish_completed_attempts()
        for _ in range(_MAX_ATTEMPTS_PER_DRAIN):
            try:
                message = self._commands.get_nowait()
            except queue.Empty:
                break
            self._handle_prepare(message)
        self._cleanup_expired()

    def _handle_prepare(self, message: list[bytes]) -> None:
        try:
            header, token_ids, source_indices = parse_reverse_kv_prepare_message(
                message, max_tokens=self.max_tokens
            )
        except Exception as error:
            logger.warning("Rejected malformed reverse-KV prepare: %s", error)
            return

        attempt_id = header["attempt_id"]
        previous = self._attempts.get(attempt_id)
        if previous is not None:
            if previous.terminal_message is not None:
                self._send_response(previous.peer_name, previous.terminal_message)
            return

        if self._admission_disabled:
            self._send_terminal(
                peer_name=header["agent_name"],
                attempt_id=attempt_id,
                status="rejected",
                reason="reverse-KV imports are disabled after an unsettled attempt",
            )
            return

        room = int(header["bootstrap_room"])
        ticket = self._tickets.get(room)
        rejection = self._validate_ticket(ticket, header, len(token_ids))
        if rejection is not None:
            self._send_terminal(
                peer_name=header["agent_name"],
                attempt_id=attempt_id,
                status="rejected",
                reason=rejection,
            )
            return
        assert ticket is not None

        peer = self.kv_manager.decode_kv_args_table.get(header["agent_name"])
        if peer is None:
            self._send_terminal(
                peer_name=header["agent_name"],
                attempt_id=attempt_id,
                status="rejected",
                reason="decode NIXL peer is not registered",
            )
            return
        if reverse_kv_fingerprint_from_peer(peer) != self.fingerprint:
            self._send_terminal(
                peer_name=header["agent_name"],
                attempt_id=attempt_id,
                status="rejected",
                reason="decode KV layout does not match the ticket",
            )
            return

        active_imports = sum(
            attempt.future is not None and not attempt.future.done()
            for attempt in self._attempts.values()
        )
        if active_imports >= _MAX_CONCURRENT_IMPORTS:
            self._send_terminal(
                peer_name=header["agent_name"],
                attempt_id=attempt_id,
                status="rejected",
                reason="reverse-KV import capacity exhausted",
            )
            return

        key = RadixKey(
            array("q", token_ids.tolist()),
            header.get("extra_key"),
            cache_salt=header.get("cache_salt"),
        )
        match = self.tree_cache.match_prefix(MatchPrefixParams(key=key))
        prefix_len = match.device_prefix_len
        ticket.attempt_id = attempt_id
        if prefix_len == len(token_ids):
            attempt = _ImportAttempt(
                attempt_id=attempt_id,
                ticket=ticket,
                peer_name=header["agent_name"],
                key=key,
                initial_prefix_len=prefix_len,
                prefix_lock=None,
                allocated_indices=torch.empty(
                    0, dtype=torch.int64, device=self.tree_cache.device
                ),
                future=None,
            )
            self._attempts[attempt_id] = attempt
            self._complete_attempt(
                attempt,
                status="committed",
                reason="prefix already present",
                imported_tokens=0,
            )
            return

        prefix_lock = self.tree_cache.lock(match.last_device_node)
        missing_tokens = len(token_ids) - prefix_len
        self.token_to_kv_pool_allocator.evict_to_free_tokens(
            self.tree_cache, missing_tokens
        )
        allocated_indices = self.token_to_kv_pool_allocator.alloc(missing_tokens)
        if allocated_indices is None:
            self.tree_cache.unlock(prefix_lock)
            ticket.attempt_id = None
            self._send_terminal(
                peer_name=header["agent_name"],
                attempt_id=attempt_id,
                status="rejected",
                reason="insufficient prefill KV capacity",
            )
            return

        destination_indices = (
            allocated_indices.to("cpu").numpy().astype(np.int32, copy=False)
        )
        future = self._executor.submit(
            self.kv_manager.read_kvcache_from_peer,
            peer_name=header["agent_name"],
            remote_source_indices=source_indices[prefix_len:],
            local_destination_indices=destination_indices,
            timeout_s=self.control_timeout,
        )
        self._attempts[attempt_id] = _ImportAttempt(
            attempt_id=attempt_id,
            ticket=ticket,
            peer_name=header["agent_name"],
            key=key,
            initial_prefix_len=prefix_len,
            prefix_lock=prefix_lock,
            allocated_indices=allocated_indices,
            future=future,
        )

    def _validate_ticket(
        self, ticket: Optional[_Ticket], header: dict, token_count: int
    ) -> Optional[str]:
        if ticket is None:
            return "reverse-KV ticket not found"
        if time.time() >= ticket.expires_at:
            return "reverse-KV ticket expired"
        if not secrets.compare_digest(ticket.nonce, header["nonce"]):
            return "reverse-KV ticket nonce mismatch"
        if header["fingerprint"] != ticket.fingerprint:
            return "reverse-KV layout fingerprint mismatch"
        if ticket.attempt_id is not None:
            return "reverse-KV ticket already consumed"
        if token_count % self.page_size != 0:
            return "reverse-KV token count is not page-aligned"
        if token_count > self.max_tokens:
            return "reverse-KV token count exceeds configured maximum"
        if token_count - ticket.prompt_tokens < self.min_tokens:
            return "reverse-KV generated suffix is below configured minimum"
        return None

    def _finish_completed_attempts(self) -> None:
        for attempt in list(self._attempts.values()):
            future = attempt.future
            if (
                future is None
                or attempt.terminal_message is not None
                or attempt.unsafe
                or not future.done()
            ):
                continue
            error = future.exception()
            if error is not None:
                if isinstance(error, ReverseKVTransferUnsettled):
                    attempt.unsafe = True
                    self._admission_disabled = True
                    logger.error(
                        "Reverse KV attempt %s did not settle; retaining decode and "
                        "prefill pages indefinitely to avoid reuse under transfer",
                        attempt.attempt_id,
                    )
                    continue
                self._abort_attempt(attempt)
                self._complete_attempt(
                    attempt,
                    status="failed",
                    reason=str(error),
                    imported_tokens=0,
                )
                continue
            try:
                imported_tokens = self._commit_attempt(attempt)
            except Exception:
                attempt.unsafe = True
                self._admission_disabled = True
                logger.exception(
                    "Reverse KV attempt %s transferred but could not be published; "
                    "retaining both sides",
                    attempt.attempt_id,
                )
                continue
            self._complete_attempt(
                attempt,
                status="committed",
                reason="reverse KV imported",
                imported_tokens=imported_tokens,
            )

    def _commit_attempt(self, attempt: _ImportAttempt) -> int:
        prefix_values = self.tree_cache.path_device_indices(attempt.prefix_lock.node)
        values = torch.cat((prefix_values, attempt.allocated_indices))
        result = self.tree_cache.insert(InsertParams(key=attempt.key, value=values))
        duplicate_tokens = result.prefix_len - attempt.initial_prefix_len
        if duplicate_tokens < 0:
            raise RuntimeError("reverse-KV radix prefix regressed during import")
        if duplicate_tokens:
            self.token_to_kv_pool_allocator.free_segment(
                attempt.allocated_indices[:duplicate_tokens],
                start_pos=attempt.initial_prefix_len,
            )
        self.tree_cache.unlock(attempt.prefix_lock)
        attempt.prefix_lock = None
        return len(attempt.key) - result.prefix_len

    def _abort_attempt(self, attempt: _ImportAttempt) -> None:
        if attempt.allocated_indices.numel():
            self.token_to_kv_pool_allocator.free_segment(
                attempt.allocated_indices, start_pos=attempt.initial_prefix_len
            )
        self.tree_cache.unlock(attempt.prefix_lock)
        attempt.prefix_lock = None

    def _complete_attempt(
        self,
        attempt: _ImportAttempt,
        *,
        status: str,
        reason: str,
        imported_tokens: int,
    ) -> None:
        attempt.terminal_message = build_reverse_kv_response_message(
            attempt_id=attempt.attempt_id,
            status=status,
            reason=reason,
            imported_tokens=imported_tokens,
        )
        self._send_response(attempt.peer_name, attempt.terminal_message)
        attempt.key = None
        attempt.initial_prefix_len = 0
        attempt.prefix_lock = None
        attempt.allocated_indices = None
        attempt.future = None

    def _send_terminal(
        self, *, peer_name: str, attempt_id: str, status: str, reason: str
    ) -> None:
        self._send_response(
            peer_name,
            build_reverse_kv_response_message(
                attempt_id=attempt_id,
                status=status,
                reason=reason,
            ),
        )

    def _send_response(self, peer_name: str, message: list[bytes]) -> None:
        peer = self.kv_manager.decode_kv_args_table.get(peer_name)
        if peer is None:
            logger.warning(
                "Cannot send reverse-KV response to unknown peer %s", peer_name
            )
            return
        endpoint = NetworkAddress(peer.endpoint, peer.dst_port).to_tcp()
        try:
            self.kv_manager._send_multipart_locked(endpoint, message)
        except Exception:
            logger.exception("Failed to send reverse-KV response to %s", peer_name)

    def _cleanup_expired(self) -> None:
        now = time.time()
        active_rooms = {
            attempt.ticket.bootstrap_room
            for attempt in self._attempts.values()
            if attempt.terminal_message is None
        }
        for room, ticket in list(self._tickets.items()):
            if ticket.expires_at <= now and room not in active_rooms:
                del self._tickets[room]
        for attempt_id, attempt in list(self._attempts.items()):
            if (
                attempt.terminal_message is not None
                and attempt.ticket.expires_at <= now
            ):
                del self._attempts[attempt_id]


class ReverseKVReturnManager:
    def __init__(
        self,
        *,
        kv_manager,
        req_to_token_pool: ReqToTokenPool,
        token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator,
        request_retirer: RequestRetirer,
        device_module,
        forward_stream,
    ) -> None:
        if kv_manager.kv_args.num_draft_entries:
            raise ValueError("Reverse KV return does not support draft KV entries")
        if kv_manager.kv_args.state_types:
            raise ValueError(
                "Reverse KV return does not support recurrent or SWA state"
            )

        cfg = get_disagg()
        self.kv_manager = kv_manager
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        self.request_retirer = request_retirer
        self.device_module = device_module
        self.forward_stream = forward_stream
        self.page_size = kv_manager.kv_args.page_size
        self.min_tokens = cfg.disaggregation_reverse_kv_min_tokens
        self.max_tokens = cfg.disaggregation_reverse_kv_max_tokens
        self.control_timeout = cfg.disaggregation_reverse_kv_control_timeout
        self.fingerprint = reverse_kv_fingerprint_from_args(kv_manager.kv_args)
        self.queue = IndexedReqList(ListKind.PD_RETURNING, abort=self._abort)
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=_MAX_CONCURRENT_RETURNS,
            thread_name_prefix="reverse-kv-return",
        )
        self._futures: Dict[Req, concurrent.futures.Future] = {}
        self._responses: Dict[str, queue.Queue[dict]] = {}
        self._responses_lock = threading.Lock()
        self._admission_disabled = False
        kv_manager.reverse_kv_return_manager = self

    def request_lists(self) -> Dict[ListKind, ReqList]:
        return {ListKind.PD_RETURNING: self.queue}

    def claim(self, req: Req) -> bool:
        if (
            self._admission_disabled
            or len(self._futures) >= _MAX_CONCURRENT_RETURNS
            or not self._eligible(req)
        ):
            return False

        return_len = req.owned_kv_len()
        if return_len > self.max_tokens:
            return False
        return_len = return_len // self.page_size * self.page_size
        if return_len <= 0:
            return False

        req.refresh_fill_ids()
        token_ids = np.asarray(
            req.full_untruncated_fill_ids[:return_len], dtype=np.int64
        )
        source_indices = self.req_to_token_pool.req_to_token[
            req.kv.req_pool_idx, :return_len
        ]
        source_indices = (
            self.token_to_kv_pool_allocator.translate_kv_indices_for_transfer(
                source_indices
            )
        )
        source_indices = source_indices.to("cpu").numpy().astype(np.int32, copy=False)
        source_ready = self.device_module.Event()
        source_ready.record(self.forward_stream)

        move([req], self.queue)
        self._futures[req] = self._executor.submit(
            self._return,
            req,
            token_ids,
            source_indices,
            source_ready,
        )
        return True

    def _eligible(self, req: Req) -> bool:
        return (
            req.kv.holds_kv
            and not isinstance(req.finished_reason, FINISH_ABORT)
            and req.multimodal_inputs is None
            and req.input_embeds is None
            and req.positional_embed_overrides is None
            and req.session is None
            and req.lora_id is None
            and req.beam_group is None
        )

    def _return(
        self,
        req: Req,
        token_ids: np.ndarray,
        source_indices: np.ndarray,
        source_ready,
    ) -> tuple[bool, str]:
        bootstrap_addr = NetworkAddress(
            req.bootstrap_host, req.bootstrap_port
        ).to_host_port_str()
        try:
            response = requests.post(
                f"http://{bootstrap_addr}/query_reverse_kv_ticket",
                json={"bootstrap_room": req.bootstrap_room},
                timeout=self.control_timeout,
            )
        except Exception as error:
            return True, f"ticket lookup failed: {error}"
        if response.status_code == 404:
            return True, "ticket unavailable"
        if response.status_code != 200:
            return (
                True,
                f"ticket lookup returned {response.status_code}: {response.text}",
            )

        try:
            ticket = response.json()
            if ticket["version"] != REVERSE_KV_PROTOCOL_VERSION:
                return True, "ticket protocol mismatch"
            if time.time() >= float(ticket["expires_at"]):
                return True, "ticket expired"
            if ticket["fingerprint"] != self.fingerprint:
                return True, "ticket layout mismatch"
            if len(token_ids) - int(ticket["prompt_tokens"]) < self.min_tokens:
                return True, "generated suffix below threshold"
            control_endpoint = str(ticket["control_endpoint"])
            nonce = str(ticket["nonce"])
        except Exception as error:
            return True, f"invalid ticket: {error}"

        attempt_id = uuid.uuid4().hex
        response_queue: queue.Queue[dict] = queue.Queue(maxsize=1)
        submitted = False
        with self._responses_lock:
            self._responses[attempt_id] = response_queue
        try:
            message = build_reverse_kv_prepare_message(
                attempt_id=attempt_id,
                bootstrap_room=req.bootstrap_room,
                nonce=nonce,
                fingerprint=self.fingerprint,
                agent_name=self.kv_manager.agent.name,
                token_ids=token_ids,
                source_indices=source_indices,
                extra_key=req.extra_key,
                cache_salt=req.cache_salt,
            )
            source_ready.synchronize()
            self.kv_manager._send_multipart_locked(control_endpoint, message)
            submitted = True
            try:
                terminal = response_queue.get(timeout=2 * self.control_timeout)
            except queue.Empty:
                return (
                    False,
                    "terminal response timed out after control submission",
                )
            return True, f"{terminal['status']}: {terminal['reason']}"
        except Exception as error:
            return (
                not submitted,
                f"control submission failed: {error}",
            )
        finally:
            with self._responses_lock:
                self._responses.pop(attempt_id, None)

    def enqueue_response(self, message: list[bytes]) -> None:
        try:
            response = parse_reverse_kv_response_message(message)
        except Exception as error:
            logger.warning("Dropped malformed reverse-KV response: %s", error)
            return
        with self._responses_lock:
            response_queue = self._responses.get(response["attempt_id"])
        if response_queue is None:
            return
        try:
            response_queue.put_nowait(response)
        except queue.Full:
            pass

    def poll(self) -> None:
        for req, future in list(self._futures.items()):
            if not future.done():
                continue
            try:
                safe_to_release, outcome = future.result()
                logger.info(
                    "Reverse KV return finished for room %s: %s",
                    req.bootstrap_room,
                    outcome,
                )
                if safe_to_release:
                    self.request_retirer.release(req, checkpoint=False)
                else:
                    self._admission_disabled = True
                    logger.error(
                        "Reverse KV return for room %s has no terminal response; "
                        "retaining its source pages and disabling new returns",
                        req.bootstrap_room,
                    )
            except Exception as error:
                self._admission_disabled = True
                logger.error(
                    "Reverse KV return for room %s failed unexpectedly; retaining "
                    "its source pages and disabling new returns",
                    req.bootstrap_room,
                    exc_info=error,
                )
            finally:
                del self._futures[req]

    def _abort(self, recv_req) -> None:
        for req in self.queue:
            if recv_req.matches(req.rid):
                logger.info(
                    "Ignoring abort for completed request %s while reverse KV is "
                    "retaining its source pages",
                    req.rid,
                )
