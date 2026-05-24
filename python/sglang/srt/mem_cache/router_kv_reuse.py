from __future__ import annotations

import atexit
import base64
import json
import logging
import socket
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import TYPE_CHECKING, Any, Dict, Iterable, Mapping, Optional, Protocol
from urllib.parse import urlparse

import numpy as np
import torch

from sglang.srt.mem_cache.base_prefix_cache import EvictParams, InsertParams
from sglang.srt.mem_cache.g2plus_transfer import (
    G2plusTransferBackend,
    g2plus_config,
    g2plus_config_value,
    g2plus_timeout_secs,
    g2plus_transfer_backend_name,
    make_g2plus_transfer_backend,
)
from sglang.srt.mem_cache.radix_cache import RadixKey, TreeNode
from sglang.srt.mem_cache.utils import (
    block_hash_aliases,
    compute_node_hash_values,
    hash_str_to_int64,
)
from sglang.srt.environ import envs

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)

REMOTE_KV_REUSE_PLAN_EXTRA_ARGS_KEY = "remote_kv_reuse_plan"
REMOTE_KV_REUSE_NO_PLAN_REASON_EXTRA_ARGS_KEY = "remote_kv_reuse_no_plan_reason"
REMOTE_KV_REUSE_PLAN_VERSION = 1
REMOTE_KV_REUSE_MAX_CONTROL_BODY_BYTES = 16 * 1024 * 1024
REMOTE_KV_REUSE_DIRECT_TIMEOUT_REASON = "source_transfer_timeout_maybe_inflight"

_REMOTE_G2_TIERS = {"g2", "host_pinned", "hostpinned", "cpu_pinned", "cpu_tier1"}


def _now_ms() -> int:
    return int(time.time() * 1000)


def _normalize_tier(tier: str) -> str:
    return str(tier).strip().lower().replace("-", "_")


def _normalize_metric_label(value: Any, default: str = "unknown") -> str:
    value = str(value or default).split(":", 1)[0].strip().lower()
    chars = [ch if ch.isalnum() or ch == "_" else "_" for ch in value]
    value = "".join(chars).strip("_")
    return (value or default)[:80]


def _is_remote_g2_tier(tier: str) -> bool:
    return _normalize_tier(tier) in _REMOTE_G2_TIERS


def _coerce_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be an integer, got {value!r}")
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, str):
        value = value.strip()
        if not value:
            raise ValueError(f"{field_name} must be an integer, got empty string")
        try:
            return int(value, 10)
        except ValueError as err:
            raise ValueError(
                f"{field_name} must be an integer, got {value!r}"
            ) from err
    raise ValueError(f"{field_name} must be an integer, got {value!r}")


def _coerce_array(value: Any, field_name: str) -> list[Any]:
    if isinstance(value, (str, bytes, Mapping)):
        raise ValueError(f"{field_name} must be an array")
    try:
        return list(value)
    except TypeError as err:
        raise ValueError(f"{field_name} must be an array") from err


def _coerce_block_hash(value: Any) -> int:
    if isinstance(value, Mapping):
        for key in ("block_hash", "hash", "value", "0"):
            if key in value:
                return _coerce_int(value[key], "block_hash")
        if len(value) == 1:
            return _coerce_int(next(iter(value.values())), "block_hash")
    return _coerce_int(value, "block_hash")


def _expand_block_hash_aliases(values: Iterable[int]) -> set[int]:
    aliases: set[int] = set()
    for value in values:
        aliases.update(block_hash_aliases(value))
    return aliases


def _first_present(data: Mapping[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        if name in data:
            return data[name]
    return default


@dataclass(frozen=True)
class RemoteKvReusePlan:
    plan_id: str
    request_id: str
    target_worker_id: int
    target_dp_rank: int
    source_worker_id: int
    source_dp_rank: int
    source_tier: str
    block_hashes: tuple[int, ...]
    planned_prefix_blocks: int
    block_size_tokens: int
    created_at_ms: int
    expires_at_ms: int
    start_block_index: int = 0
    plan_version: int = REMOTE_KV_REUSE_PLAN_VERSION
    kv_block_hashes: tuple[int, ...] = ()

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RemoteKvReusePlan":
        if not isinstance(data, Mapping):
            raise ValueError("remote KV reuse plan must be a mapping")

        block_hashes_raw = _first_present(data, "block_hashes", "hashes")
        if block_hashes_raw is None:
            raise ValueError("remote KV reuse plan missing block_hashes")
        block_hashes = tuple(
            _coerce_block_hash(item)
            for item in _coerce_array(block_hashes_raw, "block_hashes")
        )
        kv_block_hashes_raw = _first_present(
            data, "kv_block_hashes", "source_block_hashes", default=()
        )
        if kv_block_hashes_raw is None:
            kv_block_hashes_raw = ()
        kv_block_hashes = tuple(
            _coerce_block_hash(item)
            for item in _coerce_array(kv_block_hashes_raw, "kv_block_hashes")
        )
        if kv_block_hashes and len(kv_block_hashes) != len(block_hashes):
            raise ValueError(
                "kv_block_hashes length must match block_hashes when provided"
            )

        planned_prefix_blocks = _coerce_int(
            _first_present(
                data,
                "planned_prefix_blocks",
                "planned_blocks",
                "num_blocks",
                default=len(block_hashes),
            ),
            "planned_prefix_blocks",
        )
        if planned_prefix_blocks < 0:
            raise ValueError("planned_prefix_blocks must be non-negative")
        start_block_index = _coerce_int(
            _first_present(data, "start_block_index", default=0),
            "start_block_index",
        )
        if start_block_index < 0:
            raise ValueError("start_block_index must be non-negative")

        try:
            return cls(
                plan_id=str(_first_present(data, "plan_id", default="")),
                request_id=str(_first_present(data, "request_id", default="")),
                target_worker_id=_coerce_int(
                    data["target_worker_id"], "target_worker_id"
                ),
                target_dp_rank=_coerce_int(
                    _first_present(data, "target_dp_rank", default=0),
                    "target_dp_rank",
                ),
                source_worker_id=_coerce_int(
                    data["source_worker_id"], "source_worker_id"
                ),
                source_dp_rank=_coerce_int(
                    _first_present(data, "source_dp_rank", default=0),
                    "source_dp_rank",
                ),
                source_tier=str(
                    _first_present(data, "source_tier", default="host_pinned")
                ),
                block_hashes=block_hashes,
                planned_prefix_blocks=min(planned_prefix_blocks, len(block_hashes)),
                block_size_tokens=_coerce_int(
                    _first_present(data, "block_size_tokens", "block_size"),
                    "block_size_tokens",
                ),
                created_at_ms=_coerce_int(
                    _first_present(data, "created_at_ms", default=0), "created_at_ms"
                ),
                expires_at_ms=_coerce_int(data["expires_at_ms"], "expires_at_ms"),
                start_block_index=start_block_index,
                plan_version=_coerce_int(
                    _first_present(
                        data, "plan_version", default=REMOTE_KV_REUSE_PLAN_VERSION
                    ),
                    "plan_version",
                ),
                kv_block_hashes=kv_block_hashes,
            )
        except KeyError as err:
            raise ValueError(f"remote KV reuse plan missing {err.args[0]}") from err

    def to_dict(self) -> Dict[str, Any]:
        value = asdict(self)
        value["block_hashes"] = list(self.block_hashes)
        value["kv_block_hashes"] = list(self.kv_block_hashes)
        return value

    @property
    def planned_hashes(self) -> tuple[int, ...]:
        return self.block_hashes[: self.planned_prefix_blocks]

    @property
    def planned_kv_block_hashes(self) -> tuple[int, ...]:
        return self.kv_block_hashes[: self.planned_prefix_blocks]

    def is_remote_g2(self) -> bool:
        return _is_remote_g2_tier(self.source_tier)

    def is_expired(self, now_ms: Optional[int] = None) -> bool:
        return self.expires_at_ms <= (now_ms if now_ms is not None else _now_ms())


@dataclass(frozen=True)
class ResolvedHostPage:
    block_hash: int
    hash_value: str
    data: bytes


@dataclass(frozen=True)
class ResolvedHostPageLocation:
    block_hash: int
    hash_value: str
    host_index: int


@dataclass(frozen=True)
class RouterKVReuseResult:
    staged_tokens: int = 0
    pending: bool = False


@dataclass
class _RemoteG2PendingFetch:
    plan: RemoteKvReusePlan
    plan_offset: int
    target_start_block: int
    expected_hashes: tuple[int, ...]
    future: Future
    device_indices: Optional[torch.Tensor] = None
    locked_node: Optional[TreeNode] = None
    backend: str = "unknown"
    bytes_per_page: int = 0
    submitted_at: float = 0.0


def _tensor_to_bytes(tensor: torch.Tensor) -> bytes:
    tensor = tensor.detach().cpu().contiguous()
    return tensor.view(torch.uint8).numpy().tobytes()


def _tensor_from_bytes(
    raw: bytes, dtype: torch.dtype, expected_numel: int
) -> torch.Tensor:
    tensor = torch.frombuffer(bytearray(raw), dtype=dtype)
    if tensor.numel() != expected_numel:
        raise ValueError(
            f"remote host page has {tensor.numel()} elements, expected {expected_numel}"
        )
    return tensor


def _iter_tree_nodes(root: TreeNode) -> Iterable[TreeNode]:
    stack = list(root.children.values())
    while stack:
        node = stack.pop()
        yield node
        stack.extend(node.children.values())


def _build_host_block_index(
    tree_cache, wanted_hashes: set[int]
) -> Dict[int, tuple[TreeNode, int, str]]:
    lookup_index = getattr(tree_cache, "lookup_router_kv_host_blocks", None)
    if lookup_index is not None:
        index = lookup_index(wanted_hashes)
        if len(index) >= len(wanted_hashes):
            return index
        if getattr(tree_cache, "router_kv_block_index", None) is not None:
            return index
        wanted_hashes = wanted_hashes - set(index.keys())
    else:
        index: Dict[int, tuple[TreeNode, int, str]] = {}

    wanted_hashes = _expand_block_hash_aliases(wanted_hashes)
    page_size = tree_cache.page_size
    for node in _iter_tree_nodes(tree_cache.root_node):
        if node.host_value is None:
            continue
        if node.hash_value is None:
            node.hash_value = compute_node_hash_values(node, page_size)

        num_pages = min(len(node.hash_value), len(node.host_value) // page_size)
        for page_idx in range(num_pages):
            hash_value = node.hash_value[page_idx]
            block_hash = hash_str_to_int64(hash_value)
            for alias in block_hash_aliases(block_hash):
                if alias in wanted_hashes and alias not in index:
                    index[alias] = (node, page_idx, hash_value)
        if len(index) >= len(wanted_hashes):
            break
    return index


def _host_lookup_guard(tree_cache):
    return getattr(tree_cache, "router_kv_lock", nullcontext())


def _flush_hicache_write_through_acks(tree_cache) -> None:
    flush = getattr(tree_cache, "flush_write_through_acks", None)
    if callable(flush):
        flush()


def _host_page_start_indices(
    entries: list[tuple[int, str, TreeNode, int]], page_size: int
) -> list[int]:
    host_indices = [0] * len(entries)
    grouped_entries: dict[int, tuple[TreeNode, list[tuple[int, int]]]] = {}
    for output_idx, (_, _, node, page_idx) in enumerate(entries):
        group = grouped_entries.get(node.id)
        if group is None:
            grouped_entries[node.id] = (node, [(output_idx, page_idx)])
        else:
            group[1].append((output_idx, page_idx))

    for node, refs in grouped_entries.values():
        offsets = [page_idx * page_size for _, page_idx in refs]
        starts = node.host_value[offsets].detach().cpu().tolist()
        for (output_idx, _), host_index in zip(refs, starts):
            host_indices[output_idx] = int(host_index)

    return host_indices


def resolve_host_pages(
    tree_cache,
    plan: RemoteKvReusePlan,
    *,
    start_block: int,
    max_blocks: int,
    worker_id: Optional[int],
    dp_rank: int,
) -> tuple[list[ResolvedHostPage], str]:
    if worker_id is None:
        return [], "missing_source_worker_id"
    if plan.source_worker_id != worker_id:
        return [], "wrong_source_worker"
    if plan.source_dp_rank != dp_rank:
        return [], "wrong_source_dp_rank"
    if plan.is_expired():
        return [], "plan_expired"
    if not plan.is_remote_g2():
        return [], "unsupported_source_tier"
    if plan.block_size_tokens != tree_cache.page_size:
        return [], "incompatible_block_size"

    if start_block < 0 or max_blocks <= 0:
        return [], "empty_request"

    identity_hashes = plan.planned_hashes
    kv_hashes = plan.planned_kv_block_hashes or identity_hashes
    if start_block >= len(identity_hashes):
        return [], "already_local"

    requested_identity_hashes = identity_hashes[start_block : start_block + max_blocks]
    requested_kv_hashes = kv_hashes[start_block : start_block + max_blocks]
    pages: list[ResolvedHostPage] = []
    entries: list[tuple[int, str, TreeNode, int]] = []
    protected_nodes: list[TreeNode] = []
    protected_ids: set[int] = set()
    _flush_hicache_write_through_acks(tree_cache)
    with _host_lookup_guard(tree_cache):
        block_index = _build_host_block_index(
            tree_cache, set(requested_kv_hashes)
        )
        try:
            reason = "ok"
            for identity_hash, kv_hash in zip(
                requested_identity_hashes, requested_kv_hashes
            ):
                entry = block_index.get(kv_hash)
                if entry is None:
                    reason = "partial" if entries else "missing_first_block"
                    break
                node, page_idx, hash_value = entry
                if node.id not in protected_ids:
                    node.protect_host()
                    protected_nodes.append(node)
                    protected_ids.add(node.id)
                entries.append((identity_hash, hash_value, node, page_idx))

            for (identity_hash, hash_value, _, _), page_start in zip(
                entries,
                _host_page_start_indices(entries, tree_cache.page_size),
            ):
                data_page = tree_cache.cache_controller.mem_pool_host.get_data_page(
                    page_start, flat=True
                )
                pages.append(
                    ResolvedHostPage(
                        block_hash=identity_hash,
                        hash_value=hash_value,
                        data=_tensor_to_bytes(data_page),
                    )
                )
            if reason != "ok":
                return pages, reason
        finally:
            for node in protected_nodes:
                try:
                    node.release_host()
                except RuntimeError:
                    logger.exception(
                        "Failed to release remote G2 source host page protection"
                    )

    return pages, "ok"


def _resolve_host_page_locations(
    tree_cache,
    plan: RemoteKvReusePlan,
    *,
    start_block: int,
    max_blocks: int,
    worker_id: Optional[int],
    dp_rank: int,
) -> tuple[list[ResolvedHostPageLocation], str, list[TreeNode]]:
    if worker_id is None:
        return [], "missing_source_worker_id", []
    if plan.source_worker_id != worker_id:
        return [], "wrong_source_worker", []
    if plan.source_dp_rank != dp_rank:
        return [], "wrong_source_dp_rank", []
    if plan.is_expired():
        return [], "plan_expired", []
    if not plan.is_remote_g2():
        return [], "unsupported_source_tier", []
    if plan.block_size_tokens != tree_cache.page_size:
        return [], "incompatible_block_size", []

    if start_block < 0 or max_blocks <= 0:
        return [], "empty_request", []

    identity_hashes = plan.planned_hashes
    kv_hashes = plan.planned_kv_block_hashes or identity_hashes
    if start_block >= len(identity_hashes):
        return [], "already_local", []

    requested_identity_hashes = identity_hashes[start_block : start_block + max_blocks]
    requested_kv_hashes = kv_hashes[start_block : start_block + max_blocks]
    entries: list[tuple[int, str, TreeNode, int]] = []
    protected_nodes: list[TreeNode] = []
    protected_ids: set[int] = set()
    _flush_hicache_write_through_acks(tree_cache)
    with _host_lookup_guard(tree_cache):
        block_index = _build_host_block_index(
            tree_cache, set(requested_kv_hashes)
        )
        reason = "ok"
        for identity_hash, kv_hash in zip(
            requested_identity_hashes, requested_kv_hashes
        ):
            entry = block_index.get(kv_hash)
            if entry is None:
                reason = "partial" if entries else "missing_first_block"
                break
            node, page_idx, hash_value = entry
            if node.id not in protected_ids:
                node.protect_host()
                protected_nodes.append(node)
                protected_ids.add(node.id)
            entries.append((identity_hash, hash_value, node, page_idx))

        pages = [
            ResolvedHostPageLocation(
                block_hash=identity_hash,
                hash_value=hash_value,
                host_index=host_index,
            )
            for (identity_hash, hash_value, _, _), host_index in zip(
                entries,
                _host_page_start_indices(entries, tree_cache.page_size),
            )
        ]
        if reason != "ok":
            return pages, reason, protected_nodes

    return pages, "ok", protected_nodes


def _normalize_endpoint(endpoint: str) -> str:
    endpoint = endpoint.strip()
    if not endpoint:
        return endpoint
    if "://" not in endpoint:
        endpoint = f"http://{endpoint}"
    return endpoint.rstrip("/")


def _format_optional_ms(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def _is_timeout_error(err: BaseException) -> bool:
    if isinstance(err, TimeoutError):
        return True
    if isinstance(err, urllib.error.URLError):
        reason = getattr(err, "reason", None)
        if isinstance(reason, TimeoutError):
            return True
    return "timed out" in str(err).lower()


def _is_indeterminate_direct_transfer_reason(reason: str) -> bool:
    return str(reason).startswith(REMOTE_KV_REUSE_DIRECT_TIMEOUT_REASON)


def _endpoint_to_bind(endpoint: str) -> tuple[str, int]:
    parsed = urlparse(_normalize_endpoint(endpoint))
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"unsupported remote G2 endpoint scheme: {parsed.scheme}")
    if parsed.hostname is None or parsed.port is None:
        raise ValueError(f"remote G2 endpoint must include host and port: {endpoint}")
    return parsed.hostname, parsed.port


def _select_dp_endpoint(endpoint_spec: object, dp_rank: int) -> Optional[str]:
    if not endpoint_spec:
        return None
    if isinstance(endpoint_spec, Mapping):
        endpoint = endpoint_spec.get(str(dp_rank))
        if endpoint is None:
            return None
        if not isinstance(endpoint, str) or not endpoint.strip():
            raise ValueError(
                "g2plus_config.endpoint values must be non-empty strings; "
                f"got {endpoint!r} for dp_rank={dp_rank}"
            )
        return _normalize_endpoint(endpoint)
    if not isinstance(endpoint_spec, str):
        raise ValueError("g2plus_config.endpoint must be a string or JSON object")
    spec = endpoint_spec.strip()
    if not spec:
        return None
    if spec.startswith("{"):
        endpoints = json.loads(spec)
        if not isinstance(endpoints, Mapping):
            raise ValueError("g2plus_config.endpoint must be a JSON object")
        endpoint = endpoints.get(str(dp_rank))
        if endpoint is None:
            return None
        if not isinstance(endpoint, str) or not endpoint.strip():
            raise ValueError(
                "g2plus_config.endpoint values must be non-empty strings; "
                f"got {endpoint!r} for dp_rank={dp_rank}"
            )
        return _normalize_endpoint(endpoint)
    if "{dp_rank}" in spec:
        spec = spec.format(dp_rank=dp_rank)
    return _normalize_endpoint(spec)


def _parse_peer_endpoints(peer_endpoints: object) -> Dict[str, str]:
    if not peer_endpoints:
        return {}
    if isinstance(peer_endpoints, Mapping):
        raw = peer_endpoints
    elif isinstance(peer_endpoints, str):
        raw = json.loads(peer_endpoints)
    else:
        raise ValueError("g2plus_config.peer_endpoints must be a JSON object")
    if not isinstance(raw, Mapping):
        raise ValueError("g2plus_config.peer_endpoints must be a JSON object")
    parsed: Dict[str, str] = {}
    for key, endpoint in raw.items():
        if not isinstance(endpoint, str) or not endpoint.strip():
            raise ValueError(
                "g2plus_config.peer_endpoints values must be non-empty strings; "
                f"got {endpoint!r} for {key!r}"
            )
        parsed[str(key)] = _normalize_endpoint(endpoint)
    return parsed


class _ReusableThreadingHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True

    def get_request(self):
        request, client_address = super().get_request()
        try:
            request.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except OSError:
            logger.debug("Failed to set TCP_NODELAY on G2plus resolver socket")
        return request, client_address


def _router_kv_reuse_enabled(server_args: "ServerArgs") -> bool:
    config = g2plus_config(server_args)
    return bool(
        getattr(server_args, "enable_router_kv_reuse", False)
        or config
        or getattr(server_args, "enable_g2plus", False)
    )


class RouterKVReuseHandler(Protocol):
    def has_reuse_plan(self, req: "Req") -> bool: ...

    def has_pending(self) -> bool: ...

    def prefetch_remote_prefix(self, req: "Req") -> RouterKVReuseResult: ...

    def check_remote_prefix(self, req: "Req") -> RouterKVReuseResult: ...

    def maybe_stage_remote_prefix(self, req: "Req") -> int: ...

    def release_request(self, rid: str) -> None: ...


class RemoteG2ReuseHandler:
    def __init__(
        self,
        *,
        server_args: "ServerArgs",
        tree_cache,
        worker_id: Optional[int],
        dp_rank: int,
        direct_transfer: Optional[G2plusTransferBackend] = None,
        direct_transfer_diagnostics: Optional[list[str]] = None,
        metrics_collector=None,
    ):
        self.tree_cache = tree_cache
        self.worker_id = worker_id
        self.dp_rank = dp_rank
        self.timeout_secs = g2plus_timeout_secs(server_args)
        self.direct_transfer = direct_transfer
        self.metrics_collector = metrics_collector
        self.control_backend = str(
            g2plus_config_value(server_args, "control_backend", "http")
        ).lower()
        self.allow_http_staging = g2plus_transfer_backend_name(server_args) == "http"
        if not self._direct_transfer_enabled() and not self.allow_http_staging:
            diagnostic_suffix = ""
            if direct_transfer_diagnostics:
                diagnostic_suffix = (
                    " Diagnostics: " + "; ".join(direct_transfer_diagnostics)
                )
            logger.warning(
                "Router KV reuse is enabled but no direct G2plus transfer backend "
                "is available; remote KV reuse plans will be treated as cache misses.%s",
                diagnostic_suffix,
            )
        http_control = g2plus_config_value(server_args, "http_control", {}) or {}
        if not isinstance(http_control, Mapping):
            http_control = {}
        endpoint_spec = http_control.get(
            "endpoint", getattr(server_args, "g2plus_endpoint", None)
        )
        static_peer_endpoints = http_control.get(
            "static_peer_endpoints",
            getattr(server_args, "g2plus_peer_endpoints", None),
        )
        self.endpoint = _select_dp_endpoint(endpoint_spec, dp_rank)
        self.static_peer_endpoints = _parse_peer_endpoints(static_peer_endpoints)
        self._source_server: Optional[ThreadingHTTPServer] = None
        self._source_thread: Optional[threading.Thread] = None
        self._shutdown = False
        worker_limit = max(
            1,
            int(
                getattr(
                    server_args,
                    "g2plus_fetch_workers",
                    envs.SGLANG_G2PLUS_FETCH_WORKERS.get(),
                )
            ),
        )
        self._source_activity_lock = threading.Lock()
        self._active_source_resolver_ops = 0
        self._source_resolver_semaphore = threading.BoundedSemaphore(worker_limit)
        self._fetch_semaphore = threading.BoundedSemaphore(worker_limit)
        self._fetch_executor = ThreadPoolExecutor(
            max_workers=worker_limit,
            thread_name_prefix=f"g2plus-fetch-dp{dp_rank}",
        )
        self._pending_fetches: dict[str, _RemoteG2PendingFetch] = {}
        self._detached_fetches: set[Future] = set()
        self._quarantined_device_indices: list[torch.Tensor] = []
        self._quarantined_tokens_by_backend: dict[str, int] = {}
        self._finished_plan_keys: set[tuple[str, str]] = set()
        self.max_control_body_bytes = REMOTE_KV_REUSE_MAX_CONTROL_BODY_BYTES
        self._direct_transfer_shutdown_done = False
        self._direct_transfer_shutdown_deferred = False
        self._direct_transfer_shutdown_lock = threading.Lock()

        if self.endpoint is not None:
            self._start_source_resolver()
        atexit.register(self.shutdown)

    @classmethod
    def from_scheduler(cls, scheduler) -> Optional["RemoteG2ReuseHandler"]:
        server_args = scheduler.server_args
        if not _router_kv_reuse_enabled(server_args):
            return None
        if not scheduler.enable_hierarchical_cache:
            logger.warning(
                "Router KV reuse disabled because hierarchical cache is not enabled"
            )
            return None
        if not hasattr(scheduler.tree_cache, "_insert_helper_host"):
            logger.warning(
                "Router KV reuse disabled because the active tree cache does not support host inserts"
            )
            return None
        worker_id = g2plus_config_value(
            server_args, "worker_id", getattr(server_args, "g2plus_worker_id", None)
        )
        if worker_id is None:
            logger.warning(
                "Router KV reuse disabled because G2plus worker identity is not set; "
                "set worker_id in --g2plus-config when enabling G2plus remote KV reuse"
            )
            return None
        transfer_diagnostics: list[str] = []
        direct_transfer = make_g2plus_transfer_backend(
            scheduler, diagnostics=transfer_diagnostics
        )
        return cls(
            server_args=server_args,
            tree_cache=scheduler.tree_cache,
            worker_id=worker_id,
            dp_rank=scheduler.dp_rank or 0,
            direct_transfer=direct_transfer,
            direct_transfer_diagnostics=transfer_diagnostics,
            metrics_collector=(
                scheduler.metrics_collector
                if getattr(scheduler, "enable_metrics", False)
                else None
            ),
        )

    def _current_backend_label(self) -> str:
        if self._direct_transfer_enabled():
            direct_transfer = getattr(self, "direct_transfer", None)
            return str(getattr(direct_transfer, "name", "direct"))
        return "none"

    def _direct_transfer_enabled(self) -> bool:
        direct_transfer = getattr(self, "direct_transfer", None)
        return direct_transfer is not None and bool(
            getattr(direct_transfer, "enabled", False)
        )

    def _observe_reuse(
        self,
        *,
        backend: str,
        outcome: str,
        reason: str,
        tokens: int = 0,
        wait_ms: Optional[float] = None,
        insert_ms: Optional[float] = None,
        transfer_bytes: Optional[int] = None,
    ) -> None:
        metrics_collector = getattr(self, "metrics_collector", None)
        observe = getattr(metrics_collector, "observe_router_kv_reuse", None)
        if observe is None:
            return
        try:
            observe(
                backend=_normalize_metric_label(backend),
                outcome=_normalize_metric_label(outcome),
                reason=_normalize_metric_label(reason),
                tokens=max(0, int(tokens)),
                wait_ms=wait_ms,
                insert_ms=insert_ms,
                transfer_bytes=transfer_bytes,
            )
        except Exception:
            logger.debug("Failed to record router KV reuse metrics", exc_info=True)

    def _pending_wait_ms(self, pending: _RemoteG2PendingFetch) -> Optional[float]:
        submitted_at = getattr(pending, "submitted_at", 0.0)
        if submitted_at <= 0:
            return None
        return (time.perf_counter() - submitted_at) * 1000

    def _pending_ready_wait_ms(
        self, pending: _RemoteG2PendingFetch
    ) -> Optional[float]:
        done_at = getattr(pending.future, "_g2plus_done_at", 0.0)
        if done_at <= 0:
            return None
        return max(0.0, (time.perf_counter() - done_at) * 1000)

    def _transfer_bytes_for_pages(
        self, pending: _RemoteG2PendingFetch, pages: list[ResolvedHostPage]
    ) -> int:
        bytes_per_page = int(getattr(pending, "bytes_per_page", 0) or 0)
        if bytes_per_page > 0:
            return len(pages) * bytes_per_page
        return sum(len(page.data) for page in pages)

    def _max_control_body_bytes(self) -> int:
        return int(
            getattr(
                self,
                "max_control_body_bytes",
                REMOTE_KV_REUSE_MAX_CONTROL_BODY_BYTES,
            )
        )

    def _try_enter_source_resolver(self) -> bool:
        if not self._source_resolver_semaphore.acquire(blocking=False):
            return False
        if not hasattr(self, "_source_activity_lock"):
            self._source_activity_lock = threading.Lock()
            self._active_source_resolver_ops = 0
        with self._source_activity_lock:
            self._active_source_resolver_ops += 1
        return True

    def _exit_source_resolver(self) -> None:
        if not hasattr(self, "_source_activity_lock"):
            self._source_activity_lock = threading.Lock()
            self._active_source_resolver_ops = 0
        with self._source_activity_lock:
            self._active_source_resolver_ops -= 1
        self._source_resolver_semaphore.release()

    def _try_acquire_fetch_worker(self) -> bool:
        semaphore = getattr(self, "_fetch_semaphore", None)
        if semaphore is None:
            return True
        return semaphore.acquire(blocking=False)

    def _release_fetch_worker(self) -> None:
        semaphore = getattr(self, "_fetch_semaphore", None)
        if semaphore is None:
            return
        try:
            semaphore.release()
        except ValueError:
            logger.debug("G2plus fetch worker semaphore release ignored", exc_info=True)

    def _on_fetch_worker_done(self, future: Future) -> None:
        try:
            setattr(future, "_g2plus_done_at", time.perf_counter())
        finally:
            self._release_fetch_worker()

    def _submit_fetch_worker(self, fn, *args, **kwargs) -> Optional[Future]:
        if not self._try_acquire_fetch_worker():
            return None
        try:
            future = self._fetch_executor.submit(fn, *args, **kwargs)
        except Exception:
            self._release_fetch_worker()
            raise
        future.add_done_callback(self._on_fetch_worker_done)
        return future

    def _start_source_resolver(self) -> None:
        host, port = _endpoint_to_bind(self.endpoint)
        manager = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802
                request_start = time.perf_counter()
                if self.path not in {
                    "/transfer_direct",
                    "/transfer_mooncake",
                }:
                    self.send_error(404)
                    return
                if not manager._try_enter_source_resolver():
                    self._write_json(
                        503,
                        {
                            "ok": False,
                            "reason": "source_resolver_busy",
                            "pages": [],
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
                                "pages": [],
                            },
                        )
                        return
                    if content_len <= 0:
                        self._write_json(
                            400,
                            {
                                "ok": False,
                                "reason": "empty_request_body",
                                "pages": [],
                            },
                        )
                        return
                    if content_len > manager._max_control_body_bytes():
                        self._write_json(
                            413,
                            {
                                "ok": False,
                                "reason": "control_payload_too_large",
                                "pages": [],
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
                                "pages": [],
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
                                "pages": [],
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
                                "pages": [],
                            },
                        )
                        return
                    if self.path in {"/transfer_direct", "/transfer_mooncake"}:
                        if not manager._direct_transfer_enabled():
                            self._write_json(
                                501,
                                {
                                    "ok": False,
                                    "reason": "direct_transfer_unavailable",
                                    "pages": [],
                                },
                            )
                            return
                        pre_handler_ms = (
                            time.perf_counter() - request_start
                        ) * 1000
                        response = dict(manager._handle_source_transfer(payload))
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
                            "G2plus source control handled path=%s pre_handler_ms=%.3f body_read_ms=%.3f json_decode_ms=%.3f response_write_ms=%.3f request_total_ms=%.3f",
                            self.path,
                            pre_handler_ms,
                            body_read_ms,
                            decode_ms,
                            write_ms,
                            (time.perf_counter() - request_start) * 1000,
                        )
                        return
                except Exception as err:
                    logger.exception("Remote G2 source resolve failed")
                    self._write_json(
                        500, {"ok": False, "reason": str(err), "pages": []}
                    )
                finally:
                    manager._exit_source_resolver()

            def log_message(self, fmt, *args):
                logger.debug("Remote G2 source resolver: " + fmt, *args)

            def _write_json(self, status_code: int, response: Mapping[str, Any]):
                data = json.dumps(response).encode("utf-8")
                self.send_response(status_code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def _write_binary(
                self,
                status_code: int,
                response: Mapping[str, Any],
                pages: list[ResolvedHostPage],
            ):
                metadata = json.dumps(response).encode("utf-8")
                metadata_len = len(metadata).to_bytes(8, byteorder="little")
                data_len = sum(len(page.data) for page in pages)
                self.send_response(status_code)
                self.send_header(
                    "Content-Type", "application/vnd.sglang.remote-g2-pages"
                )
                self.send_header("Content-Length", str(8 + len(metadata) + data_len))
                self.end_headers()
                self.wfile.write(metadata_len)
                self.wfile.write(metadata)
                for page in pages:
                    self.wfile.write(page.data)

        self._source_server = _ReusableThreadingHTTPServer((host, port), Handler)
        self._source_thread = threading.Thread(
            target=self._source_server.serve_forever,
            name=f"g2plus-source-{host}:{port}",
            daemon=True,
        )
        self._source_thread.start()
        logger.info(
            "Remote G2 source resolver listening on %s for worker_id=%s dp_rank=%s",
            self.endpoint,
            self.worker_id,
            self.dp_rank,
        )

    def _shutdown_direct_transfer_backend(self) -> None:
        direct_transfer = getattr(self, "direct_transfer", None)
        shutdown = getattr(direct_transfer, "shutdown", None)
        if shutdown is None:
            return
        lock = getattr(self, "_direct_transfer_shutdown_lock", None)
        if lock is None:
            lock = threading.Lock()
            self._direct_transfer_shutdown_lock = lock
        with lock:
            if getattr(self, "_direct_transfer_shutdown_done", False):
                return
            shutdown()
            self._direct_transfer_shutdown_done = True

    def _defer_direct_transfer_shutdown(self) -> None:
        direct_transfer = getattr(self, "direct_transfer", None)
        if getattr(direct_transfer, "shutdown", None) is None:
            return
        if getattr(self, "_direct_transfer_shutdown_deferred", False):
            return
        self._direct_transfer_shutdown_deferred = True

        def _wait_for_pending_and_shutdown():
            while self.has_pending():
                time.sleep(0.01)
            try:
                self._release_quarantined_device_indices()
                self._shutdown_direct_transfer_backend()
            except Exception:
                logger.warning(
                    "G2plus deferred direct transfer backend shutdown failed",
                    exc_info=True,
                )

        thread = threading.Thread(
            target=_wait_for_pending_and_shutdown,
            name="g2plus-direct-transfer-shutdown",
            daemon=True,
        )
        self._direct_transfer_shutdown_thread = thread
        thread.start()

    def shutdown(self) -> None:
        if self._shutdown:
            return
        self._shutdown = True

        server = self._source_server
        self._source_server = None
        if server is not None:
            try:
                server.shutdown()
                server.server_close()
            except Exception:
                logger.debug("Remote G2 source resolver shutdown failed", exc_info=True)

        source_thread = self._source_thread
        self._source_thread = None
        if (
            source_thread is not None
            and source_thread is not threading.current_thread()
        ):
            try:
                source_thread.join(timeout=1)
            except Exception:
                logger.debug("Remote G2 source resolver join failed", exc_info=True)

        for pending in self._pending_fetches.values():
            self._release_pending_fetch(pending)
        self._pending_fetches.clear()
        self._fetch_executor.shutdown(wait=False, cancel_futures=True)

        timeout_secs = float(getattr(self, "timeout_secs", 0.0))
        deadline = time.monotonic() + min(max(timeout_secs, 0.0), 5.0)
        while self.has_pending() and time.monotonic() < deadline:
            time.sleep(0.01)

        if self.has_pending():
            logger.warning(
                "Deferring direct transfer backend shutdown while G2plus work is still pending"
            )
            self._defer_direct_transfer_shutdown()
            return

        self._release_quarantined_device_indices()
        self._shutdown_direct_transfer_backend()

    def _endpoint_for_plan(self, plan: RemoteKvReusePlan) -> Optional[str]:
        for endpoint in self._candidate_endpoints_for_plan(plan):
            return endpoint
        return None

    def _candidate_endpoints_for_plan(self, plan: RemoteKvReusePlan) -> list[str]:
        endpoints: list[str] = []

        def add(endpoint: Optional[str]) -> None:
            if endpoint and endpoint not in endpoints:
                endpoints.append(endpoint)

        endpoint = self.static_peer_endpoints.get(
            f"{plan.source_worker_id}:{plan.source_dp_rank}"
        )
        add(endpoint)
        add(self.static_peer_endpoints.get(str(plan.source_worker_id)))
        if (
            self.worker_id == plan.source_worker_id
            and self.dp_rank == plan.source_dp_rank
            and self.endpoint is not None
        ):
            add(self.endpoint)
        for endpoint in self.static_peer_endpoints.values():
            add(endpoint)
        return endpoints

    def _fetch_pages(
        self, plan: RemoteKvReusePlan, *, start_block: int, max_blocks: int
    ) -> tuple[list[ResolvedHostPage], str]:
        endpoints = self._candidate_endpoints_for_plan(plan)
        if not endpoints:
            return [], "missing_source_endpoint"

        body = json.dumps(
            {
                "plan": plan.to_dict(),
                "start_block": start_block,
                "max_blocks": max_blocks,
            }
        ).encode("utf-8")
        last_reason = "missing_source_endpoint"
        for endpoint in endpoints:
            try:
                pages, reason = self._fetch_pages_binary_from_endpoint(endpoint, body)
                if pages or reason in {"ok", "already_local"}:
                    return pages, reason
                last_reason = reason
            except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as err:
                last_reason = f"source_request_failed:{err}"
            except Exception as err:
                last_reason = f"source_binary_decode_failed:{err}"

            try:
                pages, reason = self._fetch_pages_json_from_endpoint(endpoint, body)
            except (urllib.error.URLError, TimeoutError) as err:
                last_reason = f"source_request_failed:{err}"
                continue

            last_reason = reason
            if pages or last_reason in {"ok", "already_local"}:
                return pages, last_reason

        return [], last_reason

    def _fetch_pages_json_from_endpoint(
        self, endpoint: str, body: bytes
    ) -> tuple[list[ResolvedHostPage], str]:
        request = urllib.request.Request(
            f"{endpoint}/resolve",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout_secs) as response:
            payload = json.loads(response.read().decode("utf-8"))

        reason = str(payload.get("reason", "ok"))
        if not payload.get("ok"):
            return [], reason

        return (
            [
                ResolvedHostPage(
                    block_hash=int(page["block_hash"]),
                    hash_value=str(page["hash_value"]),
                    data=base64.b64decode(page["data_b64"]),
                )
                for page in payload.get("pages", [])
            ],
            reason,
        )

    def _fetch_pages_binary_from_endpoint(
        self, endpoint: str, body: bytes
    ) -> tuple[list[ResolvedHostPage], str]:
        request = urllib.request.Request(
            f"{endpoint}/resolve_binary",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout_secs) as response:
            metadata_len_raw = response.read(8)
            if len(metadata_len_raw) != 8:
                raise ValueError("binary remote G2 response missing metadata header")
            metadata_len = int.from_bytes(metadata_len_raw, byteorder="little")
            payload = json.loads(response.read(metadata_len).decode("utf-8"))

            reason = str(payload.get("reason", "ok"))
            if not payload.get("ok"):
                return [], reason

            pages: list[ResolvedHostPage] = []
            for page in payload.get("pages", []):
                byte_length = int(page["byte_length"])
                data = response.read(byte_length)
                if len(data) != byte_length:
                    raise ValueError(
                        "binary remote G2 page truncated: "
                        f"got {len(data)} bytes, expected {byte_length}"
                    )
                pages.append(
                    ResolvedHostPage(
                        block_hash=int(page["block_hash"]),
                        hash_value=str(page["hash_value"]),
                        data=data,
                    )
                )
            return pages, reason

    def _pages_from_direct_transfer_payload(
        self,
        payload: Mapping[str, Any],
        plan: RemoteKvReusePlan,
        *,
        start_block: int,
        max_blocks: int,
    ) -> list[ResolvedHostPage]:
        page_payload = payload.get("pages")
        if page_payload is not None:
            return [
                ResolvedHostPage(
                    block_hash=int(page["block_hash"]),
                    hash_value=str(page.get("hash_value", "")),
                    data=b"",
                )
                for page in page_payload
            ]

        transferred_blocks = _coerce_int(
            payload.get("transferred_blocks", 0), "transferred_blocks"
        )
        if transferred_blocks < 0:
            raise ValueError("transferred_blocks must be non-negative")
        transferred_blocks = min(transferred_blocks, max_blocks)
        block_hashes = plan.planned_hashes[
            start_block : start_block + transferred_blocks
        ]
        return [
            ResolvedHostPage(block_hash=block_hash, hash_value="", data=b"")
            for block_hash in block_hashes
        ]

    def _request_source_transfer(
        self,
        *,
        transfer_backend: G2plusTransferBackend,
        endpoints: list[str],
        plan: RemoteKvReusePlan,
        start_block: int,
        max_blocks: int,
        target_page_indices: list[int],
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
            paths = ["/transfer_direct"]
            for path in paths:
                request = urllib.request.Request(
                    f"{endpoint}{path}",
                    data=body,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                request_start = time.perf_counter()
                http_read_ms = 0.0
                decode_ms = 0.0
                response_bytes = 0
                try:
                    with urllib.request.urlopen(
                        request, timeout=self.timeout_secs
                    ) as response:
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
                    if _is_timeout_error(err):
                        last_reason = f"{REMOTE_KV_REUSE_DIRECT_TIMEOUT_REASON}:{err}"
                        logger.warning(
                            "Remote G2 direct transfer timed out endpoint=%s path=%s ms=%.3f; target pages will be quarantined reason=%s",
                            endpoint,
                            path,
                            (time.perf_counter() - request_start) * 1000,
                            last_reason,
                        )
                        return [], last_reason
                    last_reason = f"source_transfer_failed:{err}"
                    logger.debug(
                        "Remote G2 direct transfer request failed endpoint=%s path=%s ms=%.3f request_encode_ms=%.3f http_read_ms=%.3f response_decode_ms=%.3f request_bytes=%d response_bytes=%d reason=%s",
                        endpoint,
                        path,
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
                        "Remote G2 direct transfer returned malformed response endpoint=%s path=%s ms=%.3f reason=%s",
                        endpoint,
                        path,
                        (time.perf_counter() - request_start) * 1000,
                        last_reason,
                    )
                    continue

                last_reason = str(payload.get("reason", "ok"))
                if not payload.get("ok"):
                    if _is_indeterminate_direct_transfer_reason(last_reason):
                        logger.warning(
                            "Remote G2 direct transfer rejected with indeterminate target-page state endpoint=%s path=%s ms=%.3f reason=%s",
                            endpoint,
                            path,
                            (time.perf_counter() - request_start) * 1000,
                            last_reason,
                        )
                        return [], last_reason
                    logger.debug(
                        "Remote G2 direct transfer rejected endpoint=%s path=%s ms=%.3f request_encode_ms=%.3f http_read_ms=%.3f response_decode_ms=%.3f request_bytes=%d response_bytes=%d source_resolve_ms=%s source_transfer_ms=%s source_total_ms=%s reason=%s",
                        endpoint,
                        path,
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
                    pages = self._pages_from_direct_transfer_payload(
                        payload,
                        plan,
                        start_block=start_block,
                        max_blocks=max_blocks,
                    )
                    parse_ms = (time.perf_counter() - parse_start) * 1000
                except (TypeError, KeyError, ValueError) as err:
                    last_reason = f"malformed_source_transfer_response:{err}"
                    logger.debug(
                        "Remote G2 direct transfer returned malformed pages endpoint=%s path=%s ms=%.3f request_encode_ms=%.3f http_read_ms=%.3f response_decode_ms=%.3f request_bytes=%d response_bytes=%d reason=%s",
                        endpoint,
                        path,
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
                    "Remote G2 direct transfer response endpoint=%s path=%s pages=%d ms=%.3f request_encode_ms=%.3f http_read_ms=%.3f response_decode_ms=%.3f response_parse_ms=%.3f request_bytes=%d response_bytes=%d source_resolve_ms=%s source_transfer_ms=%s source_total_ms=%s source_bytes=%s reason=%s",
                    endpoint,
                    path,
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

    def _parse_target_kv_metadata(
        self, payload: Mapping[str, Any], transfer_backend: G2plusTransferBackend
    ) -> tuple[Optional[str], Optional[list[int]], Optional[list[int]], Optional[str]]:
        try:
            target_session_id_raw = payload["target_session_id"]
            target_kv_ptrs_raw = payload["target_kv_ptrs"]
            target_kv_item_lens_raw = payload["target_kv_item_lens"]
        except KeyError as err:
            return None, None, None, f"target_kv_metadata_missing:{err}"

        target_session_id = str(target_session_id_raw)
        if not target_session_id or target_session_id_raw is None:
            return None, None, None, "target_session_id_empty"
        target_metadata = payload.get("target_metadata")
        if isinstance(target_metadata, Mapping) and "session_id" in target_metadata:
            metadata_session_id = str(target_metadata["session_id"])
            if not metadata_session_id or target_metadata["session_id"] is None:
                return None, None, None, "target_metadata_session_id_empty"
            if metadata_session_id != target_session_id:
                return None, None, None, "target_session_id_mismatch"

        try:
            target_kv_ptrs = self._coerce_transfer_int_list(
                target_kv_ptrs_raw, "target_kv_ptrs"
            )
            target_kv_item_lens = self._coerce_transfer_int_list(
                target_kv_item_lens_raw, "target_kv_item_lens"
            )
        except ValueError as err:
            return None, None, None, str(err)

        if not target_kv_ptrs:
            return None, None, None, "target_kv_ptrs_empty"
        if len(target_kv_ptrs) != len(target_kv_item_lens):
            return (
                None,
                None,
                None,
                f"target_kv_ptrs_len_{len(target_kv_ptrs)}!=target_kv_item_lens_len_{len(target_kv_item_lens)}",
            )

        uint64_max = int(np.iinfo(np.uint64).max)
        if any(ptr <= 0 or ptr > uint64_max for ptr in target_kv_ptrs):
            return None, None, None, "target_kv_ptr_out_of_range"
        if any(length <= 0 or length > uint64_max for length in target_kv_item_lens):
            return None, None, None, "target_kv_item_len_out_of_range"

        expected_item_lens = getattr(transfer_backend, "target_kv_item_lens", None)
        if expected_item_lens is not None:
            try:
                expected_item_lens = [int(length) for length in expected_item_lens]
            except (TypeError, ValueError) as err:
                return None, None, None, f"local_target_kv_item_lens:{err}"
            if len(expected_item_lens) != len(target_kv_item_lens):
                return (
                    None,
                    None,
                    None,
                    f"target_kv_item_lens_count_mismatch:expected={len(expected_item_lens)}:got={len(target_kv_item_lens)}",
                )
            for idx, (expected, actual) in enumerate(
                zip(expected_item_lens, target_kv_item_lens)
            ):
                if expected != actual:
                    return (
                        None,
                        None,
                        None,
                        "target_kv_item_lens_mismatch:"
                        f"idx={idx}:expected={expected}:got={actual}",
                    )

        return target_session_id, target_kv_ptrs, target_kv_item_lens, None

    def _coerce_transfer_int_list(self, raw: Any, field_name: str) -> list[int]:
        if isinstance(raw, (str, bytes, Mapping)):
            raise ValueError(f"{field_name}_must_be_array")
        try:
            values = list(raw)
        except TypeError as err:
            raise ValueError(f"{field_name}_must_be_array") from err
        return [
            self._coerce_transfer_int(value, f"{field_name}[{idx}]")
            for idx, value in enumerate(values)
        ]

    def _coerce_transfer_int(self, value: Any, field_name: str) -> int:
        if isinstance(value, bool):
            raise ValueError(f"{field_name}_contains_non_integer:{value!r}")
        if isinstance(value, (int, np.integer)):
            return int(value)
        if isinstance(value, str):
            value = value.strip()
            if not value:
                raise ValueError(f"{field_name}_contains_non_integer:empty")
            try:
                return int(value, 10)
            except ValueError as err:
                raise ValueError(
                    f"{field_name}_contains_non_integer:{value!r}"
                ) from err
        raise ValueError(f"{field_name}_contains_non_integer:{value!r}")

    def _handle_source_transfer(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        transfer_backend = self.direct_transfer
        if transfer_backend is None or not getattr(transfer_backend, "enabled", False):
            return {"ok": False, "reason": "direct_transfer_unavailable", "pages": []}
        requested_backend = str(payload.get("transfer_backend", "mooncake")).lower()
        if requested_backend != transfer_backend.name:
            return {
                "ok": False,
                "reason": (
                    f"unsupported_transfer_backend:{requested_backend}:"
                    f"local={transfer_backend.name}"
                ),
                "pages": [],
            }

        try:
            plan = RemoteKvReusePlan.from_dict(payload["plan"])
            start_block = _coerce_int(payload.get("start_block", 0), "start_block")
            max_blocks = _coerce_int(
                payload.get("max_blocks", len(plan.block_hashes)), "max_blocks"
            )
        except (KeyError, ValueError) as err:
            return {
                "ok": False,
                "reason": f"malformed_transfer_request:plan:{err}",
                "pages": [],
            }
        (
            target_session_id,
            target_kv_ptrs,
            target_kv_item_lens,
            target_kv_metadata_error,
        ) = self._parse_target_kv_metadata(payload, transfer_backend)
        if target_kv_metadata_error is not None:
            return {
                "ok": False,
                "reason": f"malformed_transfer_request:{target_kv_metadata_error}",
                "block_size_tokens": self.tree_cache.page_size,
                "pages": [],
            }
        try:
            target_page_indices_list = self._coerce_transfer_int_list(
                payload["target_page_indices"], "target_page_indices"
            )
        except KeyError as err:
            return {
                "ok": False,
                "reason": f"malformed_transfer_request:target_page_indices_missing:{err}",
                "block_size_tokens": self.tree_cache.page_size,
                "pages": [],
            }
        except ValueError as err:
            return {
                "ok": False,
                "reason": f"malformed_transfer_request:{err}",
                "block_size_tokens": self.tree_cache.page_size,
                "pages": [],
            }
        max_int32 = np.iinfo(np.int32).max
        if any(idx < 0 or idx > max_int32 for idx in target_page_indices_list):
            return {
                "ok": False,
                "reason": "malformed_transfer_request:target_page_index_out_of_range",
                "block_size_tokens": self.tree_cache.page_size,
                "pages": [],
            }
        total_start = time.perf_counter()
        resolve_start = total_start
        pages, reason, protected_nodes = _resolve_host_page_locations(
            self.tree_cache,
            plan,
            start_block=start_block,
            max_blocks=max_blocks,
            worker_id=self.worker_id,
            dp_rank=self.dp_rank,
        )
        resolve_ms = (time.perf_counter() - resolve_start) * 1000
        transfer_ms = 0.0
        transfer_bytes = 0
        try:
            if pages:
                page_size = self.tree_cache.page_size
                source_page_indices_list: list[int] = []
                max_int32 = np.iinfo(np.int32).max
                for page in pages:
                    host_index = int(page.host_index)
                    if host_index < 0:
                        return {
                            "ok": False,
                            "reason": "source_host_page_index_out_of_range",
                            "block_size_tokens": self.tree_cache.page_size,
                            "pages": [],
                        }
                    if host_index % page_size != 0:
                        return {
                            "ok": False,
                            "reason": "source_host_page_index_unaligned",
                            "block_size_tokens": self.tree_cache.page_size,
                            "pages": [],
                        }
                    page_index = host_index // page_size
                    if page_index > max_int32:
                        return {
                            "ok": False,
                            "reason": "source_page_index_out_of_range",
                            "block_size_tokens": self.tree_cache.page_size,
                            "pages": [],
                        }
                    source_page_indices_list.append(page_index)
                source_page_indices = np.array(
                    source_page_indices_list, dtype=np.int32
                )
                transfer_bytes = len(pages) * sum(int(x) for x in target_kv_item_lens)
                if len(target_page_indices_list) < len(pages):
                    return {
                        "ok": False,
                        "reason": (
                            "malformed_transfer_request:"
                            f"target_page_indices_too_short:{len(target_page_indices_list)}<{len(pages)}"
                        ),
                        "block_size_tokens": self.tree_cache.page_size,
                        "pages": [],
                    }
                target_page_indices_list = target_page_indices_list[: len(pages)]
                target_page_indices = np.array(
                    target_page_indices_list, dtype=np.int32
                )
                transfer_start = time.perf_counter()
                try:
                    transfer_backend.transfer_pages(
                        target_session_id=target_session_id,
                        source_page_indices=source_page_indices,
                        target_page_indices=target_page_indices,
                        target_kv_ptrs=target_kv_ptrs,
                        target_kv_item_lens=target_kv_item_lens,
                        target_metadata=payload.get("target_metadata"),
                    )
                except Exception as err:
                    transfer_ms = (time.perf_counter() - transfer_start) * 1000
                    if _is_timeout_error(err):
                        failure_reason = (
                            f"{REMOTE_KV_REUSE_DIRECT_TIMEOUT_REASON}:source:{err}"
                        )
                    else:
                        failure_reason = f"direct_transfer_failed:{err}"
                    logger.warning(
                        "G2plus source direct transfer failed pages=%d resolve_ms=%.3f transfer_ms=%.3f reason=%s",
                        len(pages),
                        resolve_ms,
                        transfer_ms,
                        err,
                        exc_info=True,
                    )
                    return {
                        "ok": False,
                        "reason": failure_reason,
                        "block_size_tokens": self.tree_cache.page_size,
                        "resolve_ms": resolve_ms,
                        "transfer_ms": transfer_ms,
                        "total_ms": (time.perf_counter() - total_start) * 1000,
                        "transfer_bytes": transfer_bytes,
                        "pages": [],
                    }
                transfer_ms = (time.perf_counter() - transfer_start) * 1000
            total_ms = (time.perf_counter() - total_start) * 1000
            logger.debug(
                "G2plus source transfer handled pages=%d reason=%s resolve_ms=%.3f transfer_ms=%.3f total_ms=%.3f",
                len(pages),
                reason,
                resolve_ms,
                transfer_ms,
                total_ms,
            )
            return {
                "ok": bool(pages) or reason in {"ok", "already_local"},
                "reason": reason,
                "block_size_tokens": self.tree_cache.page_size,
                "resolve_ms": resolve_ms,
                "transfer_ms": transfer_ms,
                "total_ms": total_ms,
                "transfer_bytes": transfer_bytes,
                "transferred_blocks": len(pages),
            }
        finally:
            for node in protected_nodes:
                try:
                    node.release_host()
                except RuntimeError:
                    logger.exception(
                        "Failed to release remote G2 source host page protection"
                    )

    def _max_cacheable_blocks(self, req: "Req") -> int:
        max_prefix_len = max(len(req.fill_ids) - 1, 0)
        if req.return_logprob and req.logprob_start_len >= 0:
            max_prefix_len = min(max_prefix_len, req.logprob_start_len)
        if req.positional_embed_overrides is not None:
            max_prefix_len = 0
        return max_prefix_len // self.tree_cache.page_size

    def _validate_plan(self, plan: RemoteKvReusePlan) -> Optional[str]:
        if self.worker_id is None:
            return "missing_worker_id"
        if plan.target_worker_id != self.worker_id:
            return "wrong_target_worker"
        if plan.target_dp_rank != self.dp_rank:
            return "wrong_target_dp_rank"
        if (
            plan.source_worker_id == plan.target_worker_id
            and plan.source_dp_rank == plan.target_dp_rank
        ):
            return "source_is_target"
        if plan.plan_version != REMOTE_KV_REUSE_PLAN_VERSION:
            return "unsupported_plan_version"
        if plan.is_expired():
            return "plan_expired"
        if not plan.is_remote_g2():
            return "unsupported_source_tier"
        if plan.block_size_tokens != self.tree_cache.page_size:
            return "incompatible_block_size"
        return None

    def _plan_key(self, req: "Req", plan: RemoteKvReusePlan) -> tuple[str, str]:
        return str(req.rid), plan.plan_id

    def _alloc_device_indices(self, token_count: int) -> Optional[torch.Tensor]:
        allocator = self.tree_cache.cache_controller.mem_pool_device_allocator
        device_indices = allocator.alloc(token_count)
        if device_indices is None:
            self.tree_cache.evict(EvictParams(num_tokens=token_count))
            device_indices = allocator.alloc(token_count)
        if device_indices is None:
            logger.warning("Remote G2 failed to allocate %d device tokens", token_count)
        return device_indices

    def _free_device_indices(self, device_indices: Optional[torch.Tensor]) -> None:
        if device_indices is None:
            return
        self.tree_cache.cache_controller.mem_pool_device_allocator.free(device_indices)

    def _observe_quarantine(
        self, *, backend: str, reason: str, tokens: int, current_tokens: int
    ) -> None:
        metrics_collector = getattr(self, "metrics_collector", None)
        observe = getattr(
            metrics_collector, "observe_router_kv_reuse_quarantine", None
        )
        if observe is None:
            return
        try:
            observe(
                backend=_normalize_metric_label(backend),
                reason=_normalize_metric_label(reason),
                tokens=max(0, int(tokens)),
                current_tokens=max(0, int(current_tokens)),
            )
        except Exception:
            logger.debug(
                "Failed to record router KV reuse quarantine metrics", exc_info=True
            )

    def _quarantine_device_indices(
        self, device_indices: torch.Tensor, reason: str, *, backend: str
    ) -> None:
        quarantined = getattr(self, "_quarantined_device_indices", None)
        if quarantined is None:
            quarantined = self._quarantined_device_indices = []
        quarantined.append(device_indices)
        backend_label = _normalize_metric_label(backend)
        token_count = int(device_indices.numel())
        tokens_by_backend = getattr(self, "_quarantined_tokens_by_backend", None)
        if tokens_by_backend is None:
            tokens_by_backend = self._quarantined_tokens_by_backend = {}
        current_tokens = int(tokens_by_backend.get(backend_label, 0)) + token_count
        tokens_by_backend[backend_label] = current_tokens
        self._observe_quarantine(
            backend=backend_label,
            reason=reason,
            tokens=token_count,
            current_tokens=current_tokens,
        )
        logger.error(
            "Quarantining %d G2plus target KV indices after indeterminate direct transfer: %s",
            token_count,
            reason,
        )

    def _release_quarantined_device_indices(self) -> None:
        quarantined = getattr(self, "_quarantined_device_indices", None)
        if not quarantined:
            return
        self._quarantined_device_indices = []
        tokens_by_backend = getattr(self, "_quarantined_tokens_by_backend", {})
        self._quarantined_tokens_by_backend = {}
        for device_indices in quarantined:
            self._free_device_indices(device_indices)
        for backend in tokens_by_backend:
            self._observe_quarantine(
                backend=backend,
                reason="released",
                tokens=0,
                current_tokens=0,
            )

    def _device_indices_to_page_indices(
        self, device_indices: torch.Tensor
    ) -> Optional[list[int]]:
        page_size = self.tree_cache.page_size
        indices = device_indices.detach().cpu().numpy()
        if len(indices) == 0 or len(indices) % page_size != 0:
            return None

        page_rows = indices.reshape(-1, page_size)
        starts = page_rows[:, 0]
        offsets = np.arange(page_size, dtype=page_rows.dtype)
        if np.any(starts % page_size != 0) or np.any(
            page_rows != starts[:, None] + offsets[None, :]
        ):
            return None

        page_indices = starts // page_size
        if np.any(page_indices < 0) or np.any(page_indices > np.iinfo(np.int32).max):
            return None
        return page_indices.astype(np.int32, copy=False).tolist()

    def _active_source_resolver_count(self) -> int:
        lock = getattr(self, "_source_activity_lock", None)
        if lock is None:
            return int(getattr(self, "_active_source_resolver_ops", 0))
        with lock:
            return int(getattr(self, "_active_source_resolver_ops", 0))

    def has_pending(self) -> bool:
        detached_fetches = getattr(self, "_detached_fetches", set())
        for future in list(detached_fetches):
            if future.done():
                detached_fetches.discard(future)
        return (
            bool(getattr(self, "_pending_fetches", {}))
            or bool(detached_fetches)
            or self._active_source_resolver_count() > 0
        )

    def _lock_request_prefix(self, req: "Req") -> Optional[TreeNode]:
        last_node = getattr(req, "last_node", None)
        if last_node is None or last_node is self.tree_cache.root_node:
            return None
        self.tree_cache.inc_lock_ref(last_node)
        return last_node

    def _unlock_pending_prefix(self, pending: _RemoteG2PendingFetch) -> None:
        locked_node = getattr(pending, "locked_node", None)
        if locked_node is None:
            return
        pending.locked_node = None
        self.tree_cache.dec_lock_ref(locked_node)

    def _release_pending_fetch(self, pending: _RemoteG2PendingFetch) -> None:
        cancelled = pending.future.cancel()
        self._unlock_pending_prefix(pending)
        if pending.device_indices is None:
            return
        backend = getattr(pending, "backend", self._current_backend_label())
        if cancelled:
            self._free_device_indices(pending.device_indices)
        elif pending.future.done():
            self._release_device_indices_after_fetch_done(
                pending.future, pending.device_indices, backend=backend
            )
        else:
            detached_fetches = getattr(self, "_detached_fetches", None)
            if detached_fetches is None:
                detached_fetches = self._detached_fetches = set()
            detached_fetches.add(pending.future)

            def _free_detached_fetch(_future, device_indices=pending.device_indices):
                try:
                    self._release_device_indices_after_fetch_done(
                        _future, device_indices, backend=backend
                    )
                finally:
                    detached_fetches.discard(_future)

            pending.future.add_done_callback(_free_detached_fetch)

    def _release_device_indices_after_fetch_done(
        self, future: Future, device_indices: torch.Tensor, *, backend: str
    ) -> None:
        try:
            pages, reason = future.result()
        except Exception:
            self._free_device_indices(device_indices)
            return
        if not pages and _is_indeterminate_direct_transfer_reason(reason):
            self._quarantine_device_indices(device_indices, reason, backend=backend)
            return
        self._free_device_indices(device_indices)

    def _submit_fetch(
        self, plan: RemoteKvReusePlan, *, start_block: int, max_blocks: int
    ) -> Optional[Future]:
        return None

    def _submit_direct_transfer(
        self,
        plan: RemoteKvReusePlan,
        *,
        start_block: int,
        max_blocks: int,
        token_count: int,
    ) -> tuple[Optional[Future], Optional[torch.Tensor]]:
        direct_transfer = getattr(self, "direct_transfer", None)
        if not self._direct_transfer_enabled():
            return None, None
        endpoints = self._candidate_endpoints_for_plan(plan)
        if not endpoints:
            return None, None
        if not self._try_acquire_fetch_worker():
            return None, None

        device_indices = self._alloc_device_indices(token_count)
        if device_indices is None:
            self._release_fetch_worker()
            return None, None

        target_page_indices = self._device_indices_to_page_indices(device_indices)
        if target_page_indices is None:
            logger.warning(
                "Remote G2 direct transfer got non page-aligned target device allocation"
            )
            self._free_device_indices(device_indices)
            self._release_fetch_worker()
            return None, None
        try:
            future = self._fetch_executor.submit(
                self._request_source_transfer,
                transfer_backend=direct_transfer,
                endpoints=endpoints,
                plan=plan,
                start_block=start_block,
                max_blocks=max_blocks,
                target_page_indices=target_page_indices,
            )
        except Exception:
            self._free_device_indices(device_indices)
            self._release_fetch_worker()
            raise
        future.add_done_callback(self._on_fetch_worker_done)
        return future, device_indices

    def maybe_stage_remote_prefix(self, req: "Req") -> int:
        return self.check_remote_prefix(req).staged_tokens

    def has_reuse_plan(self, req: "Req") -> bool:
        plan_data = getattr(req, "remote_kv_reuse_plan", None)
        if plan_data is None:
            return False
        try:
            plan = RemoteKvReusePlan.from_dict(plan_data)
        except ValueError:
            return False
        if self._validate_plan(plan) is not None:
            return False
        return self._direct_transfer_enabled()

    def prefetch_remote_prefix(self, req: "Req") -> RouterKVReuseResult:
        return RouterKVReuseResult()

    def release_request(self, rid: str) -> None:
        rid = str(rid)
        pending = self._pending_fetches.pop(rid, None)

        if pending is not None:
            self._release_pending_fetch(pending)

        self._finished_plan_keys = {
            key for key in self._finished_plan_keys if key[0] != rid
        }

    def check_remote_prefix(self, req: "Req") -> RouterKVReuseResult:
        plan_data = getattr(req, "remote_kv_reuse_plan", None)
        if plan_data is None:
            return RouterKVReuseResult()

        try:
            plan = RemoteKvReusePlan.from_dict(plan_data)
        except ValueError as err:
            logger.debug("Ignoring invalid remote G2 plan for rid=%s: %s", req.rid, err)
            self._observe_reuse(
                backend=self._current_backend_label(),
                outcome="skip",
                reason="invalid_plan",
            )
            return RouterKVReuseResult()

        rejection = self._validate_plan(plan)
        if rejection is not None:
            logger.debug(
                "Ignoring remote G2 plan rid=%s plan_id=%s reason=%s",
                req.rid,
                plan.plan_id,
                rejection,
            )
            self._observe_reuse(
                backend=self._current_backend_label(),
                outcome="skip",
                reason=rejection,
            )
            return RouterKVReuseResult()

        plan_key = self._plan_key(req, plan)
        if plan_key in self._finished_plan_keys:
            return RouterKVReuseResult()

        page_size = self.tree_cache.page_size
        matched_tokens = len(req.prefix_indices) + req.host_hit_length
        if matched_tokens % page_size != 0:
            logger.debug(
                "Skipping remote G2 plan rid=%s plan_id=%s reason=unaligned_matched_tokens matched_tokens=%d page_size=%d",
                req.rid,
                plan.plan_id,
                matched_tokens,
                page_size,
            )
            self._observe_reuse(
                backend=self._current_backend_label(),
                outcome="skip",
                reason="unaligned_matched_tokens",
            )
            return RouterKVReuseResult()
        computed_blocks = matched_tokens // page_size
        if computed_blocks < plan.start_block_index:
            logger.debug(
                "Skipping remote G2 plan rid=%s plan_id=%s reason=before_plan_start computed_blocks=%d start_block_index=%d",
                req.rid,
                plan.plan_id,
                computed_blocks,
                plan.start_block_index,
            )
            self._observe_reuse(
                backend=self._current_backend_label(),
                outcome="skip",
                reason="before_plan_start",
            )
            return RouterKVReuseResult()

        max_plan_blocks = max(
            self._max_cacheable_blocks(req) - plan.start_block_index, 0
        )
        planned_blocks = min(plan.planned_prefix_blocks, max_plan_blocks)
        plan_offset = computed_blocks - plan.start_block_index
        if planned_blocks <= plan_offset:
            logger.debug(
                "Skipping remote G2 plan rid=%s plan_id=%s reason=no_remaining_planned_blocks planned_blocks=%d plan_offset=%d max_plan_blocks=%d",
                req.rid,
                plan.plan_id,
                planned_blocks,
                plan_offset,
                max_plan_blocks,
            )
            self._observe_reuse(
                backend=self._current_backend_label(),
                outcome="skip",
                reason="no_remaining_planned_blocks",
            )
            return RouterKVReuseResult()

        pending = self._pending_fetches.get(str(req.rid))
        if pending is not None:
            if pending.plan.plan_id != plan.plan_id:
                self._pending_fetches.pop(str(req.rid), None)
                self._release_pending_fetch(pending)
            elif not pending.future.done():
                return RouterKVReuseResult(pending=True)
            else:
                return self._finish_pending_fetch(req, pending)

        logger.debug(
            "Submitting remote G2 fetch rid=%s plan_id=%s source=%s:%s start_block=%d max_blocks=%d matched_tokens=%d",
            req.rid,
            plan.plan_id,
            plan.source_worker_id,
            plan.source_dp_rank,
            plan_offset,
            planned_blocks - plan_offset,
            matched_tokens,
        )
        expected_hashes = plan.planned_hashes[plan_offset:planned_blocks]
        max_blocks = planned_blocks - plan_offset
        token_count = max_blocks * page_size
        future = None
        device_indices = None
        direct_transfer_enabled = self._direct_transfer_enabled()
        if not direct_transfer_enabled:
            logger.debug(
                "Skipping remote G2 plan rid=%s plan_id=%s reason=direct_transfer_unavailable",
                req.rid,
                plan.plan_id,
            )
            self._finished_plan_keys.add(plan_key)
            self._observe_reuse(
                backend="none",
                outcome="miss",
                reason="direct_transfer_unavailable",
            )
            return RouterKVReuseResult()
        if direct_transfer_enabled and req.host_hit_length > 0:
            self._finished_plan_keys.add(plan_key)
            self._observe_reuse(
                backend=self._current_backend_label(),
                outcome="skip",
                reason="local_host_hit",
            )
            return RouterKVReuseResult()
        locked_node = None
        if req.host_hit_length == 0:
            if direct_transfer_enabled:
                locked_node = self._lock_request_prefix(req)
            try:
                future, device_indices = self._submit_direct_transfer(
                    plan,
                    start_block=plan_offset,
                    max_blocks=max_blocks,
                    token_count=token_count,
                )
            except Exception:
                if locked_node is not None:
                    self.tree_cache.dec_lock_ref(locked_node)
                raise
            if direct_transfer_enabled and future is None:
                if locked_node is not None:
                    self.tree_cache.dec_lock_ref(locked_node)
                self._finished_plan_keys.add(plan_key)
                self._observe_reuse(
                    backend=self._current_backend_label(),
                    outcome="miss",
                    reason="direct_submit_unavailable",
                )
                return RouterKVReuseResult()
        backend = "none"
        bytes_per_page = 0
        if device_indices is not None and direct_transfer_enabled:
            direct_transfer = getattr(self, "direct_transfer", None)
            backend = str(getattr(direct_transfer, "name", "direct"))
            try:
                bytes_per_page = sum(
                    int(length)
                    for length in getattr(direct_transfer, "target_kv_item_lens", [])
                )
            except Exception:
                bytes_per_page = 0
        self._pending_fetches[str(req.rid)] = _RemoteG2PendingFetch(
            plan=plan,
            plan_offset=plan_offset,
            target_start_block=plan.start_block_index + plan_offset,
            expected_hashes=expected_hashes,
            future=future,
            device_indices=device_indices,
            locked_node=locked_node,
            backend=backend,
            bytes_per_page=bytes_per_page,
            submitted_at=time.perf_counter(),
        )
        return RouterKVReuseResult(pending=True)

    def _finish_pending_fetch(
        self, req: "Req", pending: _RemoteG2PendingFetch
    ) -> RouterKVReuseResult:
        self._pending_fetches.pop(str(req.rid), None)
        plan = pending.plan

        try:
            pages, reason = pending.future.result()
        except Exception:
            logger.exception(
                "Remote G2 fetch failed rid=%s plan_id=%s", req.rid, plan.plan_id
            )
            self._unlock_pending_prefix(pending)
            if pending.device_indices is not None:
                self._free_device_indices(pending.device_indices)
            self._finished_plan_keys.add(self._plan_key(req, plan))
            self._observe_reuse(
                backend=pending.backend,
                outcome="error",
                reason="fetch_exception",
                wait_ms=self._pending_wait_ms(pending),
            )
            return RouterKVReuseResult()

        if not pages:
            logger.debug(
                "Remote G2 source returned no pages rid=%s plan_id=%s reason=%s",
                req.rid,
                plan.plan_id,
                reason,
            )
            self._unlock_pending_prefix(pending)
            indeterminate_transfer = _is_indeterminate_direct_transfer_reason(reason)
            if pending.device_indices is not None:
                if indeterminate_transfer:
                    self._quarantine_device_indices(
                        pending.device_indices,
                        reason,
                        backend=getattr(
                            pending, "backend", self._current_backend_label()
                        ),
                    )
                else:
                    self._free_device_indices(pending.device_indices)
            self._finished_plan_keys.add(self._plan_key(req, plan))
            self._observe_reuse(
                backend=pending.backend,
                outcome="error" if indeterminate_transfer else "miss",
                reason=reason,
                wait_ms=self._pending_wait_ms(pending),
            )
            return RouterKVReuseResult()

        if len(pages) > len(pending.expected_hashes):
            logger.warning(
                "Remote G2 source returned too many pages rid=%s plan_id=%s pages=%d expected=%d",
                req.rid,
                plan.plan_id,
                len(pages),
                len(pending.expected_hashes),
            )
            self._unlock_pending_prefix(pending)
            if pending.device_indices is not None:
                self._free_device_indices(pending.device_indices)
            self._finished_plan_keys.add(self._plan_key(req, plan))
            self._observe_reuse(
                backend=pending.backend,
                outcome="error",
                reason="too_many_pages",
                wait_ms=self._pending_wait_ms(pending),
                transfer_bytes=self._transfer_bytes_for_pages(pending, pages),
            )
            return RouterKVReuseResult()

        expected_hashes = pending.expected_hashes[: len(pages)]
        if tuple(page.block_hash for page in pages) != expected_hashes:
            logger.warning(
                "Remote G2 source returned non-contiguous pages rid=%s plan_id=%s",
                req.rid,
                plan.plan_id,
            )
            self._unlock_pending_prefix(pending)
            if pending.device_indices is not None:
                self._free_device_indices(pending.device_indices)
            self._finished_plan_keys.add(self._plan_key(req, plan))
            self._observe_reuse(
                backend=pending.backend,
                outcome="error",
                reason="non_contiguous_pages",
                wait_ms=self._pending_wait_ms(pending),
                transfer_bytes=self._transfer_bytes_for_pages(pending, pages),
            )
            return RouterKVReuseResult()

        insert_start = time.perf_counter()
        try:
            if pending.device_indices is not None:
                staged_tokens = self._insert_device_pages(
                    req,
                    pages,
                    device_indices=pending.device_indices,
                    start_block=pending.target_start_block,
                )
            else:
                staged_tokens = self._insert_pages(
                    req, pages, start_block=pending.target_start_block
                )
        except Exception:
            insert_ms = (time.perf_counter() - insert_start) * 1000
            logger.exception(
                "Remote G2 insert failed rid=%s plan_id=%s",
                req.rid,
                plan.plan_id,
            )
            self._finished_plan_keys.add(self._plan_key(req, plan))
            self._observe_reuse(
                backend=pending.backend,
                outcome="error",
                reason="insert_exception",
                wait_ms=self._pending_wait_ms(pending),
                insert_ms=insert_ms,
                transfer_bytes=self._transfer_bytes_for_pages(pending, pages),
            )
            return RouterKVReuseResult()
        finally:
            self._unlock_pending_prefix(pending)
        insert_ms = (time.perf_counter() - insert_start) * 1000
        if staged_tokens > 0:
            req.remote_g2_hit_length = (
                getattr(req, "remote_g2_hit_length", 0) + staged_tokens
            )
            wait_ms = self._pending_wait_ms(pending)
            ready_wait_ms = self._pending_ready_wait_ms(pending)
            logger.debug(
                "Remote G2 staged %d tokens rid=%s plan_id=%s source=%s:%s wait_ms=%s future_ready_wait_ms=%s insert_ms=%.3f direct=%s",
                staged_tokens,
                req.rid,
                plan.plan_id,
                plan.source_worker_id,
                plan.source_dp_rank,
                _format_optional_ms(wait_ms),
                _format_optional_ms(ready_wait_ms),
                insert_ms,
                pending.device_indices is not None,
            )
        self._finished_plan_keys.add(self._plan_key(req, plan))
        outcome = "hit" if staged_tokens > 0 else "miss"
        self._observe_reuse(
            backend=pending.backend,
            outcome=outcome,
            reason=reason if staged_tokens > 0 else "insert_returned_zero",
            tokens=staged_tokens,
            wait_ms=self._pending_wait_ms(pending),
            insert_ms=insert_ms,
            transfer_bytes=self._transfer_bytes_for_pages(pending, pages),
        )
        return RouterKVReuseResult(staged_tokens=staged_tokens)

    def _insert_pages(
        self, req: "Req", pages: list[ResolvedHostPage], *, start_block: int
    ) -> int:
        page_size = self.tree_cache.page_size
        token_count = len(pages) * page_size
        token_start = start_block * page_size
        token_end = token_start + token_count
        if token_end > len(req.fill_ids):
            token_count = ((len(req.fill_ids) - token_start) // page_size) * page_size
            pages = pages[: token_count // page_size]
            token_end = token_start + token_count
        if token_count <= 0:
            return 0

        host_pool = self.tree_cache.cache_controller.mem_pool_host
        host_indices = host_pool.alloc(token_count)
        if host_indices is None:
            self.tree_cache.evict_host(token_count)
            host_indices = host_pool.alloc(token_count)
        if host_indices is None:
            logger.warning("Remote G2 failed to allocate %d host tokens", token_count)
            return 0

        try:
            expected_numel = host_pool.get_dummy_flat_data_page().numel()
            for page_idx, page in enumerate(pages):
                page_tensor = _tensor_from_bytes(
                    page.data, host_pool.dtype, expected_numel
                )
                page_start = int(host_indices[page_idx * page_size].item())
                host_pool.set_from_flat_data_page(page_start, page_tensor)

            parent = req.last_host_node if req.host_hit_length > 0 else req.last_node
            if parent is None:
                parent = self.tree_cache.root_node

            key = RadixKey(
                req.fill_ids[token_start:token_end],
                extra_key=req.extra_key,
                is_bigram=self.tree_cache.is_eagle,
            )
            matched_length = self.tree_cache._insert_helper_host(
                parent,
                key,
                host_indices[:token_count],
                [page.hash_value for page in pages],
            )
            if matched_length > 0:
                host_pool.free(host_indices[:matched_length])
            staged_tokens = token_count - matched_length
            if staged_tokens <= 0:
                return 0
            return staged_tokens
        except Exception:
            host_pool.free(host_indices)
            raise

    def _insert_device_pages(
        self,
        req: "Req",
        pages: list[ResolvedHostPage],
        *,
        device_indices: torch.Tensor,
        start_block: int,
    ) -> int:
        page_size = self.tree_cache.page_size
        token_count = len(pages) * page_size
        token_start = start_block * page_size
        token_end = token_start + token_count
        allocated_tokens = len(device_indices)

        if token_end > len(req.fill_ids):
            token_count = ((len(req.fill_ids) - token_start) // page_size) * page_size
            pages = pages[: token_count // page_size]
            token_end = token_start + token_count

        if token_count <= 0:
            self._free_device_indices(device_indices)
            return 0

        if token_count < allocated_tokens:
            self._free_device_indices(device_indices[token_count:])
            device_indices = device_indices[:token_count]

        try:
            prefix_indices = getattr(
                req, "prefix_indices", torch.empty((0,), dtype=torch.int64)
            )
            if token_start != len(prefix_indices):
                logger.debug(
                    "Remote G2 direct insert cannot attach suffix rid=%s token_start=%d prefix_indices=%d",
                    getattr(req, "rid", None),
                    token_start,
                    len(prefix_indices),
                )
                self._free_device_indices(device_indices)
                return 0

            key = RadixKey(
                req.fill_ids[:token_end],
                extra_key=req.extra_key,
                is_bigram=self.tree_cache.is_eagle,
            )
            if token_start > 0:
                prefix_indices = prefix_indices.to(
                    dtype=torch.int64, device=device_indices.device, copy=False
                )
                insert_value = torch.cat([prefix_indices, device_indices])
            else:
                insert_value = device_indices

            result = self.tree_cache.insert(InsertParams(key=key, value=insert_value))
            matched_length = result.prefix_len
            matched_new_tokens = min(max(0, matched_length - token_start), token_count)
            if matched_new_tokens > 0:
                self._free_device_indices(device_indices[:matched_new_tokens])
            staged_tokens = token_count - matched_new_tokens
            if staged_tokens <= 0:
                return 0
            return staged_tokens
        except Exception:
            self._free_device_indices(device_indices)
            raise


class RouterKVReuseManager:
    """Scheduler-facing router reuse control plane.

    The external wire key stays `remote_kv_reuse_plan`. Internally, the
    scheduler only calls this generic manager so future router-directed reuse
    plans can install a different handler without changing request plumbing.
    """

    def __init__(self, handlers: Iterable[RouterKVReuseHandler]):
        self.handlers = list(handlers)

    @classmethod
    def from_scheduler(cls, scheduler) -> Optional["RouterKVReuseManager"]:
        if not _router_kv_reuse_enabled(scheduler.server_args):
            return None

        handlers: list[RouterKVReuseHandler] = []
        remote_g2_handler = RemoteG2ReuseHandler.from_scheduler(scheduler)
        if remote_g2_handler is not None:
            handlers.append(remote_g2_handler)

        if not handlers:
            return None
        return cls(handlers)

    def maybe_stage_reuse_plan(self, req: "Req") -> int:
        return self.check_reuse_plan_progress(req).staged_tokens

    def has_reuse_plan(self, req: "Req") -> bool:
        for handler in self.handlers:
            has_reuse_plan = getattr(handler, "has_reuse_plan", None)
            if has_reuse_plan is not None and has_reuse_plan(req):
                return True
        return False

    def has_pending(self) -> bool:
        for handler in self.handlers:
            has_pending = getattr(handler, "has_pending", None)
            if has_pending is None:
                continue
            try:
                if has_pending():
                    return True
            except Exception:
                logger.exception("Router KV reuse handler pending check failed")
                return True
        return False

    def prefetch_reuse_plan(self, req: "Req") -> RouterKVReuseResult:
        for handler in self.handlers:
            prefetch = getattr(handler, "prefetch_remote_prefix", None)
            if prefetch is not None:
                result = prefetch(req)
            else:
                check = getattr(handler, "check_remote_prefix", None)
                if check is None:
                    continue
                result = check(req)
            if result.pending or result.staged_tokens > 0:
                return result
        return RouterKVReuseResult()

    def check_reuse_plan_progress(self, req: "Req") -> RouterKVReuseResult:
        for handler in self.handlers:
            if hasattr(handler, "check_remote_prefix"):
                result = handler.check_remote_prefix(req)
            else:
                result = RouterKVReuseResult(
                    staged_tokens=handler.maybe_stage_remote_prefix(req)
                )
            if result.pending or result.staged_tokens > 0:
                return result
        return RouterKVReuseResult()

    def maybe_stage_remote_prefix(self, req: "Req") -> int:
        return self.maybe_stage_reuse_plan(req)

    def shutdown(self) -> None:
        for handler in self.handlers:
            shutdown = getattr(handler, "shutdown", None)
            if shutdown is not None:
                try:
                    shutdown()
                except Exception:
                    logger.exception("Router KV reuse handler shutdown failed")

    def release_request(self, rid: str) -> None:
        for handler in self.handlers:
            release = getattr(handler, "release_request", None)
            if release is not None:
                try:
                    release(rid)
                except Exception:
                    logger.exception(
                        "Router KV reuse handler failed to release rid=%s", rid
                    )


G2plusManager = RemoteG2ReuseHandler
