from __future__ import annotations

import logging
import time
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Optional

import numpy as np
import torch

from sglang.srt.mem_cache.remote_g2.transfer import RemoteG2TransferBackend
from sglang.srt.mem_cache.radix_cache import TreeNode
from sglang.srt.mem_cache.remote_g2.plan import (
    REMOTE_G2_DIRECT_TIMEOUT_REASON,
    RemoteG2Plan,
    expand_block_hash_aliases,
)
from sglang.srt.mem_cache.utils import (
    block_hash_aliases,
    compute_node_hash_values,
    hash_str_to_int64,
)

logger = logging.getLogger(__name__)


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


def _tensor_to_bytes(tensor: torch.Tensor) -> bytes:
    tensor = tensor.detach().cpu().contiguous()
    return tensor.view(torch.uint8).numpy().tobytes()


def _iter_tree_nodes(root: TreeNode) -> Iterable[TreeNode]:
    stack = list(root.children.values())
    while stack:
        node = stack.pop()
        yield node
        stack.extend(node.children.values())


def _build_host_block_index(
    tree_cache, wanted_hashes: set[int]
) -> Dict[int, tuple[TreeNode, int, str]]:
    lookup_index = getattr(tree_cache, "lookup_remote_g2_host_blocks", None)
    if lookup_index is not None:
        index = lookup_index(wanted_hashes)
        if len(index) >= len(wanted_hashes):
            return index
        if getattr(tree_cache, "remote_g2_host_block_index", None) is not None:
            return index
        wanted_hashes = wanted_hashes - set(index.keys())
    else:
        index: Dict[int, tuple[TreeNode, int, str]] = {}

    wanted_hashes = expand_block_hash_aliases(wanted_hashes)
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
    return getattr(tree_cache, "remote_g2_host_index_lock", nullcontext())


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
    plan: RemoteG2Plan,
    *,
    start_block: int,
    max_blocks: int,
    worker_id: Optional[int],
    dp_rank: int,
) -> tuple[list[ResolvedHostPage], str]:
    pages, reason, protected_nodes = resolve_host_page_locations(
        tree_cache,
        plan,
        start_block=start_block,
        max_blocks=max_blocks,
        worker_id=worker_id,
        dp_rank=dp_rank,
    )
    try:
        resolved = [
            ResolvedHostPage(
                block_hash=page.block_hash,
                hash_value=page.hash_value,
                data=_tensor_to_bytes(
                    tree_cache.cache_controller.mem_pool_host.get_data_page(
                        page.host_index, flat=True
                    )
                ),
            )
            for page in pages
        ]
        return resolved, reason
    finally:
        release_protected_host_nodes(protected_nodes)


def resolve_host_page_locations(
    tree_cache,
    plan: RemoteG2Plan,
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


def release_protected_host_nodes(nodes: Iterable[TreeNode]) -> None:
    for node in nodes:
        try:
            node.release_host()
        except RuntimeError:
            logger.exception("Failed to release remote G2 source host page protection")


def _coerce_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be an integer, got {value!r}")
    if isinstance(value, (int, np.integer)):
        return int(value)
    raise ValueError(f"{field_name} must be an integer, got {value!r}")


def _coerce_transfer_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field_name}_contains_non_integer:{value!r}")
    if isinstance(value, (int, np.integer)):
        return int(value)
    raise ValueError(f"{field_name}_contains_non_integer:{value!r}")


def _coerce_transfer_int_list(raw: Any, field_name: str) -> list[int]:
    if isinstance(raw, (str, bytes, Mapping)):
        raise ValueError(f"{field_name}_must_be_array")
    try:
        values = list(raw)
    except TypeError as err:
        raise ValueError(f"{field_name}_must_be_array") from err
    return [
        _coerce_transfer_int(value, f"{field_name}[{idx}]")
        for idx, value in enumerate(values)
    ]


def _is_timeout_error(err: BaseException) -> bool:
    if isinstance(err, TimeoutError):
        return True
    return "timed out" in str(err).lower()


def _parse_target_kv_metadata(
    payload: Mapping[str, Any], transfer_backend: RemoteG2TransferBackend
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
        target_kv_ptrs = _coerce_transfer_int_list(
            target_kv_ptrs_raw, "target_kv_ptrs"
        )
        target_kv_item_lens = _coerce_transfer_int_list(
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


def handle_source_transfer(
    *,
    payload: Mapping[str, Any],
    transfer_backend: Optional[RemoteG2TransferBackend],
    tree_cache,
    worker_id: Optional[int],
    dp_rank: int,
) -> Mapping[str, Any]:
    if transfer_backend is None or not getattr(transfer_backend, "enabled", False):
        return {"ok": False, "reason": "direct_transfer_unavailable"}
    requested_backend = str(payload.get("transfer_backend", "mooncake")).lower()
    if requested_backend != transfer_backend.name:
        return {
            "ok": False,
            "reason": (
                f"unsupported_transfer_backend:{requested_backend}:"
                f"local={transfer_backend.name}"
            ),
        }

    try:
        plan = RemoteG2Plan.from_dict(payload["plan"])
        start_block = _coerce_int(payload.get("start_block", 0), "start_block")
        max_blocks = _coerce_int(
            payload.get("max_blocks", len(plan.block_hashes)), "max_blocks"
        )
    except (KeyError, ValueError) as err:
        return {
            "ok": False,
            "reason": f"malformed_transfer_request:plan:{err}",
        }

    (
        target_session_id,
        target_kv_ptrs,
        target_kv_item_lens,
        target_kv_metadata_error,
    ) = _parse_target_kv_metadata(payload, transfer_backend)
    if target_kv_metadata_error is not None:
        return {
            "ok": False,
            "reason": f"malformed_transfer_request:{target_kv_metadata_error}",
            "block_size_tokens": tree_cache.page_size,
        }
    try:
        target_page_indices_list = _coerce_transfer_int_list(
            payload["target_page_indices"], "target_page_indices"
        )
    except KeyError as err:
        return {
            "ok": False,
            "reason": f"malformed_transfer_request:target_page_indices_missing:{err}",
            "block_size_tokens": tree_cache.page_size,
        }
    except ValueError as err:
        return {
            "ok": False,
            "reason": f"malformed_transfer_request:{err}",
            "block_size_tokens": tree_cache.page_size,
        }
    max_int32 = np.iinfo(np.int32).max
    if any(idx < 0 or idx > max_int32 for idx in target_page_indices_list):
        return {
            "ok": False,
            "reason": "malformed_transfer_request:target_page_index_out_of_range",
            "block_size_tokens": tree_cache.page_size,
        }

    total_start = time.perf_counter()
    resolve_start = total_start
    pages, reason, protected_nodes = resolve_host_page_locations(
        tree_cache,
        plan,
        start_block=start_block,
        max_blocks=max_blocks,
        worker_id=worker_id,
        dp_rank=dp_rank,
    )
    resolve_ms = (time.perf_counter() - resolve_start) * 1000
    transfer_ms = 0.0
    transfer_bytes = 0
    try:
        if pages:
            page_size = tree_cache.page_size
            source_page_indices_list: list[int] = []
            max_int32 = np.iinfo(np.int32).max
            for page in pages:
                host_index = int(page.host_index)
                if host_index < 0:
                    return {
                        "ok": False,
                        "reason": "source_host_page_index_out_of_range",
                        "block_size_tokens": tree_cache.page_size,
                    }
                if host_index % page_size != 0:
                    return {
                        "ok": False,
                        "reason": "source_host_page_index_unaligned",
                        "block_size_tokens": tree_cache.page_size,
                    }
                page_index = host_index // page_size
                if page_index > max_int32:
                    return {
                        "ok": False,
                        "reason": "source_page_index_out_of_range",
                        "block_size_tokens": tree_cache.page_size,
                    }
                source_page_indices_list.append(page_index)
            source_page_indices = np.array(source_page_indices_list, dtype=np.int32)
            transfer_bytes = len(pages) * sum(int(x) for x in target_kv_item_lens)
            if len(target_page_indices_list) < len(pages):
                return {
                    "ok": False,
                    "reason": (
                        "malformed_transfer_request:"
                        f"target_page_indices_too_short:{len(target_page_indices_list)}<{len(pages)}"
                    ),
                    "block_size_tokens": tree_cache.page_size,
                }
            target_page_indices_list = target_page_indices_list[: len(pages)]
            target_page_indices = np.array(target_page_indices_list, dtype=np.int32)
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
                        f"{REMOTE_G2_DIRECT_TIMEOUT_REASON}:source:{err}"
                    )
                else:
                    failure_reason = f"direct_transfer_failed:{err}"
                logger.warning(
                    "RemoteG2 source direct transfer failed pages=%d resolve_ms=%.3f transfer_ms=%.3f reason=%s",
                    len(pages),
                    resolve_ms,
                    transfer_ms,
                    err,
                    exc_info=True,
                )
                return {
                    "ok": False,
                    "reason": failure_reason,
                    "block_size_tokens": tree_cache.page_size,
                    "resolve_ms": resolve_ms,
                    "transfer_ms": transfer_ms,
                    "total_ms": (time.perf_counter() - total_start) * 1000,
                    "transfer_bytes": transfer_bytes,
                }
            transfer_ms = (time.perf_counter() - transfer_start) * 1000
        total_ms = (time.perf_counter() - total_start) * 1000
        logger.debug(
            "RemoteG2 source transfer handled pages=%d reason=%s resolve_ms=%.3f transfer_ms=%.3f total_ms=%.3f",
            len(pages),
            reason,
            resolve_ms,
            transfer_ms,
            total_ms,
        )
        return {
            "ok": bool(pages) or reason in {"ok", "already_local"},
            "reason": reason,
            "block_size_tokens": tree_cache.page_size,
            "resolve_ms": resolve_ms,
            "transfer_ms": transfer_ms,
            "total_ms": total_ms,
            "transfer_bytes": transfer_bytes,
            "transferred_blocks": len(pages),
        }
    finally:
        release_protected_host_nodes(protected_nodes)
