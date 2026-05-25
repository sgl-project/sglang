from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional

import numpy as np
import torch

from sglang.srt.mem_cache.shared_hicache.transfer import (
    SharedHiCacheTransferBackend,
    shared_hicache_parallel_rejection,
)
from sglang.srt.mem_cache.radix_cache import TreeNode
from sglang.srt.mem_cache.shared_hicache.plan import (
    SHARED_HICACHE_DIRECT_TIMEOUT_REASON,
    SharedHiCachePlan,
)
from sglang.srt.mem_cache.utils import block_hash_aliases

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


def _lookup_hicache_host_blocks(
    tree_cache, wanted_hashes: set[int]
) -> tuple[dict[int, tuple[TreeNode, int, str]], list[TreeNode], Optional[str]]:
    lookup_index = getattr(tree_cache, "lookup_hicache_host_blocks", None)
    if not callable(lookup_index):
        return {}, [], "hicache_host_lookup_unavailable"
    lookup_result = lookup_index(wanted_hashes, protect=True)
    if not isinstance(lookup_result, tuple) or len(lookup_result) != 2:
        return {}, [], "malformed_hicache_host_lookup"
    index, protected_nodes = lookup_result
    if not isinstance(index, dict):
        return {}, list(protected_nodes or []), "malformed_hicache_host_lookup"
    return index, list(protected_nodes or []), None


def _host_block_entry(
    block_index: Mapping[int, tuple[TreeNode, int, str]], block_hash: int
) -> Optional[tuple[TreeNode, int, str]]:
    for alias in block_hash_aliases(block_hash):
        entry = block_index.get(alias)
        if entry is not None:
            return entry
    return None


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
    plan: SharedHiCachePlan,
    *,
    start_block: int,
    max_blocks: int,
    worker_id: Optional[int],
    attn_dp_rank: int,
) -> tuple[list[ResolvedHostPage], str]:
    pages, reason, protected_nodes = resolve_host_page_locations(
        tree_cache,
        plan,
        start_block=start_block,
        max_blocks=max_blocks,
        worker_id=worker_id,
        attn_dp_rank=attn_dp_rank,
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
    plan: SharedHiCachePlan,
    *,
    start_block: int,
    max_blocks: int,
    worker_id: Optional[int],
    attn_dp_rank: int,
    attn_tp_rank: int = 0,
    attn_tp_size: int = 1,
    pp_size: int = 1,
    attn_cp_size: int = 1,
    target_attn_dp_rank: Optional[int] = None,
    target_attn_tp_rank: Optional[int] = None,
    target_attn_tp_size: Optional[int] = None,
) -> tuple[list[ResolvedHostPageLocation], str, list[TreeNode]]:
    if worker_id is None:
        return [], "missing_source_worker_id", []
    if plan.source_worker_id != worker_id:
        return [], "wrong_source_worker", []
    if plan.source_attn_dp_rank != attn_dp_rank:
        return [], "wrong_source_attn_dp_rank", []
    rank_rejection = _source_rank_rejection(
        plan,
        attn_tp_rank=attn_tp_rank,
        attn_tp_size=attn_tp_size,
        pp_size=pp_size,
        attn_cp_size=attn_cp_size,
        target_attn_dp_rank=target_attn_dp_rank,
        target_attn_tp_rank=target_attn_tp_rank,
        target_attn_tp_size=target_attn_tp_size,
    )
    if rank_rejection is not None:
        return [], rank_rejection, []
    if plan.is_expired():
        return [], "plan_expired", []
    if not plan.is_shared_hicache():
        return [], "unsupported_source_medium", []
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
    block_index, protected_nodes, lookup_error = _lookup_hicache_host_blocks(
        tree_cache, set(requested_kv_hashes)
    )
    if lookup_error is not None:
        return [], lookup_error, protected_nodes

    protected_ids = {node.id for node in protected_nodes}
    reason = "ok"
    for identity_hash, kv_hash in zip(requested_identity_hashes, requested_kv_hashes):
        entry = _host_block_entry(block_index, kv_hash)
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
            logger.exception("Failed to release shared HiCache source host page protection")


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


def _target_metadata_int(
    metadata: Any,
    field_name: str,
) -> Optional[int]:
    if not isinstance(metadata, Mapping) or field_name not in metadata:
        return None
    return _coerce_transfer_int(metadata[field_name], f"target_metadata.{field_name}")


def _is_timeout_error(err: BaseException) -> bool:
    if isinstance(err, TimeoutError):
        return True
    return "timed out" in str(err).lower()


def _source_rank_rejection(
    plan: SharedHiCachePlan,
    *,
    attn_tp_rank: int,
    attn_tp_size: int,
    pp_size: int,
    attn_cp_size: int,
    target_attn_dp_rank: Optional[int],
    target_attn_tp_rank: Optional[int],
    target_attn_tp_size: Optional[int],
) -> Optional[str]:
    topology_rejection = shared_hicache_parallel_rejection(
        pp_size=int(pp_size),
        attn_cp_size=int(attn_cp_size),
    )
    if topology_rejection is not None:
        return f"unsupported_source_topology:{topology_rejection}"

    local_attn_tp_size = int(attn_tp_size)
    local_attn_tp_rank = int(attn_tp_rank)
    if plan.source_attn_tp_size != local_attn_tp_size:
        return (
            "wrong_source_attn_tp_size:"
            f"plan={plan.source_attn_tp_size}:local={local_attn_tp_size}"
        )
    if (
        target_attn_dp_rank is not None
        and plan.target_attn_dp_rank != int(target_attn_dp_rank)
    ):
        return (
            "wrong_target_attn_dp_rank:"
            f"plan={plan.target_attn_dp_rank}:target={int(target_attn_dp_rank)}"
        )

    if target_attn_tp_size is None:
        if local_attn_tp_size > 1:
            return "missing_target_attn_tp_size"
        target_attn_tp_size = 1
    if target_attn_tp_rank is None:
        if local_attn_tp_size > 1:
            return "missing_target_attn_tp_rank"
        target_attn_tp_rank = 0

    target_attn_tp_size = int(target_attn_tp_size)
    target_attn_tp_rank = int(target_attn_tp_rank)
    if plan.target_attn_tp_size != target_attn_tp_size:
        return (
            "wrong_target_attn_tp_size:"
            f"plan={plan.target_attn_tp_size}:target={target_attn_tp_size}"
        )
    if target_attn_tp_size != local_attn_tp_size:
        return (
            "incompatible_attn_tp_size:"
            f"source={local_attn_tp_size}:target={target_attn_tp_size}"
        )
    if target_attn_tp_rank != local_attn_tp_rank:
        return (
            "wrong_source_attn_tp_rank_for_target:"
            f"source={local_attn_tp_rank}:target={target_attn_tp_rank}"
        )
    return None


def _parse_target_kv_metadata(
    payload: Mapping[str, Any], transfer_backend: SharedHiCacheTransferBackend
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
    transfer_backend: Optional[SharedHiCacheTransferBackend],
    tree_cache,
    worker_id: Optional[int],
    attn_dp_rank: int,
    attn_tp_rank: int = 0,
    attn_tp_size: int = 1,
    pp_size: int = 1,
    attn_cp_size: int = 1,
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
        plan = SharedHiCachePlan.from_dict(payload["plan"])
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
    target_metadata = payload.get("target_metadata")
    try:
        target_attn_dp_rank = _target_metadata_int(
            target_metadata, "attn_dp_rank"
        )
        target_attn_tp_rank = _target_metadata_int(
            target_metadata, "attn_tp_rank"
        )
        target_attn_tp_size = _target_metadata_int(
            target_metadata, "attn_tp_size"
        )
    except ValueError as err:
        return {
            "ok": False,
            "reason": f"malformed_transfer_request:{err}",
            "block_size_tokens": tree_cache.page_size,
        }
    if target_attn_dp_rank is None:
        return {
            "ok": False,
            "reason": "malformed_transfer_request:target_metadata.attn_dp_rank_missing",
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
        attn_dp_rank=attn_dp_rank,
        attn_tp_rank=attn_tp_rank,
        attn_tp_size=attn_tp_size,
        pp_size=pp_size,
        attn_cp_size=attn_cp_size,
        target_attn_dp_rank=target_attn_dp_rank,
        target_attn_tp_rank=target_attn_tp_rank,
        target_attn_tp_size=target_attn_tp_size,
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
                        f"{SHARED_HICACHE_DIRECT_TIMEOUT_REASON}:source:{err}"
                    )
                else:
                    failure_reason = f"direct_transfer_failed:{err}"
                logger.warning(
                    "SharedHiCache source direct transfer failed pages=%d resolve_ms=%.3f transfer_ms=%.3f reason=%s",
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
            "SharedHiCache source transfer handled pages=%d reason=%s resolve_ms=%.3f transfer_ms=%.3f total_ms=%.3f",
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
