from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Mapping, Optional, Protocol

import numpy as np
import torch

from sglang.srt.disaggregation.common.staging_buffer import StagingBuffer
from sglang.srt.disaggregation.common.utils import group_concurrent_contiguous
from sglang.srt.environ import default_shared_hicache_transfer_parallelism
from sglang.srt.mem_cache.shared_hicache.config import (
    shared_hicache_transfer_backend_name,
)

logger = logging.getLogger(__name__)


class SharedHiCacheTransferBackend(Protocol):
    name: str
    target_session_id: str
    target_kv_ptrs: list[int]
    target_kv_item_lens: list[int]

    @property
    def enabled(self) -> bool: ...

    def target_descriptor(self) -> dict[str, Any]: ...

    def transfer_pages(
        self,
        *,
        target_session_id: str,
        source_page_indices: np.ndarray,
        target_page_indices: np.ndarray,
        target_kv_ptrs: list[int],
        target_kv_item_lens: list[int],
        target_metadata: Optional[Mapping[str, Any]] = None,
    ) -> None: ...

    def shutdown(self) -> None: ...


def _target_kv_pool_from_scheduler(scheduler):
    target_pool = scheduler.token_to_kv_pool_allocator.get_kvcache()
    if hasattr(target_pool, "full_kv_pool") and hasattr(target_pool, "full_layer_nums"):
        raise RuntimeError(
            "SharedHiCache direct transfer does not support hybrid linear-attention KV pools"
        )
    return target_pool


def _server_arg(scheduler, name: str, default: int) -> int:
    server_args = getattr(scheduler, "server_args", None)
    value = getattr(server_args, name, default)
    return int(value if value is not None else default)


def _parallel_value(scheduler, name: str, default: int) -> int:
    ps = getattr(scheduler, "ps", None)
    if ps is not None and hasattr(ps, name):
        value = getattr(ps, name)
    else:
        value = getattr(scheduler, name, default)
    return int(value if value is not None else default)


def scheduler_parallel_metadata(scheduler) -> dict[str, int]:
    """Return the same parallel-rank vocabulary used by disagg and HiCache."""

    return {
        "attn_dp_rank": _parallel_value(scheduler, "attn_dp_rank", 0),
        "attn_dp_size": _parallel_value(
            scheduler, "attn_dp_size", _server_arg(scheduler, "dp_size", 1)
        ),
        "attn_tp_rank": _parallel_value(scheduler, "attn_tp_rank", 0),
        "attn_tp_size": _parallel_value(
            scheduler, "attn_tp_size", _server_arg(scheduler, "tp_size", 1)
        ),
        "tp_rank": _parallel_value(scheduler, "tp_rank", 0),
        "tp_size": _parallel_value(
            scheduler, "tp_size", _server_arg(scheduler, "tp_size", 1)
        ),
        "pp_rank": _parallel_value(scheduler, "pp_rank", 0),
        "pp_size": _parallel_value(
            scheduler, "pp_size", _server_arg(scheduler, "pp_size", 1)
        ),
        "attn_cp_rank": _parallel_value(scheduler, "attn_cp_rank", 0),
        "attn_cp_size": _parallel_value(
            scheduler, "attn_cp_size", _server_arg(scheduler, "attn_cp_size", 1)
        ),
        "moe_ep_rank": _parallel_value(scheduler, "moe_ep_rank", 0),
        "moe_ep_size": _parallel_value(
            scheduler, "moe_ep_size", _server_arg(scheduler, "ep_size", 1)
        ),
    }


def shared_hicache_parallel_rejection(
    *, pp_size: int, attn_cp_size: int
) -> Optional[str]:
    unsupported = []
    if pp_size != 1:
        unsupported.append(f"pp_size={pp_size}")
    if attn_cp_size != 1:
        unsupported.append(f"attn_cp_size={attn_cp_size}")
    if unsupported:
        return (
            "SharedHiCache direct transfer supports TP, DP-attn, and EP, but "
            f"PP/CP are deferred; got {', '.join(unsupported)}"
        )
    return None


def _direct_topology_rejection(scheduler) -> Optional[str]:
    return shared_hicache_parallel_rejection(
        pp_size=_server_arg(scheduler, "pp_size", 1),
        attn_cp_size=_server_arg(scheduler, "attn_cp_size", 1),
    )


def _get_or_init_mooncake_transfer_engine(scheduler):
    from sglang.srt.distributed.device_communicators.mooncake_transfer_engine import (
        get_mooncake_transfer_engine,
        init_mooncake_transfer_engine,
    )
    from sglang.srt.utils.network import get_local_ip_auto

    engine = get_mooncake_transfer_engine()
    if engine is not None:
        return engine
    gpu_id = getattr(scheduler, "gpu_id", None)
    if gpu_id is None:
        gpu_id = getattr(getattr(scheduler, "ps", None), "gpu_id", None)
    return init_mooncake_transfer_engine(
        get_local_ip_auto(),
        gpu_id=gpu_id,
        ib_device=scheduler.server_args.mooncake_ib_device,
    )


def _source_host_buf_infos(tree_cache) -> tuple[list[int], list[int]]:
    refs = _source_host_tensors(tree_cache)
    page_size = tree_cache.page_size
    ptrs = [int(ref.data_ptr()) for ref in refs]
    item_lens = [int(ref[0].nbytes) * page_size for ref in refs]
    return ptrs, item_lens


def _source_host_tensors(tree_cache) -> list[torch.Tensor]:
    host_pool = tree_cache.cache_controller.mem_pool_host
    if getattr(host_pool, "layout", None) != "layer_first":
        raise RuntimeError(
            "SharedHiCache direct transfer requires layer_first host layout, "
            f"got {getattr(host_pool, 'layout', None)!r}"
        )

    if hasattr(host_pool, "k_data_refs") and hasattr(host_pool, "v_data_refs"):
        refs = host_pool.k_data_refs + host_pool.v_data_refs
    elif hasattr(host_pool, "data_refs"):
        refs = host_pool.data_refs
    else:
        raise RuntimeError("Unsupported HiCache host pool for SharedHiCache direct transfer")
    return list(refs)


def _tensor_byte_view(tensor: torch.Tensor) -> torch.Tensor:
    if not tensor.is_contiguous():
        raise RuntimeError("SharedHiCache staging requires contiguous host KV tensors")
    return tensor.view(torch.uint8).reshape(-1)


def _validate_kv_item_lens_match(
    source_kv_item_lens: list[int], target_kv_item_lens: list[int]
) -> None:
    if len(source_kv_item_lens) != len(target_kv_item_lens):
        raise RuntimeError(
            "KV item length count mismatch: "
            f"source={len(source_kv_item_lens)} target={len(target_kv_item_lens)}"
        )

    src_item_lens = np.asarray(source_kv_item_lens, dtype=np.uint64)
    dst_item_lens = np.asarray(target_kv_item_lens, dtype=np.uint64)
    mismatched_items = np.nonzero(src_item_lens != dst_item_lens)[0]
    if len(mismatched_items) > 0:
        idx = int(mismatched_items[0])
        raise RuntimeError(
            "KV item length mismatch: "
            f"source={int(src_item_lens[idx])} target={int(dst_item_lens[idx])}"
        )


def _build_grouped_transfer_arrays(
    *,
    source_page_indices: np.ndarray,
    target_page_indices: np.ndarray,
    source_kv_ptrs: list[int],
    target_kv_ptrs: list[int],
    source_kv_item_lens: list[int],
    target_kv_item_lens: list[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    if len(source_kv_ptrs) != len(target_kv_ptrs):
        raise RuntimeError(
            "KV pointer count mismatch: "
            f"source={len(source_kv_ptrs)} target={len(target_kv_ptrs)}"
        )
    _validate_kv_item_lens_match(source_kv_item_lens, target_kv_item_lens)

    src_item_lens = np.asarray(source_kv_item_lens, dtype=np.uint64)
    dst_item_lens = np.asarray(target_kv_item_lens, dtype=np.uint64)
    src_blocks, dst_blocks = group_concurrent_contiguous(
        source_page_indices.astype(np.int32, copy=False),
        target_page_indices.astype(np.int32, copy=False),
    )
    if not src_blocks:
        empty = np.empty((0,), dtype=np.uint64)
        return empty, empty, empty, 0

    # Keep address arithmetic in uint64, matching existing disagg transfer code.
    # Some platforms can expose addresses beyond signed int64.
    src_ptrs = np.asarray(source_kv_ptrs, dtype=np.uint64)
    dst_ptrs = np.asarray(target_kv_ptrs, dtype=np.uint64)
    src_starts = np.fromiter((block[0] for block in src_blocks), dtype=np.uint64)
    dst_starts = np.fromiter((block[0] for block in dst_blocks), dtype=np.uint64)
    block_lens = np.fromiter((len(block) for block in src_blocks), dtype=np.uint64)

    src_addrs = (
        src_ptrs[:, None] + src_starts[None, :] * src_item_lens[:, None]
    ).reshape(-1)
    dst_addrs = (
        dst_ptrs[:, None] + dst_starts[None, :] * dst_item_lens[:, None]
    ).reshape(-1)
    lengths = (src_item_lens[:, None] * block_lens[None, :]).reshape(-1)
    return src_addrs, dst_addrs, lengths, len(src_blocks)


class MooncakeSharedHiCacheTransferBackend:
    """Mooncake-backed source-HiCache-host to target-GPU-device transfer helper."""

    name = "mooncake"

    def __init__(
        self,
        *,
        engine,
        tree_cache,
        target_kv_ptrs,
        target_kv_item_lens,
        parallel_metadata: Optional[Mapping[str, int]] = None,
        target_registered: bool = False,
        custom_mem_pool=None,
        transfer_parallelism: Optional[int] = None,
    ):
        self.engine = engine
        self.tree_cache = tree_cache
        self.target_session_id = engine.get_session_id()
        self.target_kv_ptrs = [int(ptr) for ptr in target_kv_ptrs]
        self.target_kv_item_lens = [int(length) for length in target_kv_item_lens]
        self.parallel_metadata = {
            key: int(value) for key, value in (parallel_metadata or {}).items()
        }
        self._target_registered = bool(target_registered)
        self._source_registered = False
        self._source_registered_ptrs: list[int] = []
        transport_info = engine.get_transport_info()
        self.ib_device = transport_info.get("ib_device")
        self.protocol = str(transport_info.get("protocol") or "rdma").lower()
        self.path_hint = str(
            transport_info.get("path_hint") or self.protocol
        ).lower()
        if transfer_parallelism is None:
            transfer_parallelism = default_shared_hicache_transfer_parallelism()
        self._transfer_parallelism = max(1, int(transfer_parallelism))
        self._transfer_executor: Optional[ThreadPoolExecutor] = None
        self._transfer_executor_lock = threading.Lock()
        self._staging_lock = threading.Lock()
        self._staging_buffer: Optional[StagingBuffer] = None
        self._staging_registered_ptr: Optional[int] = None
        self._custom_mem_pool = custom_mem_pool
        gpu_id = getattr(engine, "gpu_id", None)
        self._gpu_id = int(gpu_id if gpu_id is not None else 0)
        self._device = f"cuda:{self._gpu_id}"
        self._shutdown = False

    @classmethod
    def from_scheduler(cls, scheduler) -> Optional["MooncakeSharedHiCacheTransferBackend"]:
        server_args = scheduler.server_args
        backend = shared_hicache_transfer_backend_name(server_args)
        if backend not in {"auto", "mooncake"}:
            return None
        topology_rejection = _direct_topology_rejection(scheduler)
        if topology_rejection is not None:
            if backend == "mooncake":
                logger.warning(
                    "SharedHiCache Mooncake direct transfer disabled: %s",
                    topology_rejection,
                )
            else:
                logger.debug(
                    "SharedHiCache Mooncake direct transfer disabled: %s",
                    topology_rejection,
                )
            return None

        try:
            engine = _get_or_init_mooncake_transfer_engine(scheduler)

            target_pool = _target_kv_pool_from_scheduler(scheduler)
            target_kv_ptrs, target_kv_lens, target_kv_item_lens = (
                target_pool.get_contiguous_buf_infos()
            )
            registered, register_reason = engine.register_regions_checked(
                target_kv_ptrs, target_kv_lens
            )
            if not registered:
                logger.warning(
                    "SharedHiCache Mooncake disabled: target KV registration failed (%s)",
                    register_reason,
                )
                return None
            transfer = cls(
                engine=engine,
                tree_cache=scheduler.tree_cache,
                target_kv_ptrs=target_kv_ptrs,
                target_kv_item_lens=target_kv_item_lens,
                parallel_metadata=scheduler_parallel_metadata(scheduler),
                target_registered=True,
                custom_mem_pool=(
                    target_pool.maybe_get_custom_mem_pool()
                    if callable(getattr(target_pool, "maybe_get_custom_mem_pool", None))
                    else None
                ),
            )
            transfer._register_source_host_pool()
            if transfer.enabled:
                transfer._log_ready()
                return transfer
            transfer.shutdown()
            return None
        except Exception:
            if backend == "mooncake":
                logger.exception(
                    "SharedHiCache Mooncake direct transfer initialization failed"
                )
            else:
                logger.debug(
                    "SharedHiCache Mooncake direct transfer unavailable; using fallback",
                    exc_info=True,
                )
            return None

    @property
    def enabled(self) -> bool:
        return self._source_registered

    def target_descriptor(self) -> dict[str, Any]:
        return {
            "backend": self.name,
            "session_id": self.target_session_id,
            "transport": {
                "protocol": self.protocol,
                "ib_device": self.ib_device,
                "path_hint": self.path_hint,
                "transfer_parallelism": self._transfer_parallelism,
            },
            **self.parallel_metadata,
        }

    def _log_ready(self) -> None:
        if self.protocol == "rdma" and self.ib_device is None:
            logger.warning(
                "SharedHiCache Mooncake direct transfer enabled session=%s "
                "protocol=rdma ib_device=<none> path_hint=no_explicit_ib_device "
                "parallelism=%d; benchmark labels should not treat this as a "
                "configured RDMA/GDR path",
                self.target_session_id,
                self._transfer_parallelism,
            )
            return
        logger.info(
            "SharedHiCache Mooncake direct transfer enabled session=%s protocol=%s "
            "ib_device=%s path_hint=%s parallelism=%d",
            self.target_session_id,
            self.protocol,
            self.ib_device if self.ib_device is not None else "<none>",
            self.path_hint,
            self._transfer_parallelism,
        )

    def _register_source_host_pool(self) -> None:
        host_pool = self.tree_cache.cache_controller.mem_pool_host
        if getattr(host_pool, "layout", None) != "layer_first":
            logger.info(
                "SharedHiCache Mooncake direct transfer disabled for HiCache layout=%s; "
                "source host pool must be layer_first",
                getattr(host_pool, "layout", None),
            )
            return
        if not hasattr(host_pool, "kv_buffer"):
            logger.info("SharedHiCache Mooncake direct transfer disabled: no host kv_buffer")
            return
        try:
            _, source_kv_item_lens = _source_host_buf_infos(self.tree_cache)
            _validate_kv_item_lens_match(
                source_kv_item_lens, self.target_kv_item_lens
            )
        except RuntimeError as err:
            logger.info("SharedHiCache Mooncake direct transfer disabled: %s", err)
            return
        registered, register_reason = self.engine.register_regions_checked(
            [host_pool.kv_buffer.data_ptr()], [host_pool.kv_buffer.nbytes]
        )
        if not registered:
            logger.warning(
                "SharedHiCache Mooncake direct transfer disabled: host registration failed (%s)",
                register_reason,
            )
            return
        self._source_registered = True
        self._source_registered_ptrs = [int(host_pool.kv_buffer.data_ptr())]

    def shutdown(self) -> None:
        if self._shutdown:
            return
        self._shutdown = True

        # Keep Mooncake memory registrations process-lifetime, matching existing
        # Mooncake disagg/staging users of the shared transfer engine. The same
        # pointer can be registered by multiple in-process features, and SharedHiCache
        # cannot safely infer global ownership at scheduler shutdown.
        self._source_registered_ptrs = []
        self._source_registered = False

        self._target_registered = False
        executor = self._transfer_executor
        self._transfer_executor = None
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)

    def _should_stage_source_host_for_nvlink(self) -> bool:
        if "nvlink" not in self.path_hint and "nvlink" not in self.protocol:
            return False
        source_refs = _source_host_tensors(self.tree_cache)
        return any(not ref.is_cuda for ref in source_refs)

    def _ensure_staging_buffer(self, required_bytes: int) -> StagingBuffer:
        staging = self._staging_buffer
        if staging is not None and staging.fits(required_bytes):
            return staging

        if self._custom_mem_pool is None:
            raise RuntimeError(
                "SharedHiCache NVLink staging requires a Mooncake custom memory pool"
            )

        staging = StagingBuffer(
            required_bytes,
            self._device,
            self._gpu_id,
            custom_mem_pool=self._custom_mem_pool,
        )
        registered, reason = self.engine.register_regions_checked(
            [staging.get_ptr()], [staging.get_size()]
        )
        if not registered:
            raise RuntimeError(f"SharedHiCache staging registration failed: {reason}")
        self._staging_buffer = staging
        self._staging_registered_ptr = staging.get_ptr()
        return staging

    def _get_transfer_executor(self) -> ThreadPoolExecutor:
        with self._transfer_executor_lock:
            if self._transfer_executor is None:
                self._transfer_executor = ThreadPoolExecutor(
                    max_workers=self._transfer_parallelism,
                    thread_name_prefix="shared_hicache-mooncake-transfer",
                )
            return self._transfer_executor

    def _batch_transfer_sync(
        self,
        target_session_id: str,
        src_addrs: np.ndarray,
        dst_addrs: np.ndarray,
        lengths: np.ndarray,
    ) -> int:
        parallelism = min(self._transfer_parallelism, int(src_addrs.size))
        if parallelism <= 1:
            return self.engine.batch_transfer_sync(
                target_session_id,
                src_addrs.tolist(),
                dst_addrs.tolist(),
                lengths.tolist(),
            )

        executor = self._get_transfer_executor()
        futures = [
            executor.submit(
                self.engine.batch_transfer_sync,
                target_session_id,
                src_addrs[chunk].tolist(),
                dst_addrs[chunk].tolist(),
                lengths[chunk].tolist(),
            )
            for chunk in np.array_split(np.arange(src_addrs.size), parallelism)
            if chunk.size > 0
        ]
        # Drain all submitted chunks before reporting failure; the target may free
        # pages immediately after an error response.
        ret = 0
        first_error: Optional[BaseException] = None
        for future in as_completed(futures):
            try:
                chunk_ret = future.result()
            except Exception as err:
                if first_error is None:
                    first_error = err
                continue
            if chunk_ret != 0:
                ret = ret or int(chunk_ret)
        if first_error is not None:
            raise first_error
        return ret

    def _transfer_pages_via_gpu_staging(
        self,
        *,
        target_session_id: str,
        source_page_indices: np.ndarray,
        target_page_indices: np.ndarray,
        target_kv_ptrs: list[int],
        target_kv_item_lens: list[int],
    ) -> None:
        source_refs = _source_host_tensors(self.tree_cache)
        source_kv_item_lens = [
            int(ref[0].nbytes) * self.tree_cache.page_size for ref in source_refs
        ]
        _validate_kv_item_lens_match(source_kv_item_lens, target_kv_item_lens)
        if len(source_refs) != len(target_kv_ptrs):
            raise RuntimeError(
                "KV pointer count mismatch: "
                f"source={len(source_refs)} target={len(target_kv_ptrs)}"
            )

        src_blocks, dst_blocks = group_concurrent_contiguous(
            source_page_indices.astype(np.int32, copy=False),
            target_page_indices.astype(np.int32, copy=False),
        )
        if not src_blocks:
            return

        lengths: list[int] = []
        dst_addrs: list[int] = []
        staging_offsets: list[int] = []
        total_bytes = 0
        for src_block, dst_block in zip(src_blocks, dst_blocks):
            block_len = len(src_block)
            src_start = int(src_block[0])
            dst_start = int(dst_block[0])
            for item_len, dst_ptr in zip(source_kv_item_lens, target_kv_ptrs):
                length = int(item_len) * block_len
                lengths.append(length)
                dst_addrs.append(int(dst_ptr) + dst_start * int(item_len))
                staging_offsets.append(total_bytes)
                total_bytes += length

        with self._staging_lock:
            staging = self._ensure_staging_buffer(total_bytes)
            staging_base = int(staging.get_ptr())
            staging_src_addrs: list[int] = []
            transfer_index = 0
            for src_block in src_blocks:
                block_len = len(src_block)
                src_start = int(src_block[0])
                for source_ref, item_len in zip(source_refs, source_kv_item_lens):
                    length = int(item_len) * block_len
                    staging_offset = staging_offsets[transfer_index]
                    src_offset = src_start * int(item_len)
                    source_bytes = _tensor_byte_view(source_ref).narrow(
                        0, src_offset, length
                    )
                    staging.buffer.narrow(0, staging_offset, length).copy_(
                        source_bytes,
                        non_blocking=True,
                    )
                    staging_src_addrs.append(staging_base + staging_offset)
                    transfer_index += 1

            torch.cuda.synchronize(self._device)
            start = time.perf_counter()
            ret = self._batch_transfer_sync(
                target_session_id,
                np.asarray(staging_src_addrs, dtype=np.uint64),
                np.asarray(dst_addrs, dtype=np.uint64),
                np.asarray(lengths, dtype=np.uint64),
            )
            transfer_ms = (time.perf_counter() - start) * 1000
            logger.info(
                "SharedHiCache staged host pages through GPU staging blocks=%d slices=%d bytes=%d parallelism=%d ms=%.3f",
                len(src_blocks),
                len(staging_src_addrs),
                int(sum(lengths)),
                min(self._transfer_parallelism, len(staging_src_addrs)),
                transfer_ms,
            )
            if ret != 0:
                raise RuntimeError(f"Mooncake staged KV transfer failed with ret={ret}")

    def _source_host_buf_infos(self) -> tuple[list[int], list[int]]:
        return _source_host_buf_infos(self.tree_cache)

    def transfer_pages(
        self,
        *,
        target_session_id: str,
        source_page_indices: np.ndarray,
        target_page_indices: np.ndarray,
        target_kv_ptrs: list[int],
        target_kv_item_lens: list[int],
        target_metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        if self._should_stage_source_host_for_nvlink():
            self._transfer_pages_via_gpu_staging(
                target_session_id=target_session_id,
                source_page_indices=source_page_indices,
                target_page_indices=target_page_indices,
                target_kv_ptrs=target_kv_ptrs,
                target_kv_item_lens=target_kv_item_lens,
            )
            return

        source_kv_ptrs, source_kv_item_lens = self._source_host_buf_infos()
        src_addrs, dst_addrs, lengths, num_blocks = _build_grouped_transfer_arrays(
            source_page_indices=source_page_indices,
            target_page_indices=target_page_indices,
            source_kv_ptrs=source_kv_ptrs,
            target_kv_ptrs=target_kv_ptrs,
            source_kv_item_lens=source_kv_item_lens,
            target_kv_item_lens=target_kv_item_lens,
        )

        if src_addrs.size > 0:
            start = time.perf_counter()
            ret = self._batch_transfer_sync(
                target_session_id, src_addrs, dst_addrs, lengths
            )
            transfer_ms = (time.perf_counter() - start) * 1000
            logger.info(
                "SharedHiCache Mooncake transferred blocks=%d slices=%d bytes=%d parallelism=%d ms=%.3f",
                num_blocks,
                len(src_addrs),
                int(lengths.sum()),
                min(self._transfer_parallelism, int(src_addrs.size)),
                transfer_ms,
            )
            if ret != 0:
                raise RuntimeError(f"Mooncake direct KV transfer failed with ret={ret}")


def make_shared_hicache_transfer_backend(scheduler) -> Optional[SharedHiCacheTransferBackend]:
    backend = shared_hicache_transfer_backend_name(scheduler.server_args)
    topology_rejection = _direct_topology_rejection(scheduler)
    if topology_rejection is not None:
        if backend == "mooncake":
            raise RuntimeError(topology_rejection)
        logger.warning("SharedHiCache direct transfer unavailable: %s", topology_rejection)
        return None

    if backend == "mooncake":
        transfer = MooncakeSharedHiCacheTransferBackend.from_scheduler(scheduler)
        if transfer is None:
            raise RuntimeError(
                "SharedHiCache Mooncake transfer backend was requested but unavailable"
            )
        return transfer

    if backend != "auto":
        raise RuntimeError(
            f"SharedHiCache transfer backend {backend!r} is not supported; "
            "this path supports only 'mooncake'"
        )

    transfer = MooncakeSharedHiCacheTransferBackend.from_scheduler(scheduler)
    if transfer is not None:
        return transfer
    return None
