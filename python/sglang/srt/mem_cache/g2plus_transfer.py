from __future__ import annotations

import base64
import json
import logging
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Mapping, Optional, Protocol

import numpy as np

from sglang.srt.disaggregation.common.utils import group_concurrent_contiguous
from sglang.srt.environ import default_g2plus_transfer_parallelism, envs

logger = logging.getLogger(__name__)


class G2plusTransferBackend(Protocol):
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


def _target_kv_infos_from_scheduler(scheduler):
    from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool

    target_pool = scheduler.token_to_kv_pool_allocator.get_kvcache()
    if isinstance(target_pool, HybridLinearKVPool):
        target_pool = target_pool.full_kv_pool
    return target_pool.get_contiguous_buf_infos()


def _get_topology_value(scheduler, name: str) -> int:
    server_args = getattr(scheduler, "server_args", None)
    value = getattr(server_args, name, getattr(scheduler, name, 1))
    return int(value if value is not None else 1)


def _direct_topology_rejection(scheduler) -> Optional[str]:
    unsupported = []
    for name in ("tp_size", "pp_size", "attn_cp_size"):
        value = _get_topology_value(scheduler, name)
        if value != 1:
            unsupported.append(f"{name}={value}")
    if not unsupported:
        return None
    return (
        "G2plus direct transfer V0 supports only tp_size=1, pp_size=1, "
        f"and attn_cp_size=1; got {', '.join(unsupported)}"
    )


def _record_diagnostic(diagnostics: Optional[list[str]], message: str) -> None:
    if diagnostics is not None:
        diagnostics.append(message)


def _diagnostic_suffix(diagnostics: Optional[list[str]]) -> str:
    if not diagnostics:
        return ""
    return ": " + "; ".join(diagnostics)


def g2plus_config(server_args) -> Mapping[str, Any]:
    config = getattr(server_args, "g2plus_config", None)
    return config if isinstance(config, Mapping) else {}


def g2plus_config_value(server_args, key: str, default=None):
    config = g2plus_config(server_args)
    legacy_name = f"g2plus_{key}"
    return config.get(key, getattr(server_args, legacy_name, default))


def g2plus_transfer_backend_name(server_args, default: str = "auto") -> str:
    return str(g2plus_config_value(server_args, "transfer_backend", default)).lower()


def g2plus_timeout_secs(server_args, default: float = 1.0) -> float:
    return float(g2plus_config_value(server_args, "timeout_secs", default))


def _source_host_buf_infos(tree_cache) -> tuple[list[int], list[int]]:
    host_pool = tree_cache.cache_controller.mem_pool_host
    if getattr(host_pool, "layout", None) != "layer_first":
        raise RuntimeError(
            "G2plus direct transfer requires layer_first host layout, "
            f"got {getattr(host_pool, 'layout', None)!r}"
        )

    if hasattr(host_pool, "k_data_refs") and hasattr(host_pool, "v_data_refs"):
        refs = host_pool.k_data_refs + host_pool.v_data_refs
    elif hasattr(host_pool, "data_refs"):
        refs = host_pool.data_refs
    else:
        raise RuntimeError("Unsupported HiCache host pool for G2plus direct transfer")

    page_size = tree_cache.page_size
    ptrs = [int(ref.data_ptr()) for ref in refs]
    item_lens = [int(ref[0].nbytes) * page_size for ref in refs]
    return ptrs, item_lens


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


def _mooncake_ib_device(engine) -> Optional[str]:
    get_ib_device = getattr(engine, "get_ib_device", None)
    if callable(get_ib_device):
        ib_device = get_ib_device()
    else:
        ib_device = getattr(engine, "ib_device", None)
    if ib_device is None:
        return None
    ib_device = str(ib_device).strip()
    return ib_device or None


def _mooncake_protocol(engine) -> str:
    get_protocol = getattr(engine, "get_protocol", None)
    if callable(get_protocol):
        protocol = get_protocol()
    else:
        protocol = getattr(engine, "protocol", None)
    if protocol is None:
        return "rdma"
    protocol = str(protocol).strip().lower()
    return protocol or "rdma"


def _mooncake_path_hint(protocol: str, ib_device: Optional[str]) -> str:
    if protocol == "rdma":
        return (
            "explicit_ib_device"
            if ib_device is not None
            else "no_explicit_ib_device"
        )
    return protocol


def _mooncake_register_regions(
    engine, ptrs: list[int], lengths: list[int]
) -> tuple[bool, str]:
    ptrs = [int(ptr) for ptr in ptrs]
    lengths = [int(length) for length in lengths]
    if len(ptrs) != len(lengths):
        return (
            False,
            f"registration_length_mismatch:ptrs={len(ptrs)}:lengths={len(lengths)}",
        )

    # Mooncake 0.3.10 can return success from batch_register_memory even when
    # individual CUDA registrations fail. Prefer the scalar API when available
    # so G2plus does not advertise a target whose GPU KV addresses are absent
    # from Mooncake's segment descriptor.
    underlying_engine = getattr(engine, "engine", None)
    register_memory = getattr(underlying_engine, "register_memory", None)
    if callable(register_memory):
        for index, (ptr, length) in enumerate(zip(ptrs, lengths)):
            try:
                ret = int(register_memory(ptr, length))
            except Exception as err:
                return False, f"register_memory_exception:index={index}:error={err}"
            if ret != 0:
                return False, f"register_memory_failed:index={index}:ret={ret}"
        return True, "ok"

    ret = int(engine.batch_register(ptrs, lengths))
    if ret != 0:
        return False, f"batch_register_failed:ret={ret}"
    return True, "ok"


def _default_nixl_num_threads() -> int:
    value = envs.SGLANG_DISAGGREGATION_THREAD_POOL_SIZE.get()
    if value is None:
        value = 8
    return max(0, int(value))


def _apply_nixl_backend_thread_params(
    backend_name: str, backend_params: dict[str, str], num_threads: int
) -> None:
    if num_threads <= 0:
        return
    if backend_name in {"UCX", "OBJ"}:
        backend_params.setdefault("num_threads", str(num_threads))
    elif backend_name == "GDS_MT":
        backend_params.setdefault("thread_count", str(num_threads))
    elif backend_name == "UCCL":
        backend_params.setdefault("num_cpus", str(num_threads))


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

    # Keep address arithmetic in uint64, matching the existing NIXL disagg path.
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


def _make_nixl_req_array(
    addrs: np.ndarray, lengths: np.ndarray, gpu_id: int
) -> np.ndarray:
    if addrs.size == 0:
        return np.empty((0, 3), dtype=np.uint64)
    return np.column_stack(
        (
            addrs.astype(np.uint64, copy=False),
            lengths.astype(np.uint64, copy=False),
            np.full(addrs.shape, int(gpu_id), dtype=np.uint64),
        )
    )


class MooncakeG2plusTransferBackend:
    """Mooncake-backed source-G2-host to target-G1-device transfer helper."""

    name = "mooncake"

    def __init__(
        self,
        *,
        engine,
        tree_cache,
        target_kv_ptrs,
        target_kv_item_lens,
        target_registered: bool = False,
        transfer_parallelism: Optional[int] = None,
    ):
        self.engine = engine
        self.tree_cache = tree_cache
        self.target_session_id = engine.get_session_id()
        self.target_kv_ptrs = [int(ptr) for ptr in target_kv_ptrs]
        self.target_kv_item_lens = [int(length) for length in target_kv_item_lens]
        self._target_registered = bool(target_registered)
        self._source_registered = False
        self._source_registered_ptrs: list[int] = []
        self.ib_device = _mooncake_ib_device(engine)
        self.protocol = _mooncake_protocol(engine)
        if transfer_parallelism is None:
            transfer_parallelism = default_g2plus_transfer_parallelism()
        self._transfer_parallelism = max(1, int(transfer_parallelism))
        self._transfer_executor: Optional[ThreadPoolExecutor] = None
        self._transfer_executor_lock = threading.Lock()
        self._shutdown = False

    @classmethod
    def from_scheduler(
        cls, scheduler, diagnostics: Optional[list[str]] = None
    ) -> Optional["MooncakeG2plusTransferBackend"]:
        server_args = scheduler.server_args
        backend = g2plus_transfer_backend_name(server_args)
        if backend not in {"auto", "mooncake"}:
            return None
        topology_rejection = _direct_topology_rejection(scheduler)
        if topology_rejection is not None:
            if backend == "mooncake":
                logger.warning(
                    "G2plus Mooncake direct transfer disabled: %s",
                    topology_rejection,
                )
            else:
                logger.debug(
                    "G2plus Mooncake direct transfer disabled: %s",
                    topology_rejection,
                )
            _record_diagnostic(diagnostics, f"mooncake topology: {topology_rejection}")
            return None

        try:
            from sglang.srt.distributed.device_communicators.mooncake_transfer_engine import (
                init_mooncake_transfer_engine,
            )
            from sglang.srt.distributed.parallel_state import (
                get_mooncake_transfer_engine,
            )
            from sglang.srt.utils.network import get_local_ip_auto

            engine = get_mooncake_transfer_engine()
            if engine is None:
                engine = init_mooncake_transfer_engine(
                    get_local_ip_auto(),
                    gpu_id=scheduler.gpu_id,
                    ib_device=server_args.mooncake_ib_device,
                )

            target_kv_ptrs, target_kv_lens, target_kv_item_lens = (
                _target_kv_infos_from_scheduler(scheduler)
            )
            registered, register_reason = _mooncake_register_regions(
                engine, target_kv_ptrs, target_kv_lens
            )
            if not registered:
                logger.warning(
                    "G2plus Mooncake disabled: target KV registration failed (%s)",
                    register_reason,
                )
                _record_diagnostic(
                    diagnostics,
                    "mooncake target KV registration failed",
                )
                return None
            transfer = cls(
                engine=engine,
                tree_cache=scheduler.tree_cache,
                target_kv_ptrs=target_kv_ptrs,
                target_kv_item_lens=target_kv_item_lens,
                target_registered=True,
            )
            transfer._register_source_host_pool()
            if transfer.enabled:
                transfer._log_ready()
                return transfer
            transfer.shutdown()
            _record_diagnostic(
                diagnostics,
                "mooncake source host pool registration did not enable direct transfer",
            )
            return None
        except Exception as err:
            if backend == "mooncake":
                logger.exception(
                    "G2plus Mooncake direct transfer initialization failed"
                )
            else:
                logger.debug(
                    "G2plus Mooncake direct transfer unavailable; using fallback",
                    exc_info=True,
                )
            _record_diagnostic(
                diagnostics,
                f"mooncake initialization failed: {err}",
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
                "path_hint": _mooncake_path_hint(self.protocol, self.ib_device),
                "transfer_parallelism": self._transfer_parallelism,
            },
        }

    def _log_ready(self) -> None:
        path_hint = _mooncake_path_hint(self.protocol, self.ib_device)
        if self.protocol == "rdma" and self.ib_device is None:
            logger.warning(
                "G2plus Mooncake direct transfer enabled session=%s "
                "protocol=rdma ib_device=<none> path_hint=no_explicit_ib_device "
                "parallelism=%d; benchmark labels should not treat this as a "
                "configured RDMA/GDR path",
                self.target_session_id,
                self._transfer_parallelism,
            )
            return
        logger.info(
            "G2plus Mooncake direct transfer enabled session=%s protocol=%s "
            "ib_device=%s path_hint=%s parallelism=%d",
            self.target_session_id,
            self.protocol,
            self.ib_device if self.ib_device is not None else "<none>",
            path_hint,
            self._transfer_parallelism,
        )

    def _register_source_host_pool(self) -> None:
        host_pool = self.tree_cache.cache_controller.mem_pool_host
        if getattr(host_pool, "layout", None) != "layer_first":
            logger.info(
                "G2plus Mooncake direct transfer disabled for HiCache layout=%s; "
                "source host pool must be layer_first",
                getattr(host_pool, "layout", None),
            )
            return
        if not hasattr(host_pool, "kv_buffer"):
            logger.info("G2plus Mooncake direct transfer disabled: no host kv_buffer")
            return
        try:
            _, source_kv_item_lens = _source_host_buf_infos(self.tree_cache)
            _validate_kv_item_lens_match(
                source_kv_item_lens, self.target_kv_item_lens
            )
        except RuntimeError as err:
            logger.info("G2plus Mooncake direct transfer disabled: %s", err)
            return
        registered, register_reason = _mooncake_register_regions(
            self.engine, [host_pool.kv_buffer.data_ptr()], [host_pool.kv_buffer.nbytes]
        )
        if not registered:
            logger.warning(
                "G2plus Mooncake direct transfer disabled: host registration failed (%s)",
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
        # pointer can be registered by multiple in-process features, and G2plus
        # cannot safely infer global ownership at scheduler shutdown.
        self._source_registered_ptrs = []
        self._source_registered = False

        self._target_registered = False
        executor = self._transfer_executor
        self._transfer_executor = None
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)

    def _get_transfer_executor(self) -> ThreadPoolExecutor:
        with self._transfer_executor_lock:
            if self._transfer_executor is None:
                self._transfer_executor = ThreadPoolExecutor(
                    max_workers=self._transfer_parallelism,
                    thread_name_prefix="g2plus-mooncake-transfer",
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
            logger.debug(
                "G2plus Mooncake transferred blocks=%d slices=%d bytes=%d parallelism=%d ms=%.3f",
                num_blocks,
                len(src_addrs),
                int(lengths.sum()),
                min(self._transfer_parallelism, int(src_addrs.size)),
                transfer_ms,
            )
            if ret != 0:
                raise RuntimeError(f"Mooncake direct KV transfer failed with ret={ret}")

class NixlG2plusTransferBackend:
    """NIXL-backed source-G2-host to target-G1-device transfer helper."""

    name = "nixl"

    def __init__(
        self,
        *,
        agent,
        tree_cache,
        target_kv_ptrs,
        target_kv_lens,
        target_kv_item_lens,
        gpu_id: int,
        timeout_secs: float,
        backend_name: str = "unknown",
        transfer_parallelism: Optional[int] = None,
    ):
        self.agent = agent
        self.tree_cache = tree_cache
        self.agent_name = str(getattr(agent, "name", f"g2plus-{uuid.uuid4().hex}"))
        self.backend_name = str(backend_name)
        self.target_session_id = self.agent_name
        self.target_kv_ptrs = [int(ptr) for ptr in target_kv_ptrs]
        self.target_kv_lens = [int(length) for length in target_kv_lens]
        self.target_kv_item_lens = [int(length) for length in target_kv_item_lens]
        self.gpu_id = int(gpu_id)
        self.timeout_secs = float(timeout_secs)
        if transfer_parallelism is None:
            transfer_parallelism = default_g2plus_transfer_parallelism()
        self._transfer_parallelism = max(1, int(transfer_parallelism))
        self._target_registered = False
        self._source_registered = False
        self._target_descs = None
        self._source_descs = None
        self._remote_agent_lock = threading.Lock()
        self._remote_agent_names: set[str] = set()
        self._shutdown = False

    @classmethod
    def from_scheduler(
        cls, scheduler, diagnostics: Optional[list[str]] = None
    ) -> Optional["NixlG2plusTransferBackend"]:
        server_args = scheduler.server_args
        backend = g2plus_transfer_backend_name(server_args)
        if backend not in {"auto", "nixl"}:
            return None
        topology_rejection = _direct_topology_rejection(scheduler)
        if topology_rejection is not None:
            if backend == "nixl":
                logger.warning(
                    "G2plus NIXL direct transfer disabled: %s",
                    topology_rejection,
                )
            else:
                logger.debug(
                    "G2plus NIXL direct transfer disabled: %s",
                    topology_rejection,
                )
            _record_diagnostic(diagnostics, f"nixl topology: {topology_rejection}")
            return None

        try:
            from nixl._api import nixl_agent, nixl_agent_config

            backend_name = envs.SGLANG_DISAGGREGATION_NIXL_BACKEND.get()
            backend_params = json.loads(
                envs.SGLANG_DISAGGREGATION_NIXL_BACKEND_PARAMS.get()
            )
            if not isinstance(backend_params, dict) or not all(
                isinstance(key, str) and isinstance(value, str)
                for key, value in backend_params.items()
            ):
                raise ValueError(
                    "SGLANG_DISAGGREGATION_NIXL_BACKEND_PARAMS must be a JSON "
                    "object with string keys and string values"
                )
            num_threads = _default_nixl_num_threads()
            agent = nixl_agent(
                f"sglang-g2plus-rank{scheduler.tp_rank}-{uuid.uuid4().hex[:8]}",
                nixl_agent_config(backends=[], num_threads=num_threads),
            )
            _apply_nixl_backend_thread_params(backend_name, backend_params, num_threads)
            agent.create_backend(backend_name, backend_params)
            available_plugins = agent.get_plugin_list()
            if backend_name not in available_plugins:
                raise ValueError(
                    f"NIXL backend {backend_name!r} not found. "
                    f"Available: {available_plugins}"
                )

            target_kv_ptrs, target_kv_lens, target_kv_item_lens = (
                _target_kv_infos_from_scheduler(scheduler)
            )
            transfer = cls(
                agent=agent,
                tree_cache=scheduler.tree_cache,
                target_kv_ptrs=target_kv_ptrs,
                target_kv_lens=target_kv_lens,
                target_kv_item_lens=target_kv_item_lens,
                gpu_id=scheduler.gpu_id,
                timeout_secs=g2plus_timeout_secs(server_args),
                backend_name=backend_name,
            )
            transfer._register_target_device_pool()
            transfer._register_source_host_pool()
            if transfer.enabled:
                logger.info(
                    "G2plus NIXL direct transfer enabled agent=%s backend=%s gpu_id=%d timeout_secs=%.3f parallelism=%d num_threads=%d",
                    transfer.agent_name,
                    transfer.backend_name,
                    transfer.gpu_id,
                    transfer.timeout_secs,
                    transfer._transfer_parallelism,
                    num_threads,
                )
                return transfer
            transfer.shutdown()
            _record_diagnostic(
                diagnostics,
                "nixl memory registration did not enable direct transfer",
            )
            return None
        except Exception as err:
            if backend == "nixl":
                logger.exception("G2plus NIXL direct transfer initialization failed")
            else:
                logger.debug(
                    "G2plus NIXL direct transfer unavailable; using fallback",
                    exc_info=True,
                )
            _record_diagnostic(diagnostics, f"nixl initialization failed: {err}")
            return None

    @property
    def enabled(self) -> bool:
        return self._target_registered and self._source_registered

    def target_descriptor(self) -> dict[str, Any]:
        return {
            "backend": self.name,
            "agent_name": self.agent_name,
            "agent_metadata_b64": base64.b64encode(
                self.agent.get_agent_metadata()
            ).decode("ascii"),
            "gpu_id": self.gpu_id,
            "transport": {
                "backend": self.backend_name,
                "gpu_id": self.gpu_id,
                "transfer_parallelism": self._transfer_parallelism,
            },
        }

    def _register_target_device_pool(self) -> None:
        addrs = [
            (ptr, length, self.gpu_id, "")
            for ptr, length in zip(self.target_kv_ptrs, self.target_kv_lens)
        ]
        if not addrs:
            logger.warning("G2plus NIXL disabled: no target KV buffers")
            return
        descs = self.agent.register_memory(addrs, "VRAM")
        if not descs:
            logger.warning("G2plus NIXL disabled: target KV registration failed")
            return
        self._target_descs = descs
        self._target_registered = True

    def _register_source_host_pool(self) -> None:
        host_pool = self.tree_cache.cache_controller.mem_pool_host
        if getattr(host_pool, "layout", None) != "layer_first":
            logger.info(
                "G2plus NIXL direct transfer disabled for HiCache layout=%s; "
                "source host pool must be layer_first",
                getattr(host_pool, "layout", None),
            )
            return
        if not hasattr(host_pool, "kv_buffer"):
            logger.info("G2plus NIXL direct transfer disabled: no host kv_buffer")
            return
        try:
            _, source_kv_item_lens = _source_host_buf_infos(self.tree_cache)
            _validate_kv_item_lens_match(
                source_kv_item_lens, self.target_kv_item_lens
            )
        except RuntimeError as err:
            logger.info("G2plus NIXL direct transfer disabled: %s", err)
            return
        descs = self.agent.register_memory(
            [
                (
                    int(host_pool.kv_buffer.data_ptr()),
                    int(host_pool.kv_buffer.nbytes),
                    0,
                    "",
                )
            ],
            "DRAM",
        )
        if not descs:
            logger.warning("G2plus NIXL disabled: source host registration failed")
            return
        self._source_descs = descs
        self._source_registered = True

    def _unregister_memory(self, descs, label: str) -> None:
        if not descs:
            return
        unregister = getattr(self.agent, "unregister_memory", None) or getattr(
            self.agent, "deregister_memory", None
        )
        if unregister is None:
            logger.debug(
                "G2plus NIXL agent has no memory unregister hook for %s", label
            )
            return
        try:
            unregister(descs)
        except Exception:
            logger.warning(
                "G2plus NIXL failed to unregister %s memory", label, exc_info=True
            )

    def shutdown(self) -> None:
        if self._shutdown:
            return
        self._shutdown = True
        self._unregister_memory(self._source_descs, "source host")
        self._source_descs = None
        self._source_registered = False
        self._unregister_memory(self._target_descs, "target KV")
        self._target_descs = None
        self._target_registered = False

    def _parse_target_metadata(
        self, target_metadata: Optional[Mapping[str, Any]]
    ) -> tuple[str, bytes, int]:
        if not isinstance(target_metadata, Mapping):
            raise RuntimeError("NIXL direct transfer requires target metadata")

        try:
            agent_name_raw = target_metadata["agent_name"]
            metadata_b64_raw = target_metadata["agent_metadata_b64"]
        except KeyError as err:
            raise RuntimeError(f"NIXL target metadata missing {err}") from err

        agent_name = str(agent_name_raw)
        if not agent_name or agent_name_raw is None:
            raise RuntimeError("NIXL target metadata has empty agent_name")

        metadata_b64 = str(metadata_b64_raw)
        if not metadata_b64 or metadata_b64_raw is None:
            raise RuntimeError("NIXL target metadata has empty agent_metadata_b64")
        try:
            agent_metadata = base64.b64decode(metadata_b64, validate=True)
        except Exception as err:
            raise RuntimeError("NIXL target metadata has invalid agent_metadata_b64") from err
        if not agent_metadata:
            raise RuntimeError("NIXL target metadata decoded to empty agent metadata")

        try:
            target_gpu_id = int(target_metadata.get("gpu_id", 0))
        except (TypeError, ValueError) as err:
            raise RuntimeError("NIXL target metadata has invalid gpu_id") from err
        if target_gpu_id < 0 or target_gpu_id > np.iinfo(np.uint32).max:
            raise RuntimeError("NIXL target metadata gpu_id out of range")

        return agent_name, agent_metadata, target_gpu_id

    def _ensure_remote_agent(self, agent_name: str, agent_metadata: bytes) -> str:
        with self._remote_agent_lock:
            if agent_name not in self._remote_agent_names:
                self.agent.add_remote_agent(agent_metadata)
                self._remote_agent_names.add(agent_name)
        return agent_name

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
        target_agent_name, target_agent_metadata, target_gpu_id = (
            self._parse_target_metadata(target_metadata)
        )
        target_agent_name = self._ensure_remote_agent(
            target_agent_name, target_agent_metadata
        )
        source_kv_ptrs, source_kv_item_lens = _source_host_buf_infos(self.tree_cache)
        src_addrs, dst_addrs, lengths, num_blocks = _build_grouped_transfer_arrays(
            source_page_indices=source_page_indices,
            target_page_indices=target_page_indices,
            source_kv_ptrs=source_kv_ptrs,
            target_kv_ptrs=target_kv_ptrs,
            source_kv_item_lens=source_kv_item_lens,
            target_kv_item_lens=target_kv_item_lens,
        )

        if src_addrs.size == 0:
            return

        parallelism = min(self._transfer_parallelism, int(src_addrs.size))
        chunks = [
            chunk
            for chunk in np.array_split(np.arange(src_addrs.size), parallelism)
            if chunk.size > 0
        ]
        start = time.perf_counter()
        handles = []
        first_error: Optional[BaseException] = None
        try:
            for chunk_index, chunk in enumerate(chunks):
                src_reqs = _make_nixl_req_array(src_addrs[chunk], lengths[chunk], 0)
                dst_reqs = _make_nixl_req_array(
                    dst_addrs[chunk], lengths[chunk], target_gpu_id
                )
                src_descs = self.agent.get_xfer_descs(src_reqs, "DRAM")
                dst_descs = self.agent.get_xfer_descs(dst_reqs, "VRAM")
                notif = f"g2plus_{time.time_ns()}_{chunk_index}".encode("ascii")
                handle = self.agent.initialize_xfer(
                    "WRITE", src_descs, dst_descs, target_agent_name, notif
                )
                if not handle:
                    first_error = RuntimeError(
                        "NIXL direct KV transfer failed to create transfer"
                    )
                    break
                handles.append(handle)

                state = self.agent.transfer(handle)
                if state == "ERR":
                    first_error = RuntimeError(
                        "NIXL direct KV transfer failed to post transfer"
                    )
                    break

            deadline = time.monotonic() + self.timeout_secs
            pending_handles = list(handles)
            while pending_handles:
                next_pending = []
                for handle in pending_handles:
                    state = self.agent.check_xfer_state(handle)
                    if state == "DONE":
                        continue
                    if state == "ERR":
                        if first_error is None:
                            first_error = RuntimeError(
                                "NIXL direct KV transfer encountered ERR"
                            )
                        continue
                    next_pending.append(handle)
                pending_handles = next_pending
                if not pending_handles:
                    break
                if time.monotonic() > deadline:
                    timeout_error = TimeoutError(
                        "NIXL direct KV transfer timed out waiting for "
                        f"{len(pending_handles)} handles to drain"
                    )
                    if first_error is not None:
                        timeout_error.__cause__ = first_error
                    first_error = timeout_error
                    break
                time.sleep(0)
            if first_error is not None:
                raise first_error
        finally:
            for handle in handles:
                try:
                    self.agent.release_xfer_handle(handle)
                except Exception:
                    logger.warning(
                        "G2plus NIXL failed to release transfer handle", exc_info=True
                    )

        transfer_ms = (time.perf_counter() - start) * 1000
        logger.debug(
            "G2plus NIXL transferred blocks=%d slices=%d bytes=%d parallelism=%d ms=%.3f",
            num_blocks,
            len(src_addrs),
            int(lengths.sum()),
            parallelism,
            transfer_ms,
        )


def make_g2plus_transfer_backend(
    scheduler, diagnostics: Optional[list[str]] = None
) -> Optional[G2plusTransferBackend]:
    backend = g2plus_transfer_backend_name(scheduler.server_args)
    topology_rejection = _direct_topology_rejection(scheduler)
    if topology_rejection is not None:
        if backend in {"mooncake", "nixl"}:
            raise RuntimeError(topology_rejection)
        logger.warning("G2plus direct transfer unavailable: %s", topology_rejection)
        _record_diagnostic(diagnostics, f"topology: {topology_rejection}")
        return None

    if backend == "mooncake":
        transfer = MooncakeG2plusTransferBackend.from_scheduler(
            scheduler, diagnostics=diagnostics
        )
        if transfer is None:
            raise RuntimeError(
                "G2plus Mooncake transfer backend was requested but unavailable"
                f"{_diagnostic_suffix(diagnostics)}"
            )
        return transfer

    if backend == "nixl":
        transfer = NixlG2plusTransferBackend.from_scheduler(
            scheduler, diagnostics=diagnostics
        )
        if transfer is None:
            raise RuntimeError(
                "G2plus NIXL transfer backend was requested but unavailable"
                f"{_diagnostic_suffix(diagnostics)}"
            )
        return transfer

    if backend == "auto":
        diagnostic_count = len(diagnostics) if diagnostics is not None else 0
        transfer = MooncakeG2plusTransferBackend.from_scheduler(
            scheduler, diagnostics=diagnostics
        )
        if transfer is not None:
            return transfer
        if diagnostics is not None and len(diagnostics) == diagnostic_count:
            _record_diagnostic(diagnostics, "mooncake unavailable")
        diagnostic_count = len(diagnostics) if diagnostics is not None else 0
        transfer = NixlG2plusTransferBackend.from_scheduler(
            scheduler, diagnostics=diagnostics
        )
        if transfer is not None:
            return transfer
        if diagnostics is not None and len(diagnostics) == diagnostic_count:
            _record_diagnostic(diagnostics, "nixl unavailable")
        return None

    return None
