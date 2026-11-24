# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to SGLang project

"""HiCache L3 backend ``egm``: KV pages in host memory exported over NVLink.

For systems whose GPUs share an NVLink domain with fabric handles supported. A
provider process (``provider.py``) allocates host memory with
``CU_MEM_LOCATION_TYPE_HOST_NUMA`` and ``CU_MEM_HANDLE_TYPE_FABRIC`` and writes
one handle file per segment. Each tensor-parallel rank imports one segment, maps
it into its GPU's address space and moves pages between its pinned L2 host pages
and the segment with driver copies. The page index lives only in the rank's
memory, so it is empty at every start.

Pages are checksummed when written and verified when read; a page is a hit only
after its bytes landed in L2 and passed the check. A copy error, checksum
mismatches over ``egm_max_mismatch``, a lost provider heartbeat or a provider
stop refuses the tier for the rest of the run: every query then misses and the
server keeps serving from L1 / L2.

Extra-config keys:
  egm_handles        handle file path template; "{rank}" is the TP rank and
                     "{node}" the node index (required)
  egm_rank_map       list of node indices, one per rank (default: the rank)
  egm_streams        copy streams per direction (default 2)
  egm_hb_timeout_s   provider heartbeat timeout in seconds (default 5)
  egm_max_mismatch   checksum mismatches tolerated before refusing (default 0)
  egm_crc_threads    checksum worker threads per rank (default 4)
  egm_region_shares  {pool name: fraction of the data area}; default
                     proportional to the host pool bytes
  egm_status_dir     directory for per-rank JSON status files
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

import torch

from sglang.srt.mem_cache.hicache_storage import (
    HiCacheStorage,
    HiCacheStorageConfig,
    HiCacheStorageExtraInfo,
    PoolHitPolicy,
    PoolName,
    PoolTransfer,
    PoolTransferResult,
)
from sglang.srt.mem_cache.storage.egm import store

logger = logging.getLogger(__name__)

# Region order inside the data area; other pools follow by name.
_POOL_ORDER = {PoolName.KV.value: 0, PoolName.DRAFT.value: 1}
# How long close() waits for a shut already under way on another thread.
_CLOSE_WAIT_S = 10.0
_DRAIN_TIMEOUT_S = 30.0
_MONITOR_PERIOD_S = 1.0
_STATUS_PERIOD_S = 10.0
_SUPPORTED_LAYOUTS = ("page_first", "page_first_direct")


class EgmStorage(HiCacheStorage):
    def __init__(self, storage_config: HiCacheStorageConfig, mem_pool_host: Any = None):
        cfg = dict(storage_config.extra_config or {})
        self.cfg = cfg
        self.rank = int(storage_config.tp_rank)
        why = self._config_problems(storage_config, cfg)
        if why:
            raise ValueError(self._refusal("; ".join(why)))

        rank_map = cfg.get("egm_rank_map")
        self.node = int(rank_map[self.rank]) if rank_map else self.rank
        self.handle_path = str(cfg["egm_handles"]).format(
            rank=self.rank, node=self.node
        )
        self.handle = self._load_handle(self.handle_path)
        self.epoch = int(self.handle["epoch"], 16)
        self.hb_timeout = float(cfg.get("egm_hb_timeout_s", 5.0))
        self.max_mismatch = int(cfg.get("egm_max_mismatch", 0))
        self.status_dir = cfg.get("egm_status_dir")
        streams = int(cfg.get("egm_streams", 2))
        self.seg, self.copier, self.small = self._open_device(self.handle, streams)
        nthreads = int(cfg.get("egm_crc_threads", 4))
        self.pool_exec = (
            ThreadPoolExecutor(max_workers=nthreads, thread_name_prefix="egm-crc")
            if nthreads > 1
            else None
        )

        self.pools: Dict[str, Any] = {}
        self.specs: Dict[str, store.PoolSpec] = {}
        self.store: Optional[store.Store] = None
        self.pools_ready = False
        self.build_lock = threading.Lock()
        self.closed = False
        self.monitor_state = {
            "hb": None,
            "hb_changed": time.monotonic(),
            "bad_reads": 0,
            "last": None,
        }
        self.monitor: Optional[threading.Thread] = None
        self.stop_evt = threading.Event()
        self.unmapped = False
        # One shut path: the unmap runs once, and no control-block access
        # overlaps or follows it.
        self.shut_lock = threading.Lock()
        self.shut_started = False
        self.shut_done = threading.Event()
        self.ctrl_lock = threading.Lock()
        self.ctrl_closed = False
        logger.info(
            "EGM rank %d: node %d, epoch %x, segment %.1f GiB (data %.1f GiB), "
            "import %.3f s",
            self.rank,
            self.node,
            self.epoch,
            self.seg.total / 2**30,
            self.handle["data_size"] / 2**30,
            self.seg.times.get("total_s", 0.0),
        )

    def _refusal(self, why: str) -> str:
        return "EGM storage refused on rank %d: %s" % (self.rank, why)

    @staticmethod
    def _config_problems(storage_config: HiCacheStorageConfig, cfg: dict) -> List[str]:
        why = []
        if storage_config.is_rank_replicated:
            why.append(
                "rank-replicated device pool (ranks other than 0 skip the backup)"
            )
        if storage_config.should_split_heads:
            why.append("head splitting across heterogeneous TP is not supported")
        if int(storage_config.pp_size) != 1 or int(storage_config.attn_cp_size) != 1:
            why.append("pipeline / context parallelism is not supported")
        if not cfg.get("egm_handles"):
            why.append("egm_handles is missing")
        return why

    def _load_handle(self, path: str) -> dict:
        try:
            with open(path, "r") as fh:
                handle = json.load(fh)
        except (OSError, ValueError) as e:
            raise ValueError(self._refusal("handle %s: %r" % (path, e))) from e
        bad = store.validate_handle(handle, node_index=self.node)
        if bad:
            raise ValueError(self._refusal("handle %s: %s" % (path, ", ".join(bad))))
        return handle

    def _open_device(
        self, handle: dict, streams: int
    ) -> Tuple[Any, store.DirCopier, Any]:
        """(segment, copier, control-block IO) on the current device."""
        from sglang.srt.mem_cache.storage.egm import cuda_vmm

        ctx = cuda_vmm.Context(torch.cuda.current_device())
        seg = cuda_vmm.ImportedSegment(ctx, handle)
        copier = store.DirCopier(
            get_copier=cuda_vmm.Copier(ctx, streams),
            set_copier=cuda_vmm.Copier(ctx, streams),
        )
        return seg, copier, cuda_vmm.SmallIO(ctx)

    # ------------------------------------------------------------ registration
    def register_mem_pool_host(self, mem_pool_host):
        super().register_mem_pool_host(mem_pool_host)
        self._check_pool(mem_pool_host, PoolName.KV.value)
        self.pools[PoolName.KV.value] = mem_pool_host

    def register_mem_host_pool_v2(self, host_pool, host_pool_name):
        super().register_mem_host_pool_v2(host_pool, host_pool_name)
        key = str(host_pool_name)
        if key == PoolName.KV.value and key in self.pools:
            return
        if self.store is not None:
            # Regions are fixed at the first storage call; switch the tier off
            # instead of failing the caller.
            self._shut("pool %s registered after the regions were fixed" % key)
            return
        self._check_pool(host_pool, key)
        self.pools[key] = host_pool

    def _check_pool(self, pool, key: str) -> None:
        if pool.layout not in _SUPPORTED_LAYOUTS:
            raise ValueError(
                self._refusal(
                    "pool %s layout %r (needs --hicache-mem-layout page_first)"
                    % (key, pool.layout)
                )
            )
        if pool.kv_buffer is None:
            raise ValueError(self._refusal("pool %s has no KV buffer" % key))

    def on_host_pools_registered(self) -> None:
        # Sidecar pools may still arrive through register_mem_host_pool_v2
        # until the server serves, so regions are planned at the first call.
        if PoolName.KV.value not in self.pools:
            raise ValueError(self._refusal("no KV anchor pool registered"))
        if self.pools_ready:
            return
        self.pools_ready = True
        self._write_rank_block(store.RANK_ATTACHED)
        self._start_monitor()

    def _start_monitor(self) -> None:
        self.monitor = threading.Thread(
            target=self._monitor, name="egm-monitor", daemon=True
        )
        self.monitor.start()

    def _ensure_store(self) -> bool:
        if self.store is not None:
            return True
        if not self.pools_ready or self.unmapped:
            return False
        with self.build_lock:
            if self.store is not None:
                return True
            try:
                self.store = self._build_store()
            except Exception as e:
                self._shut("region plan failed: %r" % (e,))
                return False
        return True

    def _pool_spec(self, key: str) -> store.PoolSpec:
        pool = self.pools[key]
        ps = int(pool.page_size)
        ptrs, sizes = pool.get_page_buffer_meta(torch.arange(ps, dtype=torch.int64))
        if any(isinstance(p, (list, tuple)) for p in ptrs):
            raise ValueError(self._refusal("pool %s uses multi-buffer pages" % key))
        comp_sizes = tuple(int(s) for s in sizes)
        weight = float(sum(comp_sizes)) * max(1, int(pool.size) // ps)
        return store.PoolSpec(name=key, comp_sizes=comp_sizes, weight=weight)

    def _build_store(self) -> store.Store:
        keys = sorted(self.pools, key=lambda k: (_POOL_ORDER.get(k, 9), k))
        specs = [self._pool_spec(k) for k in keys]
        self.specs = {s.name: s for s in specs}
        regions = store.plan_regions(
            specs,
            data_off=int(self.handle["data_off"]),
            data_size=int(self.handle["data_size"]),
            shares=self.cfg.get("egm_region_shares"),
        )
        for r in regions:
            if r.slots <= 0:
                raise ValueError(self._refusal("region %s has no slot" % r.name))
        crc_map = None
        if self.pool_exec is not None:
            pool_exec = self.pool_exec

            def crc_map(fn, items):
                return list(pool_exec.map(fn, items, chunksize=8))

        logger.info(
            "EGM rank %d: regions %s",
            self.rank,
            ", ".join(
                "%s %d slots x %d B" % (r.name, r.slots, r.page_bytes) for r in regions
            ),
        )
        return store.Store(
            regions,
            seg_va=self.seg.va,
            copier=self.copier,
            host_view=store.host_view,
            max_mismatch=self.max_mismatch,
            crc_map=crc_map,
            log=lambda m: logger.error("EGM rank %d: %s", self.rank, m),
        )

    # ------------------------------------------------- control block + monitor
    def _write_rank_block(self, state: int) -> None:
        with self.ctrl_lock:
            if self.ctrl_closed:
                return
            self._write_rank_block_now(state)

    def _write_rank_block_now(self, state: int) -> None:
        self.small.write(
            self.seg.va + store.rank_slot_offset(self.rank),
            store.pack_rank_block(self.rank, state, self.epoch, int(time.time())),
        )

    def _read_provider(self) -> Optional[store.ProviderBlock]:
        with self.ctrl_lock:
            if self.ctrl_closed:
                return None
            raw = self.small.read(self.seg.va, store.PROV_BLOCK)
        return store.unpack_provider_block(raw)

    def _monitor_once(self, now: float) -> Optional[str]:
        """The reason to refuse the tier, or None while the provider is healthy."""
        st = self.monitor_state
        try:
            blk = self._read_provider()
        except Exception as e:
            return "control block read failed: %r" % (e,)
        st["last"] = blk
        if blk is None:
            st["bad_reads"] += 1
            if now - st["hb_changed"] > self.hb_timeout:
                return "no valid control block for %.1f s" % (now - st["hb_changed"])
            return None
        if blk.epoch != self.epoch:
            return "segment epoch %x != handle epoch %x" % (blk.epoch, self.epoch)
        if blk.state != store.PROV_RUN:
            return "provider state %d (stopping)" % blk.state
        if blk.heartbeat != st["hb"]:
            st["hb"] = blk.heartbeat
            st["hb_changed"] = now
        elif now - st["hb_changed"] > self.hb_timeout:
            return "provider heartbeat stale for %.1f s" % (now - st["hb_changed"])
        return None

    def _monitor_step(self, now: float) -> Optional[str]:
        why = self._monitor_once(now)
        if why is None and self.store is not None and self.store.refused is not None:
            why = "store: " + self.store.refused
        return why

    def _monitor(self) -> None:
        last_status = 0.0
        while not self.stop_evt.wait(_MONITOR_PERIOD_S):
            now = time.monotonic()
            why = self._monitor_step(now)
            if why is not None:
                self._shut("monitor: " + why)
                return
            if self.status_dir and now - last_status >= _STATUS_PERIOD_S:
                last_status = now
                self._write_status()

    def _shut(self, why: str, wait_s: float = 0.0) -> None:
        """Refuse the tier, wait for in-flight copies, ack the provider and unmap.

        The first caller does the work; a later one returns at once, or after
        waiting up to ``wait_s`` for the first to finish.
        """
        if self.store is not None:
            self.store.refuse(why)
        with self.shut_lock:
            first = not self.shut_started
            self.shut_started = True
        if not first:
            if wait_s > 0:
                self.shut_done.wait(wait_s)
            return
        try:
            self.stop_evt.set()
            drained = (
                self.store.drain(_DRAIN_TIMEOUT_S) if self.store is not None else True
            )
            if self.unmapped:
                return
            if not drained:
                logger.error(
                    "EGM rank %d: copies still in flight after refusal; "
                    "segment left mapped",
                    self.rank,
                )
                self._write_status()
                return
            self._detach_and_unmap()
            logger.error(
                "EGM rank %d: tier off for this run (%s); serving continues without L3",
                self.rank,
                why,
            )
            self._write_status()
        finally:
            self.shut_done.set()

    def _detach_and_unmap(self) -> None:
        # The control block is closed before the unmap, under its lock, so no
        # monitor read or write reaches the range during or after the unmap.
        with self.ctrl_lock:
            try:
                self._write_rank_block_now(store.RANK_DETACHED)
            except Exception as e:
                logger.error("EGM rank %d: detach ack failed: %r", self.rank, e)
            self.ctrl_closed = True
            errs = self.seg.close()
            self.unmapped = True
        if errs:
            logger.error("EGM rank %d: unmap errors %s", self.rank, errs)

    def _write_status(self) -> None:
        if not self.status_dir:
            return
        try:
            snap = self.store.snapshot() if self.store is not None else {}
            s = sorted(self.copier.get_copier.issue_ms[-256:])
            snap.update(
                {
                    "rank": self.rank,
                    "node": self.node,
                    "epoch": "%x" % self.epoch,
                    "time": time.time(),
                    "unmapped": self.unmapped,
                    "monitor": dict(self.monitor_state),
                    "get_issue_ms_p50": s[len(s) // 2] if s else None,
                    "get_issue_ms_p99": (
                        s[min(len(s) - 1, int(0.99 * len(s)))] if s else None
                    ),
                }
            )
            os.makedirs(self.status_dir, exist_ok=True)
            p = os.path.join(self.status_dir, "status_rank%d.json" % self.rank)
            with open(p + ".tmp", "w") as fh:
                json.dump(snap, fh, sort_keys=True, default=str)
            os.replace(p + ".tmp", p)
        except Exception as e:
            logger.warning("EGM rank %d: status write failed: %r", self.rank, e)

    # ------------------------------------------------------------------ I/O
    def _ptrs(self, key: str, host_indices, npages: int) -> List[List[int]]:
        spec = self.specs[key]
        ptr_list, sizes = self.pools[key].get_page_buffer_meta(host_indices)
        nc = len(spec.comp_sizes)
        if len(ptr_list) != npages * nc:
            raise ValueError(
                "pool %s: %d objects for %d pages" % (key, len(ptr_list), npages)
            )
        for q, s in enumerate(sizes):
            if int(s) != spec.comp_sizes[q % nc]:
                raise ValueError(
                    "pool %s: object size %d != %d"
                    % (key, int(s), spec.comp_sizes[q % nc])
                )
        return [[int(ptr_list[i * nc + j]) for j in range(nc)] for i in range(npages)]

    def _io(self, key: str, keys: List[str], host_indices, is_set: bool) -> List[bool]:
        n = len(keys)
        if n == 0 or not self._ensure_store():
            return [False] * n
        pool = self.pools.get(key)
        if pool is None:
            logger.error(
                "EGM rank %d: transfer for unregistered pool %s", self.rank, key
            )
            return [False] * n
        try:
            if len(host_indices) != n * int(pool.page_size):
                raise ValueError(
                    "pool %s: %d indices for %d pages" % (key, len(host_indices), n)
                )
            ptrs = self._ptrs(key, host_indices, n)
        except Exception as e:
            # A geometry mismatch is a bug: switch the tier off.
            self._shut("geometry: %r" % (e,))
            return [False] * n
        if is_set:
            return self.store.put(key, keys, ptrs)
        return self.store.get(key, keys, ptrs)

    # ------------------------------------------------- HiCacheStorage v1 / v2
    def batch_exists(
        self, keys: List[str], extra_info: Optional[HiCacheStorageExtraInfo] = None
    ) -> int:
        if not self._ensure_store():
            return 0
        return self.store.present_prefix(PoolName.KV.value, list(keys))

    def batch_exists_v2(
        self,
        keys: List[str],
        pool_transfers: Optional[List[PoolTransfer]] = None,
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> PoolTransferResult:
        keys = list(keys)
        kv_pages = self.batch_exists(keys, extra_info)
        pools = []
        for t in pool_transfers or []:
            trailing = len(t.keys) if t.keys else 1
            if self.store is not None:
                flags = self.store.present_flags(t.name, keys[:kv_pages])
            else:
                flags = [False] * kv_pages
            policy = PoolHitPolicy(t.hit_policy).value
            pools.append((t.name, policy, trailing, flags))
        final, hit_count, restorable = store.exists_v2_core(kv_pages, pools)
        out = {PoolName.KV: kv_pages} if kv_pages else {}
        out.update(hit_count)
        return PoolTransferResult(final, out, restorable)

    def batch_get_v1(
        self,
        keys: List[str],
        host_indices: torch.Tensor,
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> List[bool]:
        return self._io(PoolName.KV.value, list(keys), host_indices, is_set=False)

    def batch_set_v1(
        self,
        keys: List[str],
        host_indices: torch.Tensor,
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> List[bool]:
        return self._io(PoolName.KV.value, list(keys), host_indices, is_set=True)

    def batch_get_v2(
        self,
        transfers: List[PoolTransfer],
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> Dict[str, List[bool]]:
        return {
            t.name: self._io(t.name, list(t.keys or []), t.host_indices, is_set=False)
            for t in transfers
        }

    def batch_set_v2(
        self,
        transfers: List[PoolTransfer],
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> Dict[str, List[bool]]:
        return {
            t.name: self._io(t.name, list(t.keys or []), t.host_indices, is_set=True)
            for t in transfers
        }

    # Only the zero-copy v1 / v2 interface is served.
    def get(self, key, target_location=None, target_sizes=None):
        raise NotImplementedError("EGM storage serves the v1 / v2 interface only")

    def batch_get(self, keys, target_locations=None, target_sizes=None):
        raise NotImplementedError("EGM storage serves the v1 / v2 interface only")

    def set(self, key, value=None, target_location=None, target_sizes=None):
        raise NotImplementedError("EGM storage serves the v1 / v2 interface only")

    def batch_set(self, keys, values=None, target_locations=None, target_sizes=None):
        raise NotImplementedError("EGM storage serves the v1 / v2 interface only")

    def exists(self, key: str) -> bool:
        return self.batch_exists([key]) == 1

    def clear(self) -> None:
        if self.store is not None and not self.store.clear():
            logger.warning(
                "EGM rank %d: clear skipped (operations in flight)", self.rank
            )

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self.stop_evt.set()
        self._shut("backend closed", wait_s=_CLOSE_WAIT_S)
        if self.pool_exec is not None:
            self.pool_exec.shutdown(wait=False)
