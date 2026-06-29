"""Wrapper around FlexKV ``KVManager`` for sglang.

The public surface is small (see "Public API" below). The class owns:

* the FlexKV ``KVManager`` (server-client mode when ``dp_size > 1`` or
  multi-instance; in-process otherwise — handled by FlexKV itself);
* the per-rank ``KVTPClient`` that registers this rank's GPU KV cache
  with the FlexKV TransferManager;
* an optional ``FlexKVLayerDoneCounter`` plus the UDS-side handshake
  that wires its eventfds into the FlexKV layerwise transfer worker.

Cross-rank sync uses :class:`FlexKVComm`. Only the **sync leader**
(rank 0 of every PP × CP × TP axis) talks to ``KVManager``; other
ranks block on broadcast / barrier.

Modes:
  * **MP / synchronous** (default): ``retrieve_kv`` fires ``launch``
    and blocks on ``wait`` so the device slots are ready by the time
    sglang's prefill runs.
  * **Layerwise** (``FLEXKV_ENABLE_LAYERWISE_TRANSFER=1``): ``launch``
    is fired with ``layerwise_transfer=True`` and the per-layer hook
    registered via ``register_layer_transfer_counter`` blocks each
    forward layer on its own eventfd.
"""

from __future__ import annotations

import logging
import os
import socket
import struct
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from sglang.srt.mem_cache.storage.flexkv.flexkv_comm import (
    CMD_LAYERWISE,
    CMD_PUT_META,
    CMD_STORE_COMPLETE,
    FlexKVComm,
    FlexKVLayerDoneCounter,
    send_fds,
)

try:
    from flexkv.common.request import KVResponseStatus
    from flexkv.common.storage import KVCacheLayout, KVCacheLayoutType
    from flexkv.integration.config import FlexKVConfig
    from flexkv.kvmanager import KVManager
    from flexkv.server.client import KVTPClient
    from flexkv.transfer.layerwise import build_layerwise_eventfd_socket_path
    from flexkv.transfer_manager import TransferManagerOnRemote
except ImportError as exc:  # pragma: no cover - runtime check
    raise RuntimeError(
        "FlexKV is not installed. Please install the FlexKV package to use "
        "--enable-flexkv."
    ) from exc

logger = logging.getLogger(__name__)


class FlexKVConnector:
    """A FlexKV-side façade used by :class:`FlexKVRadixCache`.

    This class manages connection lifecycle and provides a small,
    sgl-friendly contract over FlexKV's task-based API:

      * ``lookup_kv`` — page-aligned hit count + a held task id.
      * ``retrieve_kv`` — synchronous load (launch + wait).
      * ``start_load_kv_layerwise`` — layerwise async load.
      * ``store_kv`` — page-aligned write back.
      * ``check_completed_stores`` — drain async store completions.
      * ``prefetch_async`` / ``check_prefetch_progress`` /
        ``cancel_prefetch`` — opportunistic CPU↔SSD/Remote staging.
      * ``release_pending`` — cancel a held task whose load won't run.
      * ``reset`` / ``shutdown``.
    """

    def __init__(
        self,
        *,
        sgl_model_config: Any,
        server_args: Any,
        page_size: int,
        kvcache: Any,
        tp_rank: int,
        dp_rank: Optional[int],
        pp_rank: int,
        attn_cp_rank: int,
        pp_group: Any = None,
        attn_tp_group: Any = None,
        attn_cp_group: Any = None,
    ) -> None:
        self.page_size = int(page_size)

        # 1. Resolve FlexKV config from env + sglang server args.
        self.flexkv_config = FlexKVConfig.from_env()
        self.rank_info = self.flexkv_config.post_init_from_sglang_config(
            sglang_config=sgl_model_config,
            server_args=server_args,
            page_size=self.page_size,
            tp_rank=tp_rank,
            pp_rank=pp_rank,
            dp_rank=dp_rank if dp_rank is not None else 0,
            attn_cp_rank=attn_cp_rank,
        )
        self.model_config = self.flexkv_config.model_config
        self.cache_config = self.flexkv_config.cache_config
        self._label = (
            f"[model_config={self.model_config}, rank_info={self.rank_info}]"
        )

        # 2. Cross-rank sync context.
        world_rank = (
            torch.distributed.get_rank()
            if torch.distributed.is_initialized()
            else 0
        )
        self._sync_ctx = FlexKVComm(
            rank_info=self.rank_info,
            world_rank=world_rank,
            pp_group=pp_group,
            attn_tp_group=attn_tp_group,
            attn_cp_group=attn_cp_group,
        )

        # 3. Align block counts across all ranks (MIN reduce) so each
        # rank's KVManager registers compatible sizes.
        for attr in ("num_cpu_blocks", "num_ssd_blocks", "num_remote_blocks"):
            orig = getattr(self.cache_config, attr, None)
            if orig is None or orig <= 0:
                continue
            aligned = self._sync_ctx.all_reduce_min(int(orig))
            if aligned != orig:
                logger.info(
                    "[FlexKV] Block count MIN alignment '%s': %d -> %d",
                    attr,
                    orig,
                    aligned,
                )
            setattr(self.cache_config, attr, aligned)

        # 4. Extract MLA/MHA KV buffers + optional indexer buffers.
        indexer_buffers = getattr(kvcache, "index_k_with_scale_buffer", None)
        if hasattr(kvcache, "kv_buffer"):
            # MLA: K and V share the same buffer (per-layer tensor).
            kv_caches = list(kvcache.kv_buffer)
        elif hasattr(kvcache, "k_buffer"):
            # MHA: K buffers concatenated with V buffers, layer-first.
            kv_caches = list(kvcache.k_buffer) + list(kvcache.v_buffer)
        else:
            raise AttributeError(
                f"Unsupported KV cache type {type(kvcache).__name__}: "
                f"expected kv_buffer (MLA/NSA) or k_buffer/v_buffer (MHA)."
            )
        self._kvcache = kvcache

        # 5. On multi-node setups, every node beyond node 0 needs a
        # TransferManagerOnRemote process (FlexKV side) before any rank
        # on that node can register GPU buffers.
        self._remote_process = None
        if (
            self.model_config.nnodes > 1
            and self.rank_info.node_rank > 0
            and self.rank_info.local_rank == 0
        ):
            self._remote_process = TransferManagerOnRemote.create_process(
                master_host=self.model_config.master_host,
                master_ports=self.model_config.master_ports,
            )
            logger.info(
                "[FlexKV] Launched TransferManagerOnRemote on node_rank=%d %s",
                self.rank_info.node_rank,
                self._label,
            )

        # 6. Bring up KVManager on the sync leader only.
        self.kv_manager: Optional[KVManager] = None
        if self._sync_ctx.is_sync_leader:
            self.kv_manager = KVManager(
                model_config=self.model_config,
                cache_config=self.cache_config,
                dp_client_id=self.rank_info.dp_client_id,
                server_recv_port=self.flexkv_config.server_recv_port,
                gpu_register_port=self.flexkv_config.gpu_register_port,
            )
            self.kv_manager.start()

        # 7. Per-rank TP client registers this rank's GPU buffers.
        self.tp_client = KVTPClient(
            self.flexkv_config.gpu_register_port,
            dp_client_id=self.rank_info.dp_client_id,
            pp_rank=self.rank_info.pp_rank,
            device_id=self.rank_info.local_rank,
        )
        self._register_with_retry(kv_caches, indexer_buffers)

        # 8. Layerwise transfer plumbing.
        self.enable_layerwise = bool(
            int(os.environ.get("FLEXKV_ENABLE_LAYERWISE_TRANSFER", "0"))
        )
        self._layerwise_socket = build_layerwise_eventfd_socket_path(
            dp_client_id=self.rank_info.dp_client_id,
            pp_rank=self.rank_info.pp_rank,
            model_config=self.model_config,
        )
        self._layerwise_eventfd_connect_max_retries = max(
            360,
            int(os.environ.get("FLEXKV_LAYERWISE_EVENTFD_CONNECT_MAX_RETRIES", "0")),
        )
        self.layer_done_counter: Optional[FlexKVLayerDoneCounter] = None
        if self.enable_layerwise:
            self.layer_done_counter = FlexKVLayerDoneCounter(
                self.rank_info.num_layers_per_pp_stage
            )
            self._send_eventfds_to_worker()

        # 9. Wait for the KVManager (and its remote subprocess) to be ready.
        if self._sync_ctx.is_sync_leader:
            self._wait_kv_manager_ready()

        # 10. Per-rank in-flight tracking.
        # Loads
        self._pending_lookups: Dict[str, int] = {}  # rid -> fkv_task_id
        self._inflight_loads: Dict[int, int] = {}  # producer_id -> rid hashlike
        self._completed_layerwise: List[int] = []
        self._launched_load_tids: List[int] = []  # leader-only, for periodic drain
        # Stores
        self._inflight_stores: Dict[str, int] = {}  # rid -> fkv_task_id
        # Prefetches
        self._ongoing_prefetches: Dict[str, int] = {}  # rid -> fkv_task_id
        self._prefetch_enabled = bool(
            self.cache_config.enable_ssd
            or self.cache_config.enable_remote
            or self.cache_config.enable_kv_sharing
        )

        logger.info(
            "[FlexKV] Connector ready %s: layerwise=%s, prefetch=%s",
            self._label,
            self.enable_layerwise,
            self._prefetch_enabled,
        )

    # ------------------------------------------------------------------
    # Public API — lookup / load
    # ------------------------------------------------------------------

    def lookup_kv(
        self,
        token_ids: List[int],
        token_mask: torch.Tensor,
        rid: Optional[str] = None,
    ) -> Tuple[int, int]:
        """Page-aligned prefix lookup against FlexKV.

        Args:
          token_ids: full token id sequence we'd like to check.
          token_mask: 1-D bool tensor or array, True for "this token is
            *not* already on GPU and is a candidate for load-back".
          rid: if set and hit > 0, the held FlexKV task id is stashed
            under this key so a later ``retrieve_kv(rid, slots)`` call
            can resolve it. If not set, the held task is cancelled when
            hit > 0 and the caller didn't ask to track it.

        Returns:
          ``(fkv_task_id, hit_count)``. ``hit_count`` is page-aligned
          and may be smaller than the raw FlexKV match if the page
          floor truncated it.
        """
        fkv_task_id = -1
        hit_length = 0

        if self._sync_ctx.is_sync_leader and self.kv_manager is not None:
            tids_np = np.asarray(token_ids, dtype=np.int64)
            mask_np = self._as_numpy_mask(token_mask)
            try:
                res = self.kv_manager.get_match(token_ids=tids_np, token_mask=mask_np)
            except Exception as exc:  # noqa: BLE001
                logger.warning("[FlexKV] get_match raised: %s", exc)
                res = None
            if res is None:
                fkv_task_id = -1
                hit_length = 0
            else:
                fkv_task_id, matched_mask = res
                hit_length = (
                    int(matched_mask.sum()) if matched_mask is not None else 0
                )

        if self._sync_ctx.needs_sync:
            payload = self._sync_ctx.scatter(
                {"task_id": fkv_task_id, "hit": hit_length}
            )
            fkv_task_id = payload["task_id"]
            hit_length = payload["hit"]

        # Page-align: FlexKV transfers whole pages.
        if hit_length > 0 and self.page_size > 1:
            aligned = (hit_length // self.page_size) * self.page_size
            if aligned < hit_length:
                logger.debug(
                    "[FlexKV] lookup_kv: page-aligning hit %d -> %d (page=%d)",
                    hit_length,
                    aligned,
                    self.page_size,
                )
            hit_length = aligned

        # Decide what to do with the held task. Three cases:
        #   1. hit_length > 0 and rid given → stash for retrieve_kv later.
        #   2. hit_length > 0 and rid is None → cancel; caller can't use it.
        #   3. hit_length == 0 → no work to do; FlexKV already marked the
        #      empty graph COMPLETED inside get_match, cancel would warn.
        if hit_length > 0 and rid is not None and fkv_task_id >= 0:
            self._pending_lookups[rid] = fkv_task_id
        elif hit_length > 0 and fkv_task_id >= 0 and self._sync_ctx.is_sync_leader:
            assert self.kv_manager is not None
            self.kv_manager.cancel([fkv_task_id])

        return fkv_task_id, hit_length

    def release_pending(self, rid: str) -> None:
        """Cancel the task held by an earlier ``lookup_kv(rid=...)`` that
        won't be followed by a ``retrieve_kv`` (e.g. allocation failed)."""
        fkv_task_id = self._pending_lookups.pop(rid, -1)
        if fkv_task_id >= 0 and self._sync_ctx.is_sync_leader:
            assert self.kv_manager is not None
            self.kv_manager.cancel([fkv_task_id])

    def retrieve_kv(
        self,
        rid: str,
        slot_mapping: torch.Tensor,
    ) -> int:
        """Synchronous load: ``launch`` + ``wait``.

        Returns the number of slots actually loaded. The caller is
        responsible for having allocated ``slot_mapping`` of length
        equal to ``hit_length`` from a prior ``lookup_kv``.
        """
        fkv_task_id = self._pending_lookups.pop(rid, -1)
        if fkv_task_id < 0:
            return 0

        slot_mapping_cpu = self._to_cpu_int64(slot_mapping)

        # Cross-node PP receivers must send their slot mapping back to
        # the TransferManagerOnRemote so the remote side knows where to
        # land the H2D copies on its own GPUs.
        if self._sync_ctx.should_send_slot_mapping_to_remote:
            self._send_slot_mapping_to_remote(fkv_task_id, slot_mapping_cpu)

        n = slot_mapping_cpu.numel()
        if self._sync_ctx.is_sync_leader and self.kv_manager is not None:
            self.kv_manager.launch(
                task_ids=[fkv_task_id],
                slot_mappings=[slot_mapping_cpu],
                as_batch=True,
                layerwise_transfer=False,
            )
            resp = self.kv_manager.wait([fkv_task_id], timeout=30.0)
            if not (
                fkv_task_id in resp
                and resp[fkv_task_id].status == KVResponseStatus.SUCCESS
            ):
                logger.warning(
                    "[FlexKV] retrieve_kv: task %d failed/timed out",
                    fkv_task_id,
                )
                n = 0
        if self._sync_ctx.needs_sync:
            self._sync_ctx.barrier()
        return n

    def start_load_kv_layerwise(
        self,
        rid: str,
        slot_mapping: torch.Tensor,
    ) -> Tuple[int, int]:
        """Layerwise load. Fires ``launch(layerwise_transfer=True)`` and
        returns ``(n_slots, producer_id)``. The caller registers
        ``producer_id`` with the layer hook so the KV pool blocks on
        the right eventfds during forward."""
        assert self.enable_layerwise and self.layer_done_counter is not None, (
            "start_load_kv_layerwise called but layerwise transfer is "
            "disabled. Set FLEXKV_ENABLE_LAYERWISE_TRANSFER=1."
        )
        fkv_task_id = self._pending_lookups.pop(rid, -1)
        if fkv_task_id < 0:
            return 0, -1

        slot_mapping_cpu = self._to_cpu_int64(slot_mapping)
        n = slot_mapping_cpu.numel()

        if self._sync_ctx.should_send_slot_mapping_to_remote:
            self._send_slot_mapping_to_remote(fkv_task_id, slot_mapping_cpu)

        # Allocate / receive producer slot.
        if self._sync_ctx.is_pp_receiver:
            payload = self._sync_ctx.scatter_pp(None)
            if payload.get("cmd") != CMD_LAYERWISE:
                raise RuntimeError(
                    f"Tag mismatch: expected CMD_LAYERWISE, got "
                    f"{payload.get('cmd')}"
                )
            producer_id = int(payload["counter_id"])
            self.layer_done_counter.register_task_with_explicit_counter_id(
                fkv_task_id, producer_id
            )
        else:
            producer_id = self.layer_done_counter.update_producer()
            self.layer_done_counter.events[producer_id].reset_for_new_transfer()
            self.layer_done_counter.register_task(fkv_task_id, producer_id)

        if self._sync_ctx.is_pp_sender:
            self._sync_ctx.scatter_pp(
                {
                    "cmd": CMD_LAYERWISE,
                    "fkv_task_id": fkv_task_id,
                    "counter_id": producer_id,
                }
            )

        if self._sync_ctx.is_sync_leader and self.kv_manager is not None:
            self.kv_manager.launch(
                task_ids=[fkv_task_id],
                slot_mappings=[slot_mapping_cpu],
                as_batch=True,
                layerwise_transfer=True,
                counter_id=producer_id,
            )
            self._launched_load_tids.append(fkv_task_id)

        # Tell the layer hook which counter slot to wait on.
        self.layer_done_counter.set_consumer(fkv_task_id)
        return n, producer_id

    def drain_launched_loads(self, threshold: int = 100) -> None:
        """Periodic non-blocking sweep on long-lived launched tasks so the
        FlexKV pipe doesn't accumulate. No-op on non-leader ranks."""
        if not self._sync_ctx.is_sync_leader or self.kv_manager is None:
            return
        if len(self._launched_load_tids) < threshold:
            return
        try:
            self.kv_manager.try_wait(task_ids=list(self._launched_load_tids))
        except Exception as exc:  # noqa: BLE001
            logger.debug("[FlexKV] drain_launched_loads try_wait: %s", exc)
        self._launched_load_tids.clear()

    # ------------------------------------------------------------------
    # Public API — store
    # ------------------------------------------------------------------

    def store_kv(
        self,
        rid: str,
        token_ids: List[int],
        kv_indices: torch.Tensor,
    ) -> int:
        """Schedule a write back from GPU into FlexKV.

        On the sync leader this runs ``put_match`` to discover which
        tokens are NOT yet in FlexKV's CPU cache (= the "unmatched"
        slice), then ``launch`` on those. On non-leaders the unmatched
        mask is received over the PP fan-out so cross-node PP can
        forward its slot mappings.

        Returns the FlexKV task id of the in-flight store, or -1 if
        nothing needed to be written.
        """
        token_ids_np = np.asarray(token_ids, dtype=np.int64)
        n = len(token_ids_np)
        if n != len(kv_indices):
            raise ValueError(
                f"store_kv: token_ids has {n} entries but kv_indices "
                f"has {len(kv_indices)} entries"
            )

        # Page-align inputs *before* put_match so the FlexKV allocator
        # only reserves slots that line up with the slot_mapping we send.
        if self.page_size > 1:
            aligned_len = (n // self.page_size) * self.page_size
            if aligned_len == 0:
                self._send_pp_put_meta(-1, [])
                return -1
            if aligned_len < n:
                token_ids_np = token_ids_np[:aligned_len]
                kv_indices = kv_indices[:aligned_len]

        fkv_task_id = -1
        if self._sync_ctx.is_sync_leader and self.kv_manager is not None:
            try:
                res = self.kv_manager.put_match(
                    token_ids=token_ids_np, token_mask=None
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("[FlexKV] put_match raised: %s", exc)
                res = None
            if res is None:
                self._send_pp_put_meta(-1, [])
                return -1
            fkv_task_id, unmatched_mask = res

            self._send_pp_put_meta(fkv_task_id, unmatched_mask)

            if int(unmatched_mask.sum()) > 0:
                filtered = kv_indices[unmatched_mask]
                slot_mapping_cpu = self._to_cpu_int64(filtered)
                self.kv_manager.launch(
                    task_ids=[fkv_task_id],
                    slot_mappings=[slot_mapping_cpu],
                    as_batch=False,
                    layerwise_transfer=False,
                )
                self._inflight_stores[rid] = fkv_task_id
                return fkv_task_id
            return -1

        # Non-leader path: receive the unmatched mask + maybe forward
        # slot_mapping to the remote-side TransferManager.
        if self._sync_ctx.is_pp_receiver:
            payload = self._sync_ctx.scatter_pp(None)
            if payload.get("cmd") != CMD_PUT_META:
                raise RuntimeError(
                    f"Tag mismatch: expected CMD_PUT_META, got "
                    f"{payload.get('cmd')}"
                )
            fkv_task_id = int(payload["fkv_task_id"])
            mask_list = payload.get("unmatched_mask", [])
            unmatched_mask = torch.tensor(mask_list, dtype=torch.bool)
            if (
                int(unmatched_mask.sum()) > 0
                and fkv_task_id >= 0
                and self._sync_ctx.should_send_slot_mapping_to_remote
            ):
                filtered = kv_indices[unmatched_mask]
                slot_mapping_cpu = self._to_cpu_int64(filtered)
                self._send_slot_mapping_to_remote(fkv_task_id, slot_mapping_cpu)
                self._inflight_stores[rid] = fkv_task_id
        return fkv_task_id

    def check_completed_stores(self) -> List[str]:
        """Return rids whose stores have completed since the last call."""
        completed_rids: List[str] = []
        completed_dict: Dict[int, Any] = {}

        if self._sync_ctx.is_sync_leader and self.kv_manager is not None:
            if self._inflight_stores:
                fk_to_rid = {v: k for k, v in self._inflight_stores.items()}
                try:
                    completed_dict = self.kv_manager.try_wait(
                        task_ids=list(fk_to_rid.keys())
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.debug("[FlexKV] check_completed_stores: %s", exc)
                    completed_dict = {}
                for fk_tid in completed_dict:
                    rid = fk_to_rid[fk_tid]
                    completed_rids.append(rid)
                    self._inflight_stores.pop(rid, None)

        if self._sync_ctx.is_pp_sender:
            self._sync_ctx.scatter_pp(
                {
                    "cmd": CMD_STORE_COMPLETE,
                    "completed_fk_ids": list(completed_dict),
                }
            )
        elif self._sync_ctx.is_pp_receiver:
            payload = self._sync_ctx.scatter_pp(None)
            if payload.get("cmd") != CMD_STORE_COMPLETE:
                raise RuntimeError(
                    f"Tag mismatch: expected CMD_STORE_COMPLETE, got "
                    f"{payload.get('cmd')}"
                )
            fk_ids = payload.get("completed_fk_ids", [])
            if fk_ids and self._inflight_stores:
                fk_to_rid = {v: k for k, v in self._inflight_stores.items()}
                for fk_tid in fk_ids:
                    if fk_tid in fk_to_rid:
                        rid = fk_to_rid[fk_tid]
                        completed_rids.append(rid)
                        self._inflight_stores.pop(rid, None)

        if self._sync_ctx.needs_sync:
            completed_rids = self._sync_ctx.scatter(completed_rids)
        return completed_rids

    def wait_store(self, rid: str, timeout: float = 30.0) -> bool:
        """Block until a single store task identified by ``rid`` finishes."""
        fkv_task_id = self._inflight_stores.pop(rid, -1)
        if fkv_task_id < 0:
            return True
        if not self._sync_ctx.is_sync_leader or self.kv_manager is None:
            return True
        try:
            resp = self.kv_manager.wait([fkv_task_id], timeout=timeout)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[FlexKV] wait_store: %s", exc)
            return False
        return (
            fkv_task_id in resp
            and resp[fkv_task_id].status == KVResponseStatus.SUCCESS
        )

    # ------------------------------------------------------------------
    # Public API — prefetch
    # ------------------------------------------------------------------

    def prefetch_async(self, rid: str, token_ids: List[int]) -> int:
        if not self._prefetch_enabled or not rid:
            return -1
        task_id = -1
        if self._sync_ctx.is_sync_leader and self.kv_manager is not None:
            try:
                task_id = self.kv_manager.prefetch_async(
                    token_ids=np.asarray(token_ids, dtype=np.int64)
                )
            except Exception as exc:  # noqa: BLE001
                logger.debug("[FlexKV] prefetch_async: %s", exc)
                task_id = -1
        if self._sync_ctx.needs_sync:
            payload = self._sync_ctx.scatter({"task_id": task_id})
            task_id = payload["task_id"]
        if task_id >= 0:
            self._ongoing_prefetches[rid] = task_id
        return task_id

    def check_prefetch_progress(self, rid: str) -> bool:
        if not self._prefetch_enabled:
            return True
        task_id = self._ongoing_prefetches.get(rid, -1)
        if task_id < 0:
            return True
        done = False
        if self._sync_ctx.is_sync_leader and self.kv_manager is not None:
            try:
                completed = self.kv_manager.try_wait(task_ids=[task_id])
            except Exception:  # noqa: BLE001
                completed = {}
            if task_id in completed:
                done = True
        if self._sync_ctx.needs_sync:
            payload = self._sync_ctx.scatter({"done": done})
            done = payload["done"]
        if done:
            self._ongoing_prefetches.pop(rid, None)
        return done

    def cancel_prefetch(self, rid: str) -> None:
        self._pending_lookups.pop(rid, None)
        # FlexKV doesn't currently support prefetch cancellation, but
        # we still drop our tracking entry.
        self._ongoing_prefetches.pop(rid, None)

    # ------------------------------------------------------------------
    # Layerwise transfer hooks
    # ------------------------------------------------------------------

    def register_layer_transfer_counter(self, kvcache: Any) -> None:
        """Register the FlexKVLayerDoneCounter onto sglang's KV pool so
        each forward layer blocks on its eventfd. No-op when layerwise
        is disabled."""
        if (
            self.layer_done_counter is None
            or kvcache is None
            or not hasattr(kvcache, "register_layer_transfer_counter")
        ):
            return
        kvcache.register_layer_transfer_counter(self.layer_done_counter)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self) -> None:
        # Drop pending lookups (cancel their held tasks on the leader).
        if self._sync_ctx.is_sync_leader and self.kv_manager is not None:
            pending = [tid for tid in self._pending_lookups.values() if tid >= 0]
            if pending:
                try:
                    self.kv_manager.cancel(pending)
                except Exception as exc:  # noqa: BLE001
                    logger.debug("[FlexKV] reset cancel: %s", exc)
        self._pending_lookups.clear()
        self._ongoing_prefetches.clear()
        self._inflight_loads.clear()
        self._completed_layerwise.clear()
        self._launched_load_tids.clear()
        if self._sync_ctx.is_sync_leader and self.kv_manager is not None:
            for fk_tid in list(self._inflight_stores.values()):
                if fk_tid >= 0:
                    try:
                        self.kv_manager.wait([fk_tid], timeout=20.0)
                    except Exception:  # noqa: BLE001
                        pass
        self._inflight_stores.clear()
        if self.layer_done_counter is not None:
            self.layer_done_counter.reset()

    def shutdown(self) -> None:
        if self._sync_ctx.is_sync_leader and self.kv_manager is not None:
            try:
                self.kv_manager.shutdown()
            except Exception as exc:  # noqa: BLE001
                logger.warning("[FlexKV] kv_manager.shutdown: %s", exc)
        if self._remote_process is not None:
            try:
                self._remote_process.terminate()
                self._remote_process.join(timeout=5.0)
                if self._remote_process.is_alive():
                    self._remote_process.kill()
                    self._remote_process.join()
            except Exception as exc:  # noqa: BLE001
                logger.warning("[FlexKV] remote process shutdown: %s", exc)
            self._remote_process = None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _as_numpy_mask(mask) -> np.ndarray:
        if mask is None:
            return None
        if isinstance(mask, torch.Tensor):
            return mask.detach().cpu().numpy()
        return np.asarray(mask)

    @staticmethod
    def _to_cpu_int64(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.is_cuda:
            tensor = tensor.cpu()
        return tensor.to(torch.int64)

    def _wait_kv_manager_ready(self, poll_interval: float = 10.0) -> None:
        assert self.kv_manager is not None
        wait_count = 0
        while not self.kv_manager.is_ready():
            time.sleep(poll_interval)
            wait_count += 1
            logger.info(
                "[FlexKV] Waiting for FlexKV ready %s (waited %.0fs)",
                self._label,
                wait_count * poll_interval,
            )
        logger.info("[FlexKV] FlexKV is ready %s", self._label)

    def _register_with_retry(
        self,
        kv_caches: List[torch.Tensor],
        indexer_buffers: Optional[List[torch.Tensor]] = None,
        max_retries: int = 360,
    ) -> None:
        """Retry GPU registration. On node_rank>0, the
        TransferManagerOnRemote may not be ready immediately; retry up
        to ~6 minutes."""
        for attempt in range(max_retries):
            try:
                self._register_to_server(kv_caches, indexer_buffers)
                return
            except Exception as exc:  # noqa: BLE001
                if attempt == max_retries - 1:
                    raise
                if attempt % 30 == 0:
                    logger.info(
                        "[FlexKV] GPU register retry %s attempt=%d/%d "
                        "error=%s",
                        self._label,
                        attempt + 1,
                        max_retries,
                        exc,
                    )
                time.sleep(1.0)

    def _register_to_server(
        self,
        kv_caches: List[torch.Tensor],
        indexer_buffers: Optional[List[torch.Tensor]] = None,
    ) -> None:
        assert len(kv_caches) > 0
        assert kv_caches[0].ndim == 3, (
            f"Expected 3D KV cache tensor, got shape={kv_caches[0].shape}"
        )

        is_mla = self.model_config.use_mla
        num_blocks, num_kv_heads, head_size = kv_caches[0].shape

        gpu_layout = KVCacheLayout(
            type=KVCacheLayoutType.LAYERFIRST,
            num_layer=self.rank_info.num_layers_per_pp_stage,
            num_block=num_blocks // self.page_size,
            tokens_per_block=self.page_size,
            num_head=num_kv_heads,
            head_size=head_size,
            is_mla=is_mla,
        )

        indexer_layout = None
        if indexer_buffers is not None and len(indexer_buffers) > 0:
            indexer_tensor = indexer_buffers[0]
            assert indexer_tensor.ndim == 2, (
                f"Expected 2D indexer tensor (num_pages, page_stride_size), "
                f"got shape={indexer_tensor.shape}"
            )
            indexer_layout = KVCacheLayout(
                type=KVCacheLayoutType.LAYERFIRST,
                num_layer=len(indexer_buffers),
                num_block=indexer_tensor.shape[0],
                tokens_per_block=1,
                num_head=1,
                head_size=indexer_tensor.shape[1],
                is_mla=True,
            )

        self.tp_client.register_to_server(
            kv_caches=kv_caches,
            kv_layout=gpu_layout,
            indexer_buffers=indexer_buffers,
            indexer_layout=indexer_layout,
        )
        logger.info("[FlexKV] Registered KV caches to server %s", self._label)

    def _send_pp_put_meta(self, fkv_task_id: int, unmatched_mask) -> None:
        if not self._sync_ctx.is_pp_active:
            return
        if hasattr(unmatched_mask, "tolist"):
            mask_list = unmatched_mask.tolist()
        else:
            mask_list = list(unmatched_mask)
        self._sync_ctx.scatter_pp(
            {
                "cmd": CMD_PUT_META,
                "fkv_task_id": fkv_task_id,
                "unmatched_mask": mask_list,
            }
        )

    def _send_slot_mapping_to_remote(
        self, task_id: int, slot_mapping_cpu: torch.Tensor
    ) -> None:
        np_arr = slot_mapping_cpu.numpy()
        self.tp_client.set_slot_mapping(task_id, np_arr)

    def _send_eventfds_to_worker(self, retry_interval: float = 1.0) -> None:
        """UDS handshake with the FlexKV layerwise transfer worker.

        Sends per-counter eventfd FDs over a unix domain socket using
        ``SCM_RIGHTS``. Retries connect (worker may not yet be up) and
        retries the whole connect+send sequence on send error.
        """
        max_retries = self._layerwise_eventfd_connect_max_retries
        max_send_retries = 3
        last_error: Optional[BaseException] = None

        assert self.layer_done_counter is not None

        for send_attempt in range(max_send_retries):
            sock: Optional[socket.socket] = None
            try:
                # Phase 1: connect (worker may not yet be up).
                for attempt in range(max_retries):
                    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                    try:
                        sock.connect(self._layerwise_socket)
                        logger.info(
                            "[FlexKV] Eventfd connected %s socket=%s attempts=%d",
                            self._label,
                            self._layerwise_socket,
                            attempt + 1,
                        )
                        break
                    except (FileNotFoundError, ConnectionRefusedError) as exc:
                        sock.close()
                        sock = None
                        if attempt == max_retries - 1:
                            raise RuntimeError(
                                f"[FlexKV] Failed to connect to eventfd socket "
                                f"{self._layerwise_socket} after {max_retries} attempts"
                            ) from exc
                        time.sleep(retry_interval)
                assert sock is not None

                # Phase 2: send 16-byte metadata + per-counter FDs + read ACK.
                num_counters = self.layer_done_counter.num_counters
                metadata = struct.pack(
                    "iiii",
                    self.rank_info.tp_rank_per_node,
                    self.model_config.tp_size_per_node,
                    self.rank_info.num_layers_per_pp_stage,
                    num_counters,
                )
                sock.sendall(metadata)
                for counter_id in range(num_counters):
                    fds = self.layer_done_counter.events[counter_id].load_event_fds
                    send_fds(sock, fds, struct.pack("i", counter_id))
                sock.settimeout(30.0)
                try:
                    ack = sock.recv(1)
                except socket.timeout as exc:
                    raise RuntimeError(
                        "Timed out waiting for ACK from FlexKV layerwise worker"
                    ) from exc
                if not ack or ack[0] != 1:
                    raise RuntimeError(
                        f"FlexKV layerwise worker NACK'd eventfd transfer "
                        f"(ack={ack!r})"
                    )
                logger.info(
                    "[FlexKV] Eventfd handshake complete %s counters=%d layers=%d",
                    self._label,
                    num_counters,
                    self.rank_info.num_layers_per_pp_stage,
                )
                return
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                logger.warning(
                    "[FlexKV] Eventfd handshake send_attempt=%d/%d failed: %s",
                    send_attempt + 1,
                    max_send_retries,
                    exc,
                )
            finally:
                if sock is not None:
                    sock.close()
                time.sleep(retry_interval)

        raise RuntimeError(
            f"[FlexKV] Failed to send eventfds to {self._layerwise_socket} "
            f"after {max_send_retries} attempts: {last_error}"
        )
