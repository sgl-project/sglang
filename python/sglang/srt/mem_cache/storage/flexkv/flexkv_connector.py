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
from dataclasses import dataclass
from enum import Enum
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


@dataclass(frozen=True)
class _PendingFlexKVLookup:
    task_id: int
    expected_slots: int


class FlexKVRetrieveStatus(Enum):
    SUCCESS = "success"
    DEFINITE_TERMINAL_FAILURE = "definite_terminal_failure"
    AMBIGUOUS = "ambiguous"


@dataclass(frozen=True)
class FlexKVRetrieveResult:
    status: FlexKVRetrieveStatus
    num_slots: int
    observation_task_ids: Tuple[int, ...] = ()
    reason: Optional[str] = None


@dataclass(frozen=True)
class _PreparedFlexKVLoad:
    pending: Optional[_PendingFlexKVLookup]
    slot_mapping: Optional[torch.Tensor]
    producer_id: Optional[int]
    failure: Optional[FlexKVRetrieveResult]


@dataclass(frozen=True)
class _LayerwiseProducerSelection:
    producer_id: int
    reason: Optional[str]


class FlexKVAmbiguousLoadError(RuntimeError):
    pass


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
        allocator_page_size: int,
        kvcache: Any,
        tp_rank: int,
        dp_rank: Optional[int],
        pp_rank: int,
        attn_cp_rank: int,
        pp_group: Any = None,
        attn_tp_group: Any = None,
        attn_cp_group: Any = None,
    ) -> None:
        self.storage_page_size = int(page_size)
        self.allocator_page_size = int(allocator_page_size)
        self.enable_layerwise = bool(
            int(os.environ.get("FLEXKV_ENABLE_LAYERWISE_TRANSFER", "0"))
        )
        if self.enable_layerwise and self.allocator_page_size > 1:
            raise ValueError("FlexKV layerwise transfer requires allocator page size 1")
        if self.allocator_page_size % self.storage_page_size != 0:
            raise ValueError(
                "FlexKV requires the storage page size to divide the allocator "
                "page size"
            )

        # 1. Resolve FlexKV config from env + sglang server args.
        self.flexkv_config = FlexKVConfig.from_env()
        self.rank_info = self.flexkv_config.post_init_from_sglang_config(
            sglang_config=sgl_model_config,
            server_args=server_args,
            page_size=self.storage_page_size,
            tp_rank=tp_rank,
            pp_rank=pp_rank,
            dp_rank=dp_rank if dp_rank is not None else 0,
            attn_cp_rank=attn_cp_rank,
        )
        self.model_config = self.flexkv_config.model_config
        self.cache_config = self.flexkv_config.cache_config
        self._label = f"[model_config={self.model_config}, rank_info={self.rank_info}]"

        # 2. Cross-rank sync context.
        world_rank = (
            torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
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
        self._pending_lookups: Dict[str, _PendingFlexKVLookup] = {}
        self._ambiguous_loads: Dict[str, Tuple[int, ...]] = {}
        self._poison_reason: Optional[str] = None
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
          ``(fkv_task_id, hit_count)``. ``hit_count`` is the exact
          allocator-page-aligned match represented by the held task.
        """
        if self._poison_reason is not None:
            raise RuntimeError(f"FlexKV load-back is poisoned: {self._poison_reason}")

        payload = {"task_id": -1, "hit": 0, "error": None}
        if self._sync_ctx.is_sync_leader and self.kv_manager is not None:
            try:
                tids_np = np.asarray(token_ids, dtype=np.int64)
                mask_np = np.asarray(self._as_numpy_mask(token_mask), dtype=np.bool_)
                res = self.kv_manager.get_match(token_ids=tids_np, token_mask=mask_np)
                if res is None:
                    raise RuntimeError("FlexKV get_match returned no result")
                task_id, matched_mask = res
                error = self._validate_lookup_match(
                    token_ids=tids_np,
                    candidate_mask=mask_np,
                    task_id=task_id,
                    matched_mask=matched_mask,
                )
                hit_length = (
                    int(np.asarray(matched_mask, dtype=np.bool_).sum())
                    if error is None
                    else 0
                )
                payload = {
                    "task_id": self._normalize_task_id(task_id),
                    "hit": hit_length,
                    "error": error,
                }
            except Exception as exc:  # noqa: BLE001
                payload["error"] = f"FlexKV get_match failed: {exc}"

        if self._sync_ctx.needs_sync:
            payload = self._sync_ctx.scatter(payload)

        if payload["error"] is not None:
            self._poison_reason = str(payload["error"])
            raise RuntimeError(f"Invalid FlexKV lookup result: {self._poison_reason}")

        fkv_task_id = int(payload["task_id"])
        hit_length = int(payload["hit"])

        # Decide what to do with the held task. Three cases:
        #   1. hit_length > 0 and rid given → stash for retrieve_kv later.
        #   2. hit_length > 0 and rid is None → cancel; caller can't use it.
        #   3. hit_length == 0 → no work to do; FlexKV already marked the
        #      empty graph COMPLETED inside get_match, cancel would warn.
        if hit_length > 0 and rid is not None and fkv_task_id >= 0:
            self._pending_lookups[rid] = _PendingFlexKVLookup(
                task_id=fkv_task_id,
                expected_slots=hit_length,
            )
        elif hit_length > 0 and fkv_task_id >= 0 and self._sync_ctx.is_sync_leader:
            assert self.kv_manager is not None
            try:
                self.kv_manager.cancel([fkv_task_id])
            except Exception as exc:  # noqa: BLE001
                logger.warning("[FlexKV] untracked lookup cancel: %s", exc)

        return fkv_task_id, hit_length

    def release_pending(self, rid: str) -> None:
        """Cancel the task held by an earlier ``lookup_kv(rid=...)`` that
        won't be followed by a ``retrieve_kv`` (e.g. allocation failed)."""
        pending = self._pending_lookups.pop(rid, None)
        if pending is not None and self._sync_ctx.is_sync_leader:
            assert self.kv_manager is not None
            try:
                self.kv_manager.cancel([pending.task_id])
            except Exception as exc:  # noqa: BLE001
                logger.warning("[FlexKV] release_pending: %s", exc)

    def retrieve_kv(
        self,
        rid: str,
        slot_mapping: Optional[torch.Tensor],
    ) -> FlexKVRetrieveResult:
        """Synchronous load: ``launch`` + ``wait``.

        All ranks make one combined pre-launch decision. Once launch has
        been attempted, every non-success outcome is ambiguous.
        """
        prepared = self._prepare_load(
            rid=rid,
            slot_mapping=slot_mapping,
            layerwise=False,
        )
        if prepared.failure is not None:
            return prepared.failure
        assert prepared.pending is not None and prepared.slot_mapping is not None
        pending = prepared.pending
        slot_mapping_cpu = prepared.slot_mapping
        outcome = {
            "status": FlexKVRetrieveStatus.AMBIGUOUS.value,
            "observation_task_ids": [],
            "reason": None,
        }
        if self._sync_ctx.is_sync_leader:
            assert self.kv_manager is not None
            try:
                observation_task_ids = self._normalize_task_ids(
                    self.kv_manager.launch(
                        task_ids=[pending.task_id],
                        slot_mappings=[slot_mapping_cpu],
                        as_batch=False,
                        layerwise_transfer=False,
                    )
                )
                outcome["observation_task_ids"] = observation_task_ids
                responses = self.kv_manager.wait(
                    observation_task_ids,
                    timeout=30.0,
                    completely=True,
                )
                if responses is None or set(responses) != set(observation_task_ids):
                    raise RuntimeError("KVManager.wait returned unexpected task ids")
                if any(
                    responses[task_id].status != KVResponseStatus.SUCCESS
                    for task_id in observation_task_ids
                ):
                    raise RuntimeError("FlexKV load did not complete successfully")
                outcome = {
                    "status": FlexKVRetrieveStatus.SUCCESS.value,
                    "observation_task_ids": outcome["observation_task_ids"],
                    "reason": None,
                }
            except Exception as exc:  # noqa: BLE001
                outcome["reason"] = str(exc)
        if self._sync_ctx.needs_sync:
            outcome = self._sync_ctx.scatter(outcome)

        self._pending_lookups.pop(rid, None)
        status = FlexKVRetrieveStatus(outcome["status"])
        observation_task_ids = tuple(outcome["observation_task_ids"])
        reason = outcome["reason"]
        if status is FlexKVRetrieveStatus.AMBIGUOUS:
            self._ambiguous_loads[rid] = observation_task_ids
            self._poison_reason = reason or "FlexKV launch outcome is ambiguous"

        return FlexKVRetrieveResult(
            status=status,
            num_slots=pending.expected_slots,
            observation_task_ids=observation_task_ids,
            reason=reason,
        )

    def start_load_kv_layerwise(
        self,
        rid: str,
        slot_mapping: Optional[torch.Tensor],
    ) -> Tuple[int, int]:
        """Layerwise load. Fires ``launch(layerwise_transfer=True)`` and
        returns ``(n_slots, producer_id)``. The caller registers
        ``producer_id`` with the layer hook so the KV pool blocks on
        the right eventfds during forward."""
        assert self.enable_layerwise and self.layer_done_counter is not None, (
            "start_load_kv_layerwise called but layerwise transfer is "
            "disabled. Set FLEXKV_ENABLE_LAYERWISE_TRANSFER=1."
        )
        prepared = self._prepare_load(
            rid=rid,
            slot_mapping=slot_mapping,
            layerwise=True,
        )
        if prepared.failure is not None:
            return 0, -1
        assert (
            prepared.pending is not None
            and prepared.slot_mapping is not None
            and prepared.producer_id is not None
        )
        pending = prepared.pending
        fkv_task_id = pending.task_id
        slot_mapping_cpu = prepared.slot_mapping
        producer_id = prepared.producer_id
        n = slot_mapping_cpu.numel()

        outcome = {"observation_task_ids": [], "reason": None}
        if self._sync_ctx.is_sync_leader:
            assert self.kv_manager is not None
            try:
                observation_task_ids = self._normalize_task_ids(
                    self.kv_manager.launch(
                        task_ids=[fkv_task_id],
                        slot_mappings=[slot_mapping_cpu],
                        as_batch=True,
                        layerwise_transfer=True,
                        counter_id=producer_id,
                    )
                )
                outcome["observation_task_ids"] = observation_task_ids
            except Exception as exc:  # noqa: BLE001
                outcome["reason"] = str(exc)
        if self._sync_ctx.needs_sync:
            outcome = self._sync_ctx.scatter(outcome)

        self._pending_lookups.pop(rid, None)
        observation_task_ids = tuple(outcome["observation_task_ids"])
        if outcome["reason"] is not None:
            self._ambiguous_loads[rid] = observation_task_ids
            self._poison_reason = str(outcome["reason"])
            raise FlexKVAmbiguousLoadError(self._poison_reason)
        if self._sync_ctx.is_sync_leader:
            self._launched_load_tids.extend(observation_task_ids)

        return n, producer_id

    def drain_launched_loads(self, threshold: int = 100) -> None:
        """Release only layerwise tasks proven fully terminal and successful."""
        if not self._sync_ctx.is_sync_leader or self.kv_manager is None:
            return
        if len(self._launched_load_tids) < threshold:
            return
        tracked_ids = set(self._launched_load_tids)
        try:
            responses = self.kv_manager.wait(
                task_ids=list(self._launched_load_tids),
                timeout=0,
                completely=True,
            )
        except TimeoutError:
            return
        except Exception as exc:  # noqa: BLE001
            reason = f"FlexKV layerwise terminal wait failed: {exc}"
            self._poison_reason = reason
            for task_id in tracked_ids:
                self._ambiguous_loads[f"layerwise:{task_id}"] = (task_id,)
            return

        normalized_responses: Dict[int, Any] = {}
        try:
            if responses is None:
                raise ValueError("FlexKV layerwise terminal wait returned no result")
            for task_id, response in responses.items():
                normalized_task_id = self._normalize_task_id(task_id)
                response_task_id = self._normalize_task_id(response.task_id)
                if response_task_id != normalized_task_id:
                    raise ValueError(
                        "FlexKV layerwise terminal response changed task identity"
                    )
                normalized_responses[normalized_task_id] = response
            if set(normalized_responses) != tracked_ids:
                raise ValueError(
                    "FlexKV layerwise terminal wait returned unexpected task ids"
                )
        except Exception as exc:  # noqa: BLE001
            self._poison_reason = str(exc)
            for task_id in tracked_ids:
                self._ambiguous_loads[f"layerwise:{task_id}"] = (task_id,)
            return

        completed_ids: set[int] = set()
        for task_id, response in normalized_responses.items():
            try:
                status = response.status
            except AttributeError:
                status = None
            if status is KVResponseStatus.SUCCESS:
                completed_ids.add(task_id)
            elif status is not KVResponseStatus.TIMEOUT:
                self._poison_reason = (
                    f"FlexKV layerwise task {task_id} completed with status {status}"
                )
                self._ambiguous_loads[f"layerwise:{task_id}"] = (task_id,)
        self._launched_load_tids = [
            task_id
            for task_id in self._launched_load_tids
            if task_id not in completed_ids
        ]

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
        if self.storage_page_size > 1:
            aligned_len = n // self.storage_page_size * self.storage_page_size
            if aligned_len == 0:
                self._send_pp_put_meta(-1, [])
                return -1
            if aligned_len < n:
                token_ids_np = token_ids_np[:aligned_len]
                kv_indices = kv_indices[:aligned_len]

        fkv_task_id = -1
        if self._sync_ctx.is_sync_leader and self.kv_manager is not None:
            try:
                res = self.kv_manager.put_match(token_ids=token_ids_np, token_mask=None)
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
                    f"Tag mismatch: expected CMD_PUT_META, got " f"{payload.get('cmd')}"
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
            fkv_task_id in resp and resp[fkv_task_id].status == KVResponseStatus.SUCCESS
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

    def ensure_reset_safe(self) -> None:
        if self._poison_reason is not None or self._ambiguous_loads:
            raise RuntimeError(
                "Cannot reset FlexKV after an ambiguous load: " f"{self._poison_reason}"
            )

    def reset(self) -> None:
        self.ensure_reset_safe()
        # Drop pending lookups (cancel their held tasks on the leader).
        if self._sync_ctx.is_sync_leader and self.kv_manager is not None:
            pending = [lookup.task_id for lookup in self._pending_lookups.values()]
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

    def _prepare_load(
        self,
        *,
        rid: str,
        slot_mapping: Optional[torch.Tensor],
        layerwise: bool,
    ) -> _PreparedFlexKVLoad:
        pending = self._pending_lookups.get(rid)
        local_status = 1
        local_reason: Optional[str] = None
        slot_mapping_cpu: Optional[torch.Tensor] = None
        page_starts: List[int] = []
        producer_id = -1

        if self._poison_reason is not None:
            local_status = 0
            local_reason = self._poison_reason
        elif pending is None:
            local_status = 0
            local_reason = "missing held lookup"
        elif slot_mapping is None:
            local_status = 0
            local_reason = "allocator could not acquire the exact slot mapping"
        else:
            try:
                slot_mapping_cpu = self._to_cpu_int64(slot_mapping)
                local_reason = self._validate_slot_mapping(
                    slot_mapping=slot_mapping_cpu,
                    expected_slots=pending.expected_slots,
                )
                if local_reason is None:
                    page_starts = slot_mapping_cpu[:: self.allocator_page_size].tolist()
                else:
                    local_status = 0
            except Exception as exc:  # noqa: BLE001
                local_status = 0
                local_reason = str(exc)

        if (
            local_status == 1
            and self._sync_ctx.should_send_slot_mapping_to_remote
            and pending is not None
            and slot_mapping_cpu is not None
        ):
            try:
                self._send_slot_mapping_to_remote(pending.task_id, slot_mapping_cpu)
            except Exception as exc:  # noqa: BLE001
                local_status = 0
                local_reason = str(exc)

        if layerwise:
            producer_selection = self._select_layerwise_producer(pending=pending)
            producer_id = producer_selection.producer_id
            if producer_selection.reason is not None:
                local_status = 0
                local_reason = producer_selection.reason

        local_manifest = {
            "task_id": pending.task_id if pending is not None else -1,
            "expected_slots": pending.expected_slots if pending is not None else -1,
            "page_starts": page_starts,
            "producer_id": producer_id,
        }
        leader_manifest = local_manifest
        if self._sync_ctx.needs_sync:
            leader_manifest = self._sync_ctx.scatter(local_manifest)
        if local_status == 1 and local_manifest != leader_manifest:
            local_status = 0
            local_reason = "lookup or slot manifest differs across ranks"

        if local_status == 1 and layerwise and pending is not None:
            try:
                self._register_layerwise_counter(
                    pending=pending,
                    producer_id=producer_id,
                )
            except Exception as exc:  # noqa: BLE001
                local_status = 0
                local_reason = str(exc)

        combined_status = self._sync_ctx.all_reduce_min(local_status)
        if combined_status == 0:
            if pending is not None and self._sync_ctx.is_sync_leader:
                assert self.kv_manager is not None
                try:
                    self.kv_manager.cancel([pending.task_id])
                except Exception as exc:  # noqa: BLE001
                    logger.warning("[FlexKV] pre-launch cancel: %s", exc)
            self._pending_lookups.pop(rid, None)
            return _PreparedFlexKVLoad(
                pending=pending,
                slot_mapping=slot_mapping_cpu,
                producer_id=producer_id if layerwise else None,
                failure=FlexKVRetrieveResult(
                    status=FlexKVRetrieveStatus.DEFINITE_TERMINAL_FAILURE,
                    num_slots=0,
                    reason=local_reason
                    or "another rank rejected the pre-launch manifest",
                ),
            )

        assert pending is not None and slot_mapping_cpu is not None
        return _PreparedFlexKVLoad(
            pending=pending,
            slot_mapping=slot_mapping_cpu,
            producer_id=producer_id if layerwise else None,
            failure=None,
        )

    def _select_layerwise_producer(
        self, *, pending: Optional[_PendingFlexKVLookup]
    ) -> _LayerwiseProducerSelection:
        counter = self.layer_done_counter
        task_id = pending.task_id if pending is not None else -1
        producer_id = -1
        reason: Optional[str] = None

        if self._sync_ctx.is_pp_receiver:
            try:
                payload = self._sync_ctx.scatter_pp(None)
                if not isinstance(payload, dict):
                    raise ValueError("FlexKV layerwise counter payload is invalid")
                if payload.get("error") is not None:
                    raise RuntimeError(str(payload["error"]))
                if payload.get("cmd") != CMD_LAYERWISE:
                    raise ValueError(
                        "FlexKV layerwise counter payload has the wrong tag"
                    )
                received_task_id = self._normalize_task_id(payload.get("fkv_task_id"))
                if received_task_id != task_id:
                    raise ValueError(
                        "FlexKV layerwise counter payload changed task identity"
                    )
                if counter is None:
                    raise RuntimeError("FlexKV layerwise counter is not initialized")
                producer_id = self._normalize_counter_id(
                    payload.get("counter_id"),
                    num_counters=counter.num_counters,
                )
            except Exception as exc:  # noqa: BLE001
                reason = str(exc)
            return _LayerwiseProducerSelection(
                producer_id=producer_id,
                reason=reason,
            )

        try:
            if pending is None:
                raise RuntimeError("missing held lookup")
            if counter is None:
                raise RuntimeError("FlexKV layerwise counter is not initialized")
            producer_id = self._normalize_counter_id(
                counter.update_producer(),
                num_counters=counter.num_counters,
            )
        except Exception as exc:  # noqa: BLE001
            reason = str(exc)

        if self._sync_ctx.is_pp_sender:
            payload = {
                "cmd": CMD_LAYERWISE,
                "fkv_task_id": task_id,
                "counter_id": producer_id,
                "error": reason,
            }
            try:
                self._sync_ctx.scatter_pp(payload)
            except Exception as exc:  # noqa: BLE001
                reason = str(exc)

        return _LayerwiseProducerSelection(
            producer_id=producer_id,
            reason=reason,
        )

    def _register_layerwise_counter(
        self, *, pending: _PendingFlexKVLookup, producer_id: int
    ) -> None:
        counter = self.layer_done_counter
        if counter is None:
            raise RuntimeError("FlexKV layerwise counter is not initialized")
        normalized_producer_id = self._normalize_counter_id(
            producer_id,
            num_counters=counter.num_counters,
        )
        if self._sync_ctx.is_pp_receiver:
            counter.register_task_with_explicit_counter_id(
                task_id=pending.task_id,
                counter_id=normalized_producer_id,
            )
        else:
            counter.events[normalized_producer_id].reset_for_new_transfer()
            counter.register_task(
                task_id=pending.task_id,
                producer_id=normalized_producer_id,
            )
        counter.set_consumer(pending.task_id)

    def _validate_lookup_match(
        self,
        *,
        token_ids: np.ndarray,
        candidate_mask: np.ndarray,
        task_id: Any,
        matched_mask: Any,
    ) -> Optional[str]:
        try:
            self._normalize_task_id(task_id)
        except ValueError as exc:
            return str(exc)
        if token_ids.ndim != 1 or candidate_mask.shape != token_ids.shape:
            return "FlexKV lookup inputs must be matching one-dimensional arrays"
        matched = np.asarray(matched_mask, dtype=np.bool_)
        if matched.shape != candidate_mask.shape:
            return "FlexKV get_match returned a mask with the wrong shape"
        candidate_positions = np.flatnonzero(candidate_mask)
        if candidate_positions.size > 0 and not np.array_equal(
            candidate_positions,
            np.arange(candidate_positions[0], token_ids.size),
        ):
            return "FlexKV lookup candidates must form a contiguous suffix"
        if np.any(matched & ~candidate_mask):
            return "FlexKV matched tokens outside the candidate suffix"
        matched_candidates = matched[candidate_positions]
        false_positions = np.flatnonzero(~matched_candidates)
        if false_positions.size > 0 and np.any(
            matched_candidates[false_positions[0] :]
        ):
            return "FlexKV matched mask must be a contiguous candidate prefix"
        hit_length = int(matched.sum())
        if hit_length % self.allocator_page_size != 0:
            return "FlexKV matched mask is not allocator-page-aligned"
        return None

    @staticmethod
    def _normalize_task_id(task_id: Any) -> int:
        if isinstance(task_id, bool) or not isinstance(task_id, (int, np.integer)):
            raise ValueError("FlexKV returned an invalid task id")
        normalized_task_id = int(task_id)
        if normalized_task_id < 0:
            raise ValueError("FlexKV returned an invalid task id")
        return normalized_task_id

    @classmethod
    def _normalize_task_ids(cls, task_ids: Any) -> List[int]:
        if not isinstance(task_ids, list) or not task_ids:
            raise ValueError("KVManager.launch returned invalid task ids")
        return [cls._normalize_task_id(task_id) for task_id in task_ids]

    @staticmethod
    def _normalize_counter_id(counter_id: Any, *, num_counters: int) -> int:
        if isinstance(counter_id, bool) or not isinstance(
            counter_id, (int, np.integer)
        ):
            raise ValueError("FlexKV returned an invalid layerwise counter id")
        normalized_counter_id = int(counter_id)
        if not 0 <= normalized_counter_id < num_counters:
            raise ValueError("FlexKV returned an invalid layerwise counter id")
        return normalized_counter_id

    def _validate_slot_mapping(
        self, *, slot_mapping: torch.Tensor, expected_slots: int
    ) -> Optional[str]:
        if slot_mapping.ndim != 1 or slot_mapping.numel() != expected_slots:
            return "FlexKV slot mapping does not match the held lookup"
        if expected_slots % self.allocator_page_size != 0:
            return "FlexKV slot mapping length is not allocator-page-aligned"
        pages = slot_mapping.reshape(-1, self.allocator_page_size)
        page_starts = pages[:, 0]
        if torch.any(page_starts % self.allocator_page_size != 0).item():
            return "FlexKV slot mapping contains an unaligned page start"
        expected_offsets = torch.arange(
            self.allocator_page_size,
            dtype=slot_mapping.dtype,
            device=slot_mapping.device,
        )
        if not torch.equal(pages, page_starts[:, None] + expected_offsets[None, :]):
            return "FlexKV slot mapping does not contain complete pages"
        return None

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
                        "[FlexKV] GPU register retry %s attempt=%d/%d " "error=%s",
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
        assert (
            kv_caches[0].ndim == 3
        ), f"Expected 3D KV cache tensor, got shape={kv_caches[0].shape}"

        is_mla = self.model_config.use_mla
        num_blocks, num_kv_heads, head_size = kv_caches[0].shape

        gpu_layout = KVCacheLayout(
            type=KVCacheLayoutType.LAYERFIRST,
            num_layer=self.rank_info.num_layers_per_pp_stage,
            num_block=num_blocks // self.storage_page_size,
            tokens_per_block=self.storage_page_size,
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
