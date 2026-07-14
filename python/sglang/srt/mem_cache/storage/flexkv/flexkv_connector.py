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


@dataclass(frozen=True)
class _InflightFlexKVStore:
    version: int
    remaining_task_ids: Tuple[int, ...]
    successful_task_ids: Tuple[int, ...]
    terminal_ready: bool


@dataclass(frozen=True)
class _StoreFinalizationPlan:
    inflight_stores: Dict[str, _InflightFlexKVStore]
    owner_required: set[str]
    ambiguous_stores: Dict[str, str]


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
        self._launched_load_tids: List[int] = []
        # Stores
        self._inflight_stores: Dict[str, _InflightFlexKVStore] = {}
        self._ambiguous_stores: Dict[str, str] = {}
        self._store_owner_required: set[str] = set()
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
        self._launched_load_tids.extend(observation_task_ids)

        return n, producer_id

    def drain_launched_loads(self) -> None:
        """Release only layerwise tasks proven fully terminal and successful."""
        tracked_ids = tuple(self._launched_load_tids)
        outcome = {
            "tracked_task_ids": list(tracked_ids),
            "completed_task_ids": [],
            "reason": None,
        }
        if self._sync_ctx.is_sync_leader:
            try:
                if tracked_ids:
                    if self.kv_manager is None:
                        raise RuntimeError("FlexKV KVManager is not initialized")
                    responses = self.kv_manager.wait(
                        task_ids=list(tracked_ids),
                        timeout=0,
                        completely=True,
                    )
                    normalized_responses = self._normalize_terminal_responses(
                        responses=responses,
                        expected_task_ids=tracked_ids,
                    )
                    completed_task_ids: List[int] = []
                    for task_id, response in normalized_responses.items():
                        if response.status is KVResponseStatus.SUCCESS:
                            completed_task_ids.append(task_id)
                        elif response.status is not KVResponseStatus.TIMEOUT:
                            raise RuntimeError(
                                f"FlexKV layerwise task {task_id} completed with "
                                f"status {response.status}"
                            )
                    outcome["completed_task_ids"] = completed_task_ids
            except TimeoutError:
                pass
            except Exception as exc:  # noqa: BLE001
                outcome["reason"] = f"FlexKV layerwise terminal wait failed: {exc}"
        if self._sync_ctx.needs_sync:
            outcome = self._sync_ctx.scatter(outcome)

        local_valid = True
        completed_task_ids: set[int] = set()
        try:
            outcome_tracked_ids = tuple(
                self._normalize_task_ids_allow_empty(outcome["tracked_task_ids"])
            )
            completed_task_ids = set(
                self._normalize_task_ids_allow_empty(outcome["completed_task_ids"])
            )
            if outcome_tracked_ids != tracked_ids:
                raise RuntimeError(
                    "FlexKV layerwise task tracking differs across ranks"
                )
            if not completed_task_ids.issubset(set(tracked_ids)):
                raise RuntimeError(
                    "FlexKV layerwise drain completed unexpected task ids"
                )
            if outcome["reason"] is not None and not isinstance(outcome["reason"], str):
                raise RuntimeError("FlexKV layerwise drain returned an invalid reason")
        except Exception:  # noqa: BLE001
            local_valid = False
            completed_task_ids = set()

        validation_consistent = self.coordinate_load_publication_step(
            local_success=local_valid
        )
        if not validation_consistent:
            reason = "FlexKV layerwise drain validation differs across ranks"
            self._poison_reason = reason
            for task_id in tracked_ids:
                self._ambiguous_loads[f"layerwise:{task_id}"] = (task_id,)
            raise RuntimeError(reason)

        reason = outcome["reason"]
        if reason is not None:
            self._poison_reason = str(reason)
            for task_id in tracked_ids:
                self._ambiguous_loads[f"layerwise:{task_id}"] = (task_id,)
            raise RuntimeError(str(reason))

        self._launched_load_tids = [
            task_id for task_id in tracked_ids if task_id not in completed_task_ids
        ]

    def ensure_load_back_safe(self) -> None:
        if self._poison_reason is not None or self._ambiguous_loads:
            raise RuntimeError(f"FlexKV load-back is poisoned: {self._poison_reason}")

    def ensure_layerwise_evict_safe(self) -> None:
        self.drain_launched_loads()
        self.ensure_load_back_safe()
        if self._launched_load_tids:
            raise RuntimeError(
                "Cannot evict FlexKV slots while layerwise loads are active"
            )

    def coordinate_load_publication_classification(
        self, local_classification: int
    ) -> Optional[int]:
        minimum_classification = self._sync_ctx.all_reduce_min(local_classification)
        maximum_classification = -self._sync_ctx.all_reduce_min(-local_classification)
        if minimum_classification != maximum_classification:
            return None
        return minimum_classification

    def coordinate_load_match_state(
        self,
        *,
        key_length: int,
        device_length: int,
        lookup_enabled: bool,
    ) -> Optional[Tuple[int, int, bool]]:
        local_state = (key_length, device_length, lookup_enabled)
        leader_state = local_state
        if self._sync_ctx.needs_sync:
            leader_state = self._sync_ctx.scatter(local_state)
        local_valid = int(
            isinstance(leader_state, tuple)
            and len(leader_state) == 3
            and local_state == leader_state
            and key_length >= 0
            and 0 <= device_length <= key_length
        )
        if self._sync_ctx.all_reduce_min(local_valid) == 0:
            return None
        return leader_state

    def coordinate_load_publication_step(self, *, local_success: bool) -> bool:
        return self._sync_ctx.all_reduce_min(int(local_success)) == 1

    def coordinate_store_owner_release(self, *, local_success: bool) -> bool:
        return self._sync_ctx.all_reduce_min(int(local_success)) == 1

    def store_requires_owner_lock(self, rid: str) -> bool:
        return rid in self._store_owner_required

    def poison_load_back(self, reason: str) -> None:
        self._poison_reason = reason

    # ------------------------------------------------------------------
    # Public API — store
    # ------------------------------------------------------------------

    def store_kv(
        self,
        rid: str,
        token_ids: List[int],
        kv_indices: torch.Tensor,
        *,
        local_preparation_error: Optional[str] = None,
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
        self.ensure_load_back_safe()
        local_reason = local_preparation_error
        token_ids_np = np.empty((0,), dtype=np.int64)
        aligned_kv_indices: Optional[torch.Tensor] = None
        try:
            if not rid:
                raise ValueError("FlexKV store requires a non-empty request id")
            if rid in self._inflight_stores or rid in self._ambiguous_stores:
                raise ValueError("FlexKV store request id is already active")
            token_ids_np = np.asarray(token_ids, dtype=np.int64)
            if token_ids_np.ndim != 1 or kv_indices.ndim != 1:
                raise ValueError("FlexKV store inputs must be one-dimensional")
            if len(token_ids_np) != len(kv_indices):
                raise ValueError(
                    "FlexKV store token ids and indices must have equal length"
                )
            aligned_len = (
                len(token_ids_np) // self.storage_page_size * self.storage_page_size
            )
            token_ids_np = token_ids_np[:aligned_len]
            aligned_kv_indices = kv_indices[:aligned_len]
        except Exception as exc:  # noqa: BLE001
            if local_reason is None:
                local_reason = str(exc)

        local_manifest = {
            "rid": rid,
            "token_ids": token_ids_np.tolist(),
            "reason": local_reason,
        }
        leader_manifest = local_manifest
        if self._sync_ctx.needs_sync:
            leader_manifest = self._sync_ctx.scatter(local_manifest)
        try:
            local_preflight_valid = int(
                local_reason is None
                and aligned_kv_indices is not None
                and isinstance(leader_manifest, dict)
                and local_manifest == leader_manifest
                and leader_manifest.get("reason") is None
            )
        except Exception:  # noqa: BLE001
            local_preflight_valid = 0
        if self._sync_ctx.all_reduce_min(local_preflight_valid) == 0:
            return -1
        if not token_ids_np.size:
            return -1

        put_outcome = {
            "task_id": -1,
            "unmatched_mask": [],
            "reason": None,
        }
        if self._sync_ctx.is_sync_leader:
            try:
                if self.kv_manager is None:
                    raise RuntimeError("FlexKV KVManager is not initialized")
                match_result = self.kv_manager.put_match(
                    token_ids=token_ids_np,
                    token_mask=None,
                )
                if match_result is None:
                    raise RuntimeError("FlexKV put_match returned no result")
                task_id, unmatched_mask = match_result
                normalized_task_id = self._normalize_task_id(task_id)
                put_outcome["task_id"] = normalized_task_id
                normalized_mask = np.asarray(unmatched_mask, dtype=np.bool_)
                if normalized_mask.shape != token_ids_np.shape:
                    raise RuntimeError("FlexKV put_match returned an invalid mask")
                put_outcome["unmatched_mask"] = normalized_mask.tolist()
            except Exception as exc:  # noqa: BLE001
                put_outcome["reason"] = str(exc)
        if self._sync_ctx.needs_sync:
            put_outcome = self._sync_ctx.scatter(put_outcome)

        task_id = -1
        unmatched_mask = np.empty((0,), dtype=np.bool_)
        local_put_valid = True
        try:
            if not isinstance(put_outcome, dict):
                raise ValueError("FlexKV store match outcome is invalid")
            reason = put_outcome.get("reason")
            if reason is not None and not isinstance(reason, str):
                raise ValueError("FlexKV store match reason is invalid")
            if reason is None:
                task_id = self._normalize_task_id(put_outcome.get("task_id"))
                unmatched_mask = np.asarray(
                    put_outcome.get("unmatched_mask"),
                    dtype=np.bool_,
                )
                if unmatched_mask.shape != token_ids_np.shape:
                    raise ValueError("FlexKV store match mask differs across ranks")
            elif put_outcome.get("task_id") != -1:
                task_id = self._normalize_task_id(put_outcome.get("task_id"))
        except Exception:  # noqa: BLE001
            local_put_valid = False

        put_valid = self._sync_ctx.all_reduce_min(int(local_put_valid)) == 1
        if not put_valid:
            if not self._cancel_prelaunch_store(task_id=task_id):
                self.poison_load_back(
                    "FlexKV store match validation and cancellation failed"
                )
            return -1
        if put_outcome["reason"] is not None:
            if task_id >= 0 and not self._cancel_prelaunch_store(task_id=task_id):
                self.poison_load_back("FlexKV store match failure cancellation failed")
            return -1
        if not unmatched_mask.any():
            return -1

        slot_mapping_cpu: Optional[torch.Tensor] = None
        local_mapping_valid = True
        local_mapping_manifest: Optional[List[int]] = None
        try:
            if aligned_kv_indices is None:
                raise RuntimeError("FlexKV store indices are unavailable")
            mask_tensor = torch.as_tensor(
                unmatched_mask,
                dtype=torch.bool,
                device=aligned_kv_indices.device,
            )
            slot_mapping_cpu = self._to_cpu_int64(aligned_kv_indices[mask_tensor])
            if slot_mapping_cpu.numel() != int(unmatched_mask.sum()):
                raise RuntimeError("FlexKV store slot mapping has an invalid length")
            local_mapping_manifest = slot_mapping_cpu.tolist()
        except Exception as exc:  # noqa: BLE001
            local_mapping_valid = False
            logger.warning(
                "[FlexKV] store pre-launch mapping failed: %s",
                exc,
                exc_info=True,
            )

        stage_mapping_manifest = self._sync_ctx.scatter_stage(local_mapping_manifest)
        if local_mapping_manifest != stage_mapping_manifest:
            local_mapping_valid = False

        mapping_valid = self._sync_ctx.all_reduce_min(int(local_mapping_valid)) == 1
        if not mapping_valid:
            if not self._cancel_prelaunch_store(task_id=task_id):
                self.poison_load_back("FlexKV store pre-launch cancellation failed")
            return -1

        local_remote_mapping_valid = True
        try:
            if self._sync_ctx.should_send_slot_mapping_to_remote:
                if slot_mapping_cpu is None:
                    raise RuntimeError("FlexKV store slot mapping is unavailable")
                self._send_slot_mapping_to_remote(task_id, slot_mapping_cpu)
        except Exception as exc:  # noqa: BLE001
            local_remote_mapping_valid = False
            logger.warning(
                "[FlexKV] store remote mapping failed: %s",
                exc,
                exc_info=True,
            )

        remote_mapping_valid = (
            self._sync_ctx.all_reduce_min(int(local_remote_mapping_valid)) == 1
        )
        if not remote_mapping_valid:
            if not self._cancel_prelaunch_store(task_id=task_id):
                self.poison_load_back("FlexKV store remote mapping cancellation failed")
            return -1

        local_owner_install_valid = True
        try:
            self._inflight_stores[rid] = _InflightFlexKVStore(
                version=0,
                remaining_task_ids=(),
                successful_task_ids=(),
                terminal_ready=False,
            )
            self._store_owner_required.add(rid)
        except Exception as exc:  # noqa: BLE001
            local_owner_install_valid = False
            logger.warning(
                "[FlexKV] store owner installation failed: %s",
                exc,
                exc_info=True,
            )

        owner_install_valid = (
            self._sync_ctx.all_reduce_min(int(local_owner_install_valid)) == 1
        )
        if not owner_install_valid:
            cancel_valid = self._cancel_prelaunch_store(task_id=task_id)
            local_cleanup_valid = True
            try:
                self._inflight_stores.pop(rid, None)
                self._store_owner_required.discard(rid)
            except Exception as exc:  # noqa: BLE001
                local_cleanup_valid = False
                logger.warning(
                    "[FlexKV] provisional store owner cleanup failed: %s",
                    exc,
                    exc_info=True,
                )
            cleanup_valid = self._sync_ctx.all_reduce_min(int(local_cleanup_valid)) == 1
            if not cancel_valid or not cleanup_valid:
                self.poison_load_back("FlexKV provisional store owner cleanup failed")
            return -1

        try:
            return self._launch_store_after_owner_install(
                rid=rid,
                task_id=task_id,
                slot_mapping_cpu=slot_mapping_cpu,
            )
        except Exception as exc:  # noqa: BLE001
            reason = f"FlexKV post-attempt store launch failed: {exc}"
            self._ambiguous_stores[rid] = reason
            self.poison_load_back(reason)
            raise

    def check_completed_stores(self) -> List[str]:
        """Return rids whose stores have completed since the last call."""
        self.ensure_load_back_safe()
        tracked_stores = {
            rid: self._serialize_store_state(state)
            for rid, state in self._inflight_stores.items()
        }
        outcome = {
            "tracked_stores": tracked_stores,
            "transitions": {},
            "reason": None,
        }
        if self._sync_ctx.is_sync_leader:
            try:
                normalized_responses: Dict[int, Any] = {}
                if any(
                    not state.terminal_ready and not state.remaining_task_ids
                    for state in self._inflight_stores.values()
                ):
                    raise RuntimeError(
                        "FlexKV store terminal drain found an unlaunched state"
                    )
                observation_task_ids = [
                    task_id
                    for state in self._inflight_stores.values()
                    if not state.terminal_ready
                    for task_id in state.remaining_task_ids
                ]
                if observation_task_ids:
                    if self.kv_manager is None:
                        raise RuntimeError("FlexKV KVManager is not initialized")
                    responses = self.kv_manager.wait(
                        task_ids=observation_task_ids,
                        timeout=0,
                        completely=True,
                    )
                    normalized_responses = self._normalize_terminal_responses(
                        responses=responses,
                        expected_task_ids=tuple(observation_task_ids),
                    )
                transitions = {}
                for rid, state in self._inflight_stores.items():
                    if state.terminal_ready:
                        continue
                    newly_successful: List[int] = []
                    remaining_task_ids: List[int] = []
                    for task_id in state.remaining_task_ids:
                        status = normalized_responses[task_id].status
                        if status is KVResponseStatus.SUCCESS:
                            newly_successful.append(task_id)
                        elif status is KVResponseStatus.TIMEOUT:
                            remaining_task_ids.append(task_id)
                        else:
                            raise RuntimeError(
                                f"FlexKV store {rid} completed with status {status}"
                            )
                    successful_task_ids = (
                        list(state.successful_task_ids) + newly_successful
                    )
                    transitions[rid] = {
                        "old": self._serialize_store_state(state),
                        "new": {
                            "version": state.version + 1,
                            "remaining_task_ids": remaining_task_ids,
                            "successful_task_ids": successful_task_ids,
                            "terminal_ready": not remaining_task_ids,
                        },
                    }
                outcome["transitions"] = transitions
            except Exception as exc:  # noqa: BLE001
                outcome["reason"] = f"FlexKV store terminal wait failed: {exc}"
        try:
            if self._sync_ctx.needs_sync:
                outcome = self._sync_ctx.scatter(outcome)
        except Exception as exc:  # noqa: BLE001
            reason = f"FlexKV store terminal publication failed: {exc}"
            self._poison_stores(reason)
            raise

        local_valid = True
        prepared_states: Dict[str, _InflightFlexKVStore] = {}
        try:
            if not isinstance(outcome, dict):
                raise ValueError("FlexKV store terminal outcome is invalid")
            if outcome.get("tracked_stores") != tracked_stores:
                raise ValueError("FlexKV store tracking differs across ranks")
            reason = outcome.get("reason")
            if reason is not None and not isinstance(reason, str):
                raise ValueError("FlexKV store terminal reason is invalid")
            transitions = outcome.get("transitions")
            if not isinstance(transitions, dict):
                raise ValueError("FlexKV store transitions are invalid")
            if reason is None:
                expected_transition_rids = {
                    rid
                    for rid, state in self._inflight_stores.items()
                    if not state.terminal_ready
                }
                if set(transitions) != expected_transition_rids:
                    raise ValueError("FlexKV store transitions are incomplete")
                for rid, transition in transitions.items():
                    state = self._inflight_stores[rid]
                    prepared_states[rid] = self._validate_store_transition(
                        state=state,
                        transition=transition,
                    )
        except Exception:  # noqa: BLE001
            local_valid = False
            prepared_states = {}

        try:
            terminal_valid = self._sync_ctx.all_reduce_min(int(local_valid)) == 1
        except Exception as exc:  # noqa: BLE001
            reason = f"FlexKV store terminal validation failed: {exc}"
            self._poison_stores(reason)
            raise
        if not terminal_valid:
            reason = "FlexKV store terminal validation differs across ranks"
            self._poison_stores(reason)
            raise RuntimeError(reason)
        if outcome.get("reason") is not None:
            self._poison_stores(str(outcome.get("reason")))
            raise RuntimeError(str(outcome.get("reason")))

        local_commit_valid = True
        try:
            for rid, state in prepared_states.items():
                self._inflight_stores[rid] = state
        except Exception as exc:  # noqa: BLE001
            local_commit_valid = False
            logger.warning(
                "[FlexKV] store terminal transition commit failed: %s",
                exc,
                exc_info=True,
            )
        try:
            commit_valid = self._sync_ctx.all_reduce_min(int(local_commit_valid)) == 1
        except Exception as exc:  # noqa: BLE001
            reason = f"FlexKV store terminal commit failed: {exc}"
            self._poison_stores(reason)
            raise
        if not commit_valid:
            reason = "FlexKV store terminal transition commit differs across ranks"
            self._poison_stores(reason)
            raise RuntimeError(reason)

        return [
            rid for rid, state in self._inflight_stores.items() if state.terminal_ready
        ]

    def prepare_store_finalization(self, rids: List[str]) -> _StoreFinalizationPlan:
        self.ensure_load_back_safe()
        local_valid = True
        plan = _StoreFinalizationPlan(
            inflight_stores=self._inflight_stores,
            owner_required=self._store_owner_required,
            ambiguous_stores=self._ambiguous_stores,
        )
        local_manifest: Dict[str, Dict[str, Any]] = {}
        try:
            if len(set(rids)) != len(rids):
                raise ValueError("FlexKV store finalization contains duplicate ids")
            local_manifest = {
                rid: self._serialize_store_state(self._inflight_stores[rid])
                for rid in rids
            }
        except Exception as exc:  # noqa: BLE001
            local_valid = False
            logger.warning(
                "[FlexKV] store finalization manifest failed: %s",
                exc,
                exc_info=True,
            )

        leader_manifest = local_manifest
        try:
            if self._sync_ctx.needs_sync:
                leader_manifest = self._sync_ctx.scatter(local_manifest)
        except Exception as exc:  # noqa: BLE001
            reason = f"FlexKV store finalization publication failed: {exc}"
            self._poison_stores(reason)
            raise
        try:
            if not local_valid:
                raise ValueError("FlexKV store finalization manifest is invalid")
            if local_manifest != leader_manifest:
                raise ValueError("FlexKV store finalization differs across ranks")
            for rid in rids:
                state = self._inflight_stores[rid]
                if (
                    not state.terminal_ready
                    or state.remaining_task_ids
                    or rid not in self._store_owner_required
                    or rid in self._ambiguous_stores
                ):
                    raise ValueError("FlexKV store is not ready for finalization")
            rid_set = set(rids)
            plan = _StoreFinalizationPlan(
                inflight_stores={
                    rid: state
                    for rid, state in self._inflight_stores.items()
                    if rid not in rid_set
                },
                owner_required=self._store_owner_required - rid_set,
                ambiguous_stores={
                    rid: reason
                    for rid, reason in self._ambiguous_stores.items()
                    if rid not in rid_set
                },
            )
        except Exception as exc:  # noqa: BLE001
            local_valid = False
            logger.warning(
                "[FlexKV] store finalization preparation failed: %s",
                exc,
                exc_info=True,
            )

        try:
            prepare_valid = self._sync_ctx.all_reduce_min(int(local_valid)) == 1
        except Exception as exc:  # noqa: BLE001
            reason = f"FlexKV store finalization validation failed: {exc}"
            self._poison_stores(reason)
            raise
        if not prepare_valid:
            reason = "FlexKV store finalization preparation differs across ranks"
            self._poison_stores(reason)
            raise RuntimeError(reason)
        return plan

    def commit_store_finalization(self, plan: _StoreFinalizationPlan) -> None:
        self._inflight_stores = plan.inflight_stores
        self._store_owner_required = plan.owner_required
        self._ambiguous_stores = plan.ambiguous_stores

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

    def ensure_reset_safe(
        self,
        *,
        has_active_store_nodes: bool = False,
        has_quarantined_load_slots: bool = False,
    ) -> None:
        local_safe = int(
            self._poison_reason is None
            and not self._ambiguous_loads
            and not self._ambiguous_stores
            and not self._launched_load_tids
            and not self._inflight_stores
            and not self._store_owner_required
            and not has_active_store_nodes
            and not has_quarantined_load_slots
        )
        combined_safe = self._sync_ctx.all_reduce_min(local_safe)
        if combined_safe == 0:
            raise RuntimeError(
                "Cannot reset FlexKV with ambiguous loads, active layerwise loads, "
                "active stores, or quarantined load slots"
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
        self._inflight_stores.clear()
        self._ambiguous_stores.clear()
        self._store_owner_required.clear()
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
    def _serialize_store_state(state: _InflightFlexKVStore) -> Dict[str, Any]:
        return {
            "version": state.version,
            "remaining_task_ids": list(state.remaining_task_ids),
            "successful_task_ids": list(state.successful_task_ids),
            "terminal_ready": state.terminal_ready,
        }

    def _validate_store_transition(
        self,
        *,
        state: _InflightFlexKVStore,
        transition: Any,
    ) -> _InflightFlexKVStore:
        if not isinstance(transition, dict):
            raise ValueError("FlexKV store transition is invalid")
        if transition.get("old") != self._serialize_store_state(state):
            raise ValueError("FlexKV store transition changed old state identity")
        new_state = transition.get("new")
        if not isinstance(new_state, dict):
            raise ValueError("FlexKV store transition new state is invalid")
        version = new_state.get("version")
        if (
            isinstance(version, bool)
            or not isinstance(version, (int, np.integer))
            or int(version) != state.version + 1
        ):
            raise ValueError("FlexKV store transition version is invalid")
        remaining_task_ids = tuple(
            self._normalize_task_ids_allow_empty(new_state.get("remaining_task_ids"))
        )
        successful_task_ids = tuple(
            self._normalize_task_ids_allow_empty(new_state.get("successful_task_ids"))
        )
        terminal_ready = new_state.get("terminal_ready")
        if not isinstance(terminal_ready, bool):
            raise ValueError("FlexKV store terminal-ready state is invalid")
        if len(set(remaining_task_ids + successful_task_ids)) != len(
            remaining_task_ids + successful_task_ids
        ):
            raise ValueError("FlexKV store transition reused task ids")
        old_successful = state.successful_task_ids
        if successful_task_ids[: len(old_successful)] != old_successful:
            raise ValueError("FlexKV store transition changed successful task ids")
        newly_successful = successful_task_ids[len(old_successful) :]
        if set(newly_successful).intersection(remaining_task_ids):
            raise ValueError("FlexKV store transition task partition overlaps")
        if set(newly_successful).union(remaining_task_ids) != set(
            state.remaining_task_ids
        ):
            raise ValueError("FlexKV store transition task partition is incomplete")
        remaining_set = set(remaining_task_ids)
        successful_set = set(newly_successful)
        if remaining_task_ids != tuple(
            task_id for task_id in state.remaining_task_ids if task_id in remaining_set
        ):
            raise ValueError("FlexKV store remaining task order changed")
        if newly_successful != tuple(
            task_id for task_id in state.remaining_task_ids if task_id in successful_set
        ):
            raise ValueError("FlexKV store successful task order changed")
        if terminal_ready != (not remaining_task_ids):
            raise ValueError("FlexKV store terminal-ready state is inconsistent")
        return _InflightFlexKVStore(
            version=int(version),
            remaining_task_ids=remaining_task_ids,
            successful_task_ids=successful_task_ids,
            terminal_ready=terminal_ready,
        )

    def _poison_stores(self, reason: str) -> None:
        for rid in self._inflight_stores:
            self._ambiguous_stores[rid] = reason
        self.poison_load_back(reason)

    def _launch_store_after_owner_install(
        self,
        *,
        rid: str,
        task_id: int,
        slot_mapping_cpu: Optional[torch.Tensor],
    ) -> int:
        launch_outcome = {
            "observation_task_ids": [],
            "reason": None,
        }
        if self._sync_ctx.is_sync_leader:
            try:
                if self.kv_manager is None or slot_mapping_cpu is None:
                    raise RuntimeError("FlexKV store launch is not prepared")
                launch_outcome["observation_task_ids"] = self._normalize_task_ids(
                    self.kv_manager.launch(
                        task_ids=[task_id],
                        slot_mappings=[slot_mapping_cpu],
                        as_batch=False,
                        layerwise_transfer=False,
                    )
                )
            except Exception as exc:  # noqa: BLE001
                launch_outcome["reason"] = str(exc)
        if self._sync_ctx.needs_sync:
            launch_outcome = self._sync_ctx.scatter(launch_outcome)

        observation_task_ids: Tuple[int, ...] = ()
        local_launch_valid = True
        try:
            if not isinstance(launch_outcome, dict):
                raise ValueError("FlexKV store launch outcome is invalid")
            reason = launch_outcome.get("reason")
            if reason is not None and not isinstance(reason, str):
                raise ValueError("FlexKV store launch reason is invalid")
            raw_observation_task_ids = launch_outcome.get("observation_task_ids")
            if reason is None:
                observation_task_ids = tuple(
                    self._normalize_task_ids(raw_observation_task_ids)
                )
                active_task_ids = {
                    observation_task_id
                    for active_rid, state in self._inflight_stores.items()
                    if active_rid != rid
                    for observation_task_id in (
                        state.remaining_task_ids + state.successful_task_ids
                    )
                }
                if active_task_ids.intersection(observation_task_ids):
                    raise ValueError("FlexKV reused an active store observation id")
            else:
                observation_task_ids = tuple(
                    self._normalize_task_ids_allow_empty(raw_observation_task_ids)
                )
        except Exception:  # noqa: BLE001
            local_launch_valid = False

        launch_valid = self._sync_ctx.all_reduce_min(int(local_launch_valid)) == 1
        if launch_valid and launch_outcome.get("reason") is None:
            self._inflight_stores[rid] = _InflightFlexKVStore(
                version=1,
                remaining_task_ids=observation_task_ids,
                successful_task_ids=(),
                terminal_ready=False,
            )
            return observation_task_ids[0]

        if not launch_valid:
            observation_task_ids = ()
        ambiguous_reason = (
            str(launch_outcome.get("reason"))
            if launch_valid
            else "FlexKV store launch validation differs across ranks"
        )
        self._inflight_stores[rid] = _InflightFlexKVStore(
            version=1,
            remaining_task_ids=observation_task_ids,
            successful_task_ids=(),
            terminal_ready=False,
        )
        self._ambiguous_stores[rid] = ambiguous_reason
        self.poison_load_back(ambiguous_reason)
        return task_id

    def _cancel_prelaunch_store(self, *, task_id: int) -> bool:
        outcome = {"success": False, "reason": None}
        if self._sync_ctx.is_sync_leader:
            try:
                if self.kv_manager is None:
                    raise RuntimeError("FlexKV KVManager is not initialized")
                self.kv_manager.cancel([task_id])
                outcome["success"] = True
            except Exception as exc:  # noqa: BLE001
                outcome["reason"] = str(exc)
        if self._sync_ctx.needs_sync:
            outcome = self._sync_ctx.scatter(outcome)

        local_valid = int(
            isinstance(outcome, dict)
            and isinstance(outcome.get("success"), bool)
            and (
                outcome.get("reason") is None or isinstance(outcome.get("reason"), str)
            )
        )
        if self._sync_ctx.all_reduce_min(local_valid) == 0:
            return False
        return bool(outcome["success"])

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
        counter_registration_started = False

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
            counter_registration_started = True
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
            if layerwise:
                cleanup_status = 1
                try:
                    if (
                        counter_registration_started
                        and pending is not None
                        and producer_id >= 0
                    ):
                        self._abort_layerwise_counter(
                            pending=pending,
                            producer_id=producer_id,
                        )
                except Exception as exc:  # noqa: BLE001
                    cleanup_status = 0
                    logger.warning(
                        "[FlexKV] layerwise pre-launch rollback failed: %s",
                        exc,
                        exc_info=True,
                    )
                combined_cleanup_status = self._sync_ctx.all_reduce_min(cleanup_status)
                if combined_cleanup_status == 0:
                    local_reason = (
                        "FlexKV layerwise pre-launch rollback failed on at least one "
                        "rank"
                    )
                    self._poison_reason = local_reason
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
                counter.ensure_producer_ready(producer_id=producer_id)
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

    def _abort_layerwise_counter(
        self, *, pending: _PendingFlexKVLookup, producer_id: int
    ) -> None:
        counter = self.layer_done_counter
        if counter is None:
            raise RuntimeError("FlexKV layerwise counter is not initialized")
        normalized_producer_id = self._normalize_counter_id(
            producer_id,
            num_counters=counter.num_counters,
        )
        counter.abort_prepared_transfer(
            task_id=pending.task_id,
            producer_id=normalized_producer_id,
        )

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
        normalized_task_ids = [cls._normalize_task_id(task_id) for task_id in task_ids]
        if len(set(normalized_task_ids)) != len(normalized_task_ids):
            raise ValueError("KVManager.launch returned duplicate task ids")
        return normalized_task_ids

    @classmethod
    def _normalize_task_ids_allow_empty(cls, task_ids: Any) -> List[int]:
        if not isinstance(task_ids, list):
            raise ValueError("FlexKV task ids must be a list")
        if not task_ids:
            return []
        return cls._normalize_task_ids(task_ids)

    def _normalize_terminal_responses(
        self,
        *,
        responses: Any,
        expected_task_ids: Tuple[int, ...],
    ) -> Dict[int, Any]:
        if not isinstance(responses, dict):
            raise ValueError("FlexKV terminal wait returned an invalid result")
        normalized_responses: Dict[int, Any] = {}
        for task_id, response in responses.items():
            normalized_task_id = self._normalize_task_id(task_id)
            response_task_id = self._normalize_task_id(response.task_id)
            if response_task_id != normalized_task_id:
                raise ValueError("FlexKV terminal response changed task identity")
            normalized_responses[normalized_task_id] = response
        if set(normalized_responses) != set(expected_task_ids):
            raise ValueError("FlexKV terminal wait returned unexpected task ids")
        return normalized_responses

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
