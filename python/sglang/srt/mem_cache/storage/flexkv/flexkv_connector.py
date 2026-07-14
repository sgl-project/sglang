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


def is_flexkv_layerwise_transfer_enabled() -> bool:
    return bool(int(os.environ.get("FLEXKV_ENABLE_LAYERWISE_TRANSFER", "0")))


class _FlexKVLookupState(Enum):
    HELD = "held"
    LAUNCHING = "launching"
    LAUNCHED = "launched"
    AMBIGUOUS = "ambiguous"


@dataclass
class _PendingFlexKVLookup:
    lookup_task_id: int
    expected_slots: int
    state: _FlexKVLookupState = _FlexKVLookupState.HELD
    terminal_task_ids: Tuple[int, ...] = ()


@dataclass(frozen=True)
class _FlexKVRetrieveResult:
    lookup_task_id: int
    terminal_task_ids: Tuple[int, ...]
    requested_slots: int
    terminal_proof: bool
    terminal_success: bool
    prelaunch_miss: bool
    prelaunch_contract_error: bool


class _FlexKVFatalTransferError(RuntimeError):
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
        self.enable_layerwise = is_flexkv_layerwise_transfer_enabled()
        if self.enable_layerwise and self.allocator_page_size > 1:
            raise ValueError(
                "FlexKV layerwise transfer currently requires allocator page "
                "size 1; disable FLEXKV_ENABLE_LAYERWISE_TRANSFER or use MP"
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
          ``(lookup_task_id, hit_count)``. ``hit_count`` is page-aligned
          and may be smaller than the raw FlexKV match if the page
          floor truncated it.
        """
        if rid is not None and rid in self._pending_lookups:
            self.release_pending(rid)

        payload = {"lookup_task_id": -1, "hit": 0, "error": None}
        if self._sync_ctx.is_sync_leader and self.kv_manager is not None:
            if self.enable_layerwise and self.allocator_page_size == 1:
                payload = self._lookup_legacy_layerwise_on_leader(
                    token_ids=token_ids,
                    token_mask=token_mask,
                    keep_task=rid is not None,
                )
            else:
                payload = self._lookup_page_aligned_on_leader(
                    token_ids=token_ids,
                    token_mask=token_mask,
                    keep_task=rid is not None,
                )
        if self._sync_ctx.needs_sync:
            payload = self._sync_ctx.scatter(payload)

        lookup_task_id = int(payload["lookup_task_id"])
        hit_length = int(payload["hit"])
        error = payload["error"]
        if error is not None:
            if rid is not None and lookup_task_id >= 0:
                self._pending_lookups[rid] = _PendingFlexKVLookup(
                    lookup_task_id=lookup_task_id,
                    expected_slots=hit_length,
                    state=_FlexKVLookupState.AMBIGUOUS,
                )
            raise _FlexKVFatalTransferError(str(error))

        if hit_length > 0 and rid is not None and lookup_task_id >= 0:
            self._pending_lookups[rid] = _PendingFlexKVLookup(
                lookup_task_id=lookup_task_id,
                expected_slots=hit_length,
            )

        return lookup_task_id, hit_length

    def release_pending(self, rid: str) -> None:
        """Cancel the task held by an earlier ``lookup_kv(rid=...)`` that
        won't be followed by a ``retrieve_kv`` (e.g. allocation failed)."""
        pending = self._pending_lookups.get(rid)
        if self._sync_ctx.needs_sync:
            all_missing = self._sync_ctx.all_reduce_min(1 if pending is None else 0)
            if all_missing == 1:
                return
        elif pending is None:
            return

        payload = {
            "lookup_task_id": pending.lookup_task_id if pending is not None else -1,
            "cancelled": False,
            "error": None,
        }
        if self._sync_ctx.is_sync_leader:
            if pending is None:
                payload["error"] = "sync leader is missing the held lookup"
            elif pending.state is not _FlexKVLookupState.HELD:
                payload["error"] = f"lookup is in unproven state {pending.state.value}"
            else:
                assert self.kv_manager is not None
                try:
                    cancel_result = self.kv_manager.cancel([pending.lookup_task_id])
                    if cancel_result is not None:
                        raise RuntimeError(
                            "KVManager.cancel must synchronously return None"
                        )
                    payload["cancelled"] = True
                except Exception as exc:  # noqa: BLE001
                    payload["error"] = str(exc)
        if self._sync_ctx.needs_sync:
            payload = self._sync_ctx.scatter(payload)

        if payload["error"] is not None or not payload["cancelled"]:
            if pending is not None:
                pending.state = _FlexKVLookupState.AMBIGUOUS
            raise _FlexKVFatalTransferError(
                f"Failed to cancel unlaunched FlexKV lookup "
                f"{payload['lookup_task_id']}: {payload['error']}"
            )
        self._pending_lookups.pop(rid, None)

    def requires_mp_eviction(self, local_has_capacity: bool) -> bool:
        return self._sync_ctx.all_reduce_min(1 if local_has_capacity else 0) == 0

    def retrieve_kv(
        self,
        rid: str,
        slot_mapping: Optional[torch.Tensor],
        expected_lookup_task_id: int,
        local_manifest_error: Optional[str] = None,
        local_prelaunch_error: Optional[str] = None,
    ) -> _FlexKVRetrieveResult:
        """Synchronous load: ``launch`` + ``wait``.

        Returns the lookup identity, launch terminal identities, requested
        slot count, and terminal proof. A clean prelaunch miss is explicitly
        distinguished from an ambiguous launched transfer.
        """
        pending = self._pending_lookups.get(rid)
        slot_mapping_cpu: Optional[torch.Tensor] = None
        local_error: Optional[str] = None
        local_status = 1
        if local_manifest_error is not None:
            local_error = local_manifest_error
            local_status = -1
        elif local_prelaunch_error is not None:
            local_error = local_prelaunch_error
            local_status = 0
        elif pending is None:
            local_error = "missing held lookup"
            local_status = -1
        elif slot_mapping is None:
            local_error = "local allocator could not acquire the slot lease"
            local_status = 0
        elif pending.state is not _FlexKVLookupState.HELD:
            local_error = f"lookup is in state {pending.state.value}"
            local_status = -1
        else:
            try:
                slot_mapping_cpu = self._to_cpu_int64(slot_mapping)
                if slot_mapping_cpu.numel() != pending.expected_slots:
                    local_error = (
                        f"slot mapping has {slot_mapping_cpu.numel()} entries, "
                        f"expected {pending.expected_slots}"
                    )
                    local_status = -1
            except Exception as exc:  # noqa: BLE001
                local_error = str(exc)
                local_status = 0

        manifest = {
            "rid": rid,
            "lookup_task_id": pending.lookup_task_id if pending is not None else -1,
            "expected_slots": pending.expected_slots if pending is not None else -1,
            "slot_mapping": (
                slot_mapping_cpu.tolist()
                if local_status == 1 and slot_mapping_cpu is not None
                else None
            ),
        }
        if self._sync_ctx.needs_sync:
            manifest = self._sync_ctx.scatter(manifest)

        if local_status == 1:
            assert pending is not None and slot_mapping_cpu is not None
            if (
                manifest["rid"] != rid
                or int(manifest["lookup_task_id"]) != pending.lookup_task_id
                or int(manifest["expected_slots"]) != pending.expected_slots
                or expected_lookup_task_id != pending.lookup_task_id
            ):
                local_error = "lookup manifest differs across ranks"
                local_status = -1
            elif manifest["slot_mapping"] is None:
                local_error = "sync leader could not acquire the slot lease"
                local_status = 0
            elif manifest["slot_mapping"] != slot_mapping_cpu.tolist():
                local_error = "slot mapping differs across ranks"
                local_status = -1
            elif self._sync_ctx.should_send_slot_mapping_to_remote:
                try:
                    self._send_slot_mapping_to_remote(
                        pending.lookup_task_id, slot_mapping_cpu
                    )
                except Exception as exc:  # noqa: BLE001
                    local_error = str(exc)
                    local_status = 0

        all_ranks_status = self._sync_ctx.all_reduce_min(local_status)
        if all_ranks_status < 1:
            if local_error is not None:
                logger.warning(
                    "[FlexKV] retrieve_kv prelaunch validation failed: %s",
                    local_error,
                )
            lookup_task_id = pending.lookup_task_id if pending is not None else -1
            self.release_pending(rid)
            return _FlexKVRetrieveResult(
                lookup_task_id=lookup_task_id,
                terminal_task_ids=(),
                requested_slots=0,
                terminal_proof=False,
                terminal_success=False,
                prelaunch_miss=True,
                prelaunch_contract_error=all_ranks_status < 0,
            )

        assert pending is not None and slot_mapping_cpu is not None
        pending.state = _FlexKVLookupState.LAUNCHING
        launch_payload = {
            "terminal_task_ids": [],
            "requested_slots": int(slot_mapping_cpu.numel()),
            "error": None,
        }
        if self._sync_ctx.is_sync_leader:
            assert self.kv_manager is not None
            try:
                terminal_task_ids = self.kv_manager.launch(
                    task_ids=[pending.lookup_task_id],
                    slot_mappings=[slot_mapping_cpu],
                    as_batch=True,
                    layerwise_transfer=False,
                )
                if not isinstance(terminal_task_ids, list):
                    raise TypeError("KVManager.launch must return a list of task ids")
                if len(terminal_task_ids) != 1:
                    raise RuntimeError(
                        "KVManager.launch returned a terminal task count that "
                        "does not match the single lookup manifest"
                    )
                if any(
                    not isinstance(task_id, int) or task_id < 0
                    for task_id in terminal_task_ids
                ):
                    raise RuntimeError(
                        "KVManager.launch returned an invalid terminal task id"
                    )
                launch_payload["terminal_task_ids"] = terminal_task_ids
            except Exception as exc:  # noqa: BLE001
                launch_payload["error"] = str(exc)
        if self._sync_ctx.needs_sync:
            launch_payload = self._sync_ctx.scatter(launch_payload)

        terminal_task_ids = tuple(
            int(task_id) for task_id in launch_payload["terminal_task_ids"]
        )
        pending.terminal_task_ids = terminal_task_ids
        if launch_payload["error"] is not None:
            pending.state = _FlexKVLookupState.AMBIGUOUS
            raise _FlexKVFatalTransferError(
                f"FlexKV launch outcome is ambiguous for lookup "
                f"{pending.lookup_task_id}: {launch_payload['error']}"
            )
        if (
            int(launch_payload["requested_slots"]) != pending.expected_slots
            or len(terminal_task_ids) != 1
        ):
            pending.state = _FlexKVLookupState.AMBIGUOUS
            raise _FlexKVFatalTransferError(
                "FlexKV launch manifest changed across ranks; retaining slot lease"
            )
        pending.state = _FlexKVLookupState.LAUNCHED

        terminal_payload = {
            "complete": False,
            "successful": False,
            "error": None,
        }
        if self._sync_ctx.is_sync_leader:
            assert self.kv_manager is not None
            try:
                responses = self.kv_manager.wait(
                    task_ids=list(terminal_task_ids),
                    timeout=30.0,
                    completely=True,
                )
                expected_ids = set(terminal_task_ids)
                if set(responses) != expected_ids:
                    raise RuntimeError(
                        "KVManager.wait did not return every terminal task id"
                    )
                successful = True
                for task_id in terminal_task_ids:
                    response = responses[task_id]
                    if response.task_id != task_id:
                        raise RuntimeError(
                            f"terminal response identity changed for task {task_id}"
                        )
                    if response.status is KVResponseStatus.SUCCESS:
                        continue
                    if response.status in (
                        KVResponseStatus.FAILED,
                        KVResponseStatus.CANCELLED,
                    ):
                        successful = False
                    else:
                        raise RuntimeError(
                            f"terminal task {task_id} returned {response.status}"
                        )
                terminal_payload["complete"] = True
                terminal_payload["successful"] = successful
            except Exception as exc:  # noqa: BLE001
                terminal_payload["error"] = str(exc)
        if self._sync_ctx.needs_sync:
            terminal_payload = self._sync_ctx.scatter(terminal_payload)

        if not terminal_payload["complete"]:
            pending.state = _FlexKVLookupState.AMBIGUOUS
            raise _FlexKVFatalTransferError(
                f"FlexKV terminal proof failed for tasks "
                f"{list(terminal_task_ids)}: {terminal_payload['error']}"
            )

        self._pending_lookups.pop(rid, None)
        return _FlexKVRetrieveResult(
            lookup_task_id=pending.lookup_task_id,
            terminal_task_ids=terminal_task_ids,
            requested_slots=pending.expected_slots,
            terminal_proof=True,
            terminal_success=bool(terminal_payload["successful"]),
            prelaunch_miss=False,
            prelaunch_contract_error=False,
        )

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
        pending = self._pending_lookups.pop(rid, None)
        if pending is None:
            return 0, -1
        if pending.state is not _FlexKVLookupState.HELD:
            raise _FlexKVFatalTransferError(
                f"Layerwise lookup for rid={rid} is in state {pending.state.value}"
            )
        fkv_task_id = pending.lookup_task_id

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

    def reset(self) -> None:
        unproven_rids = [
            rid
            for rid, pending in self._pending_lookups.items()
            if pending.state is not _FlexKVLookupState.HELD
        ]
        if unproven_rids:
            raise _FlexKVFatalTransferError(
                "Cannot reset FlexKV while terminal proof is missing for "
                f"rids={unproven_rids}"
            )
        for rid in list(self._pending_lookups):
            self.release_pending(rid)
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
        unproven_rids = [
            rid
            for rid, pending in self._pending_lookups.items()
            if pending.state is not _FlexKVLookupState.HELD
        ]
        if unproven_rids:
            raise _FlexKVFatalTransferError(
                "Cannot shut down FlexKV while terminal proof is missing for "
                f"rids={unproven_rids}"
            )
        for rid in list(self._pending_lookups):
            self.release_pending(rid)
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

    def _lookup_legacy_layerwise_on_leader(
        self,
        *,
        token_ids: List[int],
        token_mask: torch.Tensor,
        keep_task: bool,
    ) -> Dict[str, Any]:
        assert self.kv_manager is not None

        try:
            result = self.kv_manager.get_match(
                token_ids=np.asarray(token_ids, dtype=np.int64),
                token_mask=self._as_numpy_mask(token_mask),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("[FlexKV] get_match raised: %s", exc)
            return {"lookup_task_id": -1, "hit": 0, "error": None}
        if result is None:
            return {"lookup_task_id": -1, "hit": 0, "error": None}

        lookup_task_id, matched_mask = result
        hit_length = int(matched_mask.sum()) if matched_mask is not None else 0
        if hit_length <= 0 or keep_task:
            return {
                "lookup_task_id": int(lookup_task_id),
                "hit": hit_length,
                "error": None,
            }

        cancel_error = self._cancel_unlaunched_on_leader(int(lookup_task_id))
        return {
            "lookup_task_id": int(lookup_task_id),
            "hit": hit_length,
            "error": cancel_error,
        }

    def _lookup_page_aligned_on_leader(
        self,
        *,
        token_ids: List[int],
        token_mask: torch.Tensor,
        keep_task: bool,
    ) -> Dict[str, Any]:
        assert self.kv_manager is not None

        token_ids_array = np.asarray(token_ids, dtype=np.int64)
        current_mask = np.asarray(self._as_numpy_mask(token_mask), dtype=np.bool_)
        if current_mask.ndim != 1 or current_mask.shape != token_ids_array.shape:
            return {
                "lookup_task_id": -1,
                "hit": 0,
                "error": "FlexKV lookup mask must match the token id shape",
            }

        for attempt in range(4):
            try:
                result = self.kv_manager.get_match(
                    token_ids=token_ids_array,
                    token_mask=current_mask,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("[FlexKV] get_match raised: %s", exc)
                return {"lookup_task_id": -1, "hit": 0, "error": None}
            if result is None:
                return {"lookup_task_id": -1, "hit": 0, "error": None}

            lookup_task_id, matched_mask = result
            if matched_mask is None:
                return {
                    "lookup_task_id": int(lookup_task_id),
                    "hit": 0,
                    "error": None,
                }

            matched_mask_array = np.asarray(matched_mask, dtype=np.bool_)
            valid_mask = (
                matched_mask_array.ndim == 1
                and matched_mask_array.shape == current_mask.shape
                and not np.any(matched_mask_array & ~current_mask)
            )
            if not valid_mask:
                cancel_error = self._cancel_unlaunched_on_leader(int(lookup_task_id))
                return {
                    "lookup_task_id": (
                        int(lookup_task_id) if cancel_error is not None else -1
                    ),
                    "hit": 0,
                    "error": cancel_error
                    or "FlexKV get_match returned an invalid matched mask",
                }

            candidate_positions = np.flatnonzero(current_mask)
            matched_candidates = matched_mask_array[candidate_positions]
            false_positions = np.flatnonzero(~matched_candidates)
            leading_length = (
                int(false_positions[0])
                if false_positions.size > 0
                else int(matched_candidates.size)
            )
            if leading_length <= 0:
                if np.any(matched_mask_array):
                    cancel_error = self._cancel_unlaunched_on_leader(
                        int(lookup_task_id)
                    )
                    return {
                        "lookup_task_id": (
                            int(lookup_task_id) if cancel_error is not None else -1
                        ),
                        "hit": 0,
                        "error": cancel_error,
                    }
                return {
                    "lookup_task_id": int(lookup_task_id),
                    "hit": 0,
                    "error": None,
                }

            aligned_length = (
                leading_length // self.allocator_page_size * self.allocator_page_size
            )
            aligned_prefix_mask = np.zeros_like(current_mask)
            aligned_prefix_mask[candidate_positions[:aligned_length]] = True
            if np.array_equal(matched_mask_array, aligned_prefix_mask):
                if keep_task:
                    return {
                        "lookup_task_id": int(lookup_task_id),
                        "hit": aligned_length,
                        "error": None,
                    }
                cancel_error = self._cancel_unlaunched_on_leader(int(lookup_task_id))
                return {
                    "lookup_task_id": int(lookup_task_id),
                    "hit": aligned_length,
                    "error": cancel_error,
                }

            cancel_error = self._cancel_unlaunched_on_leader(int(lookup_task_id))
            if cancel_error is not None:
                return {
                    "lookup_task_id": int(lookup_task_id),
                    "hit": aligned_length,
                    "error": cancel_error,
                }
            if aligned_length == 0 or attempt == 3:
                return {"lookup_task_id": -1, "hit": 0, "error": None}

            current_mask = aligned_prefix_mask

        return {"lookup_task_id": -1, "hit": 0, "error": None}

    def _cancel_unlaunched_on_leader(self, lookup_task_id: int) -> Optional[str]:
        assert self.kv_manager is not None
        try:
            cancel_result = self.kv_manager.cancel([lookup_task_id])
            if cancel_result is not None:
                raise RuntimeError("KVManager.cancel must synchronously return None")
        except Exception as exc:  # noqa: BLE001
            return str(exc)
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
