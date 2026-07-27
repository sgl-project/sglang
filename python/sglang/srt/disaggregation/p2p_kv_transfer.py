from __future__ import annotations

import dataclasses
import enum
import logging
import random
import threading
import time
from array import array
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from typing import Any, ClassVar

import requests
import torch

from sglang.srt.disaggregation.base.conn import KVPoll, StateType
from sglang.srt.disaggregation.utils import (
    KVClassType,
    TransferBackend,
    get_kv_class,
    poll_and_all_reduce_attn_cp_tp_group,
)
from sglang.srt.managers.io_struct import (
    P2PKVTransferReqInput,
    P2PKVTransferReqOutput,
)
from sglang.srt.mem_cache.base_prefix_cache import InsertParams, MatchPrefixParams
from sglang.srt.mem_cache.common import (
    evict_from_tree_cache,
    kv_to_page_indices,
    page_align_floor,
)
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.utils.msgspec_utils import msgspec_to_builtins

logger = logging.getLogger(__name__)


def _p2p_req_to_builtins(req: P2PKVTransferReqInput):
    if dataclasses.is_dataclass(req):
        return dataclasses.asdict(req)
    return msgspec_to_builtins(req)


def _radix_key(token_ids, extra_key=None):
    return RadixKey(array("l", [int(x) for x in token_ids]), extra_key)


class _P2PCacheIntegrityError(RuntimeError):
    pass


class _TargetOwnershipPhase(enum.Enum):
    P2P_OWNED = "p2p_owned"
    INSERT_OWNERSHIP_TRANSFERRED = "insert_ownership_transferred"
    INSERT_COMMITTED = "insert_committed"


class P2PTransferState(enum.Enum):
    RESERVED = "reserved"
    WAIT_SOURCE = "wait_source"
    TRANSFERRING = "transferring"
    CONSENSUS = "consensus"
    COMMIT = "commit"
    FAILED = "failed"


@dataclasses.dataclass
class PendingP2PTransfer:
    req: P2PKVTransferReqInput
    role: str
    state: P2PTransferState
    deadline: float
    kv_manager: Any
    sender: Any = None
    receiver: Any = None
    match: Any = None
    lock_params: Any = None
    actual_len: int = 0
    prefix_len: int = 0
    allocation: _TargetAllocation | None = None
    pair_key: tuple[str, str] | None = None
    source_future: Any = None
    trigger_source: bool = False
    source_payload: dict[str, Any] | None = None
    source_error: str | None = None
    source_terminal: bool = True
    room: int = 0
    consensus_work: Any = None
    consensus_tensor: torch.Tensor | None = None
    consensus_label: str | None = None
    abort_requested: bool = False
    quarantine_allocation: bool = False
    cleanup_done: bool = False


@dataclasses.dataclass
class _TargetAllocation:
    kv: torch.Tensor
    mamba: torch.Tensor | None = None
    kv_owned_by_p2p: bool = True
    mamba_owned_by_p2p: bool = False
    phase: _TargetOwnershipPhase = _TargetOwnershipPhase.P2P_OWNED

    def attach_mamba(self, value: torch.Tensor) -> None:
        if self.phase is not _TargetOwnershipPhase.P2P_OWNED:
            raise RuntimeError("cannot attach Mamba state after ownership transfer")
        self.mamba = value
        self.mamba_owned_by_p2p = True

    def trim_kv(self, length: int, allocator) -> None:
        if self.phase is not _TargetOwnershipPhase.P2P_OWNED:
            raise RuntimeError("cannot trim KV after ownership transfer")
        if length < len(self.kv):
            allocator.free(self.kv[length:])
            self.kv = self.kv[:length]

    def transfer_to_insert(self) -> None:
        if self.phase is not _TargetOwnershipPhase.P2P_OWNED:
            raise RuntimeError(f"invalid ownership transfer from {self.phase.value}")
        self.phase = _TargetOwnershipPhase.INSERT_OWNERSHIP_TRANSFERRED
        self.kv_owned_by_p2p = False
        self.mamba_owned_by_p2p = False

    def commit(self) -> None:
        if self.phase is not _TargetOwnershipPhase.INSERT_OWNERSHIP_TRANSFERRED:
            raise RuntimeError(f"invalid ownership commit from {self.phase.value}")
        self.phase = _TargetOwnershipPhase.INSERT_COMMITTED

    def cleanup(self, kv_allocator, mamba_allocator=None) -> None:
        if self.kv_owned_by_p2p:
            self.kv_owned_by_p2p = False
            kv_allocator.free(self.kv)
        if self.mamba_owned_by_p2p and self.mamba is not None:
            self.mamba_owned_by_p2p = False
            if mamba_allocator is None:
                raise RuntimeError("missing Mamba allocator during P2P cleanup")
            mamba_allocator.free(self.mamba)


@dataclasses.dataclass(frozen=True)
class _TargetRegistrationResult:
    committed_tokens: int
    duplicate_kv_owner: str
    duplicate_kv_freed_by_p2p: bool
    cache_accepted_mamba: bool


class PrefillP2PMooncakeTransferEngine:
    """Narrow experimental Prefill->Prefill Mooncake transfer path.

    This intentionally supports only identical-layout Prefill pairs. The target
    Prefill allocates destination KV slots and optional model state, then asks the source
    Prefill to write into those slots via Mooncake using the target session id.
    """

    _PAIR_GATE_LOCK = threading.Lock()
    _PAIR_GATE_DEADLINES: ClassVar[dict[tuple[str, str], float]] = {}
    _TRANSFER_TIMEOUT_S = 120.0
    _SOURCE_REQUEST_TIMEOUT_S = 245.0
    _PAIR_GATE_TTL_S = 365.0

    def __init__(self, scheduler, http_executor=None):
        self.scheduler = scheduler
        self._source_bootstrap_addrs = {}
        self._capacity_detail_counter = 0
        self._pending_transfers: dict[str, PendingP2PTransfer] = {}
        self._unsafe_quarantine = False
        self._http_executor = http_executor or ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="sglang-p2p-control"
        )

    def _poll_receiver_consensus(self, receiver, force_failed: bool = False):
        poller = receiver
        if force_failed:
            poller = SimpleNamespace(poll=lambda: KVPoll.Failed)
        cp_group = getattr(self.scheduler, "attn_cp_cpu_group", None)
        tp_group = getattr(self.scheduler, "attn_tp_cpu_group", None)
        if cp_group is None or tp_group is None:
            return poller.poll()
        return poll_and_all_reduce_attn_cp_tp_group(
            [poller],
            cp_group,
            tp_group,
        )[0]

    def _min_target_consensus(self, value: int) -> int:
        cp_group = getattr(self.scheduler, "attn_cp_cpu_group", None)
        tp_group = getattr(self.scheduler, "attn_tp_cpu_group", None)
        if cp_group is None or tp_group is None:
            return int(value)
        value_tensor = torch.tensor([int(value)], dtype=torch.int64, device="cpu")
        torch.distributed.all_reduce(
            value_tensor,
            op=torch.distributed.ReduceOp.MIN,
            group=tp_group,
        )
        torch.distributed.all_reduce(
            value_tensor,
            op=torch.distributed.ReduceOp.MIN,
            group=cp_group,
        )
        return int(value_tensor.item())

    def _world_min_int(self, value: int) -> int:
        world_group = getattr(
            getattr(self.scheduler, "world_group", None), "cpu_group", None
        )
        if (
            world_group is None
            or not torch.distributed.is_available()
            or not torch.distributed.is_initialized()
        ):
            return int(value)
        value_tensor = torch.tensor([int(value)], dtype=torch.int64, device="cpu")
        torch.distributed.all_reduce(
            value_tensor,
            op=torch.distributed.ReduceOp.MIN,
            group=world_group,
        )
        return int(value_tensor.item())

    def _progress_world_min(
        self, pending: PendingP2PTransfer, label: str, value: int
    ) -> int | None:
        """Progress one ordered, nonblocking model-parallel consensus."""
        world_group = getattr(
            getattr(self.scheduler, "world_group", None), "cpu_group", None
        )
        if world_group is None:
            return int(value)

        if pending.consensus_work is None:
            pending.consensus_tensor = torch.tensor(
                [int(value)], dtype=torch.int64, device="cpu"
            )
            pending.consensus_label = label
            pending.consensus_work = torch.distributed.all_reduce(
                pending.consensus_tensor,
                op=torch.distributed.ReduceOp.MIN,
                group=world_group,
                async_op=True,
            )
            return None

        if pending.consensus_label != label:
            return None
        if not pending.consensus_work.is_completed():
            return None

        pending.consensus_work.wait()
        result = int(pending.consensus_tensor.item())
        pending.consensus_work = None
        pending.consensus_tensor = None
        pending.consensus_label = None
        return result

    def has_pending_transfers(self) -> bool:
        return bool(self._pending_transfers)

    def _world_transfer_consensus(
        self,
        req: P2PKVTransferReqInput,
        local_result: P2PKVTransferReqOutput,
        phase: str,
    ) -> P2PKVTransferReqOutput:
        world_group = getattr(
            getattr(self.scheduler, "world_group", None), "cpu_group", None
        )
        if (
            world_group is None
            or not torch.distributed.is_available()
            or not torch.distributed.is_initialized()
        ):
            return local_result

        local_success = bool(
            local_result.success and not local_result.fallback_recompute
        )
        values = torch.tensor(
            [
                1 if local_success else 0,
                int(local_result.transferred_tokens) if local_success else 0,
            ],
            dtype=torch.int64,
            device="cpu",
        )
        torch.distributed.all_reduce(
            values,
            op=torch.distributed.ReduceOp.MIN,
            group=world_group,
        )
        if int(values[0].item()) == 0:
            return self._fail(
                req,
                f"{phase} failed on at least one model-parallel rank",
            )
        transferred_tokens = int(values[1].item())
        return P2PKVTransferReqOutput(
            success=True,
            message=local_result.message,
            source_url=req.source_url,
            target_url=req.target_url,
            matched_tokens=req.matched_tokens,
            transferred_tokens=transferred_tokens,
            fallback_recompute=False,
            experimental_limitations=self._limitations(),
        )

    def _world_phase_consensus(self, local_success: bool, phase: str) -> bool:
        world_group = getattr(
            getattr(self.scheduler, "world_group", None), "cpu_group", None
        )
        if (
            world_group is None
            or not torch.distributed.is_available()
            or not torch.distributed.is_initialized()
        ):
            return local_success
        value = torch.tensor(
            [1 if local_success else 0], dtype=torch.int64, device="cpu"
        )
        torch.distributed.all_reduce(
            value,
            op=torch.distributed.ReduceOp.MIN,
            group=world_group,
        )
        success = bool(int(value.item()))
        if not success:
            logger.warning("p2p_world_phase_failed: phase=%s", phase)
        return success

    def _sample_capacity_detail(self) -> bool:
        self._capacity_detail_counter += 1
        return (
            self._capacity_detail_counter == 1
            or self._capacity_detail_counter % 64 == 0
        )

    def transfer(self, req: P2PKVTransferReqInput) -> P2PKVTransferReqOutput:
        if req.p2p_source_send:
            try:
                local_result = self._source_send_via_sender(req)
            # This is the rank-wide fail-safe boundary: any sender failure must
            # participate in consensus instead of stranding the other ranks.
            except Exception as exc:  # noqa: BLE001
                local_result = self._fail(
                    req, f"source transfer raised before consensus: {exc}"
                )
            return self._world_transfer_consensus(req, local_result, "source transfer")
        return self._target_pull(req)

    def start_transfer(
        self, req: P2PKVTransferReqInput
    ) -> P2PKVTransferReqOutput | None:
        if self._unsafe_quarantine:
            return self._fail(
                req,
                "P2P transfer disabled after an uncertain transport timeout; "
                "restart the worker to reclaim quarantined KV safely",
            )
        if self._pending_transfers:
            return self._fail(req, "P2P transfer inflight limit reached")

        pending_or_output = (
            self._start_source_transfer(req)
            if req.p2p_source_send
            else self._start_target_transfer(req)
        )
        if isinstance(pending_or_output, P2PKVTransferReqOutput):
            return pending_or_output
        key = req.request_id or str(id(req))
        self._pending_transfers[key] = pending_or_output
        return None

    def progress_transfers(
        self,
    ) -> list[tuple[P2PKVTransferReqInput, P2PKVTransferReqOutput]]:
        completions = []
        for key, pending in list(self._pending_transfers.items()):
            try:
                output = (
                    self._progress_source_transfer(pending)
                    if pending.role == "source"
                    else self._progress_target_transfer(pending)
                )
            except Exception as exc:  # noqa: BLE001
                logger.exception(
                    "p2p_transfer_progress_failed: request_id=%s role=%s",
                    pending.req.request_id,
                    pending.role,
                )
                pending.state = P2PTransferState.FAILED
                if (
                    pending.role == "target"
                    and pending.source_future is not None
                    and (
                        not pending.source_future.done() or not pending.source_terminal
                    )
                ):
                    pending.quarantine_allocation = True
                output = self._fail(pending.req, f"P2P progress failed safely: {exc}")
            if output is None:
                continue
            try:
                if pending.role == "source":
                    self._cleanup_source_transfer(pending)
                else:
                    self._cleanup_target_transfer(pending)
            except Exception:  # noqa: BLE001
                logger.exception(
                    "p2p_transfer_cleanup_failed: request_id=%s role=%s",
                    pending.req.request_id,
                    pending.role,
                )
            self._pending_transfers.pop(key, None)
            completions.append((pending.req, output))
        return completions

    def _start_target_transfer(
        self, req: P2PKVTransferReqInput
    ) -> PendingP2PTransfer | P2PKVTransferReqOutput:
        kv_manager = self._kv_manager()
        if kv_manager is None:
            return self._fail(req, "target has no Prefill Mooncake KV manager")
        state_error = self._unsupported_state_error(kv_manager)
        if state_error:
            return self._fail(req, state_error)
        self._ensure_receiver_state(kv_manager)

        prefix_len = self._transferable_len(req)
        if prefix_len <= 0:
            return self._fail(req, "no page-aligned transferable prefix")
        target_match = self.scheduler.tree_cache.match_prefix(
            MatchPrefixParams(key=_radix_key(req.token_ids[:prefix_len]))
        )
        cached_tokens = self._align_available_len(
            req, len(target_match.device_indices), prefix_len
        )
        if cached_tokens >= prefix_len:
            return P2PKVTransferReqOutput(
                success=True,
                message="target prefix already cached",
                source_url=req.source_url,
                target_url=req.target_url,
                matched_tokens=req.matched_tokens,
                transferred_tokens=prefix_len,
                fallback_recompute=False,
                experimental_limitations=self._limitations(),
            )

        pair_key = self._canonical_pair_key(req)
        if not self._acquire_pair_gate(req, pair_key):
            return self._fail(req, "p2p pair already has an in-flight transfer")

        allocator = self.scheduler.token_to_kv_pool_allocator
        evict_from_tree_cache(self.scheduler.tree_cache, prefix_len)
        dst_kv = allocator.alloc(prefix_len)
        if dst_kv is None:
            self._release_pair_gate(req, pair_key)
            return self._fail(req, "target failed to allocate KV slots")

        allocation = _TargetAllocation(kv=dst_kv)
        try:
            dst_mamba = None
            if self._requires_mamba(kv_manager):
                mamba_allocator = getattr(
                    self.scheduler.req_to_token_pool, "mamba_allocator", None
                )
                if mamba_allocator is None:
                    raise RuntimeError("target has no Mamba slot allocator")
                dst_mamba = mamba_allocator.alloc(1)
                if dst_mamba is None:
                    raise RuntimeError("target failed to allocate Mamba slot")
                allocation.attach_mamba(dst_mamba)

            source_bootstrap_addr = req.source_bootstrap_addr
            if source_bootstrap_addr is None:
                raise RuntimeError("source bootstrap address was not resolved")
            if not kv_manager.try_ensure_parallel_info(
                source_bootstrap_addr, p2p_identical_layout=True
            ):
                raise RuntimeError(
                    f"target could not resolve source bootstrap "
                    f"{source_bootstrap_addr}"
                )
            info = kv_manager.prefill_info_table[source_bootstrap_addr]
            layout_error = self._validate_identical_layout(kv_manager, info)
            if layout_error:
                raise RuntimeError(layout_error)

            receiver_cls = get_kv_class(TransferBackend.MOONCAKE, KVClassType.RECEIVER)
            room = int(req.p2p_bootstrap_room or random.getrandbits(63))
            receiver = receiver_cls(kv_manager, source_bootstrap_addr, room)
            receiver.init(prefill_dp_rank=0)
            trigger_source = (
                kv_manager.kv_args.engine_rank == 0
                and getattr(kv_manager, "pp_rank", 0) == 0
            )
            return PendingP2PTransfer(
                req=req,
                role="target",
                state=P2PTransferState.RESERVED,
                deadline=time.monotonic() + self._TRANSFER_TIMEOUT_S,
                kv_manager=kv_manager,
                receiver=receiver,
                actual_len=prefix_len,
                prefix_len=prefix_len,
                allocation=allocation,
                pair_key=pair_key,
                trigger_source=trigger_source,
                source_payload={"success": True, "transferred_tokens": prefix_len},
                room=room,
            )
        except Exception as exc:
            abort = locals().get("receiver")
            if abort is not None and callable(getattr(abort, "abort", None)):
                abort.abort()
            allocation.cleanup(
                allocator,
                getattr(self.scheduler.req_to_token_pool, "mamba_allocator", None),
            )
            self._release_pair_gate(req, pair_key)
            return self._fail(req, f"target transfer setup failed: {exc}")

    def _request_source_transfer(
        self, source_req: P2PKVTransferReqInput
    ) -> tuple[dict[str, Any], str | None, bool]:
        try:
            response = requests.post(
                f"{source_req.source_url.rstrip('/')}/experimental/p2p_kv_transfer",
                json=_p2p_req_to_builtins(source_req),
                timeout=self._SOURCE_REQUEST_TIMEOUT_S,
            )
            if response.status_code != 200:
                return (
                    {},
                    (
                        f"source transfer failed: HTTP {response.status_code}: "
                        f"{response.text[:512]}"
                    ),
                    True,
                )
            payload = response.json()
            if not payload.get("success"):
                return (
                    payload,
                    f"source transfer failed: {payload.get('message', '')}",
                    True,
                )
            return payload, None, True
        except Exception as exc:  # noqa: BLE001
            return {}, f"source transfer request failed: {exc}", False

    def _progress_target_transfer(
        self, pending: PendingP2PTransfer
    ) -> P2PKVTransferReqOutput | None:
        req = pending.req
        timed_out = time.monotonic() >= pending.deadline

        if pending.state == P2PTransferState.RESERVED:
            poll = pending.receiver.poll()
            init_ok = self._progress_world_min(
                pending, "target-init", int(poll != KVPoll.Failed)
            )
            if init_ok is None:
                return None
            if init_ok == 0:
                pending.state = P2PTransferState.FAILED
                return self._fail(req, "target receiver bootstrap failed")

            try:
                page_indices = kv_to_page_indices(
                    pending.allocation.kv.to(dtype=torch.int32, copy=True),
                    pending.kv_manager.kv_args.page_size,
                )
                mamba_index = (
                    int(pending.allocation.mamba[0].item())
                    if pending.allocation.mamba is not None
                    else None
                )
                state_indices = self._state_indices_for_pages(
                    pending.kv_manager, page_indices, mamba_index
                )
                pending.receiver.send_metadata(
                    page_indices, aux_index=0, state_indices=state_indices
                )
                metadata_ok = pending.receiver.poll() != KVPoll.Failed
            except Exception as exc:  # noqa: BLE001
                pending.source_error = f"target receiver metadata failed: {exc}"
                metadata_ok = False

            pending.state = P2PTransferState.WAIT_SOURCE
            metadata_ready = self._progress_world_min(
                pending, "target-metadata", int(metadata_ok)
            )
            if metadata_ready is None:
                return None
            if metadata_ready == 0:
                pending.state = P2PTransferState.FAILED
                return self._fail(
                    req,
                    pending.source_error or "target receiver metadata consensus failed",
                )

        if (
            pending.state == P2PTransferState.WAIT_SOURCE
            and pending.consensus_label == "target-metadata"
        ):
            metadata_ready = self._progress_world_min(
                pending,
                "target-metadata",
                int(pending.source_error is None),
            )
            if metadata_ready is None:
                return None
            if metadata_ready == 0:
                pending.state = P2PTransferState.FAILED
                return self._fail(
                    req,
                    pending.source_error or "target receiver metadata consensus failed",
                )

        if (
            pending.state == P2PTransferState.WAIT_SOURCE
            and pending.source_future is None
            and pending.trigger_source
        ):
            source_req = P2PKVTransferReqInput(
                source_url=req.source_url,
                target_url=req.target_url,
                token_ids=req.token_ids[: pending.prefix_len],
                matched_tokens=pending.prefix_len,
                request_id=req.request_id,
                reason=req.reason,
                p2p_bootstrap_room=pending.room,
                p2p_source_send=True,
            )
            pending.source_terminal = False
            pending.source_future = self._http_executor.submit(
                self._request_source_transfer, source_req
            )

        if timed_out and not pending.abort_requested:
            pending.abort_requested = True
            abort = getattr(pending.receiver, "abort", None)
            if callable(abort):
                abort()

        if (
            pending.trigger_source
            and pending.source_future is not None
            and pending.source_future.done()
            and not pending.source_terminal
        ):
            (
                pending.source_payload,
                pending.source_error,
                pending.source_terminal,
            ) = pending.source_future.result()

        if pending.state == P2PTransferState.WAIT_SOURCE:
            if pending.trigger_source:
                if pending.source_future is None or not pending.source_future.done():
                    source_state = 1
                elif pending.source_error is None:
                    source_state = 2
                elif pending.source_terminal:
                    source_state = 0
                else:
                    source_state = -1
            else:
                source_state = 2

            source_state = self._progress_world_min(
                pending, "target-source", source_state
            )
            if source_state is None or source_state == 1:
                return None
            if source_state <= 0:
                pending.quarantine_allocation = source_state < 0
                pending.state = P2PTransferState.FAILED
                return self._fail(
                    req,
                    pending.source_error
                    or (
                        "target receiver timed out waiting for source"
                        if timed_out
                        else "source transfer failed"
                    ),
                )
            pending.state = P2PTransferState.TRANSFERRING

        if pending.state == P2PTransferState.TRANSFERRING:
            poll = pending.receiver.poll()
            poll_state = {
                KVPoll.Failed: 0,
                KVPoll.Success: 2,
            }.get(poll, 1)
            poll_state = self._progress_world_min(
                pending, "target-transfer", poll_state
            )
            if poll_state is None or poll_state == 1:
                return None
            if poll_state == 0:
                pending.state = P2PTransferState.FAILED
                return self._fail(req, "target receiver reported transfer failure")
            pending.state = P2PTransferState.CONSENSUS

        transferred_tokens = int(
            (pending.source_payload or {}).get("transferred_tokens") or 0
        )
        transferred_tokens = self._progress_world_min(
            pending, "target-tokens", transferred_tokens
        )
        if transferred_tokens is None:
            return None
        if transferred_tokens <= 0 or transferred_tokens > pending.prefix_len:
            pending.state = P2PTransferState.FAILED
            return self._fail(
                req, f"invalid transferred token count {transferred_tokens}"
            )

        result = P2PKVTransferReqOutput(
            success=True,
            message="p2p Mooncake receiver transfer succeeded",
            source_url=req.source_url,
            target_url=req.target_url,
            matched_tokens=req.matched_tokens,
            transferred_tokens=transferred_tokens,
            fallback_recompute=False,
            experimental_limitations=self._limitations(),
        )

        pending.allocation.trim_kv(
            result.transferred_tokens,
            self.scheduler.token_to_kv_pool_allocator,
        )
        self._register_target_prefix(req, pending.allocation, result.transferred_tokens)
        pending.state = P2PTransferState.COMMIT
        logger.info(
            "p2p_target_receiver_done: request_id=%s transferred_tokens=%s "
            "fallback_recompute=%s",
            req.request_id,
            result.transferred_tokens,
            result.fallback_recompute,
        )
        return result

    def _cleanup_target_transfer(self, pending: PendingP2PTransfer) -> None:
        if pending.cleanup_done:
            return
        pending.cleanup_done = True
        if pending.state == P2PTransferState.FAILED:
            abort = getattr(pending.receiver, "abort", None)
            if callable(abort):
                abort()
        if pending.quarantine_allocation:
            self._unsafe_quarantine = True
            logger.error(
                "p2p_target_allocation_quarantined: request_id=%s tokens=%s "
                "reason=source_transport_not_proven_terminal",
                pending.req.request_id,
                len(pending.allocation.kv),
            )
        else:
            pending.allocation.cleanup(
                self.scheduler.token_to_kv_pool_allocator,
                getattr(self.scheduler.req_to_token_pool, "mamba_allocator", None),
            )
        if pending.pair_key is not None:
            self._release_pair_gate(pending.req, pending.pair_key)

    def _start_source_transfer(
        self, req: P2PKVTransferReqInput
    ) -> PendingP2PTransfer | P2PKVTransferReqOutput:
        kv_manager = self._kv_manager()
        if kv_manager is None:
            return self._fail(req, "source has no Prefill Mooncake KV manager")
        state_error = self._unsupported_state_error(kv_manager)
        if state_error:
            return self._fail(req, state_error)

        prefix_len = self._transferable_len(req)
        match = self.scheduler.tree_cache.match_prefix(
            MatchPrefixParams(key=_radix_key(req.token_ids[:prefix_len]))
        )
        match = self._restore_source_prefix_from_hicache(req, match, prefix_len)
        actual_len = self._align_available_len(
            req, len(match.device_indices), prefix_len
        )
        if actual_len <= 0:
            return self._fail(
                req,
                f"source cache miss: matched {len(match.device_indices)} "
                f"< requested {prefix_len}",
            )
        mamba_index = self._cached_mamba_index_if_needed(req, kv_manager, match)
        if isinstance(mamba_index, P2PKVTransferReqOutput):
            return mamba_index

        sender_cls = get_kv_class(TransferBackend.MOONCAKE, KVClassType.SENDER)
        sender = sender_cls(
            mgr=kv_manager,
            bootstrap_addr="",
            bootstrap_room=int(req.p2p_bootstrap_room),
            dest_tp_ranks=[kv_manager.attn_tp_rank],
            pp_rank=kv_manager.pp_rank,
            force_cp_rank_transfer=True,
        )
        num_pages = len(
            kv_to_page_indices(
                torch.zeros(actual_len, dtype=torch.int32),
                kv_manager.kv_args.page_size,
            )
        )
        sender.init(num_pages, aux_index=0)
        lock_params = self.scheduler.tree_cache.inc_lock_ref(match.last_device_node)
        return PendingP2PTransfer(
            req=req,
            role="source",
            state=P2PTransferState.RESERVED,
            deadline=time.monotonic() + self._TRANSFER_TIMEOUT_S,
            kv_manager=kv_manager,
            sender=sender,
            match=match,
            lock_params=lock_params,
            actual_len=actual_len,
        )

    def _progress_source_transfer(
        self, pending: PendingP2PTransfer
    ) -> P2PKVTransferReqOutput | None:
        req = pending.req
        timed_out = time.monotonic() >= pending.deadline
        if timed_out and not pending.abort_requested:
            abort = getattr(pending.sender, "abort", None)
            if callable(abort):
                abort()
            pending.abort_requested = True

        poll = pending.sender.poll()

        if pending.state == P2PTransferState.RESERVED:
            source_state = (
                0
                if timed_out or poll == KVPoll.Failed
                else 2 if poll == KVPoll.WaitingForInput else 1
            )
            source_state = self._progress_world_min(
                pending, "source-ready", source_state
            )
            if source_state is None or source_state == 1:
                return None
            if source_state == 0:
                pending.state = P2PTransferState.FAILED
                return self._fail(
                    req,
                    (
                        "source sender timed out waiting for target metadata"
                        if timed_out
                        else "source sender bootstrap failed before transfer"
                    ),
                )
            src_pages = kv_to_page_indices(
                pending.match.device_indices[: pending.actual_len].to(
                    dtype=torch.int32, copy=True
                ),
                pending.kv_manager.kv_args.page_size,
            )
            mamba_index = self._cached_mamba_index_if_needed(
                req, pending.kv_manager, pending.match
            )
            if isinstance(mamba_index, P2PKVTransferReqOutput):
                pending.state = P2PTransferState.FAILED
                return mamba_index
            state_indices = self._state_indices_for_pages(
                pending.kv_manager, src_pages, mamba_index
            )
            pending.sender.send(src_pages, state_indices=state_indices)
            pending.state = P2PTransferState.TRANSFERRING
            return None

        if pending.state == P2PTransferState.TRANSFERRING:
            transfer_state = (
                0
                if timed_out or poll == KVPoll.Failed
                else 2 if poll == KVPoll.Success else 1
            )
            transfer_state = self._progress_world_min(
                pending, "source-transfer", transfer_state
            )
            if transfer_state is None or transfer_state == 1:
                return None
            if transfer_state == 0:
                pending.state = P2PTransferState.FAILED
                return self._fail(
                    req,
                    (
                        "source sender timed out waiting for transfer"
                        if timed_out
                        else "source sender reported transfer failure"
                    ),
                )
            pending.state = P2PTransferState.COMMIT
            return P2PKVTransferReqOutput(
                success=True,
                message="source sent KV and optional state via Mooncake sender",
                source_url=req.source_url,
                target_url=req.target_url,
                matched_tokens=req.matched_tokens,
                transferred_tokens=pending.actual_len,
                fallback_recompute=False,
                experimental_limitations=self._limitations(),
            )
        return None

    def _cleanup_source_transfer(self, pending: PendingP2PTransfer) -> None:
        if pending.cleanup_done:
            return
        pending.cleanup_done = True
        if pending.lock_params is not None:
            self.scheduler.tree_cache.dec_lock_ref(
                pending.match.last_device_node,
                pending.lock_params.to_dec_params(),
            )

    def _kv_manager(self):
        queue = getattr(self.scheduler, "disagg_prefill_bootstrap_queue", None)
        return getattr(queue, "kv_manager", None)

    def _limitations(self):
        return [
            "experimental_prefill_to_prefill_mooncake",
            "identical_tp_pp_layout_supported",
            "identical_distributed_layout_supported",
            "tp_pp_mismatch_falls_back_to_recompute",
            "same_model_same_page_size_required",
            "failure_falls_back_to_recompute",
        ]

    def _fail(self, req: P2PKVTransferReqInput, message: str) -> P2PKVTransferReqOutput:
        kv_manager = self._kv_manager()
        logger.warning(
            "p2p_transfer_fallback: request_id=%s reason=%s source=%s target=%s "
            "matched_tokens=%s state_types=%s message=%s",
            req.request_id,
            req.reason,
            req.source_url,
            req.target_url,
            req.matched_tokens,
            self._state_type_names(kv_manager) if kv_manager is not None else None,
            message,
        )
        return P2PKVTransferReqOutput(
            success=False,
            message=message,
            source_url=req.source_url,
            target_url=req.target_url,
            matched_tokens=req.matched_tokens,
            transferred_tokens=0,
            fallback_recompute=True,
            experimental_limitations=self._limitations(),
        )

    def _transferable_len(self, req: P2PKVTransferReqInput) -> int:
        page_size = getattr(self.scheduler.tree_cache, "page_size", 1)
        prefix_len = min(req.matched_tokens, len(req.token_ids))
        prefix_len = page_align_floor(prefix_len, page_size)
        mamba_chunk = getattr(self.scheduler.tree_cache, "mamba_cache_chunk_size", 1)
        if mamba_chunk > 1:
            prefix_len = page_align_floor(prefix_len, mamba_chunk)
        return prefix_len

    def _endpoint_sort_key(self, url: str):
        host_port = url.rstrip("/").split("://", 1)[-1].split("/", 1)[0]
        host, _, port = host_port.partition(":")
        host_key = tuple(
            (0, int(part)) if part.isdigit() else (1, part) for part in host.split(".")
        )
        return (host_key, int(port) if port.isdigit() else 0, url.rstrip("/"))

    def _canonical_pair_key(self, req: P2PKVTransferReqInput):
        left = req.source_url.rstrip("/")
        right = req.target_url.rstrip("/")
        return tuple(sorted((left, right), key=self._endpoint_sort_key))

    def _acquire_pair_gate(self, req: P2PKVTransferReqInput, pair_key) -> bool:
        now = time.monotonic()
        deadline = now + self._PAIR_GATE_TTL_S
        with self._PAIR_GATE_LOCK:
            existing = self._PAIR_GATE_DEADLINES.get(pair_key)
            if existing is not None and existing > now:
                logger.info(
                    "p2p_pair_gate_busy: request_id=%s pair=%s existing_ttl=%.3f",
                    req.request_id,
                    pair_key,
                    existing - now,
                )
                return False
            self._PAIR_GATE_DEADLINES[pair_key] = deadline
        logger.info(
            "p2p_pair_gate_acquired: request_id=%s pair=%s ttl=%.3f",
            req.request_id,
            pair_key,
            self._PAIR_GATE_TTL_S,
        )
        return True

    def _release_pair_gate(self, req: P2PKVTransferReqInput, pair_key):
        with self._PAIR_GATE_LOCK:
            self._PAIR_GATE_DEADLINES.pop(pair_key, None)
        logger.info(
            "p2p_pair_gate_released: request_id=%s pair=%s", req.request_id, pair_key
        )

    def _target_pull(self, req: P2PKVTransferReqInput) -> P2PKVTransferReqOutput:
        kv_manager = self._kv_manager()
        if kv_manager is None:
            return self._fail(req, "target has no Prefill Mooncake KV manager")
        state_error = self._unsupported_state_error(kv_manager)
        if state_error:
            return self._fail(req, state_error)
        self._ensure_receiver_state(kv_manager)

        prefix_len = self._transferable_len(req)
        if prefix_len <= 0:
            return self._fail(req, "no page-aligned transferable prefix")
        target_match = self.scheduler.tree_cache.match_prefix(
            MatchPrefixParams(key=_radix_key(req.token_ids[:prefix_len]))
        )
        cached_tokens = self._align_available_len(
            req, len(target_match.device_indices), prefix_len
        )
        if cached_tokens >= prefix_len:
            logger.info(
                "p2p_target_prefix_already_cached: request_id=%s source=%s "
                "target=%s tokens=%s verify_match=%s",
                req.request_id,
                req.source_url,
                req.target_url,
                prefix_len,
                len(target_match.device_indices),
            )
            return P2PKVTransferReqOutput(
                success=True,
                message="target prefix already cached",
                source_url=req.source_url,
                target_url=req.target_url,
                matched_tokens=req.matched_tokens,
                transferred_tokens=prefix_len,
                fallback_recompute=False,
                experimental_limitations=self._limitations(),
            )
        pair_key = self._canonical_pair_key(req)
        if not self._acquire_pair_gate(req, pair_key):
            return self._fail(req, "p2p pair already has an in-flight transfer")
        logger.info(
            "p2p_target_plan: request_id=%s reason=%s source=%s target=%s "
            "matched_tokens=%s transferable_tokens=%s page_size=%s state_types=%s "
            "requires_mamba=%s",
            req.request_id,
            req.reason,
            req.source_url,
            req.target_url,
            req.matched_tokens,
            prefix_len,
            kv_manager.kv_args.page_size,
            self._state_type_names(kv_manager),
            self._requires_mamba(kv_manager),
        )

        allocator = self.scheduler.token_to_kv_pool_allocator
        capacity = self.scheduler.tree_cache.available_and_evictable_str().strip()
        capacity_log = logger.info if self._sample_capacity_detail() else logger.debug
        capacity_log(
            "p2p_target_capacity: request_id=%s requested_tokens=%s %s",
            req.request_id,
            prefix_len,
            capacity.replace("\n", "; "),
        )
        evict_from_tree_cache(self.scheduler.tree_cache, prefix_len)
        capacity_after_evict = (
            self.scheduler.tree_cache.available_and_evictable_str().strip()
        )
        capacity_log(
            "p2p_target_capacity_after_evict: request_id=%s %s",
            req.request_id,
            capacity_after_evict.replace("\n", "; "),
        )
        dst_kv = allocator.alloc(prefix_len)
        if dst_kv is None:
            self._release_pair_gate(req, pair_key)
            return self._fail(req, "target failed to allocate KV slots")

        allocation = _TargetAllocation(kv=dst_kv)
        dst_mamba = None
        try:
            if self._requires_mamba(kv_manager):
                mamba_allocator = getattr(
                    self.scheduler.req_to_token_pool, "mamba_allocator", None
                )
                if mamba_allocator is None:
                    return self._fail(req, "target has no Mamba slot allocator")
                dst_mamba = mamba_allocator.alloc(1)
                if dst_mamba is None:
                    return self._fail(req, "target failed to allocate Mamba slot")
                allocation.attach_mamba(dst_mamba)
            logger.info(
                "p2p_target_allocated: request_id=%s dst_kv_slots=%s "
                "dst_mamba_slots=%s state_types=%s",
                req.request_id,
                len(dst_kv),
                len(dst_mamba) if dst_mamba is not None else 0,
                self._state_type_names(kv_manager),
            )

            receiver_result = self._target_pull_via_receiver(
                req, kv_manager, dst_kv, dst_mamba, prefix_len
            )
            if receiver_result is None:
                receiver_result = self._fail(
                    req,
                    "source bootstrap unavailable for Mooncake receiver path; "
                    "direct P2P transfer is disabled",
                )
            if receiver_result.success:
                transferred_tokens = receiver_result.transferred_tokens
                if transferred_tokens <= 0 or transferred_tokens > len(allocation.kv):
                    receiver_result = self._fail(
                        req,
                        f"invalid transferred token count {transferred_tokens}",
                    )
            receiver_result = self._world_transfer_consensus(
                req, receiver_result, "target transfer"
            )
            if receiver_result.success:
                transferred_tokens = receiver_result.transferred_tokens
                allocation.trim_kv(transferred_tokens, allocator)
                self._register_target_prefix(req, allocation, transferred_tokens)
                logger.info(
                    "p2p_target_receiver_done: request_id=%s transferred_tokens=%s "
                    "fallback_recompute=%s",
                    req.request_id,
                    transferred_tokens,
                    receiver_result.fallback_recompute,
                )
                return receiver_result
            return receiver_result

        except _P2PCacheIntegrityError:
            raise
        # The target must fall back to recompute for transport, HTTP, allocator,
        # or cache failures rather than fail the user request.
        except Exception as e:  # noqa: BLE001
            return self._fail(req, f"target transfer failed: {e}")
        finally:
            allocation.cleanup(
                allocator,
                getattr(self.scheduler.req_to_token_pool, "mamba_allocator", None),
            )
            self._release_pair_gate(req, pair_key)

    @staticmethod
    def _mamba_component_index(kv_manager) -> int | None:
        for i, state_type in enumerate(kv_manager.kv_args.state_types):
            if state_type == StateType.MAMBA:
                return i
        return None

    @classmethod
    def _requires_mamba(cls, kv_manager) -> bool:
        return cls._mamba_component_index(kv_manager) is not None

    @staticmethod
    def _state_type_name(state_type) -> str:
        return str(getattr(state_type, "value", state_type))

    @classmethod
    def _state_type_names(cls, kv_manager):
        return [
            cls._state_type_name(state_type)
            for state_type in getattr(kv_manager.kv_args, "state_types", [])
        ]

    @classmethod
    def _unsupported_state_error(cls, kv_manager) -> str | None:
        supported = {StateType.MAMBA, StateType.DSA, StateType.MINIMAX_INDEX_K}
        unsupported = [
            cls._state_type_name(state_type)
            for state_type in getattr(kv_manager.kv_args, "state_types", [])
            if state_type not in supported
        ]
        if unsupported:
            return (
                "remote KV transfer skipped: unsupported P2P state types "
                f"{unsupported}"
            )
        return None

    @staticmethod
    def _node_mamba_value(node):
        mamba_value = getattr(node, "mamba_value", None)
        if mamba_value is not None:
            return mamba_value
        component_data = getattr(node, "component_data", {})
        if hasattr(component_data, "items"):
            entries = component_data.items()
        else:
            entries = (
                (component_type, component_data[component_type])
                for component_type in getattr(node, "tree_components", ())
            )
        for component_type, data in entries:
            if getattr(component_type, "is_mamba", False):
                return getattr(data, "value", None)
        return None

    def _cached_mamba_index_if_needed(self, req, kv_manager, match):
        if not self._requires_mamba(kv_manager):
            return None
        mamba_value = self._node_mamba_value(match.last_device_node)
        if mamba_value is None or len(mamba_value) != 1:
            return self._fail(req, "source cache has no transferable Mamba state")
        return int(mamba_value[0].item())

    @staticmethod
    def _as_int_list(indices):
        if indices is None:
            return []
        if hasattr(indices, "detach"):
            indices = indices.detach().cpu().tolist()
        elif hasattr(indices, "tolist"):
            indices = indices.tolist()
        return [int(x) for x in indices]

    def _state_indices_for_pages(
        self, kv_manager, page_indices, mamba_index: int | None
    ):
        page_indices = self._as_int_list(page_indices)
        state_indices = []
        for state_type in kv_manager.kv_args.state_types:
            if state_type == StateType.MAMBA:
                if mamba_index is None:
                    raise ValueError(
                        "Mamba state is required but no Mamba index was provided"
                    )
                state_indices.append([int(mamba_index)])
            elif state_type in (StateType.DSA, StateType.MINIMAX_INDEX_K):
                state_indices.append(page_indices)
            else:
                state_indices.append([])
        return state_indices

    def _align_available_len(
        self, req: P2PKVTransferReqInput, matched_len: int, requested_len: int
    ) -> int:
        n = min(matched_len, requested_len, len(req.token_ids))
        page_size = getattr(self.scheduler.tree_cache, "page_size", 1)
        n = page_align_floor(n, page_size)
        mamba_chunk = getattr(self.scheduler.tree_cache, "mamba_cache_chunk_size", 1)
        if mamba_chunk > 1:
            n = page_align_floor(n, mamba_chunk)
        return n

    def _register_target_prefix(
        self,
        req: P2PKVTransferReqInput,
        allocation: _TargetAllocation,
        transferred_tokens: int,
    ) -> _TargetRegistrationResult:
        allocation.transfer_to_insert()
        try:
            insert_result = self.scheduler.tree_cache.insert(
                InsertParams(
                    key=_radix_key(req.token_ids[:transferred_tokens]),
                    value=allocation.kv.to(dtype=torch.int64, copy=True),
                    mamba_value=(
                        allocation.mamba.to(dtype=torch.int64, copy=True)
                        if allocation.mamba is not None
                        else None
                    ),
                    prev_prefix_len=0,
                )
            )
            duplicate_tokens = int(insert_result.prefix_len or 0)
            if not 0 <= duplicate_tokens <= transferred_tokens:
                raise _P2PCacheIntegrityError(
                    "invalid insert prefix_len="
                    f"{duplicate_tokens} transferred={transferred_tokens}"
                )
            duplicate_kv_handled_by_cache = bool(
                getattr(insert_result, "duplicate_kv_handled_by_cache", False)
            )
            duplicate_kv_freed_by_p2p = (
                duplicate_tokens > 0 and not duplicate_kv_handled_by_cache
            )
            if duplicate_kv_freed_by_p2p:
                self.scheduler.token_to_kv_pool_allocator.free(
                    allocation.kv[:duplicate_tokens]
                )
                logger.info(
                    "p2p_target_freed_duplicate_prefix: request_id=%s source=%s "
                    "target=%s duplicate_tokens=%s transferred_tokens=%s",
                    req.request_id,
                    req.source_url,
                    req.target_url,
                    duplicate_tokens,
                    transferred_tokens,
                )
            mamba_exist = bool(getattr(insert_result, "mamba_exist", False))
            cache_accepted_mamba = allocation.mamba is not None and not mamba_exist
            if allocation.mamba is not None and mamba_exist:
                self.scheduler.req_to_token_pool.mamba_allocator.free(allocation.mamba)
            allocation.commit()
        except Exception as exc:
            logger.exception(
                "p2p_target_cache_integrity_error: request_id=%s phase=%s",
                req.request_id,
                allocation.phase.value,
            )
            raise _P2PCacheIntegrityError(str(exc)) from exc

        try:
            verify_match = self.scheduler.tree_cache.match_prefix(
                MatchPrefixParams(key=_radix_key(req.token_ids[:transferred_tokens]))
            )
            logger.info(
                "p2p_transfer_success target registered prefix: source=%s "
                "target=%s tokens=%s verify_match=%s",
                req.source_url,
                req.target_url,
                transferred_tokens,
                len(verify_match.device_indices),
            )
        except Exception:
            logger.exception(
                "p2p_target_post_commit_verify_failed: request_id=%s tokens=%s",
                req.request_id,
                transferred_tokens,
            )
        return _TargetRegistrationResult(
            committed_tokens=transferred_tokens,
            duplicate_kv_owner=(
                "none"
                if duplicate_tokens == 0
                else "cache" if duplicate_kv_handled_by_cache else "p2p"
            ),
            duplicate_kv_freed_by_p2p=duplicate_kv_freed_by_p2p,
            cache_accepted_mamba=cache_accepted_mamba,
        )

    def _target_pull_via_receiver(
        self, req: P2PKVTransferReqInput, kv_manager, dst_kv, dst_mamba, prefix_len: int
    ) -> P2PKVTransferReqOutput | None:
        source_bootstrap_addr = (
            req.source_bootstrap_addr or self._source_bootstrap_addr(req.source_url)
        )
        if source_bootstrap_addr is None:
            return None
        if not kv_manager.try_ensure_parallel_info(
            source_bootstrap_addr, p2p_identical_layout=True
        ):
            return self._fail(
                req,
                f"target could not resolve source bootstrap {source_bootstrap_addr}",
            )
        info = kv_manager.prefill_info_table[source_bootstrap_addr]
        layout_error = self._validate_identical_layout(kv_manager, info)
        if layout_error:
            return self._fail(req, layout_error)

        receiver_cls = get_kv_class(TransferBackend.MOONCAKE, KVClassType.RECEIVER)
        room = int(req.p2p_bootstrap_room or random.getrandbits(63))
        logger.info(
            "p2p_target_receiver_start: room=%s rank=%s source_bootstrap=%s source=%s target=%s prefix_len=%s",
            room,
            kv_manager.kv_args.engine_rank,
            source_bootstrap_addr,
            req.source_url,
            req.target_url,
            prefix_len,
        )
        receiver = receiver_cls(kv_manager, source_bootstrap_addr, room)
        receiver.init(prefill_dp_rank=0)
        initial_poll = self._poll_receiver_consensus(receiver)
        logger.info(
            "p2p_target_receiver_init: room=%s rank=%s poll=%s",
            room,
            kv_manager.kv_args.engine_rank,
            initial_poll,
        )
        if initial_poll == KVPoll.Failed:
            return self._fail(req, "target receiver bootstrap failed")

        page_indices = kv_to_page_indices(
            dst_kv.to(dtype=torch.int32, copy=True),
            kv_manager.kv_args.page_size,
        )
        mamba_index = int(dst_mamba[0].item()) if dst_mamba is not None else None
        state_indices = self._state_indices_for_pages(
            kv_manager, page_indices, mamba_index
        )
        metadata_error = None
        try:
            receiver.send_metadata(
                page_indices, aux_index=0, state_indices=state_indices
            )
        # Backend implementations can raise different exception types; convert
        # all of them into the same rank-consensus failure result.
        except Exception as exc:  # noqa: BLE001
            metadata_error = f"target receiver metadata failed: {exc}"
        metadata_poll = self._poll_receiver_consensus(
            receiver, force_failed=metadata_error is not None
        )
        metadata_ready = metadata_poll != KVPoll.Failed
        if not self._world_phase_consensus(metadata_ready, "target metadata"):
            return self._fail(
                req, metadata_error or "target receiver metadata consensus failed"
            )
        trigger_source = (
            kv_manager.kv_args.engine_rank == 0
            and getattr(kv_manager, "pp_rank", 0) == 0
        )
        logger.info(
            "p2p_target_metadata_sent: request_id=%s room=%s rank=%s pages=%s "
            "state_types=%s state_indices=%s trigger_source=%s",
            req.request_id,
            room,
            kv_manager.kv_args.engine_rank,
            len(page_indices),
            self._state_type_names(kv_manager),
            state_indices,
            trigger_source,
        )

        payload = {"success": True, "transferred_tokens": prefix_len}
        source_error = None
        if trigger_source:
            # Let all target TP ranks publish receiver metadata before rank 0
            # asks the source TP ranks to enter the same Mooncake room.
            logger.info(
                "p2p_target_source_trigger_barrier: room=%s rank=%s sleep_sec=1.0",
                room,
                kv_manager.kv_args.engine_rank,
            )
            time.sleep(1.0)
            source_req = P2PKVTransferReqInput(
                source_url=req.source_url,
                target_url=req.target_url,
                token_ids=req.token_ids[:prefix_len],
                matched_tokens=prefix_len,
                request_id=req.request_id,
                reason=req.reason,
                p2p_bootstrap_room=room,
                p2p_source_send=True,
            )
            try:
                response = requests.post(
                    f"{req.source_url.rstrip('/')}/experimental/p2p_kv_transfer",
                    json=_p2p_req_to_builtins(source_req),
                    timeout=self._SOURCE_REQUEST_TIMEOUT_S,
                )
                if response.status_code != 200:
                    source_error = (
                        f"source transfer failed: HTTP {response.status_code}: "
                        f"{response.text[:512]}"
                    )
                else:
                    payload = response.json()
                    if not payload.get("success"):
                        source_error = (
                            f"source transfer failed: {payload.get('message', '')}"
                        )
            # Requests may raise transport, decoding, or application errors.
            except Exception as exc:  # noqa: BLE001
                source_error = f"source transfer request failed: {exc}"

        # All target ranks enter this collective after rank 0's source request.
        # A local trigger failure is reduced to Failed so every rank cleans up.
        poll = self._poll_receiver_consensus(
            receiver, force_failed=source_error is not None
        )
        if poll == KVPoll.Failed:
            return self._fail(
                req,
                source_error or "target receiver reported transfer failure",
            )

        transferred_tokens = self._min_target_consensus(
            int(payload.get("transferred_tokens") or 0)
        )
        if transferred_tokens <= 0:
            return self._fail(req, "source transfer returned no transferred tokens")

        deadline = time.monotonic() + self._TRANSFER_TIMEOUT_S
        while True:
            if poll == KVPoll.Success:
                return P2PKVTransferReqOutput(
                    success=True,
                    message="p2p Mooncake receiver transfer succeeded",
                    source_url=req.source_url,
                    target_url=req.target_url,
                    matched_tokens=req.matched_tokens,
                    transferred_tokens=transferred_tokens,
                    fallback_recompute=False,
                    experimental_limitations=self._limitations(),
                )
            timed_out = time.monotonic() >= deadline
            poll = self._poll_receiver_consensus(receiver, force_failed=timed_out)
            if poll == KVPoll.Failed:
                return self._fail(
                    req,
                    (
                        "target receiver timed out waiting for transfer"
                        if timed_out
                        else "target receiver reported transfer failure"
                    ),
                )
            time.sleep(0.001)

    def _restore_source_prefix_from_hicache(
        self, req: P2PKVTransferReqInput, match, prefix_len: int
    ):
        """Synchronously restore an L2-only source prefix before Mooncake reads it."""
        tree_cache = self.scheduler.tree_cache
        device_tokens = len(match.device_indices)
        host_tokens = int(getattr(match, "host_hit_length", 0) or 0)
        covered_tokens = self._align_available_len(
            req, device_tokens + host_tokens, prefix_len
        )
        if host_tokens <= 0 or covered_tokens < prefix_len:
            return match

        required = (
            "load_back",
            "ready_to_load_host_cache",
            "loading_check",
            "cache_controller",
        )
        if any(not hasattr(tree_cache, name) for name in required):
            logger.warning(
                "p2p_source_hicache_load_back_unavailable: request_id=%s "
                "device_tokens=%s host_tokens=%s",
                req.request_id,
                device_tokens,
                host_tokens,
            )
            return match

        best_match_node = getattr(match, "best_match_node", None)
        if best_match_node is None:
            return match

        logger.info(
            "p2p_source_hicache_load_back_start: request_id=%s "
            "device_tokens=%s host_tokens=%s requested_tokens=%s",
            req.request_id,
            device_tokens,
            host_tokens,
            prefix_len,
        )
        try:
            queued = tree_cache.load_back(best_match_node, req=None)
            if queued is None or queued is False:
                logger.warning(
                    "p2p_source_hicache_load_back_rejected: request_id=%s",
                    req.request_id,
                )
                return match

            consumer_index = tree_cache.ready_to_load_host_cache()
            if not isinstance(consumer_index, int) or consumer_index < 0:
                logger.warning(
                    "p2p_source_hicache_load_back_not_started: request_id=%s "
                    "consumer_index=%s",
                    req.request_id,
                    consumer_index,
                )
                return match

            finish_event = tree_cache.cache_controller.layer_done_counter.events[
                consumer_index
            ].finish_event
            finish_event.synchronize()
            tree_cache.loading_check()
        except Exception:
            logger.exception(
                "p2p_source_hicache_load_back_failed: request_id=%s",
                req.request_id,
            )
            return match

        rematch = tree_cache.match_prefix(
            MatchPrefixParams(key=_radix_key(req.token_ids[:prefix_len]))
        )
        logger.info(
            "p2p_source_hicache_load_back_done: request_id=%s "
            "device_match_tokens=%s host_hit_tokens=%s mamba_host_hit=%s",
            req.request_id,
            len(rematch.device_indices),
            getattr(rematch, "host_hit_length", 0),
            getattr(rematch, "mamba_host_hit_length", 0),
        )
        return rematch

    def _source_send_via_sender(
        self, req: P2PKVTransferReqInput
    ) -> P2PKVTransferReqOutput:
        kv_manager = self._kv_manager()
        if kv_manager is None:
            return self._fail(req, "source has no Prefill Mooncake KV manager")
        state_error = self._unsupported_state_error(kv_manager)
        if state_error:
            return self._fail(req, state_error)
        room = int(req.p2p_bootstrap_room)
        prefix_len = self._transferable_len(req)
        match = self.scheduler.tree_cache.match_prefix(
            MatchPrefixParams(key=_radix_key(req.token_ids[:prefix_len]))
        )
        logger.info(
            "p2p_source_cache_match: request_id=%s requested_tokens=%s "
            "device_match_tokens=%s host_hit_tokens=%s mamba_host_hit=%s",
            req.request_id,
            prefix_len,
            len(match.device_indices),
            getattr(match, "host_hit_length", 0),
            getattr(match, "mamba_host_hit_length", 0),
        )
        match = self._restore_source_prefix_from_hicache(req, match, prefix_len)
        actual_len = self._align_available_len(
            req, len(match.device_indices), prefix_len
        )
        if actual_len <= 0:
            return self._fail(
                req,
                f"source cache miss: matched {len(match.device_indices)} < requested {prefix_len}",
            )
        mamba_index = self._cached_mamba_index_if_needed(req, kv_manager, match)
        if isinstance(mamba_index, P2PKVTransferReqOutput):
            return mamba_index
        logger.info(
            "p2p_source_sender_plan: request_id=%s reason=%s room=%s source=%s "
            "target=%s requested_tokens=%s cache_match_tokens=%s transfer_tokens=%s "
            "state_types=%s requires_mamba=%s",
            req.request_id,
            req.reason,
            room,
            req.source_url,
            req.target_url,
            prefix_len,
            len(match.device_indices),
            actual_len,
            self._state_type_names(kv_manager),
            self._requires_mamba(kv_manager),
        )

        sender_cls = get_kv_class(TransferBackend.MOONCAKE, KVClassType.SENDER)
        sender = sender_cls(
            mgr=kv_manager,
            bootstrap_addr="",
            bootstrap_room=room,
            dest_tp_ranks=[kv_manager.attn_tp_rank],
            pp_rank=kv_manager.pp_rank,
            force_cp_rank_transfer=True,
        )
        num_pages = len(
            kv_to_page_indices(
                torch.zeros(actual_len, dtype=torch.int32),
                kv_manager.kv_args.page_size,
            )
        )
        sender.init(num_pages, aux_index=0)
        logger.info(
            "p2p_source_sender_start: room=%s rank=%s actual_len=%s num_pages=%s",
            room,
            kv_manager.kv_args.engine_rank,
            actual_len,
            num_pages,
        )
        deadline = time.monotonic() + self._TRANSFER_TIMEOUT_S
        last_log = 0.0
        while time.monotonic() < deadline:
            poll = sender.poll()
            if poll == KVPoll.WaitingForInput:
                logger.info(
                    "p2p_source_sender_waiting_for_input: room=%s rank=%s transfer_info_sessions=%s",
                    room,
                    kv_manager.kv_args.engine_rank,
                    list(
                        getattr(kv_manager, "transfer_infos", {}).get(room, {}).keys()
                    ),
                )
                break
            if poll == KVPoll.Failed:
                return self._fail(req, "source sender bootstrap failed before transfer")
            now = time.monotonic()
            if now - last_log > 5:
                last_log = now
                logger.info(
                    "p2p_source_sender_wait: room=%s rank=%s poll=%s transfer_info_rooms=%s",
                    room,
                    kv_manager.kv_args.engine_rank,
                    poll,
                    list(getattr(kv_manager, "transfer_infos", {}).keys()),
                )
            time.sleep(0.001)
        else:
            return self._fail(
                req, "source sender timed out waiting for target metadata"
            )

        lock_params = self.scheduler.tree_cache.inc_lock_ref(match.last_device_node)
        try:
            src_pages = kv_to_page_indices(
                match.device_indices[:actual_len]
                .to(dtype=torch.int32, copy=True)
                .detach(),
                kv_manager.kv_args.page_size,
            )
            state_indices = self._state_indices_for_pages(
                kv_manager, src_pages, mamba_index
            )
            logger.info(
                "p2p_source_sender_send: request_id=%s room=%s tokens=%s "
                "src_pages=%s state_types=%s state_indices=%s",
                req.request_id,
                room,
                actual_len,
                len(src_pages),
                self._state_type_names(kv_manager),
                state_indices,
            )
            sender.send(src_pages, state_indices=state_indices)
            deadline = time.monotonic() + self._TRANSFER_TIMEOUT_S
            while time.monotonic() < deadline:
                poll = sender.poll()
                if poll == KVPoll.Success:
                    logger.info(
                        "p2p_transfer_success source sent prefix via Mooncake sender: source=%s target=%s tokens=%s",
                        req.source_url,
                        req.target_url,
                        actual_len,
                    )
                    return P2PKVTransferReqOutput(
                        success=True,
                        message="source sent KV and optional state via Mooncake sender",
                        source_url=req.source_url,
                        target_url=req.target_url,
                        matched_tokens=req.matched_tokens,
                        transferred_tokens=actual_len,
                        fallback_recompute=False,
                        experimental_limitations=self._limitations(),
                    )
                if poll == KVPoll.Failed:
                    return self._fail(req, "source sender reported transfer failure")
                time.sleep(0.001)
            return self._fail(req, "source sender timed out waiting for transfer")
        finally:
            self.scheduler.tree_cache.dec_lock_ref(
                match.last_device_node, lock_params.to_dec_params()
            )

    def _source_bootstrap_addr(self, source_url: str) -> str | None:
        source_url = source_url.rstrip("/")
        cached_addr = self._source_bootstrap_addrs.get(source_url)
        if cached_addr is not None:
            logger.info(
                "p2p_source_bootstrap_cache_hit: source=%s bootstrap=%s",
                source_url,
                cached_addr,
            )
            return cached_addr

        host = source_url.split("://", 1)[-1].split("/", 1)[0].split(":", 1)[0]
        info_url = f"{source_url.rstrip('/')}/server_info"
        try:
            response = requests.get(info_url, timeout=5)
            if response.status_code != 200:
                logger.warning(
                    "p2p_source_bootstrap_discovery_failed: source=%s status=%s",
                    source_url,
                    response.status_code,
                )
                return None
            info = response.json()
        # Discovery failures are non-fatal because the caller can recompute.
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "p2p_source_bootstrap_discovery_failed: source=%s error=%s",
                source_url,
                exc,
            )
            return None
        port = info.get("disaggregation_bootstrap_port")
        try:
            port = int(port)
        except (TypeError, ValueError):
            logger.warning(
                "p2p_source_bootstrap_invalid: source=%s port=%s", source_url, port
            )
            return None
        if not 0 < port < 65536:
            logger.warning(
                "p2p_source_bootstrap_invalid: source=%s port=%s", source_url, port
            )
            return None
        logger.info(
            "p2p_source_bootstrap_discovered: source=%s bootstrap=%s:%s",
            source_url,
            host,
            port,
        )
        addr = f"{host}:{port}"
        self._source_bootstrap_addrs[source_url] = addr
        return addr

    def _validate_identical_layout(self, kv_manager, source_info) -> str | None:
        if kv_manager.attn_tp_size != source_info.attn_tp_size:
            return (
                "remote KV transfer skipped: source/target TP mismatch "
                f"source={source_info.attn_tp_size} target={kv_manager.attn_tp_size}"
            )
        if kv_manager.attn_cp_size != source_info.attn_cp_size:
            return (
                "remote KV transfer skipped: source/target CP mismatch "
                f"source={source_info.attn_cp_size} target={kv_manager.attn_cp_size}"
            )
        if kv_manager.pp_size != source_info.pp_size:
            return (
                "remote KV transfer skipped: source/target PP mismatch "
                f"source={source_info.pp_size} target={kv_manager.pp_size}"
            )
        return None

    @staticmethod
    def _ensure_receiver_state(kv_manager) -> None:
        """Add the small Decode-mode state surface CommonKVReceiver expects.

        P2P target runs on a Prefill scheduler, but reuses MooncakeKVReceiver to
        avoid rebuilding bootstrap/rank-mapping/metadata registration. The
        receiver only needs these tables and locks; they are normally created
        in CommonKVManager's Decode-mode branch.
        """
        if not hasattr(kv_manager, "prefill_info_table"):
            kv_manager.prefill_info_table = {}
        if not hasattr(kv_manager, "connection_pool"):
            kv_manager.connection_pool = {}
        if not hasattr(kv_manager, "connection_lock"):
            kv_manager.connection_lock = threading.Lock()
        if not hasattr(kv_manager, "required_prefill_response_num_table"):
            kv_manager.required_prefill_response_num_table = {}
        if not hasattr(kv_manager, "addr_to_rooms_tracker"):
            kv_manager.addr_to_rooms_tracker = defaultdict(set)
        if not hasattr(kv_manager, "prefill_response_tracker"):
            kv_manager.prefill_response_tracker = defaultdict(set)
        if not hasattr(kv_manager, "heartbeat_failures"):
            kv_manager.heartbeat_failures = {}
        if not hasattr(kv_manager, "waiting_timeout"):
            kv_manager.waiting_timeout = 120
