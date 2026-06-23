"""
Decode-to-Prefill (D->P) KV cache replication via D2PKVManager.

After decode finishes generating tokens for a request, the decode server
fire-and-forget sends the decode-generated KV cache back to the prefill
server. The prefill server inserts it into its radix cache so future
multi-turn requests get higher prefill-side cache hits.

Mooncake backend only for V1.

Architecture:
- D2PKVManager subclasses MooncakeKVManager. No separate Replicator or
  Receiver classes — the manager itself handles both roles via the
  existing Mooncake threading infrastructure (bootstrap_thread,
  transfer_worker, decode_thread).

- Decode side: D2PKVManager in PREFILL mode. capture_and_enqueue()
  sends D2P_REQ and stores the task. The base bootstrap_thread receives
  TransferInfo and calls update_status(WaitingForInput). The override
  hooks into that transition to call add_transfer_request(), which
  enqueues work to the base transfer_worker for RDMA. On completion
  (Success/Failed), update_status handles dec_lock_ref cleanup.

- Prefill side: D2PKVManager in DECODE mode. The forward manager's
  bootstrap_thread dispatches D2P_REQ via _d2p_receiver.handle_d2p_request().
  process_d2p_incoming() (called from the scheduler event loop) handles
  allocation, receiver creation, and radix cache insertion on completion.

- Bootstrap: A second MooncakeKVBootstrapServer instance on the decode
  side at port (disaggregation_bootstrap_port + 1). No subclassing needed.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional

import numpy as np
import numpy.typing as npt
import torch

from sglang.srt.disaggregation.base.conn import KVArgs, KVPoll
from sglang.srt.disaggregation.common.conn import CommonKVReceiver
from sglang.srt.disaggregation.mooncake.conn import (
    MooncakeKVManager,
    MooncakeKVReceiver,
)
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.environ import envs
from sglang.srt.mem_cache.base_prefix_cache import InsertParams, MatchPrefixParams
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey
from sglang.srt.utils.network import NetworkAddress

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.managers.scheduler import Scheduler

logger = logging.getLogger(__name__)

D2P_REQUEST_HEADER = b"D2P_REQ"


def get_d2p_bootstrap_port(server_args) -> int:
    return server_args.disaggregation_bootstrap_port + 1


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class D2PReplicationTask:
    token_ids: List[int]
    prompt_len: int
    src_kv_indices: npt.NDArray[np.int32]
    bootstrap_addr: str


@dataclass
class D2PPendingRequest:
    room: int
    token_ids: List[int]
    prompt_len: int
    num_tokens: int
    d2p_bootstrap_addr: str


# ---------------------------------------------------------------------------
# Shared: build KVArgs from a KV pool for reverse managers
# ---------------------------------------------------------------------------


def _build_reverse_kv_args(scheduler: Scheduler, forward_mgr: MooncakeKVManager) -> KVArgs:
    """Build KVArgs for a reverse manager from the scheduler's KV pool.

    Mirrors the _init_kv_manager() pattern in prefill.py / decode.py but
    skips aux/state buffers (D2P only transfers KV cache data).
    """
    kv_args = KVArgs()
    kv_args.engine_rank = forward_mgr.kv_args.engine_rank
    kv_args.pp_rank = forward_mgr.pp_rank
    kv_args.system_dp_rank = forward_mgr.system_dp_rank

    kv_pool = scheduler.token_to_kv_pool_allocator.get_kvcache()
    kv_args.prefill_start_layer = getattr(kv_pool, "start_layer", 0)
    kv_args.prefill_end_layer = getattr(kv_pool, "end_layer", None)

    kv_data_ptrs, kv_data_lens, kv_item_lens = kv_pool.get_contiguous_buf_infos()
    kv_args.kv_data_ptrs = kv_data_ptrs
    kv_args.kv_data_lens = kv_data_lens
    kv_args.kv_item_lens = kv_item_lens
    kv_args.page_size = kv_pool.page_size

    if not forward_mgr.is_mla_backend:
        kv_args.kv_head_num = getattr(kv_pool, "head_num", 0)
        kv_args.total_kv_head_num = getattr(
            forward_mgr.kv_args, "total_kv_head_num",
            kv_args.kv_head_num * forward_mgr.attn_tp_size,
        )

    kv_args.aux_data_ptrs = []
    kv_args.aux_data_lens = []
    kv_args.aux_item_lens = []
    kv_args.state_types = []
    kv_args.state_data_ptrs = []
    kv_args.state_data_lens = []
    kv_args.state_item_lens = []
    kv_args.state_dim_per_tensor = []

    kv_args.ib_device = scheduler.server_args.disaggregation_ib_device
    kv_args.gpu_id = scheduler.ps.gpu_id
    kv_args.mla_compression_ratios = None

    from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool

    if isinstance(kv_pool, DeepSeekV4TokenToKVPool):
        kv_args.mla_compression_ratios = list(kv_pool.compression_ratios)

    return kv_args


# ---------------------------------------------------------------------------
# D2PKVManager
# ---------------------------------------------------------------------------


class D2PKVManager(MooncakeKVManager):
    """Reverse KV manager for D2P replication.

    Shares the forward manager's transfer engine (register_buffer_to_engine
    is a no-op). Registers to the D2P bootstrap server at
    (disaggregation_bootstrap_port + 1).

    PREFILL mode (decode side — sender):
      capture_and_enqueue() stores a task and sends D2P_REQ. The base
      bootstrap_thread receives TransferInfo and calls update_status
      (WaitingForInput). The override hooks into that transition to call
      add_transfer_request(), dispatching work to the base transfer_worker
      for RDMA. On Success/Failed, update_status calls dec_lock_ref.

    DECODE mode (prefill side — receiver):
      The forward manager's bootstrap_thread dispatches D2P_REQ via
      _d2p_receiver.handle_d2p_request(). process_d2p_incoming() is called
      from the scheduler event loop to handle allocation, receiver
      creation, completion polling, and radix cache insertion.
    """

    def __init__(
        self,
        args: KVArgs,
        disaggregation_mode: DisaggregationMode,
        server_args,
        is_mla_backend: Optional[bool] = False,
    ):
        self._d2p_bootstrap_port = get_d2p_bootstrap_port(server_args)

        self.tree_cache = None
        self.forward_mgr = None

        if disaggregation_mode == DisaggregationMode.PREFILL:
            # Sender state must exist before super().__init__() because the
            # inherited Mooncake bootstrap thread starts during construction.
            self._d2p_pending_sends: Dict[int, D2PReplicationTask] = {}
            self._d2p_inflight: Dict[int, D2PReplicationTask] = {}
            self._room_counter = 1_000_000_000
            self._forward_bootstrap_port = None
            self._d2p_bootstrap_addr = None
        elif disaggregation_mode == DisaggregationMode.DECODE:
            # Receiver state is driven by the prefill scheduler event loop.
            self._pending_alloc: deque = deque()
            self._active_receivers: Dict[int, tuple] = {}
            self._allocated_rooms: Dict[int, dict] = {}
            self.token_to_kv_pool_allocator = None

        transfer_thread_pool_size, transfer_queue_size = \
            envs.SGLANG_DISAGGREGATION_THREAD_POOL_SIZE.get(), envs.SGLANG_DISAGGREGATION_QUEUE_SIZE.get()
        envs.SGLANG_DISAGGREGATION_THREAD_POOL_SIZE.set(1)
        envs.SGLANG_DISAGGREGATION_QUEUE_SIZE.set(1)
        super().__init__(args, disaggregation_mode, server_args, is_mla_backend)
        envs.SGLANG_DISAGGREGATION_THREAD_POOL_SIZE.set(transfer_thread_pool_size)
        envs.SGLANG_DISAGGREGATION_QUEUE_SIZE.set(transfer_queue_size)
        self.bootstrap_port = self._d2p_bootstrap_port

    def register_to_bootstrap(self):
        self.bootstrap_port = self._d2p_bootstrap_port
        super().register_to_bootstrap()

    def register_buffer_to_engine(self):
        pass

    def _get_transfer_thread_pool_size(self) -> int:
        return 1

    def _get_transfer_queue_size(self) -> int:
        return 1

    def update_status(self, bootstrap_room, status):
        super().update_status(bootstrap_room, status)

        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            if status == KVPoll.WaitingForInput:
                task = self._d2p_pending_sends.pop(bootstrap_room, None)
                if task is not None:
                    self.add_transfer_request(
                        bootstrap_room,
                        task.src_kv_indices,
                        slice(0, len(task.src_kv_indices)),
                        is_last_chunk=True,
                        aux_index=None,
                    )
                    self._d2p_inflight[bootstrap_room] = task
            elif status in (KVPoll.Success, KVPoll.Failed):
                task = self._d2p_inflight.pop(bootstrap_room, None)
                if task is not None:
                    if status == KVPoll.Success:
                        logger.info(f"D2P: transfer success room={bootstrap_room}")
                    else:
                        logger.warning(f"D2P: transfer failed room={bootstrap_room}")
                    self.tree_cache.dec_lock_ref(task._locked_node)

    # ------------------------------------------------------------------
    # Init helpers (called from scheduler after construction)
    # ------------------------------------------------------------------

    def init_d2p_sender(self, scheduler: Scheduler, forward_mgr: MooncakeKVManager):
        self.tree_cache = scheduler.tree_cache
        self.forward_mgr = forward_mgr
        self._forward_bootstrap_port = scheduler.server_args.disaggregation_bootstrap_port

        if scheduler.server_args.dist_init_addr:
            d2p_host = NetworkAddress.parse(
                scheduler.server_args.dist_init_addr
            ).resolved().host
        else:
            # In single-node mode the D2P bootstrap server binds to
            # server_args.host. Advertise that same address for loopback/local
            # tests; only replace wildcard binds with a routable interface IP.
            bind_host = scheduler.server_args.host
            d2p_host = (
                self.local_ip
                if bind_host in ("0.0.0.0", "::", "")
                else bind_host
            )
        d2p_port = get_d2p_bootstrap_port(scheduler.server_args)
        self._d2p_bootstrap_addr = NetworkAddress(
            d2p_host, d2p_port
        ).to_host_port_str()
        logger.info("D2P sender initialized (reverse PREFILL manager)")

    def init_d2p_receiver(self, scheduler: Scheduler, forward_mgr: MooncakeKVManager):
        self.tree_cache = scheduler.tree_cache
        self.forward_mgr = forward_mgr
        self.token_to_kv_pool_allocator = scheduler.token_to_kv_pool_allocator
        logger.info("D2P receiver initialized (reverse DECODE manager)")

    # ------------------------------------------------------------------
    # PREFILL mode (decode side — sender)
    # ------------------------------------------------------------------

    def _next_room(self) -> int:
        self._room_counter += 1
        return self._room_counter

    def capture_and_enqueue(self, req: Req):
        """Called AFTER release_kv_cache from the scheduler thread.

        Captures the decode-generated KV indices from the radix tree, sends
        D2P_REQ to prefill, and stores the task. The base bootstrap_thread
        receives TransferInfo and triggers the transfer via update_status.
        """
        token_ids = list(
            (req.origin_input_ids + req.output_ids)[: req.kv_committed_len_saved]
        )
        prompt_len = len(req.origin_input_ids)
        if len(token_ids) <= prompt_len:
            return

        match_result = self.tree_cache.match_prefix(
            MatchPrefixParams(key=RadixKey(token_ids))
        )
        matched_len = len(match_result.device_indices)
        if matched_len <= prompt_len:
            return

        last_node = match_result.last_device_node
        self.tree_cache.inc_lock_ref(last_node)

        src_kv_indices = (
            match_result.device_indices[prompt_len:].cpu().numpy().astype(np.int32)
        )

        bootstrap_addr = NetworkAddress(
            req.bootstrap_host, self._forward_bootstrap_port
        ).to_host_port_str()

        task = D2PReplicationTask(
            token_ids=token_ids,
            prompt_len=prompt_len,
            src_kv_indices=src_kv_indices,
            bootstrap_addr=bootstrap_addr,
        )
        task._locked_node = last_node
        task._room = self._next_room()

        if not self._send_d2p_request(task):
            self.tree_cache.dec_lock_ref(last_node)
            return

        self._d2p_pending_sends[task._room] = task
        self.update_status(task._room, KVPoll.Bootstrapping)
        logger.info(
            f"D2P: enqueued room={task._room}, "
            f"new_tokens={len(src_kv_indices)}, bootstrap={bootstrap_addr}"
        )

    def _find_bootstrap_info(self, bootstrap_addr: str) -> Optional[dict]:
        for key, infos in self.forward_mgr.connection_pool.items():
            if key.startswith(bootstrap_addr + "_"):
                for info in infos:
                    if not info.get("is_dummy", False):
                        return info
        return None

    def _send_d2p_request(self, task: D2PReplicationTask) -> bool:
        bootstrap_info = self._find_bootstrap_info(task.bootstrap_addr)
        if bootstrap_info is None:
            logger.info("D2P: no bootstrap connection available, skipping")
            return False

        sock, lock = CommonKVReceiver._connect_to_bootstrap_server(bootstrap_info)
        token_ids_bytes = np.array(task.token_ids, dtype=np.int32).tobytes()

        with lock:
            sock.send_multipart([
                D2P_REQUEST_HEADER,
                str(task._room).encode("ascii"),
                token_ids_bytes,
                str(task.prompt_len).encode("ascii"),
                str(len(task.src_kv_indices)).encode("ascii"),
                self._d2p_bootstrap_addr.encode("ascii"),
            ])
        return True

    # ------------------------------------------------------------------
    # DECODE mode (prefill side — receiver)
    # ------------------------------------------------------------------

    def handle_d2p_request(self, msg):
        """Called from forward manager's bootstrap_thread on D2P_REQ.

        msg layout: [D2P_REQ, room, token_ids_bytes, prompt_len,
                      num_tokens, d2p_bootstrap_addr]
        """
        room = int(msg[1].decode("ascii"))
        token_ids = list(np.frombuffer(msg[2], dtype=np.int32))
        prompt_len = int(msg[3].decode("ascii"))
        num_tokens = int(msg[4].decode("ascii"))
        d2p_bootstrap_addr = msg[5].decode("ascii")
        pending = D2PPendingRequest(
            room=room,
            token_ids=token_ids,
            prompt_len=prompt_len,
            num_tokens=num_tokens,
            d2p_bootstrap_addr=d2p_bootstrap_addr,
        )
        self._pending_alloc.append(pending)
        logger.info(f"D2P prefill: request room={room}, tokens={num_tokens}")

    def process_d2p_incoming(self):
        """Called from the prefill scheduler event loop each tick."""
        self._process_allocations()
        self._process_active_receivers()

    def _process_allocations(self):
        processed = 0
        while self._pending_alloc and processed < 8:
            req = self._pending_alloc[0]
            processed += 1

            if not self.try_ensure_parallel_info(req.d2p_bootstrap_addr):
                logger.debug("D2P: decode reverse manager not registered yet, deferring")
                break

            self._pending_alloc.popleft()
            try:
                self._handle_alloc(req)
            except Exception as e:
                logger.warning(f"D2P alloc failed for room {req.room}: {e}")

    def _handle_alloc(self, req: D2PPendingRequest):
        num_tokens = req.num_tokens
        if self.token_to_kv_pool_allocator.available_size() < num_tokens:
            logger.info(f"D2P: not enough pool space for {num_tokens} tokens, skipping")
            return

        dst_indices = self.token_to_kv_pool_allocator.alloc(num_tokens)
        if dst_indices is None:
            logger.info("D2P: pool allocation returned None, skipping")
            return

        dst_kv_indices = dst_indices.cpu().numpy().astype(np.int32)

        alloc_info = {
            "token_ids": req.token_ids,
            "prompt_len": req.prompt_len,
            "dst_kv_indices": dst_kv_indices,
        }
        self._allocated_rooms[req.room] = alloc_info

        receiver = MooncakeKVReceiver(
            self, req.d2p_bootstrap_addr, req.room
        )
        receiver.init(prefill_dp_rank=0)

        if getattr(receiver, "conclude_state", None) == KVPoll.Failed:
            logger.warning(f"D2P: receiver init failed for room {req.room}")
            self.token_to_kv_pool_allocator.free(dst_indices)
            self._allocated_rooms.pop(req.room, None)
            return

        receiver.send_metadata(dst_kv_indices)

        self._active_receivers[req.room] = (receiver, alloc_info)
        logger.info(
            f"D2P prefill: receiver created room={req.room}, "
            f"dst_tokens={num_tokens}"
        )

    def _process_active_receivers(self):
        done_rooms = []
        for room, (receiver, alloc_info) in self._active_receivers.items():
            status = receiver.poll()
            if status == KVPoll.Success:
                logger.info(f"D2P prefill: transfer complete room={room}")
                try:
                    self._insert_into_radix_cache(
                        token_ids=alloc_info["token_ids"],
                        prompt_len=alloc_info["prompt_len"],
                        new_kv_indices=alloc_info["dst_kv_indices"],
                    )
                except Exception as e:
                    logger.warning(f"D2P radix insert failed room={room}: {e}")
                receiver.clear()
                done_rooms.append(room)
            elif status == KVPoll.Failed:
                logger.warning(f"D2P prefill: transfer failed room={room}")
                dst_indices = torch.from_numpy(
                    alloc_info["dst_kv_indices"].astype(np.int64)
                )
                self.token_to_kv_pool_allocator.free(dst_indices)
                receiver.clear()
                done_rooms.append(room)

        for room in done_rooms:
            self._active_receivers.pop(room, None)
            self._allocated_rooms.pop(room, None)

    def _insert_into_radix_cache(
        self,
        token_ids: List[int],
        prompt_len: int,
        new_kv_indices: npt.NDArray[np.int32],
    ):
        if not isinstance(self.tree_cache, RadixCache):
            return

        existing_match = self.tree_cache.match_prefix(
            MatchPrefixParams(key=RadixKey(token_ids[:prompt_len]))
        )
        existing_indices = existing_match.device_indices

        new_indices_tensor = torch.from_numpy(new_kv_indices.astype(np.int64)).to(
            existing_indices.device
        )
        full_indices = torch.cat([existing_indices, new_indices_tensor])

        full_key = RadixKey(token_ids[: len(full_indices)])
        result = self.tree_cache.insert(
            InsertParams(key=full_key, value=full_indices)
        )
        if result.prefix_len > len(existing_indices):
            dup_indices = full_indices[len(existing_indices) : result.prefix_len]
            if len(dup_indices) > 0:
                self.token_to_kv_pool_allocator.free(dup_indices)

        logger.debug(
            f"D2P: inserted {len(new_kv_indices)} tokens into radix cache "
            f"(prefix_len={result.prefix_len}, total={len(full_indices)})"
        )
