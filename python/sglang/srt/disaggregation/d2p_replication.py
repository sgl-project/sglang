"""
Decode-to-Prefill (D→P) KV cache replication.

After decode finishes generating tokens for a request, the decode server
fire-and-forget sends the decode-generated KV cache back to the prefill
server. The prefill server inserts it into its radix cache so future
multi-turn requests get higher prefill-side cache hits.

Mooncake backend only for V1.

Architecture:
- Decode side: DecodeToPrefillKVReplicator captures KV data from finished
  requests and uses a background thread to RDMA-write them to prefill.
- Prefill side: PrefillD2PReceiver listens for D2P requests, allocates pool
  slots, sends allocation info back to decode, and inserts into the radix
  cache after RDMA transfer completes.
- Communication: ZMQ between decode scheduler and prefill scheduler.
"""

from __future__ import annotations

import logging
import struct
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np
import numpy.typing as npt
import requests
import torch
import zmq

from sglang.srt.disaggregation.common.utils import group_concurrent_contiguous
from sglang.srt.disaggregation.utils import is_mla_backend
from sglang.srt.mem_cache.base_prefix_cache import InsertParams, MatchPrefixParams
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey
from sglang.srt.utils.network import NetworkAddress, get_local_ip_auto, get_zmq_socket_on_host

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.managers.scheduler import Scheduler

logger = logging.getLogger(__name__)

D2P_REQUEST_HEADER = b"D2P_REQ"
D2P_ALLOC_RSP_HEADER = b"D2P_ALLOC"
D2P_COMPLETE_HEADER = b"D2P_DONE"


@dataclass
class D2PReplicationTask:
    token_ids: List[int]
    prompt_len: int
    src_kv_indices: npt.NDArray[np.int32]
    bootstrap_addr: str


@dataclass
class D2PAllocResponse:
    room: int
    dst_kv_indices: npt.NDArray[np.int32]
    prefill_session_id: str
    prefill_kv_data_ptrs: List[int]
    prefill_kv_item_lens: List[int]


@dataclass
class D2PPendingRequest:
    room: int
    token_ids: List[int]
    prompt_len: int
    num_tokens: int
    decode_session_id: str
    decode_kv_data_ptrs: List[int]
    decode_kv_item_lens: List[int]
    src_kv_indices: npt.NDArray[np.int32]
    decode_endpoint: str


def _rdma_transfer_kv(
    engine,
    session_id: str,
    src_data_ptrs: List[int],
    dst_data_ptrs: List[int],
    item_lens: List[int],
    src_indices: npt.NDArray[np.int32],
    dst_indices: npt.NDArray[np.int32],
    is_mla: bool,
) -> int:
    """Low-level RDMA transfer of KV cache between pools."""
    src_blocks, dst_blocks = group_concurrent_contiguous(src_indices, dst_indices)

    transfer_blocks = []
    if is_mla:
        num_layers = len(src_data_ptrs)
        for layer_id in range(num_layers):
            for src_block, dst_block in zip(src_blocks, dst_blocks):
                src_addr = src_data_ptrs[layer_id] + int(src_block[0]) * item_lens[layer_id]
                dst_addr = dst_data_ptrs[layer_id] + int(dst_block[0]) * item_lens[layer_id]
                length = item_lens[layer_id] * len(src_block)
                transfer_blocks.append((src_addr, dst_addr, length))
    else:
        num_layers = len(src_data_ptrs) // 2
        for layer_id in range(num_layers):
            for src_block, dst_block in zip(src_blocks, dst_blocks):
                src_addr = src_data_ptrs[layer_id] + int(src_block[0]) * item_lens[layer_id]
                dst_addr = dst_data_ptrs[layer_id] + int(dst_block[0]) * item_lens[layer_id]
                length = item_lens[layer_id] * len(src_block)
                transfer_blocks.append((src_addr, dst_addr, length))
                v_layer = num_layers + layer_id
                src_addr = src_data_ptrs[v_layer] + int(src_block[0]) * item_lens[v_layer]
                dst_addr = dst_data_ptrs[v_layer] + int(dst_block[0]) * item_lens[v_layer]
                length = item_lens[v_layer] * len(src_block)
                transfer_blocks.append((src_addr, dst_addr, length))

    if not transfer_blocks:
        return 0

    src_addrs, dst_addrs, lengths = zip(*transfer_blocks)
    return engine.batch_transfer_sync(
        session_id, list(src_addrs), list(dst_addrs), list(lengths)
    )


# ---------------------------------------------------------------------------
# Decode side
# ---------------------------------------------------------------------------


class DecodeToPrefillKVReplicator:
    """Decode-side D→P KV cache replicator.

    Captures KV data from finished requests and uses a background thread
    to coordinate with prefill and RDMA-write the data to prefill's KV pool.
    """

    def __init__(self, scheduler: Scheduler):
        from sglang.srt.distributed.parallel_state import get_mooncake_transfer_engine

        self.engine = get_mooncake_transfer_engine()
        self.session_id = self.engine.get_session_id()
        kv_pool = scheduler.token_to_kv_pool_allocator.get_kvcache()
        self.is_mla = is_mla_backend(kv_pool)
        self.kv_data_ptrs, _, self.kv_item_lens = kv_pool.get_contiguous_buf_infos()
        self.tree_cache = scheduler.tree_cache
        self.bootstrap_port = scheduler.server_args.disaggregation_bootstrap_port

        local_ip = get_local_ip_auto()
        zmq_ctx = zmq.Context()
        self.recv_port, self.recv_sock = get_zmq_socket_on_host(
            zmq_ctx, zmq.PULL, host=local_ip
        )
        self.local_endpoint = NetworkAddress(local_ip, self.recv_port).to_tcp()
        self.local_ip = local_ip

        self._send_sock_cache: Dict[str, zmq.Socket] = {}
        self._send_sock_lock = threading.Lock()
        self._zmq_ctx = zmq_ctx

        self._prefill_endpoint_cache: Dict[str, str] = {}

        self._task_queue: deque = deque()
        self._room_counter = 1_000_000_000

        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()
        self._recv_thread = threading.Thread(target=self._recv_loop, daemon=True)
        self._recv_thread.start()

        self._alloc_responses: Dict[int, D2PAllocResponse] = {}
        self._alloc_lock = threading.Lock()

        logger.info("D2P replicator started on decode side")

    def _next_room(self) -> int:
        self._room_counter += 1
        return self._room_counter

    def _get_send_sock(self, endpoint: str) -> zmq.Socket:
        with self._send_sock_lock:
            if endpoint not in self._send_sock_cache:
                sock = self._zmq_ctx.socket(zmq.PUSH)
                sock.connect(endpoint)
                self._send_sock_cache[endpoint] = sock
            return self._send_sock_cache[endpoint]

    def _get_prefill_d2p_endpoint(self, bootstrap_addr: str) -> Optional[str]:
        if bootstrap_addr in self._prefill_endpoint_cache:
            return self._prefill_endpoint_cache[bootstrap_addr]
        try:
            url = f"http://{bootstrap_addr}/d2p_endpoint"
            resp = requests.get(url, timeout=5)
            if resp.status_code == 200:
                endpoint = resp.json().get("endpoint")
                if endpoint:
                    self._prefill_endpoint_cache[bootstrap_addr] = endpoint
                    return endpoint
        except Exception as e:
            logger.debug(f"D2P: failed to fetch prefill endpoint from {bootstrap_addr}: {e}")
        return None

    def capture_and_enqueue(self, req: Req):
        """Called AFTER release_kv_cache. The tree now owns the KV indices."""
        logger.info(
            f"D2P capture_and_enqueue: origin_input_ids={len(req.origin_input_ids)}, "
            f"output_ids={len(req.output_ids)}, "
            f"kv_committed_len_saved={req.kv_committed_len_saved}, "
            f"bootstrap_host={req.bootstrap_host}"
        )
        token_ids = list(
            (req.origin_input_ids + req.output_ids)[: req.kv_committed_len_saved]
        )
        prompt_len = len(req.origin_input_ids)
        if len(token_ids) <= prompt_len:
            logger.info(f"D2P: skip, token_ids({len(token_ids)}) <= prompt_len({prompt_len})")
            return

        match_result = self.tree_cache.match_prefix(
            MatchPrefixParams(key=RadixKey(token_ids))
        )
        matched_len = len(match_result.device_indices)
        if matched_len <= prompt_len:
            logger.info(f"D2P: skip, matched_len({matched_len}) <= prompt_len({prompt_len})")
            return

        last_node = match_result.last_device_node
        self.tree_cache.inc_lock_ref(last_node)

        src_kv_indices = match_result.device_indices[prompt_len :].cpu().numpy().astype(np.int32)

        bootstrap_addr = NetworkAddress(
            req.bootstrap_host, self.bootstrap_port
        ).to_host_port_str()

        task = D2PReplicationTask(
            token_ids=token_ids,
            prompt_len=prompt_len,
            src_kv_indices=src_kv_indices,
            bootstrap_addr=bootstrap_addr,
        )
        task._locked_node = last_node
        task._room = self._next_room()
        self._task_queue.append(task)
        logger.info(
            f"D2P: enqueued task room={task._room}, tokens={len(token_ids)}, "
            f"bootstrap={bootstrap_addr}"
        )

    def _recv_loop(self):
        """Receives allocation responses from prefill."""
        while True:
            try:
                msg = self.recv_sock.recv_multipart()
                header = msg[0]
                if header == D2P_ALLOC_RSP_HEADER:
                    room = int(msg[1].decode("ascii"))
                    dst_kv_indices = np.frombuffer(msg[2], dtype=np.int32)
                    prefill_session_id = msg[3].decode("ascii")
                    prefill_kv_data_ptrs = list(
                        struct.unpack(f"{len(msg[4]) // 8}Q", msg[4])
                    )
                    prefill_kv_item_lens = list(
                        struct.unpack(f"{len(msg[5]) // 4}I", msg[5])
                    )
                    rsp = D2PAllocResponse(
                        room=room,
                        dst_kv_indices=dst_kv_indices,
                        prefill_session_id=prefill_session_id,
                        prefill_kv_data_ptrs=prefill_kv_data_ptrs,
                        prefill_kv_item_lens=prefill_kv_item_lens,
                    )
                    with self._alloc_lock:
                        self._alloc_responses[room] = rsp
            except Exception as e:
                logger.warning(f"D2P recv error: {e}")

    def _worker_loop(self):
        while True:
            if not self._task_queue:
                time.sleep(0.005)
                continue
            task = self._task_queue.popleft()
            try:
                self._process_task(task)
            except Exception as e:
                logger.warning(
                    f"D2P replication failed for {len(task.token_ids)} tokens: {e}"
                )
            finally:
                self.tree_cache.dec_lock_ref(task._locked_node)

    def _process_task(self, task: D2PReplicationTask):
        logger.info(f"D2P _process_task: room={task._room}, bootstrap={task.bootstrap_addr}")
        prefill_endpoint = self._get_prefill_d2p_endpoint(task.bootstrap_addr)
        if prefill_endpoint is None:
            logger.info("D2P: no prefill endpoint available, skipping")
            return

        logger.info(f"D2P: got prefill endpoint={prefill_endpoint}")
        room = task._room
        src_kv_indices = task.src_kv_indices

        packed_ptrs = struct.pack(f"{len(self.kv_data_ptrs)}Q", *self.kv_data_ptrs)
        packed_item_lens = struct.pack(f"{len(self.kv_item_lens)}I", *self.kv_item_lens)
        token_ids_bytes = np.array(task.token_ids, dtype=np.int32).tobytes()

        sock = self._get_send_sock(prefill_endpoint)
        logger.info(f"D2P: sending request room={room}, src_indices={len(src_kv_indices)}")
        sock.send_multipart([
            D2P_REQUEST_HEADER,
            str(room).encode("ascii"),
            token_ids_bytes,
            str(task.prompt_len).encode("ascii"),
            src_kv_indices.tobytes(),
            self.session_id.encode("ascii"),
            packed_ptrs,
            packed_item_lens,
            self.local_endpoint.encode("ascii"),
        ])

        deadline = time.time() + 30.0
        while time.time() < deadline:
            with self._alloc_lock:
                rsp = self._alloc_responses.pop(room, None)
            if rsp is not None:
                break
            time.sleep(0.002)
        else:
            logger.info(f"D2P: allocation timeout for room {room}")
            return

        logger.info(f"D2P: got alloc response room={room}, dst_indices={len(rsp.dst_kv_indices)}")
        ret = _rdma_transfer_kv(
            engine=self.engine,
            session_id=rsp.prefill_session_id,
            src_data_ptrs=self.kv_data_ptrs,
            dst_data_ptrs=rsp.prefill_kv_data_ptrs,
            item_lens=self.kv_item_lens,
            src_indices=src_kv_indices,
            dst_indices=rsp.dst_kv_indices,
            is_mla=self.is_mla,
        )

        if ret == 0:
            logger.info(f"D2P: RDMA transfer success room={room}, sending completion")
            sock.send_multipart([
                D2P_COMPLETE_HEADER,
                str(room).encode("ascii"),
            ])
        else:
            logger.warning(f"D2P: RDMA transfer failed with code {ret}")


# ---------------------------------------------------------------------------
# Prefill side
# ---------------------------------------------------------------------------


class PrefillD2PReceiver:
    """Prefill-side D→P KV cache receiver.

    Listens for D2P requests from decode, allocates pool slots, responds
    with allocation info, and inserts KV into the radix cache after
    RDMA transfer completes.
    """

    def __init__(self, scheduler: Scheduler):
        from sglang.srt.distributed.parallel_state import get_mooncake_transfer_engine

        self.engine = get_mooncake_transfer_engine()
        self.session_id = self.engine.get_session_id()
        self.tree_cache = scheduler.tree_cache
        self.token_to_kv_pool_allocator = scheduler.token_to_kv_pool_allocator
        kv_pool = self.token_to_kv_pool_allocator.get_kvcache()
        self.is_mla = is_mla_backend(kv_pool)
        self.kv_data_ptrs, _, self.kv_item_lens = kv_pool.get_contiguous_buf_infos()
        self.page_size = kv_pool.page_size

        local_ip = get_local_ip_auto()
        self._zmq_ctx = zmq.Context()
        self.port, self.sock = get_zmq_socket_on_host(
            self._zmq_ctx, zmq.PULL, host=local_ip
        )
        self.endpoint = NetworkAddress(local_ip, self.port).to_tcp()
        self.host_port = NetworkAddress(local_ip, self.port).to_host_port_str()

        self._send_sock_cache: Dict[str, zmq.Socket] = {}
        self._send_sock_lock = threading.Lock()

        self._pending_alloc: deque = deque()
        self._pending_insert: deque = deque()
        self._allocated_rooms: Dict[int, dict] = {}
        self._inflight_tokens: int = 0

        self._recv_thread = threading.Thread(target=self._recv_loop, daemon=True)
        self._recv_thread.start()

        logger.info(f"D2P receiver started on prefill side at {self.host_port}")

    def _get_send_sock(self, endpoint: str) -> zmq.Socket:
        with self._send_sock_lock:
            if endpoint not in self._send_sock_cache:
                sock = self._zmq_ctx.socket(zmq.PUSH)
                sock.connect(endpoint)
                self._send_sock_cache[endpoint] = sock
            return self._send_sock_cache[endpoint]

    def _recv_loop(self):
        """Receives D2P requests and completion notifications from decode."""
        while True:
            try:
                msg = self.sock.recv_multipart()
                header = msg[0]
                if header == D2P_REQUEST_HEADER:
                    room = int(msg[1].decode("ascii"))
                    token_ids = list(np.frombuffer(msg[2], dtype=np.int32))
                    prompt_len = int(msg[3].decode("ascii"))
                    src_kv_indices = np.frombuffer(msg[4], dtype=np.int32).copy()
                    decode_session_id = msg[5].decode("ascii")
                    decode_kv_data_ptrs = list(
                        struct.unpack(f"{len(msg[6]) // 8}Q", msg[6])
                    )
                    decode_kv_item_lens = list(
                        struct.unpack(f"{len(msg[7]) // 4}I", msg[7])
                    )
                    decode_endpoint = msg[8].decode("ascii")
                    pending = D2PPendingRequest(
                        room=room,
                        token_ids=token_ids,
                        prompt_len=prompt_len,
                        num_tokens=len(src_kv_indices),
                        decode_session_id=decode_session_id,
                        decode_kv_data_ptrs=decode_kv_data_ptrs,
                        decode_kv_item_lens=decode_kv_item_lens,
                        src_kv_indices=src_kv_indices,
                        decode_endpoint=decode_endpoint,
                    )
                    self._pending_alloc.append(pending)
                    logger.info(f"D2P prefill recv: request room={room}, tokens={len(src_kv_indices)}")

                elif header == D2P_COMPLETE_HEADER:
                    room = int(msg[1].decode("ascii"))
                    self._pending_insert.append(room)
                    logger.info(f"D2P prefill recv: completion room={room}")

            except Exception as e:
                logger.warning(f"D2P prefill recv error: {e}")

    def process_d2p_incoming(self):
        """Called from the prefill scheduler event loop each tick.

        Processes pending allocation requests and completed transfers.
        Must run on the scheduler thread (owns KV pool and radix cache).
        """
        self._process_allocations()
        self._process_completions()

    def _process_allocations(self):
        processed = 0
        while self._pending_alloc and processed < 8:
            req = self._pending_alloc.popleft()
            processed += 1
            try:
                self._handle_alloc(req)
            except Exception as e:
                logger.warning(f"D2P alloc failed for room {req.room}: {e}")

    def _handle_alloc(self, req: D2PPendingRequest):
        num_tokens = req.num_tokens
        logger.info(f"D2P prefill _handle_alloc: room={req.room}, num_tokens={num_tokens}")
        can_alloc = self.token_to_kv_pool_allocator.available_size() >= num_tokens
        if not can_alloc:
            logger.info(f"D2P: not enough pool space for {num_tokens} tokens, skipping")
            return

        dst_indices = self.token_to_kv_pool_allocator.alloc(num_tokens)
        if dst_indices is None:
            logger.info("D2P: pool allocation returned None, skipping")
            return

        dst_kv_indices = dst_indices.cpu().numpy().astype(np.int32)

        self._allocated_rooms[req.room] = {
            "token_ids": req.token_ids,
            "prompt_len": req.prompt_len,
            "dst_kv_indices": dst_kv_indices,
        }
        self._inflight_tokens += num_tokens

        packed_ptrs = struct.pack(f"{len(self.kv_data_ptrs)}Q", *self.kv_data_ptrs)
        packed_item_lens = struct.pack(f"{len(self.kv_item_lens)}I", *self.kv_item_lens)

        sock = self._get_send_sock(req.decode_endpoint)
        sock.send_multipart([
            D2P_ALLOC_RSP_HEADER,
            str(req.room).encode("ascii"),
            dst_kv_indices.tobytes(),
            self.session_id.encode("ascii"),
            packed_ptrs,
            packed_item_lens,
        ])

    def _process_completions(self):
        processed = 0
        while self._pending_insert and processed < 8:
            room = self._pending_insert.popleft()
            processed += 1
            alloc_info = self._allocated_rooms.pop(room, None)
            if alloc_info is None:
                logger.info(f"D2P prefill: completion for unknown room {room}")
                continue
            num_tokens = len(alloc_info["dst_kv_indices"])
            logger.info(f"D2P prefill: inserting room={room}, tokens={num_tokens}")
            try:
                self._insert_into_radix_cache(
                    token_ids=alloc_info["token_ids"],
                    prompt_len=alloc_info["prompt_len"],
                    new_kv_indices=alloc_info["dst_kv_indices"],
                )
                logger.info(f"D2P prefill: insert success room={room}")
            except Exception as e:
                logger.warning(f"D2P radix insert failed for room {room}: {e}")
            self._inflight_tokens -= num_tokens

    def _insert_into_radix_cache(
        self,
        token_ids: List[int],
        prompt_len: int,
        new_kv_indices: npt.NDArray[np.int32],
    ):
        if not isinstance(self.tree_cache, RadixCache):
            logger.debug("D2P: tree_cache is not RadixCache, skipping insert")
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
