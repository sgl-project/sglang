import ctypes
import logging
import os
import socket
import struct
import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.mem_cache.kv_connector import BaseKVConnector, LoadOperation
from sglang.srt.utils import broadcast_pyobj

try:
    from flexkv.common.request import KVResponseStatus
    from flexkv.common.storage import KVCacheLayout, KVCacheLayoutType
    from flexkv.integration.config import FlexKVConfig
    from flexkv.kvmanager import KVManager
    from flexkv.server.client import KVTPClient
except ImportError as e:
    raise RuntimeError("FlexKV is not installed. Please install it.") from e

logger = logging.getLogger(__name__)


# ---- libc / eventfd ----
libc = ctypes.CDLL("libc.so.6", use_errno=True)

libc.eventfd.argtypes = [ctypes.c_uint, ctypes.c_int]
libc.eventfd.restype = ctypes.c_int

libc.read.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t]
libc.read.restype = ctypes.c_ssize_t

libc.write.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t]
libc.write.restype = ctypes.c_ssize_t

EFD_SEMAPHORE = 0x1
EFD_NONBLOCK = 0x800


def eventfd(initval=0, flags=0):
    fd = libc.eventfd(ctypes.c_uint(initval), ctypes.c_int(flags))
    if fd == -1:
        err = ctypes.get_errno()
        raise OSError(err, os.strerror(err))
    return fd


def eventfd_write(fd, val):
    v = ctypes.c_uint64(val)
    buf = ctypes.byref(v)
    n = libc.write(fd, buf, ctypes.sizeof(v))
    if n != ctypes.sizeof(v):
        err = ctypes.get_errno()
        raise OSError(err, f"eventfd write failed: {os.strerror(err)}")


def eventfd_read(fd):
    """Blocking read from eventfd."""
    v = ctypes.c_uint64()
    buf = ctypes.byref(v)
    n = libc.read(fd, buf, ctypes.sizeof(v))
    if n != ctypes.sizeof(v):
        err = ctypes.get_errno()
        if err == 11:  # EAGAIN
            return 0
        raise OSError(err, f"eventfd read failed: {os.strerror(err)}")
    return v.value


def send_fds(sock: socket.socket, fds: list, extra_data: bytes = b"x"):
    """Send multiple fds + extra_data via Unix domain socket."""
    fds_packed = struct.pack(f"{len(fds)}i", *fds)
    ancdata = [(socket.SOL_SOCKET, socket.SCM_RIGHTS, fds_packed)]
    sock.sendmsg([extra_data], ancdata)


def recv_fds(sock: socket.socket, num_fds: int):
    """Receive multiple fds + extra_data via Unix domain socket."""
    data_buf = bytearray(256)
    anc_buf_size = socket.CMSG_SPACE(num_fds * struct.calcsize("i"))

    nbytes, ancdata, flags, addr = sock.recvmsg_into(
        [data_buf], anc_buf_size, anc_buf_size
    )
    data = bytes(data_buf[:nbytes])

    fds = []
    for level, ctype, cdata in ancdata:
        if level == socket.SOL_SOCKET and ctype == socket.SCM_RIGHTS:
            num_received = len(cdata) // struct.calcsize("i")
            fds = list(
                struct.unpack(
                    f"{num_received}i", cdata[: num_received * struct.calcsize("i")]
                )
            )
            break
    if not fds:
        raise RuntimeError("did not receive fds via SCM_RIGHTS")
    return fds, data


# ---- CUDA Runtime (via ctypes) ----
def load_cudart():
    candidates = [
        "libcudart.so",
        "libcudart.so.12",
        "libcudart.so.11.0",
        "/usr/local/cuda/lib64/libcudart.so",
    ]
    for lib in candidates:
        try:
            return ctypes.CDLL(lib)
        except OSError:
            continue
    return None


cudart = load_cudart()

if cudart:
    cudart.cudaLaunchHostFunc.argtypes = [
        ctypes.c_void_p,
        ctypes.CFUNCTYPE(None, ctypes.c_void_p),
        ctypes.c_void_p,
    ]
    cudart.cudaLaunchHostFunc.restype = ctypes.c_int


# ---- Layer-wise transfer components ----


class FlexKVLayerLoadingEvent:
    def __init__(self, num_layers: int):
        self._num_layers = num_layers
        self.load_event_fds: List[int] = [
            eventfd(0, EFD_SEMAPHORE) for _ in range(num_layers)
        ]
        self._finished = True
        self._last_layer_wait_count = 0
        self.wait_remaining: List[int] = [2] * num_layers

    def reset_for_new_transfer(self):
        self._finished = False
        self._last_layer_wait_count = 0
        self.wait_remaining = [2] * self._num_layers

    def wait(self, layer_index: int):
        assert 0 <= layer_index < self._num_layers
        eventfd_read(self.load_event_fds[layer_index])
        if layer_index == self._num_layers - 1:
            self._last_layer_wait_count += 1
            if self._last_layer_wait_count >= 2:
                self._finished = True

    def close(self):
        for fd in self.load_event_fds:
            try:
                os.close(fd)
            except Exception:
                pass
        self.load_event_fds.clear()

    def __del__(self):
        self.close()


class FlexKVLayerDoneCounter:
    """Triple-buffered layer-wise transfer counter using eventfds.

    Provides the same ``set_consumer`` / ``wait_until`` interface expected by
    the KV cache memory pool so that the attention backend can synchronize
    per-layer with an in-flight host->device transfer.

    Because ``ExtendedRadixCache`` assigns monotonically increasing *task_ids*
    while this counter cycles through a fixed number of producer slots, a
    ``_task_to_producer`` mapping translates between the two id spaces.
    """

    def __init__(self, num_layers: int):
        self.num_layers = num_layers
        self.num_counters = 3
        self.events: List[FlexKVLayerLoadingEvent] = [
            FlexKVLayerLoadingEvent(num_layers) for _ in range(self.num_counters)
        ]
        self.producer_index = -1
        self.consumer_index = -1
        self._task_to_producer: Dict[int, int] = {}

    def register_task(self, task_id: int, producer_id: int):
        self._task_to_producer[task_id] = producer_id

    def update_producer(self) -> int:
        self.producer_index = (self.producer_index + 1) % self.num_counters
        assert self.events[
            self.producer_index
        ]._finished, "Producer event should be finished before reuse"
        return self.producer_index

    def set_consumer(self, index: int):
        if index < 0:
            self.consumer_index = -1
            return
        producer_id = self._task_to_producer.pop(index, None)
        if producer_id is not None:
            self.consumer_index = producer_id
        else:
            self.consumer_index = index

    def wait_until(self, threshold: int):
        if self.consumer_index < 0:
            return
        event = self.events[self.consumer_index]
        if event.wait_remaining[threshold] <= 0:
            return
        event.wait_remaining[threshold] -= 1
        event.wait(threshold)

    def reset(self):
        self.producer_index = -1
        self.consumer_index = -1
        self._task_to_producer.clear()

    def __del__(self):
        for event in self.events:
            event.close()
        self.events.clear()


# ---- FlexKV Connector ----


class FlexKVConnector(BaseKVConnector):
    """KV cache connector backed by FlexKV's distributed cache system.

    Implements ``BaseKVConnector`` so it can be used with
    ``ExtendedRadixCache`` via ``--kv-connector-cls``.
    """

    def __init__(
        self,
        params: Any = None,
        server_args: Any = None,
        tp_group: Any = None,
        tp_rank: int = 0,
        kvcache: Any = None,
    ):
        super().__init__(params, server_args, tp_group, tp_rank, kvcache)

        model_config = ModelConfig.from_server_args(server_args)

        self.flexkv_config = FlexKVConfig.from_env()
        self.flexkv_config.post_init_from_sglang_config(
            sglang_config=model_config,
            tp_size=server_args.tp_size,
            page_size=params.page_size,
        )

        self.tp_size = server_args.tp_size
        self.rank = tp_rank

        self.k_pool = getattr(kvcache, "k_buffer", None)
        self.v_pool = getattr(kvcache, "v_buffer", None)

        if self.rank == 0:
            self.kv_manager = KVManager(
                model_config=self.flexkv_config.model_config,
                cache_config=self.flexkv_config.cache_config,
                server_recv_port=self.flexkv_config.server_recv_port,
            )
            self.kv_manager.start()

        self.tp_client = KVTPClient(self.flexkv_config.gpu_register_port, 0, self.rank)
        self._register_to_server(self.k_pool, self.v_pool)

        self.num_layers = model_config.num_hidden_layers if model_config else 0
        self.enable_layerwise_transfer = bool(
            int(os.getenv("FLEXKV_ENABLE_LAYERWISE_TRANSFER", "1"))
        )
        self.layerwise_eventfd_socket = os.getenv(
            "FLEXKV_LAYERWISE_EVENTFD_SOCKET", "/tmp/flexkv_layerwise_eventfd.sock"
        )
        self._layer_done_counter: Optional[FlexKVLayerDoneCounter] = None
        self._worker_connected = False

        self._init_layer_transfer_components()

        if self._layer_done_counter is not None and kvcache is not None:
            kvcache.register_layer_transfer_counter(self._layer_done_counter)

        # rid -> flexkv_task_id (pending loads awaiting start_load_kv)
        self._pending_loads: Dict[str, int] = {}
        # ext_task_id -> producer_id (layerwise loads in flight)
        self._ongoing_loads: Dict[int, int] = {}
        # ext_task_ids whose load has completed
        self._completed_loads: List[int] = []
        # ext_task_id -> flexkv_task_id (stores in flight, rank 0 only)
        self._ongoing_stores: Dict[int, int] = {}
        # ext_task_ids whose store has completed or was skipped (rank 0 only)
        self._completed_stores: List[int] = []

        if self.rank == 0:
            while not self.kv_manager.is_ready():
                time.sleep(3)
                logger.info("[FlexKV] Waiting for FlexKV to be ready...")
            logger.info("[FlexKV] FlexKV is ready")

        logger.info(
            "[FlexKV] Connector initialized for rank %d, layerwise_transfer=%s",
            self.rank,
            self.enable_layerwise_transfer,
        )

    # ---- BaseKVConnector abstract methods ----

    def get_new_hit_length(
        self,
        token_ids: List[int],
        token_mask: torch.Tensor,
        update_state_for_load: bool = False,
        rid: Optional[str] = None,
    ) -> int:
        hit_length = 0
        flexkv_task_id = -1

        if self.rank == 0:
            token_ids_np = np.array(token_ids, dtype=np.int64)
            flexkv_task_id, matched_mask = self.kv_manager.get_match(
                token_ids=token_ids_np,
                token_mask=token_mask,
            )
            hit_length = int(matched_mask.sum()) if matched_mask is not None else 0
            if not update_state_for_load:
                self.kv_manager.cancel([flexkv_task_id])

        if self.tp_group is not None and self.tp_size > 1:
            data = broadcast_pyobj(
                [{"hit_length": hit_length, "task_id": flexkv_task_id}],
                self.rank,
                self.tp_group,
                src=0,
            )[0]
            hit_length = data["hit_length"]
            flexkv_task_id = data["task_id"]

        if update_state_for_load and rid is not None and hit_length > 0:
            self._pending_loads[rid] = flexkv_task_id
        return hit_length

    def release_load_state(self, rid: str) -> None:
        self._pending_loads.pop(rid, None)

    def start_load_kv(
        self,
        task_id: int,
        load_ops: List[LoadOperation],
    ) -> None:
        flexkv_task_ids: List[int] = []
        slot_mappings: List[torch.Tensor] = []

        for op in load_ops:
            fkv_tid = self._pending_loads.pop(op.rid, -1)
            if fkv_tid < 0:
                continue
            flexkv_task_ids.append(fkv_tid)
            indices = op.device_indices
            slot_mappings.append(indices.cpu() if indices.is_cuda else indices)

        if not flexkv_task_ids:
            self._completed_loads.append(task_id)
            return

        if self.enable_layerwise_transfer and self._layer_done_counter is not None:
            producer_id = self._layer_done_counter.update_producer()
            self._layer_done_counter.events[producer_id].reset_for_new_transfer()
            self._layer_done_counter.register_task(task_id, producer_id)

            if self.rank == 0:
                self.kv_manager.launch(
                    task_ids=flexkv_task_ids,
                    slot_mappings=slot_mappings,
                    as_batch=True,
                    layerwise_transfer=True,
                    counter_id=producer_id,
                )

            self._ongoing_loads[task_id] = producer_id
        else:
            if self.rank == 0:
                self.kv_manager.launch(
                    task_ids=flexkv_task_ids,
                    slot_mappings=slot_mappings,
                    as_batch=True,
                    layerwise_transfer=False,
                )
                response = self.kv_manager.wait(flexkv_task_ids, timeout=30.0)
                if not all(
                    tid in response and response[tid].status == KVResponseStatus.SUCCESS
                    for tid in flexkv_task_ids
                ):
                    logger.warning(
                        "[FlexKV] Some tasks failed in non-layerwise transfer"
                    )

            if self.tp_group is not None and self.tp_size > 1:
                torch.distributed.barrier(self.tp_group)

            self._completed_loads.append(task_id)

    def check_completed_load_tasks(self) -> List[int]:
        for ext_tid, producer_id in list(self._ongoing_loads.items()):
            if self._layer_done_counter.events[producer_id]._finished:
                self._completed_loads.append(ext_tid)
                del self._ongoing_loads[ext_tid]

        result = list(self._completed_loads)
        self._completed_loads.clear()
        return result

    def start_store_kv(
        self,
        task_id: int,
        token_ids: List[int],
        kv_indices: torch.Tensor,
    ) -> None:
        if self.rank != 0:
            return

        try:
            token_ids_np = np.array(token_ids, dtype=np.int64)
            fkv_task_id, unmatched_mask = self.kv_manager.put_match(
                token_ids=token_ids_np, token_mask=None
            )

            if unmatched_mask.sum() > 0:
                filtered = kv_indices[unmatched_mask]
                slot_mapping = filtered.cpu() if filtered.is_cuda else filtered
                self.kv_manager.launch(
                    task_ids=[fkv_task_id], slot_mappings=[slot_mapping]
                )
                self._ongoing_stores[task_id] = fkv_task_id
            else:
                self._completed_stores.append(task_id)
        except Exception as e:
            logger.error("[FlexKV] start_store_kv failed: %s", e)
            self._completed_stores.append(task_id)

    def check_completed_store_tasks(self) -> List[int]:
        completed_ext_ids = list(self._completed_stores)
        self._completed_stores.clear()

        if self.rank == 0 and self._ongoing_stores:
            fk_to_ext = {v: k for k, v in self._ongoing_stores.items()}
            completed_dict = self.kv_manager.try_wait(task_ids=list(fk_to_ext.keys()))
            for fk_tid in completed_dict:
                ext_tid = fk_to_ext[fk_tid]
                completed_ext_ids.append(ext_tid)
                del self._ongoing_stores[ext_tid]

        if self.tp_group is not None and self.tp_size > 1:
            completed_ext_ids = broadcast_pyobj(
                [completed_ext_ids] if self.rank == 0 else [None],
                self.rank,
                self.tp_group,
                src=0,
            )[0]

        return completed_ext_ids

    # ---- Optional overrides ----

    def cancel_prefetch(self, rid: str) -> None:
        self._pending_loads.pop(rid, None)

    @property
    def layer_done_counter(self) -> Any:
        return self._layer_done_counter

    def register_layer_transfer_counter(self, kvcache: Any) -> None:
        if self._layer_done_counter is not None:
            kvcache.register_layer_transfer_counter(self._layer_done_counter)

    def reset(self) -> None:
        self._pending_loads.clear()
        self._ongoing_loads.clear()
        self._completed_loads.clear()

        if self.rank == 0:
            for fk_tid in list(self._ongoing_stores.values()):
                if fk_tid >= 0:
                    self._wait_flexkv_task(fk_tid)
        self._ongoing_stores.clear()
        self._completed_stores.clear()

        if self._layer_done_counter is not None:
            self._layer_done_counter.reset()

    def shutdown(self) -> None:
        if self.rank == 0:
            self.kv_manager.shutdown()

    # ---- Private helpers ----

    def _wait_flexkv_task(self, fk_task_id: int, timeout: float = 20.0) -> bool:
        if fk_task_id < 0 or self.rank != 0:
            return True
        try:
            response = self.kv_manager.wait([fk_task_id], timeout=timeout)
            return (
                fk_task_id in response
                and response[fk_task_id].status == KVResponseStatus.SUCCESS
            )
        except Exception as e:
            logger.error("[FlexKV] wait task failed: %s", e)
            return False

    def _register_to_server(
        self,
        k_caches: List[torch.Tensor],
        v_caches: List[torch.Tensor],
    ) -> None:
        assert len(k_caches) == len(v_caches)
        assert (
            k_caches[0].ndim == 3
        ), f"Expected 3D tensor, got shape={k_caches[0].shape}"

        num_layer = len(k_caches)
        num_blocks, num_kv_heads, head_size = k_caches[0].shape

        gpu_layout = KVCacheLayout(
            type=KVCacheLayoutType.LAYERFIRST,
            num_layer=num_layer,
            num_block=num_blocks,
            tokens_per_block=1,
            num_head=num_kv_heads,
            head_size=head_size,
            is_mla=False,
        )
        self.tp_client.register_to_server(k_caches + v_caches, gpu_layout)
        logger.info("[FlexKV] Registered KV caches to server")

    def _init_layer_transfer_components(self):
        if not self.enable_layerwise_transfer:
            self._layer_done_counter = None
            self._worker_connected = False
            logger.info("[FlexKV] Rank %d: Layerwise transfer disabled", self.rank)
            return

        self._layer_done_counter = FlexKVLayerDoneCounter(self.num_layers)
        self._send_eventfds_to_worker()
        logger.info("[FlexKV] Rank %d: Initialized layerwise transfer", self.rank)

    def _send_eventfds_to_worker(
        self, max_retries: int = 180, retry_interval: float = 1.0
    ):
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)

        for attempt in range(max_retries):
            try:
                sock.connect(self.layerwise_eventfd_socket)
                logger.info("[FlexKV] Rank %d: Connected to worker socket", self.rank)
                break
            except (FileNotFoundError, ConnectionRefusedError):
                if attempt == max_retries - 1:
                    sock.close()
                    raise RuntimeError(
                        f"[FlexKV] Rank {self.rank}: Failed to connect "
                        f"after {max_retries} attempts"
                    )
                if attempt % 10 == 0:
                    logger.info(
                        "[FlexKV] Rank %d: Worker not ready, retrying...",
                        self.rank,
                    )
                time.sleep(retry_interval)

        try:
            num_counters = self._layer_done_counter.num_counters
            metadata = struct.pack(
                "iiii", self.rank, self.tp_size, self.num_layers, num_counters
            )
            sock.sendall(metadata)

            for counter_id in range(num_counters):
                fds = self._layer_done_counter.events[counter_id].load_event_fds
                send_fds(sock, fds, struct.pack("i", counter_id))

            self._worker_connected = True
            logger.info(
                "[FlexKV] Rank %d: Sent %d sets of eventfds",
                self.rank,
                num_counters,
            )
        except Exception as e:
            sock.close()
            raise RuntimeError(
                f"[FlexKV] Rank {self.rank}: Failed to send eventfds: {e}"
            )
