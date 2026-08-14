import logging
import threading
import time
from multiprocessing import shared_memory
from typing import Tuple

import numpy as np
import torch
from torch.multiprocessing.reductions import reduce_tensor

from sglang.srt.utils.stale_shm_cleanup import make_shm_name

logger = logging.getLogger(__name__)


class ShmSyncBuffer:
    def __init__(self, byte_size: int = 4):
        self.buffer = shared_memory.SharedMemory(
            create=True, size=byte_size, name=make_shm_name("sync")
        )
        self.buffer_wrapper = np.ndarray(1, dtype=np.float32, buffer=self.buffer.buf)
        self.buffer_wrapper *= 0
        self.meta_data = {
            "handle": self.buffer.name,
            "shape": self.buffer_wrapper.shape,
            "dtype": self.buffer_wrapper.dtype,
        }

    def close(self):
        if self.buffer:
            self.buffer.close()
            self.buffer.unlink()


class MmItemMemoryChunk:
    def __init__(self, area: Tuple, sync_buffer: ShmSyncBuffer, tp_size: int):
        self.area = area
        self.sync_flag = sync_buffer
        self.tp_size = tp_size

    @property
    def mem_size(self):
        return self.area[1] - self.area[0]

    @property
    def start(self):
        return self.area[0]

    @property
    def end(self):
        return self.area[1]

    def try_to_recycle(self) -> bool:
        val = float(self.sync_flag.buffer_wrapper.item())
        logger.debug(
            f"[try_to_recycle] area={self.area}, flag={val}, tp_size={self.tp_size}"
        )

        if val == float(self.tp_size):
            self.sync_flag.buffer_wrapper *= 0.0
            return True

        return False


class MmItemMemoryPool:
    def __init__(self, memory_size, recycle_interval, base_npu_id, tp_size):
        self.tp_size = tp_size
        self.memory_pool = torch.empty(
            memory_size, dtype=torch.int8, device=f"npu:{base_npu_id}"
        ).contiguous()
        self._pool_ipc_handle = reduce_tensor(self.memory_pool)
        self._pool_device_index = self.memory_pool.device.index

        self.sync_flag_list = []

        init_chunk = MmItemMemoryChunk(
            (0, memory_size), self.pop_sync_buffer(), tp_size
        )
        self.available_chunks = [init_chunk]
        self.occupied_chunks = []

        self._lock = threading.Lock()
        self._pool_full_warned = False

        self._recycle_interval = recycle_interval
        self._stop_recycler = False
        self._recycle_thread = threading.Thread(
            target=self._recycle_loop, name="MmItemMemoryPoolRecycler", daemon=True
        )
        self._recycle_thread.start()

        logger.info(
            f"[NPU IPC Pool] Allocated NPU memory pool: size={memory_size / (1024*1024):.1f} MB, "
            f"device=npu:{base_npu_id}, recycle_interval={recycle_interval}s"
        )

    def shutdown(self):
        self._stop_recycler = True
        if self._recycle_thread.is_alive():
            self._recycle_thread.join(timeout=1.0)

    def _recycle_loop(self):
        while not self._stop_recycler:
            try:
                with self._lock:
                    self.recycle_chunks()
                    self.merge_chunks()
            except Exception as e:
                logger.warning(
                    f"[MmItemMemoryPool(NPU)] recycle loop error: {e}", exc_info=True
                )

            time.sleep(self._recycle_interval)

    def clear_sync_flag_list(self):
        self.sync_flag_list.clear()

    def pop_sync_buffer(self):
        if len(self.sync_flag_list) == 0:
            try:
                new_sync_buffer = ShmSyncBuffer()
                return new_sync_buffer
            except:
                logger.info("allocate shm buffer failed")
                raise RuntimeError
        else:
            return self.sync_flag_list.pop()

    def push_sync_buffer(self, sync_buffer):
        self.sync_flag_list.append(sync_buffer)

    def get_available_chunk(self, src_tensor: torch.Tensor) -> MmItemMemoryChunk:
        src_tensor_size = src_tensor.numel() * src_tensor.element_size()
        min_size = self.memory_pool.numel() * self.memory_pool.element_size() + 1
        selected_chunk = None
        for chunk in self.available_chunks:
            if chunk.mem_size >= src_tensor_size:
                if chunk.mem_size < min_size:
                    min_size = chunk.mem_size
                    selected_chunk = chunk

        if not selected_chunk:
            return None

        occupied_chunk_area = (
            selected_chunk.start,
            selected_chunk.start + src_tensor_size,
        )
        occupied_chunk_sync_flag = selected_chunk.sync_flag
        new_occupied_chunk = MmItemMemoryChunk(
            occupied_chunk_area, occupied_chunk_sync_flag, self.tp_size
        )

        self.occupied_chunks.append(new_occupied_chunk)
        self.available_chunks.remove(selected_chunk)

        available_split_chunk_area = (new_occupied_chunk.end, selected_chunk.end)
        if available_split_chunk_area[0] != available_split_chunk_area[1]:
            split_available_chunk = MmItemMemoryChunk(
                available_split_chunk_area, selected_chunk.sync_flag, self.tp_size
            )
            self.available_chunks.append(split_available_chunk)
            self.occupied_chunks.pop()
        else:
            self.occupied_chunks.pop()

        return new_occupied_chunk

    def return_a_slice_tensor_with_flag(self, tensor: torch.Tensor):
        assert tensor.is_contiguous(), "tensor must be contiguous"
        sync_flag = None
        available_slice = None
        byte_offset = None
        with self._lock:
            selected_chunk = self.get_available_chunk(tensor)
            if selected_chunk is not None:
                available_slice = self.memory_pool[
                    selected_chunk.start : selected_chunk.end
                ]
                byte_offset = selected_chunk.start
                sync_flag = selected_chunk.sync_flag
                logger.info(
                    f"[NPU IPC] Pool allocated: tensor_size={tensor.numel() * tensor.element_size() / (1024*1024):.2f} MB, "
                    f"chunk=[{selected_chunk.start}, {selected_chunk.end}], "
                    f"pool_remaining={self.memory_pool.numel() * self.memory_pool.element_size() / (1024*1024):.1f} MB"
                )
            else:
                if not self._pool_full_warned:
                    logger.warning(
                        "[NPU IPC] Pool is FULL, falling back to CPU transport"
                    )
                    self._pool_full_warned = True

        return sync_flag, available_slice, byte_offset

    def wrap_tensor(self, tensor: torch.Tensor, *, use_pool_handle_cache: bool):
        sync_flag, available_slice, byte_offset = self.return_a_slice_tensor_with_flag(
            tensor
        )
        if isinstance(available_slice, torch.Tensor):
            available_slice.copy_(tensor.view(torch.int8).view(-1), non_blocking=True)
            return NpuIpcTensorTransportProxy(
                data=available_slice,
                info_data=tensor,
                sync_buffer_meta=sync_flag,
                pool_ipc_handle=(
                    self._pool_ipc_handle if use_pool_handle_cache else None
                ),
                pool_byte_offset=byte_offset,
                pool_device_index=self._pool_device_index,
            )
        return None

    def reclaim_chunk(self, chunk: MmItemMemoryChunk):
        self.occupied_chunks.remove(chunk)
        self.available_chunks.append(chunk)

    def recycle_chunks(self):
        to_recycle_chunks = []
        for chunk in self.occupied_chunks:
            if chunk.try_to_recycle():
                to_recycle_chunks.append(chunk)

        for chunk in to_recycle_chunks:
            self.reclaim_chunk(chunk)

    def merge_chunks(self):
        if len(self.available_chunks) <= 1:
            return

        self.available_chunks.sort(key=lambda chunk: chunk.start)
        merged_chunks = []
        for chunk in self.available_chunks:
            if not merged_chunks:
                merged_chunks.append(chunk)
            else:
                if chunk.start == merged_chunks[-1].end:
                    to_merge_chunk = merged_chunks.pop()
                    to_merge_chunk_sync = to_merge_chunk.sync_flag
                    merged_chunk_area = (to_merge_chunk.start, chunk.end)
                    merged_chunks.append(
                        MmItemMemoryChunk(
                            merged_chunk_area, to_merge_chunk_sync, self.tp_size
                        )
                    )
                    self.push_sync_buffer(chunk.sync_flag)
                else:
                    merged_chunks.append(chunk)

        self.available_chunks = merged_chunks


class NpuIpcTensorTransportProxy:
    """
    A torch.tensor's proxy used to do inter-process data-sharing on NPU
    including:

    torch.tensor(on npu)'s IPC handle infos
    a shm sync buffer's meta data which is used to sync between different process
    """

    def __init__(
        self,
        data: torch.Tensor,
        info_data: torch.Tensor,
        sync_buffer_meta,
        pool_ipc_handle=None,
        pool_byte_offset: int = 0,
        pool_device_index: int = 0,
    ):
        if (not isinstance(data, torch.Tensor)) or (
            not isinstance(info_data, torch.Tensor)
        ):
            raise TypeError(
                f"Input 'data' must be a torch.Tensor, but got {type(data)}"
            )

        if pool_ipc_handle is not None:
            self.proxy_state = {
                "ipc_extra": {
                    "pool_handle": pool_ipc_handle,
                    "pool_byte_offset": pool_byte_offset,
                    "pool_device_index": pool_device_index,
                    "shape": data.shape,
                    "dtype": data.dtype,
                    "stride": data.stride(),
                    "storage_offset": 0,
                    "nbytes": data.numel() * data.element_size(),
                    "recons_shape": info_data.shape,
                    "recons_dtype": info_data.dtype,
                    "device_type": "npu",
                },
                "tensor_data": None,
            }
        else:
            self.proxy_state = self.get_proxy_state(data, info_data)
        self.reconstruct_tensor = None
        self._consumer_acknowledged = False

    @property
    def get_sync_flag(self):
        if not hasattr(self, "_sync_flag"):
            shm_name = (
                self.proxy_state["tensor_data"]
                .get("sync_buffer_meta", {})
                .get("handle")
            )
            if not shm_name:
                shm_name = self.proxy_state.get("sync_buffer_meta", {}).get("handle")
            if shm_name:
                self._sync_flag = ShmSyncBuffer.__new__(ShmSyncBuffer)
                self._sync_flag.buffer = shared_memory.SharedMemory(name=shm_name)
                self._sync_flag.buffer_wrapper = np.ndarray(
                    1, dtype=np.float32, buffer=self._sync_flag.buffer.buf
                )
                self._sync_flag.meta_data = self.proxy_state["tensor_data"][
                    "sync_buffer_meta"
                ]
            else:
                self._sync_flag = None
        return self._sync_flag

    @property
    def sync_data_meta(self):
        return self.proxy_state.get("tensor_data", {}).get("sync_buffer_meta", {})

    @property
    def sync_buffer(self):
        if not hasattr(self, "_sync_buffer"):
            shm_name = self.sync_data_meta.get("handle")
            if shm_name:
                self._sync_buffer = shared_memory.SharedMemory(name=shm_name)
            else:
                self._sync_buffer = None
        return self._sync_buffer

    @property
    def get_sync_buffer_data(self):
        if not self.sync_buffer:
            return None
        shape = self.sync_data_meta["shape"]
        dtype = self.sync_data_meta["dtype"]
        return np.ndarray(shape, dtype=dtype, buffer=self.sync_buffer.buf)

    def close_shm(self):
        if hasattr(self, "_sync_flag") and self._sync_flag:
            self._sync_flag.buffer.close()
            self._sync_flag = None
        if hasattr(self, "_sync_buffer") and self._sync_buffer:
            self._sync_buffer.close()
            self._sync_buffer = None

    def get_proxy_state(self, data, info_data):
        state = {}

        try:
            handle = reduce_tensor(data)
            state["ipc_extra"] = {
                "handle": handle,
                "shape": data.shape,
                "dtype": data.dtype,
                "stride": data.stride(),
                "device_index": data.device.index,
                "storage_offset": data.storage_offset(),
                "recons_shape": info_data.shape,
                "recons_dtype": info_data.dtype,
                "device_type": "npu",
            }
            state["tensor_data"] = None
            logger.info(
                f"[NPU IPC] Created IPC handle: data_shape={data.shape}, "
                f"device={data.device}, handle_type={type(handle).__name__}, "
                f"handle_len={len(handle) if isinstance(handle, tuple) else 'N/A'}"
            )
        except Exception:
            state["ipc_extra"] = None
            state["tensor_data"] = data
            logger.warning(
                f"[NPU IPC] Failed to create IPC handle, falling back to CPU"
            )

        return state

    def _reconstruct_from_ipc_extra(
        self, ipc_extra, *, use_cache: bool, rebuild_device_idx: int
    ):
        shape = ipc_extra["shape"]
        dtype = ipc_extra["dtype"]
        stride = ipc_extra["stride"]
        pool_handle = ipc_extra["pool_handle"]

        target_device = torch.device(f"npu:{rebuild_device_idx}")
        func, args = pool_handle
        list_args = list(args)
        for i, arg in enumerate(list_args):
            if isinstance(arg, torch.device):
                list_args[i] = target_device
            elif isinstance(arg, str) and arg.startswith("npu:"):
                list_args[i] = str(target_device)
        rebuild_tensor = func(*tuple(list_args))
        slice_tensor = rebuild_tensor.as_strided(
            size=shape,
            stride=stride,
            storage_offset=ipc_extra["storage_offset"],
        )

        return slice_tensor, target_device, None, None

    def _acknowledge_consumption(self, consumer_count: int = 1):
        try:
            sync_data = self.get_sync_buffer_data
            if sync_data is not None:
                sync_data += consumer_count
                self._consumer_acknowledged = True
        except Exception:
            pass

    def _copy_slice_tensor_to_target(
        self, slice_tensor, target_device, recons_shape, recons_dtype, consumer_count
    ):
        reconstructed_tensor = torch.zeros(
            recons_shape, dtype=recons_dtype, device=target_device
        )
        start = 0
        for i in range(consumer_count):
            end = start + slice_tensor.numel()
            reconstructed_tensor.view(torch.int8).view(-1)[start:end] = (
                slice_tensor.view(torch.int8).view(-1)
            )
            start = end
        return reconstructed_tensor

    def reconstruct_on_target_device(self, rebuild_device_idx, consumer_count: int = 1):
        rebuild_device = torch.device(f"npu:{rebuild_device_idx}")
        if (
            isinstance(self.reconstruct_tensor, torch.Tensor)
            and self.reconstruct_tensor.device == rebuild_device
        ):
            return self.reconstruct_tensor

        if self.proxy_state["ipc_extra"]:
            ipc_extra = self.proxy_state["ipc_extra"]
            recons_shape = ipc_extra["recons_shape"]
            recons_dtype = ipc_extra["recons_dtype"]

            if "pool_handle" in ipc_extra:
                logger.info(
                    f"[NPU IPC] Reconstruct from pool: shape={recons_shape}, device={rebuild_device}"
                )
                (
                    slice_tensor,
                    _target_device,
                    _cache_key,
                    _storage_to_cache,
                ) = self._reconstruct_from_ipc_extra(
                    ipc_extra,
                    use_cache=False,
                    rebuild_device_idx=rebuild_device_idx,
                )
            else:
                try:
                    original_handle = ipc_extra["handle"]
                    target_device = torch.device(f"npu:{rebuild_device_idx}")
                    logger.info(
                        f"[NPU IPC] Reconstruct from handle: device={target_device}"
                    )
                    func, args = original_handle
                    list_args = list(args)
                    for i, arg in enumerate(list_args):
                        if isinstance(arg, torch.device):
                            list_args[i] = target_device
                        elif isinstance(arg, str) and arg.startswith("npu:"):
                            list_args[i] = str(target_device)
                    rebuild_tensor = func(*tuple(list_args))
                    slice_tensor = rebuild_tensor.as_strided(
                        size=ipc_extra["shape"],
                        stride=ipc_extra["stride"],
                        storage_offset=ipc_extra["storage_offset"],
                    )
                except Exception as e:
                    logger.info("Failed to deserialize from NPU IPC handle (%s).", e)
                    raise

            reconstructed_tensor = self._copy_slice_tensor_to_target(
                slice_tensor,
                rebuild_device,
                recons_shape,
                recons_dtype,
                consumer_count,
            )
        elif isinstance(self.proxy_state["tensor_data"], torch.Tensor):
            logger.info(
                f"[NPU IPC] Reconstruct from tensor_data: device={rebuild_device}"
            )
            reconstructed_tensor = self.proxy_state["tensor_data"].to(
                rebuild_device, non_blocking=True
            )
        else:
            raise TypeError("invalid proxy_state")

        self.reconstruct_tensor = reconstructed_tensor
        logger.info(
            f"[NPU IPC] Reconstruct SUCCESS: final_shape={reconstructed_tensor.shape}, "
            f"device={reconstructed_tensor.device}, dtype={reconstructed_tensor.dtype}"
        )
        return self.reconstruct_tensor

    def get_reconstructed_tensor(
        self, use_cache: bool = False, rebuild_device_idx: int = None
    ):
        if self.reconstruct_tensor is not None:
            return self.reconstruct_tensor

        ipc_extra = self.proxy_state.get("ipc_extra")
        tensor_data = self.proxy_state.get("tensor_data")
        if ipc_extra is not None:
            if rebuild_device_idx is None:
                rebuild_device_idx = ipc_extra["pool_device_index"]
            (
                self.reconstruct_tensor,
                target_device,
                cache_key,
                storage_to_cache,
            ) = self._reconstruct_from_ipc_extra(
                ipc_extra, use_cache=use_cache, rebuild_device_idx=rebuild_device_idx
            )
        elif tensor_data is not None:
            self.reconstruct_tensor = tensor_data

        return self.reconstruct_tensor

    def is_lazy(self):
        return (
            self.proxy_state.get("ipc_extra") is not None
            and self.proxy_state.get("tensor_data") is None
            and not self._consumer_acknowledged
        )

    def serialize(self):
        serialized = {}
        if self.proxy_state.get("ipc_extra") is not None:
            serialized["ipc_extra"] = self.proxy_state["ipc_extra"]
        if self.proxy_state.get("tensor_data") is not None:
            serialized["tensor_data"] = self.proxy_state["tensor_data"]
        serialized["consumer_acknowledged"] = self._consumer_acknowledged
        return serialized

    @staticmethod
    def deserialize(data):
        proxy = NpuIpcTensorTransportProxy.__new__(NpuIpcTensorTransportProxy)
        proxy.reconstruct_tensor = None
        proxy._consumer_acknowledged = data.get("consumer_acknowledged", False)
        proxy.proxy_state = {}
        if "ipc_extra" in data:
            proxy.proxy_state["ipc_extra"] = data["ipc_extra"]
        if "tensor_data" in data:
            sync_buffer_meta = data["tensor_data"].pop("sync_buffer_meta", None)
            if sync_buffer_meta is not None:
                proxy.proxy_state["tensor_data"] = data["tensor_data"]
                proxy.proxy_state["tensor_data"]["sync_buffer_meta"] = sync_buffer_meta
            else:
                proxy.proxy_state["tensor_data"] = data["tensor_data"]
        return proxy
