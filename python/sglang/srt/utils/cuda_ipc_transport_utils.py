import logging
import threading
import time

import torch

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)


_ipc_collect_thread = None
_ipc_collect_lock = threading.Lock()


def start_ipc_collect_sweeper(interval_sec: float = 0.1):
    """Periodically run torch.cuda.ipc_collect() so the producer reclaims the
    per-tensor IPC limbo; without it producer GPU memory grows with the backlog."""
    global _ipc_collect_thread
    with _ipc_collect_lock:
        if _ipc_collect_thread is not None and _ipc_collect_thread.is_alive():
            return

        def _loop():
            while True:
                try:
                    torch.cuda.ipc_collect()
                except Exception as e:
                    logger.warning("ipc_collect sweep failed: %s", e)
                time.sleep(interval_sec)

        _ipc_collect_thread = threading.Thread(
            target=_loop, name="CudaIpcCollectSweeper", daemon=True
        )
        _ipc_collect_thread.start()
        logger.info(
            "Started CUDA IPC collect sweeper (interval=%.0f ms)", interval_sec * 1000
        )


class CudaIpcTensorTransportProxy:
    """Per-tensor CUDA IPC proxy: producer shares a CUDA tensor's handle, the
    consumer copies it out to host and releases the producer."""

    def __init__(
        self,
        handle,
        shape,
        dtype,
        stride,
        device_index,
        storage_offset,
    ):
        self.ipc_state = {
            "handle": handle,
            "shape": shape,
            "dtype": dtype,
            "stride": stride,
            "device_index": device_index,
            "storage_offset": storage_offset,
        }
        self._reconstructed = None

    def __getstate__(self):
        # The reconstructed tensor is consumer-local; never pickle it.
        state = self.__dict__.copy()
        state["_reconstructed"] = None
        return state

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor):
        """Producer side: share the tensor's IPC handle (no copy). Keep no storage
        ref; CudaIPCSentDataLimbo owns it, else the GPU stays pinned per request."""
        storage = tensor.untyped_storage()
        handle = storage._share_cuda_()
        return cls(
            handle=handle,
            shape=tensor.shape,
            dtype=tensor.dtype,
            stride=tensor.stride(),
            device_index=tensor.device.index,
            storage_offset=tensor.storage_offset(),
        )

    def reconstruct_on_target_device(self, rebuild_device_idx):
        """Consumer side: open the handle, copy to host, release the producer so
        queued requests hold no GPU memory. SGLANG_CUDA_IPC_KEEP_ON_DEVICE=1 keeps it on device."""
        if self._reconstructed is not None:
            return self._reconstructed

        handle = self.ipc_state["handle"]
        target_device = torch.device(f"cuda:{rebuild_device_idx}")
        redirected_handle = (rebuild_device_idx,) + tuple(handle)[1:]
        shape = self.ipc_state["shape"]
        dtype = self.ipc_state["dtype"]
        stride = self.ipc_state["stride"]
        s_offset = self.ipc_state["storage_offset"]

        keep_on_device = envs.SGLANG_CUDA_IPC_KEEP_ON_DEVICE.get()

        with torch.cuda.device(target_device):
            storage = torch.UntypedStorage._new_shared_cuda(*redirected_handle)
            view = torch.empty(0, dtype=dtype, device=target_device).set_(
                storage, storage_offset=s_offset, size=shape, stride=stride
            )

            if keep_on_device:
                # Keep a zero-copy view on device; producer stays pinned until freed.
                self._reconstructed = view
                return self._reconstructed

            # Copy out to host, then drop the view so the producer can reclaim.
            host_tensor = torch.empty(view.shape, dtype=view.dtype, device="cpu")
            host_tensor.copy_(view)
            del view
            del storage

        self._reconstructed = host_tensor
        return self._reconstructed
