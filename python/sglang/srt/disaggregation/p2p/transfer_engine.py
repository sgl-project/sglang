import logging
import time
from typing import List

import cuda_p2p_transfer

from sglang.srt.utils.network import get_free_port

logger = logging.getLogger(__name__)


class SyncCompletedHandle:
    def is_done(self) -> bool:
        return True

    def wait(self) -> None:
        return


class CompositeTransferHandle:
    def __init__(self, handles):
        self._handles = list(handles or [])

    def is_done(self) -> bool:
        for h in self._handles:
            fn = getattr(h, "is_done", None)
            if fn is None or not fn():
                return False
        return True

    def wait(self) -> None:
        for h in self._handles:
            fn = getattr(h, "wait", None)
            if fn is not None:
                fn()
            else:
                while not getattr(h, "is_done", lambda: True)():
                    time.sleep(0.001)


class P2PTransferEngine:
    def __init__(self, hostname: str, physical_gpu_id: int):
        self.physical_gpu_id = physical_gpu_id
        self._rpc_port = get_free_port()
        self.hostname = hostname
        self.session_id = f"{self.hostname}:{self._rpc_port}"
        try:
            self.p2p_engine = cuda_p2p_transfer.CudaP2PTransfer(physical_gpu_id)
            self.supports_transfer_many = hasattr(self.p2p_engine, "transfer_many")
            logger.info(
                f"P2P engine initialized successfully "
                f"(GPU {physical_gpu_id}, transfer_many: {self.supports_transfer_many})"
            )
        except Exception as e:
            logger.exception(
                f"Failed to initialize P2P engine for GPU {physical_gpu_id}"
            )
            raise RuntimeError(f"P2P engine initialization failed: {e}") from e

    def register_buffer(self, ptr):
        try:
            handle = self.p2p_engine.register_buffer(ptr)
            handle_size = len(handle) if isinstance(handle, bytes) else "N/A"
            logger.debug(
                f"Successfully registered buffer at ptr={ptr:#x}, "
                f"handle size={handle_size}"
            )
            return handle
        except Exception as e:
            logger.error(f"Failed to register buffer at ptr={ptr}: {e}")
            raise RuntimeError(f"Buffer registration failed: {e}")

    def transfer(
        self,
        src_ptr: int,
        src_dev: int,
        dst_handle: bytes,
        dst_dev: int,
        dst_offset: int,
        length: int,
    ):
        try:
            h = self.p2p_engine.transfer(
                src_ptr, src_dev, dst_handle, dst_dev, dst_offset, length
            )
            if isinstance(h, int):
                if h != 0:
                    logger.error(
                        f"P2P transfer failed with code {h} "
                        f"(src_ptr={src_ptr:#x}, length={length})"
                    )
                    raise RuntimeError(f"P2P transfer failed with code {h}")
                logger.debug(
                    f"P2P transfer completed immediately "
                    f"(src_ptr={src_ptr:#x}, length={length})"
                )
                return SyncCompletedHandle()
            return h
        except Exception as e:
            logger.exception(
                f"Transfer failed (src_ptr={src_ptr:#x}, "
                f"dst_dev={dst_dev}, length={length}): {e}"
            )
            raise

    def transfer_many(
        self,
        src_ptrs: List[int],
        src_devs: List[int],
        dst_handles: List[bytes],
        dst_devs: List[int],
        dst_offsets: List[int],
        lengths: List[int],
    ):
        try:
            if self.supports_transfer_many:
                h = self.p2p_engine.transfer_many(
                    src_ptrs, src_devs, dst_handles, dst_devs, dst_offsets, lengths
                )
                logger.debug(
                    f"Submitted batch transfer with {len(src_ptrs)} operations"
                )
                return h
            handles = []
            for sp, sd, dh, dd, off, ln in zip(
                src_ptrs, src_devs, dst_handles, dst_devs, dst_offsets, lengths
            ):
                handles.append(self.transfer(sp, sd, dh, dd, off, ln))
            return CompositeTransferHandle(handles)
        except Exception as e:
            logger.exception(f"transfer_many failed: {e}")
            raise

    def register_d_handle(self, dst_handle: bytes) -> int:
        try:
            result = self.p2p_engine.register_d_handle(dst_handle)
            if result != 0:
                logger.error(
                    f"Failed to register destination handle "
                    f"{dst_handle.hex()} with code {result}"
                )
            return result
        except Exception as e:
            logger.exception(f"Destination handle registration failed: {e}")
            return 1

    def get_session_id(self):
        return self.session_id
