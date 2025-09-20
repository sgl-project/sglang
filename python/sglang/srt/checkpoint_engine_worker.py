# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
Checkpoint-engine integration for SGLang.
This module provides weight update functionality via IPC for checkpoint-engine compatibility.
"""
import gc
import logging
from typing import Callable, Dict, List, Optional, Tuple, TypedDict
import torch
import zmq
logger = logging.getLogger(__name__)
class FlattenedTensorMetadata(TypedDict):
    name: str
    shape: torch.Size
    dtype: torch.dtype
    # specify the start offset of this tensor in shared ipc_buffer tensor
    offset: int
def _rebuild_ipc(handle: tuple[Callable, tuple], device_id: Optional[int] = None) -> torch.Tensor:
    """Rebuild a tensor from IPC handle, adapting to current device."""
    func, args = handle
    list_args = list(args)
    if device_id is not None:
        # the key is to change device id to the current device id
        # in case two processes have different CUDA_VISIBLE_DEVICES
        list_args[6] = device_id
    buffer = func(*list_args)
    return buffer
def _extract_weights(payload: List[FlattenedTensorMetadata], buffer: torch.Tensor) -> List[Tuple[str, torch.Tensor]]:
    """Extract named weights from flattened buffer based on metadata."""
    assert buffer is not None
    weights: List[Tuple[str, torch.Tensor]] = []
    for item in payload:
        shape = item["shape"]
        if isinstance(shape, (list, tuple)):
            shape = torch.Size(shape)
        assert isinstance(shape, torch.Size)
        dtype, offset = item["dtype"], item["offset"]
        size = dtype.itemsize * shape.numel()
        tensor = buffer[offset : offset + size].view(dtype=dtype).view(shape)
        weights.append((item["name"], tensor))
    return weights
def update_weights_from_ipc(
    zmq_ctx: zmq.Context,
    zmq_handle: str,
    device_id: int,
    *,
    run: Callable[[List[Tuple[str, torch.Tensor]]], None],
    post_hook: Callable[[], None] = None,
):
    """
    Core IPC weight update logic for SGLang.
    Args:
        zmq_ctx: ZMQ context for communication
        zmq_handle: ZMQ socket path for this device
        device_id: Current device ID
        run: Function to apply weights to the model (model.load_weights)
        post_hook: Optional post-processing function
    """
    socket = zmq_ctx.socket(zmq.REP)
    socket.connect(zmq_handle)
    buffer: Optional[torch.Tensor] = None
    logger.info(f"Starting IPC weight update on device {device_id}, socket: {zmq_handle}")
    try:
        while True:
            payload: tuple[Callable, tuple] | List[FlattenedTensorMetadata] | None = socket.recv_pyobj()
            if payload is None:
                # means the update is done
                logger.info(f"Weight update complete on device {device_id}")
                if post_hook is not None:
                    post_hook()
                torch.cuda.synchronize()
                socket.send(b"")
                break
            if isinstance(payload, tuple):
                # an ipc handle that we can use to rebuild GPU tensor
                logger.debug(f"Received IPC handle on device {device_id}")
                buffer = _rebuild_ipc(payload, device_id)
                assert buffer.dtype == torch.uint8
                socket.send(b"")
                continue
            assert isinstance(payload, list)
            # weight metadata list - extract and load weights
            logger.debug(f"Received {len(payload)} weight tensors on device {device_id}")
            weights = _extract_weights(payload, buffer)
            run(weights)
            torch.cuda.synchronize()
            socket.send(b"")
    except Exception as e:
        logger.error(f"Error in IPC weight update on device {device_id}: {e}")
        raise
    finally:
        socket.close()
        del buffer
        gc.collect()
        torch.cuda.empty_cache()
        logger.info(f"Cleaned up IPC weight update on device {device_id}")
class SGLangCheckpointEngineWorkerExtension:
    """
    Worker extension for SGLang to support checkpoint-engine IPC weight updates.
    This class provides the interface needed for checkpoint-engine integration.
    """
    def __init__(self):
        self._zmq_ctx: Optional[zmq.Context] = None
    def get_device_uuid(self) -> str:
        """Get the UUID of current device."""
        # We need to implement this to get the device UUID
        # This will be overridden when integrated into SGLang's worker
        raise NotImplementedError("This method should be overridden by SGLang integration")
    def get_device_id(self) -> int:
        """Get the device ID."""
        raise NotImplementedError("This method should be overridden by SGLang integration")
    def get_model_loader(self) -> Callable:
        """Get the model weight loader function."""
        raise NotImplementedError("This method should be overridden by SGLang integration")
    def get_post_hook(self) -> Optional[Callable]:
        """Get the post-processing hook after weight loading."""
        return None
    def update_weights_from_ipc(self, zmq_handles: Dict[str, str]):
        """
        Update weights from IPC communication.
        Args:
            zmq_handles: Dict mapping device UUID to ZMQ socket path
        """
        if self._zmq_ctx is None:
            self._zmq_ctx = zmq.Context()
        device_uuid = self.get_device_uuid()
        device_id = self.get_device_id()
        if device_uuid not in zmq_handles:
            raise ValueError(f"Device UUID {device_uuid} not found in zmq_handles: {list(zmq_handles.keys())}")
        update_weights_from_ipc(
            self._zmq_ctx,
            zmq_handles[device_uuid],
            device_id=device_id,
            run=self.get_model_loader(),
            post_hook=self.get_post_hook(),
        )
