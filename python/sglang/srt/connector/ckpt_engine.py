# SPDX-License-Identifier: Apache-2.0

import json
import logging
import subprocess
from collections import OrderedDict
from typing import Callable, Dict, Generator, List, Optional, Tuple, TypedDict
from urllib.parse import urlparse

import torch
import torch.distributed as dist
import zmq

from sglang.srt.connector import BaseConnector
from sglang.srt.utils import init_custom_process_group

logger = logging.getLogger(__name__)

# TODO: using dynamic port
CKPTENGINE_PORT = 33001


def _get_physical_gpu_id(rank: int) -> str:
    result = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True)
    if result.returncode != 0:
        raise ValueError(result.stdout)
    lines = result.stdout.strip().split("\n")
    for line in lines:
        if f"GPU {rank}" in line:
            uuid = line.split("UUID: ")[1].strip(")")
            return uuid
    raise ValueError(f"not found gpu{rank} uuid")


def _rebuild_ipc(
    handle: tuple[Callable, tuple], device_id: Optional[int] = None
) -> torch.Tensor:
    """
    Rebuilds a tensor from a shared memory IPC handle on the correct GPU device.
    """
    func, args = handle
    list_args = list(args)
    if device_id is not None:
        # This ensures the tensor is mapped to the current process's specific GPU.
        list_args[6] = device_id
    buffer = func(*list_args)
    return buffer


class FlattenedTensorMetadata(TypedDict):
    name: str
    shape: torch.Size
    dtype: torch.dtype
    # The starting offset of this tensor's data in the shared buffer.
    offset: int


class CkptEngineConnector(BaseConnector):

    def __init__(self, url: str, device: torch.device = "cpu"):
        super().__init__(url)
        self.url = url
        self.device = device
        self.zmq_handle = None
        self.zmq_ctx = None
        self.device_uuid = None
        self.socket = None
        self.buffer: Optional[torch.Tensor] = None
        self.local_rank = None
        self.final_state_dict = OrderedDict()
        self.pending_weights: Dict[str, torch.Tensor] = {}

    def get_zmq_handle(self, tp_rank: int):
        # FIXME: There needs a local rank
        self.device_uuid = _get_physical_gpu_id(tp_rank)
        socket = zmq.Context().socket(zmq.PULL)
        socket.bind(f"tcp://*:{CKPTENGINE_PORT + tp_rank}")
        try:
            raw_message = socket.recv()

            try:
                data = json.loads(raw_message.decode("utf-8"))

                if not isinstance(data, dict):
                    logger.warning("CKPTENGINE: Not exactly the socket handle.")
                else:
                    self.zmq_handle = data[self.device_uuid]

            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                logger.error(f"can not parse the socket raw message: {e}")

        except KeyboardInterrupt:
            print("\n shutting down the server.")
        finally:
            socket.close()

    def get_socket_handle(self, tp_rank: int):
        # FIXME: local_rank is not tp_rank
        self.local_rank = tp_rank
        self.get_zmq_handle(tp_rank)
        self.zmq_ctx = zmq.Context()
        self.socket = self.zmq_ctx.socket(zmq.REP)
        self.socket.connect(self.zmq_handle)

    # Implemented as a no-op to make BaseConnector interface consistent.
    def pull_files(
        self,
        allow_pattern: Optional[list[str]] = None,
        ignore_pattern: Optional[list[str]] = None,
    ) -> None:
        return

    def _merge_and_store(self, gate_key, gate_tensor, up_key, up_tensor):
        new_key = gate_key.replace("gate_proj", "gate_up_proj")
        merged_tensor = torch.cat([gate_tensor, up_tensor], dim=0)
        self.final_state_dict[new_key] = merged_tensor

    def _extract_weights(
        self, payload: list[FlattenedTensorMetadata], buffer: torch.Tensor
    ) -> list[tuple[str, torch.Tensor]]:
        """
        Extracts individual weight tensors from a shared buffer using metadata.
        """
        assert buffer is not None
        weights: list[tuple[str, torch.Tensor]] = []
        for item in payload:
            shape = torch.Size(item["shape"])
            dtype, offset = item["dtype"], item["offset"]

            # Calculate the size in bytes to slice the buffer.
            size = dtype.itemsize * shape.numel()

            # Slice the buffer, cast to the correct data type, and reshape.
            tensor_data = buffer[offset : offset + size]
            tensor = tensor_data.view(dtype=dtype).view(shape)

            if "mlp.gate_proj.weight" in item["name"]:
                up_key = item["name"].replace("gate_proj", "up_proj")
                if up_key in self.pending_weights:
                    up_tensor = self.pending_weights.pop(up_key)
                    self._merge_and_store(item["name"], tensor, up_key, up_tensor)
                else:
                    self.pending_weights[item["name"]] = tensor

            elif "mlp.up_proj.weight" in item["name"]:
                gate_key = item["name"].replace("up_proj", "gate_proj")
                if gate_key in self.pending_weights:
                    gate_tensor = self.pending_weights.pop(gate_key)
                    self._merge_and_store(gate_key, gate_tensor, item["name"], tensor)
                else:
                    self.pending_weights[item["name"]] = tensor

            else:
                weights.append((item["name"], tensor))
        return weights

    # Implemented as a no-op to make BaseConnector interface consistent.
    def weight_iterator(
        self, rank: int = 0
    ) -> Generator[Tuple[str, torch.Tensor], None, None]:
        if self.socket is None:
            self.get_socket_handle(rank)
        while True:
            payload: tuple | list | None = self.socket.recv_pyobj()

            # Handle termination signal
            if payload is None:
                self.socket.send(b"")
                break

            # Handle IPC buffer setup
            if isinstance(payload, tuple):
                buffer = _rebuild_ipc(payload, self.local_rank)
                assert buffer.dtype == torch.uint8
                self.socket.send(b"")
                continue

            # Handle weight metadata payload
            assert isinstance(payload, list)

            # Extract and yield all weights from the current payload.
            # The 'for' loop iterates over the list returned by _extract_weights.
            for key, tensor in self._extract_weights(payload, buffer):
                yield key, tensor

            # Acknowledge receipt of this batch
            torch.cuda.synchronize()
            self.socket.send(b"")

        for key, tensor in self.final_state_dict.items():
            yield key, tensor
