# SPDX-License-Identifier: Apache-2.0

import gc
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

logger = logging.getLogger(__name__)


def _get_physical_gpu_id(device_index: int | None = None) -> str:
    try:
        return f"GPU-{torch.cuda.get_device_properties(device_index).uuid!s}"
    except AssertionError as e:
        raise ValueError(f"fail to get physical gpu id {device_index}") from e


def _resolve_zmq_handle(
    device_uuid: str,
    all_device_uuids: List[str],
    received_handles: Dict[str, str],
) -> str:
    if device_uuid in received_handles:
        logger.info(f"Rank for UUID {device_uuid}: Found direct ZMQ handle match.")
        return received_handles[device_uuid]

    logger.warning(
        f"Rank for UUID {device_uuid}: Direct match failed. Attempting fallback mapping for unshared GPUs."
    )

    device_uuids_set = set(all_device_uuids)
    sender_uuids_set = set(received_handles.keys())

    unmatched_my_uuids = sorted(list(device_uuids_set - sender_uuids_set))
    unmatched_sender_uuids = sorted(list(sender_uuids_set - device_uuids_set))

    if len(unmatched_my_uuids) != len(unmatched_sender_uuids):
        raise RuntimeError(
            f"Unmatched GPU count mismatch. My unmatched: {len(unmatched_my_uuids)} "
            f"({unmatched_my_uuids}), Sender's unmatched: {len(unmatched_sender_uuids)} "
            f"({unmatched_sender_uuids}). Cannot establish a 1-to-1 mapping."
        )

    if not unmatched_my_uuids:
        raise RuntimeError(
            f"UUID {device_uuid} not found in received handles, but there are no "
            "unmatched GPUs to perform fallback mapping. This indicates a logic error."
        )

    mapping = dict(zip(unmatched_my_uuids, unmatched_sender_uuids))

    target_sender_uuid = mapping.get(device_uuid)
    if not target_sender_uuid:
        raise RuntimeError(
            f"Failed to find UUID {device_uuid} in the fallback mapping. Mapping: {mapping}"
        )

    handle = received_handles[target_sender_uuid]
    logger.info(
        f"Rank for UUID {device_uuid}: Mapped to sender's UUID {target_sender_uuid} via fallback."
    )
    return handle


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

    def __init__(
        self, url: str, device: torch.device = "cpu", ckpt_engine_port: int = 33001
    ):
        super().__init__(url)
        self.url = url
        self.device = device
        self.ckpt_engine_port = ckpt_engine_port
        self.zmq_handle = None
        self.zmq_ctx = None
        self.device_uuid = None
        self.socket = None
        self.buffer: Optional[torch.Tensor] = None
        self.local_rank = None

    def get_zmq_handle(self, tp_rank: int):
        # FIXME: There needs a local rank
        self.device_uuid = _get_physical_gpu_id(tp_rank)

        data_container = [None]
        if tp_rank == 0:
            socket = zmq.Context().socket(zmq.PULL)
            socket.bind(f"tcp://*:{self.ckpt_engine_port}")

            data = None
            try:
                raw_message = socket.recv()

                try:
                    data = json.loads(raw_message.decode("utf-8"))

                    if not isinstance(data, dict):
                        logger.warning("CKPTENGINE: Not exactly the socket handle.")

                except (json.JSONDecodeError, UnicodeDecodeError) as e:
                    logger.error(f"can not parse the socket raw message: {e}")

            except KeyboardInterrupt:
                logger.info("\n shutting down the server.")
            finally:
                socket.close()

            if data is None:
                raise RuntimeError(
                    "Rank 0 failed to receive or parse the ZMQ handle data."
                )

            data_container[0] = data
            logger.info("Rank 0: Received handle data. Broadcasting to other ranks...")

        dist.broadcast_object_list(data_container, src=0)
        received_data = data_container[0]
        world_size = dist.get_world_size()
        all_device_uuids = [None] * world_size
        dist.all_gather_object(all_device_uuids, self.device_uuid)
        self.zmq_handle = _resolve_zmq_handle(
            self.device_uuid, all_device_uuids, received_data
        )

    def get_socket_handle(self, tp_rank: int):
        # FIXME: local_rank is not tp_rank
        if self.zmq_handle is not None:
            return
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

    def _extract_weights(
        self, payload: list[FlattenedTensorMetadata], buffer: torch.Tensor
    ) -> list[tuple[str, torch.Tensor]]:
        """
        Extracts individual weight tensors from a shared buffer using metadata.
        """
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

    # Implemented as a no-op to make BaseConnector interface consistent.
    def weight_iterator(
        self, rank: int = 0
    ) -> Generator[Tuple[str, torch.Tensor], None, None]:
        return

    def update_weights_from_ipc(
        self, model, rank: int = 0, post_hook: Callable[[], None] = None
    ):
        self.get_socket_handle(rank)
        try:
            while True:
                payload: tuple | list | None = self.socket.recv_pyobj()

                # Handle termination signal
                if payload is None:
                    if post_hook is not None:
                        post_hook()
                    torch.cuda.synchronize()
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

                model.load_weights(self._extract_weights(payload, buffer))

                torch.cuda.synchronize()
                self.socket.send(b"")
        except Exception as e:
            logger.error(f"Error in IPC weight update on device {rank}: {e}")
            raise
        finally:
            self.socket.close()
            del self.buffer
            gc.collect()
            torch.cuda.empty_cache()
            logger.info(f"Cleaned up IPC weight update on device {rank}")
