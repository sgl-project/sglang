# SPDX-License-Identifier: Apache-2.0
"""Direct worker-to-worker transfers using the existing Mooncake buffer manager."""

import asyncio
import logging
import pickle
import uuid

import torch
import zmq
import zmq.asyncio

from sglang.multimodal_gen.runtime.disaggregation.deployment.payload import (
    pack_fields,
    unpack_fields,
)
from sglang.multimodal_gen.runtime.disaggregation.transport import (
    engine as transfer_engines,
)
from sglang.multimodal_gen.runtime.disaggregation.transport.buffer import (
    TransferTensorBuffer,
    tensor_payload_size,
)
from sglang.multimodal_gen.runtime.disaggregation.transport.manager import (
    DiffusionTransferManager,
)

logger = logging.getLogger(__name__)


async def rpc(context, endpoint, message, timeout):
    """one private socket per call so cancellation cannot disturb another request"""
    socket = context.socket(zmq.REQ)
    socket.setsockopt(zmq.LINGER, 0)
    socket.connect(endpoint)
    try:
        await socket.send(pickle.dumps(message))
        reply = pickle.loads(await asyncio.wait_for(socket.recv(), timeout))
        if "error" in reply:
            raise RuntimeError(reply["error"] or "Remote deployment request failed")
        return reply
    finally:
        socket.close()


class TensorTransport:
    """own registered staging/receive slots; release only after transfer acknowledgement"""

    def __init__(self, device, hostname, pool_size, ib_device=None):
        self.device = torch.device(device)
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)
        engine = transfer_engines.create_transfer_engine(
            hostname, self.device.index or 0, ib_device
        )
        buffer = TransferTensorBuffer(pool_size, device=str(self.device))
        self.manager = DiffusionTransferManager(engine, buffer)
        self.receive_stream = (
            torch.cuda.Stream(device=self.device)
            if self.device.type == "cuda"
            else None
        )

    def push(self, values, target):
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)
        tensors, scalar = pack_fields(values)
        key = target["key"]
        size = tensor_payload_size(tensors)
        if size > target["size"]:
            raise ValueError("Tensor payload exceeds the advertised receive allocation")
        staged, event = self.manager.stage_tensors_async(key, tensors)
        if staged is None:
            raise RuntimeError(
                "Transfer pool exhausted; reduce concurrency or increase pool size"
            )
        try:
            if event is not None:
                event.synchronize()
            if not self.manager.push_to_peer(
                key, target["session"], target["address"], size
            ):
                raise RuntimeError("GPU tensor transfer failed")
            return {"manifest": staged.manifest, "scalar": scalar}
        finally:
            self.manager.free_staged(key)

    async def receive(self, context, source, timeout):
        """pull selected descriptor fields into an exclusively owned receive slot"""
        fields = source["fields"]
        size = sum(source["sizes"][field] for field in fields)
        # each consumer owns a separate receive slot, including fan-out consumers
        key = uuid.uuid4().hex
        pending = self.manager.allocate_receive_slot(key, max(size, 1))
        if pending is None:
            raise RuntimeError("Receive pool exhausted")
        target = {
            "key": key,
            "session": self.manager.session_id,
            "address": self.manager.pool_data_ptr + pending.slot.offset,
            "size": size,
        }
        acknowledged = False
        try:
            reply = await rpc(
                context,
                source["endpoint"],
                {
                    "method": "push",
                    "ticket": source["ticket"],
                    "fields": fields,
                    "target": target,
                },
                timeout,
            )
            acknowledged = True
            if self.device.type == "cuda":
                torch.cuda.set_device(self.device)
            tensors, event = self.manager.load_tensors_async(
                key, reply["manifest"], self.device, stream=self.receive_stream
            )
            if event is not None:
                try:
                    # leave the event loop available for other pools and client prefetch
                    await asyncio.to_thread(event.synchronize)
                finally:
                    # cancellation cannot return this slot while its D2D copy is pending
                    event.synchronize()
            return unpack_fields(tensors, reply["scalar"], self.device)
        finally:
            if acknowledged:
                self.manager.free_receive_slot(key)
            else:
                # a timed-out sender may still be writing; never reuse its address
                logger.error(
                    "Quarantining receive slot %s after an unacknowledged transfer", key
                )

    def close(self):
        self.manager.cleanup()
