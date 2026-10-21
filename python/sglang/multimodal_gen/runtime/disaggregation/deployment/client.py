# SPDX-License-Identifier: Apache-2.0
"""Async encoding client returning named tensors on the trainer's GPU."""

import asyncio
import logging
import uuid

import zmq.asyncio

from sglang.multimodal_gen.runtime.disaggregation.deployment.transport import (
    TensorTransport,
    rpc,
)

logger = logging.getLogger(__name__)


class EncodingClient:
    """pull named results onto the caller's device and release their remote leases"""

    def __init__(
        self,
        endpoint,
        *,
        device="cuda:0",
        hostname="127.0.0.1",
        pool_size=256 << 20,
        ib_device=None,
        timeout=600,
    ):
        self.endpoint = endpoint
        self.timeout = timeout
        self.context = zmq.asyncio.Context()
        self.transport = TensorTransport(device, hostname, pool_size, ib_device)
        self.pending = set()
        self.closing = False

    async def encode(self, request, outputs=None):
        if self.closing:
            raise RuntimeError("Encoding client is closing")
        task = asyncio.current_task()
        self.pending.add(task)
        try:
            return await self._encode(request, outputs)
        finally:
            self.pending.discard(task)

    async def _encode(self, request, outputs):
        job_id = uuid.uuid4().hex
        result = None
        try:
            result = await rpc(
                self.context,
                self.endpoint,
                {
                    "method": "encode",
                    "job_id": job_id,
                    "request": request,
                    "outputs": outputs,
                },
                self.timeout,
            )
            values = {}
            for source in result["sources"]:
                values.update(
                    await self.transport.receive(self.context, source, self.timeout)
                )
            return values
        finally:
            # before a reply, cancel by job ID; afterwards, release the retained lease
            cleanup = (
                {"method": "release", "lease": result["lease"]}
                if result is not None
                else {"method": "cancel", "job_id": job_id}
            )
            try:
                await rpc(self.context, self.endpoint, cleanup, min(self.timeout, 5))
            except Exception:
                logger.warning(
                    "Encoder cleanup was not acknowledged; its lease will expire",
                    exc_info=True,
                )

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        # wait for each request's cleanup RPC before destroying its transport
        self.closing = True
        for task in self.pending:
            task.cancel()
        await asyncio.gather(*self.pending, return_exceptions=True)
        try:
            self.transport.close()
        finally:
            self.context.destroy(linger=0)
