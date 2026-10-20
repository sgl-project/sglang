# SPDX-License-Identifier: Apache-2.0
"""Async encoding client returning named tensors on the trainer's GPU."""

import zmq.asyncio

from sglang.multimodal_gen.runtime.disaggregation.deployment.transport import (
    TensorTransport,
    rpc,
)


class EncodingClient:
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

    async def encode(self, request, outputs=None):
        result = await rpc(
            self.context,
            self.endpoint,
            {"method": "encode", "request": request, "outputs": outputs},
            self.timeout,
        )
        try:
            values = {}
            for source in result["sources"]:
                fields = source["fields"]
                values.update(
                    await self.transport.receive(
                        self.context,
                        source["endpoint"],
                        source["ticket"],
                        fields,
                        sum(source["sizes"][field] for field in fields),
                        self.timeout,
                    )
                )
            return values
        finally:
            await rpc(
                self.context,
                self.endpoint,
                {"method": "release", "lease": result["lease"]},
                min(self.timeout, 5),
            )

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        self.transport.close()
        self.context.term()
