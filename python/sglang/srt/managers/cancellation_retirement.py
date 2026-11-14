"""Bounded, process-local cancellation receipts, independent of output lifetime."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass

from sglang.srt.managers.io_struct import CancellationRetirementOutput


@dataclass
class _Receipt:
    rid: str
    instance_id: str
    created: float
    result: asyncio.Future


class CancellationRetirementRegistry:
    """Keep one shared result per cancellation ID for a bounded retry window.

    Waiting clients never own the shared future. Expiry and scheduler failure
    cannot establish retirement; they leave an explicit unknown result.
    """

    def __init__(self, capacity: int = 256, ttl_seconds: float = 300.0):
        self.capacity = capacity
        self.ttl_seconds = ttl_seconds
        self._receipts: dict[str, _Receipt] = {}

    def _expire(self):
        now = time.monotonic()
        expired = [
            key
            for key, receipt in self._receipts.items()
            if now - receipt.created >= self.ttl_seconds
        ]
        for key in expired:
            receipt = self._receipts.pop(key)
            if not receipt.result.done():
                receipt.result.set_result(
                    CancellationRetirementOutput(
                        rid=receipt.rid,
                        cancellation_id=key,
                        request_instance_id=receipt.instance_id,
                        status="unknown",
                        reason="Receipt retry window expired",
                    )
                )

    def lookup(self, cancellation_id: str, rid: str) -> _Receipt | None:
        self._expire()
        receipt = self._receipts.get(cancellation_id)
        if receipt is not None and receipt.rid != rid:
            raise ValueError("cancellation_id is already bound to another request")
        return receipt

    def create(self, cancellation_id: str, rid: str, instance_id: str) -> _Receipt:
        self._expire()
        if len(self._receipts) >= self.capacity:
            raise ValueError("Cancellation receipt capacity reached; retry later")
        receipt = _Receipt(
            rid,
            instance_id,
            time.monotonic(),
            asyncio.get_running_loop().create_future(),
        )
        self._receipts[cancellation_id] = receipt
        return receipt

    def resolve(self, output: CancellationRetirementOutput):
        self._expire()
        receipt = self._receipts.get(output.cancellation_id)
        if (
            receipt is not None
            and receipt.rid == output.rid
            and receipt.instance_id == output.request_instance_id
            and not receipt.result.done()
        ):
            receipt.result.set_result(output)

    async def wait(self, receipt: _Receipt, timeout_seconds: float):
        if receipt.result.done():
            return receipt.result.result()
        try:
            return await asyncio.wait_for(
                asyncio.shield(receipt.result), timeout=timeout_seconds
            )
        except asyncio.TimeoutError:
            return None
