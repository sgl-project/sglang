# SPDX-License-Identifier: Apache-2.0
"""
Public runtime service facade for diffusion generation paths.

This module intentionally starts as a thin adapter over existing internals.
It provides a stable surface for future incremental refactors without
changing current behavior.
"""

from typing import Any


class GenerationService:
    """
    Thin facade over request preparation and scheduler communication.

    This service does not alter generation semantics; it only centralizes
    existing behavior behind a reusable API.
    """

    def __init__(
        self,
        sync_client: Any = None,
        async_client: Any = None,
    ) -> None:
        # Local import avoids import cycles during startup.
        from sglang.multimodal_gen.runtime.scheduler_client import (
            async_scheduler_client,
            sync_scheduler_client,
        )

        self._sync_client = sync_client or sync_scheduler_client
        self._async_client = async_client or async_scheduler_client

    @staticmethod
    def _prepare_request(server_args: Any, sampling_params: Any) -> Any:
        from sglang.multimodal_gen.runtime.entrypoints.utils import prepare_request

        return prepare_request(server_args=server_args, sampling_params=sampling_params)

    def build_request(self, server_args: Any, sampling_params: Any) -> Any:
        """Build a runtime Req from server and sampling parameters."""
        return self._prepare_request(
            server_args=server_args, sampling_params=sampling_params
        )

    def forward_sync(self, payload: Any) -> Any:
        """Forward payload to the scheduler via the sync transport."""
        return self._sync_client.forward(payload)

    async def forward_async(self, payload: Any) -> Any:
        """Forward payload to the scheduler via the async transport."""
        return await self._async_client.forward(payload)

    @staticmethod
    def ensure_success(output: Any, failure_message: str) -> Any:
        """
        Raise a RuntimeError when scheduler output contains an error field.
        """
        error = getattr(output, "error", None)
        if error is not None:
            raise RuntimeError(f"{failure_message}: {error}")
        return output

    def control_sync(self, req: Any, failure_message: str) -> Any:
        """
        Send a non-generation control request via sync transport.
        """
        output = self.forward_sync(req)
        return self.ensure_success(output, failure_message=failure_message)


__all__ = ["GenerationService"]
