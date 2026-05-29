# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from uuid import uuid4

from sglang.multimodal_gen.runtime.entrypoints.openai.protocol import (
    RealtimeVideoInitRequest,
)
from sglang.multimodal_gen.runtime.pipelines_core.realtime_session import (
    RealtimeSession,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

if TYPE_CHECKING:
    from sglang.multimodal_gen.runtime.entrypoints.openai.realtime.realtime_adapter import (
        RealtimeModelAdapter,
    )


class RealtimeGenerateSession:
    def __init__(self):
        self.id = uuid4().hex
        self.request_id = None
        self.request: RealtimeVideoInitRequest | None = None
        self.generate_chunk_cnt = 0
        self.realtime_session = RealtimeSession()
        self.adapter: RealtimeModelAdapter | None = None
        self.adapter_state: Any = None

    def set_adapter(self, adapter: RealtimeModelAdapter):
        self.adapter = adapter
        self.adapter_state = adapter.create_state()

    def set_request(self, request: RealtimeVideoInitRequest):
        self.request = request

    def dispose(self):
        if self.adapter is not None:
            try:
                self.adapter.dispose(self)
            except Exception:
                logger.warning(
                    "Failed to dispose realtime adapter for session %s",
                    self.id,
                    exc_info=True,
                )

        try:
            self.realtime_session.dispose()
        except Exception:
            logger.warning(
                "Failed to dispose realtime session state for session %s",
                self.id,
                exc_info=True,
            )
        finally:
            self.request = None
            self.request_id = None
            self.generate_chunk_cnt = 0
            self.adapter = None
            self.adapter_state = None

    def new_request(self):
        self.request_id = f"{self.id}_{uuid4().hex}"

    def generate_chunk_completed(self):
        self.generate_chunk_cnt += 1

    def build_sampling_params(self):
        if self.adapter is None:
            raise ValueError("realtime adapter is not initialized")
        return self.adapter.build_sampling_params(self)
