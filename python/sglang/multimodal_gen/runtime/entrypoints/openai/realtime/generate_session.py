# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from uuid import uuid4

from sglang.multimodal_gen.runtime.entrypoints.openai.protocol import (
    RealtimeVideoGenerationsRequest,
)
from sglang.multimodal_gen.runtime.realtime.session import (
    RealtimeSession,
)

if TYPE_CHECKING:
    from sglang.multimodal_gen.runtime.entrypoints.openai.realtime.realtime_adapter import (
        RealtimeModelAdapter,
    )


class GenerateSession:
    def __init__(self):
        self.id = uuid4().hex
        self.request_id = None
        self.request: RealtimeVideoGenerationsRequest | None = None
        self.input_temp_dir: str | None = None
        self.generate_chunk_cnt = 0
        self.realtime_session = RealtimeSession()
        self.adapter: RealtimeModelAdapter | None = None
        self.adapter_state: Any = None

    def set_adapter(self, adapter: RealtimeModelAdapter):
        self.adapter = adapter
        self.adapter_state = adapter.create_state()

    def set_request(self, request: RealtimeVideoGenerationsRequest):
        self.request = request

    def dispose(self):
        if self.adapter is not None:
            self.adapter.dispose(self)
        self.request = None
        self.request_id = None
        self.input_temp_dir = None
        self.generate_chunk_cnt = 0
        self.adapter = None
        self.adapter_state = None
        self.realtime_session.dispose()

    def new_request(self):
        self.request_id = f"{self.id}_{uuid4().hex}"

    def generate_chunk_completed(self):
        self.generate_chunk_cnt += 1

    def build_sampling_params(self):
        if self.adapter is None:
            raise ValueError("realtime adapter is not initialized")
        return self.adapter.build_sampling_params(self)
