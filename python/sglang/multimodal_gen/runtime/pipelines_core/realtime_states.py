# SPDX-License-Identifier: Apache-2.0

from sglang.multimodal_gen.runtime.realtime.session import (
    BaseRealtimeState,
)


class RealtimeCausalDiTState(BaseRealtimeState):
    """Persist causal DiT KV/cache position across realtime chunks."""

    def __init__(self):
        super().__init__()
        self.kv_cache = None
        self.crossattn_cache = None
        self.current_chunk_start_frame: int = 0
        self.chunk_idx: int = 0

    def dispose(self):
        self.kv_cache = None
        self.crossattn_cache = None
        self.current_chunk_start_frame = 0
        self.chunk_idx = 0
