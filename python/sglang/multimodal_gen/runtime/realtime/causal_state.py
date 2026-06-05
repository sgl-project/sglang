# SPDX-License-Identifier: Apache-2.0

from sglang.multimodal_gen.runtime.realtime.session import BaseRealtimeState


class RealtimeCausalDiTState(BaseRealtimeState):
    """persist causal DiT cache and frame position across realtime chunks"""

    def __init__(self):
        super().__init__()
        self.kv_cache = None
        self.crossattn_cache = None
        self.runtime_cache: dict = {}
        self.current_chunk_start_frame: int = 0
        self.chunk_idx: int = 0

    def dispose(self) -> None:
        self.kv_cache = None
        self.crossattn_cache = None
        self.runtime_cache.clear()
        self.current_chunk_start_frame = 0
        self.chunk_idx = 0
