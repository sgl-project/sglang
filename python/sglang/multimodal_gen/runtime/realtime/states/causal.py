# SPDX-License-Identifier: Apache-2.0

from typing import Any, TypeVar, cast

from sglang.multimodal_gen.runtime.realtime.session import (
    BaseRealtimeState,
    RealtimeSession,
)


class RealtimeCausalDiTState(BaseRealtimeState):
    """persist causal DiT cache, chunk frontier, and output buffer"""

    def __init__(self):
        super().__init__()
        self.kv_cache: Any = None
        self.crossattn_cache: Any = None
        self.runtime_cache: dict = {}
        self.current_chunk_start_frame: int = 0
        self.chunk_idx: int = 0
        self.chunk_indices: list[int] = [0]
        self.latents: Any = None
        self.scheduler: Any = None

    def dispose(self) -> None:
        self.kv_cache = None
        self.crossattn_cache = None
        self.runtime_cache.clear()
        self.current_chunk_start_frame = 0
        self.chunk_idx = 0
        self.chunk_indices = [0]
        self.latents = None
        self.scheduler = None


RealtimeCausalDiTStateT = TypeVar(
    "RealtimeCausalDiTStateT", bound=RealtimeCausalDiTState
)


def get_realtime_causal_dit_state(
    session: RealtimeSession,
    state_cls: type[RealtimeCausalDiTStateT] = RealtimeCausalDiTState,
) -> RealtimeCausalDiTStateT:
    return cast(RealtimeCausalDiTStateT, session.get_or_create_state(state_cls))


class RealtimeCausalDecodeState(BaseRealtimeState):
    """persist causal VAE decode cache and output frontier across chunks"""

    def __init__(self):
        super().__init__()
        self.conv_cache: dict | None = None
        self.next_dec_idx: int = 0

    def dispose(self) -> None:
        self.conv_cache = None
        self.next_dec_idx = 0
