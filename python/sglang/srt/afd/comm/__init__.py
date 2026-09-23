"""AFD data-plane communication backends (attn instance <-> ffn instance)."""

from sglang.srt.afd.comm.base import (
    AFDCommBackend,
    AFDCommConfig,
    create_afd_comm_backend,
)

__all__ = [
    "AFDCommBackend",
    "AFDCommConfig",
    "create_afd_comm_backend",
]
