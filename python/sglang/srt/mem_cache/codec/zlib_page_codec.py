from __future__ import annotations

import zlib
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ZlibCodecConfig:
    level: int = 1


class ZlibPageCodec:
    def __init__(self, *, config: Optional[ZlibCodecConfig] = None):
        self.config = config or ZlibCodecConfig()

    def encode_bytes(self, raw: bytes) -> bytes:
        return zlib.compress(raw, level=self.config.level)

    def decode_bytes(self, blob: bytes) -> bytes:
        return zlib.decompress(blob)
