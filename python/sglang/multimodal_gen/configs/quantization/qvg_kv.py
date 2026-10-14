# SPDX-License-Identifier: Apache-2.0
"""CLI-facing configuration for Quant-VideoGen PRQ KV-cache quantization.

Mirrors the SRT `--kv-cache-dtype` pattern: the on/off + tuning knobs live on a
typed config object carried by ServerArgs (see `kv_cache_quant_config`), instead
of a pile of raw environment variables.

Defaults match the tuned per-chunk setting
(kmeans 1 stage, 128 centroids, block 64, symmetric, 2 iters, recent 1,
per-chunk sink) — i.e. `--kv-cache-quant int4` alone reproduces it.
"""

from __future__ import annotations

from dataclasses import dataclass

_BITS = {"off": None, "none": None, "bf16": None, "int4": 4, "int2": 2}


def _parse_bits(val: str | None) -> int | None:
    if val is None:
        return None
    key = str(val).strip().lower()
    if key not in _BITS:
        raise ValueError(f"kv-cache-quant must be one of {list(_BITS)}, got {val!r}")
    return _BITS[key]


@dataclass
class QVGKVQuantArgs:
    """PRQ (multi-stage k-means) KV-cache quantization settings.

    ``bits is None`` means quantization is OFF (plain bf16 cache). Defaults
    reproduce the tuned per-chunk config from offline sweeps.
    """

    bits: int | None = None  # None => off; 2 or 4 (master switch)
    centroids: int = 128  # k-means centroids per stage
    block_size: int = 64  # residual scale block size
    stages: int = 1  # PRQ k-means stages
    kmeans_iters: int = 2  # k-means iterations
    asymmetric: bool = False  # KIVI-style asymmetric residual quant
    keep_recent_chunks: int = 1  # completed chunks kept bf16 (recency guard)
    sink: bool = True  # quantize the attention sink too
    sink_keep_chunks: int = 0  # leading sink chunks kept bf16 forever

    @property
    def enabled(self) -> bool:
        return self.bits is not None

    def validate(self) -> QVGKVQuantArgs:
        if self.bits not in (None, 2, 4):
            raise ValueError(f"kv-cache-quant bits must be 2 or 4, got {self.bits}")
        if self.centroids <= 0:
            raise ValueError("kv-cache-quant-centroids must be > 0")
        if self.block_size <= 0:
            raise ValueError("kv-cache-quant-block-size must be > 0")
        if self.stages <= 0:
            raise ValueError("kv-cache-quant-stages must be > 0")
        if self.kmeans_iters <= 0:
            raise ValueError("kv-cache-quant-iters must be > 0")
        if self.keep_recent_chunks < 0 or self.sink_keep_chunks < 0:
            raise ValueError("keep_recent_chunks / sink_keep_chunks must be >= 0")
        return self

    def describe(self) -> str:
        if not self.enabled:
            return "off"
        return (
            f"int{self.bits} centroids={self.centroids} block={self.block_size} "
            f"stages={self.stages} iters={self.kmeans_iters} "
            f"asym={self.asymmetric} recent={self.keep_recent_chunks} "
            f"sink={self.sink} sink_keep={self.sink_keep_chunks}"
        )

    @classmethod
    def from_dict(cls, kwargs: dict) -> QVGKVQuantArgs:
        """Build from flat CLI kwargs (dest names ``kv_cache_quant*``)."""
        master = kwargs.get("kv_cache_quant")
        if master is None:
            return cls()
        inst = cls(bits=_parse_bits(master))
        if kwargs.get("kv_cache_quant_centroids") is not None:
            inst.centroids = kwargs["kv_cache_quant_centroids"]
        if kwargs.get("kv_cache_quant_block_size") is not None:
            inst.block_size = kwargs["kv_cache_quant_block_size"]
        if kwargs.get("kv_cache_quant_stages") is not None:
            inst.stages = kwargs["kv_cache_quant_stages"]
        if kwargs.get("kv_cache_quant_iters") is not None:
            inst.kmeans_iters = kwargs["kv_cache_quant_iters"]
        if kwargs.get("kv_cache_quant_keep_recent") is not None:
            inst.keep_recent_chunks = kwargs["kv_cache_quant_keep_recent"]
        if kwargs.get("kv_cache_quant_sink_keep") is not None:
            inst.sink_keep_chunks = kwargs["kv_cache_quant_sink_keep"]
        if kwargs.get("kv_cache_quant_asymmetric") is not None:
            inst.asymmetric = kwargs["kv_cache_quant_asymmetric"]
        if kwargs.get("kv_cache_quant_sink") is not None:
            inst.sink = kwargs["kv_cache_quant_sink"]
        inst.sink = bool(inst.sink)
        inst.asymmetric = bool(inst.asymmetric)
        return inst.validate()
