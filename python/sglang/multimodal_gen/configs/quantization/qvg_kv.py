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

from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


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
    sink_span: str = "chunk"  # "chunk" (per-chunk) | "full" (one span)
    sink_keep_chunks: int = 0  # leading sink chunks kept bf16 forever

    @property
    def enabled(self) -> bool:
        return self.bits is not None

    def validate(self) -> QVGKVQuantArgs:
        if self.bits not in (None, 2, 4):
            raise ValueError(f"kv-cache-quant bits must be 2 or 4, got {self.bits}")
        if self.sink_span not in ("chunk", "full"):
            raise ValueError(
                f"kv-cache-quant-sink-span must be 'chunk' or 'full', got {self.sink_span!r}"
            )
        for name in ("centroids", "block_size", "stages", "kmeans_iters"):
            if getattr(self, name) <= 0:
                raise ValueError(f"kv-cache-quant-{name} must be > 0")
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
            f"sink={self.sink} sink_span={self.sink_span} sink_keep={self.sink_keep_chunks}"
        )

    @classmethod
    def from_dict(cls, kwargs: dict) -> QVGKVQuantArgs:
        """Build from flat CLI kwargs (dest names ``kv_cache_quant*``).

        If the master flag ``--kv-cache-quant`` was not supplied, fall back to
        the legacy env vars so existing scripts keep working.
        """
        master = kwargs.get("kv_cache_quant")
        if master is None:
            return cls()  # disabled by default (no env fallback)
        inst = cls(bits=_parse_bits(master))
        cli_map = {
            "kv_cache_quant_centroids": "centroids",
            "kv_cache_quant_block_size": "block_size",
            "kv_cache_quant_stages": "stages",
            "kv_cache_quant_iters": "kmeans_iters",
            "kv_cache_quant_keep_recent": "keep_recent_chunks",
            "kv_cache_quant_sink_keep": "sink_keep_chunks",
            "kv_cache_quant_asymmetric": "asymmetric",
            "kv_cache_quant_sink": "sink",
            "kv_cache_quant_sink_span": "sink_span",
        }
        for cli_name, field_name in cli_map.items():
            val = kwargs.get(cli_name)
            if val is not None:
                setattr(inst, field_name, val)
        inst.sink = bool(inst.sink)
        inst.asymmetric = bool(inst.asymmetric)
        return inst.validate()
