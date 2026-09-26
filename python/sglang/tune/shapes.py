"""Attention profile + the representative shape grid.

Two axes, kept deliberately separate (this separation is the heart of the design):

* The **coarse key** — the model's attention *profile* + device — identifies one
  config file. Low cardinality (SM-class x attn-family x head-config x dtype x
  parallelism), so the committed corpus stays small and maintainable. SGLang's own
  ``_get_default_attn_backend`` already branches on model family (issue #5064), which
  is why the key carries a model/attn-family axis rather than being purely hardware.

* The **fine grid** — phase x batch x context-length — lives *inside* one file as the
  body. This is where attention backends actually cross over (a measured H20 example:
  FA3 wins at short seqlen, FlashInfer at long). A single per-file choice would be
  wrong; a full cross-product would explode. Buckets are ~power-of-two so runtime
  nearest-match interpolates sanely, exactly like MoE's M-key set.
"""

from __future__ import annotations

import dataclasses
from typing import List, Tuple

# ---- the fine grid (body axis) ------------------------------------------------
# Decode is memory-bound -> sweep batch and KV length. Prefill is compute-bound ->
# sweep batch and sequence length. Kept small (~a few dozen cells) so a full sweep
# on one GPU is minutes, not hours — viable as an install/init step.
DECODE_BATCH: Tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128, 256)
DECODE_CTXLEN: Tuple[int, ...] = (1024, 4096, 16384, 65536)
PREFILL_BATCH: Tuple[int, ...] = (1, 4, 16)
PREFILL_SEQLEN: Tuple[int, ...] = (128, 512, 2048, 8192, 32768)


@dataclasses.dataclass(frozen=True)
class AttnProfile:
    """The coarse key: what makes a kernel behave differently for a given model.

    Pulled from ``ServerArgs`` + ``ModelConfig`` exactly as the engine sees them, so a
    file emitted by ``sglang tune`` is guaranteed key-compatible with the runtime lookup.
    """

    num_qo_heads: int  # query/output heads AFTER tensor-parallel sharding
    num_kv_heads: int  # KV heads after TP (GQA factor = qo/kv)
    head_dim: int
    dtype: str  # e.g. "bfloat16"
    kv_cache_dtype: str = "auto"
    is_mla: bool = False  # DeepSeek-style Multi-head Latent Attention
    tp_size: int = 1
    ep_size: int = 1
    dp_size: int = 1

    def family(self) -> str:
        """Coarse attention-family label the config key carries."""
        if self.is_mla:
            return "mla"
        return "gqa" if self.num_kv_heads < self.num_qo_heads else "mha"

    def key_fields(self) -> dict:
        """Ordered, explicit fields for the filename key — TP/EP/DP stated literally,
        never derived into an opaque shape integer the way MoE bakes topology into N.
        """
        return {
            "family": self.family(),
            "qo": self.num_qo_heads,
            "kv": self.num_kv_heads,
            "hd": self.head_dim,
            "dtype": self.dtype,
            "kv_dtype": self.kv_cache_dtype,
            "tp": self.tp_size,
            "ep": self.ep_size,
            "dp": self.dp_size,
        }


@dataclasses.dataclass(frozen=True)
class DecodeShape:
    batch: int
    ctx_len: int

    def bucket_key(self) -> str:
        return f"{self.batch}:{self.ctx_len}"


@dataclasses.dataclass(frozen=True)
class PrefillShape:
    batch: int
    seq_len: int

    def bucket_key(self) -> str:
        return f"{self.batch}:{self.seq_len}"


def decode_grid() -> List[DecodeShape]:
    return [DecodeShape(b, c) for c in DECODE_CTXLEN for b in DECODE_BATCH]


def prefill_grid() -> List[PrefillShape]:
    return [PrefillShape(b, s) for s in PREFILL_SEQLEN for b in PREFILL_BATCH]


def parse_decode_key(key: str) -> DecodeShape:
    b, c = key.split(":")
    return DecodeShape(int(b), int(c))


def parse_prefill_key(key: str) -> PrefillShape:
    b, s = key.split(":")
    return PrefillShape(int(b), int(s))
