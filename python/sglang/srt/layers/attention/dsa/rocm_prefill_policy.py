# SPDX-License-Identifier: Apache-2.0
"""Host-only, opt-in policies for ROCm sparse MLA prefill.

The profiles were fitted on GLM-5.3-MXFP4 with MI350X TP4. They select an
implementation; they neither change the KV layout nor repair its numerics.
In particular, the FP8 profile needs a separately validated AITER kernel
(see ROCm/aiter#5535). It must not be treated as an FP8 correctness guarantee.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

RocmDSAPrefillBackend = Literal["aiter", "tilelang"]
ROCM_DSA_PREFILL_POLICIES = (
    "glm53-mi350x-tp4-bf16-v1",
    "glm53-mi350x-tp4-fp8-v1",
)


@dataclass(frozen=True)
class DSAPrefillShape:
    """Realized query rows and total KV length, not requested concurrency.

    ``total_q_tokens`` excludes cached prefix tokens. ``max_kv_len`` includes
    the prefix and this extend. All values must already reside on the host.
    """

    total_q_tokens: int
    max_q_len: int
    batch_size: int
    max_kv_len: int

    def __post_init__(self) -> None:
        for name in ("total_q_tokens", "max_q_len", "batch_size", "max_kv_len"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")


@dataclass(frozen=True)
class RocmDSAPrefillPolicy:
    """Bounded offline policy; unsupported batch shapes use TileLang.

    The 32K fresh-row ceiling is a workspace-safety bound, not a measured
    performance crossover. No policy is enabled or inferred automatically.
    """

    policy_id: str

    def __post_init__(self) -> None:
        if self.policy_id not in ROCM_DSA_PREFILL_POLICIES:
            raise ValueError(f"Unknown ROCm DSA prefill policy: {self.policy_id!r}")

    @property
    def kv_dtype(self) -> str:
        return "float8_e4m3fn" if self.policy_id.endswith("fp8-v1") else "bfloat16"

    def validate_runtime(
        self,
        *,
        architecture: str,
        attn_tp_size: int,
        attn_cp_size: int,
        dcp_size: int,
        kv_dtype: str,
        prefill_backend: str,
        prefill_graph_enabled: bool,
        deterministic: bool,
        index_topk: int,
        index_kpool: int,
        num_q_heads: int,
        kv_lora_rank: int,
        qk_rope_head_dim: int,
    ) -> None:
        """Reject untuned configurations at startup, before the first batch."""
        if architecture.split(":", 1)[0] != "gfx950":
            raise ValueError("ROCm DSA prefill policies require gfx950")
        if attn_tp_size != 4:
            raise ValueError("ROCm DSA prefill policies require attention TP4")
        if attn_cp_size != 1 or dcp_size != 1:
            raise ValueError("ROCm DSA prefill policies do not support CP or DCP")
        if kv_dtype != self.kv_dtype:
            raise ValueError(
                f"{self.policy_id} requires KV dtype {self.kv_dtype}; got {kv_dtype}"
            )
        if prefill_backend != "tilelang":
            raise ValueError(
                "--dsa-prefill-policy requires --dsa-prefill-backend tilelang "
                "as its fallback"
            )
        if prefill_graph_enabled:
            raise ValueError("ROCm DSA prefill policies support eager prefill only")
        if deterministic:
            raise ValueError(
                "ROCm DSA prefill policies are incompatible with deterministic "
                "inference because backend selection depends on batch shape"
            )
        if index_topk != 2048 or index_kpool != 1:
            raise ValueError(
                "ROCm DSA prefill policies require index_topk=2048 and index_kpool=1"
            )
        if (num_q_heads, kv_lora_rank, qk_rope_head_dim) != (16, 512, 64):
            raise ValueError(
                "ROCm DSA prefill policies require 16 local query heads, "
                "kv_lora_rank=512 and qk_rope_head_dim=64"
            )

    def select(self, shape: DSAPrefillShape) -> RocmDSAPrefillBackend:
        if (
            shape.total_q_tokens == 0
            or shape.batch_size == 0
            or shape.total_q_tokens > 32768
            or shape.batch_size > 8
        ):
            return "tilelang"

        if self.kv_dtype == "bfloat16":
            if shape.max_kv_len >= 65536:
                min_fresh_rows = 1024
            elif shape.max_kv_len >= 32768:
                min_fresh_rows = 1536
            else:
                min_fresh_rows = 2049
            return "aiter" if shape.total_q_tokens >= min_fresh_rows else "tilelang"

        # FP8 crossover is driven by the longest fresh row, not aggregate M.
        if shape.max_q_len >= 3072:
            return "aiter"
        if shape.max_kv_len >= 65536 and shape.total_q_tokens >= 1024:
            return "aiter"
        if shape.max_kv_len >= 8192 and shape.total_q_tokens >= 4096:
            return "aiter"
        if shape.max_kv_len >= 16384 and shape.max_q_len >= 2048:
            return "aiter"
        return "tilelang"
