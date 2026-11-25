# SPDX-License-Identifier: Apache-2.0
"""Data contracts of snapshot loading.

msgspec types only, importing nothing from sglang: load-diff and the validator
share this module and nothing else.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Optional

import msgspec


class WeightCacheConfig(
    msgspec.Struct, frozen=True, kw_only=True, forbid_unknown_fields=True
):
    """Identity and format of the model weights, shared across ranks.

    The checkpoint ID follows the contents across staging paths and revision aliases.
    """

    checkpoint_id: str  # pinned Hub commit or local checkpoint content digest
    model_arch: str
    hf_config_hash: str  # effective HF config after model overrides
    dtype: str
    quant_method: str
    quant_config_hash: str
    draft_model_idx: Optional[int]  # layer selected from a multi-layer draft checkpoint

    def __post_init__(self):
        if not self.checkpoint_id:
            raise ValueError("A snapshot requires an immutable checkpoint identifier")


class LoadSpec(msgspec.Struct, frozen=True, kw_only=True, forbid_unknown_fields=True):
    """A model and the conditions that determine its loaded weights on this rank.

    Shard lookup uses the whole spec. Ranks vote on the model and shared load
    settings while excluding their logical rank coordinates.
    """

    model: WeightCacheConfig

    # Parallel layout and logical shard coordinates.
    tp_size: int
    tp_rank: int
    pp_size: int
    pp_rank: int
    dp_size: int
    ep_size: int
    moe_dp_size: int
    moe_dp_rank: int
    moe_ep_rank: int
    attn_dp_size: int
    attn_cp_size: int
    moe_dense_tp_size: Optional[int]
    moe_cp_size: int

    # Runtime and kernel choices that affect the representation of the weights.
    device_capability: str
    torch_version: str
    moe_a2a_backend: str
    moe_runner_backend: str
    fp4_gemm_runner_backend: str
    fp8_gemm_runner_backend: str
    bf16_gemm_backend: str
    enable_dp_lm_head: bool
    enable_fp32_lm_head: bool
    ep_num_redundant_experts: int
    enable_eplb: bool
    init_expert_location: Optional[str]
    flashinfer_cutedsl_swap_ab: Optional[bool]

    def digest(self, *, include_rank: bool = True) -> str:
        """Identify this shard for lookup, or its shared configuration for a vote."""
        fields = msgspec.to_builtins(self)
        # Keep the flat key encoding while exposing model settings as a nested struct.
        fields.update(fields.pop("model"))
        if not include_rank:
            for name in ("tp_rank", "pp_rank", "moe_dp_rank", "moe_ep_rank"):
                del fields[name]
        encoded = json.dumps(fields, sort_keys=True)
        return hashlib.sha256(encoded.encode()).hexdigest()


class StorageMeta(msgspec.Struct, frozen=True, kw_only=True):
    """One allocation of the owner's model, moved as raw bytes."""

    nbytes: int
    device: str  # "cuda" or "cpu"


class SnapshotManifest(msgspec.Struct, frozen=True, kw_only=True):
    """What an owner published for one rank; each part reads only its own fields."""

    # Minted by the owners' publish vote; equal on every rank of one publish.
    version: str
    # Filled by load-diff, moved by the transport, digested by the validator.
    storages: dict[str, StorageMeta]
    # Tensors, changed values and the expert map; only load-diff decodes it.
    load_diff: bytes
    # The owner's raw load as the validator described it; only the validator reads it.
    attestation: bytes
    # Where the bytes are; only the transport reads it.
    source: dict[str, Any]
