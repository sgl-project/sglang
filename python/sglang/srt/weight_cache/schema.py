# SPDX-License-Identifier: Apache-2.0
"""Data contracts of weight caching and snapshot loading.

msgspec types only, importing nothing from sglang: load-diff and the validator
share this module and nothing else.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Optional

import msgspec


def _digest(fields: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(fields, sort_keys=True).encode()).hexdigest()


class WeightCacheConfig(
    msgspec.Struct, frozen=True, kw_only=True, forbid_unknown_fields=True
):
    """The model: its checkpoint and the model options that decide its weights.

    Built from ModelConfig, the checkpoint ID and the draft layer, and equal on
    every rank. How a rank loads the model is in LoadSpec.
    """

    # A content digest or pinned commit; the cache daemon uses path_checkpoint_id.
    checkpoint_id: str
    model_arch: str
    hf_config_hash: str  # effective HF config after model overrides, without paths
    dtype: str
    quant_method: str  # "" for unquantized
    quant_config_hash: str
    draft_model_idx: Optional[int]  # layer selected from a multi-layer draft checkpoint

    def __post_init__(self):
        if not self.checkpoint_id:
            raise ValueError("A weight cache config requires a checkpoint ID")

    def digest(self) -> str:
        return _digest(msgspec.to_builtins(self))


class SnapshotRank(
    msgspec.Struct, frozen=True, kw_only=True, forbid_unknown_fields=True
):
    """The parallel shard described by a snapshot manifest."""

    tp_rank: int
    pp_rank: int
    moe_dp_rank: int
    moe_ep_rank: int


class LoadSpec(msgspec.Struct, frozen=True, kw_only=True, forbid_unknown_fields=True):
    """How this rank loads the model: everything that is not in WeightCacheConfig.

    Built from the runtime context alone.
    """

    # Parallel layout, equal on every rank.
    tp_size: int
    pp_size: int
    dp_size: int
    ep_size: int
    moe_dp_size: int
    attn_dp_size: int
    attn_cp_size: int
    moe_cp_size: int
    moe_dense_tp_size: Optional[int]
    enable_dp_lm_head: bool

    # This rank's shard.
    tp_rank: int
    pp_rank: int
    moe_dp_rank: int
    moe_ep_rank: int

    # Expert placement.
    ep_num_redundant_experts: int
    enable_eplb: bool
    init_expert_location: Optional[str]

    # Kernels and features that post-process the weights differently.
    moe_a2a_backend: str
    moe_runner_backend: str
    fp4_gemm_runner_backend: str
    fp8_gemm_runner_backend: str
    bf16_gemm_backend: str
    flashinfer_cutedsl_swap_ab: Optional[bool]
    enable_fp32_lm_head: bool

    # The process that post-processes the weights.
    device_capability: str
    torch_version: str

    @property
    def rank(self) -> SnapshotRank:
        return SnapshotRank(
            tp_rank=self.tp_rank,
            pp_rank=self.pp_rank,
            moe_dp_rank=self.moe_dp_rank,
            moe_ep_rank=self.moe_ep_rank,
        )

    def digest(self, *, include_rank: bool = True) -> str:
        """Identify how this rank loads the model, or what every rank shares."""
        fields = msgspec.to_builtins(self)
        if not include_rank:
            for name in SnapshotRank.__struct_fields__:
                del fields[name]
        return _digest(fields)


def snapshot_tag(
    model: WeightCacheConfig, spec: LoadSpec, *, include_rank: bool = True
) -> str:
    """Name the snapshot shard or, for a vote, the load shared by all ranks."""
    return _digest(
        {"model": model.digest(), "spec": spec.digest(include_rank=include_rank)}
    )


class StorageMeta(msgspec.Struct, frozen=True, kw_only=True):
    """One allocation of the owner's model, moved as raw bytes."""

    nbytes: int
    device: str  # "cuda" or "cpu"


class SnapshotManifest(msgspec.Struct, frozen=True, kw_only=True):
    """What an owner published for one rank; each part reads only its own fields."""

    # Minted by the owners' publish vote; equal on every rank of one publish.
    version: str
    rank: SnapshotRank
    # Filled by load-diff, moved by the transport, digested by the validator.
    storages: dict[str, StorageMeta]
    # Tensors, changed values and the expert map; only load-diff decodes it.
    load_diff: bytes
    # The owner's raw load as the validator described it; only the validator reads it.
    attestation: bytes
    # Where the bytes are; only the transport reads it.
    source: dict[str, Any]
