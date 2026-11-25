# SPDX-License-Identifier: Apache-2.0
"""Data contracts of snapshot loading.

msgspec types only, importing nothing from sglang: load-diff and the validator
share this module and nothing else.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Optional, Self

import msgspec


class WeightCacheConfig(msgspec.Struct):
    """Fingerprint of one rank's weights, shared by the daemon and snapshot loaders."""

    model_path: str
    model_arch: str
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
    enable_dp_lm_head: bool
    attn_cp_size: int
    moe_dense_tp_size: Optional[int]
    moe_a2a_backend: str
    quant_method: str  # e.g. "fp8", "gptq_marlin", "" for unquantized
    quant_config_hash: str  # SHA-256 hash of quantization config
    dtype: str  # e.g. "torch.float16"
    revision: str  # model revision the weights were loaded from ("" if unset)
    # Environment stamp: a daemon and a client that ran different post-processing
    # branches (different GPU compute capability or torch/kernel version) can
    # produce incompatible weights that would map cleanly yet serve garbage.
    # Comparing these turns that into a clean mismatch. See compute_env_stamp().
    device_capability: str  # local compute capability, e.g. "8.0" ("" if N/A)
    torch_version: str  # torch.__version__ of the process that built the weights

    def matches(self, other: WeightCacheConfig) -> bool:
        """Compare all fields for an IPC weight-cache request."""
        return self == other

    def to_dict(self) -> dict[str, Any]:
        return msgspec.structs.asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Self:
        return cls(**d)


class LoadSpec(
    WeightCacheConfig, frozen=True, kw_only=True, forbid_unknown_fields=True
):
    """Inputs to the snapshot lookup key. Sources index snapshots by digest()."""

    checkpoint_id: str
    hf_config_hash: str
    draft_model_idx: Optional[int]
    moe_runner_backend: str
    fp4_gemm_runner_backend: str
    fp8_gemm_runner_backend: str
    bf16_gemm_backend: str
    moe_cp_size: int
    enable_fp32_lm_head: bool
    ep_num_redundant_experts: int
    enable_eplb: bool
    init_expert_location: Optional[str]
    flashinfer_cutedsl_swap_ab: Optional[bool]

    def __post_init__(self):
        if not self.checkpoint_id:
            raise ValueError("A snapshot requires an immutable checkpoint identifier")

    def digest(self, *, include_rank: bool = True) -> str:
        """Identify this shard for lookup, or its shared configuration for a vote."""
        fields = msgspec.to_builtins(self)
        # A remote checkpoint can be staged in a different directory on each rank.
        # checkpoint_id already identifies the contents across paths and revision aliases.
        del fields["model_path"], fields["revision"]
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
