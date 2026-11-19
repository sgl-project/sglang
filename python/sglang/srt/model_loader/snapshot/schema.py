# SPDX-License-Identifier: Apache-2.0
"""Data contracts of snapshot loading.

msgspec types only, importing nothing from sglang: load-diff and the validator
share this module and nothing else.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import msgspec
import msgspec.structs


class LoadSpec(msgspec.Struct, frozen=True, kw_only=True):
    """What this rank is about to build: the key a snapshot is published under."""

    model: dict[str, str]  # checkpoint identity, hf_config digest, dtype, quantization
    settings: dict[str, str]  # resolved parallel, kernel and environment settings
    rank: dict[str, int]  # which shard this rank builds: tp, pp, moe_ep, ...

    def without_rank(self) -> LoadSpec:
        """What every rank of one load must agree on."""
        return msgspec.structs.replace(self, rank={})

    def digest(self) -> str:
        encoded = json.dumps(msgspec.to_builtins(self), sort_keys=True)
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
