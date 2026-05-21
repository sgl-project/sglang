from __future__ import annotations

import hashlib
from typing import List

import torch


def _hash_rids_to_i64_tensor(*, rids: List[str], device: torch.device) -> torch.Tensor:
    values: List[int] = [_stable_hash_rid_i64(rid) for rid in rids]
    return torch.tensor(values, dtype=torch.int64, device=device)


def _stable_hash_rid_i64(rid: str) -> int:
    digest = hashlib.blake2b(rid.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little", signed=True)
