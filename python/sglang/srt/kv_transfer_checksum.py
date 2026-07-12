# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""SHA-256 checksum verification for PD disaggregation KV cache transfers.

When ``SGLANG_ENABLE_KV_TRANSFER_CHECKSUM`` is set, each KV cache transfer
between the prefill and decode workers is verified with a deterministic
checksum. This catches silent data corruption that can occur over high-speed
network links during disaggregated serving.

Extended by JoyFuture: initial implementation of KV transfer checksum
verification for PD disaggregation debugging.
"""

import hashlib
import logging
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch

logger = logging.getLogger(__name__)


def compute_kv_checksum(kv_tensor: torch.Tensor) -> str:
    """Compute a deterministic SHA-256 checksum of a KV cache tensor.

    The tensor is moved to CPU and hashed over its raw bytes. The first
    16 hex characters of the hash are returned as a compact checksum
    sufficient for collision detection in typical workloads.

    Args:
        kv_tensor: The KV cache tensor to hash. Can be on any device.

    Returns:
        A 16-character hex string checksum, or ``"none"`` if the input
        is ``None``.
    """
    if kv_tensor is None:
        return "none"
    cpu_tensor = kv_tensor.detach().cpu()
    h = hashlib.sha256()
    h.update(cpu_tensor.numpy().tobytes())
    return h.hexdigest()[:16]


@dataclass
class KVTransferChecksumRecord:
    """Record of a single KV transfer with before/after checksums."""

    request_id: str
    layer_idx: int
    pre_checksum: str
    post_checksum: str
    num_tokens: int = 0
    matched: bool = True
    timestamp: float = 0.0

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = __import__("time").monotonic()
        if self.pre_checksum != self.post_checksum:
            self.matched = False


@dataclass
class KVTransferChecksumVerifier:
    """Batch checksum verifier for PD disaggregation transfers.

    Tracks checksums across multiple layers and requests, and reports
    any mismatches. Designed to be called from the transfer code path
    when ``SGLANG_ENABLE_KV_TRANSFER_CHECKSUM`` is active.

    Usage::

        verifier = KVTransferChecksumVerifier()
        # Before transfer
        pre = compute_kv_checksum(kv_cache)
        verifier.record_pre(request_id, layer_idx, pre)
        # ... perform transfer ...
        # After transfer
        post = compute_kv_checksum(destination_kv_cache)
        record = verifier.record_post(request_id, layer_idx, post)
        if not record.matched:
            logger.warning("Checksum mismatch: %s", record)
    """

    records: List[KVTransferChecksumRecord] = field(default_factory=list)
    _pre_checksums: Dict[str, str] = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def _key(self, request_id: str, layer_idx: int) -> str:
        return f"{request_id}:{layer_idx}"

    def record_pre(self, request_id: str, layer_idx: int, checksum: str) -> None:
        """Record the checksum before a KV transfer."""
        with self.lock:
            self._pre_checksums[self._key(request_id, layer_idx)] = checksum

    def record_post(
        self, request_id: str, layer_idx: int, post_checksum: str, num_tokens: int = 0
    ) -> KVTransferChecksumRecord:
        """Record the checksum after a KV transfer and compare.

        Returns the checksum record. If there is a mismatch, the record's
        ``matched`` field will be ``False`` and a warning is logged.
        """
        key = self._key(request_id, layer_idx)
        with self.lock:
            pre_checksum = self._pre_checksums.pop(key, "unknown")
        record = KVTransferChecksumRecord(
            request_id=request_id,
            layer_idx=layer_idx,
            pre_checksum=pre_checksum,
            post_checksum=post_checksum,
            num_tokens=num_tokens,
        )
        with self.lock:
            self.records.append(record)
        if not record.matched:
            logger.warning(
                "KV transfer checksum mismatch: request=%s layer=%d pre=%s post=%s",
                request_id,
                layer_idx,
                pre_checksum,
                post_checksum,
            )
        return record

    def get_mismatches(self) -> List[KVTransferChecksumRecord]:
        """Return all checksum records that did not match."""
        return [r for r in self.records if not r.matched]

    def get_summary(self) -> dict:
        """Return summary statistics for all recorded transfers."""
        total = len(self.records)
        mismatches = len(self.get_mismatches())
        return {
            "total_transfers": total,
            "mismatches": mismatches,
            "match_rate_pct": round((total - mismatches) / total * 100, 2) if total > 0 else 0,
        }
