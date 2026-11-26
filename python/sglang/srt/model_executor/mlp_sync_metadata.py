# SPDX-License-Identifier: Apache-2.0
"""Host metadata for a scheduler's attention-DP MLP sync exchange."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


@dataclass(frozen=True)
class MLPSyncBatchMetadata:
    """The cohort's batches before graph padding or decode-to-extend conversion."""

    forward_modes: tuple[int, ...]
    num_tokens: tuple[int, ...]
    num_requests: tuple[int, ...]
    num_decode_requests: tuple[int, ...]
    max_prefix_tokens: tuple[int, ...]

    @classmethod
    def from_exchange(cls, gathered_rows: "torch.Tensor") -> "MLPSyncBatchMetadata":
        rows = gathered_rows.tolist()
        return cls(
            forward_modes=tuple(row[5] for row in rows),
            num_tokens=tuple(row[0] for row in rows),
            num_requests=tuple(row[-3] for row in rows),
            num_decode_requests=tuple(row[-2] for row in rows),
            max_prefix_tokens=tuple(row[-1] for row in rows),
        )
