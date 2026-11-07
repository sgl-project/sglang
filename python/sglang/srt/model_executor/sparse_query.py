# SPDX-License-Identifier: Apache-2.0
"""Original query positions for selective prefill over a complete KV cache."""

import msgspec
import torch


class SparseQueryMetadata(msgspec.Struct, frozen=True):
    """Describe request-major queries without changing the KV sequence lengths.

    Args:
        query_lens: Number of selected queries in each request, in batch order.
        positions: Flattened original token positions, on the attention device.
            Positions within each request must be sorted, nonnegative, and less
            than that request's KV length. They also determine RoPE positions
            and the caller's selected output-cache locations.

    Callers compact model inputs and bind the selected write columns with
    ``IterationIds.bind`` before planning attention. KV lengths remain full;
    this metadata does not change the model inputs or the KV write plan.
    """

    query_lens: tuple[int, ...]
    positions: torch.Tensor

    def causal_mask(self, kv_lens: list[int]) -> torch.Tensor:
        """Build a request-major mask that preserves original causal positions.

        Args:
            kv_lens: Full KV length of each request, known on the host.

        Returns:
            Flattened boolean mask on the positions device. Each request owns
            a query_lens[i] by kv_lens[i] matrix; no row attends another request.

        Raises:
            ValueError: Query/KV batch sizes, counts, or position shape disagree.

        This eager reference path materializes O(sum(query_len * kv_len)) entries
        as boolean elements before FlashInfer packs them.
        """
        if (
            len(self.query_lens) != len(kv_lens)
            or not self.query_lens
            or any(
                q <= 0 or q > k for q, k in zip(self.query_lens, kv_lens, strict=True)
            )
            or self.positions.ndim != 1
            or self.positions.numel() != sum(self.query_lens)
        ):
            raise ValueError("Sparse query counts must match positions and KV lengths")
        return torch.cat(
            [
                (
                    torch.arange(kv_len, device=positions.device)[None, :]
                    <= positions[:, None]
                ).flatten()
                for positions, kv_len in zip(
                    self.positions.split(self.query_lens), kv_lens, strict=True
                )
            ]
        )
