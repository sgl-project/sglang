"""GPU-only committed-token relay for repetition penalties under overlap."""

from typing import Optional

import torch


class RepetitionPenaltyRelay:
    """One pending block per request-pool slot, gathered in the next batch order.

    The overlap scheduler can have one unprocessed result. Its CPU penalizer
    snapshot covers the older history; this relay supplies the missing block.
    Repetition uses set membership, so overlap with the snapshot is harmless.
    Stashing a request without a penalty payload clears the length, even when a
    freed slot is reused. Storage is O(pool size * block width), not O(vocab).
    """

    def __init__(self, pool_size: int, device):
        self.pool_size = pool_size
        self.device = device
        self.tokens: Optional[torch.Tensor] = None
        self.lengths: Optional[torch.Tensor] = None

    def stash(self, indices, tokens=None, lengths=None):
        if tokens is None:
            if self.lengths is not None:
                self.lengths[indices] = 0
            return
        width = tokens.shape[1]
        if self.tokens is None or self.tokens.shape[1] < width:
            expanded = torch.zeros(
                (self.pool_size, width), dtype=torch.int64, device=self.device
            )
            if self.tokens is not None:
                expanded[:, : self.tokens.shape[1]] = self.tokens
            self.tokens = expanded
        if self.lengths is None:
            self.lengths = torch.zeros(
                self.pool_size, dtype=torch.int64, device=self.device
            )
        self.tokens[indices, :width] = tokens
        self.lengths[indices] = lengths if lengths is not None else width

    def resolve(self, indices):
        if self.tokens is None:
            return None, None
        # Advanced indexing owns its storage: a later stash cannot overwrite a
        # forward's snapshot, and filtering/merging only changes these indices.
        return self.tokens[indices], self.lengths[indices]
