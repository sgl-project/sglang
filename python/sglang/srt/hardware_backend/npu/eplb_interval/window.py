import random

import torch


class IntervalWindow:
    """Rank-local logical expert counts, retained across layout checks and exports."""

    def __init__(self, metadata, buffer_size, device):
        self._metadata = metadata
        self._buffer = torch.zeros(
            (buffer_size, metadata.num_layers, metadata.num_logical_experts),
            dtype=torch.int32,
            device=device,
        )
        self.samples_written = 0
        self.generation = 0

    def append(self, physical_counts):
        # Resolve this forward's IDs before another layout is installed.
        row = self._buffer[self.samples_written % len(self._buffer)]
        slots = self._metadata.physical_to_logical_map
        row.zero_().scatter_add_(1, slots.long(), physical_counts.to(row.dtype))
        self.samples_written += 1

    def sum_steps(self):
        # Some device backends widen the entire input before reducing. Bound
        # that temporary to 8 MiB (or one row) even for a long decode window.
        rows = max(1, (8 << 20) // (8 * self._buffer[0].numel()))
        chunks = iter(self._buffer.split(rows))
        total = next(chunks).sum(dim=0, dtype=torch.int64)
        for chunk in chunks:
            total.add_(chunk.sum(dim=0, dtype=torch.int64))
        return total

    def latest(self):
        """Borrow the last appended row until the next forward."""
        if not self.samples_written:
            raise ValueError("The recorder window is empty")
        return self._buffer[(self.samples_written - 1) % len(self._buffer)]

    def sample_recent(self, steps: int, sample_count: int, seed: int = 0):
        """Stratified rows from the last block, in chronological order."""
        if not 0 < steps <= min(len(self._buffer), self.samples_written):
            raise ValueError("Requested steps are outside the recorder window")
        n = min(steps, sample_count)
        rng = random.Random(seed)
        offsets = (
            range(steps)
            if steps == n
            else [rng.randrange(i * steps // n, (i + 1) * steps // n) for i in range(n)]
        )
        indices = [
            (self.samples_written - steps + i) % len(self._buffer) for i in offsets
        ]
        return self._buffer[indices]

    def snapshot(self):
        """Owned chronological rows for export; unused rows follow the valid prefix."""
        oldest = max(0, self.samples_written - len(self._buffer)) % len(self._buffer)
        return self._buffer.roll(-oldest, dims=0)

    def reset(self):
        self._buffer[...] = 0
        self.samples_written = 0
        self.generation += 1
