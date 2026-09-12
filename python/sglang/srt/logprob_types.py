"""Internal requested-token layouts shared by tokenizer, scheduler and scorer."""

from dataclasses import dataclass


@dataclass(frozen=True)
class PerPositionTokenIds:
    """Token IDs keyed by the absolute position of the token being predicted."""

    positions: list[list[int]]
    start: int = 0

    def __len__(self):
        return len(self.positions)

    def rows(self, offset: int, count: int) -> list[list[int]]:
        start = self.start + offset
        rows = self.positions[start : start + count]
        return rows + [[] for _ in range(count - len(rows))]
