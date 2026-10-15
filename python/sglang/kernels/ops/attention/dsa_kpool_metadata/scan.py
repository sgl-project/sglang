"""Capture-safe launch bounds shared by KPool metadata kernels."""

_TILE_PROGRAM_TARGET = 8192


def bounded_scan_num_splits(rows: int, num_col_blocks: int) -> int:
    """Keep the grid capture-safe while bounding traversal by replay-time data."""
    assert rows > 0
    return max(1, min(num_col_blocks, _TILE_PROGRAM_TARGET // rows))
