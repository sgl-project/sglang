"""Capture-safe launch bounds shared by KPool metadata kernels."""

_TILE_PROGRAM_TARGET = 8192


def bounded_scan_num_splits(rows: int, num_col_blocks: int) -> int:
    """Keep the grid capture-safe while bounding traversal by replay-time data."""
    assert rows > 0
    return max(1, min(num_col_blocks, _TILE_PROGRAM_TARGET // rows))


# The metadata kernels take the batch size at run time and size their row blocks
# with these floors, so one compiled variant serves every decode batch size up to
# the floor; a per-size specialization would JIT on the serving path the first
# time each batch size appears.
_MIN_BLOCK_BS = 64
_MIN_BLOCK_EXPANDED = 512


def _metadata_block(n: int, floor: int) -> int:
    """Power-of-two row block covering ``n``, at least ``floor``."""
    block = 1
    while block < n:
        block *= 2
    return max(block, floor)
