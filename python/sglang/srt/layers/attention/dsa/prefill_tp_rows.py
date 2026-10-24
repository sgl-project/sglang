"""Partition replicated indexer rows without changing their global row indices."""


def shard_row_chunks(
    row_chunks: tuple[slice, ...], world_size: int, rank: int
) -> tuple[tuple[slice, ...], int]:
    num_rows = row_chunks[-1].stop
    rows_per_rank = (num_rows + world_size - 1) // world_size
    start = rank * rows_per_rank
    end = min(start + rows_per_rank, num_rows)
    chunks = tuple(
        slice(max(start, chunk.start), min(end, chunk.stop))
        for chunk in row_chunks
        if max(start, chunk.start) < min(end, chunk.stop)
    )
    return chunks, rows_per_rank * world_size
