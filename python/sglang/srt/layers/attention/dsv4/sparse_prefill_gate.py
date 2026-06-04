from typing import List, Optional


def can_use_sparse_prefill(
    *,
    q_num_rows: int,
    batch_size: int,
    extend_seq_lens_cpu: Optional[List[int]],
    is_cp_round_robin: bool,
    cp_num_rows: Optional[int] = None,
    cp_size: int = 1,
) -> bool:
    if extend_seq_lens_cpu is None:
        return False
    if is_cp_round_robin:
        # Single request only (the round-robin interleave can't be described
        # by one query_start_loc across multiple requests), local row count
        # must match the reindexed positions, AND the chunk must be unpadded:
        # when the global token count isn't a multiple of cp_size the batch is
        # ceil-aligned with trailing padding rows (pos=0, all-(-1) page masks)
        # that poison the c128 mask selection and write OOB SWA combine
        # indices. Those padded (ragged-tail) chunks fall back to dense.
        return (
            batch_size == 1
            and cp_num_rows == q_num_rows
            and sum(extend_seq_lens_cpu) % cp_size == 0
        )
    return sum(extend_seq_lens_cpu) == q_num_rows
