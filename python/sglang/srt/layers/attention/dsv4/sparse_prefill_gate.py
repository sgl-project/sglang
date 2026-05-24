from typing import List, Optional


def can_use_sparse_prefill(
    *,
    q_num_rows: int,
    batch_size: int,
    extend_seq_lens_cpu: Optional[List[int]],
    is_cp_round_robin: bool,
    cp_num_rows: Optional[int] = None,
) -> bool:
    if extend_seq_lens_cpu is None:
        return False
    if is_cp_round_robin:
        return batch_size == 1 and cp_num_rows == q_num_rows
    return sum(extend_seq_lens_cpu) == q_num_rows
