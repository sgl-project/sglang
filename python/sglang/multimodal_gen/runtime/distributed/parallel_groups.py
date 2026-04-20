# Reference: https://github.com/feifeibear/long-context-attention/blob/main/yunchang/globals.py


import torch


class Singleton:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(Singleton, cls).__new__(cls, *args, **kwargs)
        return cls._instance


class ProcessGroupSingleton(Singleton):
    def __init__(self):
        self.ULYSSES_PG = None
        self.RING_PG = None


PROCESS_GROUP = ProcessGroupSingleton()


def set_seq_parallel_pg_by_sp_groups(
    sp_ulysses_degree,
    sp_ring_degree,
    rank: int,
    sp_groups: list[list[int]],
    use_ulysses_low: bool = True,
):
    """Create Ulysses/Ring process groups inside each SP group.

    This is required when TP>1, because SP groups are not necessarily made of
    consecutive global ranks (e.g., tp-sp order makes SP ranks strided).

    Args:
        sp_ulysses_degree: ulysses degree inside SP.
        sp_ring_degree: ring degree inside SP.
        rank: global rank of current process.
        sp_groups: list of global-rank lists for each SP group.
        use_ulysses_low: keep the same semantics as the original function.
    """
    sp_degree = sp_ring_degree * sp_ulysses_degree
    assert sp_degree > 0
    assert all(
        len(g) == sp_degree for g in sp_groups
    ), f"Each SP group must have size {sp_degree}, got sizes {[len(g) for g in sp_groups]}"

    ulyssess_pg = None
    ring_pg = None

    num_ulysses_pgs = sp_ring_degree
    num_ring_pgs = sp_ulysses_degree

    def _map_indices_to_ranks(ranks: list[int], indices: list[int]) -> list[int]:
        return [ranks[i] for i in indices]

    # Important: call torch.distributed.new_group in the same order on all ranks.
    for sp_ranks in sp_groups:
        if use_ulysses_low:
            for i in range(num_ulysses_pgs):
                idx = list(range(i * sp_ulysses_degree, (i + 1) * sp_ulysses_degree))
                ulysses_ranks = _map_indices_to_ranks(sp_ranks, idx)
                group = torch.distributed.new_group(ulysses_ranks)
                if rank in ulysses_ranks:
                    ulyssess_pg = group

            for i in range(num_ring_pgs):
                idx = list(range(i, sp_degree, num_ring_pgs))
                ring_ranks = _map_indices_to_ranks(sp_ranks, idx)
                group = torch.distributed.new_group(ring_ranks)
                if rank in ring_ranks:
                    ring_pg = group
        else:
            for i in range(num_ring_pgs):
                idx = list(range(i * sp_ring_degree, (i + 1) * sp_ring_degree))
                ring_ranks = _map_indices_to_ranks(sp_ranks, idx)
                group = torch.distributed.new_group(ring_ranks)
                if rank in ring_ranks:
                    ring_pg = group

            for i in range(num_ulysses_pgs):
                idx = list(range(i, sp_degree, num_ulysses_pgs))
                ulysses_ranks = _map_indices_to_ranks(sp_ranks, idx)
                group = torch.distributed.new_group(ulysses_ranks)
                if rank in ulysses_ranks:
                    ulyssess_pg = group

    PROCESS_GROUP.ULYSSES_PG = ulyssess_pg
    PROCESS_GROUP.RING_PG = ring_pg
