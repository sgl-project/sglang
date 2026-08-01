# Reference: https://github.com/feifeibear/long-context-attention/blob/main/yunchang/globals.py


import torch

from sglang.multimodal_gen.runtime.distributed.utils import (
    NCCL2_DEVICE_BACKEND,
    is_nccl2_world,
)


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

    # Collect the rank lists first. Both families are partitions: the ulysses
    # groups partition each SP group, and so do the ring groups (with a strided
    # index pattern), and the SP groups are themselves disjoint.
    ulysses_rank_groups: list[list[int]] = []
    ring_rank_groups: list[list[int]] = []
    for sp_ranks in sp_groups:
        if use_ulysses_low:
            for i in range(num_ulysses_pgs):
                idx = list(range(i * sp_ulysses_degree, (i + 1) * sp_ulysses_degree))
                ulysses_rank_groups.append(_map_indices_to_ranks(sp_ranks, idx))
            for i in range(num_ring_pgs):
                idx = list(range(i, sp_degree, num_ring_pgs))
                ring_rank_groups.append(_map_indices_to_ranks(sp_ranks, idx))
        else:
            for i in range(num_ring_pgs):
                idx = list(range(i * sp_ring_degree, (i + 1) * sp_ring_degree))
                ring_rank_groups.append(_map_indices_to_ranks(sp_ranks, idx))
            for i in range(num_ulysses_pgs):
                idx = list(range(i, sp_degree, num_ulysses_pgs))
                ulysses_rank_groups.append(_map_indices_to_ranks(sp_ranks, idx))

    if is_nccl2_world():
        # These groups have asymmetric membership, so an eager new_group() over
        # an nccl2 world would blow up on the non-member ranks with
        # "'ProcessGroupNCCL2' object has no attribute
        # 'perform_nocolor_split'". Carve them off the world PG instead: one
        # collective split per family (NOT one per group -- split_group already
        # handles the whole partition in a single call).
        #
        # split_group preserves the order of ranks within each split, so the
        # strided ring ordering is kept. Note this differs from new_group, which
        # sorts by default (sort_ranks=True); for the index patterns used here
        # both orderings coincide because sp_groups are ascending.
        def _split(rank_groups: list[list[int]]):
            if not rank_groups:
                return None
            pg = torch.distributed.split_group(
                split_ranks=rank_groups, backend=NCCL2_DEVICE_BACKEND
            )
            # Match the legacy behaviour: ranks outside every group get None.
            if pg is torch.distributed.GroupMember.NON_GROUP_MEMBER:
                return None
            return pg

        if use_ulysses_low:
            ulyssess_pg = _split(ulysses_rank_groups)
            ring_pg = _split(ring_rank_groups)
        else:
            ring_pg = _split(ring_rank_groups)
            ulyssess_pg = _split(ulysses_rank_groups)
    else:
        # Important: call torch.distributed.new_group in the same order on all ranks.
        def _new_groups(rank_groups: list[list[int]]):
            my_pg = None
            for group_ranks in rank_groups:
                group = torch.distributed.new_group(group_ranks)
                if rank in group_ranks:
                    my_pg = group
            return my_pg

        if use_ulysses_low:
            ulyssess_pg = _new_groups(ulysses_rank_groups)
            ring_pg = _new_groups(ring_rank_groups)
        else:
            ring_pg = _new_groups(ring_rank_groups)
            ulyssess_pg = _new_groups(ulysses_rank_groups)

    PROCESS_GROUP.ULYSSES_PG = ulyssess_pg
    PROCESS_GROUP.RING_PG = ring_pg
