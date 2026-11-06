"""DFLASH verify commit with rank 0's decision broadcast fused in (csrc/distributed/dflash_commit.cuh)."""

from __future__ import annotations

import functools

import torch


@functools.lru_cache(maxsize=None)
def _use_pdl() -> bool:
    from sglang.kernels.jit.utils import is_arch_support_pdl

    return bool(is_arch_support_pdl())


@functools.lru_cache(maxsize=None)
def module(world: int, pdl: bool):
    from sglang.kernels.jit.utils import load_jit, make_cpp_args

    args = make_cpp_args(max(int(world), 1), pdl)
    return load_jit(
        "dflash_commit",
        *args,
        cuda_files=["distributed/dflash_commit.cuh"],
        cuda_wrappers=[
            ("run", f"DflashCommit<{args}>::run"),
            ("run_local", f"DflashCommit<{args}>::run_local"),
        ],
    )


def commit(candidates, accept_len, bonus, comm=None):
    """Returns (out_tokens, commit_lens). With ``comm``, rank 0's accept_len / bonus are first delivered to every rank
    of its push plane (collective; ``accept_len`` / ``bonus`` are overwritten with rank 0's values)."""
    bs = candidates.shape[0]
    out_tokens = torch.empty_like(candidates, dtype=torch.int64)
    commit_lens = torch.empty((bs,), dtype=torch.int32, device=candidates.device)
    if comm is not None:
        module(comm.world_size, _use_pdl()).run(
            comm.obj, candidates, accept_len, bonus, out_tokens, commit_lens
        )
    else:
        module(1, _use_pdl()).run_local(
            candidates, accept_len, bonus, out_tokens, commit_lens
        )
    return out_tokens, commit_lens
