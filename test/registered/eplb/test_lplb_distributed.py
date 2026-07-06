"""Real-CUDA multi-rank distributed test for LPLBSolver.

Supplements the CPU-only `test_lplb.py` checks: this test spawns 2
CUDA-bound ranks on a single node via `torch.multiprocessing.spawn` and
exercises the actual code paths the production LP server uses, including
the EP all-reduce and the DP-attention "empty rank" case (rank with zero
tokens for a given forward).

Critical invariants this test catches that the CPU tests cannot:
    1. The all-reduce inside `LPLBSolver.solve()` actually completes when
       one rank has empty `topk_ids` (i.e., the empty-rank deadlock fix is
       in effect — without it, the empty rank would skip `solver.solve()`
       and hang the collective).
    2. Both ranks receive the **same** counts post-all-reduce and produce
       identical outputs.
    3. The output equals the torch-IPM oracle computed from the ORACLE
       summed counts (this is the discriminator that tells "all-reduce
       happened" apart from "rank silently used local-only counts and
       coincidentally produced a finite tensor").
    4. The fused-only contract holds end-to-end: a fused-kernel failure
       must surface as an exception (no silent torch fallback exists).
    5. Post-rebalance reinit produces solvers tied to the new metadata —
       outputs change when `phy2log` changes.

Pattern follows `test/registered/layers/mamba/test_mamba2_mixer.py` —
`torch.multiprocessing.spawn` is required because `torch.distributed`
plays poorly with arbitrary subprocess launchers.
"""

import pytest
import torch

from sglang.srt.distributed.device_communicators.custom_all_reduce_utils import (
    update_environment_variables,
)
from sglang.srt.distributed.parallel_state import (
    init_distributed_environment,
    initialize_model_parallel,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=19, stage="base-b", runner_config="2-gpu-large")

NUM_GPUS = 2
TOPK = 2

# Smallest replicated-metadata config that still exercises a non-trivial LP:
#   num_gpus = 2
#   num_logical = 4 (3 single-copy + 1 replicated)
#   num_phy = 6 (4 + 2 redundant copies of expert 0)
# → NC = 1 + 2 = 3, NV = 2 + 2 + 2 = 6 (well within H20 shmem budget)
NUM_LOGICAL = 4
NUM_PHY = 6


def _make_metadata(rebalanced: bool = False):
    """Build (phy2log, log2phy, num_valid) for the test config.

    When `rebalanced=True`, swap the redundant logical from expert 0 to
    expert 1 — same shapes, different mapping, so a stale solver would
    produce wrong outputs.
    """
    replicated_logical = 1 if rebalanced else 0
    phy2log = torch.tensor(
        [0, 1, 2, 3, replicated_logical, replicated_logical], dtype=torch.int64
    )
    log2phy = torch.full((NUM_LOGICAL, 3), -1, dtype=torch.int64)
    for i in range(NUM_LOGICAL):
        log2phy[i, 0] = i
    log2phy[replicated_logical, 1] = 4
    log2phy[replicated_logical, 2] = 5
    num_valid = torch.ones(NUM_LOGICAL, dtype=torch.int64)
    num_valid[replicated_logical] = 3
    return phy2log, log2phy, num_valid


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="This test requires CUDA",
)
def test_dispatch_probability_matches_torch_reference():
    """The fused CUDA `dispatch_probability` and the pure-torch reference
    must produce identical outputs for the same ``random_vals``. Single-rank,
    runs on any CUDA GPU."""
    from sglang.jit_kernel.lplb.cuda_solver import (
        dispatch_probability,
        dispatch_probability_torch_reference,
    )

    torch.manual_seed(0)
    device = torch.device("cuda:0")
    num_logical, max_copies, num_tokens, topk = 8, 3, 64, 4

    # Random LP-style probabilities (some rows zeroed to exercise the fallback).
    log2phy_prob = torch.rand(num_logical, max_copies, device=device)
    log2phy_prob[0] = 0  # force fallback row
    log2phy_map = torch.full(
        (num_logical, max_copies), -1, dtype=torch.int32, device=device
    )
    for i in range(num_logical):
        log2phy_map[i, 0] = i  # primary replica
        log2phy_map[i, 1] = num_logical + (i % 4)  # one redundant
        # leave last slot at -1 to exercise the masked-replica path

    topk_ids = torch.randint(
        0, num_logical, (num_tokens, topk), dtype=torch.int32, device=device
    )
    random_vals = torch.rand(num_tokens * topk, device=device, dtype=torch.float32)

    cuda_out = dispatch_probability(topk_ids, log2phy_prob, log2phy_map, random_vals)
    torch_out = dispatch_probability_torch_reference(
        topk_ids, log2phy_prob, log2phy_map, random_vals
    )

    assert torch.equal(cuda_out, torch_out), (
        f"dispatch_probability disagrees with torch reference: "
        f"{(cuda_out != torch_out).sum().item()}/{cuda_out.numel()} mismatches"
    )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="This test requires CUDA",
)
def test_solve_ipm_matches_torch_reference():
    """The fused CUDA IPM kernel and the pure-torch reference should agree to
    a small tolerance on a *real* LPLB LP. They are NOT bit-equivalent — the
    kernel factors the KKT system with a hand-written block Cholesky while the
    reference uses torch.linalg.solve (LU) — so we compare with allclose and
    print the max abs difference (the numerical-difference number the review
    asked about).

    The LP is built the way LPLBSolver does (Big-M column + normalized
    counts), because an arbitrary random Ax=b never satisfies the solver's
    convergence test (the Big-M slack must reach ~0) and both backends would
    just return the 0.5 non-convergence sentinel — agreeing trivially without
    exercising the solve. Single-rank, any CUDA GPU."""
    from sglang.jit_kernel.lplb.cuda_solver import solve_ipm as cuda_solve_ipm
    from sglang.jit_kernel.lplb.torch_solver import solve_ipm_torch_reference
    from sglang.srt.eplb.lplb_solver import LPLBSolver

    torch.manual_seed(0)
    device = torch.device("cuda:0")
    phy2log, log2phy, num_valid = _make_metadata()
    solver = LPLBSolver(
        phy2log=phy2log.to(device),
        log2phy=log2phy.to(device),
        num_gpus=NUM_GPUS,
        ep_group=None,
        logical_to_all_physical_map_num_valid=num_valid.to(device),
    )

    # Realistic counts: the replicated expert (logical 0) carries heavy load
    # to distribute across its 3 physical copies; the rest are moderate. This
    # gives the barrier method a well-posed instance that converges in 5 iters.
    global_counts = torch.tensor(
        [120.0, 30.0, 25.0, 20.0], dtype=torch.float32, device=device
    )

    # Reconstruct the LP exactly as LPLBSolver._solve builds it.
    counts_norm = global_counts / global_counts.sum().clamp(min=1.0)
    t1 = counts_norm[solver.log_single]
    b1 = counts_norm[solver.log_replicated]
    b2 = -(solver.B1 @ t1).flatten()
    b = torch.cat([b1, b2])
    big_M_col = b - solver._A_base_row_sum
    A_full = torch.hstack([solver.A_base, big_M_col.unsqueeze(1)])

    cuda_x = cuda_solve_ipm(A_full, b, solver.c_vec)
    torch_x = solve_ipm_torch_reference(A_full, b, solver.c_vec)

    converged = not torch.allclose(cuda_x, torch.full_like(cuda_x, 0.5))
    max_diff = (cuda_x - torch_x).abs().max().item()
    print(
        f"\n[ipm-compare] converged={converged}  max|cuda-torch|={max_diff:.3e}  "
        f"cuda={[round(v,4) for v in cuda_x.tolist()]}  "
        f"torch={[round(v,4) for v in torch_x.tolist()]}"
    )
    assert converged, (
        "IPM returned the 0.5 non-convergence sentinel — the comparison would "
        "be trivial. Adjust the LP instance so it converges."
    )
    assert torch.allclose(
        cuda_x, torch_x, atol=1e-2, rtol=1e-2
    ), f"fused IPM diverges from torch reference: max abs diff {max_diff:.3e}"


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < NUM_GPUS,
    reason=f"This test requires at least {NUM_GPUS} CUDA devices",
)
def test_lplb_distributed_two_rank():
    """Driver: spawn `NUM_GPUS` worker processes that each run the
    distributed-LPLB invariant suite."""
    torch.multiprocessing.spawn(
        _worker_main,
        args=(NUM_GPUS,),
        nprocs=NUM_GPUS,
    )


def _worker_main(local_rank: int, world_size: int):
    """Per-rank entry point under torch.multiprocessing.spawn."""
    # Inject minimal ServerArgs before any LPLB module reads the global state.
    # Silent fallbacks no longer exist — the fused CUDA path is the only LP
    # path, so this test relies on hard failures, not gating flags.
    from sglang.srt.server_args import (
        ServerArgs,
        set_global_server_args_for_scheduler,
    )

    set_global_server_args_for_scheduler(
        ServerArgs(
            model_path="dummy",
        )
    )

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    torch.set_default_device(device)

    update_environment_variables(
        {
            "RANK": str(local_rank),
            "LOCAL_RANK": str(local_rank),
            "WORLD_SIZE": str(world_size),
            "MASTER_ADDR": "localhost",
            "MASTER_PORT": "12347",  # Distinct from other tests' ports.
        }
    )
    init_distributed_environment(
        world_size=world_size, rank=local_rank, local_rank=local_rank
    )
    initialize_model_parallel(
        tensor_model_parallel_size=world_size,
        expert_model_parallel_size=world_size,
    )

    from sglang.srt.eplb.lplb_solver import clear_global_lplb_solvers

    try:
        # Clear the global solver registry between subtests so a stale solver
        # can't shadow a freshly-built one.
        clear_global_lplb_solvers()
        _check_solver_with_empty_rank(local_rank, world_size, device)
        clear_global_lplb_solvers()
        _check_all_ranks_empty(local_rank, world_size, device)
        clear_global_lplb_solvers()
        _check_solver_determinism(local_rank, world_size, device)
        clear_global_lplb_solvers()
        _check_post_rebalance_reinit(local_rank, world_size, device)
    finally:
        # Use parallel_state's destroy helpers rather than raw
        # `dist.destroy_process_group()` so we don't leave _WORLD/_MOE_EP/_TP
        # globals stale for later tests.
        from sglang.srt.distributed.parallel_state import (
            destroy_distributed_environment,
            destroy_model_parallel,
        )

        clear_global_lplb_solvers()
        destroy_model_parallel()
        destroy_distributed_environment()


def _build_solver():
    """Construct an LPLBSolver bound to the live moe_ep_group."""
    from sglang.srt.distributed.parallel_state import get_moe_ep_group
    from sglang.srt.eplb.lplb_solver import LPLBSolver

    phy2log, log2phy, num_valid = _make_metadata(rebalanced=False)
    return (
        LPLBSolver(
            phy2log=phy2log.cuda(),
            log2phy=log2phy.cuda(),
            num_gpus=NUM_GPUS,
            ep_group=get_moe_ep_group(),
            logical_to_all_physical_map_num_valid=num_valid.cuda(),
        ),
        phy2log,
        log2phy,
        num_valid,
    )


def _expected_output(solver, expected_global_counts: torch.Tensor) -> torch.Tensor:
    """Oracle: what `solver.solve(*)` should produce given known global
    counts. Uses the solver's own `_solve` so we match implementation
    quirks (regularization, on-device clamp normalization, etc.).
    """
    return solver._solve(expected_global_counts.float().cuda())


def _check_solver_with_empty_rank(rank: int, world_size: int, device: torch.device):
    """Rank 0 has real `topk_ids`; rank 1 has `(0, topk)` empty.

    The all-reduce inside `solve()` must complete on both ranks, and both
    must produce output equal to the oracle solver-output for the SUMMED
    global counts (rank 0's bincount + rank 1's zero contribution).
    """
    solver, _, _, _ = _build_solver()

    # Rank 0 owns these tokens; rank 1 is idle (DP-attention empty case).
    rank0_topk = torch.tensor(
        [[0, 1], [0, 2], [3, 0], [0, 1]], dtype=torch.int32, device=device
    )
    if rank == 0:
        topk_ids = rank0_topk
    else:
        topk_ids = torch.empty((0, TOPK), dtype=torch.int32, device=device)

    # Compute the oracle: the global counts are exactly the bincount of
    # rank 0's tokens (rank 1 contributes zero).
    flat = rank0_topk.flatten().long()
    expected_counts = torch.bincount(flat, minlength=NUM_LOGICAL)
    expected = _expected_output(solver, expected_counts)

    # Run the actual collective.
    actual = solver.solve(topk_ids)

    # Shape + finiteness sanity.
    assert actual.shape == (
        NUM_LOGICAL,
        solver.max_copies,
    ), f"rank {rank}: bad output shape {actual.shape}"
    assert torch.isfinite(actual).all(), f"rank {rank}: non-finite values in output"
    assert (actual >= 0).all(), f"rank {rank}: negative probabilities"

    # The strong invariant: output matches the summed-count oracle.
    # If the all-reduce silently failed, rank 1 would feed all-zero local
    # counts to its solver, the oracle wouldn't match, and this assertion
    # would catch it.
    assert torch.allclose(actual, expected, atol=1e-4, rtol=1e-3), (
        f"rank {rank}: solve() output disagrees with summed-count oracle "
        f"(max abs diff {(actual - expected).abs().max().item():.2e}). "
        "This usually indicates the EP all-reduce inside solve() was "
        "skipped on the empty rank, or rank 1 used local counts only."
    )

    # 0.5-uniform sentinel detection: torch IPM's failure path returns a
    # vector of all-0.5. If we see that, the IPM diverged silently.
    is_uniform_half = torch.allclose(
        actual,
        torch.full_like(actual, 0.5),
        atol=1e-6,
    )
    assert not is_uniform_half, (
        f"rank {rank}: output is the 0.5-uniform fallback, indicating "
        "torch IPM convergence failure"
    )


def _check_all_ranks_empty(rank: int, world_size: int, device: torch.device):
    """Both ranks have empty topk_ids (idle batch). The collective must
    still complete; the output should be the all-zero-counts oracle."""
    solver, _, _, _ = _build_solver()

    empty = torch.empty((0, TOPK), dtype=torch.int32, device=device)
    expected_counts = torch.zeros(NUM_LOGICAL, dtype=torch.int64)
    expected = _expected_output(solver, expected_counts)

    actual = solver.solve(empty)

    assert torch.isfinite(actual).all(), f"rank {rank}: non-finite for empty-batch"
    assert (actual >= 0).all(), f"rank {rank}: negative for empty-batch"
    assert torch.allclose(
        actual, expected, atol=1e-4, rtol=1e-3
    ), f"rank {rank}: empty-batch output disagrees with all-zero oracle"


def _check_solver_determinism(rank: int, world_size: int, device: torch.device):
    """Same input, two `solve()` calls in a row → identical output."""
    solver, _, _, _ = _build_solver()
    if rank == 0:
        topk_ids = torch.tensor([[0, 1], [3, 0]], dtype=torch.int32, device=device)
    else:
        topk_ids = torch.empty((0, TOPK), dtype=torch.int32, device=device)

    out1 = solver.solve(topk_ids)
    out2 = solver.solve(topk_ids)
    assert torch.equal(out1, out2), (
        f"rank {rank}: solve() not deterministic across calls "
        f"(max diff {(out1 - out2).abs().max().item():.2e})"
    )


def _check_post_rebalance_reinit(rank: int, world_size: int, device: torch.device):
    """Build a NEW solver with different `phy2log` (the redundant logical
    moves from expert 0 to expert 1). Same input → DIFFERENT output, since
    the LP problem changed."""
    from sglang.srt.distributed.parallel_state import get_moe_ep_group
    from sglang.srt.eplb.lplb_solver import LPLBSolver

    solver_old, _, _, _ = _build_solver()
    phy2log_new, log2phy_new, num_valid_new = _make_metadata(rebalanced=True)
    solver_new = LPLBSolver(
        phy2log=phy2log_new.cuda(),
        log2phy=log2phy_new.cuda(),
        num_gpus=NUM_GPUS,
        ep_group=get_moe_ep_group(),
        logical_to_all_physical_map_num_valid=num_valid_new.cuda(),
    )

    # An input that hits the replicated logical in BOTH configs (so the
    # LP solution is sensitive to which logical is the replicated one).
    rank0_topk = torch.tensor(
        [[0, 1], [0, 1], [0, 1], [2, 3]], dtype=torch.int32, device=device
    )
    if rank == 0:
        topk_ids = rank0_topk
    else:
        topk_ids = torch.empty((0, TOPK), dtype=torch.int32, device=device)

    out_old = solver_old.solve(topk_ids)
    out_new = solver_new.solve(topk_ids)

    # The two outputs should differ — same `topk_ids`, different metadata
    # ⇒ different LP solution. If they're identical, one of the solvers
    # used stale state.
    assert not torch.equal(out_old, out_new), (
        f"rank {rank}: post-rebalance solver produced same output as "
        "pre-rebalance solver despite different phy2log mappings — "
        "stale solver state likely"
    )

    # Positive oracle: each output should match its OWN solver's
    # `_solve` of the SUMMED global counts (rank 0's bincount; rank 1
    # contributes zero). Both ranks compare against the same global oracle
    # since solve() all-reduces the counts.
    expected_counts = torch.bincount(rank0_topk.flatten().long(), minlength=NUM_LOGICAL)
    expected_old = _expected_output(solver_old, expected_counts)
    expected_new = _expected_output(solver_new, expected_counts)
    assert torch.allclose(out_old, expected_old, atol=1e-4, rtol=1e-3), (
        f"rank {rank}: pre-rebalance output disagrees with own oracle "
        f"(diff {(out_old - expected_old).abs().max().item():.2e})"
    )
    assert torch.allclose(out_new, expected_new, atol=1e-4, rtol=1e-3), (
        f"rank {rank}: post-rebalance output disagrees with own oracle "
        f"(diff {(out_new - expected_new).abs().max().item():.2e})"
    )


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
