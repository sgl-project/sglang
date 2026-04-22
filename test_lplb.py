#!/usr/bin/env python3
"""Functional test for LPLB solver — verifies correctness without a full server."""
import sys
import torch

def test_torch_solver():
    """Test the IPM solver directly."""
    from sglang.jit_kernel.lplb.torch_solver import solve_ipm

    # Simple LP: min x3 s.t. x1 + x2 = 1, x1 + x3 >= 0.5 (reformulated)
    # Standard form: min c^T x s.t. Ax = b, x >= 0
    NC, NV = 4, 6
    A = torch.randn(NC, NV, dtype=torch.float32)
    b = A @ torch.ones(NV, dtype=torch.float32)  # feasible point x=1
    c = torch.zeros(NV, dtype=torch.float32)
    c[-2] = 1.0   # minimize second-to-last
    c[-1] = 1000.0  # Big-M penalty

    x = solve_ipm(A, b, c, num_iters=5)
    assert x.shape == (NV,), f"Shape mismatch: {x.shape}"
    assert x.dtype == torch.float32
    assert torch.isfinite(x).all(), "Non-finite values in solution"
    print(f"  torch_solver: OK (x range [{x.min():.4f}, {x.max():.4f}])")


def test_lplb_solver_init():
    """Test LPLBSolver initialization with a realistic config."""
    from sglang.srt.eplb.lplb_solver import LPLBSolver

    num_gpus = 4
    num_logical = 32
    num_phy_per_gpu = 10  # 40 total physical
    num_phy = num_gpus * num_phy_per_gpu  # 40

    # Build phy2log: first 32 are 1:1, next 8 are copies of experts 0-7
    phy2log = torch.arange(num_logical, dtype=torch.int64)
    extra = torch.arange(8, dtype=torch.int64)  # 8 redundant copies
    phy2log = torch.cat([phy2log, extra])  # 40 physical experts

    # Build log2phy: experts 0-7 have 2 copies each, rest have 1
    max_copies = 2
    log2phy = torch.full((num_logical, max_copies), -1, dtype=torch.int64)
    for i in range(num_logical):
        log2phy[i, 0] = i  # primary copy
    for i in range(8):
        log2phy[i, 1] = num_logical + i  # redundant copy

    num_valid = torch.ones(num_logical, dtype=torch.int64)
    num_valid[:8] = 2  # first 8 have 2 copies

    solver = LPLBSolver(
        phy2log=phy2log,
        log2phy=log2phy,
        num_gpus=num_gpus,
        ep_group=None,  # single rank
        logical_to_all_physical_map_num_valid=num_valid,
    )

    assert solver.num_logical == num_logical
    assert solver.num_phy == num_phy
    assert solver._has_redundancy
    assert solver.num_red_log == 8
    print(f"  LPLBSolver init: OK (red_log={solver.num_red_log}, "
          f"red_phy={solver.num_red_phy}, single={solver.num_single})")
    return solver, log2phy, num_valid


def test_lplb_solve(solver):
    """Test solve() with sample token counts."""
    # Simulate token counts: experts 0-7 are hot (more tokens)
    topk_ids = torch.randint(0, solver.num_logical, (64, 8), dtype=torch.int32)
    # Bias toward experts 0-7
    hot_tokens = torch.randint(0, 8, (32, 8), dtype=torch.int32)
    topk_ids[:32] = hot_tokens

    log2phy_prob = solver.solve(topk_ids)

    assert log2phy_prob.shape == (solver.num_logical, solver.max_copies)
    assert log2phy_prob.dtype == torch.float32
    assert torch.isfinite(log2phy_prob).all(), "Non-finite in log2phy_prob"
    assert (log2phy_prob >= 0).all(), "Negative probabilities"

    # Replicated experts should have non-zero probabilities for both copies
    rep_probs = log2phy_prob[:8]  # first 8 are replicated
    assert (rep_probs[:, 0] > 0).all(), "Primary copy has zero prob"
    # At least some secondary copies should have non-zero prob
    assert (rep_probs[:, 1] > 0).any(), "All secondary copies have zero prob"

    print(f"  solve(): OK (shape={log2phy_prob.shape}, "
          f"sum range [{log2phy_prob.sum(dim=1).min():.4f}, "
          f"{log2phy_prob.sum(dim=1).max():.4f}])")
    return log2phy_prob


def test_lplb_solve_empty(solver):
    """Test solve() with empty topk_ids (idle rank scenario)."""
    empty_ids = torch.empty((0, 8), dtype=torch.int32)
    log2phy_prob = solver.solve(empty_ids)

    assert log2phy_prob.shape == (solver.num_logical, solver.max_copies)
    assert torch.isfinite(log2phy_prob).all()
    print(f"  solve(empty): OK (shape={log2phy_prob.shape})")


def test_probability_dispatch():
    """Test _topk_ids_logical_to_physical_probability."""
    from sglang.srt.eplb.expert_location_dispatch import (
        _topk_ids_logical_to_physical_probability,
        ExpertLocationDispatchInfo,
    )

    num_logical = 8
    max_copies = 2
    # log2phy_map: expert i -> physical [i, i+8]
    log2phy_map = torch.zeros(num_logical, max_copies, dtype=torch.int64)
    for i in range(num_logical):
        log2phy_map[i, 0] = i
        log2phy_map[i, 1] = i + num_logical

    num_valid = torch.full((num_logical,), 2, dtype=torch.int64)

    info = ExpertLocationDispatchInfo(
        ep_dispatch_algorithm="lp",
        partial_logical_to_rank_dispatch_physical_map=None,
        partial_logical_to_all_physical_map=log2phy_map,
        partial_logical_to_all_physical_map_num_valid=num_valid,
        num_physical_experts=num_logical * 2,
        lplb_solver=None,
    )

    # Create probabilities: 70% primary, 30% secondary
    log2phy_prob = torch.zeros(num_logical, max_copies, dtype=torch.float32)
    log2phy_prob[:, 0] = 0.7
    log2phy_prob[:, 1] = 0.3

    topk_ids = torch.tensor([[0, 1, 2], [3, 4, 5]], dtype=torch.int32)
    result = _topk_ids_logical_to_physical_probability(topk_ids, info, log2phy_prob)

    assert result.shape == topk_ids.shape
    # All results should be valid physical IDs
    assert (result >= 0).all()
    assert (result < num_logical * 2).all()
    print(f"  probability_dispatch: OK (result={result.tolist()})")


def test_probability_dispatch_zero_probs():
    """Test fallback when probabilities are zero."""
    from sglang.srt.eplb.expert_location_dispatch import (
        _topk_ids_logical_to_physical_probability,
        ExpertLocationDispatchInfo,
    )

    num_logical = 4
    max_copies = 2
    log2phy_map = torch.zeros(num_logical, max_copies, dtype=torch.int64)
    for i in range(num_logical):
        log2phy_map[i, 0] = i
        log2phy_map[i, 1] = i + num_logical

    num_valid = torch.full((num_logical,), 2, dtype=torch.int64)
    info = ExpertLocationDispatchInfo(
        ep_dispatch_algorithm="lp",
        partial_logical_to_rank_dispatch_physical_map=None,
        partial_logical_to_all_physical_map=log2phy_map,
        partial_logical_to_all_physical_map_num_valid=num_valid,
        num_physical_experts=num_logical * 2,
        lplb_solver=None,
    )

    # All-zero probabilities should trigger fallback to uniform
    log2phy_prob = torch.zeros(num_logical, max_copies, dtype=torch.float32)
    topk_ids = torch.tensor([[0, 1]], dtype=torch.int32)
    result = _topk_ids_logical_to_physical_probability(topk_ids, info, log2phy_prob)
    assert result.shape == topk_ids.shape
    assert (result >= 0).all()
    print(f"  zero_probs_fallback: OK (result={result.tolist()})")


def test_global_solver_registry():
    """Test the global solver get/set/clear functions."""
    from sglang.srt.eplb.lplb_solver import (
        get_global_lplb_solver,
        set_global_lplb_solver,
        clear_global_lplb_solvers,
    )

    clear_global_lplb_solvers()
    assert get_global_lplb_solver(0) is None

    sentinel = object()
    set_global_lplb_solver(0, sentinel)
    assert get_global_lplb_solver(0) is sentinel
    assert get_global_lplb_solver(1) is None

    clear_global_lplb_solvers()
    assert get_global_lplb_solver(0) is None
    print("  global_registry: OK")


def test_solve_determinism(solver):
    """Test that same input produces same output (no randomness in solver)."""
    counts = torch.zeros(solver.num_logical, dtype=torch.float32)
    counts[:8] = torch.tensor([10, 20, 15, 5, 8, 12, 25, 3], dtype=torch.float32)
    counts[8:16] = 5.0

    # Build topk_ids from counts
    topk_ids = torch.tensor([[0, 1], [2, 3], [0, 4]], dtype=torch.int32)

    prob1 = solver.solve(topk_ids)
    prob2 = solver.solve(topk_ids)

    assert torch.allclose(prob1, prob2, atol=1e-6), \
        f"Non-deterministic: max diff = {(prob1 - prob2).abs().max()}"
    print("  determinism: OK")


if __name__ == "__main__":
    print("=" * 50)
    print("LPLB Functional Tests")
    print("=" * 50)

    tests = [
        ("torch_solver", test_torch_solver),
        ("lplb_solver_init", lambda: test_lplb_solver_init()),
        ("lplb_solve", None),  # needs solver from init
        ("lplb_solve_empty", None),
        ("solve_determinism", None),
        ("probability_dispatch", test_probability_dispatch),
        ("zero_probs_fallback", test_probability_dispatch_zero_probs),
        ("global_registry", test_global_solver_registry),
    ]

    passed = 0
    failed = 0

    # Independent tests
    for name, fn in tests:
        if fn is None:
            continue
        try:
            fn()
            passed += 1
        except Exception as e:
            print(f"  {name}: FAILED — {e}")
            import traceback; traceback.print_exc()
            failed += 1

    # Tests that depend on solver instance
    try:
        solver, log2phy, num_valid = test_lplb_solver_init()
        passed += 1  # init already counted above, skip

        test_lplb_solve(solver)
        passed += 1

        test_lplb_solve_empty(solver)
        passed += 1

        test_solve_determinism(solver)
        passed += 1
    except Exception as e:
        print(f"  solver tests: FAILED — {e}")
        import traceback; traceback.print_exc()
        failed += 1

    print()
    print(f"Results: {passed} passed, {failed} failed")
    sys.exit(1 if failed > 0 else 0)
