"""Functional tests for LPLB solver — verifies correctness without a full server.

Run with `pytest test_lplb.py -v` from the sglang/ root.
"""
import pytest
import torch


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def lplb_solver():
    """Build an LPLBSolver with a realistic small config:
    4 GPUs × 10 phy/GPU = 40 physical experts; first 8 logical experts are
    replicated 2× (giving 8 redundant logicals, 16 redundant physicals).
    """
    from sglang.srt.eplb.lplb_solver import LPLBSolver

    num_gpus = 4
    num_logical = 32
    num_phy_per_gpu = 10
    num_phy = num_gpus * num_phy_per_gpu  # 40

    # phy2log: first 32 are 1:1 mapping, next 8 are duplicate copies of 0-7.
    phy2log = torch.arange(num_logical, dtype=torch.int64)
    extra = torch.arange(8, dtype=torch.int64)
    phy2log = torch.cat([phy2log, extra])  # 40 entries

    # log2phy[i] lists the physical IDs for logical i; -1 padding when fewer
    # than max_copies entries.
    max_copies = 2
    log2phy = torch.full((num_logical, max_copies), -1, dtype=torch.int64)
    for i in range(num_logical):
        log2phy[i, 0] = i
    for i in range(8):
        log2phy[i, 1] = num_logical + i

    num_valid = torch.ones(num_logical, dtype=torch.int64)
    num_valid[:8] = 2

    return LPLBSolver(
        phy2log=phy2log,
        log2phy=log2phy,
        num_gpus=num_gpus,
        ep_group=None,
        logical_to_all_physical_map_num_valid=num_valid,
    )


@pytest.fixture
def dispatch_info_factory():
    """Build a minimal ExpertLocationDispatchInfo for probability-dispatch tests."""
    from sglang.srt.eplb.expert_location_dispatch import ExpertLocationDispatchInfo

    def _build(num_logical: int, max_copies: int = 2):
        log2phy_map = torch.zeros(num_logical, max_copies, dtype=torch.int64)
        for i in range(num_logical):
            log2phy_map[i, 0] = i
            log2phy_map[i, 1] = i + num_logical
        num_valid = torch.full((num_logical,), max_copies, dtype=torch.int64)
        return ExpertLocationDispatchInfo(
            ep_dispatch_algorithm="lp",
            partial_logical_to_rank_dispatch_physical_map=None,
            partial_logical_to_all_physical_map=log2phy_map,
            partial_logical_to_all_physical_map_num_valid=num_valid,
            num_physical_experts=num_logical * 2,
            lplb_solver=None,
        )

    return _build


# ---------------------------------------------------------------------------
# Direct IPM solver
# ---------------------------------------------------------------------------


def test_torch_solver_returns_finite_solution():
    """The IPM solver returns a finite-valued vector of the right shape."""
    from sglang.jit_kernel.lplb.torch_solver import solve_ipm

    nc, nv = 4, 6
    a = torch.randn(nc, nv, dtype=torch.float32)
    b = a @ torch.ones(nv, dtype=torch.float32)  # feasible point at x=1
    c = torch.zeros(nv, dtype=torch.float32)
    c[-2] = 1.0  # minimize second-to-last (the LP slack)
    c[-1] = 1000.0  # Big-M penalty

    x = solve_ipm(a, b, c, num_iters=5)

    assert x.shape == (nv,)
    assert x.dtype == torch.float32
    assert torch.isfinite(x).all(), "Non-finite values in solution"


# ---------------------------------------------------------------------------
# LPLBSolver — init, solve, determinism
# ---------------------------------------------------------------------------


def test_lplb_solver_init_dimensions(lplb_solver):
    """Constructor populates the expected shape attributes."""
    assert lplb_solver.num_logical == 32
    assert lplb_solver.num_phy == 40
    assert lplb_solver._has_redundancy
    assert lplb_solver.num_red_log == 8


def test_lplb_solver_rejects_non_divisible_phy_count():
    """num_phy must be divisible by num_gpus (per-rank-contiguous ownership);
    EPLB allocations that violate this should fail loudly at init time, not
    silently produce wrong B1/B2 matrices."""
    from sglang.srt.eplb.lplb_solver import LPLBSolver

    num_gpus = 4
    num_logical = 16
    # 17 physical experts — not divisible by 4.
    phy2log = torch.cat(
        [
            torch.arange(num_logical, dtype=torch.int64),
            torch.tensor([0], dtype=torch.int64),
        ]
    )
    log2phy = torch.full((num_logical, 2), -1, dtype=torch.int64)
    for i in range(num_logical):
        log2phy[i, 0] = i
    log2phy[0, 1] = num_logical
    num_valid = torch.ones(num_logical, dtype=torch.int64)
    num_valid[0] = 2

    with pytest.raises(ValueError, match="divisible by num_gpus"):
        LPLBSolver(
            phy2log=phy2log,
            log2phy=log2phy,
            num_gpus=num_gpus,
            ep_group=None,
            logical_to_all_physical_map_num_valid=num_valid,
        )


def test_lplb_solve_returns_valid_probabilities(lplb_solver):
    """solve() returns a finite, non-negative probability tensor of the right shape,
    with non-zero mass on every replicated expert's primary copy and at least
    one secondary copy."""
    topk_ids = torch.randint(0, lplb_solver.num_logical, (64, 8), dtype=torch.int32)
    # Bias toward experts 0-7 so the replicated copies see traffic.
    topk_ids[:32] = torch.randint(0, 8, (32, 8), dtype=torch.int32)

    log2phy_prob = lplb_solver.solve(topk_ids)

    assert log2phy_prob.shape == (lplb_solver.num_logical, lplb_solver.max_copies)
    assert log2phy_prob.dtype == torch.float32
    assert torch.isfinite(log2phy_prob).all(), "Non-finite in log2phy_prob"
    assert (log2phy_prob >= 0).all(), "Negative probabilities"

    rep_probs = log2phy_prob[:8]  # the 8 logically-replicated experts
    assert (rep_probs[:, 0] > 0).all(), "Primary copy has zero prob"
    assert (rep_probs[:, 1] > 0).any(), "All secondary copies have zero prob"


def test_lplb_solve_handles_empty_topk(lplb_solver):
    """An empty topk_ids tensor (idle DP rank) must still produce a valid output."""
    empty_ids = torch.empty((0, 8), dtype=torch.int32)
    log2phy_prob = lplb_solver.solve(empty_ids)

    assert log2phy_prob.shape == (lplb_solver.num_logical, lplb_solver.max_copies)
    assert torch.isfinite(log2phy_prob).all()


def test_lplb_solve_is_deterministic(lplb_solver):
    """Same input → same output (the IPM has no internal randomness)."""
    topk_ids = torch.tensor([[0, 1], [2, 3], [0, 4]], dtype=torch.int32)
    prob1 = lplb_solver.solve(topk_ids)
    prob2 = lplb_solver.solve(topk_ids)
    assert torch.allclose(prob1, prob2, atol=1e-6), (
        f"Non-deterministic: max diff = {(prob1 - prob2).abs().max()}"
    )


# ---------------------------------------------------------------------------
# Probability-based physical dispatch
# ---------------------------------------------------------------------------


def test_probability_dispatch_returns_valid_physical_ids(dispatch_info_factory):
    """_topk_ids_logical_to_physical_probability picks valid physical IDs
    given a non-degenerate probability distribution."""
    from sglang.srt.eplb.expert_location_dispatch import (
        _topk_ids_logical_to_physical_probability,
    )

    num_logical = 8
    info = dispatch_info_factory(num_logical, max_copies=2)

    log2phy_prob = torch.zeros(num_logical, 2, dtype=torch.float32)
    log2phy_prob[:, 0] = 0.7
    log2phy_prob[:, 1] = 0.3

    topk_ids = torch.tensor([[0, 1, 2], [3, 4, 5]], dtype=torch.int32)
    result = _topk_ids_logical_to_physical_probability(topk_ids, info, log2phy_prob)

    assert result.shape == topk_ids.shape
    assert (result >= 0).all()
    assert (result < num_logical * 2).all()


def test_probability_dispatch_falls_back_when_probs_zero(dispatch_info_factory):
    """All-zero probabilities trigger the uniform fallback over valid physical IDs."""
    from sglang.srt.eplb.expert_location_dispatch import (
        _topk_ids_logical_to_physical_probability,
    )

    num_logical = 4
    info = dispatch_info_factory(num_logical, max_copies=2)

    log2phy_prob = torch.zeros(num_logical, 2, dtype=torch.float32)
    topk_ids = torch.tensor([[0, 1]], dtype=torch.int32)
    result = _topk_ids_logical_to_physical_probability(topk_ids, info, log2phy_prob)

    assert result.shape == topk_ids.shape
    assert (result >= 0).all()


# ---------------------------------------------------------------------------
# Global solver registry
# ---------------------------------------------------------------------------


def test_global_solver_registry_get_set_clear():
    """The per-layer global LPLB-solver registry behaves like a normal dict."""
    from sglang.srt.eplb.lplb_solver import (
        clear_global_lplb_solvers,
        get_global_lplb_solver,
        set_global_lplb_solver,
    )

    clear_global_lplb_solvers()
    assert get_global_lplb_solver(0) is None

    sentinel = object()
    set_global_lplb_solver(0, sentinel)
    assert get_global_lplb_solver(0) is sentinel
    assert get_global_lplb_solver(1) is None

    clear_global_lplb_solvers()
    assert get_global_lplb_solver(0) is None


# ---------------------------------------------------------------------------
# Model-architecture allowlist (P1.8)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "arch",
    [
        # Direct DeepSeek targets.
        "DeepseekV2ForCausalLM",
        "DeepseekV3ForCausalLM",
        "DeepseekV32ForCausalLM",
        # DeepSeek-derived: subclass DeepseekV2/V3 and reuse safe MoE paths.
        "MistralLarge3ForCausalLM",
        "MistralLarge3ForCausalLMEagle",
        "Glm4MoeLiteForCausalLM",
        "GlmMoeDsaForCausalLM",
    ],
)
def test_lplb_supports_deepseek_architectures(arch):
    """The DeepSeek family + its direct subclasses must be allowed — they all
    inherit the empty-rank `solver.solve()` participation from deepseek_v2.py."""
    from sglang.srt.eplb.lplb_solver import assert_lplb_supported_model

    # Should not raise.
    assert_lplb_supported_model(arch)


@pytest.mark.parametrize(
    "arch",
    [
        # The 10 unsafe MoE families enumerated by the cross-cutting sweep.
        "Qwen2MoeForCausalLM",
        "Qwen3MoeForCausalLM",
        "Glm4MoeForCausalLM",
        "BailingMoEForCausalLM",
        "ExaoneMoEForCausalLM",
        "Step3p5ForCausalLM",
        "MiMoV2FlashForCausalLM",
        "LLaDA2MoeModelLM",
        "MiniMaxM2ForCausalLM",
        "SDARMoeForCausalLM",
    ],
)
def test_lplb_rejects_unsafe_non_deepseek_moe(arch):
    """Every non-DeepSeek MoE model the sweep flagged must raise
    NotImplementedError before any LP solver is created. The error must name
    the architecture, point to the dispatch flag, and explain the deadlock
    cause so users know what's wrong."""
    from sglang.srt.eplb.lplb_solver import assert_lplb_supported_model

    with pytest.raises(NotImplementedError) as excinfo:
        assert_lplb_supported_model(arch)

    msg = str(excinfo.value)
    assert arch in msg, f"Error doesn't name the architecture: {msg}"
    assert "ep-dispatch-algorithm lp" in msg, f"Error doesn't reference the flag: {msg}"
    assert "all-reduce" in msg, f"Error doesn't explain the deadlock cause: {msg}"


# ---------------------------------------------------------------------------
# CUDA IPM solver — JIT-compiled kernel via load_jit
# ---------------------------------------------------------------------------


def _cuda_solver_or_skip():
    """Skip the calling test if the CUDA IPM solver can't run on this host.

    Skipped when: no CUDA, GPU SM < 9.0 (cuBLASDx requires Hopper+),
    cuda_solver module fails to import, or Math-DX (mathdx) headers are
    missing so load_jit can't compile.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    if torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("Requires SM >= 9.0 (Hopper+) for cuBLASDx/cuSolverDx")
    try:
        from sglang.jit_kernel.lplb import cuda_solver
    except ImportError as e:
        pytest.skip(f"cuda_solver import failed: {e}")
    try:
        from sglang.jit_kernel.utils import get_mathdx_root

        if get_mathdx_root() is None:
            pytest.skip(
                "Math-DX (cuBLASDx + cuSolverDx) not found. Set MATHDX_HOME or "
                "run python/sglang/jit_kernel/lplb/resources/download-mathdx.sh."
            )
    except Exception as e:
        pytest.skip(f"Math-DX probe failed: {e}")
    return cuda_solver


def _make_feasible_lp(nc: int, nv: int, seed: int = 42, device: str = "cuda"):
    """Build a small feasible LP whose Big-M variable is the last column.

    Same shape contract as the LPLB solver's actual call site:
        A: (NC, NV) float32, last column is the Big-M slack
        b: (NC,)   float32, set so x=1 is feasible
        c: (NV,)   float32, all zero except c[-2]=1, c[-1]=1000 (Big-M)
    """
    torch.manual_seed(seed)
    A = torch.randn(nc, nv, dtype=torch.float32, device=device)
    b = A @ torch.ones(nv, dtype=torch.float32, device=device)
    c = torch.zeros(nv, dtype=torch.float32, device=device)
    c[-2] = 1.0
    c[-1] = 1000.0
    return A, b, c


def test_cuda_solver_returns_correct_shape():
    """Kernel writes a finite (NV,) float32 vector for a feasible problem."""
    cuda_solver = _cuda_solver_or_skip()
    nc, nv = 4, 6
    A, b, c = _make_feasible_lp(nc, nv)
    cuda_solver.warmup(nc, nv)

    x = cuda_solver.solve_ipm(A, b, c, num_iters=5)

    assert x.shape == (nv,)
    assert x.dtype == torch.float32
    assert x.is_cuda
    assert torch.isfinite(x).all(), f"Non-finite values in CUDA solution: {x}"


def test_cuda_solver_matches_torch_oracle():
    """The JIT'd CUDA kernel agrees with the torch IPM reference on a
    feasible LP. Discriminator: tells "kernel ran the IPM" apart from
    "kernel silently emitted the 0.5 fallback for everything"."""
    cuda_solver = _cuda_solver_or_skip()
    from sglang.jit_kernel.lplb.torch_solver import _solve_ipm_torch

    nc, nv = 4, 6
    A, b, c = _make_feasible_lp(nc, nv)
    cuda_solver.warmup(nc, nv)

    x_cuda = cuda_solver.solve_ipm(A, b, c, num_iters=5)
    x_ref = _solve_ipm_torch(A, b, c, num_iters=5)

    # If both fell back to 0.5, that's a vacuous match — assert at least one
    # diverged from 0.5, then check they agree.
    not_fallback = (x_ref - 0.5).abs().max().item() > 1e-3
    assert not_fallback, (
        "Reference solver hit the 0.5 fallback; pick a different test problem"
    )
    diff = (x_cuda - x_ref).abs().max().item()
    assert diff < 5e-2, (
        f"CUDA kernel disagrees with torch oracle: max abs diff {diff:.4f}\n"
        f"  cuda: {x_cuda.cpu().tolist()}\n"
        f"  torch: {x_ref.cpu().tolist()}"
    )


def test_cuda_solver_compile_cache_is_per_shape():
    """Re-calling warmup for the same (NC, NV, num_iters) hits the cache and
    returns the same compiled module. Different shapes yield different ones.
    """
    cuda_solver = _cuda_solver_or_skip()

    mod1 = cuda_solver._ipm_module(4, 6, 256, 5, cuda_solver._sm_ver())
    mod2 = cuda_solver._ipm_module(4, 6, 256, 5, cuda_solver._sm_ver())
    mod3 = cuda_solver._ipm_module(8, 10, 256, 5, cuda_solver._sm_ver())

    assert mod1 is mod2, "cache_once should return the same module for the same shape"
    assert mod1 is not mod3, "different shapes must compile different modules"


def test_cuda_solver_fallback_on_nonconvergence():
    """When the IPM cannot converge (e.g. infeasible LP), the kernel writes
    0.5 to every entry rather than NaN/Inf."""
    cuda_solver = _cuda_solver_or_skip()

    nc, nv = 4, 6
    # Force infeasibility: b is far outside any plausible value of A @ x for x>=0
    A = torch.randn(nc, nv, dtype=torch.float32, device="cuda")
    b = torch.full((nc,), 1e6, dtype=torch.float32, device="cuda")
    c = torch.zeros(nv, dtype=torch.float32, device="cuda")
    c[-1] = 1000.0
    cuda_solver.warmup(nc, nv)

    x = cuda_solver.solve_ipm(A, b, c, num_iters=5)
    assert torch.isfinite(x).all()
    # Either the kernel converged to something or hit the fallback. If it hit
    # the fallback every entry should be exactly 0.5.
    if (x - 0.5).abs().max().item() < 1e-6:
        return  # explicitly the fallback path — pass
    # Otherwise it claimed convergence; values must still be plausibly bounded.
    assert x.abs().max().item() < 1e6, f"Kernel produced unbounded output: {x}"
