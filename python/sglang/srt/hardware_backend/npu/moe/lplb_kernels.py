"""Device-only kernels for NPU replica load balancing."""

import torch
import triton
import triton.language as tl

_COUNT_BLOCK_SIZE = 1024
_DISPATCH_BLOCK_SIZE = 32
_SMALL_DISPATCH_BLOCK_SIZE = 16
_SMALL_DISPATCH_MAX_ROUTES = 8192


@triton.jit
def _count_routes(
    IDs,
    ValidRows,
    Counts,
    N: tl.constexpr,
    K: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    E: tl.constexpr,
    HAS_VALID: tl.constexpr,
    BLOCK: tl.constexpr,
    BINS: tl.constexpr,
):
    x = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    rows = tl.load(ValidRows) if HAS_VALID else N
    ids = tl.load(IDs + (x // K) * S0 + (x % K) * S1, x < N * K, other=-1)
    valid = (x < N * K) & (x // K < rows) & (ids >= 0) & (ids < E)
    histogram = tl.histogram(tl.where(valid, ids, -1).to(tl.int32), BINS)
    bins = tl.arange(0, BINS)
    if N * K <= BLOCK:
        # One program owns every bin, including zero-count experts.
        tl.store(Counts + bins, histogram, bins < E)
    else:
        tl.atomic_add(Counts + bins, histogram, bins < E, sem="relaxed")


def count_routes(ids, valid_rows, counts):
    # Multi-program histograms need a cleared accumulator; a single program
    # overwrites it directly and avoids a separate device memset.
    if ids.numel() == 0 or ids.numel() > _COUNT_BLOCK_SIZE:
        counts.zero_()
    if ids.numel():
        _count_routes[(triton.cdiv(ids.numel(), _COUNT_BLOCK_SIZE),)](
            ids,
            valid_rows,
            counts,
            *ids.shape,
            *ids.stride(),
            counts.numel(),
            valid_rows is not None,
            _COUNT_BLOCK_SIZE,
            triton.next_power_of_2(counts.numel()),
        )


@triton.jit
def _prepare_lp(
    Counts,
    Base,
    Replicated,
    SingleRank,
    A,
    B,
    E: tl.constexpr,
    H: tl.constexpr,
    NC: tl.constexpr,
    NV: tl.constexpr,
    BE: tl.constexpr,
    BC: tl.constexpr,
    BV: tl.constexpr,
):
    e = tl.arange(0, BE)
    count = tl.load(Counts + e, e < E, 0).to(tl.float32)
    count /= tl.maximum(tl.sum(count, 0), 1.0)
    single_rank = tl.load(SingleRank + e, e < E, -1)
    r = tl.arange(0, BC)
    v = tl.arange(0, BV)
    replicated = tl.load(Replicated + r, r < H, -1)
    demand = tl.sum(tl.where(replicated[:, None] == e[None, :], count[None, :], 0.0), 1)
    fixed_load = tl.sum(
        tl.where(single_rank[None, :] == (r[:, None] - H), count[None, :], 0.0), 1
    )
    b = tl.where(r < H, demand, -fixed_load)
    b = tl.where(r < NC, b, 0.0)
    a = tl.load(
        Base + r[:, None] * NV + v[None, :], (r[:, None] < NC) & (v[None, :] < NV), 0.0
    )
    aux = b - tl.sum(a, 1)
    a = tl.where(v[None, :] == NV - 1, aux[:, None], a)
    tl.store(A + r[:, None] * NV + v[None, :], a, (r[:, None] < NC) & (v[None, :] < NV))
    tl.store(B + r, b, r < NC)


@triton.jit
def _solve_ipm_schur(
    A,
    B,
    X,
    Diagnostics,
    H: tl.constexpr,
    G: tl.constexpr,
    NV: tl.constexpr,
    BH: tl.constexpr,
    BG: tl.constexpr,
    BV: tl.constexpr,
    ITERATIONS: tl.constexpr,
):
    """Eliminate expert constraints before solving the smaller rank system.

    Each replica belongs to exactly one logical expert. Consequently, the
    expert block of A diag(x**2) A.T is diagonal except for the auxiliary
    variable's rank-one term. Its Schur complement leaves a G x G system;
    recover the expert dual variables after solving for the rank duals.

    Columns contain replica masses, rank slack, maximum load, and auxiliary.
    Five affine-scaling steps use the same feasible start as the CUDA solver.
    """
    e = tl.arange(0, BH)
    r = tl.arange(0, BG)
    v = tl.arange(0, BV)
    ae = tl.load(
        A + e[:, None] * NV + v[None, :], (e[:, None] < H) & (v[None, :] < NV), 0.0
    )
    ar = tl.load(
        A + (H + r[:, None]) * NV + v[None, :],
        (r[:, None] < G) & (v[None, :] < NV),
        0.0,
    )
    be = tl.load(B + e, e < H, 0.0)
    br = tl.load(B + H + r, r < G, 0.0)
    aux_e = tl.sum(tl.where(v[None, :] == NV - 1, ae, 0.0), 1)
    aux_r = tl.sum(tl.where(v[None, :] == NV - 1, ar, 0.0), 1)
    expert_base = tl.where(v[None, :] < NV - 2, ae, 0.0)
    rank_base = tl.where(v[None, :] < NV - 2, ar, 0.0)
    c = tl.where(v == NV - 1, 1000.0, tl.where(v == NV - 2, 1.0, 0.0))
    x = tl.where(v < NV, 1.0, 0.0)
    dmax = 0.0
    for _ in range(ITERATIONS):
        x2 = x * x
        diagonal_e = tl.sum(expert_base * x2[None, :], 1)
        inv_e = tl.where(diagonal_e > 0.0, 1.0 / tl.maximum(diagonal_e, 1.0e-20), 0.0)
        cross = tl.dot(
            expert_base * x2[None, :], tl.trans(rank_base), input_precision="ieee"
        )
        diagonal_r = tl.sum(rank_base * x2[None, :], 1)
        t2 = tl.sum(tl.where(v == NV - 2, x2, 0.0), 0)
        z2 = tl.sum(tl.where(v == NV - 1, x2, 0.0), 0)
        beta = z2 / (1.0 + z2 * tl.sum(aux_e * aux_e * inv_e, 0))
        w = aux_r - tl.sum(cross * (inv_e * aux_e)[:, None], 0)
        normal = -tl.dot(
            tl.trans(cross), cross * inv_e[:, None], input_precision="ieee"
        )
        normal += t2 + beta * w[:, None] * w[None, :]
        normal += tl.where(r[:, None] == r[None, :], diagonal_r[:, None], 0.0)
        real = r < G
        normal = tl.where(real[:, None] & real[None, :], normal, 0.0)
        normal += tl.where((r[:, None] == r[None, :]) & ~real[:, None], 1.0, 0.0)
        rhs = tl.where(real, -t2 + 1000.0 * beta * w, 0.0)
        for k in range(G):
            column = tl.sum(tl.where(r[None, :] == k, normal, 0.0), 1)
            pivot = tl.maximum(tl.sum(tl.where(r == k, column, 0.0), 0), 1.0e-12)
            row = tl.sum(tl.where(r[:, None] == k, normal, 0.0), 0) / pivot
            value = tl.sum(tl.where(r == k, rhs, 0.0), 0) / pivot
            normal = tl.where(
                r[:, None] == k, row[None, :], normal - column[:, None] * row[None, :]
            )
            rhs = tl.where(r == k, value, rhs - column * value)
        dual_r = rhs
        auxiliary_scale = beta * (1000.0 - tl.sum(w * dual_r, 0))
        dual_e = inv_e * (aux_e * auxiliary_scale - tl.sum(cross * dual_r[None, :], 1))
        reduced_cost = (
            c - tl.sum(ae * dual_e[:, None], 0) - tl.sum(ar * dual_r[:, None], 0)
        )
        direction = x * reduced_cost
        dmax = tl.maximum(tl.max(direction, 0), 0.0)
        alpha = tl.where(dmax > 1.0e-9, 0.999 / dmax, 1.0)
        x *= 1.0 - alpha * direction
    residual = tl.maximum(
        tl.max(tl.abs(tl.sum(ae * x[None, :], 1) - be), 0),
        tl.max(tl.abs(tl.sum(ar * x[None, :], 1) - br), 0),
    )
    tl.store(X + v, x, v < NV)
    tl.store(Diagnostics, residual)
    tl.store(Diagnostics + 1, tl.sum(tl.where(v == NV - 1, x, 0.0), 0))
    tl.store(Diagnostics + 2, dmax)
    # Normalize each expert's replica masses exactly as dispatch will. Finite
    # LP iterations can satisfy constraints yet balance worse than uniform.
    positive_x = tl.maximum(x, 0.0)
    expert_mass = tl.sum(expert_base * positive_x[None, :], 1)
    copies = tl.sum(expert_base, 1)
    scale = be / tl.maximum(expert_mass, 1.0e-20)
    uniform_scale = be / tl.maximum(copies, 1.0)
    candidate = tl.sum(
        expert_base
        * tl.where(
            expert_mass[:, None] > 0.0,
            positive_x[None, :] * scale[:, None],
            uniform_scale[:, None],
        ),
        0,
    )
    uniform = tl.sum(expert_base * uniform_scale[:, None], 0)
    candidate_load = tl.sum(rank_base * candidate[None, :], 1) - br
    uniform_load = tl.sum(rank_base * uniform[None, :], 1) - br
    improvement = tl.max(uniform_load, 0) - tl.max(candidate_load, 0)
    tl.store(Diagnostics + 3, improvement)


def solve_ipm(base, replicated, single_rank, counts, a, b, x, diagnostics, iterations):
    nc, nv = a.shape
    bc, bv = max(16, triton.next_power_of_2(nc)), max(16, triton.next_power_of_2(nv))
    _prepare_lp[(1,)](
        counts,
        base,
        replicated,
        single_rank,
        a,
        b,
        counts.numel(),
        replicated.numel(),
        nc,
        nv,
        triton.next_power_of_2(counts.numel()),
        bc,
        bv,
    )
    h = replicated.numel()
    g = nc - h
    _solve_ipm_schur[(1,)](
        a,
        b,
        x,
        diagnostics,
        h,
        g,
        nv,
        max(16, triton.next_power_of_2(h)),
        max(16, triton.next_power_of_2(g)),
        bv,
        iterations,
    )


@triton.jit
def _extract_probabilities(
    Map,
    ReplicaIndex,
    X,
    Diagnostics,
    Prob,
    Fallbacks,
    E: tl.constexpr,
    C: tl.constexpr,
    BC: tl.constexpr,
):
    expert = tl.program_id(0)
    copy = tl.arange(0, BC)
    slot = tl.load(Map + expert * C + copy, copy < C, -1)
    index = tl.load(ReplicaIndex + slot, slot >= 0, -1)
    mass = tl.load(X + index, (slot >= 0) & (index >= 0), 0.0)
    mass = tl.where(index < 0, 1.0, tl.maximum(mass, 0.0))
    residual = tl.load(Diagnostics)
    auxiliary = tl.load(Diagnostics + 1)
    dmax = tl.load(Diagnostics + 2)
    improvement = tl.load(Diagnostics + 3)
    usable = (
        (residual < 1.0e-3)
        & (auxiliary >= 0.0)
        & (auxiliary < 1.0e-4)
        & (dmax < 0.1)
        & (improvement >= 0.0)
    )
    finite = (
        tl.sum(((mass != mass) | (tl.abs(mass) == float("inf"))).to(tl.int32), 0) == 0
    )
    mass = tl.where((slot >= 0) & usable & finite, mass, 0.0)
    total = tl.sum(mass, 0)
    mass = tl.where(total > 0.0, mass, (slot >= 0).to(tl.float32))
    probability = mass / tl.maximum(tl.sum(mass, 0), 1.0e-20)
    tl.store(Prob + expert * C + copy, probability, copy < C)
    if expert == 0:
        tl.atomic_add(Fallbacks, (~usable).to(tl.int32), sem="relaxed")


def extract_probabilities(
    mapping, replica_index, x, diagnostics, probabilities, fallbacks
):
    e, c = mapping.shape
    _extract_probabilities[(e,)](
        mapping,
        replica_index,
        x,
        diagnostics,
        probabilities,
        fallbacks,
        e,
        c,
        triton.next_power_of_2(c),
    )


@triton.jit
def _dispatch_probability(
    IDs,
    Map,
    Prob,
    Random,
    Out,
    N: tl.constexpr,
    K: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    MAP_STRIDE: tl.constexpr,
    C: tl.constexpr,
    BLOCK: tl.constexpr,
    COPIES: tl.constexpr,
):
    route = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    expert = tl.load(IDs + (route // K) * S0 + (route % K) * S1, route < N * K, 0)
    expert = tl.maximum(expert, 0)
    u = tl.load(Random + route, route < N * K, 0.0)
    copy = tl.arange(0, COPIES)
    mask = (route[:, None] < N * K) & (copy[None, :] < C)
    slot = tl.load(Map + expert[:, None] * MAP_STRIDE + copy[None, :], mask, -1).to(
        tl.int32
    )
    probability = tl.load(Prob + expert[:, None] * C + copy[None, :], mask, 0.0)
    cumulative = tl.cumsum(probability, 1)
    chosen = tl.min(
        tl.where((slot >= 0) & (u[:, None] < cumulative), copy[None, :], C), 1
    )
    last = tl.max(tl.where((slot >= 0) & (probability > 0), copy[None, :], -1), 1)
    chosen = tl.where(chosen < C, chosen, last)
    physical = tl.sum(tl.where(copy[None, :] == chosen[:, None], slot, 0), 1)
    # Rounding at the CDF's upper edge must never select a padded -1 slot.
    tl.store(Out + route, physical, route < N * K)


def dispatch_probability(ids, probabilities, mapping):
    output = torch.empty_like(ids, memory_format=torch.contiguous_format)
    if ids.numel():
        # Smaller CDF tiles reduce small-batch latency; larger batches benefit
        # from amortizing each program's setup over more routes.
        block = (
            _SMALL_DISPATCH_BLOCK_SIZE
            if ids.numel() <= _SMALL_DISPATCH_MAX_ROUTES
            else _DISPATCH_BLOCK_SIZE
        )
        random = torch.rand(ids.numel(), device=ids.device)
        _dispatch_probability[(triton.cdiv(ids.numel(), block),)](
            ids,
            mapping,
            probabilities,
            random,
            output,
            *ids.shape,
            *ids.stride(),
            mapping.stride(0),
            probabilities.shape[1],
            block,
            triton.next_power_of_2(probabilities.shape[1]),
        )
    return output
