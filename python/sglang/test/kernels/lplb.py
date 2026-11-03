"""Pure-torch references for the LPLB CUDA kernels (testing only)."""

import torch


def solve_ipm_torch_reference(
    A: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    num_iters: int = 5,
) -> torch.Tensor:
    """Pure-torch reference for the fused IPM kernel — testing only.

    Mirrors the barrier-method iteration in ``csrc/lplb/ipm.cuh``
    step-for-step so the two can be compared numerically:

      x <- 1
      for _ in range(num_iters):
          ax2  = A * x^2            # (NC, NV)
          ax2a = ax2 @ A^T          # (NC, NC) KKT matrix
          delta = solve(ax2a, ax2 @ c)
          r     = delta^T @ A       # (NV,)
          d     = x * (c - r)
          alpha = 0.999 / d_max  (or 1.0 if d_max <= 1e-9)
          x    *= 1 - alpha * d
      write 0.5 everywhere on non-convergence.

    NOT bit-equivalent to the kernel: the kernel factors the KKT system
    with a hand-written block Cholesky while this uses
    ``torch.linalg.solve`` (LU). The two agree to a small tolerance
    (the numerical difference being the whole point of the comparison
    test). This function is never on the production path — the fused
    kernel is the only LP solver at runtime.
    """
    nc, nv = A.shape
    assert b.shape == (nc,), f"b shape mismatch: {b.shape} vs ({nc},)"
    assert c.shape == (nv,), f"c shape mismatch: {c.shape} vs ({nv},)"

    x = torch.ones(nv, device=A.device, dtype=torch.float32)
    d_max = torch.tensor(0.0, device=A.device, dtype=torch.float32)
    for _ in range(num_iters):
        ax2 = A * (x * x).unsqueeze(0)  # (NC, NV)
        ax2a = ax2 @ A.t()  # (NC, NC)
        ax2c = ax2 @ c  # (NC,)
        # Match the kernel's 1e-12 pivot clamp via a tiny diagonal jitter so
        # a (near-)singular KKT system stays solvable instead of raising.
        ax2a = ax2a + 1e-12 * torch.eye(nc, device=A.device, dtype=torch.float32)
        delta = torch.linalg.solve(ax2a, ax2c)  # (NC,)
        r = A.t() @ delta  # (NV,)
        d = x * (c - r)  # (NV,)
        d_max = d.max()
        alpha = 0.999 / d_max if d_max > 1e-9 else torch.tensor(1.0, device=A.device)
        x = x * (1.0 - alpha * d)

    max_residual = (A @ x - b).abs().max()
    converged = (d_max < 0.1) and (0 <= x[-1] < 1e-4) and (max_residual < 0.05)
    if not converged:
        return torch.full((nv,), 0.5, device=A.device, dtype=torch.float32)
    return x


def dispatch_probability_torch_reference(
    topk_ids: torch.Tensor,
    log2phy_prob: torch.Tensor,
    log2phy_map: torch.Tensor,
    random_vals: torch.Tensor,
) -> torch.Tensor:
    """Pure-torch reference of :func:`dispatch_probability`.

    Mirrors the CUDA kernel's algorithm exactly (inverse-CDF via cumsum
    + threshold) so the two paths are bit-equivalent for identical
    ``random_vals``, modulo floating-point rounding in the cumsum. Kept
    for numerical comparison and testing — not on the production hot
    path (it allocates and runs ~8 torch ops; the fused kernel collapses
    them into one launch).

    Algorithm (matches ``csrc/lplb/dispatch_probability.cuh``):

      1. Gather the per-row probability vector and physical-id map for
         each logical id in ``topk_ids``.
      2. If the row sum is zero (LP gave no signal), fall back to
         uniform over valid replicas (``log2phy_map != -1``).
      3. Sample: smallest ``c`` such that ``cumsum[0..c] > u * row_sum``,
         where ``u = random_vals[i]``. Ties favor advancing ``c``,
         matching the CUDA kernel.
      4. Return ``log2phy_map[logical_id, c]``.

    Args:
        topk_ids: (num_tokens, topk) int, on CUDA or CPU. Logical expert ids.
        log2phy_prob: (num_logical, max_copies) float32. LP solver output.
        log2phy_map: (num_logical, max_copies) int. -1 = unused replica.
        random_vals: (N,) float32, where N = ``topk_ids.numel()``. Uniform
            samples in [0, 1) — same shape and semantics as the CUDA kernel.

    Returns:
        Physical topk ids tensor with the same shape and dtype as ``topk_ids``.
    """
    original_shape = topk_ids.shape
    flat_ids = topk_ids.reshape(-1).long()
    n = flat_ids.shape[0]
    num_logical, max_copies = log2phy_prob.shape
    assert log2phy_map.shape == (num_logical, max_copies)
    assert random_vals.shape == (n,), (
        f"random_vals must be shape ({n},), got {tuple(random_vals.shape)}"
    )

    # Gather per-row probabilities and physical maps.
    probs = log2phy_prob[flat_ids]  # (N, max_copies), float32
    maps = log2phy_map[flat_ids]  # (N, max_copies), same int dtype as input

    # Fallback when row_sum == 0: uniform over valid replicas.
    row_sum = probs.sum(dim=-1, keepdim=True)  # (N, 1)
    fallback_probs = (maps >= 0).to(probs.dtype)  # (N, max_copies)
    probs = torch.where(row_sum > 0, probs, fallback_probs)
    row_sum = probs.sum(dim=-1)  # (N,)

    # Inverse-CDF sample: smallest c such that cumsum[..c] > u.
    # ``(cum <= u).sum(dim=-1)`` counts how many slots are still below u,
    # which equals the CUDA kernel's ``chosen`` after its for-loop.
    u = (random_vals * row_sum).unsqueeze(-1)  # (N, 1)
    cum = probs.cumsum(dim=-1)  # (N, max_copies)
    chosen = (cum <= u).sum(dim=-1).clamp(max=max_copies - 1)  # (N,)

    out = maps.gather(1, chosen.unsqueeze(-1)).squeeze(-1)
    return out.view(original_shape).to(topk_ids.dtype)
