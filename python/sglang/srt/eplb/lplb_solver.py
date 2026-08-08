"""
LPLBSolver — Linear-Programming Load Balancer for Expert Parallelism.

Encapsulates LP matrix construction (offline, at init/rebalance) and
per-batch solving (online, per MoE layer forward pass).

Design for DP-attention:
    Each EP rank counts its local tokens, then all ranks participate in an
    all-reduce to obtain identical global counts.  Every rank then solves
    the same LP independently, producing the same log2phy_prob — no
    broadcast is needed.  Empty-token ranks contribute zeros in the
    all-reduce so the collective never deadlocks.

Usage:
    solver = LPLBSolver(phy2log, log2phy, num_gpus, ep_group)
    log2phy_prob = solver.solve(topk_ids)  # per batch
"""

from __future__ import annotations

import logging
from typing import Optional

import torch

logger = logging.getLogger(__name__)

# Global per-layer LPLB solvers


# LP dispatch requires every EP rank to call solver.solve() on every forward
# pass (including empty-topk ranks under DP-attention) — the all-reduce inside
# would otherwise hang. Only the DeepSeek-v2 family and its subclasses route
# empty-rank paths through solver.solve(); other MoE families would deadlock.
_LPLB_SUPPORTED_MODEL_ARCHS: frozenset[str] = frozenset(
    {
        "DeepseekV2ForCausalLM",
        "DeepseekV3ForCausalLM",
        "DeepseekV32ForCausalLM",
        "MistralLarge3ForCausalLM",
        "MistralLarge3ForCausalLMEagle",
        "Glm4MoeLiteForCausalLM",
        "GlmMoeDsaForCausalLM",
    }
)


def assert_lplb_supported_model(architecture: str) -> None:
    if architecture not in _LPLB_SUPPORTED_MODEL_ARCHS:
        supported = ", ".join(sorted(_LPLB_SUPPORTED_MODEL_ARCHS))
        raise NotImplementedError(
            f"{architecture} does not support --ep-dispatch-algorithm lp. "
            f"Validated targets: {supported}. Other MoE families have "
            "empty-token early returns that don't participate in the EP "
            "all-reduce inside LPLBSolver.solve(), which would deadlock "
            "under DP-attention."
        )


def get_global_lplb_solver(layer_id: int) -> Optional[LPLBSolver]:
    from sglang.srt.runtime_context import get_resources

    return get_resources().lplb_solvers.get(layer_id)


def set_global_lplb_solver(layer_id: int, solver: LPLBSolver):
    from sglang.srt.runtime_context import get_resources

    get_resources().lplb_solvers[layer_id] = solver


def clear_global_lplb_solvers():
    from sglang.srt.runtime_context import get_resources

    get_resources().lplb_solvers.clear()


class LPLBSolver:
    """
    Per-layer LPLB solver.

    At init: pre-computes LP constraint matrices from expert-to-GPU mapping.
    At solve: takes topk_ids, counts tokens, all-reduces, runs LP,
              returns log2phy_prob for probability-based token dispatch.
    """

    def __init__(
        self,
        phy2log: torch.Tensor,
        log2phy: torch.Tensor,
        num_gpus: int,
        ep_group=None,
        logical_to_all_physical_map_num_valid=None,
    ):
        """
        Args:
            phy2log: (num_physical_experts,) physical-to-logical expert mapping.
            log2phy: (num_logical_experts, max_copies) logical-to-physical mapping (-1 padded).
            num_gpus: Number of GPUs in the EP group.
            ep_group: GroupCoordinator for EP communication (all-reduce).
            logical_to_all_physical_map_num_valid: (num_logical_experts,) number of valid physical copies.
        """
        device = phy2log.device
        self.num_gpus = num_gpus
        self.ep_group = ep_group
        self._has_redundancy = False
        if logical_to_all_physical_map_num_valid is not None:
            self._has_redundancy = bool(
                (logical_to_all_physical_map_num_valid > 1).any()
            )

        self.num_logical = log2phy.shape[0]
        self.max_copies = log2phy.shape[1]
        self.num_phy = phy2log.shape[0]
        # B1/B2 GPU-assignment matrices below assume each rank owns a
        # contiguous block of num_phy // num_gpus physical experts.
        if self.num_phy % num_gpus != 0:
            raise ValueError(
                f"LPLBSolver requires num_phy ({self.num_phy}) to be divisible "
                f"by num_gpus ({num_gpus}); per-rank-contiguous ownership is "
                "currently the only supported allocation."
            )
        num_phy_per_gpu = self.num_phy // num_gpus

        # Count copies per logical expert
        logcnt = torch.bincount(phy2log, minlength=self.num_logical)

        # Separate single-copy vs replicated experts. Stored as int64 so they
        # can be used directly as index tensors in _solve without per-call
        # .long() casts.
        self.log_single = torch.nonzero(logcnt == 1).flatten().to(torch.int64)
        self.phy_single = log2phy[self.log_single, 0].to(torch.int64)
        self.log_replicated = torch.nonzero(logcnt > 1).flatten().to(torch.int64)
        self.phy_replicated = (
            torch.nonzero(logcnt[phy2log] > 1).flatten().to(torch.int64)
        )

        self.num_single = len(self.log_single)
        self.num_red_log = len(self.log_replicated)
        self.num_red_phy = len(self.phy_replicated)

        # Build GPU assignment matrices
        B_full = torch.zeros(
            (num_gpus, self.num_phy), dtype=torch.float32, device=device
        )
        for i in range(num_gpus):
            B_full[i, i * num_phy_per_gpu : (i + 1) * num_phy_per_gpu] = 1
        self.B1 = B_full[:, self.phy_single].contiguous()
        B2 = B_full[:, self.phy_replicated]

        # Build C matrix (copy-to-logical mapping)
        C = torch.zeros(
            (self.num_red_log, self.num_red_phy), dtype=torch.float32, device=device
        )
        phy2log_rep = phy2log[self.phy_replicated]
        for i in range(self.num_red_log):
            C[i, phy2log_rep == self.log_replicated[i]] = 1.0

        # Build A_base = [[C, 0, 0], [B2, I, -1]]  (without Big-M column)
        zeros_top_g = torch.zeros(
            (self.num_red_log, num_gpus), dtype=torch.float32, device=device
        )
        zeros_top_1 = torch.zeros(
            (self.num_red_log, 1), dtype=torch.float32, device=device
        )
        I_g = torch.eye(num_gpus, dtype=torch.float32, device=device)
        neg_ones = torch.full((num_gpus, 1), -1.0, dtype=torch.float32, device=device)

        A_top = torch.hstack([C, zeros_top_g, zeros_top_1])
        A_bottom = torch.hstack([B2, I_g, neg_ones])
        self.A_base = torch.vstack([A_top, A_bottom]).contiguous()

        # Objective: minimize M (second-to-last var), penalize Big-M auxiliary
        nv = self.A_base.shape[1] + 1  # +1 for Big-M column
        self.c_vec = torch.zeros(nv, dtype=torch.float32, device=device)
        self.c_vec[-2] = 1.0
        self.c_vec[-1] = 1000.0

        # Store log2phy as int64 so it can be used directly as index tensor
        # without per-call .long() casts.
        self.log2phy = log2phy.to(torch.int64).contiguous()

        # Pick the LP solve backend once, at init:
        #   _use_fused -> NVIDIA cuBLASDx fused kernel (prep/IPM/post in CUDA
        #                 C++). Per-step, CUDA-graph-capturable. Needs a Hopper+
        #                 GPU with Math-DX headers.
        #   _hip_fused -> ROCm HIP block-Cholesky IPM kernel (torch prep/post +
        #                 the hand-written HIP kernel for the solve). Per-step,
        #                 CUDA-graph-capturable (no rocSOLVER workspace allocs).
        #   neither    -> pure-torch per-step solver (_solve_torch). Correct on
        #                 any backend but not graph-capturable on ROCm, so it
        #                 requires --disable-cuda-graph there.
        from sglang.kernels.ops.lplb.torch_solver import fused_backend_available

        self._use_fused = fused_backend_available()
        self._hip_fused = False

        # Pre-JIT-compile the IPM kernel for this (NC, NV) shape so the 20-40s
        # compile cost happens once at startup rather than on the first request.
        nc = self.A_base.shape[0]
        nv = self.A_base.shape[1] + 1  # +1 for Big-M column added in solve()
        _is_hip = getattr(torch.version, "hip", None) is not None
        if self._use_fused:
            from sglang.kernels.ops.lplb.torch_solver import warmup as _ipm_warmup

            _ipm_warmup(nc, nv, num_iters=5, device=device)
        elif _is_hip:
            # ROCm has no cuBLASDx, so use the hand-written HIP block-Cholesky
            # IPM kernel: it is CUDA-graph-capturable (allocates no workspace
            # mid-launch, unlike rocSOLVER) and fast (~80us/solve vs ~28ms for
            # torch.linalg.cholesky_ex on these tiny matrices). If it can't build
            # (e.g. no hipcc toolchain) fall back to the pure-torch per-step
            # solver, which is correct but requires --disable-cuda-graph.
            try:
                from sglang.kernels.ops.lplb import cuda_solver

                cuda_solver.warmup(nc, nv, num_iters=5, device=str(device))
                self._hip_fused = True
                logger.info(
                    "LPLBSolver: HIP block-Cholesky IPM kernel enabled "
                    "(per-step, CUDA-graph-capturable)."
                )
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "LPLBSolver: HIP IPM kernel unavailable (%s); using the "
                    "pure-torch per-step LP solver (requires --disable-cuda-graph).",
                    e,
                )
        else:
            logger.info(
                "LPLBSolver: fused IPM backend unavailable (non-Hopper GPU); "
                "using the pure-torch per-step LP solver."
            )

        # Pre-compute A_base row sum (used in every prep call).
        self._A_base_row_sum = self.A_base.sum(dim=1).contiguous()  # (NC,)

        # Pre-allocate the buffers the fused / HIP prep / IPM / post steps write
        # into. All writes are contiguous full-tensor stores (no strided ``out=``
        # semantics), so reusing them across forwards is safe under concurrency
        # and avoids per-forward allocations that would otherwise be retained in
        # each captured decode graph's memory pool and OOM the GPU. The pure-torch
        # fallback (_solve_torch) allocates its own scratch instead.
        self._A_full = torch.empty(nc, nv, dtype=torch.float32, device=device)
        self._A_full[:, : nv - 1].copy_(self.A_base)
        self._b = torch.empty(nc, dtype=torch.float32, device=device)
        self._t1 = torch.empty(self.num_single, dtype=torch.float32, device=device)
        self._x = torch.empty(nv, dtype=torch.float32, device=device)
        # Persistent log2phy_prob buffer, overwritten in place by each per-step
        # fused / HIP solve.
        self._log2phy_prob = torch.zeros(
            log2phy.shape, dtype=torch.float32, device=device
        )
        # Scratch for the in-place HIP per-step post step. HIP path only; the
        # NV fused and pure-torch paths never read it, so it is not allocated
        # there (keeps the CUDA path's footprint identical to upstream).
        self._phy_prob = (
            torch.zeros(
                self.num_single + self.num_red_phy + 1,
                dtype=torch.float32,
                device=device,
            )
            if self._hip_fused
            else None
        )

    def solve(self, topk_ids: torch.Tensor) -> torch.Tensor:
        """
        Full LPLB pipeline: count -> all-reduce -> LP solve -> return log2phy_prob.

        All EP ranks must call this method every MoE layer forward pass,
        including empty-token ranks (which pass an empty topk_ids tensor).
        This ensures the all-reduce collective does not deadlock under
        DP-attention where different ranks may have different token counts.

        Args:
            topk_ids: (num_tokens, topk) int32 tensor of logical expert IDs.
                      Can be empty (shape (0, topk)) for idle ranks.

        Returns:
            log2phy_prob: (num_logical, max_copies) float32 probability tensor.
        """
        device = topk_ids.device

        # Step 1: Count local tokens per logical expert.
        # topk_ids comes from the router and is by construction in
        # [0, num_logical), so we can scatter_add directly without filtering.
        # scatter_add on the flattened tensor is async and a no-op when topk_ids
        # is empty (DP-attention idle rank case); a boolean-mask + numel() form
        # would instead force a GPU->host sync on every forward pass.
        local_counts = torch.zeros(self.num_logical, dtype=torch.int32, device=device)
        flat_ids = topk_ids.flatten()
        local_counts.scatter_add_(
            0,
            flat_ids.long(),
            torch.ones_like(flat_ids, dtype=torch.int32),
        )

        # Step 2: All-reduce to get global counts across all EP ranks.
        # All EP ranks must participate — empty-token ranks contribute zeros.
        # After all-reduce, every rank has identical global_counts and solves
        # the same LP independently, so no broadcast is needed.
        # GroupCoordinator.all_reduce may be in-place (pynccl) or out-of-place
        # (ca_comm / pymscclpp / ...) depending on tensor size; small tensors
        # like ours (~num_logical * 4 B) typically take the out-of-place path,
        # so we must capture the return value.
        global_counts = local_counts.float()
        if self.ep_group is not None:
            global_counts = self.ep_group.all_reduce(global_counts)

        # Step 3: Run LP solver
        return self._solve(global_counts)

    def _solve_hip(self, global_counts: torch.Tensor) -> torch.Tensor:
        """ROCm per-step solve: torch prep/post around the HIP IPM kernel.

        prep and post are cheap capturable torch ops (lp_prep.cuh uses warp
        shuffles that assume a 32-wide warp, so it is not reused on HIP); only
        the IPM solve — the part that would otherwise need rocSOLVER — runs in
        the hand-written HIP block-Cholesky kernel. All three are
        CUDA-graph-capturable, so this runs inside the decode graph every step.

        Writes into pre-allocated static buffers (``_A_full``/``_b``/``_x``/
        ``_phy_prob``/``_log2phy_prob``) instead of allocating per forward: with
        the decode graph captured across ~50 batch sizes, per-step allocations
        would be retained in each graph's private memory pool and OOM the GPU.
        """
        from sglang.kernels.ops.lplb import cuda_solver

        # prep in place: only the Big-M last column of A and the RHS b depend on
        # the counts; A's first NV-1 columns hold the constant A_base from init.
        total = global_counts.sum().clamp(min=1.0)
        counts_norm = global_counts / total
        t1 = counts_norm[self.log_single]  # (num_single,)
        b1 = counts_norm[self.log_replicated]  # (num_red_log,)
        b2 = -(self.B1 @ t1).flatten()  # (num_gpus,)
        torch.cat([b1, b2], out=self._b)  # (nc,) -> static buffer
        self._A_full[:, -1] = self._b - self._A_base_row_sum  # Big-M column

        x = cuda_solver.solve_ipm(
            self._A_full, self._b, self.c_vec, num_iters=5, result=self._x
        )

        # post in place -> static _log2phy_prob.
        self._phy_prob.zero_()
        self._phy_prob[self.phy_replicated] = x[: self.num_red_phy].clamp(min=0)
        self._phy_prob[self.phy_single] = t1
        self._log2phy_prob.copy_(torch.take(self._phy_prob, self.log2phy))
        return self._log2phy_prob

    def _solve(self, global_counts: torch.Tensor) -> torch.Tensor:
        """Three CUDA kernel launches replace ~14 torch ops.

        Pipeline (all writes go into pre-allocated buffers from __init__):
            prep_lp_inputs → solve_ipm → extract_log2phy_prob
        Falls back to the HIP per-step path or the pure-torch pipeline when the
        fused cuBLASDx backend is unavailable (ROCm/HIP or non-Hopper GPU).
        """
        if self._hip_fused:
            return self._solve_hip(global_counts)
        if not self._use_fused:
            return self._solve_torch(global_counts)

        from sglang.kernels.ops.lplb import cuda_solver

        cuda_solver.prep_lp_inputs(
            self._A_full,
            self._b,
            self._t1,
            global_counts,
            self.log_single,
            self.log_replicated,
            self.B1,
            self._A_base_row_sum,
        )
        cuda_solver.solve_ipm(self._A_full, self._b, self.c_vec, result=self._x)
        cuda_solver.extract_log2phy_prob(
            self._log2phy_prob,
            self._x,
            self._t1,
            self.phy_single,
            self.phy_replicated,
            self.log2phy,
        )
        return self._log2phy_prob

    def _prep_torch(self, global_counts: torch.Tensor):
        """Build the IPM inputs (A, b) and the single-copy ratios t1 from the
        global token counts (mirrors lp_prep.cuh, ~8 torch ops). Cheap
        elementwise work with no host sync."""
        total = global_counts.sum().clamp(min=1.0)
        counts_norm = global_counts / total
        t1 = counts_norm[self.log_single]  # (num_single,)
        b1 = counts_norm[self.log_replicated]  # (num_red_log,)
        b2 = -(self.B1 @ t1).flatten()  # (num_gpus,)
        b = torch.cat([b1, b2])  # (nc,)
        # A = [A_base | Big-M column], where the Big-M column = b - A_base_row_sum.
        A = torch.cat(
            [self.A_base, (b - self._A_base_row_sum).unsqueeze(1)], dim=1
        )  # (nc, nv)
        return A, b, t1

    def _post_torch(self, x: torch.Tensor, t1: torch.Tensor) -> torch.Tensor:
        """Turn the IPM solution x into log2phy_prob (mirrors lp_post.cuh)."""
        device = x.device
        x_ratios = x[: self.num_red_phy].clamp(min=0)
        # phy_prob has a trailing always-zero "sink" slot; log2phy's -1 padding
        # is routed there via torch.take's negative-index wrap-around.
        phy_prob = torch.zeros(
            self.num_single + self.num_red_phy + 1, dtype=torch.float32, device=device
        )
        phy_prob[self.phy_replicated] = x_ratios
        phy_prob[self.phy_single] = t1
        return torch.take(phy_prob, self.log2phy)  # (num_logical, max_copies)

    def _solve_torch(self, global_counts: torch.Tensor) -> torch.Tensor:
        """Pure-torch per-step equivalent of the fused prep → IPM → post
        pipeline. Used when neither the fused cuBLASDx kernel nor the HIP kernel
        is available; correct on any backend but not CUDA-graph-capturable on
        ROCm (see _ipm_solve_robust), so it requires --disable-cuda-graph there.
        """
        A, b, t1 = self._prep_torch(global_counts)
        x = self._ipm_solve_robust(A, b, self.c_vec)  # (nv,)
        return self._post_torch(x, t1)

    def _ipm_solve_robust(
        self, A: torch.Tensor, b: torch.Tensor, c: torch.Tensor, num_iters: int = 5
    ) -> torch.Tensor:
        """Barrier-method IPM for ``min c^T x s.t. Ax=b, x>=0``.

        Solves the KKT system with a dense Cholesky factorization
        (``torch.linalg.cholesky_ex`` + ``torch.cholesky_solve``) of the
        1e-6-regularized SPD normal equations, mirroring the fused cuBLASDx
        kernel's block-Cholesky POSV. ``cholesky_ex`` (``check_errors=False``)
        returns the error ``info`` as a device tensor and never raises, so —
        unlike ``torch.linalg.solve``'s LU pivoting — it neither host-syncs nor
        aborts on a rank-deficient system (common with sparse per-batch expert
        counts); the 1e-6 diagonal keeps the normal equations SPD. The step size
        and the non-convergence fallback use ``torch.where`` (not a Python
        ``if``, which would force a ``.item()`` host sync).

        Matches the fused kernel's behavior: returns 0.5 everywhere on
        non-convergence, so the downstream dispatch spreads tokens uniformly
        over valid replicas.
        """
        nc, nv = A.shape
        device = A.device
        x = torch.ones(nv, device=device, dtype=torch.float32)
        eye = torch.eye(nc, device=device, dtype=torch.float32)
        one = torch.ones((), device=device, dtype=torch.float32)
        d_max = torch.zeros((), device=device, dtype=torch.float32)
        for _ in range(num_iters):
            ax2 = A * (x * x).unsqueeze(0)  # (nc, nv)
            ax2a = ax2 @ A.t() + 1e-6 * eye  # (nc, nc) SPD regularized KKT
            ax2c = ax2 @ c  # (nc,)
            L, _info = torch.linalg.cholesky_ex(ax2a)  # (nc, nc) lower factor
            delta = torch.cholesky_solve(ax2c.unsqueeze(1), L).squeeze(1)  # (nc,)
            r = A.t() @ delta  # (nv,)
            d = x * (c - r)  # (nv,)
            d_max = d.max()
            # alpha = 0.999 / d_max when d_max > 1e-9 else 1.0 — branch-free so
            # no host sync. The barrier step keeps 0.999 * d / d_max < 1, so x
            # stays strictly positive and diag(x^2) keeps the KKT SPD.
            alpha = torch.where(d_max > 1e-9, 0.999 / d_max.clamp(min=1e-9), one)
            x = x * (1.0 - alpha * d)

        # Device-side convergence test; pick x or the all-0.5 fallback via where
        # (no Python branch -> capturable).
        max_residual = (A @ x - b).abs().max()
        converged = (
            (d_max < 0.1) & (x[-1] >= 0) & (x[-1] < 1e-4) & (max_residual < 0.05)
        )
        half = torch.full((nv,), 0.5, device=device, dtype=torch.float32)
        return torch.where(converged, x, half)
