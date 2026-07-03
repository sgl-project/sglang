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
_global_lplb_solvers: dict[int, LPLBSolver] = {}


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
    return _global_lplb_solvers.get(layer_id)


def set_global_lplb_solver(layer_id: int, solver: LPLBSolver):
    _global_lplb_solvers[layer_id] = solver


def clear_global_lplb_solvers():
    _global_lplb_solvers.clear()


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

        # Separate single-copy vs replicated experts.
        # Stored as int64 so they can be used directly as index tensors in
        # _solve without per-call .long() casts (Tier 1 optimization).
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
        # without per-call .long() casts (Tier 1 optimization).
        self.log2phy = log2phy.to(torch.int64).contiguous()

        # Pre-JIT-compile the fused IPM kernel for this (NC, NV) shape so the
        # 20-40s compile cost happens once at startup rather than on the first
        # real request. No-op when the fused backend is unavailable.
        nc = self.A_base.shape[0]
        nv = self.A_base.shape[1] + 1  # +1 for Big-M column added in solve()
        from sglang.jit_kernel.lplb.torch_solver import warmup as _ipm_warmup

        _ipm_warmup(nc, nv, num_iters=5, device=device)

        # Pre-compute A_base row sum (used in every prep call).
        self._A_base_row_sum = self.A_base.sum(dim=1).contiguous()  # (NC,)

        # Pre-allocate the buffers the JIT CUDA prep / IPM / post kernels write
        # into. All writes are contiguous full-tensor stores (no strided
        # ``out=`` semantics), so the reuse is safe under high concurrency.
        # Constructed lazily on the first solve() call (we don't know the
        # device-side log2phy_prob shape until then) — see _solve.
        self._A_full = torch.empty(nc, nv, dtype=torch.float32, device=device)
        self._A_full[:, : nv - 1].copy_(self.A_base)
        self._b = torch.empty(nc, dtype=torch.float32, device=device)
        self._t1 = torch.empty(self.num_single, dtype=torch.float32, device=device)
        self._x = torch.empty(nv, dtype=torch.float32, device=device)
        self._log2phy_prob = torch.empty(
            log2phy.shape, dtype=torch.float32, device=device
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
        # Boolean masking + numel() (the previous defensive form) forced a
        # GPU->host sync on every forward pass via aten::nonzero and a
        # tensor-shape read; scatter_add on the flattened tensor is async
        # and a no-op when topk_ids is empty (DP-attention idle rank case).
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

    def _solve(self, global_counts: torch.Tensor) -> torch.Tensor:
        """Three CUDA kernel launches replace ~14 torch ops.

        Pipeline (all writes go into pre-allocated buffers from __init__):
            prep_lp_inputs → solve_ipm → extract_log2phy_prob
        Raises if the JIT CUDA backend is unavailable.
        """
        from sglang.jit_kernel.lplb import cuda_solver

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
