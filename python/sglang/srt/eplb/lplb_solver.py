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
_global_lplb_solvers: dict[int, "LPLBSolver"] = {}


def get_global_lplb_solver(layer_id: int) -> Optional["LPLBSolver"]:
    return _global_lplb_solvers.get(layer_id)


def set_global_lplb_solver(layer_id: int, solver: "LPLBSolver"):
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
        num_phy_per_gpu = self.num_phy // num_gpus

        # Count copies per logical expert
        logcnt = torch.bincount(phy2log, minlength=self.num_logical)

        # Separate single-copy vs replicated experts
        self.log_single = torch.nonzero(logcnt == 1).flatten().to(torch.int32)
        self.phy_single = log2phy[self.log_single.long(), 0].to(torch.int32)
        self.log_replicated = torch.nonzero(logcnt > 1).flatten().to(torch.int32)
        self.phy_replicated = (
            torch.nonzero(logcnt[phy2log] > 1).flatten().to(torch.int32)
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
        self.B1 = B_full[:, self.phy_single.long()].contiguous()
        B2 = B_full[:, self.phy_replicated.long()]

        # Build C matrix (copy-to-logical mapping)
        C = torch.zeros(
            (self.num_red_log, self.num_red_phy), dtype=torch.float32, device=device
        )
        phy2log_rep = phy2log[self.phy_replicated.long()]
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

        # Store log2phy as int32 for kernel
        self.log2phy = log2phy.to(torch.int32).contiguous()

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

        # Step 1: Count local tokens per logical expert
        local_counts = torch.zeros(self.num_logical, dtype=torch.int32, device=device)
        flat_ids = topk_ids.flatten()
        valid_mask = (flat_ids >= 0) & (flat_ids < self.num_logical)
        valid_ids = flat_ids[valid_mask]
        if valid_ids.numel() > 0:
            local_counts.scatter_add_(
                0,
                valid_ids.long(),
                torch.ones(valid_ids.shape[0], dtype=torch.int32, device=device),
            )

        # Step 2: All-reduce to get global counts across all EP ranks.
        # All EP ranks must participate — empty-token ranks contribute zeros.
        # After all-reduce, every rank has identical global_counts and solves
        # the same LP independently, so no broadcast is needed.
        global_counts = local_counts.float()
        if self.ep_group is not None:
            self.ep_group.all_reduce(global_counts)

        # Step 3: Run LP solver
        log2phy_prob = self._solve_torch(global_counts)

        return log2phy_prob

    def _solve_torch(self, global_counts: torch.Tensor) -> torch.Tensor:
        """PyTorch fallback solver."""
        from sglang.jit_kernel.lplb.torch_solver import solve_ipm

        device = global_counts.device
        total = global_counts.sum()
        if total > 0:
            counts_norm = global_counts / total
        else:
            counts_norm = global_counts

        t1 = counts_norm[self.log_single.long()]
        b2 = -(self.B1 @ t1).flatten()
        b1 = counts_norm[self.log_replicated.long()]
        b = torch.cat([b1, b2])

        big_M_col = b - self.A_base.sum(dim=1)
        A_full = torch.hstack([self.A_base, big_M_col.unsqueeze(1)])

        x = solve_ipm(A_full, b, self.c_vec)

        x_ratios = x[: self.num_red_phy].clamp(min=0)
        phy_prob = torch.zeros(
            self.num_single + self.num_red_phy + 1,
            dtype=torch.float32,
            device=device,
        )
        phy_prob[self.phy_replicated.long()] = x_ratios
        phy_prob[self.phy_single.long()] = t1
        log2phy_prob = torch.take(phy_prob, self.log2phy.long())
        return log2phy_prob
