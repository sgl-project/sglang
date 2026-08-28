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
        self._num_phy_per_gpu = self.num_phy // num_gpus

        # ---- Static, placement-independent shape bounds -------------------
        # The LP's natural size depends on how EPLB spread the redundant
        # slots, which changes on every rebalance. Recompiling and
        # reallocating per placement invalidates captured CUDA graphs, so we
        # size everything for the worst case once and zero-pad each actual
        # placement up to it (see the padding contract in csrc/lplb/ipm.cuh).
        #
        # With k = num_red_log and R = num_phy - num_logical:
        #   num_red_phy = k + R   and   num_single = num_logical - k
        # (each logical expert is either single-copy or replicated, and the
        # replicated ones absorb exactly the R redundant slots). Every
        # replicated expert needs at least one redundant slot, so k <= R.
        num_redundant = self.num_phy - self.num_logical
        if num_redundant < 0:
            raise ValueError(
                f"LPLBSolver requires num_phy ({self.num_phy}) >= num_logical "
                f"({self.num_logical})."
            )
        self.max_red_log = min(num_redundant, self.num_logical)
        self.max_red_phy = self.max_red_log + num_redundant
        self.max_single = self.num_logical
        self.nc = self.max_red_log + num_gpus
        self.nv = self.max_red_phy + num_gpus + 2

        # ---- Fixed-address buffers ----------------------------------------
        # Every tensor below is passed to a JIT kernel, so a captured CUDA
        # graph bakes in its data_ptr(). They are allocated exactly once here
        # and only ever mutated in place by update_placement() -- never
        # reassigned -- so the graph stays valid across rebalances.
        f32 = dict(dtype=torch.float32, device=device)
        i64 = dict(dtype=torch.int64, device=device)
        self._A_full = torch.zeros(self.nc, self.nv, **f32)
        self._b = torch.zeros(self.nc, **f32)
        self._t1 = torch.zeros(self.max_single, **f32)
        self._x = torch.zeros(self.nv, **f32)
        self._log2phy_prob = torch.zeros(self.num_logical, self.max_copies, **f32)
        self._A_base_row_sum = torch.zeros(self.nc, **f32)
        self.B1 = torch.zeros(num_gpus, self.max_single, **f32)
        # Index tensors are int64 so they can be used directly as index
        # tensors without per-call .long() casts, and are -1-padded in their
        # tail so the kernels can skip the unused slots.
        self.log_single = torch.full((self.max_single,), -1, **i64)
        self.phy_single = torch.full((self.max_single,), -1, **i64)
        self.log_replicated = torch.full((self.max_red_log,), -1, **i64)
        self.phy_replicated = torch.full((self.max_red_phy,), -1, **i64)
        self.log2phy = torch.full((self.num_logical, self.max_copies), -1, **i64)

        # Objective: minimize M (second-to-last var), penalize Big-M auxiliary.
        # Both live at fixed tail offsets, so this vector is placement-
        # independent and never needs updating.
        self.c_vec = torch.zeros(self.nv, **f32)
        self.c_vec[-2] = 1.0
        self.c_vec[-1] = 1000.0

        # Pre-JIT-compile the fused IPM kernel for the static (NC, NV) shape so
        # the 20-40s compile cost happens once at startup rather than on the
        # first real request. Because the shape no longer depends on the
        # placement, every layer shares one compiled kernel and a rebalance
        # never triggers a recompile.
        from sglang.kernels.ops.lplb.torch_solver import warmup as _ipm_warmup

        _ipm_warmup(self.nc, self.nv, num_iters=5, device=device)

        self.update_placement(phy2log, log2phy, logical_to_all_physical_map_num_valid)

    @property
    def A_base(self) -> torch.Tensor:
        """The constraint matrix without the Big-M column, as a view.

        ``_A_full``'s first ``nv - 1`` columns hold A_base persistently; only
        the last column is rewritten per solve by the prep kernel.
        """
        return self._A_full[:, : self.nv - 1]

    def update_placement(
        self,
        phy2log: torch.Tensor,
        log2phy: torch.Tensor,
        logical_to_all_physical_map_num_valid=None,
    ) -> None:
        """Rebuild the LP constraint data for a new expert placement, in place.

        Called at init and again after every EPLB rebalance. Mutates the
        existing buffers rather than allocating new ones, so CUDA graphs
        captured against this solver stay valid. The caller must ensure no
        forward pass is in flight (the scheduler serializes rebalance against
        the forward loop on the same stream).
        """
        device = self._A_full.device
        phy2log = phy2log.to(device)
        log2phy = log2phy.to(device)

        if phy2log.shape[0] != self.num_phy or log2phy.shape != (
            self.num_logical,
            self.max_copies,
        ):
            raise ValueError(
                "LPLBSolver.update_placement cannot change the expert-map "
                f"shapes: got phy2log {tuple(phy2log.shape)}, log2phy "
                f"{tuple(log2phy.shape)}, expected ({self.num_phy},) and "
                f"({self.num_logical}, {self.max_copies})."
            )

        self._has_redundancy = False
        if logical_to_all_physical_map_num_valid is not None:
            self._has_redundancy = bool(
                (logical_to_all_physical_map_num_valid > 1).any()
            )

        # Count copies per logical expert, then split single vs replicated.
        logcnt = torch.bincount(phy2log, minlength=self.num_logical)
        log_single = torch.nonzero(logcnt == 1).flatten()
        phy_single = log2phy[log_single, 0].to(torch.int64)
        log_replicated = torch.nonzero(logcnt > 1).flatten()
        phy_replicated = torch.nonzero(logcnt[phy2log] > 1).flatten()

        self.num_single = int(log_single.numel())
        self.num_red_log = int(log_replicated.numel())
        self.num_red_phy = int(phy_replicated.numel())
        if (
            self.num_single > self.max_single
            or self.num_red_log > self.max_red_log
            or self.num_red_phy > self.max_red_phy
        ):
            raise ValueError(
                "LPLB placement exceeds the static shape bounds: "
                f"num_single={self.num_single}/{self.max_single}, "
                f"num_red_log={self.num_red_log}/{self.max_red_log}, "
                f"num_red_phy={self.num_red_phy}/{self.max_red_phy}."
            )

        # -1-padded index tensors: the kernels skip negative slots.
        for buf, src, n in (
            (self.log_single, log_single, self.num_single),
            (self.phy_single, phy_single, self.num_single),
            (self.log_replicated, log_replicated, self.num_red_log),
            (self.phy_replicated, phy_replicated, self.num_red_phy),
        ):
            buf.fill_(-1)
            buf[:n].copy_(src)

        # GPU assignment matrices. B_full is scratch, not a kernel argument.
        B_full = torch.zeros(
            (self.num_gpus, self.num_phy), dtype=torch.float32, device=device
        )
        for i in range(self.num_gpus):
            B_full[i, i * self._num_phy_per_gpu : (i + 1) * self._num_phy_per_gpu] = 1
        self.B1.zero_()
        self.B1[:, : self.num_single].copy_(B_full[:, phy_single])
        B2 = B_full[:, phy_replicated]

        # C matrix (copy-to-logical mapping).
        C = torch.zeros(
            (self.num_red_log, self.num_red_phy), dtype=torch.float32, device=device
        )
        phy2log_rep = phy2log[phy_replicated]
        for i in range(self.num_red_log):
            C[i, phy2log_rep == log_replicated[i]] = 1.0

        # A_base = [[C, 0, 0], [B2, I, -1]], written into the padded buffer at
        # FIXED block offsets so that the slack / M / Big-M columns and the
        # per-GPU rows never move when the placement changes:
        #
        #   rows: [0, num_red_log) replicated-logical | [., max_red_log) PAD
        #         | [max_red_log, max_red_log + num_gpus) per-GPU
        #   cols: [0, num_red_phy) replicated-physical | [., max_red_phy) PAD
        #         | [max_red_phy, +num_gpus) slack | M | Big-M
        #
        # Everything outside those blocks stays zero, which is exactly the
        # inert padding the IPM kernel requires.
        k, rp, g = self.max_red_log, self.max_red_phy, self.num_gpus
        self._A_full.zero_()
        self._A_full[: self.num_red_log, : self.num_red_phy].copy_(C)
        self._A_full[k : k + g, : self.num_red_phy].copy_(B2)
        self._A_full[k : k + g, rp : rp + g].copy_(
            torch.eye(g, dtype=torch.float32, device=device)
        )
        self._A_full[k : k + g, rp + g] = -1.0
        # Column nv-1 is the Big-M column, rewritten by the prep kernel on
        # every solve; it is zero here and excluded from the row sum below.

        self._A_base_row_sum.copy_(self._A_full[:, : self.nv - 1].sum(dim=1))
        self.log2phy.copy_(log2phy.to(torch.int64))

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
            self.num_phy,
        )
        return self._log2phy_prob
