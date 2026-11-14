"""Per-batch replica balancing with graph-stable NPU workspaces."""

import torch

from sglang.srt.hardware_backend.npu.moe import lplb_kernels

# Same iteration budget as the community CUDA LP solver.
_IPM_ITERATIONS = 5


class NpuLPLBSolver:
    """Keep graph-captured buffers alive across expert placement updates.

    Logical counts choose replica probabilities. The EPLB recorder separately
    receives physical IDs after sampling, so it records actual dispatches.

    Only replicated experts are LP variables. Experts with one copy contribute
    fixed load to their owning rank. The workspace reserves enough room for any
    placement with the same number of redundant slots, so EPLB can change which
    experts are replicated without invalidating captured graphs.
    """

    def __init__(self, *, phy2log, log2phy, num_gpus, ep_group=None):
        self.num_logical, _ = log2phy.shape
        self.num_phy = phy2log.numel()
        self.num_ranks = num_gpus
        self.ep_group = ep_group
        redundant = self.num_phy - self.num_logical
        if redundant < 0 or self.num_phy % num_gpus:
            raise ValueError(
                "NPU LPLB requires all experts and equal slots per EP rank."
            )
        self._has_redundancy = redundant > 0
        # With R redundant slots, at most min(R, E) experts have replicas.
        # Their original copies plus redundant copies occupy at most R + H slots.
        h = min(redundant, self.num_logical)
        m = redundant + h
        self._replica_capacity = m
        # Rows: H expert-demand constraints, then one load constraint per rank.
        # Columns: M replica masses, rank slack, maximum load, and an auxiliary
        # variable that supplies a feasible starting point for the IPM solver.
        nc, nv = h + num_gpus, m + num_gpus + 2
        device = log2phy.device

        def zeros(*shape, dtype=torch.float32):
            return torch.zeros(shape, dtype=dtype, device=device)

        self._base = zeros(nc, nv)
        self._replicated = zeros(h, dtype=torch.int32)
        self._single_rank = zeros(self.num_logical, dtype=torch.int32)
        self._replica_index = zeros(self.num_phy, dtype=torch.int32)
        self._mapping = zeros(self.num_logical, redundant + 1, dtype=torch.int32)
        self._counts = zeros(self.num_logical, dtype=torch.int32)
        self._a, self._b, self._x = zeros(nc, nv), zeros(nc), zeros(nv)
        # Residual, auxiliary, direction, and predicted max-load improvement.
        self.diagnostics = zeros(4)
        self.fallback_count = zeros(dtype=torch.int32)
        self._probabilities = zeros(*self._mapping.shape)
        self.update_layout(phy2log=phy2log, log2phy=log2phy)

    def update_layout(self, *, phy2log, log2phy):
        """Refresh at EPLB's completed-update boundary, never during forward."""
        if phy2log.numel() != self.num_phy or log2phy.shape[0] != self.num_logical:
            raise ValueError(
                "NPU LPLB graph workspaces require a fixed expert capacity."
            )
        owners = phy2log.detach().cpu().tolist()
        copies = [[] for _ in range(self.num_logical)]
        for slot, expert in enumerate(owners):
            if not 0 <= expert < self.num_logical:
                raise ValueError("Invalid logical expert in NPU LPLB placement.")
            copies[expert].append(slot)
        if any(not slots for slots in copies):
            raise ValueError("NPU LPLB placement must contain every logical expert.")

        h = self._replicated.numel()
        m = self._replica_capacity
        base = torch.zeros(self._base.shape)
        replicated = torch.full((h,), -1, dtype=torch.int32)
        single_rank = torch.full((self.num_logical,), -1, dtype=torch.int32)
        replica_index = torch.full((self.num_phy,), -1, dtype=torch.int32)
        row = column = 0
        for expert, slots in enumerate(copies):
            if len(slots) == 1:
                single_rank[expert] = slots[0] // (self.num_phy // self.num_ranks)
                continue
            replicated[row] = expert
            for slot in slots:
                base[row, column] = 1
                base[h + slot // (self.num_phy // self.num_ranks), column] = 1
                replica_index[slot] = column
                column += 1
            row += 1
        base[h:, m : m + self.num_ranks] = torch.eye(self.num_ranks)
        base[h:, -2] = -1
        # Use metadata's copy order: dispatch consumes that same ordered map.
        mapping = log2phy[:, : self._mapping.shape[1]].to(dtype=torch.int32)
        for destination, value in (
            (self._base, base),
            (self._replicated, replicated),
            (self._single_rank, single_rank),
            (self._replica_index, replica_index),
            (self._mapping, mapping),
        ):
            destination.copy_(value)
        if not self._has_redundancy:
            self._probabilities.fill_(1)

    def solve(self, topk_ids, num_token_non_padded=None):
        """All target EP ranks call this, including ranks with no valid rows."""
        if not self._has_redundancy:
            return self._probabilities
        lplb_kernels.count_routes(topk_ids, num_token_non_padded, self._counts)
        counts = self._counts
        if self.ep_group is not None:
            counts = self.ep_group.all_reduce(counts)
        return self.solve_counts(counts)

    def solve_counts(self, global_counts):
        """Solve already-reduced counts; also used to precompile before capture."""
        lplb_kernels.solve_ipm(
            self._base,
            self._replicated,
            self._single_rank,
            global_counts,
            self._a,
            self._b,
            self._x,
            self.diagnostics,
            _IPM_ITERATIONS,
        )
        lplb_kernels.extract_probabilities(
            self._mapping,
            self._replica_index,
            self._x,
            self.diagnostics,
            self._probabilities,
            self.fallback_count,
        )
        return self._probabilities


def solve_npu_lplb(topk_ids, info, layer_id, num_token_non_padded):
    """Shared entry for ordinary, hash and vision routers after logical top-k."""
    if info is None or info.ep_dispatch_algorithm != "lp":
        return None
    from sglang.srt.eplb.lplb_solver import get_global_lplb_solver

    solver = get_global_lplb_solver(layer_id)
    if solver is None:
        raise RuntimeError(f"NPU LPLB solver was not initialized for layer {layer_id}.")
    return solver.solve(topk_ids, num_token_non_padded)


def _restore_all_replicas(metadata):
    """LP must see all copies, including those initial locality rules omit."""
    from sglang.srt.eplb.expert_location import _compute_logical_to_all_physical_map

    mapping = _compute_logical_to_all_physical_map(
        physical_to_logical_map=metadata.physical_to_logical_map_cpu,
        num_logical_experts=metadata.num_logical_experts,
        ep_size=metadata.ep_size,
        moe_ep_rank=None,
    )
    mapping = torch.nn.functional.pad(
        mapping, (0, metadata.num_physical_experts - mapping.shape[-1]), value=-1
    )
    metadata.logical_to_all_physical_map.copy_(mapping)
    metadata.logical_to_all_physical_map_cpu.copy_(mapping)
    metadata.logical_to_all_physical_map_num_valid.copy_((mapping >= 0).sum(-1))


def init_npu_lplb_solvers(model_config, update_layer_ids=None):
    from sglang.srt.eplb.expert_location import get_global_expert_location_metadata
    from sglang.srt.eplb.lplb_solver import (
        assert_lplb_supported_model,
        clear_global_lplb_solvers,
        get_global_lplb_solver,
        set_global_lplb_solver,
    )
    from sglang.srt.runtime_context import get_exec, get_parallel

    parallel = get_parallel()
    if (
        parallel.pp_size != 1
        or parallel.attn_cp_size != 1
        or parallel.attn_dcp_size != 1
        or parallel.tp_size != parallel.moe_ep_size
        or get_exec().moe.moe_a2a_backend != "ascend_ops"
    ):
        raise ValueError("NPU LPLB requires ascend_ops, TP=EP, PP=CP=DCP=1.")
    for architecture in model_config.hf_config.architectures:
        # V4 and V4.1 share this model class and its ordinary/hash/vision routers.
        if architecture != "DeepseekV4ForCausalLM":
            assert_lplb_supported_model(architecture)
    metadata = get_global_expert_location_metadata()
    if metadata is None:
        raise ValueError("NPU LPLB requires expert location metadata.")
    if update_layer_ids is None:
        _restore_all_replicas(metadata)
        clear_global_lplb_solvers()
    layer_ids = (
        range(metadata.num_layers) if update_layer_ids is None else update_layer_ids
    )
    for layer_id in layer_ids:
        layout = dict(
            # EPLB updates the CPU mirror before refreshing these solvers.
            # Build constraints from it without a per-layer device readback.
            phy2log=metadata.physical_to_logical_map_cpu[layer_id],
            log2phy=metadata.logical_to_all_physical_map[layer_id],
        )
        solver = get_global_lplb_solver(layer_id)
        if solver is None:
            solver = NpuLPLBSolver(
                **layout,
                num_gpus=metadata.ep_size,
                ep_group=parallel.moe_ep_group,
            )
            if solver._has_redundancy:
                solver.solve_counts(solver._counts)
                solver.fallback_count.zero_()
            set_global_lplb_solver(layer_id, solver)
        else:
            solver.update_layout(**layout)
