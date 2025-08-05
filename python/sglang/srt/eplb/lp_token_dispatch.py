import ecos
import numpy as np
import scipy.sparse as sp
import torch
import torch.distributed as dist

from sglang.srt.distributed import get_world_group
from sglang.srt.eplb.expert_location_dispatch import ExpertLocationDispatchInfo


def run_lp_solver(
    global_counts: torch.Tensor,
    lp_metadata: dict[str, np.ndarray],
    log2phy: torch.Tensor,
):
    """Use LP to get the redundant token distribution probability

    Args:
        global_counts: Tensor of shape (num_logical_experts,) containing global token counts for each logical expert

    Returns:
        Tensor of shape log2phy (num_logical_experts, max_redundant_count) containing redundant token distribution probability for each logical expert
    """
    global_counts = global_counts.cpu().numpy()

    t1: np.ndarray = global_counts[lp_metadata["single_expert_array"]]
    # print(f"t1 shape: {t1.shape}")

    left = lp_metadata["B1"] @ t1

    if hasattr(left, "toarray"):
        h1 = -left.toarray().flatten()
    else:
        h1 = -left.flatten()

    h = np.concatenate(
        [h1, np.zeros(lp_metadata["phy_replicated_expert_array"].shape[0])]
    )
    b = global_counts[lp_metadata["log_replicated_expert_array"]].astype(np.float64)
    G = sp.csc_matrix(lp_metadata["G"])
    A = sp.csc_matrix(lp_metadata["A"])

    solution = ecos.solve(
        c=lp_metadata["c"],
        G=G,
        h=h,
        dims=lp_metadata["dims"],
        A=A,
        b=b,
        **lp_metadata["ecos_opts"]
    )
    if solution["info"]["exitFlag"] in [0, 10]:
        tol = lp_metadata["ecos_opts"]["abstol"]
        t2_value = solution["x"][1:]
        t2_value[t2_value < tol] = 0
        phy_prob = np.ones(
            lp_metadata["single_expert_array"].shape[0]
            + lp_metadata["phy_replicated_expert_array"].shape[0]
        )
        phy_prob[lp_metadata["phy_replicated_expert_array"]] = t2_value
        phy_prob[lp_metadata["single_expert_array"]] = t1

        log2phy_prob = np.full_like(log2phy, fill_value=0, dtype=float)
        mask = log2phy != -1
        log2phy_prob[mask] = np.take(phy_prob, log2phy[mask])
        log2phy_prob = (torch.from_numpy(log2phy_prob) * 100).to(torch.int8)
        return log2phy_prob
    else:
        # Fall back to random dispatch
        # copy log2phy
        log2phy_prob = log2phy.clone()
        # replace -1 with 0, all other values to 1
        log2phy_prob[log2phy_prob == -1] = 0
        log2phy_prob[log2phy_prob != -1] = 1
        return log2phy_prob.to(torch.int8)


def count_logical_expert_tokens(
    logical_expert_ids: torch.Tensor, num_logical_experts: int
) -> torch.Tensor:
    """Count logical expert token occurrences from topk selection

    Args:
        logical_expert_ids: Tensor of shape (num_tokens, topk) containing logical expert IDs
        num_logical_experts: Number of logical experts

    Returns:
        Tensor of shape (num_logical_experts,) containing token counts for each expert
    """
    device = logical_expert_ids.device
    logical_counts = torch.zeros(num_logical_experts, dtype=torch.int32, device=device)

    # Flatten the expert IDs and count occurrences
    flat_ids = logical_expert_ids.flatten()
    # Filter out invalid IDs (like -1 for padding)
    valid_mask = flat_ids >= 0
    valid_ids = flat_ids[valid_mask]

    logical_counts.scatter_add_(
        dim=0,
        index=valid_ids.long(),
        src=torch.ones_like(valid_ids, dtype=torch.int32),
    )

    return logical_counts


def get_global_logical_counts_on_rank0(local_counts: torch.Tensor) -> torch.Tensor:
    """Get global logical counts using SGLang's parallel state system.

    All ranks move local_counts to CPU, then use the CPU communication group for reduce.
    The result is only correct on rank 0.

    Args:
        local_counts: Local logical counts tensor (on GPU)

    Returns:
        Global logical counts tensor on GPU
    """
    # Get the tensor parallel group from SGLang
    group = get_world_group()

    if group.world_size == 1:
        # Single rank case, just return local counts
        return local_counts

    # Use the CPU communication group for all-reduce
    torch.distributed.reduce(
        local_counts,
        dst=0,
        group=group.device_group,
        op=torch.distributed.ReduceOp.SUM,
    )
    return local_counts


def send_log2phy_prob_broadcast(log2phy_prob: torch.Tensor):
    """Send log2phy_prob to all ranks"""
    group = get_world_group()
    torch.distributed.broadcast(log2phy_prob, src=0, group=group.device_group)
    return log2phy_prob


def get_log2phy_prob(
    topk_ids: torch.Tensor,
    num_logical_experts: int,
    expert_location_dispatch_info: ExpertLocationDispatchInfo,
):
    """Using Linear Programming to get the redundant token distribution probability

    Args:
        topk_ids: Tensor of shape (num_tokens, topk) containing logical expert IDs
        num_logical_experts: Number of logical experts

    Returns:
        Tensor of shape (num_logical_experts,) containing global token counts for each expert
    """
    device = topk_ids.device
    # Step 1: Count local logical expert tokens
    local_counts = count_logical_expert_tokens(topk_ids, num_logical_experts)

    # Step 2: All-reduce to get global counts
    global_counts = get_global_logical_counts_on_rank0(local_counts)

    # Step 3: Use LP to get the redundant token distribution probability
    if dist.get_rank() == 0:
        log2phy_prob = run_lp_solver(
            global_counts,
            expert_location_dispatch_info.lp_metadata,
            expert_location_dispatch_info.partial_logical_to_all_physical_map.cpu(),
        )
    else:
        log2phy_prob = torch.empty_like(
            expert_location_dispatch_info.partial_logical_to_all_physical_map,
            device=device,
            dtype=torch.int8,
        )

    # Step 4: Broadcast to all ranks
    log2phy_prob = log2phy_prob.to(device=device)
    log2phy_prob = send_log2phy_prob_broadcast(log2phy_prob)
    return log2phy_prob
