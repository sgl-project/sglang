import torch
from sglang.srt.distributed import get_world_group


def _process_global_counts_placeholder(global_counts: torch.Tensor):
    """Placeholder function to process global counts on CPU"""
    # Return the processed global counts (you can modify this to return any processed result)
    return global_counts

def count_logical_expert_tokens(
    logical_expert_ids: torch.Tensor, 
    num_logical_experts: int
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
    
    if valid_ids.numel() > 0:
        # Use scatter_add to count occurrences
        logical_counts.scatter_add_(
            dim=0, 
            index=valid_ids.long(), 
            src=torch.ones_like(valid_ids, dtype=torch.int32)
        )
    
    return logical_counts

def get_global_logical_counts_cpu_allreduce(local_counts: torch.Tensor) -> torch.Tensor:
    """Get global logical counts using SGLang's parallel state system.
    
    All ranks move local_counts to CPU, then use the CPU communication group for all-reduce.
    All ranks return the global result.
    
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
    
    # Move local counts to CPU for CPU-based communication
    local_counts_cpu = local_counts.cpu()
    
    # Use the CPU communication group for all-reduce
    torch.distributed.all_reduce(local_counts_cpu, group=group.cpu_group, op=torch.distributed.ReduceOp.SUM)
    # Move result back to GPU
    global_counts = local_counts_cpu.to(local_counts.device)
    return global_counts

def get_log2phy_prob(
        topk_ids: torch.Tensor,
        num_logical_experts: int
):
    """Using Linear Programming to get the redundant token distribution probability

    Args:
        topk_ids: Tensor of shape (num_tokens, topk) containing logical expert IDs
        num_logical_experts: Number of logical experts

    Returns:
        Tensor of shape (num_logical_experts,) containing global token counts for each expert
    """
    # Step 1: Count local logical expert tokens
    local_counts = count_logical_expert_tokens(topk_ids, num_logical_experts)

    # Step 2: All-reduce to get global counts
    global_counts = get_global_logical_counts_cpu_allreduce(local_counts)

    # Step 3: Use LP to get the redundant token distribution probability

    return global_counts
