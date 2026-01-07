#!/usr/bin/env python3
"""
Analyze DeepEP Waterfill algorithm performance.

This script:
1. Simulates realistic token distributions
2. Runs waterfill algorithm
3. Analyzes load distribution before/after waterfill
4. Checks for tile utilization issues (shared tokens < 128)
"""

import torch
import os
import sys
from typing import Dict, List, Tuple
import importlib.util

# Import directly from the file
spec = importlib.util.spec_from_file_location(
    "deepep_waterfill", 
    os.path.join(os.path.dirname(__file__), "python/sglang/srt/layers/moe/deepep_waterfill.py")
)
deepep_waterfill = importlib.util.module_from_spec(spec)
spec.loader.exec_module(deepep_waterfill)

count_routed_per_rank_pytorch = deepep_waterfill.count_routed_per_rank_pytorch
assign_shared_destination_pytorch = deepep_waterfill.assign_shared_destination_pytorch
expand_topk_with_shared_expert = deepep_waterfill.expand_topk_with_shared_expert
identify_shared_expert_tokens = deepep_waterfill.identify_shared_expert_tokens
DeepEPWaterfillBalancer = deepep_waterfill.DeepEPWaterfillBalancer
LOCAL_SHARED_MARKER = deepep_waterfill.LOCAL_SHARED_MARKER


def generate_realistic_topk(
    num_tokens: int,
    num_experts: int = 256,
    topk: int = 8,
    skew_factor: float = 0.0,  # 0 = uniform, higher = more skewed
) -> torch.Tensor:
    """
    Generate realistic topk_ids with optional load skew.
    
    Args:
        num_tokens: Number of tokens
        num_experts: Number of experts
        topk: Number of experts per token
        skew_factor: How skewed the distribution is (0=uniform, 1=heavy skew)
    """
    if skew_factor == 0:
        # Uniform distribution
        topk_ids = torch.randint(0, num_experts, (num_tokens, topk))
    else:
        # Skewed distribution - some experts are more popular
        # Create popularity weights
        weights = torch.ones(num_experts)
        # Make first 25% of experts 2-4x more popular
        popular_count = num_experts // 4
        weights[:popular_count] *= (1 + 3 * skew_factor)
        weights = weights / weights.sum()
        
        # Sample experts based on weights
        topk_ids = torch.multinomial(
            weights.unsqueeze(0).expand(num_tokens, -1),
            topk,
            replacement=False
        )
    
    return topk_ids.to(torch.int64)


def analyze_distribution(
    topk_ids: torch.Tensor,
    num_experts: int,
    world_size: int,
    source_rank: int,
    routed_scaling_factor: float = 2.5,
) -> Dict:
    """
    Analyze token distribution with and without waterfill.
    """
    num_tokens = topk_ids.shape[0]
    experts_per_rank = num_experts // world_size
    
    # Count routed tokens per rank
    routed_counts = count_routed_per_rank_pytorch(topk_ids, num_experts, world_size)
    
    # Create balancer
    balancer = DeepEPWaterfillBalancer(
        num_experts=num_experts,
        world_size=world_size,
        rank=source_rank,
        routed_scaling_factor=routed_scaling_factor,
    )
    
    # Prepare dispatch
    topk_weights = torch.ones(num_tokens, topk_ids.shape[1], dtype=torch.float32) / topk_ids.shape[1]
    expanded_ids, expanded_weights, local_mask = balancer.prepare_dispatch(
        topk_ids, topk_weights, routed_counts
    )
    
    # Analyze shared expert destinations
    shared_dest = torch.zeros(num_tokens, dtype=torch.int64)
    shared_dest[local_mask] = source_rank
    remote_mask = ~local_mask
    if remote_mask.any():
        shared_dest[remote_mask] = expanded_ids[remote_mask, -1] // experts_per_rank
    
    shared_counts = torch.bincount(shared_dest, minlength=world_size)
    
    # Calculate total load per rank (routed + shared)
    total_counts = routed_counts + shared_counts
    
    # Baseline: all shared tokens on source rank
    baseline_shared_counts = torch.zeros(world_size, dtype=torch.int64)
    baseline_shared_counts[source_rank] = num_tokens
    baseline_total = routed_counts + baseline_shared_counts
    
    return {
        "num_tokens": num_tokens,
        "routed_counts": routed_counts,
        "shared_counts_waterfill": shared_counts,
        "shared_counts_baseline": baseline_shared_counts,
        "total_counts_waterfill": total_counts,
        "total_counts_baseline": baseline_total,
        "local_shared_count": local_mask.sum().item(),
        "remote_shared_count": remote_mask.sum().item(),
    }


def compute_load_balance_metrics(counts: torch.Tensor) -> Dict:
    """Compute load balance metrics."""
    counts_float = counts.float()
    mean_load = counts_float.mean().item()
    max_load = counts_float.max().item()
    min_load = counts_float.min().item()
    std_load = counts_float.std().item()
    
    # Load imbalance ratio
    imbalance_ratio = max_load / mean_load if mean_load > 0 else float('inf')
    
    # Coefficient of variation
    cv = std_load / mean_load if mean_load > 0 else float('inf')
    
    return {
        "mean": mean_load,
        "max": max_load,
        "min": min_load,
        "std": std_load,
        "imbalance_ratio": imbalance_ratio,
        "cv": cv,
    }


def check_tile_utilization(shared_counts: torch.Tensor, tile_size: int = 128) -> Dict:
    """Check for potential tile utilization issues."""
    issues = []
    for rank, count in enumerate(shared_counts.tolist()):
        if 0 < count < tile_size:
            issues.append({
                "rank": rank,
                "count": count,
                "wasted_slots": tile_size - count,
                "utilization": count / tile_size * 100,
            })
    
    return {
        "tile_size": tile_size,
        "issues": issues,
        "num_ranks_with_issues": len(issues),
    }


def print_analysis_report(
    scenario_name: str,
    result: Dict,
    world_size: int,
):
    """Print detailed analysis report."""
    print("\n" + "=" * 80)
    print(f"Scenario: {scenario_name}")
    print("=" * 80)
    
    print(f"\nTotal tokens: {result['num_tokens']}")
    
    # Per-rank breakdown
    print("\n" + "-" * 60)
    print("Per-Rank Token Distribution:")
    print("-" * 60)
    print(f"{'Rank':<6} {'Routed':<10} {'Shared(WF)':<12} {'Shared(BL)':<12} {'Total(WF)':<12} {'Total(BL)':<12}")
    print("-" * 60)
    
    for rank in range(world_size):
        routed = result['routed_counts'][rank].item()
        shared_wf = result['shared_counts_waterfill'][rank].item()
        shared_bl = result['shared_counts_baseline'][rank].item()
        total_wf = result['total_counts_waterfill'][rank].item()
        total_bl = result['total_counts_baseline'][rank].item()
        print(f"{rank:<6} {routed:<10} {shared_wf:<12} {shared_bl:<12} {total_wf:<12} {total_bl:<12}")
    
    # Local vs Remote shared
    print(f"\nShared Expert Distribution:")
    print(f"  Local (computed on source rank): {result['local_shared_count']}")
    print(f"  Remote (sent to other ranks):    {result['remote_shared_count']}")
    
    # Load balance metrics
    print("\n" + "-" * 60)
    print("Load Balance Metrics:")
    print("-" * 60)
    
    metrics_wf = compute_load_balance_metrics(result['total_counts_waterfill'])
    metrics_bl = compute_load_balance_metrics(result['total_counts_baseline'])
    
    print(f"{'Metric':<25} {'Waterfill':<15} {'Baseline':<15} {'Improvement':<15}")
    print("-" * 60)
    
    for key in ['mean', 'max', 'min', 'std', 'imbalance_ratio', 'cv']:
        wf_val = metrics_wf[key]
        bl_val = metrics_bl[key]
        if key in ['imbalance_ratio', 'cv', 'std', 'max']:
            # Lower is better
            if bl_val != 0:
                improvement = (bl_val - wf_val) / bl_val * 100
                imp_str = f"{improvement:+.1f}%"
            else:
                imp_str = "N/A"
        else:
            imp_str = "-"
        print(f"{key:<25} {wf_val:<15.2f} {bl_val:<15.2f} {imp_str:<15}")
    
    # Tile utilization check
    print("\n" + "-" * 60)
    print("Tile Utilization Analysis (tile_size=128):")
    print("-" * 60)
    
    tile_check = check_tile_utilization(result['shared_counts_waterfill'])
    
    if tile_check['issues']:
        print(f"⚠️  Found {tile_check['num_ranks_with_issues']} rank(s) with potential tile waste:")
        for issue in tile_check['issues']:
            print(f"   Rank {issue['rank']}: {issue['count']} tokens "
                  f"({issue['utilization']:.1f}% utilization, {issue['wasted_slots']} slots wasted)")
    else:
        print("✓ No tile utilization issues (all ranks have 0 or ≥128 shared tokens)")
    
    return metrics_wf, metrics_bl


def run_analysis():
    """Run comprehensive waterfill analysis."""
    print("=" * 80)
    print("DeepEP Waterfill Algorithm Performance Analysis")
    print("=" * 80)
    
    num_experts = 256
    world_size = 8
    source_rank = 0
    
    # Test scenarios
    scenarios = [
        ("Uniform Distribution (1024 tokens)", 1024, 0.0),
        ("Uniform Distribution (4096 tokens)", 4096, 0.0),
        ("Slightly Skewed (1024 tokens)", 1024, 0.3),
        ("Heavily Skewed (1024 tokens)", 1024, 0.7),
        ("Heavily Skewed (4096 tokens)", 4096, 0.7),
        ("Small Batch (128 tokens)", 128, 0.0),
        ("Very Small Batch (32 tokens)", 32, 0.0),
    ]
    
    all_results = []
    
    for name, num_tokens, skew in scenarios:
        torch.manual_seed(42)  # Reproducibility
        topk_ids = generate_realistic_topk(num_tokens, num_experts, topk=8, skew_factor=skew)
        
        result = analyze_distribution(
            topk_ids, num_experts, world_size, source_rank
        )
        
        metrics_wf, metrics_bl = print_analysis_report(name, result, world_size)
        
        all_results.append({
            "name": name,
            "num_tokens": num_tokens,
            "skew": skew,
            "result": result,
            "metrics_wf": metrics_wf,
            "metrics_bl": metrics_bl,
        })
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY: Load Imbalance Improvement")
    print("=" * 80)
    print(f"{'Scenario':<40} {'BL Imbalance':<15} {'WF Imbalance':<15} {'Reduction':<15}")
    print("-" * 80)
    
    for r in all_results:
        bl_imb = r['metrics_bl']['imbalance_ratio']
        wf_imb = r['metrics_wf']['imbalance_ratio']
        reduction = (bl_imb - wf_imb) / bl_imb * 100 if bl_imb > 0 else 0
        print(f"{r['name']:<40} {bl_imb:<15.2f} {wf_imb:<15.2f} {reduction:<15.1f}%")
    
    # Tile utilization summary
    print("\n" + "=" * 80)
    print("SUMMARY: Tile Utilization Issues")
    print("=" * 80)
    
    issues_found = False
    for r in all_results:
        tile_check = check_tile_utilization(r['result']['shared_counts_waterfill'])
        if tile_check['issues']:
            issues_found = True
            print(f"\n{r['name']}:")
            for issue in tile_check['issues']:
                print(f"  ⚠️  Rank {issue['rank']}: {issue['count']} tokens ({issue['utilization']:.1f}% tile utilization)")
    
    if not issues_found:
        print("✓ No tile utilization issues found in any scenario!")
    
    # Multi-rank simulation
    print("\n" + "=" * 80)
    print("MULTI-RANK SIMULATION: What each rank sends")
    print("=" * 80)
    
    # Simulate from each rank's perspective
    torch.manual_seed(42)
    num_tokens = 2048
    topk_ids = generate_realistic_topk(num_tokens, num_experts, topk=8, skew_factor=0.5)
    
    # Calculate global routed counts (simulated AllReduce)
    global_routed_counts = count_routed_per_rank_pytorch(topk_ids, num_experts, world_size) * world_size
    
    print(f"\nGlobal routed counts (after AllReduce): {global_routed_counts.tolist()}")
    print(f"Tokens per rank: {num_tokens}")
    print()
    
    # Simulate each rank
    all_shared_recv = torch.zeros(world_size, world_size, dtype=torch.int64)  # [src, dst]
    
    for src_rank in range(world_size):
        balancer = DeepEPWaterfillBalancer(
            num_experts=num_experts,
            world_size=world_size,
            rank=src_rank,
            routed_scaling_factor=2.5,
        )
        
        topk_weights = torch.ones(num_tokens, 8, dtype=torch.float32) / 8
        expanded_ids, _, local_mask = balancer.prepare_dispatch(
            topk_ids, topk_weights, global_routed_counts
        )
        
        # Count destinations
        remote_mask = ~local_mask
        for i in range(num_tokens):
            if local_mask[i]:
                all_shared_recv[src_rank, src_rank] += 1
            else:
                dst_rank = expanded_ids[i, -1].item() // (num_experts // world_size)
                all_shared_recv[src_rank, dst_rank] += 1
    
    print("Shared Expert Token Flow (rows=source, cols=destination):")
    print(f"{'Src\\Dst':<8}", end="")
    for dst in range(world_size):
        print(f"{'R'+str(dst):<8}", end="")
    print("Total")
    print("-" * (8 + 8 * world_size + 8))
    
    for src in range(world_size):
        print(f"R{src:<7}", end="")
        for dst in range(world_size):
            print(f"{all_shared_recv[src, dst].item():<8}", end="")
        print(f"{all_shared_recv[src].sum().item()}")
    
    # Total received by each rank
    print("-" * (8 + 8 * world_size + 8))
    print(f"{'Recv':<8}", end="")
    for dst in range(world_size):
        print(f"{all_shared_recv[:, dst].sum().item():<8}", end="")
    print()
    
    # Check tile utilization for received tokens
    print("\nShared tokens received per rank:")
    recv_per_rank = all_shared_recv.sum(dim=0)
    for rank in range(world_size):
        recv = recv_per_rank[rank].item()
        status = "✓" if recv == 0 or recv >= 128 else f"⚠️  ({recv}<128)"
        print(f"  Rank {rank}: {recv} tokens {status}")


if __name__ == "__main__":
    run_analysis()

