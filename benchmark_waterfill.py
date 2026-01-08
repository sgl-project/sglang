#!/usr/bin/env python3
"""Benchmark waterfill algorithm performance."""

import sys
import os
import time

module_path = os.path.join(os.path.dirname(__file__), "python/sglang/srt/layers/moe")
sys.path.insert(0, module_path)

import torch

from deepep_waterfill import (
    count_routed_per_rank_pytorch,
    assign_shared_destination_pytorch,
    expand_topk_with_shared_expert,
    DeepEPWaterfillBalancer,
)


def benchmark_function(fn, *args, warmup=5, repeat=100, **kwargs):
    """Benchmark a function."""
    # Warmup
    for _ in range(warmup):
        fn(*args, **kwargs)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    start = time.perf_counter()
    for _ in range(repeat):
        fn(*args, **kwargs)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    elapsed = (time.perf_counter() - start) / repeat
    return elapsed * 1000  # ms


def main():
    print("=" * 70)
    print("Waterfill Algorithm Performance Benchmark")
    print("=" * 70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")
    
    num_experts = 256
    world_size = 8
    topk = 8
    
    batch_sizes = [128, 512, 1024, 2048, 4096, 8192]
    
    # Benchmark each function
    print("-" * 70)
    print(f"{'Batch':<10} {'count_routed':<15} {'assign_dest':<15} {'expand_topk':<15} {'prepare_all':<15}")
    print(f"{'Size':<10} {'(ms)':<15} {'(ms)':<15} {'(ms)':<15} {'(ms)':<15}")
    print("-" * 70)
    
    for batch_size in batch_sizes:
        topk_ids = torch.randint(0, num_experts, (batch_size, topk), dtype=torch.int64, device=device)
        topk_weights = torch.rand(batch_size, topk, dtype=torch.float32, device=device)
        routed_counts = torch.randint(1000, 5000, (world_size,), dtype=torch.int64, device=device)
        
        # Benchmark count_routed_per_rank
        t_count = benchmark_function(
            count_routed_per_rank_pytorch, topk_ids, num_experts, world_size
        )
        
        # Benchmark assign_shared_destination
        t_assign = benchmark_function(
            assign_shared_destination_pytorch, topk_ids, routed_counts, num_experts, world_size, 0
        )
        
        # Benchmark expand_topk
        shared_dest = torch.randint(0, world_size, (batch_size,), dtype=torch.int64, device=device)
        t_expand = benchmark_function(
            expand_topk_with_shared_expert, topk_ids, topk_weights, shared_dest, 
            num_experts, world_size, 0, 0.4
        )
        
        # Benchmark full prepare_dispatch
        balancer = DeepEPWaterfillBalancer(num_experts, world_size, 0, 2.5)
        t_prepare = benchmark_function(
            balancer.prepare_dispatch, topk_ids, topk_weights, routed_counts
        )
        
        print(f"{batch_size:<10} {t_count:<15.4f} {t_assign:<15.4f} {t_expand:<15.4f} {t_prepare:<15.4f}")
    
    print("-" * 70)
    
    # Compare old vs new implementation
    print("\n" + "=" * 70)
    print("Optimization Comparison: Old (loop) vs New (vectorized)")
    print("=" * 70)
    
    def assign_shared_destination_old(topk_ids, routed_counts, num_experts, world_size, source_rank):
        """OLD implementation with for loop."""
        num_tokens = topk_ids.shape[0]
        topk = topk_ids.shape[1]
        experts_per_rank = num_experts // world_size
        device = topk_ids.device
        
        candidate_mask = torch.zeros(num_tokens, world_size, dtype=torch.bool, device=device)
        candidate_mask[:, source_rank] = True
        
        valid_mask = topk_ids >= 0
        rank_ids = torch.where(
            valid_mask,
            torch.clamp(topk_ids // experts_per_rank, 0, world_size - 1),
            torch.zeros_like(topk_ids),
        )
        
        # OLD: for loop (slow)
        for k in range(topk):
            token_indices = torch.arange(num_tokens, device=device)
            valid = valid_mask[:, k]
            ranks = rank_ids[:, k]
            candidate_mask[token_indices[valid], ranks[valid]] = True
        
        INF = routed_counts.max() + 1
        candidate_counts = torch.where(candidate_mask, routed_counts.unsqueeze(0), INF)
        return candidate_counts.argmin(dim=1).to(torch.int64)
    
    print(f"\n{'Batch':<10} {'Old (loop)':<15} {'New (vec)':<15} {'Speedup':<10}")
    print("-" * 50)
    
    for batch_size in batch_sizes:
        topk_ids = torch.randint(0, num_experts, (batch_size, topk), dtype=torch.int64, device=device)
        routed_counts = torch.randint(1000, 5000, (world_size,), dtype=torch.int64, device=device)
        
        t_old = benchmark_function(
            assign_shared_destination_old, topk_ids, routed_counts, num_experts, world_size, 0
        )
        t_new = benchmark_function(
            assign_shared_destination_pytorch, topk_ids, routed_counts, num_experts, world_size, 0
        )
        
        speedup = t_old / t_new
        print(f"{batch_size:<10} {t_old:<15.4f} {t_new:<15.4f} {speedup:<10.2f}x")


if __name__ == "__main__":
    main()

