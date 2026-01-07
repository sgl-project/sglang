#!/usr/bin/env python3
"""
Comprehensive test suite for DeepEP Waterfill implementation.
"""

import sys
import os

# Add sglang to path - only the specific module path
module_path = os.path.join(os.path.dirname(__file__), "python/sglang/srt/layers/moe")
sys.path.insert(0, module_path)

import torch

# Direct import
from deepep_waterfill import (
    count_routed_per_rank_pytorch,
    assign_shared_destination_pytorch,
    expand_topk_with_shared_expert,
    identify_shared_expert_tokens,
    compute_local_shared_expert,
    DeepEPWaterfillBalancer,
    LOCAL_SHARED_MARKER,
)


def print_test_header(name):
    print(f"\n{'='*60}")
    print(f"Test: {name}")
    print("=" * 60)


def print_pass():
    print("✓ PASSED")


def print_fail(msg):
    print(f"✗ FAILED: {msg}")
    return False


# ============== Test Functions ==============


def test_count_routed_per_rank():
    """Test that routed token counting is correct."""
    print_test_header("count_routed_per_rank_pytorch")
    
    num_experts = 256
    world_size = 8
    
    topk_ids = torch.tensor([
        [0, 32, 64],  # ranks 0, 1, 2
        [0, 1, 2],    # rank 0, 0, 0
        [-1, -1, -1], # invalid
    ], dtype=torch.int64)
    
    counts = count_routed_per_rank_pytorch(topk_ids, num_experts, world_size)
    expected = torch.tensor([4, 1, 1, 0, 0, 0, 0, 0], dtype=torch.int64)
    
    if torch.equal(counts, expected):
        print(f"Counts: {counts.tolist()}")
        print_pass()
        return True
    else:
        return print_fail(f"Expected {expected.tolist()}, got {counts.tolist()}")


def test_assign_shared_destination_basic():
    """Test basic waterfill assignment."""
    print_test_header("assign_shared_destination - basic")
    
    num_experts = 256
    world_size = 8
    source_rank = 0
    
    topk_ids = torch.tensor([
        [32, 64, 96, -1, -1, -1, -1, -1],  # ranks 1, 2, 3
    ], dtype=torch.int64)
    
    routed_counts = torch.tensor([100, 80, 20, 90, 85, 70, 75, 60], dtype=torch.int64)
    
    dest = assign_shared_destination_pytorch(
        topk_ids, routed_counts, num_experts, world_size, source_rank
    )
    
    expected = 2  # rank 2 has lowest count among candidates
    
    if dest[0].item() == expected:
        print(f"Destination: {dest[0].item()}")
        print_pass()
        return True
    else:
        return print_fail(f"Expected {expected}, got {dest[0].item()}")


def test_assign_shared_destination_source_rank():
    """Test that source rank can be selected when it has lowest count."""
    print_test_header("assign_shared_destination - prefer source rank")
    
    num_experts = 256
    world_size = 8
    source_rank = 0
    
    topk_ids = torch.tensor([
        [32, 64, 96, -1, -1, -1, -1, -1],
    ], dtype=torch.int64)
    
    routed_counts = torch.tensor([10, 80, 90, 100, 85, 70, 75, 60], dtype=torch.int64)
    
    dest = assign_shared_destination_pytorch(
        topk_ids, routed_counts, num_experts, world_size, source_rank
    )
    
    if dest[0].item() == source_rank:
        print(f"Source rank {source_rank} selected (count={routed_counts[source_rank].item()})")
        print_pass()
        return True
    else:
        return print_fail(f"Expected source rank {source_rank}, got {dest[0].item()}")


def test_expand_topk_local_marker():
    """Test that local shared experts get LOCAL_SHARED_MARKER."""
    print_test_header("expand_topk - local marker")
    
    num_experts = 256
    world_size = 8
    source_rank = 0
    experts_per_rank = 32
    shared_weight = 0.4
    
    topk_ids = torch.tensor([
        [0, 32, 64, -1, -1, -1, -1, -1],
        [1, 33, 65, -1, -1, -1, -1, -1],
    ], dtype=torch.int64)
    topk_weights = torch.ones(2, 8, dtype=torch.float32) * 0.125
    
    shared_destination = torch.tensor([source_rank, 2], dtype=torch.int64)
    
    expanded_ids, expanded_weights, local_mask = expand_topk_with_shared_expert(
        topk_ids, topk_weights, shared_destination,
        num_experts, world_size, source_rank, shared_weight
    )
    
    success = True
    
    if expanded_ids[0, -1].item() != LOCAL_SHARED_MARKER:
        print_fail(f"Token 0 should have LOCAL_SHARED_MARKER, got {expanded_ids[0, -1].item()}")
        success = False
    
    expected_virtual_id = 2 * experts_per_rank
    if expanded_ids[1, -1].item() != expected_virtual_id:
        print_fail(f"Token 1 should have virtual ID {expected_virtual_id}, got {expanded_ids[1, -1].item()}")
        success = False
    
    expected_mask = torch.tensor([True, False])
    if not torch.equal(local_mask, expected_mask):
        print_fail(f"Local mask mismatch")
        success = False
    
    if success:
        print(f"9th column: {expanded_ids[:, -1].tolist()}")
        print(f"Local mask: {local_mask.tolist()}")
        print_pass()
    
    return success


def test_identify_shared_expert_tokens():
    """Test identification of remote shared expert tokens."""
    print_test_header("identify_shared_expert_tokens")
    
    num_experts = 256
    world_size = 8
    current_rank = 2
    
    recv_topk_ids = torch.tensor([
        [0, 1, 2, 3, 4, 5, 6, 7, 64],   # rank 2
        [0, 1, 2, 3, 4, 5, 6, 7, 32],   # rank 1
        [0, 1, 2, 3, 4, 5, 6, 7, LOCAL_SHARED_MARKER],
        [0, 1, 2, 3, 4, 5, 6, 7, 64],   # rank 2
    ], dtype=torch.int64)
    
    indices = identify_shared_expert_tokens(
        recv_topk_ids, num_experts, world_size, current_rank
    )
    
    expected = torch.tensor([0, 3])
    
    if torch.equal(indices, expected):
        print(f"Identified: {indices.tolist()}")
        print_pass()
        return True
    else:
        return print_fail(f"Expected {expected.tolist()}, got {indices.tolist()}")


def test_virtual_id_to_rank_mapping():
    """Test virtual expert ID to rank mapping."""
    print_test_header("Virtual ID to rank mapping")
    
    num_experts = 256
    world_size = 8
    experts_per_rank = 32
    
    success = True
    
    for target_rank in range(world_size):
        virtual_id = target_rank * experts_per_rank
        recv_topk_ids = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, virtual_id]], dtype=torch.int64)
        
        for check_rank in range(world_size):
            indices = identify_shared_expert_tokens(recv_topk_ids, num_experts, world_size, check_rank)
            should_identify = (check_rank == target_rank)
            actually_identified = len(indices) > 0
            
            if should_identify != actually_identified:
                success = False
                print_fail(f"Mismatch for virtual_id={virtual_id}, check_rank={check_rank}")
        
        print(f"  Rank {target_rank} -> Virtual ID {virtual_id} ✓")
    
    if success:
        print_pass()
    return success


def test_min_batch_optimization():
    """Test small batch optimization."""
    print_test_header("MIN_BATCH_FOR_BALANCE optimization")
    
    balancer = DeepEPWaterfillBalancer(256, 8, 0, 2.5)
    
    batch_size = 32
    topk_ids = torch.randint(0, 256, (batch_size, 8), dtype=torch.int64)
    topk_weights = torch.rand(batch_size, 8, dtype=torch.float32)
    routed_counts = torch.randint(100, 200, (8,), dtype=torch.int64)
    
    _, _, local_mask = balancer.prepare_dispatch(topk_ids, topk_weights, routed_counts)
    
    if local_mask.all():
        print(f"Batch {batch_size} < MIN={balancer.MIN_BATCH_FOR_BALANCE}: all local ✓")
        print_pass()
        return True
    else:
        return print_fail(f"Local count: {local_mask.sum().item()}/{batch_size}")


def test_shared_weight_calculation():
    """Test shared weight = 1/rsf."""
    print_test_header("Shared weight = 1/rsf")
    
    test_cases = [(2.5, 0.4), (1.0, 1.0), (4.0, 0.25)]
    success = True
    
    for rsf, expected in test_cases:
        balancer = DeepEPWaterfillBalancer(256, 8, 0, rsf)
        if not torch.isclose(torch.tensor(balancer.shared_weight), torch.tensor(expected)):
            success = False
        else:
            print(f"  rsf={rsf} -> weight={balancer.shared_weight} ✓")
    
    if success:
        print_pass()
    return success


def test_empty_batch():
    """Test empty batch handling."""
    print_test_header("Empty batch handling")
    
    balancer = DeepEPWaterfillBalancer(256, 8, 0, 2.5)
    
    topk_ids = torch.empty(0, 8, dtype=torch.int64)
    topk_weights = torch.empty(0, 8, dtype=torch.float32)
    routed_counts = torch.zeros(8, dtype=torch.int64)
    
    expanded_ids, expanded_weights, local_mask = balancer.prepare_dispatch(
        topk_ids, topk_weights, routed_counts
    )
    
    if expanded_ids.shape == (0, 9):
        print(f"Shape: {expanded_ids.shape}")
        print_pass()
        return True
    else:
        return print_fail(f"Wrong shape: {expanded_ids.shape}")


def test_compute_local_shared_expert():
    """Test local shared expert computation."""
    print_test_header("compute_local_shared_expert")
    
    hidden_states = torch.randn(10, 128)
    local_mask = torch.tensor([False, True, False, True, True, False, False, True, False, False])
    
    def mock_fn(x):
        return x * 2
    
    output, indices = compute_local_shared_expert(hidden_states, local_mask, mock_fn)
    
    expected_indices = torch.tensor([1, 3, 4, 7])
    
    if output is None or indices is None:
        return print_fail("None returned")
    
    if not torch.equal(indices, expected_indices):
        return print_fail(f"Indices: {indices.tolist()}")
    
    expected_output = hidden_states[expected_indices] * 2
    if not torch.allclose(output, expected_output):
        return print_fail("Output values wrong")
    
    print(f"Indices: {indices.tolist()}")
    print_pass()
    return True


def test_no_local_tokens():
    """Test when no tokens are local."""
    print_test_header("No local tokens")
    
    hidden_states = torch.randn(10, 128)
    local_mask = torch.zeros(10, dtype=torch.bool)
    
    output, indices = compute_local_shared_expert(hidden_states, local_mask, lambda x: x)
    
    if output is None and indices is None:
        print("Returns (None, None) ✓")
        print_pass()
        return True
    else:
        return print_fail("Should return (None, None)")


def test_weights_preservation():
    """Test that original topk_weights are preserved."""
    print_test_header("Weights preservation")
    
    balancer = DeepEPWaterfillBalancer(256, 8, 0, 2.5)
    
    topk_ids = torch.randint(0, 256, (100, 8), dtype=torch.int64)
    topk_weights = torch.rand(100, 8, dtype=torch.float32)
    routed_counts = torch.randint(100, 200, (8,), dtype=torch.int64)
    
    expanded_ids, expanded_weights, _ = balancer.prepare_dispatch(
        topk_ids, topk_weights, routed_counts
    )
    
    if torch.equal(expanded_ids[:, :8], topk_ids) and torch.allclose(expanded_weights[:, :8], topk_weights):
        print("First 8 columns preserved ✓")
        print_pass()
        return True
    else:
        return print_fail("Columns modified")


def test_waterfill_effectiveness():
    """Test waterfill load balancing.
    
    Waterfill can only select from: source_rank OR ranks the token routes to.
    So we need tokens that route to multiple ranks including low-load ones.
    """
    print_test_header("Waterfill effectiveness")
    
    num_experts = 256
    world_size = 8
    num_tokens = 1024
    
    # High load on ranks 0, 1; low load on ranks 2, 7
    routed_counts = torch.tensor([1000, 900, 100, 500, 500, 500, 500, 100], dtype=torch.int64)
    
    # Tokens route to rank 0 (high load), rank 2 (low load), rank 7 (low load)
    topk_ids = torch.zeros(num_tokens, 8, dtype=torch.int64)
    topk_ids[:, 0] = torch.randint(0, 32, (num_tokens,))    # rank 0 (high load)
    topk_ids[:, 1] = torch.randint(64, 96, (num_tokens,))   # rank 2 (low load)
    topk_ids[:, 2] = torch.randint(224, 256, (num_tokens,)) # rank 7 (low load)
    topk_ids[:, 3:] = -1
    
    # Source rank = 0 (high load)
    # Candidates for each token: rank 0, 2, 7
    # Waterfill should prefer ranks 2 and 7 (lowest counts: 100)
    dest = assign_shared_destination_pytorch(topk_ids, routed_counts, num_experts, world_size, 0)
    dest_counts = torch.bincount(dest, minlength=world_size)
    
    print(f"Routed counts: {routed_counts.tolist()}")
    print(f"Shared dests:  {dest_counts.tolist()}")
    
    # Low load ranks (2, 7) should get most shared expert tokens
    low_load = dest_counts[2].item() + dest_counts[7].item()
    high_load = dest_counts[0].item()  # Only source rank 0 is high load candidate
    
    print(f"Low load ranks (2,7): {low_load}")
    print(f"High load rank (0): {high_load}")
    
    if low_load > high_load:
        print_pass()
        return True
    else:
        return print_fail(f"Low: {low_load}, High: {high_load}")


def test_invalid_expert_ids():
    """Test handling of -1 expert IDs."""
    print_test_header("Invalid expert IDs (-1)")
    
    topk_ids = torch.tensor([
        [0, -1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1],
        [32, 64, -1, -1, -1, -1, -1, -1],
    ], dtype=torch.int64)
    
    counts = count_routed_per_rank_pytorch(topk_ids, 256, 8)
    expected = torch.tensor([1, 1, 1, 0, 0, 0, 0, 0], dtype=torch.int64)
    
    if torch.equal(counts, expected):
        print(f"Counts: {counts.tolist()}")
        print_pass()
        return True
    else:
        return print_fail(f"Expected {expected.tolist()}, got {counts.tolist()}")


def test_large_batch_performance():
    """Test large batch performance."""
    print_test_header("Large batch performance")
    
    import time
    
    balancer = DeepEPWaterfillBalancer(256, 8, 0, 2.5)
    
    batch_size = 4096
    topk_ids = torch.randint(0, 256, (batch_size, 8), dtype=torch.int64)
    topk_weights = torch.rand(batch_size, 8, dtype=torch.float32)
    routed_counts = torch.randint(1000, 5000, (8,), dtype=torch.int64)
    
    start = time.time()
    _, _, _ = balancer.prepare_dispatch(topk_ids, topk_weights, routed_counts)
    elapsed = time.time() - start
    
    print(f"Batch: {batch_size}, Time: {elapsed*1000:.2f} ms")
    
    if elapsed < 1.0:
        print_pass()
        return True
    else:
        return print_fail(f"Too slow: {elapsed:.2f}s")


# ============== Main ==============


def main():
    print("=" * 60)
    print("DeepEP Waterfill Comprehensive Test Suite")
    print("=" * 60)
    
    tests = [
        test_count_routed_per_rank,
        test_assign_shared_destination_basic,
        test_assign_shared_destination_source_rank,
        test_expand_topk_local_marker,
        test_identify_shared_expert_tokens,
        test_virtual_id_to_rank_mapping,
        test_min_batch_optimization,
        test_shared_weight_calculation,
        test_empty_batch,
        test_compute_local_shared_expert,
        test_no_local_tokens,
        test_weights_preservation,
        test_waterfill_effectiveness,
        test_invalid_expert_ids,
        test_large_batch_performance,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"✗ EXCEPTION: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
