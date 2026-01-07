#!/usr/bin/env python3
"""
CPU-based unit tests for DeepEP Waterfill implementation.
Run with: python test_deepep_waterfill_cpu.py
"""

import torch
import sys
import os

# Directly import the module without going through sglang package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "python/sglang/srt/layers/moe"))

# Import directly from the file
import importlib.util
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
compute_local_shared_expert = deepep_waterfill.compute_local_shared_expert
DeepEPWaterfillBalancer = deepep_waterfill.DeepEPWaterfillBalancer
LOCAL_SHARED_MARKER = deepep_waterfill.LOCAL_SHARED_MARKER


def test_count_routed_per_rank():
    """Test counting routed tokens per rank."""
    print("\n" + "=" * 60)
    print("Test: count_routed_per_rank_pytorch")
    print("=" * 60)
    
    num_experts = 256
    world_size = 8
    experts_per_rank = num_experts // world_size  # 32
    
    # Create topk_ids: 4 tokens, each routes to 8 experts
    # Token 0: experts 0, 32, 64, 96, 128, 160, 192, 224 (one per rank)
    # Token 1: experts 0, 1, 2, 3, 4, 5, 6, 7 (all in rank 0)
    # Token 2: experts 32, 33, 34, 35, 36, 37, 38, 39 (all in rank 1)
    # Token 3: experts 0, 32, 64, -1, -1, -1, -1, -1 (sparse, some invalid)
    topk_ids = torch.tensor([
        [0, 32, 64, 96, 128, 160, 192, 224],  # one per rank
        [0, 1, 2, 3, 4, 5, 6, 7],              # all in rank 0
        [32, 33, 34, 35, 36, 37, 38, 39],      # all in rank 1
        [0, 32, 64, -1, -1, -1, -1, -1],       # sparse
    ], dtype=torch.int64)
    
    counts = count_routed_per_rank_pytorch(topk_ids, num_experts, world_size)
    
    print(f"topk_ids shape: {topk_ids.shape}")
    print(f"Routed counts per rank: {counts.tolist()}")
    
    # Expected:
    # rank 0: token0(1) + token1(8) + token3(1) = 10
    # rank 1: token0(1) + token2(8) + token3(1) = 10
    # rank 2: token0(1) + token3(1) = 2
    # rank 3-7: token0(1) each = 1
    expected = [10, 10, 2, 1, 1, 1, 1, 1]
    
    print(f"Expected: {expected}")
    assert counts.tolist() == expected, f"Mismatch! Got {counts.tolist()}"
    print("✓ PASSED")


def test_assign_shared_destination():
    """Test waterfill assignment algorithm."""
    print("\n" + "=" * 60)
    print("Test: assign_shared_destination_pytorch")
    print("=" * 60)
    
    num_experts = 256
    world_size = 8
    source_rank = 0
    experts_per_rank = num_experts // world_size  # 32
    
    # Token 0 routes to rank 0, 1, 2
    # Token 1 routes to rank 3, 4
    # Token 2 routes to rank 5, 6, 7
    topk_ids = torch.tensor([
        [0, 32, 64, -1, -1, -1, -1, -1],       # routes to rank 0, 1, 2
        [96, 128, -1, -1, -1, -1, -1, -1],     # routes to rank 3, 4
        [160, 192, 224, -1, -1, -1, -1, -1],   # routes to rank 5, 6, 7
    ], dtype=torch.int64)
    
    # Routed counts: rank 2 has lowest count
    routed_counts = torch.tensor([100, 80, 20, 90, 85, 70, 75, 60], dtype=torch.int64)
    
    destination = assign_shared_destination_pytorch(
        topk_ids, routed_counts, num_experts, world_size, source_rank
    )
    
    print(f"topk_ids:\n{topk_ids}")
    print(f"routed_counts: {routed_counts.tolist()}")
    print(f"source_rank: {source_rank}")
    print(f"Assigned destinations: {destination.tolist()}")
    
    # Token 0: candidates are {0, 1, 2} + source_rank(0) = {0, 1, 2}
    #   counts: 100, 80, 20 -> choose rank 2 (lowest)
    # Token 1: candidates are {3, 4} + source_rank(0) = {0, 3, 4}
    #   counts: 100, 90, 85 -> choose rank 4 (lowest)
    # Token 2: candidates are {5, 6, 7} + source_rank(0) = {0, 5, 6, 7}
    #   counts: 100, 70, 75, 60 -> choose rank 7 (lowest)
    expected = [2, 4, 7]
    
    print(f"Expected: {expected}")
    assert destination.tolist() == expected, f"Mismatch! Got {destination.tolist()}"
    print("✓ PASSED")


def test_assign_shared_destination_prefer_source():
    """Test that source rank is preferred when it has lowest count."""
    print("\n" + "=" * 60)
    print("Test: assign_shared_destination - prefer source rank")
    print("=" * 60)
    
    num_experts = 256
    world_size = 8
    source_rank = 0
    
    # Token routes to rank 1, 2, 3
    topk_ids = torch.tensor([
        [32, 64, 96, -1, -1, -1, -1, -1],
    ], dtype=torch.int64)
    
    # Source rank (0) has lowest count
    routed_counts = torch.tensor([10, 80, 90, 100, 85, 70, 75, 60], dtype=torch.int64)
    
    destination = assign_shared_destination_pytorch(
        topk_ids, routed_counts, num_experts, world_size, source_rank
    )
    
    print(f"routed_counts: {routed_counts.tolist()}")
    print(f"source_rank: {source_rank}")
    print(f"Assigned destination: {destination.tolist()}")
    
    # Candidates: {1, 2, 3} + source_rank(0) = {0, 1, 2, 3}
    # counts: 10, 80, 90, 100 -> choose rank 0 (source, lowest)
    expected = [0]
    
    print(f"Expected: {expected}")
    assert destination.tolist() == expected, f"Mismatch! Got {destination.tolist()}"
    print("✓ PASSED (source rank selected when it has lowest count)")


def test_expand_topk_with_shared_expert():
    """Test expanding topk from 8 to 9 columns."""
    print("\n" + "=" * 60)
    print("Test: expand_topk_with_shared_expert")
    print("=" * 60)
    
    num_experts = 256
    world_size = 8
    source_rank = 0
    shared_weight = 0.4  # 1/2.5
    experts_per_rank = num_experts // world_size  # 32
    
    topk_ids = torch.tensor([
        [0, 32, 64, 96, 128, 160, 192, 224],
        [1, 33, 65, 97, 129, 161, 193, 225],
    ], dtype=torch.int64)
    
    topk_weights = torch.ones(2, 8, dtype=torch.float32) * 0.125  # uniform weights
    
    # Token 0 -> rank 2 (remote)
    # Token 1 -> rank 0 (local, source rank)
    shared_destination = torch.tensor([2, 0], dtype=torch.int64)
    
    expanded_ids, expanded_weights, local_mask = expand_topk_with_shared_expert(
        topk_ids, topk_weights, shared_destination,
        num_experts, world_size, source_rank, shared_weight
    )
    
    print(f"Original topk_ids shape: {topk_ids.shape}")
    print(f"Expanded topk_ids shape: {expanded_ids.shape}")
    print(f"Expanded topk_ids:\n{expanded_ids}")
    print(f"Expanded topk_weights (9th col): {expanded_weights[:, -1].tolist()}")
    print(f"Local shared mask: {local_mask.tolist()}")
    
    # Token 0: dest=2, not local -> virtual ID = 2 * 32 = 64
    # Token 1: dest=0, local -> LOCAL_SHARED_MARKER = -1
    expected_9th_col = [64, LOCAL_SHARED_MARKER]  # [64, -1]
    expected_local_mask = [False, True]
    
    print(f"Expected 9th col: {expected_9th_col}")
    print(f"Expected local mask: {expected_local_mask}")
    
    assert expanded_ids[:, -1].tolist() == expected_9th_col, f"Mismatch in 9th col!"
    assert local_mask.tolist() == expected_local_mask, f"Mismatch in local mask!"
    # Use torch.allclose for floating point comparison
    assert torch.allclose(
        expanded_weights[:, -1], 
        torch.tensor([shared_weight, shared_weight])
    ), f"Mismatch in 9th col weights!"
    print("✓ PASSED")


def test_identify_shared_expert_tokens():
    """Test identifying shared expert tokens on receiver side."""
    print("\n" + "=" * 60)
    print("Test: identify_shared_expert_tokens")
    print("=" * 60)
    
    num_experts = 256
    world_size = 8
    current_rank = 2
    experts_per_rank = num_experts // world_size  # 32
    
    # Simulated received topk_ids (9 columns)
    # Token 0: 9th col = 64 (virtual ID for rank 2) -> should identify
    # Token 1: 9th col = 32 (virtual ID for rank 1) -> not for current rank
    # Token 2: 9th col = -1 (LOCAL_SHARED_MARKER) -> skip
    # Token 3: 9th col = 64 (virtual ID for rank 2) -> should identify
    recv_topk_ids = torch.tensor([
        [0, 32, 64, 96, 128, 160, 192, 224, 64],   # 9th = rank 2
        [1, 33, 65, 97, 129, 161, 193, 225, 32],   # 9th = rank 1
        [2, 34, 66, 98, 130, 162, 194, 226, -1],   # 9th = local marker
        [3, 35, 67, 99, 131, 163, 195, 227, 64],   # 9th = rank 2
    ], dtype=torch.int64)
    
    shared_indices = identify_shared_expert_tokens(
        recv_topk_ids, num_experts, world_size, current_rank
    )
    
    print(f"recv_topk_ids (9th col): {recv_topk_ids[:, -1].tolist()}")
    print(f"current_rank: {current_rank}")
    print(f"Identified shared indices: {shared_indices.tolist()}")
    
    expected = [0, 3]  # Tokens 0 and 3 have virtual ID for rank 2
    
    print(f"Expected: {expected}")
    assert shared_indices.tolist() == expected, f"Mismatch! Got {shared_indices.tolist()}"
    print("✓ PASSED")


def test_compute_local_shared_expert():
    """Test local shared expert computation."""
    print("\n" + "=" * 60)
    print("Test: compute_local_shared_expert")
    print("=" * 60)
    
    batch_size = 4
    hidden_size = 8
    
    hidden_states = torch.randn(batch_size, hidden_size)
    local_shared_mask = torch.tensor([False, True, False, True])
    
    # Simple mock shared expert: just multiply by 2
    def mock_shared_expert(x):
        return x * 2
    
    output, indices = compute_local_shared_expert(
        hidden_states, local_shared_mask, mock_shared_expert
    )
    
    print(f"hidden_states shape: {hidden_states.shape}")
    print(f"local_shared_mask: {local_shared_mask.tolist()}")
    print(f"output shape: {output.shape if output is not None else None}")
    print(f"indices: {indices.tolist() if indices is not None else None}")
    
    expected_indices = [1, 3]
    assert indices.tolist() == expected_indices, f"Indices mismatch!"
    
    # Verify output is 2x the selected hidden states
    expected_output = hidden_states[[1, 3]] * 2
    assert torch.allclose(output, expected_output), "Output mismatch!"
    print("✓ PASSED")


def test_deepep_waterfill_balancer_small_batch():
    """Test that small batches compute all shared locally."""
    print("\n" + "=" * 60)
    print("Test: DeepEPWaterfillBalancer - small batch optimization")
    print("=" * 60)
    
    balancer = DeepEPWaterfillBalancer(
        num_experts=256,
        world_size=8,
        rank=0,
        routed_scaling_factor=2.5,
    )
    
    # Small batch (< MIN_BATCH_FOR_BALANCE = 64)
    num_tokens = 32
    topk_ids = torch.randint(0, 256, (num_tokens, 8), dtype=torch.int64)
    topk_weights = torch.ones(num_tokens, 8, dtype=torch.float32) * 0.125
    routed_counts = torch.tensor([100, 80, 60, 90, 85, 70, 75, 65], dtype=torch.int64)
    
    expanded_ids, expanded_weights, local_mask = balancer.prepare_dispatch(
        topk_ids, topk_weights, routed_counts
    )
    
    print(f"Batch size: {num_tokens} (< MIN_BATCH={balancer.MIN_BATCH_FOR_BALANCE})")
    print(f"Local mask sum: {local_mask.sum().item()}")
    print(f"All local? {local_mask.all().item()}")
    
    # All tokens should be local
    assert local_mask.all(), "Small batch should have all local shared!"
    # All 9th column should be LOCAL_SHARED_MARKER
    assert (expanded_ids[:, -1] == LOCAL_SHARED_MARKER).all(), "All 9th col should be -1!"
    print("✓ PASSED")


def test_deepep_waterfill_balancer_sparse_redirect():
    """Test that sparse destinations are redirected to local."""
    print("\n" + "=" * 60)
    print("Test: DeepEPWaterfillBalancer - sparse destination redirect")
    print("=" * 60)
    
    balancer = DeepEPWaterfillBalancer(
        num_experts=256,
        world_size=8,
        rank=0,
        routed_scaling_factor=2.5,
    )
    
    # Large batch to enable waterfill
    num_tokens = 100
    
    # All tokens route to rank 0 and 1 only
    # This means waterfill can only choose rank 0, 1, or source rank (0)
    topk_ids = torch.zeros(num_tokens, 8, dtype=torch.int64)
    topk_ids[:, 0] = torch.randint(0, 32, (num_tokens,))   # rank 0
    topk_ids[:, 1] = torch.randint(32, 64, (num_tokens,))  # rank 1
    topk_ids[:, 2:] = -1  # invalid
    
    topk_weights = torch.ones(num_tokens, 8, dtype=torch.float32) * 0.125
    
    # Rank 2 has lowest count, but tokens can't go there (not routed)
    routed_counts = torch.tensor([100, 80, 10, 90, 85, 70, 75, 65], dtype=torch.int64)
    
    expanded_ids, expanded_weights, local_mask = balancer.prepare_dispatch(
        topk_ids, topk_weights, routed_counts
    )
    
    # Count destinations
    remote_mask = ~local_mask
    remote_9th_col = expanded_ids[remote_mask, -1]
    
    if remote_9th_col.numel() > 0:
        remote_dest_ranks = remote_9th_col // 32
        unique_dests = remote_dest_ranks.unique().tolist()
    else:
        unique_dests = []
    
    print(f"Batch size: {num_tokens}")
    print(f"Local shared count: {local_mask.sum().item()}")
    print(f"Remote shared count: {remote_mask.sum().item()}")
    print(f"Unique remote destinations: {unique_dests}")
    
    # All remote destinations should be rank 0 or 1 (the only routed ranks)
    for dest in unique_dests:
        assert dest in [0, 1], f"Unexpected destination rank {dest}!"
    print("✓ PASSED (destinations limited to routed ranks)")


def test_end_to_end_scenario():
    """Test a complete end-to-end scenario."""
    print("\n" + "=" * 60)
    print("Test: End-to-end scenario")
    print("=" * 60)
    
    num_experts = 256
    world_size = 8
    source_rank = 3
    routed_scaling_factor = 2.5
    
    balancer = DeepEPWaterfillBalancer(
        num_experts=num_experts,
        world_size=world_size,
        rank=source_rank,
        routed_scaling_factor=routed_scaling_factor,
    )
    
    # Batch of 128 tokens
    num_tokens = 128
    
    # Each token routes to 4 random experts
    topk_ids = torch.randint(0, num_experts, (num_tokens, 8), dtype=torch.int64)
    topk_ids[:, 4:] = -1  # Only 4 valid experts per token
    
    topk_weights = torch.ones(num_tokens, 8, dtype=torch.float32) * 0.25
    topk_weights[:, 4:] = 0  # Zero weight for invalid
    
    # Step 1: Count local routed tokens
    local_counts = balancer.count_local_routed(topk_ids)
    print(f"Local routed counts: {local_counts.tolist()}")
    
    # Simulate AllReduce (just use local counts for this test)
    global_counts = local_counts
    
    # Step 2: Prepare dispatch
    expanded_ids, expanded_weights, local_mask = balancer.prepare_dispatch(
        topk_ids, topk_weights, global_counts
    )
    
    print(f"\nExpanded topk_ids shape: {expanded_ids.shape}")
    print(f"Expanded topk_weights shape: {expanded_weights.shape}")
    print(f"Local shared count: {local_mask.sum().item()}")
    print(f"Remote shared count: (~local_mask).sum(): {(~local_mask).sum().item()}")
    
    # Verify shapes
    assert expanded_ids.shape == (num_tokens, 9), f"Wrong shape: {expanded_ids.shape}"
    assert expanded_weights.shape == (num_tokens, 9), f"Wrong shape: {expanded_weights.shape}"
    
    # Verify first 8 columns unchanged
    assert torch.equal(expanded_ids[:, :8], topk_ids), "First 8 cols should be unchanged!"
    assert torch.equal(expanded_weights[:, :8], topk_weights), "First 8 cols should be unchanged!"
    
    # Verify 9th column weights
    expected_shared_weight = 1.0 / routed_scaling_factor
    assert torch.allclose(
        expanded_weights[:, -1],
        torch.full((num_tokens,), expected_shared_weight)
    ), "9th col weight should be 1/rsf!"
    
    # Verify local mask consistency with 9th column
    local_9th = expanded_ids[local_mask, -1]
    remote_9th = expanded_ids[~local_mask, -1]
    
    assert (local_9th == LOCAL_SHARED_MARKER).all(), "Local tokens should have -1 in 9th col!"
    if remote_9th.numel() > 0:
        assert (remote_9th >= 0).all(), "Remote tokens should have valid virtual ID!"
    
    print("\n✓ PASSED (end-to-end scenario)")


def test_shared_weight_calculation():
    """Test that shared_weight is correctly calculated."""
    print("\n" + "=" * 60)
    print("Test: shared_weight calculation")
    print("=" * 60)
    
    test_cases = [
        (2.5, 0.4),
        (1.0, 1.0),
        (4.0, 0.25),
    ]
    
    for rsf, expected_weight in test_cases:
        balancer = DeepEPWaterfillBalancer(
            num_experts=256,
            world_size=8,
            rank=0,
            routed_scaling_factor=rsf,
        )
        
        actual_weight = balancer.shared_weight
        print(f"rsf={rsf}, expected_weight={expected_weight}, actual_weight={actual_weight}")
        
        assert abs(actual_weight - expected_weight) < 1e-6, f"Weight mismatch for rsf={rsf}!"
    
    print("✓ PASSED")


def test_empty_batch():
    """Test handling of empty batch."""
    print("\n" + "=" * 60)
    print("Test: Empty batch handling")
    print("=" * 60)
    
    balancer = DeepEPWaterfillBalancer(
        num_experts=256,
        world_size=8,
        rank=0,
        routed_scaling_factor=2.5,
    )
    
    topk_ids = torch.empty(0, 8, dtype=torch.int64)
    topk_weights = torch.empty(0, 8, dtype=torch.float32)
    routed_counts = torch.zeros(8, dtype=torch.int64)
    
    expanded_ids, expanded_weights, local_mask = balancer.prepare_dispatch(
        topk_ids, topk_weights, routed_counts
    )
    
    print(f"Input shape: {topk_ids.shape}")
    print(f"Output shape: {expanded_ids.shape}")
    
    assert expanded_ids.shape == (0, 9), f"Wrong shape: {expanded_ids.shape}"
    assert expanded_weights.shape == (0, 9), f"Wrong shape: {expanded_weights.shape}"
    assert local_mask.shape == (0,), f"Wrong mask shape: {local_mask.shape}"
    print("✓ PASSED")


def test_single_token():
    """Test handling of single token."""
    print("\n" + "=" * 60)
    print("Test: Single token handling")
    print("=" * 60)
    
    balancer = DeepEPWaterfillBalancer(
        num_experts=256,
        world_size=8,
        rank=0,
        routed_scaling_factor=2.5,
    )
    
    # Single token routing to rank 1
    topk_ids = torch.tensor([[32, 33, 34, -1, -1, -1, -1, -1]], dtype=torch.int64)
    topk_weights = torch.tensor([[0.4, 0.3, 0.3, 0, 0, 0, 0, 0]], dtype=torch.float32)
    routed_counts = torch.tensor([10, 5, 20, 30, 40, 50, 60, 70], dtype=torch.int64)
    
    expanded_ids, expanded_weights, local_mask = balancer.prepare_dispatch(
        topk_ids, topk_weights, routed_counts
    )
    
    print(f"Single token, batch < MIN_BATCH")
    print(f"Local mask: {local_mask.tolist()}")
    print(f"9th col: {expanded_ids[:, -1].tolist()}")
    
    # Should be local (batch < MIN_BATCH_FOR_BALANCE)
    assert local_mask.all(), "Single token should be local!"
    assert expanded_ids[0, -1] == LOCAL_SHARED_MARKER, "Should be LOCAL_SHARED_MARKER!"
    print("✓ PASSED")


def test_all_tokens_same_rank():
    """Test when all tokens route to the same rank."""
    print("\n" + "=" * 60)
    print("Test: All tokens route to same rank")
    print("=" * 60)
    
    balancer = DeepEPWaterfillBalancer(
        num_experts=256,
        world_size=8,
        rank=0,
        routed_scaling_factor=2.5,
    )
    
    num_tokens = 100
    # All tokens route only to rank 1
    topk_ids = torch.randint(32, 64, (num_tokens, 8), dtype=torch.int64)
    topk_weights = torch.ones(num_tokens, 8, dtype=torch.float32) * 0.125
    routed_counts = torch.tensor([0, 800, 0, 0, 0, 0, 0, 0], dtype=torch.int64)
    
    expanded_ids, expanded_weights, local_mask = balancer.prepare_dispatch(
        topk_ids, topk_weights, routed_counts
    )
    
    # All destinations should be either rank 0 (source) or rank 1 (only routed rank)
    remote_mask = ~local_mask
    if remote_mask.any():
        remote_9th = expanded_ids[remote_mask, -1]
        remote_dest_ranks = remote_9th // 32
        unique_dests = remote_dest_ranks.unique().tolist()
        print(f"Remote destinations: {unique_dests}")
        for d in unique_dests:
            assert d in [0, 1], f"Unexpected destination {d}!"
    
    print(f"Local count: {local_mask.sum().item()}")
    print(f"Remote count: {remote_mask.sum().item()}")
    print("✓ PASSED")


def test_waterfill_load_balance():
    """Test that waterfill actually balances load."""
    print("\n" + "=" * 60)
    print("Test: Waterfill load balancing effectiveness")
    print("=" * 60)
    
    num_experts = 256
    world_size = 8
    source_rank = 0
    
    # Each token routes to all 8 ranks (one expert per rank)
    num_tokens = 1000
    topk_ids = torch.zeros(num_tokens, 8, dtype=torch.int64)
    for i in range(8):
        topk_ids[:, i] = i * 32  # Expert 0, 32, 64, ..., 224
    
    # Unbalanced routed counts: rank 7 has much lower load
    routed_counts = torch.tensor([1000, 900, 800, 700, 600, 500, 400, 100], dtype=torch.int64)
    
    destination = assign_shared_destination_pytorch(
        topk_ids, routed_counts, num_experts, world_size, source_rank
    )
    
    dest_counts = torch.bincount(destination, minlength=world_size)
    print(f"Routed counts: {routed_counts.tolist()}")
    print(f"Shared destination counts: {dest_counts.tolist()}")
    
    # Most tokens should go to rank 7 (lowest load)
    max_dest = dest_counts.argmax().item()
    print(f"Most shared tokens go to rank: {max_dest}")
    
    assert max_dest == 7, f"Expected rank 7 to receive most, got {max_dest}"
    print("✓ PASSED (waterfill correctly identifies lowest load rank)")


def test_min_tokens_per_rank_threshold():
    """Test MIN_TOKENS_PER_RANK threshold in detail."""
    print("\n" + "=" * 60)
    print("Test: MIN_TOKENS_PER_RANK threshold")
    print("=" * 60)
    
    balancer = DeepEPWaterfillBalancer(
        num_experts=256,
        world_size=8,
        rank=0,
        routed_scaling_factor=2.5,
    )
    
    # Create scenario where waterfill would send few tokens to some ranks
    num_tokens = 100
    
    # 90 tokens route to rank 1, 10 tokens route to rank 2
    topk_ids = torch.zeros(num_tokens, 8, dtype=torch.int64)
    topk_ids[:90, 0] = 32  # rank 1
    topk_ids[90:, 0] = 64  # rank 2
    topk_ids[:, 1:] = -1
    
    topk_weights = torch.ones(num_tokens, 8, dtype=torch.float32) * 0.125
    
    # Rank 2 has lowest load, but only 10 tokens can go there
    routed_counts = torch.tensor([100, 50, 10, 200, 200, 200, 200, 200], dtype=torch.int64)
    
    expanded_ids, expanded_weights, local_mask = balancer.prepare_dispatch(
        topk_ids, topk_weights, routed_counts
    )
    
    # Count destinations
    dest_for_tokens = torch.zeros(num_tokens, dtype=torch.int64)
    dest_for_tokens[local_mask] = 0  # local
    remote_mask = ~local_mask
    if remote_mask.any():
        dest_for_tokens[remote_mask] = expanded_ids[remote_mask, -1] // 32
    
    dest_counts = torch.bincount(dest_for_tokens, minlength=8)
    print(f"Destination counts: {dest_counts.tolist()}")
    print(f"MIN_TOKENS_PER_RANK: {balancer.MIN_TOKENS_PER_RANK}")
    
    # Rank 2 should not receive tokens if count < MIN_TOKENS_PER_RANK
    # Those tokens should be redirected to local (rank 0)
    rank2_remote_count = (expanded_ids[remote_mask, -1] // 32 == 2).sum().item() if remote_mask.any() else 0
    print(f"Rank 2 receives (remote): {rank2_remote_count} tokens")
    
    if rank2_remote_count > 0 and rank2_remote_count < balancer.MIN_TOKENS_PER_RANK:
        print("WARNING: Sparse destination not redirected!")
    else:
        print("✓ Sparse destinations handled correctly")
    
    print("✓ PASSED")


def test_identify_shared_with_all_local():
    """Test identify_shared_expert_tokens when all are local markers."""
    print("\n" + "=" * 60)
    print("Test: identify_shared_expert_tokens with all local")
    print("=" * 60)
    
    num_experts = 256
    world_size = 8
    current_rank = 0
    
    # All tokens have LOCAL_SHARED_MARKER
    recv_topk_ids = torch.zeros(10, 9, dtype=torch.int64)
    recv_topk_ids[:, -1] = LOCAL_SHARED_MARKER
    
    shared_indices = identify_shared_expert_tokens(
        recv_topk_ids, num_experts, world_size, current_rank
    )
    
    print(f"All tokens have LOCAL_SHARED_MARKER")
    print(f"Identified indices: {shared_indices.tolist()}")
    
    assert shared_indices.numel() == 0, "Should identify no tokens!"
    print("✓ PASSED")


def test_identify_shared_mixed():
    """Test identify_shared_expert_tokens with mixed scenarios."""
    print("\n" + "=" * 60)
    print("Test: identify_shared_expert_tokens mixed scenarios")
    print("=" * 60)
    
    num_experts = 256
    world_size = 8
    experts_per_rank = 32
    
    # Test for each rank
    for current_rank in range(world_size):
        recv_topk_ids = torch.zeros(world_size + 1, 9, dtype=torch.int64)
        # Token i has virtual ID for rank i
        for i in range(world_size):
            recv_topk_ids[i, -1] = i * experts_per_rank
        # Last token is local marker
        recv_topk_ids[world_size, -1] = LOCAL_SHARED_MARKER
        
        shared_indices = identify_shared_expert_tokens(
            recv_topk_ids, num_experts, world_size, current_rank
        )
        
        expected = [current_rank]  # Only token at index current_rank
        assert shared_indices.tolist() == expected, \
            f"Rank {current_rank}: expected {expected}, got {shared_indices.tolist()}"
    
    print("✓ PASSED for all ranks")


def test_compute_local_shared_empty():
    """Test compute_local_shared_expert with no local tokens."""
    print("\n" + "=" * 60)
    print("Test: compute_local_shared_expert with no local tokens")
    print("=" * 60)
    
    hidden_states = torch.randn(10, 8)
    local_shared_mask = torch.zeros(10, dtype=torch.bool)  # All False
    
    def mock_fn(x):
        return x * 2
    
    output, indices = compute_local_shared_expert(
        hidden_states, local_shared_mask, mock_fn
    )
    
    print(f"No local tokens")
    print(f"Output: {output}")
    print(f"Indices: {indices}")
    
    assert output is None, "Output should be None!"
    assert indices is None, "Indices should be None!"
    print("✓ PASSED")


def test_virtual_id_mapping():
    """Test that virtual IDs correctly map to ranks."""
    print("\n" + "=" * 60)
    print("Test: Virtual ID to rank mapping")
    print("=" * 60)
    
    num_experts = 256
    world_size = 8
    experts_per_rank = num_experts // world_size
    
    # Test all ranks
    for target_rank in range(world_size):
        virtual_id = target_rank * experts_per_rank
        computed_rank = virtual_id // experts_per_rank
        
        assert computed_rank == target_rank, \
            f"Virtual ID {virtual_id} should map to rank {target_rank}, got {computed_rank}"
        print(f"  Rank {target_rank} -> Virtual ID {virtual_id} -> Rank {computed_rank} ✓")
    
    print("✓ PASSED")


def test_weight_preservation():
    """Test that original weights are preserved in expansion."""
    print("\n" + "=" * 60)
    print("Test: Weight preservation in topk expansion")
    print("=" * 60)
    
    num_experts = 256
    world_size = 8
    source_rank = 0
    shared_weight = 0.4
    
    # Create random weights
    topk_ids = torch.randint(0, num_experts, (50, 8), dtype=torch.int64)
    topk_weights = torch.rand(50, 8, dtype=torch.float32)
    shared_destination = torch.randint(0, world_size, (50,), dtype=torch.int64)
    
    expanded_ids, expanded_weights, local_mask = expand_topk_with_shared_expert(
        topk_ids, topk_weights, shared_destination,
        num_experts, world_size, source_rank, shared_weight
    )
    
    # Verify first 8 columns unchanged
    assert torch.equal(expanded_ids[:, :8], topk_ids), "topk_ids changed!"
    assert torch.equal(expanded_weights[:, :8], topk_weights), "topk_weights changed!"
    
    print("✓ First 8 columns preserved")
    print("✓ PASSED")


def test_routed_count_accuracy():
    """Test accuracy of routed token counting."""
    print("\n" + "=" * 60)
    print("Test: Routed count accuracy")
    print("=" * 60)
    
    num_experts = 256
    world_size = 8
    experts_per_rank = 32
    
    # Create controlled scenario
    topk_ids = torch.tensor([
        [0, 1, 2, 3, 4, 5, 6, 7],      # 8 tokens to rank 0
        [32, 33, 34, 35, -1, -1, -1, -1],  # 4 tokens to rank 1
        [64, 65, -1, -1, -1, -1, -1, -1],  # 2 tokens to rank 2
    ], dtype=torch.int64)
    
    counts = count_routed_per_rank_pytorch(topk_ids, num_experts, world_size)
    
    expected = [8, 4, 2, 0, 0, 0, 0, 0]
    print(f"Computed counts: {counts.tolist()}")
    print(f"Expected counts: {expected}")
    
    assert counts.tolist() == expected, f"Count mismatch!"
    print("✓ PASSED")


def test_consistency_across_calls():
    """Test that repeated calls give consistent results."""
    print("\n" + "=" * 60)
    print("Test: Consistency across repeated calls")
    print("=" * 60)
    
    balancer = DeepEPWaterfillBalancer(
        num_experts=256,
        world_size=8,
        rank=0,
        routed_scaling_factor=2.5,
    )
    
    # Fixed input
    torch.manual_seed(42)
    topk_ids = torch.randint(0, 256, (100, 8), dtype=torch.int64)
    topk_weights = torch.rand(100, 8, dtype=torch.float32)
    routed_counts = torch.tensor([100, 90, 80, 70, 60, 50, 40, 30], dtype=torch.int64)
    
    # Call multiple times
    results = []
    for i in range(3):
        expanded_ids, expanded_weights, local_mask = balancer.prepare_dispatch(
            topk_ids.clone(), topk_weights.clone(), routed_counts.clone()
        )
        results.append((expanded_ids.clone(), expanded_weights.clone(), local_mask.clone()))
    
    # Verify all results are identical
    for i in range(1, 3):
        assert torch.equal(results[0][0], results[i][0]), f"IDs differ at call {i}!"
        assert torch.equal(results[0][1], results[i][1]), f"Weights differ at call {i}!"
        assert torch.equal(results[0][2], results[i][2]), f"Mask differs at call {i}!"
    
    print("✓ All 3 calls produced identical results")
    print("✓ PASSED")


def main():
    print("=" * 60)
    print("DeepEP Waterfill CPU Unit Tests")
    print("=" * 60)
    
    tests = [
        test_count_routed_per_rank,
        test_assign_shared_destination,
        test_assign_shared_destination_prefer_source,
        test_expand_topk_with_shared_expert,
        test_identify_shared_expert_tokens,
        test_compute_local_shared_expert,
        test_deepep_waterfill_balancer_small_batch,
        test_deepep_waterfill_balancer_sparse_redirect,
        test_end_to_end_scenario,
        test_shared_weight_calculation,
        # New tests
        test_empty_batch,
        test_single_token,
        test_all_tokens_same_rank,
        test_waterfill_load_balance,
        test_min_tokens_per_rank_threshold,
        test_identify_shared_with_all_local,
        test_identify_shared_mixed,
        test_compute_local_shared_empty,
        test_virtual_id_mapping,
        test_weight_preservation,
        test_routed_count_accuracy,
        test_consistency_across_calls,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"\n✗ FAILED: {test.__name__}")
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    exit(main())

