"""Phase 5: P1-2 Capture-index semantic proof.

Tests that capture-layer ordering is preserved across PP partitions
using synthetic layers where each layer writes a unique identifiable value.
"""
from __future__ import annotations

import os
import sys
import torch
import torch.nn as nn

sys.path.insert(0, "/home/liang/sglang/python")

from sglang.srt.speculative.glm52_eagle3_pp import (
    build_layer_to_slot_map,
    build_slot_ownership_map,
    get_local_capture_layers,
    allocate_packed_aux_buffer,
    pack_aux_into_buffer,
    unpack_aux_from_buffer,
    validate_capture_layers,
)


class SyntheticLayer(nn.Module):
    def __init__(self, layer_id: int, hidden_size: int):
        super().__init__()
        self.layer_id = layer_id
        self.bias = nn.Parameter(torch.full((hidden_size,), float(layer_id * 1000)))
    def forward(self, x):
        return x + self.bias


def run_full_model(num_layers, hidden_size, num_tokens, capture_layers):
    """Run the full model (no PP) and return hidden + aux list."""
    layers = nn.ModuleList([SyntheticLayer(i, hidden_size) for i in range(num_layers)])
    hidden = torch.zeros(num_tokens, hidden_size, dtype=torch.float32)
    aux = []
    for i, layer in enumerate(layers):
        if i in capture_layers:
            aux.append(hidden.clone())
        hidden = layer(hidden)
    return hidden, aux


def run_pp_model(num_layers, hidden_size, num_tokens, capture_layers, partition):
    """Run model in PP stages, properly chaining hidden states."""
    pp_size = len(partition)
    layer_to_slot = build_layer_to_slot_map(capture_layers)
    slot_ownership = build_slot_ownership_map(capture_layers, pp_size, num_layers)
    num_capture = len(capture_layers)
    
    packed_aux = torch.zeros(num_tokens, num_capture, hidden_size, dtype=torch.float32)
    hidden = torch.zeros(num_tokens, hidden_size, dtype=torch.float32)
    
    for pp_rank in range(pp_size):
        start = sum(partition[:pp_rank])
        end = start + partition[pp_rank]
        local_capture = get_local_capture_layers(capture_layers, start, end)
        
        layers = nn.ModuleList([SyntheticLayer(i, hidden_size) for i in range(start, end)])
        local_aux = []
        
        for i, layer in enumerate(layers):
            global_id = start + i
            if global_id in capture_layers:
                local_aux.append(hidden.clone())
            hidden = layer(hidden)
        
        # Pack local aux into the shared buffer
        if local_aux:
            pack_aux_into_buffer(packed_aux, local_aux, local_capture, layer_to_slot)
    
    # Unpack
    result = unpack_aux_from_buffer(
        packed_aux, capture_layers, layer_to_slot, slot_ownership,
        get_local_capture_layers(capture_layers, 0, num_layers),
        pp_rank=pp_size - 1, pp_size=pp_size,
    )
    return hidden, result


def test_capture_ordering():
    """Test that PP and non-PP capture ordering are equivalent."""
    num_layers = 10
    hidden_size = 32
    num_tokens = 4
    capture_layers = [2, 5, 8]
    
    ref_hidden, ref_aux = run_full_model(num_layers, hidden_size, num_tokens, capture_layers)
    
    partitions = [
        [5, 5], [1, 9], [3, 7], [7, 3], [9, 1], [4, 6], [6, 4],
    ]
    
    for partition in partitions:
        pp_hidden, pp_aux = run_pp_model(num_layers, hidden_size, num_tokens, capture_layers, partition)
        
        assert len(pp_aux) == len(ref_aux), f"Partition {partition}: count mismatch"
        
        for i, (ref, pp) in enumerate(zip(ref_aux, pp_aux)):
            assert torch.equal(ref, pp), (
                f"Partition {partition}, capture layer {capture_layers[i]}: "
                f"mismatch. ref[0,0]={ref[0,0].item()}, pp[0,0]={pp[0,0].item()}"
            )
        
        # Also verify final hidden state
        assert torch.equal(ref_hidden, pp_hidden), (
            f"Partition {partition}: final hidden mismatch"
        )
        
        print(f"  Partition {partition}: PASSED")


def test_capture_at_boundary():
    """Test capture layers at PP boundaries."""
    num_layers = 10
    hidden_size = 32
    num_tokens = 4
    
    test_cases = [
        ([0, 5, 9], [5, 5]),
        ([5, 6], [5, 5]),
        ([1, 9], [1, 9]),
        ([1, 2], [3, 7]),
        ([8, 9], [7, 3]),
        ([0, 1], [2, 8]),
        ([9], [9, 1]),
        ([0], [1, 9]),
    ]
    
    for capture_layers, partition in test_cases:
        ref_hidden, ref_aux = run_full_model(num_layers, hidden_size, num_tokens, capture_layers)
        pp_hidden, pp_aux = run_pp_model(num_layers, hidden_size, num_tokens, capture_layers, partition)
        
        assert len(pp_aux) == len(ref_aux)
        for i, (ref, pp) in enumerate(zip(ref_aux, pp_aux)):
            assert torch.equal(ref, pp), (
            f"Capture {capture_layers}, partition {partition}, layer {capture_layers[i]}: "
                f"ref={ref[0,0].item()}, pp={pp[0,0].item()}"
            )
        print(f"  Capture {capture_layers}, partition {partition}: PASSED")


def test_off_by_one():
    """Explicitly test for local/global index off-by-one errors."""
    num_layers = 6
    hidden_size = 16
    num_tokens = 2
    capture_layers = [1, 3, 5]
    
    ref_hidden, ref_aux = run_full_model(num_layers, hidden_size, num_tokens, capture_layers)
    pp_hidden, pp_aux = run_pp_model(num_layers, hidden_size, num_tokens, capture_layers, [3, 3])
    
    for i, (ref, pp) in enumerate(zip(ref_aux, pp_aux)):
        # Capture layer 1 sees output of layer 0 = 0
        # Capture layer 3 sees output of layer 2 = 0+1000+2000 = 3000
        # Capture layer 5 sees output of layer 4 = 0+1000+2000+3000+4000 = 10000
        expected_val = float(sum(j * 1000 for j in range(capture_layers[i])))
        actual_val = pp[0, 0].item()
        assert abs(actual_val - expected_val) < 1e-4, (
            f"Capture layer {capture_layers[i]}: expected {expected_val}, got {actual_val}"
        )
    
    print("  Off-by-one test PASSED")


def test_capture_layer_validation():
    """Test validate_capture_layers catches errors."""
    try: validate_capture_layers([2, 2, 5], 10, 2, 0, 5, 32); assert False
    except ValueError: pass
    try: validate_capture_layers([5, 2, 8], 10, 2, 0, 5, 32); assert False
    except ValueError: pass
    try: validate_capture_layers([2, 5, 10], 10, 2, 0, 5, 32); assert False
    except ValueError: pass
    try: validate_capture_layers([], 10, 2, 0, 5, 32); assert False
    except ValueError: pass
    
    ownership = validate_capture_layers([2, 5, 8], 10, 2, 0, 5, 32)
    assert ownership[2] == 0
    assert ownership[5] == 1
    assert ownership[8] == 1
    
    print("  Capture layer validation PASSED")


def test_4_layer_partitions():
    """Test 4-layer model with different partitions (tiny model)."""
    num_layers = 4
    hidden_size = 16
    num_tokens = 2
    capture_layers = [1, 2]
    
    ref_hidden, ref_aux = run_full_model(num_layers, hidden_size, num_tokens, capture_layers)
    
    for partition in [[1, 3], [2, 2], [3, 1]]:
        pp_hidden, pp_aux = run_pp_model(num_layers, hidden_size, num_tokens, capture_layers, partition)
        for i, (ref, pp) in enumerate(zip(ref_aux, pp_aux)):
            assert torch.equal(ref, pp), (
                f"4-layer partition {partition}, capture {capture_layers[i]}: "
                f"ref={ref[0,0].item()}, pp={pp[0,0].item()}"
            )
        print(f"  4-layer partition {partition}: PASSED")


if __name__ == "__main__":
    print("=== Phase 5: P1-2 Capture-Index Semantic Proof ===")
    test_capture_ordering()
    test_capture_at_boundary()
    test_off_by_one()
    test_capture_layer_validation()
    test_4_layer_partitions()
    print("\n=== All Phase 5 tests PASSED ===")
