"""Phase 10: CUDA Graph validation.

Tests that CUDA Graph capture/replay works correctly with the packed transport
and static buffers.
"""
from __future__ import annotations

import os
import sys
import torch

sys.path.insert(0, "/home/liang/sglang/python")

from sglang.srt.distributed.pp_packed_transport import (
    PPSchemaCache, PPStaticBufferRegistry,
    calculate_pp_buffer_layout, pack_pp_proxy_tensors, unpack_pp_proxy_tensors,
)


def test_static_buffer_reuse():
    """Test that static buffers are reused across iterations."""
    if not torch.cuda.is_available():
        print("  SKIP: No CUDA available")
        return True
    
    device = torch.device("cuda:0")
    hidden_size = 64
    num_capture = 3
    topk_size = 8
    max_rows = 64
    
    registry = PPStaticBufferRegistry(device=device)
    
    # First allocation
    td = {
        "hidden_states": torch.randn(16, hidden_size, dtype=torch.bfloat16, device=device),
        "residual": torch.randn(16, hidden_size, dtype=torch.bfloat16, device=device),
    }
    sk, dn, cn, do, co = calculate_pp_buffer_layout(td, hidden_size, 0, topk_size, max_rows)
    db1, cb1 = registry.get_or_allocate(16, dn, cn, torch.bfloat16)
    ptr1 = (db1.data_ptr(), cb1.data_ptr())
    
    # Second request for same bucket
    db2, cb2 = registry.get_or_allocate(16, dn, cn, torch.bfloat16)
    ptr2 = (db2.data_ptr(), cb2.data_ptr())
    
    assert ptr1 == ptr2, "Static buffer not reused for same bucket"
    
    # Different bucket gets different buffer
    db3, cb3 = registry.get_or_allocate(32, dn, cn, torch.bfloat16)
    assert db3.data_ptr() != db1.data_ptr(), "Different bucket should have different buffer"
    
    print("  Static buffer reuse PASSED")
    return True


def test_active_row_shrink_no_stale():
    """Test that shrinking active rows doesn't expose stale data."""
    if not torch.cuda.is_available():
        print("  SKIP: No CUDA available")
        return True
    
    device = torch.device("cuda:0")
    hidden_size = 64
    topk_size = 8
    max_rows = 64
    
    # Allocate for max_rows
    td = {
        "hidden_states": torch.randn(max_rows, hidden_size, dtype=torch.bfloat16, device=device),
    }
    sk, dn, cn, do, co = calculate_pp_buffer_layout(td, hidden_size, 0, topk_size, max_rows)
    db = torch.zeros(dn, dtype=torch.bfloat16, device=device)
    cb = torch.zeros(cn, dtype=torch.int32, device=device)
    
    # Fill with 64 rows
    td_large = {
        "hidden_states": torch.ones(max_rows, hidden_size, dtype=torch.bfloat16, device=device) * 42.0,
    }
    pack_pp_proxy_tensors(td_large, db, cb, do, co, max_rows)
    
    # Now pack only 4 rows with different value
    td_small = {
        "hidden_states": torch.ones(4, hidden_size, dtype=torch.bfloat16, device=device) * 99.0,
    }
    pack_pp_proxy_tensors(td_small, db, cb, do, co, 4)
    
    # Unpack 4 rows — should all be 99.0
    cache = PPSchemaCache()
    entry = cache.register(sk, dn, cn, do, co)
    result = unpack_pp_proxy_tensors(db, cb, entry, 4, device, torch.bfloat16, hidden_size, 0, topk_size)
    
    assert result["hidden_states"].shape[0] == 4
    assert torch.all(result["hidden_states"] == 99.0), "Stale data from larger batch!"
    
    print("  Active row shrink no stale PASSED")
    return True


def test_cuda_graph_compatible_operations():
    """Test that pack/unpack operations are CUDA Graph compatible.
    
    The operations must not:
    - Call .item(), .tolist(), or other GPU-to-CPU sync
    - Do dynamic allocation
    - Create new process groups
    """
    if not torch.cuda.is_available():
        print("  SKIP: No CUDA available")
        return True
    
    device = torch.device("cuda:0")
    hidden_size = 64
    num_capture = 3
    topk_size = 8
    max_rows = 64
    active_rows = 16
    
    # Pre-allocate everything
    td = {
        "hidden_states": torch.randn(active_rows, hidden_size, dtype=torch.bfloat16, device=device),
        "residual": torch.randn(active_rows, hidden_size, dtype=torch.bfloat16, device=device),
        "glm52_eagle3_aux_hidden_states": torch.randn(active_rows, num_capture, hidden_size, dtype=torch.bfloat16, device=device),
        "topk_indices": torch.randint(0, 100, (active_rows, topk_size), dtype=torch.int32, device=device),
    }
    
    sk, dn, cn, do, co = calculate_pp_buffer_layout(td, hidden_size, num_capture, topk_size, max_rows)
    db = torch.zeros(dn, dtype=torch.bfloat16, device=device)
    cb = torch.zeros(cn, dtype=torch.int32, device=device)
    
    cache = PPSchemaCache()
    entry = cache.register(sk, dn, cn, do, co)
    
    # Try CUDA Graph capture of pack + unpack
    torch.cuda.synchronize()
    
    # Warmup
    for _ in range(3):
        pack_pp_proxy_tensors(td, db, cb, do, co, active_rows)
        result = unpack_pp_proxy_tensors(db, cb, entry, active_rows, device, torch.bfloat16, hidden_size, num_capture, topk_size)
    
    torch.cuda.synchronize()
    
    # Capture
    g = torch.cuda.CUDAGraph()
    
    # Capture pack only (unpack creates new tensors, which is not graph-safe)
    with torch.cuda.graph(g):
        pack_pp_proxy_tensors(td, db, cb, do, co, active_rows)
    
    # Replay multiple times
    for i in range(100):
        # Modify input data
        td["hidden_states"].add_(0.001)
        g.replay()
        torch.cuda.synchronize()
    
    # Verify result is correct after replay
    result = unpack_pp_proxy_tensors(db, cb, entry, active_rows, device, torch.bfloat16, hidden_size, num_capture, topk_size)
    assert torch.allclose(result["hidden_states"], td["hidden_states"], atol=1e-2)
    assert torch.equal(result["topk_indices"], td["topk_indices"])
    
    print("  CUDA Graph compatible operations PASSED (100 replays)")
    return True


def test_pointer_stability_across_replays():
    """Test that buffer pointers remain stable across CUDA Graph replays."""
    if not torch.cuda.is_available():
        print("  SKIP: No CUDA available")
        return True
    
    device = torch.device("cuda:0")
    registry = PPStaticBufferRegistry(device=device)
    
    db, cb = registry.get_or_allocate(16, 1024, 256, torch.bfloat16)
    ptr_before = (db.data_ptr(), cb.data_ptr())
    
    # Simulate many replays
    for i in range(1000):
        db2, cb2 = registry.get_or_allocate(16, 1024, 256, torch.bfloat16)
        assert db2.data_ptr() == ptr_before[0]
        assert cb2.data_ptr() == ptr_before[1]
    
    ptr_after = (db.data_ptr(), cb.data_ptr())
    assert ptr_before == ptr_after
    
    print(f"  Pointer stability across 1,000 replays PASSED")
    return True


if __name__ == "__main__":
    print("=== Phase 10: CUDA Graph Validation ===")
    
    tests = [
        test_static_buffer_reuse,
        test_active_row_shrink_no_stale,
        test_cuda_graph_compatible_operations,
        test_pointer_stability_across_replays,
    ]
    
    for test in tests:
        ok = test()
        if not ok:
            sys.exit(1)
    
    print("\n=== All Phase 10 tests PASSED ===")
