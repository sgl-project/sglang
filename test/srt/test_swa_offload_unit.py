# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

#!/usr/bin/env python3
"""
Unit test for SWATokenToKVPoolAllocator get_cpu_copy/load_cpu_copy.
This is a faster test that doesn't require running servers.
"""

import sys
from pathlib import Path

import torch

# Add sglang to path
sys.path.insert(0, str(Path(__file__).parent / "python"))

from sglang.srt.mem_cache.allocator import SWATokenToKVPoolAllocator
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool, SWAKVPool


def test_swa_allocator_cpu_copy():
    """Test get_cpu_copy and load_cpu_copy for SWA allocator."""
    print("=" * 70)
    print("Testing SWATokenToKVPoolAllocator CPU offload/load")
    print("=" * 70)

    # Setup parameters
    size = 1000
    size_swa = 800
    dtype = torch.float16
    head_num = 8
    head_dim = 64
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    if device == "cpu":
        print("[WARNING] CUDA not available, skipping test")
        return True

    print(f"\n[INFO] Creating SWAKVPool...")
    print(f"  - Size: {size}, SWA size: {size_swa}")
    print(f"  - Device: {device}")
    print(f"  - Dtype: {dtype}")

    # Create SWA KV pool
    # Simulate a model with 4 full attention layers and 4 SWA layers
    full_attention_layer_ids = [0, 1, 6, 7]
    swa_attention_layer_ids = [2, 3, 4, 5]

    try:
        kv_pool = SWAKVPool(
            size=size,
            size_swa=size_swa,
            dtype=dtype,
            head_num=head_num,
            head_dim=head_dim,
            swa_attention_layer_ids=swa_attention_layer_ids,
            full_attention_layer_ids=full_attention_layer_ids,
            enable_kvcache_transpose=False,
            device=device,
            token_to_kv_pool_class=MHATokenToKVPool,
        )
        print("[OK] SWAKVPool created successfully")
    except Exception as e:
        print(f"[ERROR] Failed to create SWAKVPool: {e}")
        import traceback

        traceback.print_exc()
        return False

    print(f"\n[INFO] Creating SWATokenToKVPoolAllocator...")
    try:
        allocator = SWATokenToKVPoolAllocator(
            size=size,
            size_swa=size_swa,
            dtype=dtype,
            device=device,
            kvcache=kv_pool,
            need_sort=False,
        )
        print("[OK] Allocator created successfully")
    except Exception as e:
        print(f"[ERROR] Failed to create allocator: {e}")
        import traceback

        traceback.print_exc()
        return False

    # Allocate some indices
    print(f"\n[INFO] Allocating indices for testing...")
    num_tokens = 100
    indices = allocator.alloc(num_tokens)

    if indices is None:
        print(f"[ERROR] Failed to allocate {num_tokens} tokens")
        return False

    print(f"[OK] Allocated {len(indices)} indices")

    # Fill KV cache with random data
    print(f"\n[INFO] Filling KV cache with test data...")
    for layer_id in range(len(full_attention_layer_ids) + len(swa_attention_layer_ids)):
        k_buffer = kv_pool.get_key_buffer(layer_id)
        v_buffer = kv_pool.get_value_buffer(layer_id)

        # Map full indices to SWA indices if needed
        if (
            layer_id in swa_attention_layer_ids
            and kv_pool.full_to_swa_index_mapping is not None
        ):
            actual_indices = kv_pool.full_to_swa_index_mapping[indices]
            # Filter out zero indices
            mask = actual_indices > 0
            if mask.any():
                actual_indices = actual_indices[mask]
            else:
                continue
        else:
            actual_indices = indices

        # Fill with random data
        k_buffer[actual_indices] = torch.randn_like(k_buffer[actual_indices])
        v_buffer[actual_indices] = torch.randn_like(v_buffer[actual_indices])

    print("[OK] KV cache filled with test data")

    # Test get_cpu_copy
    print(f"\n[INFO] Testing get_cpu_copy()...")
    try:
        kv_cache_cpu = allocator.get_cpu_copy(indices)
        print(f"[OK] get_cpu_copy() succeeded")
        print(f"  - Type: {type(kv_cache_cpu)}")
        if isinstance(kv_cache_cpu, dict):
            print(f"  - Keys: {kv_cache_cpu.keys()}")
    except NotImplementedError as e:
        print(f"[FAILURE] get_cpu_copy() not implemented: {e}")
        return False
    except Exception as e:
        print(f"[ERROR] get_cpu_copy() failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    # Save original data for comparison
    print(f"\n[INFO] Saving original GPU data for comparison...")
    original_data = {}
    for layer_id in range(len(full_attention_layer_ids) + len(swa_attention_layer_ids)):
        k_buffer = kv_pool.get_key_buffer(layer_id)
        v_buffer = kv_pool.get_value_buffer(layer_id)

        if (
            layer_id in swa_attention_layer_ids
            and kv_pool.full_to_swa_index_mapping is not None
        ):
            actual_indices = kv_pool.full_to_swa_index_mapping[indices]
            mask = actual_indices > 0
            if mask.any():
                actual_indices = actual_indices[mask]
            else:
                continue
        else:
            actual_indices = indices

        original_data[layer_id] = {
            "k": k_buffer[actual_indices].clone(),
            "v": v_buffer[actual_indices].clone(),
        }

    # Clear GPU data
    print(f"\n[INFO] Clearing GPU data (simulating memory release)...")
    for layer_id in range(len(full_attention_layer_ids) + len(swa_attention_layer_ids)):
        k_buffer = kv_pool.get_key_buffer(layer_id)
        v_buffer = kv_pool.get_value_buffer(layer_id)

        if (
            layer_id in swa_attention_layer_ids
            and kv_pool.full_to_swa_index_mapping is not None
        ):
            actual_indices = kv_pool.full_to_swa_index_mapping[indices]
            mask = actual_indices > 0
            if mask.any():
                actual_indices = actual_indices[mask]
            else:
                continue
        else:
            actual_indices = indices

        k_buffer[actual_indices] = 0
        v_buffer[actual_indices] = 0

    print("[OK] GPU data cleared")

    # Test load_cpu_copy
    print(f"\n[INFO] Testing load_cpu_copy()...")
    try:
        allocator.load_cpu_copy(kv_cache_cpu, indices)
        print(f"[OK] load_cpu_copy() succeeded")
    except NotImplementedError as e:
        print(f"[FAILURE] load_cpu_copy() not implemented: {e}")
        return False
    except Exception as e:
        print(f"[ERROR] load_cpu_copy() failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    # Verify data was restored correctly
    print(f"\n[INFO] Verifying restored data matches original...")
    all_match = True
    for layer_id in range(len(full_attention_layer_ids) + len(swa_attention_layer_ids)):
        if layer_id not in original_data:
            continue

        k_buffer = kv_pool.get_key_buffer(layer_id)
        v_buffer = kv_pool.get_value_buffer(layer_id)

        if (
            layer_id in swa_attention_layer_ids
            and kv_pool.full_to_swa_index_mapping is not None
        ):
            actual_indices = kv_pool.full_to_swa_index_mapping[indices]
            mask = actual_indices > 0
            if mask.any():
                actual_indices = actual_indices[mask]
            else:
                continue
        else:
            actual_indices = indices

        restored_k = k_buffer[actual_indices]
        restored_v = v_buffer[actual_indices]

        if not torch.allclose(restored_k, original_data[layer_id]["k"], rtol=1e-3):
            print(f"[ERROR] Layer {layer_id} K data mismatch!")
            all_match = False
        if not torch.allclose(restored_v, original_data[layer_id]["v"], rtol=1e-3):
            print(f"[ERROR] Layer {layer_id} V data mismatch!")
            all_match = False

    if all_match:
        print(f"[SUCCESS] All data restored correctly!")
    else:
        print(f"[FAILURE] Data mismatch after restore")
        return False

    print("\n" + "=" * 70)
    print("[SUCCESS] All tests PASSED!")
    print("=" * 70)
    return True


if __name__ == "__main__":
    success = test_swa_allocator_cpu_copy()
    sys.exit(0 if success else 1)
