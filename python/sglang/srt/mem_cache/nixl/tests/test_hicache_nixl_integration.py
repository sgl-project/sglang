#!/usr/bin/env python3

import sys
import os
import tempfile
import torch

# Set up the path
import test_utils
test_utils.setup_path()

def test_hicache_nixl_integration():
    """Test HiCacheNixl integration with all components."""
    print("=== Testing HiCacheNixl Integration ===")
    
    try:
        from sglang.srt.mem_cache.nixl.hicache_nixl import HiCacheNixl
        
        # Test HiCacheNixl with POSIX plugin
        hicache = HiCacheNixl(file_path="/tmp/hicache_nixl_test", file_plugin="POSIX")
        print("✓ Created HiCacheNixl instance")
        
        # Test basic operations
        key = "test_key_123"
        value = torch.randn(10, 10)
        
        # Test set operation
        result = hicache.set(key, value, overwrite=True)
        print(f"✓ Set operation: {'Success' if result else 'Failed'}")
        if not result:
            print("✗ Set operation failed - this is a critical error")
            return False
        
        # Test exists operation
        exists = hicache.exists(key)
        print(f"✓ Exists operation: {'Success' if exists else 'Failed'}")
        
        # Test get operation
        dst_tensor = torch.zeros_like(value)
        retrieved = hicache.get(key, dst_tensor)
        if retrieved is not None:
            # Check if tensors are equal
            is_equal = torch.allclose(value, retrieved, atol=1e-6)
            print(f"✓ Get operation: {'Success' if is_equal else 'Failed - tensors not equal'}")
            if not is_equal:
                print("✗ Get operation failed - tensors not equal")
                return False
        else:
            print("✗ Get operation: Failed - returned None")
            return False
        
        # Test delete operation
        hicache.delete(key)
        exists_after_delete = hicache.exists(key)
        print(f"✓ Delete operation: {'Success' if not exists_after_delete else 'Failed'}")
        
        # Test batch operations
        keys = ["batch_key_1", "batch_key_2", "batch_key_3"]
        values = [torch.randn(5, 5), torch.randn(3, 3), torch.randn(7, 7)]
        
        # Test batch set
        result = hicache.batch_set(keys, values, overwrite=True)
        print(f"✓ Batch set operation: {'Success' if result else 'Failed'}")
        if not result:
            print("✗ Batch set operation failed - this is a critical error")
            return False
        
        # Test batch get
        dst_tensors = [torch.zeros_like(v) for v in values]
        retrieved_batch = hicache.batch_get(keys, dst_tensors)
        if all(t is not None for t in retrieved_batch):
            # Check if all tensors are equal
            all_equal = all(torch.allclose(v, t, atol=1e-6) for v, t in zip(values, retrieved_batch))
            print(f"✓ Batch get operation: {'Success' if all_equal else 'Failed - tensors not equal'}")
            if not all_equal:
                print("✗ Batch get operation failed - tensors not equal")
                return False
        else:
            print("✗ Batch get operation: Failed - some returned None")
            return False
        
        # Test clear operation
        hicache.clear()
        all_deleted = all(not hicache.exists(key) for key in keys)
        print(f"✓ Clear operation: {'Success' if all_deleted else 'Failed'}")
        
        # Test tensor registration
        test_tensor = torch.randn(10, 10)
        result = hicache.register(test_tensor)
        print(f"✓ Tensor registration: {'Success' if result else 'Failed'}")
        
        # Test tensor deregistration
        result = hicache.deregister(test_tensor)
        print(f"✓ Tensor deregistration: {'Success' if result else 'Failed'}")
        
        print("✓ All HiCacheNixl integration tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ Error during HiCacheNixl integration test: {e}")
        return False

if __name__ == "__main__":
    success = test_hicache_nixl_integration()
    sys.exit(0 if success else 1) 