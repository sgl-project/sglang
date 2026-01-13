import torch
import unittest
# Try to import sgl_kernel, if not available (build failed/skipped), skip tests
try:
    import sgl_kernel
except ImportError:
    sgl_kernel = None

@unittest.skipIf(sgl_kernel is None, "sgl_kernel not installed")
class TestCacheScheduler(unittest.TestCase):
    def test_cache_scheduler(self):
        device = "cuda"
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
            
        K = 4
        B = 8
        
        # Tokens: 0..99
        # top_k: [10, 20, 30, 40]
        top_k_indices = torch.tensor([10, 20, 30, 40], dtype=torch.int64, device=device)
        
        # Hot buffer: 
        # [10, 20, 99, 98, 97, 96, 95, 94]
        # Locations:
        # [0,  1,  2,  3,  4,  5,  6,  7]
        hot_buffer_token_indices = torch.tensor([10, 20, 99, 98, 97, 96, 95, 94], dtype=torch.int64, device=device)
        hot_buffer_device_locations = torch.arange(B, dtype=torch.int64, device=device)
        
        # CPU locations (mock)
        cache_cpu_locations = torch.zeros(100, dtype=torch.int64, device=device)
        
        # Expected:
        # 10 -> Hit (Loc 0)
        # 20 -> Hit (Loc 1)
        # 30 -> Miss (Evict e.g. 99 at Loc 2)
        # 40 -> Miss (Evict e.g. 98 at Loc 3)
        
        top_k_device_locations = torch.full((K,), -1, dtype=torch.int64, device=device)
        
        torch.ops.sgl_kernel.manage_sparse_cache(
            top_k_indices,
            hot_buffer_token_indices,
            hot_buffer_device_locations,
            cache_cpu_locations,
            top_k_device_locations
        )
        
        # Check hits
        self.assertEqual(top_k_device_locations[0].item(), 0)
        self.assertEqual(top_k_device_locations[1].item(), 1)
        
        # Check misses (should have valid locations from buffer)
        loc2 = top_k_device_locations[2].item()
        loc3 = top_k_device_locations[3].item()
        
        self.assertTrue(loc2 in [2, 3, 4, 5, 6, 7])
        self.assertTrue(loc3 in [2, 3, 4, 5, 6, 7])
        self.assertNotEqual(loc2, loc3)
        
        # Check buffer update
        # 30 and 40 should be in buffer now
        mask_30 = (hot_buffer_token_indices == 30)
        mask_40 = (hot_buffer_token_indices == 40)
        self.assertTrue(mask_30.any())
        self.assertTrue(mask_40.any())
        
        # Check locations match
        self.assertEqual(hot_buffer_device_locations[mask_30].item(), loc2)
        self.assertEqual(hot_buffer_device_locations[mask_40].item(), loc3)

if __name__ == '__main__':
    unittest.main()
