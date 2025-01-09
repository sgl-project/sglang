import unittest
import torch
import itertools

from sglang.srt.layers.attention.triton_ops.decode_attention import quantize_cache_kv

class TestQuantizeKVCache(unittest.TestCase):
    DTYPES = [torch.float32, torch.half, torch.bfloat16]
    BATCH_SIZES = [1, 16, 32, 128]
    HEAD_NUMS = [1, 8, 16, 32] 
    HEAD_DIMS = [32, 64, 128, 256] 
    SEEDS = [0, 42, 123]


    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is not available")
        torch.set_default_device("cuda")

    def _native_quantize_cache_kv(self, k_status, v_status, dest_idx):
        """Native PyTorch implementation of quantize_cache_kv for reference"""
        bs = dest_idx.shape[0]
        k_head_num = k_status.shape[1]
        k_head_dim = k_status.shape[2]
        
        # 确保使用float32进行计算以保持精度
        k_status = k_status.to(torch.float32)
        v_status = v_status.to(torch.float32)
        
        k_quantized = torch.empty(
            (bs, k_head_num, k_head_dim),
            dtype=torch.uint8,
            device=k_status.device
        )
        v_quantized = torch.empty(
            (bs, k_head_num, k_head_dim),
            dtype=torch.uint8,
            device=v_status.device
        )
        k_scale_zeros = torch.empty(
            (bs, k_head_num, 2),
            dtype=torch.float32,
            device=k_status.device
        )
        v_scale_zeros = torch.empty(
            (bs, k_head_num, 2),
            dtype=torch.float32,
            device=v_status.device
        )

        for i in range(bs):
            for h in range(k_head_num):
                # Quantize K
                k_min = k_status[i, h].min().to(torch.float32)
                k_max = k_status[i, h].max().to(torch.float32)
                k_scale = (k_max - k_min) / 255.0  # 使用255.0保持浮点精度
                k_zero = -k_min / k_scale
                k_quantized[i, h] = ((k_status[i, h] / k_scale + k_zero + 0.5)
                    .clamp(0, 255)
                    .to(torch.uint8))
                k_scale_zeros[i, h, 0] = k_scale
                k_scale_zeros[i, h, 1] = k_zero

                # Quantize V
                v_min = v_status[i, h].min().to(torch.float32)
                v_max = v_status[i, h].max().to(torch.float32)
                v_scale = (v_max - v_min) / 255.0
                v_zero = -v_min / v_scale
                
                v_quantized[i, h] = ((v_status[i, h] / v_scale + v_zero + 0.5)
                    .clamp(0, 255)
                    .to(torch.uint8))
                v_scale_zeros[i, h, 0] = v_scale
                v_scale_zeros[i, h, 1] = v_zero

        return k_quantized, k_scale_zeros, v_quantized, v_scale_zeros

    def _test_quantize_cache_kv(self, batch_size, head_num, head_dim, dtype, seed):
        # 设置打印选项，显示所有元素
        torch.set_printoptions(profile="full", linewidth=120)
        
        print(f"\nTesting with params: batch_size={batch_size}, head_num={head_num}, "
              f"head_dim={head_dim}, dtype={dtype}, seed={seed}")
        
        torch.manual_seed(seed)
        
        # 创建输入数据
        print("Creating input tensors...")
        k_status = torch.randn(
            batch_size, head_num, head_dim,
            dtype=dtype,
            device="cuda"
        )
        v_status = torch.randn(
            batch_size, head_num, head_dim,
            dtype=dtype,
            device="cuda"
        )
        dest_idx = torch.arange(batch_size, device="cuda")

        # 创建输出tensor
        print("Creating output tensors...")
        k_quantized = torch.empty(
            (batch_size, head_num, head_dim),
            dtype=torch.uint8,
            device="cuda"
        )
        v_quantized = torch.empty(
            (batch_size, head_num, head_dim),
            dtype=torch.uint8,
            device="cuda"
        )
        k_scale_zeros = torch.empty(
            (batch_size, head_num, 2),
            dtype=torch.float32,
            device="cuda"
        )
        v_scale_zeros = torch.empty(
            (batch_size, head_num, 2),
            dtype=torch.float32,
            device="cuda"
        )

        # 运行triton实现
        print("Running Triton implementation...")
        with torch.inference_mode():
            quantize_cache_kv(
                k_status,
                v_status,
                dest_idx,
                k_quantized,
                k_scale_zeros,
                v_quantized,
                v_scale_zeros
            )

        # 运行参考实现
        print("Running reference implementation...")
        ref_k_quantized, ref_k_scale_zeros, ref_v_quantized, ref_v_scale_zeros = (
            self._native_quantize_cache_kv(k_status, v_status, dest_idx)
        )

        # 验证结果
        print("Verifying results...")
        
        # 检查k_quantized
        if not torch.allclose(k_quantized, ref_k_quantized, atol=1):
            mismatch = (torch.abs(k_quantized - ref_k_quantized) > 1)
            print("\nK_quantized mismatch:")
            print("Triton output at mismatch positions:")
            print(f"Original shape: {k_quantized.shape}")
            print(f"Values: {k_quantized[mismatch].tolist()}")
            print("\nReference output at mismatch positions:")
            print(f"Values: {ref_k_quantized[mismatch].tolist()}")
            print("\nDifference (as int8):")
            print(f"Values: {(k_quantized[mismatch].to(torch.int16) - ref_k_quantized[mismatch].to(torch.int16)).tolist()}")
            
        # 检查v_quantized
        if not torch.allclose(v_quantized, ref_v_quantized, atol=1):
            mismatch = (torch.abs(v_quantized - ref_v_quantized) > 1)
            print("\nV_quantized mismatch:")
            print("Triton output at mismatch positions:")
            print(f"Original shape: {v_quantized.shape}")
            print(f"Values: {v_quantized[mismatch].tolist()}")
            print("\nReference output at mismatch positions:")
            print(f"Values: {ref_v_quantized[mismatch].tolist()}")
            print("\nDifference (as int8):")
            print(f"Values: {(v_quantized[mismatch].to(torch.int16) - ref_v_quantized[mismatch].to(torch.int16)).tolist()}")
        
        # 检查k_scale_zeros
        if not torch.allclose(k_scale_zeros, ref_k_scale_zeros, rtol=1e-2):
            mismatch = ~torch.isclose(k_scale_zeros, ref_k_scale_zeros, rtol=1e-2)
            print("\nK_scale_zeros mismatch:")
            print("Triton output at mismatch positions:")
            print(f"Original shape: {k_scale_zeros.shape}")
            print(f"Values: {k_scale_zeros[mismatch].tolist()}")
            print("\nReference output at mismatch positions:")
            print(f"Values: {ref_k_scale_zeros[mismatch].tolist()}")
            print("\nRelative difference:")
            rel_diff = ((k_scale_zeros[mismatch] - ref_k_scale_zeros[mismatch])/ref_k_scale_zeros[mismatch]).tolist()
            print(f"Values: {rel_diff}")
        
        # 检查v_scale_zeros
        if not torch.allclose(v_scale_zeros, ref_v_scale_zeros, rtol=1e-2):
            mismatch = ~torch.isclose(v_scale_zeros, ref_v_scale_zeros, rtol=1e-2)
            print("\nV_scale_zeros mismatch:")
            print("Triton output at mismatch positions:")
            print(f"Original shape: {v_scale_zeros.shape}")
            print(f"Values: {v_scale_zeros[mismatch].tolist()}")
            print("\nReference output at mismatch positions:")
            print(f"Values: {ref_v_scale_zeros[mismatch].tolist()}")
            print("\nRelative difference:")
            rel_diff = ((v_scale_zeros[mismatch] - ref_v_scale_zeros[mismatch])/ref_v_scale_zeros[mismatch]).tolist()
            print(f"Values: {rel_diff}")

        # 恢复默认打印选项
        torch.set_printoptions(profile="default")
        
        # 执行断言
        self.assertTrue(torch.allclose(k_quantized, ref_k_quantized, atol=1))
        self.assertTrue(torch.allclose(v_quantized, ref_v_quantized, atol=1))
        self.assertTrue(torch.allclose(k_scale_zeros, ref_k_scale_zeros, rtol=1e-2))
        self.assertTrue(torch.allclose(v_scale_zeros, ref_v_scale_zeros, rtol=1e-2))
        print("Test passed!")

    def test_quantize_cache_kv(self):
        total_tests = (len(self.BATCH_SIZES) * len(self.HEAD_NUMS) * 
                      len(self.HEAD_DIMS) * len(self.DTYPES) * len(self.SEEDS))
        print(f"\nRunning {total_tests} test combinations...")
        
        current_test = 0
        for params in itertools.product(
            self.BATCH_SIZES,
            self.HEAD_NUMS, 
            self.HEAD_DIMS,
            self.DTYPES,
            self.SEEDS
        ):
            current_test += 1
            print(f"\nTest {current_test}/{total_tests}")
            with self.subTest(
                batch_size=params[0],
                head_num=params[1],
                head_dim=params[2],
                dtype=params[3],
                seed=params[4]
            ):
                self._test_quantize_cache_kv(*params)

if __name__ == "__main__":
    unittest.main(verbosity=2)
