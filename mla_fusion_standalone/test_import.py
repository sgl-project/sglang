#!/usr/bin/env python3
"""
Test script to verify the fusion kernel works
"""
import torch
import mla_fusion_kernel

print("✅ Module imported successfully!")
print(f"Available functions: {dir(mla_fusion_kernel)}")

# Test basic call
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    print(f"\n✅ CUDA is available")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    
    # Create dummy inputs
    nnz, Dn, Dr = 4, 512, 64
    q_nope = torch.randn(nnz, Dn, device=device, dtype=torch.float16)
    q_rope = torch.randn(nnz, Dr, device=device, dtype=torch.float16)
    k_nope = torch.randn(nnz, Dn, device=device, dtype=torch.float16)
    k_rope = torch.randn(nnz, Dr, device=device, dtype=torch.float16)
    
    cos_sin = torch.randn(2048, Dr*2, device=device, dtype=torch.float32)
    pos_ids = torch.arange(nnz, device=device, dtype=torch.int64)
    
    q_out = torch.empty(nnz, Dn+Dr, device=device, dtype=torch.uint8)
    k_nope_out = torch.empty(nnz, Dn, device=device, dtype=torch.uint8)
    k_rope_out = torch.empty(nnz, Dr, device=device, dtype=torch.uint8)
    
    try:
        mla_fusion_kernel.mla_rope_quantize_fp8_fused(
            q_nope, q_rope, k_nope, k_rope,
            cos_sin, pos_ids, False,
            q_out, k_nope_out, k_rope_out,
            None, None
        )
        print("✅ Kernel executed successfully!")
    except Exception as e:
        print(f"❌ Kernel execution failed: {e}")
else:
    print("⚠️ CUDA not available, skipping kernel test")

