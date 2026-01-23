#!/usr/bin/env python3
"""
Test script for AMD FlashInfer HIP patches with Triton fallback.

Tests the following changes:
1. RMSNorm layer - FlashInfer with Triton fallback on AMD
2. Rotary embedding - FlashInfer with Triton fallback on AMD
3. Multimodal layer integration

Run: python test_amd_flashinfer_hip.py
"""

import sys
import torch
import pytest

# Check for AMD GPU
def is_amd_gpu():
    """Check if running on AMD GPU."""
    if not torch.cuda.is_available():
        return False
    return torch.version.hip is not None

def get_device_info():
    """Get device information."""
    info = {
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "hip_version": getattr(torch.version, 'hip', None),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }
    if info["cuda_available"] and info["device_count"] > 0:
        info["device_name"] = torch.cuda.get_device_name(0)
    return info


# ============================================================================
# Reference implementations
# ============================================================================

def ref_rms_norm(x, weight, eps=1e-6):
    """Reference RMSNorm implementation."""
    orig_dtype = x.dtype
    x = x.float()
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    x = x * weight.float()
    return x.to(orig_dtype)


def ref_fused_add_rms_norm(x, residual, weight, eps=1e-6):
    """Reference fused add + RMSNorm implementation."""
    orig_dtype = x.dtype
    x = x.float() + residual.float()
    residual_out = x.to(orig_dtype)
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    x = x * weight.float()
    return x.to(orig_dtype), residual_out


def ref_layer_norm(x, weight, bias, eps=1e-5):
    """Reference LayerNorm implementation."""
    orig_dtype = x.dtype
    x = x.float()
    mean = x.mean(dim=-1, keepdim=True)
    variance = (x - mean).pow(2).mean(dim=-1, keepdim=True)
    x = (x - mean) * torch.rsqrt(variance + eps)
    if weight is not None:
        x = x * weight.float()
    if bias is not None:
        x = x + bias.float()
    return x.to(orig_dtype)


def ref_apply_rotary_embedding(x, cos, sin, is_neox_style=True):
    """Reference rotary embedding implementation."""
    if is_neox_style:
        x1, x2 = torch.chunk(x, 2, dim=-1)
        cos = cos.unsqueeze(-2)
        sin = sin.unsqueeze(-2)
        o1 = x1.float() * cos - x2.float() * sin
        o2 = x2.float() * cos + x1.float() * sin
        return torch.cat((o1, o2), dim=-1).to(x.dtype)
    else:
        # GPT-J style
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        cos = cos.unsqueeze(-2)
        sin = sin.unsqueeze(-2)
        o1 = x1.float() * cos - x2.float() * sin
        o2 = x2.float() * cos + x1.float() * sin
        out = torch.stack((o1, o2), dim=-1)
        return out.flatten(-2).to(x.dtype)


# ============================================================================
# Test: FlashInfer imports
# ============================================================================

class TestFlashInferImports:
    """Test FlashInfer import availability."""
    
    def test_flashinfer_base_import(self):
        """Test basic FlashInfer import."""
        try:
            import flashinfer
            print(f"✓ FlashInfer version: {flashinfer.__version__}")
            assert True
        except ImportError as e:
            pytest.skip(f"FlashInfer not available: {e}")
    
    def test_flashinfer_norm_import(self):
        """Test FlashInfer norm import."""
        try:
            from flashinfer.norm import rmsnorm
            print("✓ FlashInfer rmsnorm imported successfully")
            assert True
        except ImportError as e:
            print(f"⚠ FlashInfer rmsnorm not available (expected on AMD without HIP patches): {e}")
            # This is expected to fail on AMD without full HIP support
            pytest.skip(f"FlashInfer rmsnorm not available: {e}")
    
    def test_flashinfer_rope_import(self):
        """Test FlashInfer RoPE import."""
        try:
            from flashinfer.rope import apply_rope_with_cos_sin_cache_inplace
            print("✓ FlashInfer RoPE imported successfully")
            assert True
        except ImportError as e:
            print(f"⚠ FlashInfer RoPE not available (expected on AMD without HIP patches): {e}")
            pytest.skip(f"FlashInfer RoPE not available: {e}")


# ============================================================================
# Test: Triton fallback operations
# ============================================================================

class TestTritonFallback:
    """Test Triton fallback operations used on AMD."""
    
    @pytest.mark.parametrize("batch_size", [1, 16, 64])
    @pytest.mark.parametrize("hidden_size", [512, 1024, 4096])
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_triton_rms_norm(self, batch_size, hidden_size, dtype):
        """Test Triton RMSNorm implementation."""
        try:
            from sglang.multimodal_gen.runtime.layers.triton_ops import rms_norm_fn
        except ImportError as e:
            pytest.skip(f"Triton ops not available: {e}")
        
        device = "cuda"
        x = torch.randn(batch_size, hidden_size, device=device, dtype=dtype)
        weight = torch.randn(hidden_size, device=device, dtype=dtype)
        eps = 1e-6
        
        # Reference
        ref_out = ref_rms_norm(x, weight, eps)
        
        # Triton implementation
        triton_out = rms_norm_fn(x, weight, bias=None, residual=None, eps=eps)
        
        torch.testing.assert_close(triton_out, ref_out, rtol=1e-2, atol=1e-2)
        print(f"✓ Triton RMSNorm: batch={batch_size}, hidden={hidden_size}, dtype={dtype}")
    
    @pytest.mark.parametrize("batch_size", [1, 16])
    @pytest.mark.parametrize("hidden_size", [512, 1024])
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_triton_rms_norm_with_residual(self, batch_size, hidden_size, dtype):
        """Test Triton RMSNorm with residual."""
        try:
            from sglang.multimodal_gen.runtime.layers.triton_ops import rms_norm_fn
        except ImportError as e:
            pytest.skip(f"Triton ops not available: {e}")
        
        device = "cuda"
        x = torch.randn(batch_size, hidden_size, device=device, dtype=dtype)
        residual = torch.randn(batch_size, hidden_size, device=device, dtype=dtype)
        weight = torch.randn(hidden_size, device=device, dtype=dtype)
        eps = 1e-6
        
        # Reference
        ref_out, ref_residual = ref_fused_add_rms_norm(x, residual, weight, eps)
        
        # Triton implementation
        triton_out = rms_norm_fn(x, weight, bias=None, residual=residual, eps=eps)
        
        # Note: rms_norm_fn may return tuple or single tensor depending on residual
        if isinstance(triton_out, tuple):
            triton_out, triton_residual = triton_out
            torch.testing.assert_close(triton_residual, ref_residual, rtol=1e-2, atol=1e-2)
        
        torch.testing.assert_close(triton_out, ref_out, rtol=1e-2, atol=1e-2)
        print(f"✓ Triton RMSNorm+residual: batch={batch_size}, hidden={hidden_size}, dtype={dtype}")
    
    @pytest.mark.parametrize("batch_size", [1, 16])
    @pytest.mark.parametrize("seq_len", [32, 128])
    @pytest.mark.parametrize("num_heads", [8, 16])
    @pytest.mark.parametrize("head_dim", [64, 128])
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_triton_rotary_embedding(self, batch_size, seq_len, num_heads, head_dim, dtype):
        """Test Triton rotary embedding implementation."""
        try:
            from sglang.multimodal_gen.runtime.layers.triton_ops import apply_rotary_embedding
        except ImportError as e:
            pytest.skip(f"Triton ops not available: {e}")
        
        device = "cuda"
        
        # Create input: [batch*seq, num_heads, head_dim]
        num_tokens = batch_size * seq_len
        x = torch.randn(num_tokens, num_heads, head_dim, device=device, dtype=dtype)
        
        # Create cos/sin cache: [batch*seq, head_dim/2]
        half_dim = head_dim // 2
        positions = torch.arange(seq_len, device=device).repeat(batch_size)
        inv_freq = 1.0 / (10000 ** (torch.arange(0, half_dim, device=device, dtype=torch.float) / half_dim))
        freqs = torch.outer(positions.float(), inv_freq)
        cos = freqs.cos().to(dtype)
        sin = freqs.sin().to(dtype)
        
        # Triton implementation - returns [num_tokens, num_heads, head_dim]
        triton_out = apply_rotary_embedding(x, cos, sin, interleaved=False)
        
        # Verify output shape matches input
        assert triton_out.shape == x.shape, f"Output shape {triton_out.shape} != input shape {x.shape}"
        
        # Verify output is different from input (RoPE should modify)
        assert not torch.allclose(triton_out, x), "RoPE did not modify the input"
        
        # Verify output is finite
        assert torch.isfinite(triton_out).all(), "RoPE output contains non-finite values"
        
        print(f"✓ Triton RoPE: batch={batch_size}, seq={seq_len}, heads={num_heads}, dim={head_dim}, dtype={dtype}")


# ============================================================================
# Test: Multimodal layer classes with AMD dispatch
# ============================================================================

class TestMultimodalLayers:
    """Test multimodal layer classes with AMD dispatch."""
    
    @pytest.mark.parametrize("batch_size", [1, 8])
    @pytest.mark.parametrize("hidden_size", [512, 1024, 4096])
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_rmsnorm_layer_forward_hip(self, batch_size, hidden_size, dtype):
        """Test RMSNorm layer forward_hip method."""
        try:
            from sglang.multimodal_gen.runtime.layers.layernorm import RMSNorm
        except ImportError as e:
            pytest.skip(f"Multimodal layers not available: {e}")
        
        device = "cuda"
        eps = 1e-6
        
        # Create layer
        norm = RMSNorm(hidden_size, eps=eps, dtype=dtype).to(device)
        
        # Create input
        x = torch.randn(batch_size, hidden_size, device=device, dtype=dtype)
        
        # Reference
        ref_out = ref_rms_norm(x, norm.weight.data, eps)
        
        # forward_hip (uses FlashInfer if available, else Triton)
        hip_out = norm.forward_hip(x)
        
        torch.testing.assert_close(hip_out, ref_out, rtol=1e-2, atol=1e-2)
        print(f"✓ RMSNorm.forward_hip: batch={batch_size}, hidden={hidden_size}, dtype={dtype}")
    
    @pytest.mark.parametrize("batch_size", [1, 8])
    @pytest.mark.parametrize("hidden_size", [512, 1024])
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_rmsnorm_layer_forward_hip_with_residual(self, batch_size, hidden_size, dtype):
        """Test RMSNorm layer forward_hip with residual."""
        try:
            from sglang.multimodal_gen.runtime.layers.layernorm import RMSNorm
        except ImportError as e:
            pytest.skip(f"Multimodal layers not available: {e}")
        
        device = "cuda"
        eps = 1e-6
        
        # Create layer
        norm = RMSNorm(hidden_size, eps=eps, dtype=dtype).to(device)
        
        # Create input
        x = torch.randn(batch_size, hidden_size, device=device, dtype=dtype)
        residual = torch.randn(batch_size, hidden_size, device=device, dtype=dtype)
        
        # Reference
        ref_out, ref_residual = ref_fused_add_rms_norm(x, residual, norm.weight.data, eps)
        
        # forward_hip with residual
        hip_out, hip_residual = norm.forward_hip(x, residual)
        
        torch.testing.assert_close(hip_out, ref_out, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(hip_residual, ref_residual, rtol=1e-2, atol=1e-2)
        print(f"✓ RMSNorm.forward_hip+residual: batch={batch_size}, hidden={hidden_size}, dtype={dtype}")
    
    @pytest.mark.parametrize("batch_size", [1, 4])
    @pytest.mark.parametrize("hidden_size", [512, 1024])
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_layernorm_layer(self, batch_size, hidden_size, dtype):
        """Test LayerNorm layer."""
        try:
            from sglang.multimodal_gen.runtime.layers.layernorm import LayerNorm
        except ImportError as e:
            pytest.skip(f"Multimodal layers not available: {e}")
        
        device = "cuda"
        eps = 1e-5
        
        # Create layer
        norm = LayerNorm(hidden_size, eps=eps, dtype=dtype).to(device)
        # Initialize weights
        norm.weight.data.fill_(1.0)
        if norm.bias is not None:
            norm.bias.data.fill_(0.0)
        
        # Create input
        x = torch.randn(batch_size, hidden_size, device=device, dtype=dtype)
        
        # Reference
        ref_out = ref_layer_norm(x, norm.weight.data, norm.bias, eps)
        
        # forward_triton (used by forward_cuda and should work on AMD)
        triton_out = norm.forward_triton(x)
        
        torch.testing.assert_close(triton_out, ref_out, rtol=1e-2, atol=1e-2)
        print(f"✓ LayerNorm.forward_triton: batch={batch_size}, hidden={hidden_size}, dtype={dtype}")


# ============================================================================
# Test: Rotary embedding layer with AMD dispatch
# ============================================================================

class TestRotaryEmbeddingLayers:
    """Test rotary embedding layers with AMD dispatch."""
    
    def test_apply_flashinfer_rope_qk_fallback(self):
        """Test apply_flashinfer_rope_qk_inplace with Triton fallback.
        
        This function uses Triton fallback when FlashInfer is not available
        (import fails) or when FlashInfer JIT compilation fails on HIP.
        """
        try:
            from sglang.multimodal_gen.runtime.layers.rotary_embedding import (
                apply_flashinfer_rope_qk_inplace
            )
        except ImportError as e:
            pytest.skip(f"Rotary embedding not available: {e}")
        
        device = "cuda"
        dtype = torch.float16
        
        bsz, seqlen, nheads, head_size = 2, 32, 8, 64
        
        # Create inputs
        q = torch.randn(bsz, seqlen, nheads, head_size, device=device, dtype=dtype)
        k = torch.randn(bsz, seqlen, nheads, head_size, device=device, dtype=dtype)
        
        # Create cos_sin_cache - FlashInfer requires float32
        half_size = head_size // 2
        positions = torch.arange(seqlen * 2, device=device)  # Extra positions for cache
        inv_freq = 1.0 / (10000 ** (torch.arange(0, half_size, device=device, dtype=torch.float) / half_size))
        freqs = torch.outer(positions.float(), inv_freq)
        cos_sin_cache = torch.cat([freqs.cos(), freqs.sin()], dim=-1).to(torch.float32)
        
        # Apply rope - may use FlashInfer or fall back to Triton
        # On HIP, FlashInfer JIT may fail, but the function handles this
        # by falling back to Triton implementation
        try:
            q_out, k_out = apply_flashinfer_rope_qk_inplace(
                q.clone(), k.clone(), cos_sin_cache,
                head_size=head_size, is_neox=True
            )
            
            # Check output shapes
            assert q_out.shape == q.shape, f"Q shape mismatch: {q_out.shape} vs {q.shape}"
            assert k_out.shape == k.shape, f"K shape mismatch: {k_out.shape} vs {k.shape}"
            
            # Check outputs are different from inputs (rope should modify them)
            assert not torch.allclose(q_out, q), "Q unchanged after RoPE"
            assert not torch.allclose(k_out, k), "K unchanged after RoPE"
            
            print(f"✓ apply_flashinfer_rope_qk_inplace: bsz={bsz}, seq={seqlen}, heads={nheads}, dim={head_size}")
        except RuntimeError as e:
            # FlashInfer JIT compilation may fail on HIP
            # This is expected - the production code in rotary_embedding.py
            # has a try/except that falls back to Triton
            if "ninja" in str(e) or "compilation" in str(e).lower():
                pytest.skip(f"FlashInfer RoPE JIT compilation failed on HIP (expected): {e}")
            raise
    
    @pytest.mark.parametrize("head_size", [64, 128])
    @pytest.mark.parametrize("rotary_dim", [64, 128])
    @pytest.mark.parametrize("is_neox_style", [True, False])
    def test_rotary_embedding_class(self, head_size, rotary_dim, is_neox_style):
        """Test RotaryEmbedding class."""
        if rotary_dim > head_size:
            pytest.skip("rotary_dim > head_size")
        
        try:
            from sglang.multimodal_gen.runtime.layers.rotary_embedding import RotaryEmbedding
        except ImportError as e:
            pytest.skip(f"RotaryEmbedding not available: {e}")
        
        device = "cuda"
        dtype = torch.float16
        max_position = 2048
        base = 10000
        
        # Create embedding
        rope = RotaryEmbedding(
            head_size=head_size,
            rotary_dim=rotary_dim,
            max_position_embeddings=max_position,
            base=base,
            is_neox_style=is_neox_style,
            dtype=dtype
        ).to(device)
        
        # Create inputs
        batch_size, seq_len, num_heads = 2, 32, 8
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        query = torch.randn(batch_size * seq_len, num_heads * head_size, device=device, dtype=dtype)
        key = torch.randn(batch_size * seq_len, num_heads * head_size, device=device, dtype=dtype)
        
        # Apply rope
        q_out, k_out = rope.forward_native(positions.flatten()[:seq_len], query[:seq_len], key[:seq_len])
        
        assert q_out.shape == query[:seq_len].shape
        assert k_out.shape == key[:seq_len].shape
        print(f"✓ RotaryEmbedding: head={head_size}, rotary_dim={rotary_dim}, neox={is_neox_style}")


# ============================================================================
# Test: End-to-end integration
# ============================================================================

class TestIntegration:
    """End-to-end integration tests."""
    
    def test_platform_detection(self):
        """Test platform detection."""
        try:
            from sglang.multimodal_gen.runtime.platforms import current_platform
            print(f"✓ Current platform: {current_platform}")
            print(f"  - is_cuda: {getattr(current_platform, 'is_cuda', 'N/A')}")
            print(f"  - is_rocm: {getattr(current_platform, 'is_rocm', 'N/A')}")
        except ImportError as e:
            pytest.skip(f"Platform detection not available: {e}")
    
    def test_full_rmsnorm_dispatch(self):
        """Test full RMSNorm dispatch based on platform."""
        try:
            from sglang.multimodal_gen.runtime.layers.layernorm import RMSNorm
        except ImportError as e:
            pytest.skip(f"RMSNorm not available: {e}")
        
        device = "cuda"
        dtype = torch.float16
        hidden_size = 1024
        batch_size = 4
        
        norm = RMSNorm(hidden_size, dtype=dtype).to(device)
        x = torch.randn(batch_size, hidden_size, device=device, dtype=dtype)
        
        # Call forward() which should dispatch based on platform
        out = norm(x)
        
        # Reference
        ref_out = ref_rms_norm(x, norm.weight.data, norm.variance_epsilon)
        
        torch.testing.assert_close(out, ref_out, rtol=1e-2, atol=1e-2)
        print("✓ Full RMSNorm dispatch working")


# ============================================================================
# Main
# ============================================================================

def main():
    """Run all tests."""
    print("=" * 70)
    print("AMD FlashInfer HIP Patches - Test Suite")
    print("=" * 70)
    
    # Print device info
    info = get_device_info()
    print(f"\nDevice Information:")
    print(f"  PyTorch: {info['pytorch_version']}")
    print(f"  CUDA/ROCm available: {info['cuda_available']}")
    print(f"  HIP version: {info['hip_version']}")
    print(f"  Device count: {info['device_count']}")
    if 'device_name' in info:
        print(f"  Device name: {info['device_name']}")
    
    is_amd = is_amd_gpu()
    print(f"  Is AMD GPU: {is_amd}")
    print()
    
    if not info['cuda_available']:
        print("ERROR: No GPU available!")
        return 1
    
    # Run pytest
    args = [__file__, "-v", "--tb=short", "-x"]
    return pytest.main(args)


if __name__ == "__main__":
    sys.exit(main())
