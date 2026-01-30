"""
Unit tests for flashinfer.rope.mla_rope_quantize_fp8 kernel.

This tests the non-fused RoPE + FP8 quantization path used in TRT-LLM MLA backend.
"""

import pytest
import torch

from sglang.srt.utils import is_cuda, is_flashinfer_available

_is_cuda = is_cuda()
_has_flashinfer = is_flashinfer_available()

if _has_flashinfer:
    import flashinfer


def create_cos_sin_cache(max_seq_len: int, rotary_dim: int, device: torch.device):
    """Create a cosine/sine cache for RoPE.
    
    Args:
        max_seq_len: Maximum sequence length
        rotary_dim: Dimension of the rotary embedding (qk_rope_head_dim)
        device: Target device
        
    Returns:
        cos_sin_cache: Tensor of shape [max_seq_len, rotary_dim] containing interleaved cos/sin values.
                       Always returns float32 as required by FlashInfer.
    """
    # Create position indices
    position = torch.arange(max_seq_len, device=device, dtype=torch.float32)
    
    # Create frequency bands (similar to standard RoPE)
    # freq = 1 / (base ** (2i / dim)) where i = 0, 1, ..., dim/2 - 1
    base = 10000.0
    half_dim = rotary_dim // 2
    freq = 1.0 / (base ** (torch.arange(0, half_dim, device=device, dtype=torch.float32) / half_dim))
    
    # Compute angles: position * freq
    # Shape: [max_seq_len, half_dim]
    angles = position.unsqueeze(1) * freq.unsqueeze(0)
    
    # Compute cos and sin
    cos = torch.cos(angles)
    sin = torch.sin(angles)
    
    # Interleave cos and sin: [cos0, sin0, cos1, sin1, ...]
    # Shape: [max_seq_len, rotary_dim]
    cos_sin_cache = torch.stack([cos, sin], dim=-1).view(max_seq_len, rotary_dim)
    
    # FlashInfer requires cos_sin_cache to be float32
    return cos_sin_cache


@pytest.mark.skipif(not _is_cuda, reason="CUDA not available")
@pytest.mark.skipif(not _has_flashinfer, reason="FlashInfer not available")
class TestMLARopeQuantizeFP8:
    """Tests for flashinfer.rope.mla_rope_quantize_fp8 kernel."""
    
    # DeepSeek V3 MLA dimensions
    KV_LORA_RANK = 512
    QK_ROPE_HEAD_DIM = 64
    NUM_HEADS = 16  # Reduced for testing
    
    @pytest.fixture
    def device(self):
        return torch.device("cuda")
    
    @pytest.fixture
    def dtype(self):
        return torch.bfloat16
    
    @pytest.fixture
    def cos_sin_cache(self, device):
        max_seq_len = 8192
        return create_cos_sin_cache(max_seq_len, self.QK_ROPE_HEAD_DIM, device)
    
    def test_basic_functionality(self, device, dtype, cos_sin_cache):
        """Test basic functionality with typical MLA dimensions."""
        batch_size = 8
        
        # Create input tensors
        q_nope = torch.randn(batch_size, self.NUM_HEADS, self.KV_LORA_RANK, device=device, dtype=dtype)
        q_rope = torch.randn(batch_size, self.NUM_HEADS, self.QK_ROPE_HEAD_DIM, device=device, dtype=dtype)
        k_nope = torch.randn(batch_size, self.KV_LORA_RANK, device=device, dtype=dtype)
        k_rope = torch.randn(batch_size, self.QK_ROPE_HEAD_DIM, device=device, dtype=dtype)
        
        # Position IDs
        pos_ids = torch.arange(batch_size, device=device, dtype=torch.int32)
        
        # Allocate output tensors
        attn_dtype = torch.float8_e4m3fn
        q_out = torch.empty(
            batch_size,
            self.NUM_HEADS,
            self.KV_LORA_RANK + self.QK_ROPE_HEAD_DIM,
            device=device,
            dtype=attn_dtype,
        )
        k_rope_out = torch.empty(k_rope.shape, device=device, dtype=attn_dtype)
        k_nope_out = torch.empty(k_nope.shape, device=device, dtype=attn_dtype)
        
        # Call the kernel
        flashinfer.rope.mla_rope_quantize_fp8(
            q_rope=q_rope,
            k_rope=k_rope,
            q_nope=q_nope,
            k_nope=k_nope,
            cos_sin_cache=cos_sin_cache,
            pos_ids=pos_ids,
            is_neox=False,  # GPT-J style for DeepSeek V3
            quantize_dtype=attn_dtype,
            q_rope_out=q_out[..., self.KV_LORA_RANK:],
            k_rope_out=k_rope_out,
            q_nope_out=q_out[..., :self.KV_LORA_RANK],
            k_nope_out=k_nope_out,
            quant_scale_q=1.0,
            quant_scale_kv=1.0,
        )
        
        # Verify output shapes
        assert q_out.shape == (batch_size, self.NUM_HEADS, self.KV_LORA_RANK + self.QK_ROPE_HEAD_DIM)
        assert k_rope_out.shape == k_rope.shape
        assert k_nope_out.shape == k_nope.shape
        
        # Verify output dtypes
        assert q_out.dtype == attn_dtype
        assert k_rope_out.dtype == attn_dtype
        assert k_nope_out.dtype == attn_dtype
        
        # Verify outputs are not all zeros (kernel actually did something)
        assert not torch.all(q_out == 0)
        assert not torch.all(k_rope_out == 0)
        assert not torch.all(k_nope_out == 0)
        
        # Verify no NaN/Inf values
        q_out_float = q_out.to(torch.float32)
        k_rope_out_float = k_rope_out.to(torch.float32)
        k_nope_out_float = k_nope_out.to(torch.float32)
        
        assert not torch.isnan(q_out_float).any(), "q_out contains NaN"
        assert not torch.isinf(q_out_float).any(), "q_out contains Inf"
        assert not torch.isnan(k_rope_out_float).any(), "k_rope_out contains NaN"
        assert not torch.isinf(k_rope_out_float).any(), "k_rope_out contains Inf"
        assert not torch.isnan(k_nope_out_float).any(), "k_nope_out contains NaN"
        assert not torch.isinf(k_nope_out_float).any(), "k_nope_out contains Inf"
    
    def test_nope_passthrough(self, device, dtype, cos_sin_cache):
        """Test that nope components are quantized but not rotated."""
        batch_size = 4
        
        # Create input tensors with known values
        q_nope = torch.ones(batch_size, self.NUM_HEADS, self.KV_LORA_RANK, device=device, dtype=dtype)
        q_rope = torch.randn(batch_size, self.NUM_HEADS, self.QK_ROPE_HEAD_DIM, device=device, dtype=dtype)
        k_nope = torch.ones(batch_size, self.KV_LORA_RANK, device=device, dtype=dtype) * 0.5
        k_rope = torch.randn(batch_size, self.QK_ROPE_HEAD_DIM, device=device, dtype=dtype)
        
        pos_ids = torch.arange(batch_size, device=device, dtype=torch.int32)
        
        attn_dtype = torch.float8_e4m3fn
        q_out = torch.empty(batch_size, self.NUM_HEADS, self.KV_LORA_RANK + self.QK_ROPE_HEAD_DIM, device=device, dtype=attn_dtype)
        k_rope_out = torch.empty(k_rope.shape, device=device, dtype=attn_dtype)
        k_nope_out = torch.empty(k_nope.shape, device=device, dtype=attn_dtype)
        
        flashinfer.rope.mla_rope_quantize_fp8(
            q_rope=q_rope,
            k_rope=k_rope,
            q_nope=q_nope,
            k_nope=k_nope,
            cos_sin_cache=cos_sin_cache,
            pos_ids=pos_ids,
            is_neox=False,
            quantize_dtype=attn_dtype,
            q_rope_out=q_out[..., self.KV_LORA_RANK:],
            k_rope_out=k_rope_out,
            q_nope_out=q_out[..., :self.KV_LORA_RANK],
            k_nope_out=k_nope_out,
            quant_scale_q=1.0,
            quant_scale_kv=1.0,
        )
        
        # Check that q_nope part is close to original (just quantized)
        q_nope_result = q_out[..., :self.KV_LORA_RANK].to(torch.float32)
        q_nope_expected = q_nope.to(torch.float32)
        
        # Allow for quantization error (FP8 has limited precision)
        max_error = torch.max(torch.abs(q_nope_result - q_nope_expected)).item()
        assert max_error < 0.1, f"q_nope passthrough error too large: {max_error}"
        
        # Check k_nope similarly
        k_nope_result = k_nope_out.to(torch.float32)
        k_nope_expected = k_nope.to(torch.float32)
        max_error_k = torch.max(torch.abs(k_nope_result - k_nope_expected)).item()
        assert max_error_k < 0.1, f"k_nope passthrough error too large: {max_error_k}"
    
    def test_different_batch_sizes(self, device, dtype, cos_sin_cache):
        """Test with various batch sizes."""
        for batch_size in [1, 4, 16, 64, 256]:
            q_nope = torch.randn(batch_size, self.NUM_HEADS, self.KV_LORA_RANK, device=device, dtype=dtype)
            q_rope = torch.randn(batch_size, self.NUM_HEADS, self.QK_ROPE_HEAD_DIM, device=device, dtype=dtype)
            k_nope = torch.randn(batch_size, self.KV_LORA_RANK, device=device, dtype=dtype)
            k_rope = torch.randn(batch_size, self.QK_ROPE_HEAD_DIM, device=device, dtype=dtype)
            
            pos_ids = torch.arange(batch_size, device=device, dtype=torch.int32)
            
            attn_dtype = torch.float8_e4m3fn
            q_out = torch.empty(batch_size, self.NUM_HEADS, self.KV_LORA_RANK + self.QK_ROPE_HEAD_DIM, device=device, dtype=attn_dtype)
            k_rope_out = torch.empty(k_rope.shape, device=device, dtype=attn_dtype)
            k_nope_out = torch.empty(k_nope.shape, device=device, dtype=attn_dtype)
            
            # Should not raise
            flashinfer.rope.mla_rope_quantize_fp8(
                q_rope=q_rope,
                k_rope=k_rope,
                q_nope=q_nope,
                k_nope=k_nope,
                cos_sin_cache=cos_sin_cache,
                pos_ids=pos_ids,
                is_neox=False,
                quantize_dtype=attn_dtype,
                q_rope_out=q_out[..., self.KV_LORA_RANK:],
                k_rope_out=k_rope_out,
                q_nope_out=q_out[..., :self.KV_LORA_RANK],
                k_nope_out=k_nope_out,
                quant_scale_q=1.0,
                quant_scale_kv=1.0,
            )
            
            assert q_out.shape[0] == batch_size, f"Failed for batch_size={batch_size}"
    
    def test_neox_vs_gptj_style(self, device, dtype, cos_sin_cache):
        """Test both NeoX (interleaved) and GPT-J (half rotation) styles."""
        batch_size = 8
        
        for is_neox in [True, False]:
            q_nope = torch.randn(batch_size, self.NUM_HEADS, self.KV_LORA_RANK, device=device, dtype=dtype)
            q_rope = torch.randn(batch_size, self.NUM_HEADS, self.QK_ROPE_HEAD_DIM, device=device, dtype=dtype)
            k_nope = torch.randn(batch_size, self.KV_LORA_RANK, device=device, dtype=dtype)
            k_rope = torch.randn(batch_size, self.QK_ROPE_HEAD_DIM, device=device, dtype=dtype)
            
            pos_ids = torch.arange(batch_size, device=device, dtype=torch.int32)
            
            attn_dtype = torch.float8_e4m3fn
            q_out = torch.empty(batch_size, self.NUM_HEADS, self.KV_LORA_RANK + self.QK_ROPE_HEAD_DIM, device=device, dtype=attn_dtype)
            k_rope_out = torch.empty(k_rope.shape, device=device, dtype=attn_dtype)
            k_nope_out = torch.empty(k_nope.shape, device=device, dtype=attn_dtype)
            
            # Should not raise for either style
            flashinfer.rope.mla_rope_quantize_fp8(
                q_rope=q_rope,
                k_rope=k_rope,
                q_nope=q_nope,
                k_nope=k_nope,
                cos_sin_cache=cos_sin_cache,
                pos_ids=pos_ids,
                is_neox=is_neox,
                quantize_dtype=attn_dtype,
                q_rope_out=q_out[..., self.KV_LORA_RANK:],
                k_rope_out=k_rope_out,
                q_nope_out=q_out[..., :self.KV_LORA_RANK],
                k_nope_out=k_nope_out,
                quant_scale_q=1.0,
                quant_scale_kv=1.0,
            )
            
            assert not torch.isnan(q_out.to(torch.float32)).any(), f"NaN in q_out for is_neox={is_neox}"
    
    def test_position_ids_variation(self, device, dtype, cos_sin_cache):
        """Test with non-sequential position IDs."""
        batch_size = 8
        
        q_nope = torch.randn(batch_size, self.NUM_HEADS, self.KV_LORA_RANK, device=device, dtype=dtype)
        q_rope = torch.randn(batch_size, self.NUM_HEADS, self.QK_ROPE_HEAD_DIM, device=device, dtype=dtype)
        k_nope = torch.randn(batch_size, self.KV_LORA_RANK, device=device, dtype=dtype)
        k_rope = torch.randn(batch_size, self.QK_ROPE_HEAD_DIM, device=device, dtype=dtype)
        
        # Non-sequential position IDs (e.g., from different sequences)
        pos_ids = torch.tensor([100, 200, 300, 400, 500, 600, 700, 800], device=device, dtype=torch.int32)
        
        attn_dtype = torch.float8_e4m3fn
        q_out = torch.empty(batch_size, self.NUM_HEADS, self.KV_LORA_RANK + self.QK_ROPE_HEAD_DIM, device=device, dtype=attn_dtype)
        k_rope_out = torch.empty(k_rope.shape, device=device, dtype=attn_dtype)
        k_nope_out = torch.empty(k_nope.shape, device=device, dtype=attn_dtype)
        
        flashinfer.rope.mla_rope_quantize_fp8(
            q_rope=q_rope,
            k_rope=k_rope,
            q_nope=q_nope,
            k_nope=k_nope,
            cos_sin_cache=cos_sin_cache,
            pos_ids=pos_ids,
            is_neox=False,
            quantize_dtype=attn_dtype,
            q_rope_out=q_out[..., self.KV_LORA_RANK:],
            k_rope_out=k_rope_out,
            q_nope_out=q_out[..., :self.KV_LORA_RANK],
            k_nope_out=k_nope_out,
            quant_scale_q=1.0,
            quant_scale_kv=1.0,
        )
        
        # Verify no NaN/Inf
        assert not torch.isnan(q_out.to(torch.float32)).any()
        assert not torch.isinf(q_out.to(torch.float32)).any()


@pytest.mark.skipif(not _is_cuda, reason="CUDA not available")
@pytest.mark.skipif(not _has_flashinfer, reason="FlashInfer not available")
class TestFusedVsSeparatedPath:
    """
    Compare fused_rope_quantize_and_append_kv_cache vs quantize_and_rope_for_fp8 + set_mla_kv_buffer.
    
    These two paths should produce identical results:
    1. Fused: rope_quantize_fp8_append_paged_kv_cache (writes directly to paged KV cache)
    2. Separated: mla_rope_quantize_fp8 + manual KV cache write
    """
    
    # DeepSeek V3 MLA dimensions
    KV_LORA_RANK = 512
    QK_ROPE_HEAD_DIM = 64
    NUM_HEADS = 16
    PAGE_SIZE = 64
    
    @pytest.fixture
    def device(self):
        return torch.device("cuda")
    
    @pytest.fixture
    def dtype(self):
        return torch.bfloat16
    
    @pytest.fixture
    def cos_sin_cache(self, device):
        max_seq_len = 8192
        return create_cos_sin_cache(max_seq_len, self.QK_ROPE_HEAD_DIM, device)
    
    def _create_paged_kv_cache(self, num_pages: int, device: torch.device) -> tuple:
        """Create a paged KV cache buffer for testing.
        
        Returns:
            tuple: (ckv_cache, kpe_cache) where:
                - ckv_cache: [num_pages, page_size, kv_lora_rank] for k_nope
                - kpe_cache: [num_pages, page_size, qk_rope_head_dim] for k_rope
        """
        attn_dtype = torch.float8_e4m3fn
        ckv_cache = torch.zeros(
            num_pages, self.PAGE_SIZE, self.KV_LORA_RANK,
            device=device, dtype=attn_dtype
        )
        kpe_cache = torch.zeros(
            num_pages, self.PAGE_SIZE, self.QK_ROPE_HEAD_DIM,
            device=device, dtype=attn_dtype
        )
        return ckv_cache, kpe_cache
    
    def _run_fused_path(
        self,
        q_nope: torch.Tensor,
        q_rope: torch.Tensor,
        k_nope: torch.Tensor,
        k_rope: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        pos_ids: torch.Tensor,
        out_cache_loc: torch.Tensor,
        ckv_cache: torch.Tensor,
        kpe_cache: torch.Tensor,
        is_neox: bool = False,
    ) -> torch.Tensor:
        """Run the fused path: rope_quantize_fp8_append_paged_kv_cache.
        
        This kernel performs RoPE, FP8 quantization, and KV cache append in one call.
        """
        nnz = out_cache_loc.shape[0]
        device = out_cache_loc.device
        attn_dtype = torch.float8_e4m3fn
        
        # Allocate q_out
        q_out = torch.empty(
            nnz,
            q_rope.shape[1],
            self.KV_LORA_RANK + self.QK_ROPE_HEAD_DIM,
            device=device,
            dtype=attn_dtype,
        )
        
        # Compute metadata for paged KV cache
        kv_indices = (out_cache_loc // self.PAGE_SIZE).to(torch.int32)
        positions = (out_cache_loc % self.PAGE_SIZE).to(torch.int32)
        kv_indptr = torch.arange(nnz + 1, dtype=torch.int32, device=device)
        batch_indices = torch.arange(nnz, dtype=torch.int32, device=device)
        
        # Call fused kernel
        flashinfer.rope.rope_quantize_fp8_append_paged_kv_cache(
            q_rope=q_rope,
            k_rope=k_rope,
            q_nope=q_nope,
            k_nope=k_nope,
            v=None,  # MLA mode
            cos_sin_cache=cos_sin_cache,
            pos_ids=pos_ids,
            paged_kv_cache=(ckv_cache, kpe_cache),
            kv_indices=kv_indices,
            kv_indptr=kv_indptr,
            batch_indices=batch_indices,
            positions=positions,
            is_neox=is_neox,
            quantize_dtype=attn_dtype,
            quant_scale_q=1.0,
            quant_scale_kv=1.0,
            page_size=self.PAGE_SIZE,
            kv_layout="NHD",
            q_rope_out=q_out[..., self.KV_LORA_RANK:],
            q_nope_out=q_out[..., :self.KV_LORA_RANK],
        )
        
        return q_out
    
    def _run_separated_path(
        self,
        q_nope: torch.Tensor,
        q_rope: torch.Tensor,
        k_nope: torch.Tensor,
        k_rope: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        pos_ids: torch.Tensor,
        out_cache_loc: torch.Tensor,
        ckv_cache: torch.Tensor,
        kpe_cache: torch.Tensor,
        is_neox: bool = False,
    ) -> torch.Tensor:
        """Run the separated path: mla_rope_quantize_fp8 + manual KV cache write.
        
        This is equivalent to:
        1. flashinfer.rope.mla_rope_quantize_fp8 (RoPE + quantize)
        2. token_to_kv_pool.set_mla_kv_buffer (write to cache)
        """
        nnz = q_rope.shape[0]
        device = q_rope.device
        attn_dtype = torch.float8_e4m3fn
        
        # Allocate output tensors
        q_out = torch.empty(
            nnz,
            q_rope.shape[1],
            self.KV_LORA_RANK + self.QK_ROPE_HEAD_DIM,
            device=device,
            dtype=attn_dtype,
        )
        k_rope_out = torch.empty(k_rope.shape, device=device, dtype=attn_dtype)
        k_nope_out = torch.empty(k_nope.shape, device=device, dtype=attn_dtype)
        
        # Step 1: RoPE + quantize
        flashinfer.rope.mla_rope_quantize_fp8(
            q_rope=q_rope,
            k_rope=k_rope,
            q_nope=q_nope,
            k_nope=k_nope,
            cos_sin_cache=cos_sin_cache,
            pos_ids=pos_ids,
            is_neox=is_neox,
            quantize_dtype=attn_dtype,
            q_rope_out=q_out[..., self.KV_LORA_RANK:],
            k_rope_out=k_rope_out,
            q_nope_out=q_out[..., :self.KV_LORA_RANK],
            k_nope_out=k_nope_out,
            quant_scale_q=1.0,
            quant_scale_kv=1.0,
        )
        
        # Step 2: Write to KV cache (simulating set_mla_kv_buffer)
        # The fused kernel writes to paged cache at kv_indices[i], positions[i]
        # We need to do the same manually
        kv_indices = out_cache_loc // self.PAGE_SIZE
        positions = out_cache_loc % self.PAGE_SIZE
        
        for i in range(nnz):
            page_idx = kv_indices[i].item()
            pos_in_page = positions[i].item()
            ckv_cache[page_idx, pos_in_page, :] = k_nope_out[i]
            kpe_cache[page_idx, pos_in_page, :] = k_rope_out[i]
        
        return q_out
    
    def test_q_out_equivalence(self, device, dtype, cos_sin_cache):
        """Test that q_out is identical between fused and separated paths."""
        batch_size = 8
        num_pages = 4
        
        # Create inputs (use same random seed for both paths)
        torch.manual_seed(42)
        q_nope = torch.randn(batch_size, self.NUM_HEADS, self.KV_LORA_RANK, device=device, dtype=dtype)
        q_rope = torch.randn(batch_size, self.NUM_HEADS, self.QK_ROPE_HEAD_DIM, device=device, dtype=dtype)
        k_nope = torch.randn(batch_size, self.KV_LORA_RANK, device=device, dtype=dtype)
        k_rope = torch.randn(batch_size, self.QK_ROPE_HEAD_DIM, device=device, dtype=dtype)
        
        pos_ids = torch.arange(batch_size, device=device, dtype=torch.int32)
        # Scatter tokens across different pages
        out_cache_loc = torch.tensor([0, 65, 130, 195, 32, 97, 162, 227], device=device, dtype=torch.int32)
        
        # Create separate KV caches for each path
        ckv_cache_fused, kpe_cache_fused = self._create_paged_kv_cache(num_pages, device)
        ckv_cache_sep, kpe_cache_sep = self._create_paged_kv_cache(num_pages, device)
        
        # Run fused path
        q_out_fused = self._run_fused_path(
            q_nope, q_rope, k_nope, k_rope,
            cos_sin_cache, pos_ids, out_cache_loc,
            ckv_cache_fused, kpe_cache_fused,
        )
        
        # Run separated path
        q_out_sep = self._run_separated_path(
            q_nope, q_rope, k_nope, k_rope,
            cos_sin_cache, pos_ids, out_cache_loc,
            ckv_cache_sep, kpe_cache_sep,
        )
        
        # Compare q_out
        q_out_fused_float = q_out_fused.to(torch.float32)
        q_out_sep_float = q_out_sep.to(torch.float32)
        
        # They should be identical (same kernel, same inputs)
        max_diff = torch.max(torch.abs(q_out_fused_float - q_out_sep_float)).item()
        assert max_diff == 0.0, f"q_out differs between paths! Max diff: {max_diff}"
    
    def test_kv_cache_equivalence(self, device, dtype, cos_sin_cache):
        """Test that KV cache contents are identical between fused and separated paths."""
        batch_size = 8
        num_pages = 4
        
        torch.manual_seed(42)
        q_nope = torch.randn(batch_size, self.NUM_HEADS, self.KV_LORA_RANK, device=device, dtype=dtype)
        q_rope = torch.randn(batch_size, self.NUM_HEADS, self.QK_ROPE_HEAD_DIM, device=device, dtype=dtype)
        k_nope = torch.randn(batch_size, self.KV_LORA_RANK, device=device, dtype=dtype)
        k_rope = torch.randn(batch_size, self.QK_ROPE_HEAD_DIM, device=device, dtype=dtype)
        
        pos_ids = torch.arange(batch_size, device=device, dtype=torch.int32)
        out_cache_loc = torch.tensor([0, 65, 130, 195, 32, 97, 162, 227], device=device, dtype=torch.int32)
        
        # Create separate KV caches
        ckv_cache_fused, kpe_cache_fused = self._create_paged_kv_cache(num_pages, device)
        ckv_cache_sep, kpe_cache_sep = self._create_paged_kv_cache(num_pages, device)
        
        # Run both paths
        _ = self._run_fused_path(
            q_nope, q_rope, k_nope, k_rope,
            cos_sin_cache, pos_ids, out_cache_loc,
            ckv_cache_fused, kpe_cache_fused,
        )
        
        _ = self._run_separated_path(
            q_nope, q_rope, k_nope, k_rope,
            cos_sin_cache, pos_ids, out_cache_loc,
            ckv_cache_sep, kpe_cache_sep,
        )
        
        # Compare KV caches at the written locations
        kv_indices = out_cache_loc // self.PAGE_SIZE
        positions = out_cache_loc % self.PAGE_SIZE
        
        for i in range(batch_size):
            page_idx = kv_indices[i].item()
            pos_in_page = positions[i].item()
            
            # Compare k_nope (ckv_cache)
            ckv_fused = ckv_cache_fused[page_idx, pos_in_page].to(torch.float32)
            ckv_sep = ckv_cache_sep[page_idx, pos_in_page].to(torch.float32)
            max_diff_ckv = torch.max(torch.abs(ckv_fused - ckv_sep)).item()
            assert max_diff_ckv == 0.0, f"k_nope cache differs at loc {i}! Max diff: {max_diff_ckv}"
            
            # Compare k_rope (kpe_cache)
            kpe_fused = kpe_cache_fused[page_idx, pos_in_page].to(torch.float32)
            kpe_sep = kpe_cache_sep[page_idx, pos_in_page].to(torch.float32)
            max_diff_kpe = torch.max(torch.abs(kpe_fused - kpe_sep)).item()
            assert max_diff_kpe == 0.0, f"k_rope cache differs at loc {i}! Max diff: {max_diff_kpe}"
    
    def test_both_neox_styles(self, device, dtype, cos_sin_cache):
        """Test equivalence for both is_neox=True and is_neox=False."""
        batch_size = 4
        num_pages = 2
        
        for is_neox in [True, False]:
            torch.manual_seed(123)
            q_nope = torch.randn(batch_size, self.NUM_HEADS, self.KV_LORA_RANK, device=device, dtype=dtype)
            q_rope = torch.randn(batch_size, self.NUM_HEADS, self.QK_ROPE_HEAD_DIM, device=device, dtype=dtype)
            k_nope = torch.randn(batch_size, self.KV_LORA_RANK, device=device, dtype=dtype)
            k_rope = torch.randn(batch_size, self.QK_ROPE_HEAD_DIM, device=device, dtype=dtype)
            
            pos_ids = torch.arange(batch_size, device=device, dtype=torch.int32)
            out_cache_loc = torch.tensor([0, 65, 10, 75], device=device, dtype=torch.int32)
            
            ckv_cache_fused, kpe_cache_fused = self._create_paged_kv_cache(num_pages, device)
            ckv_cache_sep, kpe_cache_sep = self._create_paged_kv_cache(num_pages, device)
            
            q_out_fused = self._run_fused_path(
                q_nope, q_rope, k_nope, k_rope,
                cos_sin_cache, pos_ids, out_cache_loc,
                ckv_cache_fused, kpe_cache_fused,
                is_neox=is_neox,
            )
            
            q_out_sep = self._run_separated_path(
                q_nope, q_rope, k_nope, k_rope,
                cos_sin_cache, pos_ids, out_cache_loc,
                ckv_cache_sep, kpe_cache_sep,
                is_neox=is_neox,
            )
            
            max_diff = torch.max(torch.abs(q_out_fused.float() - q_out_sep.float())).item()
            assert max_diff == 0.0, f"q_out differs for is_neox={is_neox}! Max diff: {max_diff}"
    
    def test_various_batch_sizes(self, device, dtype, cos_sin_cache):
        """Test equivalence for various batch sizes."""
        num_pages = 64  # Increased to accommodate larger batch sizes
        
        for batch_size in [1, 4, 16, 64, 128, 256, 512, 1024]:
            torch.manual_seed(456)
            q_nope = torch.randn(batch_size, self.NUM_HEADS, self.KV_LORA_RANK, device=device, dtype=dtype)
            q_rope = torch.randn(batch_size, self.NUM_HEADS, self.QK_ROPE_HEAD_DIM, device=device, dtype=dtype)
            k_nope = torch.randn(batch_size, self.KV_LORA_RANK, device=device, dtype=dtype)
            k_rope = torch.randn(batch_size, self.QK_ROPE_HEAD_DIM, device=device, dtype=dtype)
            
            pos_ids = torch.arange(batch_size, device=device, dtype=torch.int32)
            # Sequential cache locations within available pages
            out_cache_loc = torch.arange(batch_size, device=device, dtype=torch.int32)
            
            ckv_cache_fused, kpe_cache_fused = self._create_paged_kv_cache(num_pages, device)
            ckv_cache_sep, kpe_cache_sep = self._create_paged_kv_cache(num_pages, device)
            
            q_out_fused = self._run_fused_path(
                q_nope, q_rope, k_nope, k_rope,
                cos_sin_cache, pos_ids, out_cache_loc,
                ckv_cache_fused, kpe_cache_fused,
            )
            
            q_out_sep = self._run_separated_path(
                q_nope, q_rope, k_nope, k_rope,
                cos_sin_cache, pos_ids, out_cache_loc,
                ckv_cache_sep, kpe_cache_sep,
            )
            
            max_diff = torch.max(torch.abs(q_out_fused.float() - q_out_sep.float())).item()
            assert max_diff == 0.0, f"q_out differs for batch_size={batch_size}! Max diff: {max_diff}"
    
    def test_deepseek_v3_full_heads(self, device, dtype, cos_sin_cache):
        """Test with DeepSeek V3's full 128 attention heads (no tensor parallelism)."""
        batch_size = 8
        num_pages = 4
        num_heads = 128  # Full DeepSeek V3 head count
        
        torch.manual_seed(789)
        q_nope = torch.randn(batch_size, num_heads, self.KV_LORA_RANK, device=device, dtype=dtype)
        q_rope = torch.randn(batch_size, num_heads, self.QK_ROPE_HEAD_DIM, device=device, dtype=dtype)
        k_nope = torch.randn(batch_size, self.KV_LORA_RANK, device=device, dtype=dtype)
        k_rope = torch.randn(batch_size, self.QK_ROPE_HEAD_DIM, device=device, dtype=dtype)
        
        pos_ids = torch.arange(batch_size, device=device, dtype=torch.int32)
        out_cache_loc = torch.tensor([0, 65, 130, 195, 32, 97, 162, 227], device=device, dtype=torch.int32)
        
        ckv_cache_fused, kpe_cache_fused = self._create_paged_kv_cache(num_pages, device)
        ckv_cache_sep, kpe_cache_sep = self._create_paged_kv_cache(num_pages, device)
        
        q_out_fused = self._run_fused_path(
            q_nope, q_rope, k_nope, k_rope,
            cos_sin_cache, pos_ids, out_cache_loc,
            ckv_cache_fused, kpe_cache_fused,
        )
        
        q_out_sep = self._run_separated_path(
            q_nope, q_rope, k_nope, k_rope,
            cos_sin_cache, pos_ids, out_cache_loc,
            ckv_cache_sep, kpe_cache_sep,
        )
        
        # Verify shape with full heads
        assert q_out_fused.shape == (batch_size, num_heads, self.KV_LORA_RANK + self.QK_ROPE_HEAD_DIM)
        
        max_diff = torch.max(torch.abs(q_out_fused.float() - q_out_sep.float())).item()
        assert max_diff == 0.0, f"q_out differs with full 128 heads! Max diff: {max_diff}"
    
    def test_long_sequence_positions(self, device, dtype, cos_sin_cache):
        """Test with long sequence positions typical of DeepSeek V3's 128K+ context."""
        batch_size = 8
        num_pages = 4
        
        torch.manual_seed(111)
        q_nope = torch.randn(batch_size, self.NUM_HEADS, self.KV_LORA_RANK, device=device, dtype=dtype)
        q_rope = torch.randn(batch_size, self.NUM_HEADS, self.QK_ROPE_HEAD_DIM, device=device, dtype=dtype)
        k_nope = torch.randn(batch_size, self.KV_LORA_RANK, device=device, dtype=dtype)
        k_rope = torch.randn(batch_size, self.QK_ROPE_HEAD_DIM, device=device, dtype=dtype)
        
        # Test with positions deep into a long sequence (within cos_sin_cache range of 8192)
        pos_ids = torch.tensor([1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000], device=device, dtype=torch.int32)
        out_cache_loc = torch.tensor([0, 65, 130, 195, 32, 97, 162, 227], device=device, dtype=torch.int32)
        
        ckv_cache_fused, kpe_cache_fused = self._create_paged_kv_cache(num_pages, device)
        ckv_cache_sep, kpe_cache_sep = self._create_paged_kv_cache(num_pages, device)
        
        q_out_fused = self._run_fused_path(
            q_nope, q_rope, k_nope, k_rope,
            cos_sin_cache, pos_ids, out_cache_loc,
            ckv_cache_fused, kpe_cache_fused,
        )
        
        q_out_sep = self._run_separated_path(
            q_nope, q_rope, k_nope, k_rope,
            cos_sin_cache, pos_ids, out_cache_loc,
            ckv_cache_sep, kpe_cache_sep,
        )
        
        max_diff = torch.max(torch.abs(q_out_fused.float() - q_out_sep.float())).item()
        assert max_diff == 0.0, f"q_out differs with long positions! Max diff: {max_diff}"
    
    def test_sparse_cache_locations(self, device, dtype, cos_sin_cache):
        """Test with sparse (non-contiguous) cache locations across many pages."""
        batch_size = 8
        num_pages = 64  # Many pages for sparse allocation
        
        torch.manual_seed(222)
        q_nope = torch.randn(batch_size, self.NUM_HEADS, self.KV_LORA_RANK, device=device, dtype=dtype)
        q_rope = torch.randn(batch_size, self.NUM_HEADS, self.QK_ROPE_HEAD_DIM, device=device, dtype=dtype)
        k_nope = torch.randn(batch_size, self.KV_LORA_RANK, device=device, dtype=dtype)
        k_rope = torch.randn(batch_size, self.QK_ROPE_HEAD_DIM, device=device, dtype=dtype)
        
        pos_ids = torch.arange(batch_size, device=device, dtype=torch.int32)
        # Sparse locations across different pages (simulating fragmented allocation)
        # page_size=64, so these span pages 0, 3, 7, 15, 31, 40, 50, 62
        out_cache_loc = torch.tensor([5, 200, 450, 1000, 2000, 2600, 3250, 4000], device=device, dtype=torch.int32)
        
        ckv_cache_fused, kpe_cache_fused = self._create_paged_kv_cache(num_pages, device)
        ckv_cache_sep, kpe_cache_sep = self._create_paged_kv_cache(num_pages, device)
        
        q_out_fused = self._run_fused_path(
            q_nope, q_rope, k_nope, k_rope,
            cos_sin_cache, pos_ids, out_cache_loc,
            ckv_cache_fused, kpe_cache_fused,
        )
        
        q_out_sep = self._run_separated_path(
            q_nope, q_rope, k_nope, k_rope,
            cos_sin_cache, pos_ids, out_cache_loc,
            ckv_cache_sep, kpe_cache_sep,
        )
        
        max_diff = torch.max(torch.abs(q_out_fused.float() - q_out_sep.float())).item()
        assert max_diff == 0.0, f"q_out differs with sparse cache locations! Max diff: {max_diff}"
    
    def test_edge_case_values(self, device, dtype, cos_sin_cache):
        """Test with edge case input values: very small, very large, negative."""
        batch_size = 4
        num_pages = 2
        
        # Test 1: Very small values (near zero)
        q_nope = torch.full((batch_size, self.NUM_HEADS, self.KV_LORA_RANK), 1e-6, device=device, dtype=dtype)
        q_rope = torch.full((batch_size, self.NUM_HEADS, self.QK_ROPE_HEAD_DIM), 1e-6, device=device, dtype=dtype)
        k_nope = torch.full((batch_size, self.KV_LORA_RANK), 1e-6, device=device, dtype=dtype)
        k_rope = torch.full((batch_size, self.QK_ROPE_HEAD_DIM), 1e-6, device=device, dtype=dtype)
        
        pos_ids = torch.arange(batch_size, device=device, dtype=torch.int32)
        out_cache_loc = torch.arange(batch_size, device=device, dtype=torch.int32)
        
        ckv_cache_fused, kpe_cache_fused = self._create_paged_kv_cache(num_pages, device)
        ckv_cache_sep, kpe_cache_sep = self._create_paged_kv_cache(num_pages, device)
        
        q_out_fused = self._run_fused_path(
            q_nope, q_rope, k_nope, k_rope,
            cos_sin_cache, pos_ids, out_cache_loc,
            ckv_cache_fused, kpe_cache_fused,
        )
        q_out_sep = self._run_separated_path(
            q_nope, q_rope, k_nope, k_rope,
            cos_sin_cache, pos_ids, out_cache_loc,
            ckv_cache_sep, kpe_cache_sep,
        )
        max_diff = torch.max(torch.abs(q_out_fused.float() - q_out_sep.float())).item()
        assert max_diff == 0.0, f"q_out differs with small values! Max diff: {max_diff}"
        
        # Test 2: Larger values (within FP8 range, max ~448 for e4m3)
        q_nope = torch.full((batch_size, self.NUM_HEADS, self.KV_LORA_RANK), 100.0, device=device, dtype=dtype)
        q_rope = torch.full((batch_size, self.NUM_HEADS, self.QK_ROPE_HEAD_DIM), 100.0, device=device, dtype=dtype)
        k_nope = torch.full((batch_size, self.KV_LORA_RANK), 100.0, device=device, dtype=dtype)
        k_rope = torch.full((batch_size, self.QK_ROPE_HEAD_DIM), 100.0, device=device, dtype=dtype)
        
        ckv_cache_fused, kpe_cache_fused = self._create_paged_kv_cache(num_pages, device)
        ckv_cache_sep, kpe_cache_sep = self._create_paged_kv_cache(num_pages, device)
        
        q_out_fused = self._run_fused_path(
            q_nope, q_rope, k_nope, k_rope,
            cos_sin_cache, pos_ids, out_cache_loc,
            ckv_cache_fused, kpe_cache_fused,
        )
        q_out_sep = self._run_separated_path(
            q_nope, q_rope, k_nope, k_rope,
            cos_sin_cache, pos_ids, out_cache_loc,
            ckv_cache_sep, kpe_cache_sep,
        )
        max_diff = torch.max(torch.abs(q_out_fused.float() - q_out_sep.float())).item()
        assert max_diff == 0.0, f"q_out differs with large values! Max diff: {max_diff}"
        
        # Test 3: Mixed positive/negative values
        torch.manual_seed(333)
        q_nope = torch.randn(batch_size, self.NUM_HEADS, self.KV_LORA_RANK, device=device, dtype=dtype) * 10
        q_rope = torch.randn(batch_size, self.NUM_HEADS, self.QK_ROPE_HEAD_DIM, device=device, dtype=dtype) * 10
        k_nope = torch.randn(batch_size, self.KV_LORA_RANK, device=device, dtype=dtype) * 10
        k_rope = torch.randn(batch_size, self.QK_ROPE_HEAD_DIM, device=device, dtype=dtype) * 10
        
        ckv_cache_fused, kpe_cache_fused = self._create_paged_kv_cache(num_pages, device)
        ckv_cache_sep, kpe_cache_sep = self._create_paged_kv_cache(num_pages, device)
        
        q_out_fused = self._run_fused_path(
            q_nope, q_rope, k_nope, k_rope,
            cos_sin_cache, pos_ids, out_cache_loc,
            ckv_cache_fused, kpe_cache_fused,
        )
        q_out_sep = self._run_separated_path(
            q_nope, q_rope, k_nope, k_rope,
            cos_sin_cache, pos_ids, out_cache_loc,
            ckv_cache_sep, kpe_cache_sep,
        )
        max_diff = torch.max(torch.abs(q_out_fused.float() - q_out_sep.float())).item()
        assert max_diff == 0.0, f"q_out differs with mixed values! Max diff: {max_diff}"
    
    def test_gptj_style_rope_only(self, device, dtype, cos_sin_cache):
        """Verify DeepSeek V3 uses is_neox=False (GPT-J style rotation) exclusively."""
        # DeepSeek V3 uses GPT-J style (is_neox=False), not NeoX style
        # This test ensures we thoroughly test the actual production configuration
        batch_size = 16
        num_pages = 8
        
        torch.manual_seed(444)
        q_nope = torch.randn(batch_size, self.NUM_HEADS, self.KV_LORA_RANK, device=device, dtype=dtype)
        q_rope = torch.randn(batch_size, self.NUM_HEADS, self.QK_ROPE_HEAD_DIM, device=device, dtype=dtype)
        k_nope = torch.randn(batch_size, self.KV_LORA_RANK, device=device, dtype=dtype)
        k_rope = torch.randn(batch_size, self.QK_ROPE_HEAD_DIM, device=device, dtype=dtype)
        
        pos_ids = torch.arange(batch_size, device=device, dtype=torch.int32)
        out_cache_loc = torch.arange(batch_size, device=device, dtype=torch.int32)
        
        ckv_cache_fused, kpe_cache_fused = self._create_paged_kv_cache(num_pages, device)
        ckv_cache_sep, kpe_cache_sep = self._create_paged_kv_cache(num_pages, device)
        
        # GPT-J style (DeepSeek V3's actual configuration)
        q_out_fused = self._run_fused_path(
            q_nope, q_rope, k_nope, k_rope,
            cos_sin_cache, pos_ids, out_cache_loc,
            ckv_cache_fused, kpe_cache_fused,
            is_neox=False,  # DeepSeek V3 uses GPT-J style
        )
        
        q_out_sep = self._run_separated_path(
            q_nope, q_rope, k_nope, k_rope,
            cos_sin_cache, pos_ids, out_cache_loc,
            ckv_cache_sep, kpe_cache_sep,
            is_neox=False,
        )
        
        max_diff = torch.max(torch.abs(q_out_fused.float() - q_out_sep.float())).item()
        assert max_diff == 0.0, f"GPT-J style RoPE differs! Max diff: {max_diff}"
        
        # Also verify outputs are different from NeoX style (sanity check)
        ckv_cache_neox, kpe_cache_neox = self._create_paged_kv_cache(num_pages, device)
        q_out_neox = self._run_fused_path(
            q_nope, q_rope, k_nope, k_rope,
            cos_sin_cache, pos_ids, out_cache_loc,
            ckv_cache_neox, kpe_cache_neox,
            is_neox=True,  # NeoX style for comparison
        )
        
        # The q_rope portion should differ between GPT-J and NeoX styles
        gptj_rope = q_out_fused[..., self.KV_LORA_RANK:].float()
        neox_rope = q_out_neox[..., self.KV_LORA_RANK:].float()
        rope_diff = torch.max(torch.abs(gptj_rope - neox_rope)).item()
        assert rope_diff > 0.0, "GPT-J and NeoX RoPE outputs should differ!"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
