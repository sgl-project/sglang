"""
Consolidated tests for MLA RoPE + FP8 quantization and fused attention pipeline.

Test classes:
    1. TestMLARopeQuantizeFP8:
       Unit tests for the non-fused mla_rope_quantize_fp8 kernel (basic functionality,
       nope passthrough, batch sizes, NeoX vs GPT-J, position IDs).

    2. TestFusedVsSeparatedPath:
       Kernel-level comparison of fused (rope_quantize_fp8_append_paged_kv_cache)
       vs separated (mla_rope_quantize_fp8 + manual KV write) paths. Tests q_out
       equivalence, KV cache equivalence, various batch sizes, edge cases,
       non-contiguous combined buffer slices, no spurious KV cache writes, and
       explicit page boundary positions.

    3. TestMLAAttentionE2E:
       End-to-end fused vs separated with full attention output comparison,
       including multi-page sequences, multi-step decode, multi-iteration without
       sync, cross-sequence isolation, and multi-step KV immutability.

    4. TestMLAAttentionWithCUDAGraphs:
       CUDA graph capture/replay tests with full DeepSeek V3/R1 dimensions (128 heads).
       Multiple replays consistency, various batch sizes (1-2048), rapid-fire stress,
       padded/partial batch sizes (metadata zeroing), and fused graph vs eager
       separated comparison.

    5. TestMLAAttentionNonCUDAGraph:
       Large batch eager-mode tests (512-4096) mimicking production when batch_size
       exceeds cuda_graph_max_bs. Multiple iterations without sync.

    6. TestMLAAttentionMixedModes:
       Graph/eager mode transition tests: alternating modes, consistency between
       graph and eager for same inputs, and rapid mode switching stress.

    7. TestMLAAttentionDraftExtend:
       Speculative decoding (draft_extend) tests with uniform and variable
       accept lengths, multiple tokens per sequence.
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

    Uses concatenated format [cos, sin] matching production code in
    RotaryEmbedding._compute_cos_sin_cache(): first half is cos values,
    second half is sin values.

    Args:
        max_seq_len: Maximum sequence length
        rotary_dim: Dimension of the rotary embedding (qk_rope_head_dim)
        device: Target device

    Returns:
        cos_sin_cache: Tensor of shape [max_seq_len, rotary_dim] containing
                       concatenated [cos0..cosN, sin0..sinN] values.
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

    # Concatenated format: [cos0, cos1, ..., cosN, sin0, sin1, ..., sinN]
    # Matches production RotaryEmbedding._compute_cos_sin_cache()
    cos_sin_cache = torch.cat([cos, sin], dim=-1)

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


    def test_combined_buffer_slices(self, device, dtype, cos_sin_cache):
        """
        Test with COMBINED buffer slices (non-contiguous kpe_cache).
        
        This mimics the production code in trtllm_mla_backend.py which uses:
        - ckv_cache = kv_buffer_paged[:, :, :kv_lora_rank]
        - kpe_cache = kv_buffer_paged[:, :, kv_lora_rank:]
        
        Note: kpe_cache is NOT contiguous because it starts at offset kv_lora_rank.
        This test verifies the FlashInfer kernel handles non-contiguous tensors correctly.
        """
        batch_size = 8
        num_pages = 4
        
        torch.manual_seed(999)
        q_nope = torch.randn(batch_size, self.NUM_HEADS, self.KV_LORA_RANK, device=device, dtype=dtype)
        q_rope = torch.randn(batch_size, self.NUM_HEADS, self.QK_ROPE_HEAD_DIM, device=device, dtype=dtype)
        k_nope = torch.randn(batch_size, self.KV_LORA_RANK, device=device, dtype=dtype)
        k_rope = torch.randn(batch_size, self.QK_ROPE_HEAD_DIM, device=device, dtype=dtype)
        
        pos_ids = torch.arange(batch_size, device=device, dtype=torch.int32)
        out_cache_loc = torch.tensor([0, 65, 130, 195, 32, 97, 162, 227], device=device, dtype=torch.int32)
        
        attn_dtype = torch.float8_e4m3fn
        kv_cache_dim = self.KV_LORA_RANK + self.QK_ROPE_HEAD_DIM
        
        # Create COMBINED buffer (like production)
        combined_buffer = torch.zeros(
            num_pages, self.PAGE_SIZE, kv_cache_dim,
            device=device, dtype=attn_dtype
        )
        
        # Slice into ckv_cache and kpe_cache (like production)
        # NOTE: Both slices are NOT contiguous since they share the last dim of the combined buffer
        ckv_cache_combined = combined_buffer[:, :, :self.KV_LORA_RANK]
        kpe_cache_combined = combined_buffer[:, :, self.KV_LORA_RANK:]
        
        # Verify both slices are NOT contiguous (slicing last dim of a combined buffer)
        # This is the key test condition: the kernel must handle non-contiguous inputs
        assert not ckv_cache_combined.is_contiguous(), "ckv_cache should NOT be contiguous (sliced from combined buffer)"
        assert not kpe_cache_combined.is_contiguous(), "kpe_cache should NOT be contiguous (sliced from combined buffer)"
        
        # Create SEPARATE buffers (like the original tests)
        ckv_cache_sep, kpe_cache_sep = self._create_paged_kv_cache(num_pages, device)
        
        # Run fused path with COMBINED buffer (non-contiguous kpe_cache)
        kv_indices = (out_cache_loc // self.PAGE_SIZE).to(torch.int32)
        positions = (out_cache_loc % self.PAGE_SIZE).to(torch.int32)
        kv_indptr = torch.arange(batch_size + 1, dtype=torch.int32, device=device)
        batch_indices = torch.arange(batch_size, dtype=torch.int32, device=device)
        
        q_out_combined = torch.empty(
            batch_size, self.NUM_HEADS, self.KV_LORA_RANK + self.QK_ROPE_HEAD_DIM,
            device=device, dtype=attn_dtype
        )
        
        flashinfer.rope.rope_quantize_fp8_append_paged_kv_cache(
            q_rope=q_rope, k_rope=k_rope, q_nope=q_nope, k_nope=k_nope,
            v=None, cos_sin_cache=cos_sin_cache, pos_ids=pos_ids,
            paged_kv_cache=(ckv_cache_combined, kpe_cache_combined),
            kv_indices=kv_indices, kv_indptr=kv_indptr,
            batch_indices=batch_indices, positions=positions,
            is_neox=False, quantize_dtype=attn_dtype,
            quant_scale_q=1.0, quant_scale_kv=1.0,
            page_size=self.PAGE_SIZE, kv_layout="NHD",
            q_rope_out=q_out_combined[..., self.KV_LORA_RANK:],
            q_nope_out=q_out_combined[..., :self.KV_LORA_RANK],
        )
        
        # Run fused path with SEPARATE buffers (contiguous)
        q_out_sep = self._run_fused_path(
            q_nope, q_rope, k_nope, k_rope,
            cos_sin_cache, pos_ids, out_cache_loc,
            ckv_cache_sep, kpe_cache_sep,
        )
        
        # Compare q_out
        q_diff = torch.max(torch.abs(q_out_combined.float() - q_out_sep.float())).item()
        assert q_diff == 0.0, f"q_out differs between combined and separate buffers! Max diff: {q_diff}"
        
        # Compare KV cache writes
        for i in range(batch_size):
            page_idx = kv_indices[i].item()
            pos_in_page = positions[i].item()
            
            # Compare ckv (k_nope)
            ckv_combined_val = ckv_cache_combined[page_idx, pos_in_page].float()
            ckv_sep_val = ckv_cache_sep[page_idx, pos_in_page].float()
            ckv_diff = torch.max(torch.abs(ckv_combined_val - ckv_sep_val)).item()
            assert ckv_diff == 0.0, f"ckv cache differs at loc {i}! Max diff: {ckv_diff}"
            
            # Compare kpe (k_rope)
            kpe_combined_val = kpe_cache_combined[page_idx, pos_in_page].float()
            kpe_sep_val = kpe_cache_sep[page_idx, pos_in_page].float()
            kpe_diff = torch.max(torch.abs(kpe_combined_val - kpe_sep_val)).item()
            assert kpe_diff == 0.0, f"kpe cache differs at loc {i}! Max diff: {kpe_diff}"

    def test_no_spurious_kv_cache_writes(self, device, dtype, cos_sin_cache):
        """Verify that the fused kernel only writes to targeted KV cache slots.

        Pre-fill the KV cache with a known sentinel value, run the fused kernel,
        and verify that only the targeted slots were modified while all other
        slots retain the sentinel. This catches off-by-one errors and stray writes.
        """
        batch_size = 8
        num_pages = 8  # More pages than needed to leave many slots untouched

        attn_dtype = torch.float8_e4m3fn
        sentinel_val = torch.tensor(0.5, dtype=torch.float32).to(attn_dtype)

        # Create KV cache filled with sentinel
        ckv_cache = torch.full(
            (num_pages, self.PAGE_SIZE, self.KV_LORA_RANK),
            sentinel_val.item(), device=device, dtype=attn_dtype,
        )
        kpe_cache = torch.full(
            (num_pages, self.PAGE_SIZE, self.QK_ROPE_HEAD_DIM),
            sentinel_val.item(), device=device, dtype=attn_dtype,
        )

        # Snapshot sentinel-filled caches
        ckv_snapshot = ckv_cache.clone()
        kpe_snapshot = kpe_cache.clone()

        torch.manual_seed(777)
        q_nope = torch.randn(batch_size, self.NUM_HEADS, self.KV_LORA_RANK, device=device, dtype=dtype)
        q_rope = torch.randn(batch_size, self.NUM_HEADS, self.QK_ROPE_HEAD_DIM, device=device, dtype=dtype)
        k_nope = torch.randn(batch_size, self.KV_LORA_RANK, device=device, dtype=dtype)
        k_rope = torch.randn(batch_size, self.QK_ROPE_HEAD_DIM, device=device, dtype=dtype)

        pos_ids = torch.arange(batch_size, device=device, dtype=torch.int32)
        # Scatter across several pages, leaving most slots untouched
        out_cache_loc = torch.tensor(
            [5, 70, 135, 200, 265, 330, 395, 460], device=device, dtype=torch.int32
        )

        _ = self._run_fused_path(
            q_nope, q_rope, k_nope, k_rope,
            cos_sin_cache, pos_ids, out_cache_loc,
            ckv_cache, kpe_cache,
        )

        # Build a mask of written locations
        kv_indices = out_cache_loc // self.PAGE_SIZE
        positions_in_page = out_cache_loc % self.PAGE_SIZE
        written_mask = torch.zeros(num_pages, self.PAGE_SIZE, dtype=torch.bool, device=device)
        for i in range(batch_size):
            written_mask[kv_indices[i], positions_in_page[i]] = True

        # All non-written locations must be unchanged
        unwritten_mask = ~written_mask
        assert torch.equal(ckv_cache[unwritten_mask], ckv_snapshot[unwritten_mask]), \
            "Spurious writes detected in ckv_cache!"
        assert torch.equal(kpe_cache[unwritten_mask], kpe_snapshot[unwritten_mask]), \
            "Spurious writes detected in kpe_cache!"

        # All written locations must have been modified
        for i in range(batch_size):
            page = kv_indices[i].item()
            pos = positions_in_page[i].item()
            assert not torch.equal(ckv_cache[page, pos], ckv_snapshot[page, pos]), \
                f"Token {i}: ckv_cache slot was not written!"
            assert not torch.equal(kpe_cache[page, pos], kpe_snapshot[page, pos]), \
                f"Token {i}: kpe_cache slot was not written!"

    def test_page_boundary_positions(self, device, dtype, cos_sin_cache):
        """Test with tokens landing exactly at page boundaries.

        Specifically targets: position 0 (start of page), position 63
        (end of page = PAGE_SIZE - 1), position 64 (start of next page),
        and position 127 (end of second page). These boundary positions
        are where off-by-one bugs in page index calculations surface.
        """
        num_pages = 4

        torch.manual_seed(888)
        boundary_positions = [0, 63, 64, 127]
        batch_size = len(boundary_positions)

        q_nope = torch.randn(batch_size, self.NUM_HEADS, self.KV_LORA_RANK, device=device, dtype=dtype)
        q_rope = torch.randn(batch_size, self.NUM_HEADS, self.QK_ROPE_HEAD_DIM, device=device, dtype=dtype)
        k_nope = torch.randn(batch_size, self.KV_LORA_RANK, device=device, dtype=dtype)
        k_rope = torch.randn(batch_size, self.QK_ROPE_HEAD_DIM, device=device, dtype=dtype)

        pos_ids = torch.tensor(boundary_positions, device=device, dtype=torch.int32)
        out_cache_loc = torch.tensor(boundary_positions, device=device, dtype=torch.int32)

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
        assert max_diff == 0.0, f"Page boundary q_out mismatch! Max diff: {max_diff}"

        # Also verify KV cache writes at boundaries
        kv_indices = out_cache_loc // self.PAGE_SIZE
        pos_in_page = out_cache_loc % self.PAGE_SIZE
        for i in range(batch_size):
            page = kv_indices[i].item()
            pos = pos_in_page[i].item()
            ckv_diff = torch.max(torch.abs(
                ckv_cache_fused[page, pos].float() - ckv_cache_sep[page, pos].float()
            )).item()
            kpe_diff = torch.max(torch.abs(
                kpe_cache_fused[page, pos].float() - kpe_cache_sep[page, pos].float()
            )).item()
            assert ckv_diff == 0.0, f"ckv boundary mismatch at flat_pos={boundary_positions[i]}!"
            assert kpe_diff == 0.0, f"kpe boundary mismatch at flat_pos={boundary_positions[i]}!"

    def test_kv_indptr_contract_mismatch_changes_writes(self, device, dtype, cos_sin_cache):
        """kv_indptr semantics must match MLA one-token-per-batch contract.

        For MLA fused append, production metadata uses:
          - kv_indptr = arange(nnz + 1)
          - batch_indices = arange(nnz)
        so token i maps to kv_indices[i].

        This test intentionally violates kv_indptr semantics (all zeros) while
        keeping all other inputs identical, and verifies that KV writes are
        corrupted. Then it verifies corrected kv_indptr restores expected writes.
        """
        batch_size = 8
        num_pages = 12
        attn_dtype = torch.float8_e4m3fn

        torch.manual_seed(7777)
        q_nope = torch.randn(batch_size, self.NUM_HEADS, self.KV_LORA_RANK, device=device, dtype=dtype)
        q_rope = torch.randn(batch_size, self.NUM_HEADS, self.QK_ROPE_HEAD_DIM, device=device, dtype=dtype)
        k_nope = torch.randn(batch_size, self.KV_LORA_RANK, device=device, dtype=dtype)
        k_rope = torch.randn(batch_size, self.QK_ROPE_HEAD_DIM, device=device, dtype=dtype)
        pos_ids = torch.arange(batch_size, device=device, dtype=torch.int32)

        # Route active writes to pages 1..batch_size (leave page 0 unused here).
        pages = torch.arange(1, batch_size + 1, device=device, dtype=torch.int32)
        positions = torch.arange(batch_size, device=device, dtype=torch.int32)
        out_cache_loc = pages * self.PAGE_SIZE + positions
        kv_indices = (out_cache_loc // self.PAGE_SIZE).to(torch.int32)
        positions_meta = (out_cache_loc % self.PAGE_SIZE).to(torch.int32)
        batch_indices = torch.arange(batch_size, dtype=torch.int32, device=device)

        # Correct contract metadata.
        good_kv_indptr = torch.arange(batch_size + 1, dtype=torch.int32, device=device)

        # Bad contract metadata: all tokens map to the first kv_indptr segment.
        # This should redirect writes away from intended pages.
        bad_kv_indptr = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)

        # Correct run
        ckv_good, kpe_good = self._create_paged_kv_cache(num_pages, device)
        q_out_good = torch.empty(
            batch_size,
            self.NUM_HEADS,
            self.KV_LORA_RANK + self.QK_ROPE_HEAD_DIM,
            device=device,
            dtype=attn_dtype,
        )
        flashinfer.rope.rope_quantize_fp8_append_paged_kv_cache(
            q_rope=q_rope, k_rope=k_rope, q_nope=q_nope, k_nope=k_nope,
            v=None, cos_sin_cache=cos_sin_cache, pos_ids=pos_ids,
            paged_kv_cache=(ckv_good, kpe_good),
            kv_indices=kv_indices, kv_indptr=good_kv_indptr,
            batch_indices=batch_indices, positions=positions_meta,
            is_neox=False, quantize_dtype=attn_dtype,
            quant_scale_q=1.0, quant_scale_kv=1.0,
            page_size=self.PAGE_SIZE, kv_layout="NHD",
            q_rope_out=q_out_good[..., self.KV_LORA_RANK:],
            q_nope_out=q_out_good[..., :self.KV_LORA_RANK],
        )

        # Mismatched kv_indptr run
        ckv_bad, kpe_bad = self._create_paged_kv_cache(num_pages, device)
        q_out_bad = torch.empty_like(q_out_good)
        flashinfer.rope.rope_quantize_fp8_append_paged_kv_cache(
            q_rope=q_rope, k_rope=k_rope, q_nope=q_nope, k_nope=k_nope,
            v=None, cos_sin_cache=cos_sin_cache, pos_ids=pos_ids,
            paged_kv_cache=(ckv_bad, kpe_bad),
            kv_indices=kv_indices, kv_indptr=bad_kv_indptr,
            batch_indices=batch_indices, positions=positions_meta,
            is_neox=False, quantize_dtype=attn_dtype,
            quant_scale_q=1.0, quant_scale_kv=1.0,
            page_size=self.PAGE_SIZE, kv_layout="NHD",
            q_rope_out=q_out_bad[..., self.KV_LORA_RANK:],
            q_nope_out=q_out_bad[..., :self.KV_LORA_RANK],
        )
        torch.cuda.synchronize()

        # q_out is independent of paged KV metadata and should remain identical.
        q_diff = torch.max(torch.abs(q_out_good.float() - q_out_bad.float())).item()
        assert q_diff == 0.0, f"q_out unexpectedly changed by kv_indptr mismatch! max_diff={q_diff}"

        # KV cache writes must differ under bad kv_indptr semantics.
        ckv_mismatch = torch.max(torch.abs(ckv_good.float() - ckv_bad.float())).item()
        kpe_mismatch = torch.max(torch.abs(kpe_good.float() - kpe_bad.float())).item()
        assert ckv_mismatch > 0.0 or kpe_mismatch > 0.0, (
            "Expected KV cache corruption with bad kv_indptr semantics, but saw no difference."
        )

        # Corrected kv_indptr should reproduce the correct run exactly.
        ckv_fixed, kpe_fixed = self._create_paged_kv_cache(num_pages, device)
        q_out_fixed = torch.empty_like(q_out_good)
        flashinfer.rope.rope_quantize_fp8_append_paged_kv_cache(
            q_rope=q_rope, k_rope=k_rope, q_nope=q_nope, k_nope=k_nope,
            v=None, cos_sin_cache=cos_sin_cache, pos_ids=pos_ids,
            paged_kv_cache=(ckv_fixed, kpe_fixed),
            kv_indices=kv_indices, kv_indptr=good_kv_indptr,
            batch_indices=batch_indices, positions=positions_meta,
            is_neox=False, quantize_dtype=attn_dtype,
            quant_scale_q=1.0, quant_scale_kv=1.0,
            page_size=self.PAGE_SIZE, kv_layout="NHD",
            q_rope_out=q_out_fixed[..., self.KV_LORA_RANK:],
            q_nope_out=q_out_fixed[..., :self.KV_LORA_RANK],
        )
        torch.cuda.synchronize()

        q_fixed_diff = torch.max(torch.abs(q_out_good.float() - q_out_fixed.float())).item()
        ckv_fixed_diff = torch.max(torch.abs(ckv_good.float() - ckv_fixed.float())).item()
        kpe_fixed_diff = torch.max(torch.abs(kpe_good.float() - kpe_fixed.float())).item()
        assert q_fixed_diff == 0.0, f"q_out mismatch after kv_indptr fix! max_diff={q_fixed_diff}"
        assert ckv_fixed_diff == 0.0, f"ckv mismatch after kv_indptr fix! max_diff={ckv_fixed_diff}"
        assert kpe_fixed_diff == 0.0, f"kpe mismatch after kv_indptr fix! max_diff={kpe_fixed_diff}"


###############################################################################
# End-to-end tests for MLA attention path with fused vs separated kernels.
#
# This tests the complete attention pipeline:
# 1. RoPE + FP8 quantization + KV cache write (fused or separated)
# 2. Attention computation (trtllm_batch_decode_with_kv_cache_mla)
# 3. Comparison of final attention outputs
#
# This catches issues that unit tests might miss, such as:
# - Race conditions between the RoPE kernel and attention kernel
# - Data format mismatches in the KV cache
# - Incorrect KV cache writes affecting attention output
###############################################################################


@pytest.mark.skipif(not _is_cuda, reason="CUDA not available")
@pytest.mark.skipif(not _has_flashinfer, reason="FlashInfer not available")
class TestMLAAttentionE2E:
    """
    End-to-end tests for the MLA attention path.

    Tests the complete pipeline:
    1. Fused path: rope_quantize_fp8_append_paged_kv_cache -> trtllm_batch_decode_with_kv_cache_mla
    2. Separated path: mla_rope_quantize_fp8 -> manual KV write -> trtllm_batch_decode_with_kv_cache_mla

    Both paths should produce identical attention outputs.
    """

    # DeepSeek V3 MLA dimensions
    KV_LORA_RANK = 512
    QK_ROPE_HEAD_DIM = 64
    QK_NOPE_HEAD_DIM = 128  # For trtllm attention kernel
    NUM_HEADS = 16  # Reduced for testing
    PAGE_SIZE = 64
    HEAD_DIM = KV_LORA_RANK + QK_ROPE_HEAD_DIM  # 576

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

    @pytest.fixture
    def workspace_buffer(self, device):
        """Create workspace buffer for trtllm attention kernel."""
        # Allocate a generous workspace (128MB)
        return torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)

    def _create_paged_kv_cache(self, num_pages: int, device: torch.device) -> tuple:
        """Create a paged KV cache buffer for testing.

        For MLA, the KV cache stores:
        - ckv_cache: [num_pages, page_size, kv_lora_rank] for compressed KV (k_nope)
        - kpe_cache: [num_pages, page_size, qk_rope_head_dim] for k_rope

        Returns:
            tuple: (ckv_cache, kpe_cache)
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

    def _create_combined_kv_cache(self, num_pages: int, device: torch.device) -> torch.Tensor:
        """Create combined KV cache for trtllm attention.

        The trtllm attention kernel expects kv_cache of shape:
        [num_pages, 1, page_size, kv_lora_rank + qk_rope_head_dim]

        The extra dimension (1) is for compatibility with the kernel interface.
        """
        attn_dtype = torch.float8_e4m3fn
        kv_cache_dim = self.KV_LORA_RANK + self.QK_ROPE_HEAD_DIM
        kv_cache = torch.zeros(
            num_pages, 1, self.PAGE_SIZE, kv_cache_dim,
            device=device, dtype=attn_dtype
        )
        return kv_cache

    def _run_fused_path(
        self,
        q_nope: torch.Tensor,
        q_rope: torch.Tensor,
        k_nope: torch.Tensor,
        k_rope: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        pos_ids: torch.Tensor,
        out_cache_loc: torch.Tensor,
        kv_cache: torch.Tensor,  # Combined KV cache
        is_neox: bool = False,
    ) -> torch.Tensor:
        """Run the fused path and return q_out (query for attention).

        This uses rope_quantize_fp8_append_paged_kv_cache to write directly
        to the KV cache in a single kernel call.
        """
        nnz = out_cache_loc.shape[0]
        device = out_cache_loc.device
        attn_dtype = torch.float8_e4m3fn

        # Allocate q_out
        q_out = torch.empty(
            nnz,
            q_rope.shape[1],  # num_heads
            self.KV_LORA_RANK + self.QK_ROPE_HEAD_DIM,
            device=device,
            dtype=attn_dtype,
        )

        # Compute metadata for paged KV cache
        kv_indices = (out_cache_loc // self.PAGE_SIZE).to(torch.int32)
        positions = (out_cache_loc % self.PAGE_SIZE).to(torch.int32)
        kv_indptr = torch.arange(nnz + 1, dtype=torch.int32, device=device)
        batch_indices = torch.arange(nnz, dtype=torch.int32, device=device)

        # Split KV cache into ckv and kpe views for the fused kernel
        # kv_cache shape: [num_pages, 1, page_size, kv_cache_dim]
        # We need: ckv [num_pages, page_size, kv_lora_rank], kpe [num_pages, page_size, qk_rope_head_dim]
        kv_cache_squeezed = kv_cache.squeeze(1)  # [num_pages, page_size, kv_cache_dim]
        ckv_cache = kv_cache_squeezed[:, :, :self.KV_LORA_RANK]
        kpe_cache = kv_cache_squeezed[:, :, self.KV_LORA_RANK:]

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

        # Copy written values back to the combined kv_cache
        # This is necessary because we created contiguous views
        kv_cache_squeezed[:, :, :self.KV_LORA_RANK] = ckv_cache
        kv_cache_squeezed[:, :, self.KV_LORA_RANK:] = kpe_cache

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
        kv_cache: torch.Tensor,  # Combined KV cache
        is_neox: bool = False,
    ) -> torch.Tensor:
        """Run the separated path and return q_out (query for attention).

        This uses mla_rope_quantize_fp8 followed by manual KV cache write.
        """
        nnz = q_rope.shape[0]
        device = q_rope.device
        attn_dtype = torch.float8_e4m3fn

        # Allocate output tensors
        q_out = torch.empty(
            nnz,
            q_rope.shape[1],  # num_heads
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

        # Step 2: Write to KV cache
        kv_indices = out_cache_loc // self.PAGE_SIZE
        positions_in_page = out_cache_loc % self.PAGE_SIZE

        # kv_cache shape: [num_pages, 1, page_size, kv_cache_dim]
        kv_cache_squeezed = kv_cache.squeeze(1)  # [num_pages, page_size, kv_cache_dim]

        for i in range(nnz):
            page_idx = kv_indices[i].item()
            pos_in_page = positions_in_page[i].item()
            # Write k_nope to ckv portion
            kv_cache_squeezed[page_idx, pos_in_page, :self.KV_LORA_RANK] = k_nope_out[i]
            # Write k_rope to kpe portion
            kv_cache_squeezed[page_idx, pos_in_page, self.KV_LORA_RANK:] = k_rope_out[i]

        return q_out

    def _run_attention(
        self,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        workspace_buffer: torch.Tensor,
        block_tables: torch.Tensor,
        seq_lens: torch.Tensor,
        max_seq_len: int,
        bmm1_scale: float = 1.0,
    ) -> torch.Tensor:
        """Run the TRT-LLM MLA attention kernel."""
        return flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla(
            query=query,
            kv_cache=kv_cache,
            workspace_buffer=workspace_buffer,
            qk_nope_head_dim=self.QK_NOPE_HEAD_DIM,
            kv_lora_rank=self.KV_LORA_RANK,
            qk_rope_head_dim=self.QK_ROPE_HEAD_DIM,
            block_tables=block_tables,
            seq_lens=seq_lens,
            max_seq_len=max_seq_len,
            bmm1_scale=bmm1_scale,
        )

    def test_attention_output_equivalence(self, device, dtype, cos_sin_cache, workspace_buffer):
        """Test that attention outputs are identical for fused vs separated paths."""
        batch_size = 4
        num_pages = 8

        # Each sequence has 64 tokens (1 page worth) for simplicity
        seq_lens = torch.full((batch_size,), 64, dtype=torch.int32, device=device)
        max_seq_len = 64

        # Create inputs
        torch.manual_seed(42)
        q_nope = torch.randn(batch_size, self.NUM_HEADS, self.KV_LORA_RANK, device=device, dtype=dtype)
        q_rope = torch.randn(batch_size, self.NUM_HEADS, self.QK_ROPE_HEAD_DIM, device=device, dtype=dtype)
        k_nope = torch.randn(batch_size, self.KV_LORA_RANK, device=device, dtype=dtype)
        k_rope = torch.randn(batch_size, self.QK_ROPE_HEAD_DIM, device=device, dtype=dtype)

        pos_ids = torch.arange(batch_size, device=device, dtype=torch.int32)
        out_cache_loc = torch.tensor([63, 127, 191, 255], device=device, dtype=torch.int32)

        block_tables = torch.tensor([
            [0, -1], [1, -1], [2, -1], [3, -1],
        ], dtype=torch.int32, device=device)

        kv_cache_fused = self._create_combined_kv_cache(num_pages, device)
        kv_cache_sep = self._create_combined_kv_cache(num_pages, device)

        # Pre-populate KV caches with some existing context (positions 0-62)
        torch.manual_seed(100)
        for page_idx in range(4):
            for pos in range(63):
                k_nope_prev = torch.randn(self.KV_LORA_RANK, device=device, dtype=dtype)
                k_rope_prev = torch.randn(self.QK_ROPE_HEAD_DIM, device=device, dtype=dtype)
                k_nope_fp8 = k_nope_prev.to(torch.float8_e4m3fn)
                k_rope_fp8 = k_rope_prev.to(torch.float8_e4m3fn)
                kv_cache_fused[page_idx, 0, pos, :self.KV_LORA_RANK] = k_nope_fp8
                kv_cache_fused[page_idx, 0, pos, self.KV_LORA_RANK:] = k_rope_fp8
                kv_cache_sep[page_idx, 0, pos, :self.KV_LORA_RANK] = k_nope_fp8
                kv_cache_sep[page_idx, 0, pos, self.KV_LORA_RANK:] = k_rope_fp8

        q_out_fused = self._run_fused_path(
            q_nope, q_rope, k_nope, k_rope,
            cos_sin_cache, pos_ids, out_cache_loc, kv_cache_fused,
        )
        q_out_sep = self._run_separated_path(
            q_nope, q_rope, k_nope, k_rope,
            cos_sin_cache, pos_ids, out_cache_loc, kv_cache_sep,
        )

        q_diff = torch.max(torch.abs(q_out_fused.float() - q_out_sep.float())).item()
        assert q_diff == 0.0, f"q_out differs! Max diff: {q_diff}"

        kv_diff = torch.max(torch.abs(kv_cache_fused.float() - kv_cache_sep.float())).item()
        assert kv_diff == 0.0, f"KV cache differs! Max diff: {kv_diff}"

        query_fused = q_out_fused.unsqueeze(1)
        query_sep = q_out_sep.unsqueeze(1)

        attn_out_fused = self._run_attention(
            query=query_fused, kv_cache=kv_cache_fused,
            workspace_buffer=workspace_buffer, block_tables=block_tables,
            seq_lens=seq_lens, max_seq_len=max_seq_len,
            bmm1_scale=1.0 / (self.HEAD_DIM ** 0.5),
        )
        attn_out_sep = self._run_attention(
            query=query_sep, kv_cache=kv_cache_sep,
            workspace_buffer=workspace_buffer, block_tables=block_tables,
            seq_lens=seq_lens, max_seq_len=max_seq_len,
            bmm1_scale=1.0 / (self.HEAD_DIM ** 0.5),
        )

        max_attn_diff = torch.max(torch.abs(attn_out_fused.float() - attn_out_sep.float())).item()
        assert max_attn_diff == 0.0, f"Attention outputs differ! Max diff: {max_attn_diff}"

    def test_attention_with_longer_sequences(self, device, dtype, cos_sin_cache, workspace_buffer):
        """Test attention with longer sequences spanning multiple pages."""
        batch_size = 2
        num_pages = 16

        seq_lens = torch.full((batch_size,), 256, dtype=torch.int32, device=device)
        max_seq_len = 256

        torch.manual_seed(123)
        q_nope = torch.randn(batch_size, self.NUM_HEADS, self.KV_LORA_RANK, device=device, dtype=dtype)
        q_rope = torch.randn(batch_size, self.NUM_HEADS, self.QK_ROPE_HEAD_DIM, device=device, dtype=dtype)
        k_nope = torch.randn(batch_size, self.KV_LORA_RANK, device=device, dtype=dtype)
        k_rope = torch.randn(batch_size, self.QK_ROPE_HEAD_DIM, device=device, dtype=dtype)

        pos_ids = torch.tensor([255, 255], device=device, dtype=torch.int32)
        out_cache_loc = torch.tensor([
            3 * self.PAGE_SIZE + 63,
            7 * self.PAGE_SIZE + 63,
        ], device=device, dtype=torch.int32)

        block_tables = torch.tensor([
            [0, 1, 2, 3], [4, 5, 6, 7],
        ], dtype=torch.int32, device=device)

        kv_cache_fused = self._create_combined_kv_cache(num_pages, device)
        kv_cache_sep = self._create_combined_kv_cache(num_pages, device)

        torch.manual_seed(200)
        for seq_idx in range(batch_size):
            base_page = seq_idx * 4
            for pos in range(255):
                page_offset = pos // self.PAGE_SIZE
                pos_in_page = pos % self.PAGE_SIZE
                page_idx = base_page + page_offset
                k_nope_prev = torch.randn(self.KV_LORA_RANK, device=device, dtype=dtype)
                k_rope_prev = torch.randn(self.QK_ROPE_HEAD_DIM, device=device, dtype=dtype)
                k_nope_fp8 = k_nope_prev.to(torch.float8_e4m3fn)
                k_rope_fp8 = k_rope_prev.to(torch.float8_e4m3fn)
                kv_cache_fused[page_idx, 0, pos_in_page, :self.KV_LORA_RANK] = k_nope_fp8
                kv_cache_fused[page_idx, 0, pos_in_page, self.KV_LORA_RANK:] = k_rope_fp8
                kv_cache_sep[page_idx, 0, pos_in_page, :self.KV_LORA_RANK] = k_nope_fp8
                kv_cache_sep[page_idx, 0, pos_in_page, self.KV_LORA_RANK:] = k_rope_fp8

        q_out_fused = self._run_fused_path(
            q_nope, q_rope, k_nope, k_rope,
            cos_sin_cache, pos_ids, out_cache_loc, kv_cache_fused,
        )
        q_out_sep = self._run_separated_path(
            q_nope, q_rope, k_nope, k_rope,
            cos_sin_cache, pos_ids, out_cache_loc, kv_cache_sep,
        )

        query_fused = q_out_fused.unsqueeze(1)
        query_sep = q_out_sep.unsqueeze(1)

        attn_out_fused = self._run_attention(
            query=query_fused, kv_cache=kv_cache_fused,
            workspace_buffer=workspace_buffer, block_tables=block_tables,
            seq_lens=seq_lens, max_seq_len=max_seq_len,
            bmm1_scale=1.0 / (self.HEAD_DIM ** 0.5),
        )
        attn_out_sep = self._run_attention(
            query=query_sep, kv_cache=kv_cache_sep,
            workspace_buffer=workspace_buffer, block_tables=block_tables,
            seq_lens=seq_lens, max_seq_len=max_seq_len,
            bmm1_scale=1.0 / (self.HEAD_DIM ** 0.5),
        )

        max_attn_diff = torch.max(torch.abs(attn_out_fused.float() - attn_out_sep.float())).item()
        assert max_attn_diff == 0.0, f"Attention outputs differ with long sequences! Max diff: {max_attn_diff}"

    def test_attention_multiple_decode_steps(self, device, dtype, cos_sin_cache, workspace_buffer):
        """Test multiple consecutive decode steps to catch cumulative errors."""
        batch_size = 2
        num_pages = 8
        num_decode_steps = 16

        kv_cache_fused = self._create_combined_kv_cache(num_pages, device)
        kv_cache_sep = self._create_combined_kv_cache(num_pages, device)

        block_tables = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32, device=device)

        torch.manual_seed(300)
        for seq_idx in range(batch_size):
            base_page = seq_idx * 2
            for pos in range(48):
                page_offset = pos // self.PAGE_SIZE
                pos_in_page = pos % self.PAGE_SIZE
                page_idx = base_page + page_offset
                k_nope_prev = torch.randn(self.KV_LORA_RANK, device=device, dtype=dtype)
                k_rope_prev = torch.randn(self.QK_ROPE_HEAD_DIM, device=device, dtype=dtype)
                k_nope_fp8 = k_nope_prev.to(torch.float8_e4m3fn)
                k_rope_fp8 = k_rope_prev.to(torch.float8_e4m3fn)
                kv_cache_fused[page_idx, 0, pos_in_page, :self.KV_LORA_RANK] = k_nope_fp8
                kv_cache_fused[page_idx, 0, pos_in_page, self.KV_LORA_RANK:] = k_rope_fp8
                kv_cache_sep[page_idx, 0, pos_in_page, :self.KV_LORA_RANK] = k_nope_fp8
                kv_cache_sep[page_idx, 0, pos_in_page, self.KV_LORA_RANK:] = k_rope_fp8

        for step in range(num_decode_steps):
            current_pos = 48 + step
            seq_len = current_pos + 1

            torch.manual_seed(400 + step)
            q_nope = torch.randn(batch_size, self.NUM_HEADS, self.KV_LORA_RANK, device=device, dtype=dtype)
            q_rope = torch.randn(batch_size, self.NUM_HEADS, self.QK_ROPE_HEAD_DIM, device=device, dtype=dtype)
            k_nope = torch.randn(batch_size, self.KV_LORA_RANK, device=device, dtype=dtype)
            k_rope = torch.randn(batch_size, self.QK_ROPE_HEAD_DIM, device=device, dtype=dtype)

            pos_ids = torch.full((batch_size,), current_pos, device=device, dtype=torch.int32)
            seq_lens = torch.full((batch_size,), seq_len, dtype=torch.int32, device=device)

            page_offset = current_pos // self.PAGE_SIZE
            pos_in_page = current_pos % self.PAGE_SIZE
            out_cache_loc = torch.tensor([
                (0 + page_offset) * self.PAGE_SIZE + pos_in_page,
                (2 + page_offset) * self.PAGE_SIZE + pos_in_page,
            ], device=device, dtype=torch.int32)

            q_out_fused = self._run_fused_path(
                q_nope, q_rope, k_nope, k_rope,
                cos_sin_cache, pos_ids, out_cache_loc, kv_cache_fused,
            )
            q_out_sep = self._run_separated_path(
                q_nope, q_rope, k_nope, k_rope,
                cos_sin_cache, pos_ids, out_cache_loc, kv_cache_sep,
            )

            query_fused = q_out_fused.unsqueeze(1)
            query_sep = q_out_sep.unsqueeze(1)

            attn_out_fused = self._run_attention(
                query=query_fused, kv_cache=kv_cache_fused,
                workspace_buffer=workspace_buffer, block_tables=block_tables,
                seq_lens=seq_lens, max_seq_len=seq_len,
                bmm1_scale=1.0 / (self.HEAD_DIM ** 0.5),
            )
            attn_out_sep = self._run_attention(
                query=query_sep, kv_cache=kv_cache_sep,
                workspace_buffer=workspace_buffer, block_tables=block_tables,
                seq_lens=seq_lens, max_seq_len=seq_len,
                bmm1_scale=1.0 / (self.HEAD_DIM ** 0.5),
            )

            max_attn_diff = torch.max(torch.abs(attn_out_fused.float() - attn_out_sep.float())).item()
            assert max_attn_diff == 0.0, f"Attention differs at step {step}! Max diff: {max_attn_diff}"

    def test_attention_without_sync_vs_with_sync(self, device, dtype, cos_sin_cache, workspace_buffer):
        """Compare attention outputs with and without explicit synchronization."""
        batch_size = 4
        num_pages = 8

        seq_lens = torch.full((batch_size,), 64, dtype=torch.int32, device=device)
        max_seq_len = 64

        torch.manual_seed(700)
        q_nope = torch.randn(batch_size, self.NUM_HEADS, self.KV_LORA_RANK, device=device, dtype=dtype)
        q_rope = torch.randn(batch_size, self.NUM_HEADS, self.QK_ROPE_HEAD_DIM, device=device, dtype=dtype)
        k_nope = torch.randn(batch_size, self.KV_LORA_RANK, device=device, dtype=dtype)
        k_rope = torch.randn(batch_size, self.QK_ROPE_HEAD_DIM, device=device, dtype=dtype)

        pos_ids = torch.arange(batch_size, device=device, dtype=torch.int32)
        out_cache_loc = torch.tensor([63, 127, 191, 255], device=device, dtype=torch.int32)
        block_tables = torch.tensor([
            [0, -1], [1, -1], [2, -1], [3, -1],
        ], dtype=torch.int32, device=device)

        # Run without sync
        kv_cache_no_sync = self._create_combined_kv_cache(num_pages, device)
        torch.manual_seed(800)
        for page_idx in range(4):
            for pos in range(63):
                k_nope_prev = torch.randn(self.KV_LORA_RANK, device=device, dtype=dtype)
                k_rope_prev = torch.randn(self.QK_ROPE_HEAD_DIM, device=device, dtype=dtype)
                kv_cache_no_sync[page_idx, 0, pos, :self.KV_LORA_RANK] = k_nope_prev.to(torch.float8_e4m3fn)
                kv_cache_no_sync[page_idx, 0, pos, self.KV_LORA_RANK:] = k_rope_prev.to(torch.float8_e4m3fn)

        q_out_no_sync = self._run_fused_path(
            q_nope, q_rope, k_nope, k_rope,
            cos_sin_cache, pos_ids, out_cache_loc, kv_cache_no_sync,
        )
        query_no_sync = q_out_no_sync.unsqueeze(1)
        attn_out_no_sync = self._run_attention(
            query=query_no_sync, kv_cache=kv_cache_no_sync,
            workspace_buffer=workspace_buffer, block_tables=block_tables,
            seq_lens=seq_lens, max_seq_len=max_seq_len,
            bmm1_scale=1.0 / (self.HEAD_DIM ** 0.5),
        )
        attn_out_no_sync_copy = attn_out_no_sync.clone()
        torch.cuda.synchronize()

        # Run with sync
        kv_cache_with_sync = self._create_combined_kv_cache(num_pages, device)
        torch.manual_seed(800)
        for page_idx in range(4):
            for pos in range(63):
                k_nope_prev = torch.randn(self.KV_LORA_RANK, device=device, dtype=dtype)
                k_rope_prev = torch.randn(self.QK_ROPE_HEAD_DIM, device=device, dtype=dtype)
                kv_cache_with_sync[page_idx, 0, pos, :self.KV_LORA_RANK] = k_nope_prev.to(torch.float8_e4m3fn)
                kv_cache_with_sync[page_idx, 0, pos, self.KV_LORA_RANK:] = k_rope_prev.to(torch.float8_e4m3fn)

        q_out_with_sync = self._run_fused_path(
            q_nope, q_rope, k_nope, k_rope,
            cos_sin_cache, pos_ids, out_cache_loc, kv_cache_with_sync,
        )
        torch.cuda.synchronize()
        query_with_sync = q_out_with_sync.unsqueeze(1)
        attn_out_with_sync = self._run_attention(
            query=query_with_sync, kv_cache=kv_cache_with_sync,
            workspace_buffer=workspace_buffer, block_tables=block_tables,
            seq_lens=seq_lens, max_seq_len=max_seq_len,
            bmm1_scale=1.0 / (self.HEAD_DIM ** 0.5),
        )
        torch.cuda.synchronize()

        max_diff = torch.max(torch.abs(attn_out_no_sync_copy.float() - attn_out_with_sync.float())).item()
        assert max_diff == 0.0, (
            f"Attention outputs differ with vs without sync! Max diff: {max_diff}\n"
            "This indicates a potential race condition between RoPE and attention kernels."
        )

    def test_multi_iteration_fused_vs_separated_no_sync(
        self, device, dtype, cos_sin_cache, workspace_buffer
    ):
        """Test many decode iterations comparing fused vs separated WITHOUT synchronization.

        In production there is no cuda.synchronize() between decode iterations.
        This test runs 32 back-to-back iterations and checks that the fused and
        separated paths produce identical attention outputs at every step, even
        when GPU operations overlap.
        """
        batch_size = 4
        num_pages = 16
        num_iterations = 32

        kv_cache_fused = self._create_combined_kv_cache(num_pages, device)
        kv_cache_sep = self._create_combined_kv_cache(num_pages, device)

        block_tables = torch.tensor(
            [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]],
            dtype=torch.int32, device=device,
        )

        # Pre-populate with shared context (96 tokens per sequence, ~1.5 pages)
        torch.manual_seed(14000)
        for seq_idx in range(batch_size):
            base_page = seq_idx * 4
            for pos in range(96):
                page_offset = pos // self.PAGE_SIZE
                pos_in_page = pos % self.PAGE_SIZE
                page_idx = base_page + page_offset
                k_nope_prev = torch.randn(self.KV_LORA_RANK, device=device, dtype=dtype)
                k_rope_prev = torch.randn(self.QK_ROPE_HEAD_DIM, device=device, dtype=dtype)
                k_nope_fp8 = k_nope_prev.to(torch.float8_e4m3fn)
                k_rope_fp8 = k_rope_prev.to(torch.float8_e4m3fn)
                kv_cache_fused[page_idx, 0, pos_in_page, :self.KV_LORA_RANK] = k_nope_fp8
                kv_cache_fused[page_idx, 0, pos_in_page, self.KV_LORA_RANK:] = k_rope_fp8
                kv_cache_sep[page_idx, 0, pos_in_page, :self.KV_LORA_RANK] = k_nope_fp8
                kv_cache_sep[page_idx, 0, pos_in_page, self.KV_LORA_RANK:] = k_rope_fp8

        max_diffs = []
        for step in range(num_iterations):
            current_pos = 96 + step
            seq_len = current_pos + 1

            torch.manual_seed(15000 + step)
            q_nope = torch.randn(batch_size, self.NUM_HEADS, self.KV_LORA_RANK, device=device, dtype=dtype)
            q_rope = torch.randn(batch_size, self.NUM_HEADS, self.QK_ROPE_HEAD_DIM, device=device, dtype=dtype)
            k_nope = torch.randn(batch_size, self.KV_LORA_RANK, device=device, dtype=dtype)
            k_rope = torch.randn(batch_size, self.QK_ROPE_HEAD_DIM, device=device, dtype=dtype)

            pos_ids = torch.full((batch_size,), current_pos, device=device, dtype=torch.int32)
            seq_lens = torch.full((batch_size,), seq_len, dtype=torch.int32, device=device)

            page_offset = current_pos // self.PAGE_SIZE
            pos_in_page = current_pos % self.PAGE_SIZE
            out_cache_loc = torch.tensor([
                (seq_idx * 4 + page_offset) * self.PAGE_SIZE + pos_in_page
                for seq_idx in range(batch_size)
            ], device=device, dtype=torch.int32)

            # Fused path (NO sync between RoPE and attention)
            q_out_fused = self._run_fused_path(
                q_nope, q_rope, k_nope, k_rope,
                cos_sin_cache, pos_ids, out_cache_loc, kv_cache_fused,
            )
            query_fused = q_out_fused.unsqueeze(1)
            attn_out_fused = self._run_attention(
                query=query_fused, kv_cache=kv_cache_fused,
                workspace_buffer=workspace_buffer, block_tables=block_tables,
                seq_lens=seq_lens, max_seq_len=seq_len,
                bmm1_scale=1.0 / (self.HEAD_DIM ** 0.5),
            )
            # NO torch.cuda.synchronize() here - intentional!

            # Separated path (NO sync between RoPE and attention)
            q_out_sep = self._run_separated_path(
                q_nope, q_rope, k_nope, k_rope,
                cos_sin_cache, pos_ids, out_cache_loc, kv_cache_sep,
            )
            query_sep = q_out_sep.unsqueeze(1)
            attn_out_sep = self._run_attention(
                query=query_sep, kv_cache=kv_cache_sep,
                workspace_buffer=workspace_buffer, block_tables=block_tables,
                seq_lens=seq_lens, max_seq_len=seq_len,
                bmm1_scale=1.0 / (self.HEAD_DIM ** 0.5),
            )
            # NO torch.cuda.synchronize() here - intentional!

            max_diff = torch.max(torch.abs(attn_out_fused.float() - attn_out_sep.float())).item()
            max_diffs.append((step, max_diff))

        # Final sync after all iterations
        torch.cuda.synchronize()

        failures = [(s, d) for s, d in max_diffs if d > 0]
        if failures:
            summary = ", ".join([f"step {s}: {d:.6e}" for s, d in failures[:5]])
            pytest.fail(
                f"Fused vs separated mismatch without sync! "
                f"{len(failures)}/{num_iterations} steps differ.\n"
                f"First differences: {summary}"
            )

    def test_cross_sequence_isolation(self, device, dtype, cos_sin_cache, workspace_buffer):
        """Verify that attention outputs are isolated per sequence.

        Changing one sequence's KV cache should not affect another sequence's
        attention output. This catches cross-sequence data leakage bugs in the
        attention kernel or KV cache addressing.
        """
        batch_size = 4
        num_pages = 16

        seq_lens = torch.full((batch_size,), 64, dtype=torch.int32, device=device)
        max_seq_len = 64

        torch.manual_seed(9999)
        q_nope = torch.randn(batch_size, self.NUM_HEADS, self.KV_LORA_RANK, device=device, dtype=dtype)
        q_rope = torch.randn(batch_size, self.NUM_HEADS, self.QK_ROPE_HEAD_DIM, device=device, dtype=dtype)
        k_nope = torch.randn(batch_size, self.KV_LORA_RANK, device=device, dtype=dtype)
        k_rope = torch.randn(batch_size, self.QK_ROPE_HEAD_DIM, device=device, dtype=dtype)

        pos_ids = torch.arange(batch_size, device=device, dtype=torch.int32)
        out_cache_loc = torch.tensor([63, 127, 191, 255], device=device, dtype=torch.int32)
        block_tables = torch.tensor([
            [0, -1], [1, -1], [2, -1], [3, -1],
        ], dtype=torch.int32, device=device)

        # Build baseline KV cache
        kv_cache_baseline = self._create_combined_kv_cache(num_pages, device)
        torch.manual_seed(10001)
        for page_idx in range(4):
            for pos in range(63):
                k_nope_prev = torch.randn(self.KV_LORA_RANK, device=device, dtype=dtype)
                k_rope_prev = torch.randn(self.QK_ROPE_HEAD_DIM, device=device, dtype=dtype)
                kv_cache_baseline[page_idx, 0, pos, :self.KV_LORA_RANK] = k_nope_prev.to(torch.float8_e4m3fn)
                kv_cache_baseline[page_idx, 0, pos, self.KV_LORA_RANK:] = k_rope_prev.to(torch.float8_e4m3fn)

        q_out_baseline = self._run_fused_path(
            q_nope, q_rope, k_nope, k_rope,
            cos_sin_cache, pos_ids, out_cache_loc, kv_cache_baseline,
        )
        query_baseline = q_out_baseline.unsqueeze(1)
        attn_baseline = self._run_attention(
            query=query_baseline, kv_cache=kv_cache_baseline,
            workspace_buffer=workspace_buffer, block_tables=block_tables,
            seq_lens=seq_lens, max_seq_len=max_seq_len,
            bmm1_scale=1.0 / (self.HEAD_DIM ** 0.5),
        )
        torch.cuda.synchronize()

        # Modify sequence 2's KV cache (page 2) with random data
        kv_cache_modified = kv_cache_baseline.clone()
        kv_cache_modified[2, 0, :, :] = torch.randn(
            self.PAGE_SIZE, self.KV_LORA_RANK + self.QK_ROPE_HEAD_DIM,
            device=device, dtype=dtype,
        ).to(torch.float8_e4m3fn)

        attn_modified = self._run_attention(
            query=query_baseline, kv_cache=kv_cache_modified,
            workspace_buffer=workspace_buffer, block_tables=block_tables,
            seq_lens=seq_lens, max_seq_len=max_seq_len,
            bmm1_scale=1.0 / (self.HEAD_DIM ** 0.5),
        )
        torch.cuda.synchronize()

        # Sequences 0, 1, 3 must be unaffected
        for seq_idx in [0, 1, 3]:
            diff = torch.max(torch.abs(
                attn_baseline[seq_idx].float() - attn_modified[seq_idx].float()
            )).item()
            assert diff == 0.0, (
                f"Cross-sequence leakage! Sequence {seq_idx}'s attention changed "
                f"when sequence 2's KV cache was modified. Max diff: {diff}"
            )

        # Sequence 2 should be different (sanity check)
        diff_seq2 = torch.max(torch.abs(
            attn_baseline[2].float() - attn_modified[2].float()
        )).item()
        assert diff_seq2 > 0, "Sequence 2's output should change when its KV cache changes!"

    def test_multi_step_kv_immutability(self, device, dtype, cos_sin_cache, workspace_buffer):
        """Verify that earlier KV cache entries are not modified by subsequent decode steps.

        Snapshot the KV cache after each step and verify that positions written
        in previous steps remain unchanged. This catches bugs where the fused
        kernel accidentally overwrites adjacent KV cache entries.
        """
        batch_size = 2
        num_pages = 8
        num_decode_steps = 8

        kv_cache = self._create_combined_kv_cache(num_pages, device)
        block_tables = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32, device=device)

        # Pre-populate first 48 positions
        torch.manual_seed(12000)
        for seq_idx in range(batch_size):
            base_page = seq_idx * 2
            for pos in range(48):
                page_offset = pos // self.PAGE_SIZE
                pos_in_page = pos % self.PAGE_SIZE
                page_idx = base_page + page_offset
                k_nope_prev = torch.randn(self.KV_LORA_RANK, device=device, dtype=dtype)
                k_rope_prev = torch.randn(self.QK_ROPE_HEAD_DIM, device=device, dtype=dtype)
                kv_cache[page_idx, 0, pos_in_page, :self.KV_LORA_RANK] = k_nope_prev.to(torch.float8_e4m3fn)
                kv_cache[page_idx, 0, pos_in_page, self.KV_LORA_RANK:] = k_rope_prev.to(torch.float8_e4m3fn)

        kv_snapshots = []

        for step in range(num_decode_steps):
            # Snapshot KV cache BEFORE this step's write
            kv_snapshots.append(kv_cache.clone())

            current_pos = 48 + step

            torch.manual_seed(13000 + step)
            q_nope = torch.randn(batch_size, self.NUM_HEADS, self.KV_LORA_RANK, device=device, dtype=dtype)
            q_rope = torch.randn(batch_size, self.NUM_HEADS, self.QK_ROPE_HEAD_DIM, device=device, dtype=dtype)
            k_nope = torch.randn(batch_size, self.KV_LORA_RANK, device=device, dtype=dtype)
            k_rope = torch.randn(batch_size, self.QK_ROPE_HEAD_DIM, device=device, dtype=dtype)

            pos_ids = torch.full((batch_size,), current_pos, device=device, dtype=torch.int32)

            page_offset = current_pos // self.PAGE_SIZE
            pos_in_page = current_pos % self.PAGE_SIZE
            out_cache_loc = torch.tensor([
                (0 + page_offset) * self.PAGE_SIZE + pos_in_page,
                (2 + page_offset) * self.PAGE_SIZE + pos_in_page,
            ], device=device, dtype=torch.int32)

            _ = self._run_fused_path(
                q_nope, q_rope, k_nope, k_rope,
                cos_sin_cache, pos_ids, out_cache_loc, kv_cache,
            )
            torch.cuda.synchronize()

            # After this step's write, verify all PREVIOUS positions are unchanged.
            # Snapshot[prev_step + 1] was taken after prev_step's write but before
            # the current step, so it contains the expected values.
            for prev_step in range(step):
                prev_pos = 48 + prev_step
                prev_page_offset = prev_pos // self.PAGE_SIZE
                prev_pos_in_page = prev_pos % self.PAGE_SIZE

                snapshot_idx = prev_step + 1  # snapshot taken AFTER prev_step wrote
                for seq_idx in range(batch_size):
                    base_page = seq_idx * 2
                    prev_page = base_page + prev_page_offset

                    expected = kv_snapshots[snapshot_idx][prev_page, 0, prev_pos_in_page]
                    actual = kv_cache[prev_page, 0, prev_pos_in_page]

                    diff = torch.max(torch.abs(expected.float() - actual.float())).item()
                    assert diff == 0.0, (
                        f"KV immutability violated! Position {prev_pos} (written at step "
                        f"{prev_step}) changed at step {step}. Seq {seq_idx}, max diff: {diff}"
                    )

            # Also check pre-populated positions (0-47) haven't been touched
            for check_pos in [0, 1, 23, 47]:
                check_page_offset = check_pos // self.PAGE_SIZE
                check_pos_in_page = check_pos % self.PAGE_SIZE
                for seq_idx in range(batch_size):
                    base_page = seq_idx * 2
                    check_page = base_page + check_page_offset

                    expected = kv_snapshots[0][check_page, 0, check_pos_in_page]
                    actual = kv_cache[check_page, 0, check_pos_in_page]
                    diff = torch.max(torch.abs(expected.float() - actual.float())).item()
                    assert diff == 0.0, (
                        f"Pre-populated KV position {check_pos} modified at step {step}!"
                    )


###############################################################################
# CUDA graph tests with full DeepSeek V3/R1 dimensions (128 heads)
###############################################################################


def _create_combined_kv_cache_standalone(
    num_pages: int, page_size: int, kv_lora_rank: int,
    qk_rope_head_dim: int, device: torch.device,
) -> torch.Tensor:
    """Create combined KV cache for trtllm attention (standalone helper)."""
    kv_cache_dim = kv_lora_rank + qk_rope_head_dim
    return torch.zeros(
        num_pages, 1, page_size, kv_cache_dim,
        device=device, dtype=torch.float8_e4m3fn,
    )


def _create_cos_sin_cache_large(qk_rope_head_dim: int, device: torch.device) -> torch.Tensor:
    """Create cos/sin cache for RoPE with 128K context (must be float32)."""
    max_seq_len = 131072
    freqs = 1.0 / (10000 ** (torch.arange(0, qk_rope_head_dim, 2, device=device).float() / qk_rope_head_dim))
    t = torch.arange(max_seq_len, device=device).float()
    freqs = torch.outer(t, freqs)
    cos = torch.cos(freqs)
    sin = torch.sin(freqs)
    return torch.cat([cos, sin], dim=-1).float()


@pytest.mark.skipif(not _is_cuda, reason="CUDA not available")
@pytest.mark.skipif(not _has_flashinfer, reason="FlashInfer not available")
class TestMLAAttentionWithCUDAGraphs:
    """
    End-to-end tests using CUDA graphs to match production behavior.

    Uses FULL DeepSeek V3/R1 dimensions:
    - 128 attention heads (no tensor parallelism)
    - kv_lora_rank = 512
    - qk_rope_head_dim = 64
    - page_size = 64
    """

    KV_LORA_RANK = 512
    QK_ROPE_HEAD_DIM = 64
    QK_NOPE_HEAD_DIM = 128
    NUM_HEADS = 128
    PAGE_SIZE = 64
    HEAD_DIM = KV_LORA_RANK + QK_ROPE_HEAD_DIM  # 576

    @pytest.fixture
    def device(self):
        return torch.device("cuda")

    @pytest.fixture
    def dtype(self):
        return torch.bfloat16

    @pytest.fixture
    def cos_sin_cache(self, device):
        return _create_cos_sin_cache_large(self.QK_ROPE_HEAD_DIM, device)

    @pytest.fixture
    def workspace_buffer(self, device):
        return torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=device)

    def _create_combined_kv_cache(self, num_pages: int, device: torch.device) -> torch.Tensor:
        return _create_combined_kv_cache_standalone(
            num_pages, self.PAGE_SIZE, self.KV_LORA_RANK, self.QK_ROPE_HEAD_DIM, device,
        )

    def _warmup_and_capture_graph(
        self, q_nope, q_rope, k_nope, k_rope, q_out, kv_cache, attn_out,
        cos_sin_cache, pos_ids, kv_indices, positions, kv_indptr, batch_indices,
        workspace_buffer, block_tables, seq_lens, max_seq_len,
    ):
        """Run warmup iterations and capture a CUDA graph."""
        attn_dtype = torch.float8_e4m3fn

        # Warmup
        for _ in range(3):
            kv_cache_squeezed = kv_cache.squeeze(1)
            ckv_cache = kv_cache_squeezed[:, :, :self.KV_LORA_RANK]
            kpe_cache = kv_cache_squeezed[:, :, self.KV_LORA_RANK:]
            flashinfer.rope.rope_quantize_fp8_append_paged_kv_cache(
                q_rope=q_rope, k_rope=k_rope, q_nope=q_nope, k_nope=k_nope,
                v=None, cos_sin_cache=cos_sin_cache, pos_ids=pos_ids,
                paged_kv_cache=(ckv_cache, kpe_cache),
                kv_indices=kv_indices, kv_indptr=kv_indptr,
                batch_indices=batch_indices, positions=positions,
                is_neox=False, quantize_dtype=attn_dtype,
                quant_scale_q=1.0, quant_scale_kv=1.0,
                page_size=self.PAGE_SIZE, kv_layout="NHD",
                q_rope_out=q_out[..., self.KV_LORA_RANK:],
                q_nope_out=q_out[..., :self.KV_LORA_RANK],
            )
            kv_cache_squeezed[:, :, :self.KV_LORA_RANK] = ckv_cache
            kv_cache_squeezed[:, :, self.KV_LORA_RANK:] = kpe_cache
            query = q_out.unsqueeze(1)
            flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla(
                query=query, kv_cache=kv_cache, workspace_buffer=workspace_buffer,
                qk_nope_head_dim=self.QK_NOPE_HEAD_DIM, kv_lora_rank=self.KV_LORA_RANK,
                qk_rope_head_dim=self.QK_ROPE_HEAD_DIM, block_tables=block_tables,
                seq_lens=seq_lens, max_seq_len=max_seq_len,
                bmm1_scale=1.0 / (self.HEAD_DIM ** 0.5),
            )

        torch.cuda.synchronize()

        # Capture graph
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())

        with torch.cuda.stream(stream):
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                kv_cache_squeezed = kv_cache.squeeze(1)
                ckv_cache_g = kv_cache_squeezed[:, :, :self.KV_LORA_RANK]
                kpe_cache_g = kv_cache_squeezed[:, :, self.KV_LORA_RANK:]
                flashinfer.rope.rope_quantize_fp8_append_paged_kv_cache(
                    q_rope=q_rope, k_rope=k_rope, q_nope=q_nope, k_nope=k_nope,
                    v=None, cos_sin_cache=cos_sin_cache, pos_ids=pos_ids,
                    paged_kv_cache=(ckv_cache_g, kpe_cache_g),
                    kv_indices=kv_indices, kv_indptr=kv_indptr,
                    batch_indices=batch_indices, positions=positions,
                    is_neox=False, quantize_dtype=attn_dtype,
                    quant_scale_q=1.0, quant_scale_kv=1.0,
                    page_size=self.PAGE_SIZE, kv_layout="NHD",
                    q_rope_out=q_out[..., self.KV_LORA_RANK:],
                    q_nope_out=q_out[..., :self.KV_LORA_RANK],
                )
                kv_cache_squeezed[:, :, :self.KV_LORA_RANK] = ckv_cache_g
                kv_cache_squeezed[:, :, self.KV_LORA_RANK:] = kpe_cache_g
                query_g = q_out.unsqueeze(1)
                attn_result = flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla(
                    query=query_g, kv_cache=kv_cache, workspace_buffer=workspace_buffer,
                    qk_nope_head_dim=self.QK_NOPE_HEAD_DIM, kv_lora_rank=self.KV_LORA_RANK,
                    qk_rope_head_dim=self.QK_ROPE_HEAD_DIM, block_tables=block_tables,
                    seq_lens=seq_lens, max_seq_len=max_seq_len,
                    bmm1_scale=1.0 / (self.HEAD_DIM ** 0.5),
                )
                attn_out.copy_(attn_result)

        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        return graph

    @pytest.mark.parametrize("batch_size", [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048])
    def test_cuda_graph_various_batch_sizes(self, device, dtype, cos_sin_cache, workspace_buffer, batch_size):
        """Test CUDA graph with various batch sizes from 1 to 2048."""
        num_replays = 10
        alignment = 128 // self.PAGE_SIZE
        attn_dtype = torch.float8_e4m3fn

        seq_len = 64
        max_seq_len = seq_len
        num_pages = ((batch_size + alignment - 1) // alignment) * alignment

        seq_lens = torch.full((batch_size,), seq_len, dtype=torch.int32, device=device)
        max_blocks_per_seq = alignment
        block_tables = torch.full((batch_size, max_blocks_per_seq), -1, dtype=torch.int32, device=device)
        for i in range(batch_size):
            if i < num_pages:
                block_tables[i, 0] = i

        out_cache_loc = torch.arange(batch_size, device=device, dtype=torch.int32) * self.PAGE_SIZE + 63
        out_cache_loc = torch.clamp(out_cache_loc, max=(num_pages - 1) * self.PAGE_SIZE + 63)
        pos_ids = (seq_lens - 1).to(torch.int32)

        kv_indices = (out_cache_loc // self.PAGE_SIZE).to(torch.int32)
        positions = (out_cache_loc % self.PAGE_SIZE).to(torch.int32)
        kv_indptr = torch.arange(batch_size + 1, dtype=torch.int32, device=device)
        batch_indices = torch.arange(batch_size, dtype=torch.int32, device=device)

        q_nope = torch.randn(batch_size, self.NUM_HEADS, self.KV_LORA_RANK, device=device, dtype=dtype)
        q_rope = torch.randn(batch_size, self.NUM_HEADS, self.QK_ROPE_HEAD_DIM, device=device, dtype=dtype)
        k_nope = torch.randn(batch_size, self.KV_LORA_RANK, device=device, dtype=dtype)
        k_rope = torch.randn(batch_size, self.QK_ROPE_HEAD_DIM, device=device, dtype=dtype)

        q_out = torch.empty(batch_size, self.NUM_HEADS, self.HEAD_DIM, device=device, dtype=attn_dtype)
        kv_cache = self._create_combined_kv_cache(num_pages, device)
        attn_out = torch.empty(batch_size, 1, self.NUM_HEADS, self.KV_LORA_RANK, device=device, dtype=dtype)
        output_buffers = [torch.empty_like(attn_out) for _ in range(num_replays)]

        torch.manual_seed(7000 + batch_size)
        for page_idx in range(min(batch_size, num_pages)):
            kv_cache[page_idx, 0, :, :] = torch.randn(
                self.PAGE_SIZE, self.KV_LORA_RANK + self.QK_ROPE_HEAD_DIM,
                device=device, dtype=dtype,
            ).to(attn_dtype)
        kv_cache_initial = kv_cache.clone()

        graph = self._warmup_and_capture_graph(
            q_nope, q_rope, k_nope, k_rope, q_out, kv_cache, attn_out,
            cos_sin_cache, pos_ids, kv_indices, positions, kv_indptr, batch_indices,
            workspace_buffer, block_tables, seq_lens, max_seq_len,
        )

        for i in range(num_replays):
            kv_cache.copy_(kv_cache_initial)
            graph.replay()
            output_buffers[i].copy_(attn_out)

        torch.cuda.synchronize()

        reference = output_buffers[0]
        for i, output in enumerate(output_buffers[1:], 1):
            max_diff = torch.max(torch.abs(reference.float() - output.float())).item()
            assert max_diff == 0.0, (
                f"batch_size={batch_size}: replay {i} differs from replay 0! Max diff: {max_diff}"
            )

    def test_cuda_graph_rapid_fire_replays(self, device, dtype, cos_sin_cache, workspace_buffer):
        """Stress test: 100 rapid-fire graph replays with batch_size=256, 128 heads."""
        batch_size = 256
        num_rapid_replays = 100
        alignment = 128 // self.PAGE_SIZE
        attn_dtype = torch.float8_e4m3fn

        seq_len = 64
        max_seq_len = seq_len
        num_pages = ((batch_size + alignment - 1) // alignment) * alignment

        seq_lens = torch.full((batch_size,), seq_len, dtype=torch.int32, device=device)
        max_blocks_per_seq = alignment
        block_tables = torch.full((batch_size, max_blocks_per_seq), -1, dtype=torch.int32, device=device)
        for i in range(batch_size):
            if i < num_pages:
                block_tables[i, 0] = i

        out_cache_loc = torch.arange(batch_size, device=device, dtype=torch.int32) * self.PAGE_SIZE + 63
        out_cache_loc = torch.clamp(out_cache_loc, max=(num_pages - 1) * self.PAGE_SIZE + 63)
        pos_ids = (seq_lens - 1).to(torch.int32)

        kv_indices = (out_cache_loc // self.PAGE_SIZE).to(torch.int32)
        positions = (out_cache_loc % self.PAGE_SIZE).to(torch.int32)
        kv_indptr = torch.arange(batch_size + 1, dtype=torch.int32, device=device)
        batch_indices = torch.arange(batch_size, dtype=torch.int32, device=device)

        q_nope = torch.randn(batch_size, self.NUM_HEADS, self.KV_LORA_RANK, device=device, dtype=dtype)
        q_rope = torch.randn(batch_size, self.NUM_HEADS, self.QK_ROPE_HEAD_DIM, device=device, dtype=dtype)
        k_nope = torch.randn(batch_size, self.KV_LORA_RANK, device=device, dtype=dtype)
        k_rope = torch.randn(batch_size, self.QK_ROPE_HEAD_DIM, device=device, dtype=dtype)

        q_out = torch.empty(batch_size, self.NUM_HEADS, self.HEAD_DIM, device=device, dtype=attn_dtype)
        kv_cache = self._create_combined_kv_cache(num_pages, device)
        attn_out = torch.empty(batch_size, 1, self.NUM_HEADS, self.KV_LORA_RANK, device=device, dtype=dtype)
        output_buffers = [torch.empty_like(attn_out) for _ in range(num_rapid_replays)]

        torch.manual_seed(9000)
        for page_idx in range(min(batch_size, num_pages)):
            kv_cache[page_idx, 0, :, :] = torch.randn(
                self.PAGE_SIZE, self.KV_LORA_RANK + self.QK_ROPE_HEAD_DIM,
                device=device, dtype=dtype,
            ).to(attn_dtype)
        kv_cache_initial = kv_cache.clone()

        graph = self._warmup_and_capture_graph(
            q_nope, q_rope, k_nope, k_rope, q_out, kv_cache, attn_out,
            cos_sin_cache, pos_ids, kv_indices, positions, kv_indptr, batch_indices,
            workspace_buffer, block_tables, seq_lens, max_seq_len,
        )

        for i in range(num_rapid_replays):
            kv_cache.copy_(kv_cache_initial)
            graph.replay()
            output_buffers[i].copy_(attn_out)

        torch.cuda.synchronize()

        reference = output_buffers[0]
        max_diffs = []
        for i, output in enumerate(output_buffers[1:], 1):
            diff = torch.max(torch.abs(reference.float() - output.float())).item()
            if diff > 0:
                max_diffs.append((i, diff))

        if max_diffs:
            diff_summary = ", ".join([f"replay {i}: {d:.6e}" for i, d in max_diffs[:5]])
            pytest.fail(
                f"RACE CONDITION DETECTED! {len(max_diffs)}/{num_rapid_replays-1} replays differ.\n"
                f"First differences: {diff_summary}\n"
                "This strongly indicates a race condition between the RoPE kernel and attention kernel."
            )

    def test_cuda_graph_padded_batch_sizes(self, device, dtype, cos_sin_cache, workspace_buffer):
        """Test CUDA graph captured at max_batch_size, replayed with smaller actual batches.

        This tests the critical 'metadata zeroing' production pattern where:
        - CUDA graph is captured at cuda_graph_max_bs
        - During replay with actual_bs < max_bs, unused slots are zeroed
        - Only first actual_bs outputs should be valid and match eager reference

        The key bug scenario: stale metadata from a previous larger batch causes
        the kernel to write to wrong KV cache pages or compute wrong RoPE positions.
        """
        max_batch_size = 128
        attn_dtype = torch.float8_e4m3fn
        alignment = 128 // self.PAGE_SIZE

        seq_len = 64
        max_seq_len = seq_len
        # Extra page for unused token writes (trash page)
        num_pages = ((max_batch_size + alignment - 1) // alignment) * alignment + 1
        trash_page = num_pages - 1

        # Allocate all buffers at max_batch_size
        q_nope = torch.zeros(max_batch_size, self.NUM_HEADS, self.KV_LORA_RANK, device=device, dtype=dtype)
        q_rope = torch.zeros(max_batch_size, self.NUM_HEADS, self.QK_ROPE_HEAD_DIM, device=device, dtype=dtype)
        k_nope = torch.zeros(max_batch_size, self.KV_LORA_RANK, device=device, dtype=dtype)
        k_rope = torch.zeros(max_batch_size, self.QK_ROPE_HEAD_DIM, device=device, dtype=dtype)

        pos_ids = torch.zeros(max_batch_size, device=device, dtype=torch.int32)
        seq_lens = torch.full((max_batch_size,), seq_len, dtype=torch.int32, device=device)

        max_blocks_per_seq = alignment
        block_tables = torch.full((max_batch_size, max_blocks_per_seq), -1, dtype=torch.int32, device=device)
        for i in range(max_batch_size):
            if i < num_pages - 1:
                block_tables[i, 0] = i

        out_cache_loc = torch.arange(max_batch_size, device=device, dtype=torch.int32) * self.PAGE_SIZE + (seq_len - 1)
        out_cache_loc = torch.clamp(out_cache_loc, max=(num_pages - 2) * self.PAGE_SIZE + (seq_len - 1))

        kv_indices = (out_cache_loc // self.PAGE_SIZE).to(torch.int32)
        positions_meta = (out_cache_loc % self.PAGE_SIZE).to(torch.int32)
        kv_indptr = torch.arange(max_batch_size + 1, dtype=torch.int32, device=device)
        batch_indices = torch.arange(max_batch_size, dtype=torch.int32, device=device)

        # Fill with valid data for warmup + capture
        torch.manual_seed(60000)
        q_nope.normal_()
        q_rope.normal_()
        k_nope.normal_()
        k_rope.normal_()
        pos_ids[:] = (seq_lens - 1).to(torch.int32)

        q_out = torch.empty(max_batch_size, self.NUM_HEADS, self.HEAD_DIM, device=device, dtype=attn_dtype)
        kv_cache = self._create_combined_kv_cache(num_pages, device)
        attn_out = torch.empty(max_batch_size, 1, self.NUM_HEADS, self.KV_LORA_RANK, device=device, dtype=dtype)

        # Pre-populate KV cache (excluding trash page)
        torch.manual_seed(60001)
        for page_idx in range(num_pages - 1):
            kv_cache[page_idx, 0, :, :] = torch.randn(
                self.PAGE_SIZE, self.KV_LORA_RANK + self.QK_ROPE_HEAD_DIM,
                device=device, dtype=dtype,
            ).to(attn_dtype)

        # Capture graph at max_batch_size
        graph = self._warmup_and_capture_graph(
            q_nope, q_rope, k_nope, k_rope, q_out, kv_cache, attn_out,
            cos_sin_cache, pos_ids, kv_indices, positions_meta, kv_indptr, batch_indices,
            workspace_buffer, block_tables, seq_lens, max_seq_len,
        )

        # Test with smaller actual batch sizes
        for actual_bs in [1, 4, 16, 32, 64]:
            # --- Prepare shared KV cache state ---
            kv_cache_clean = self._create_combined_kv_cache(num_pages, device)
            torch.manual_seed(63000 + actual_bs)
            for page_idx in range(actual_bs):
                kv_cache_clean[page_idx, 0, :, :] = torch.randn(
                    self.PAGE_SIZE, self.KV_LORA_RANK + self.QK_ROPE_HEAD_DIM,
                    device=device, dtype=dtype,
                ).to(attn_dtype)

            # --- Prepare input data ---
            torch.manual_seed(62000 + actual_bs)
            q_nope_data = torch.randn(actual_bs, self.NUM_HEADS, self.KV_LORA_RANK, device=device, dtype=dtype)
            q_rope_data = torch.randn(actual_bs, self.NUM_HEADS, self.QK_ROPE_HEAD_DIM, device=device, dtype=dtype)
            k_nope_data = torch.randn(actual_bs, self.KV_LORA_RANK, device=device, dtype=dtype)
            k_rope_data = torch.randn(actual_bs, self.QK_ROPE_HEAD_DIM, device=device, dtype=dtype)

            active_cache_locs = torch.arange(actual_bs, device=device, dtype=torch.int32) * self.PAGE_SIZE + (seq_len - 1)
            active_cache_locs = torch.clamp(active_cache_locs, max=(num_pages - 2) * self.PAGE_SIZE + (seq_len - 1))

            # --- Graph replay: fill active, zero unused ---
            q_nope[:actual_bs].copy_(q_nope_data)
            q_rope[:actual_bs].copy_(q_rope_data)
            k_nope[:actual_bs].copy_(k_nope_data)
            k_rope[:actual_bs].copy_(k_rope_data)
            q_nope[actual_bs:].zero_()
            q_rope[actual_bs:].zero_()
            k_nope[actual_bs:].zero_()
            k_rope[actual_bs:].zero_()

            pos_ids[:actual_bs] = seq_len - 1
            pos_ids[actual_bs:] = 0

            seq_lens[:actual_bs] = seq_len
            seq_lens[actual_bs:] = 1  # min 1 to avoid kernel issues

            block_tables[:] = -1
            for i in range(actual_bs):
                block_tables[i, 0] = i
            for i in range(actual_bs, max_batch_size):
                block_tables[i, 0] = trash_page

            out_cache_loc[:actual_bs] = active_cache_locs
            out_cache_loc[actual_bs:] = trash_page * self.PAGE_SIZE  # unused → trash page

            kv_indices[:actual_bs] = (active_cache_locs // self.PAGE_SIZE).to(torch.int32)
            kv_indices[actual_bs:] = trash_page
            positions_meta[:actual_bs] = (active_cache_locs % self.PAGE_SIZE).to(torch.int32)
            positions_meta[actual_bs:] = 0

            kv_indptr[:] = torch.arange(max_batch_size + 1, dtype=torch.int32, device=device)
            batch_indices[:] = torch.arange(max_batch_size, dtype=torch.int32, device=device)

            kv_cache.copy_(kv_cache_clean)
            graph.replay()
            torch.cuda.synchronize()
            graph_active_out = attn_out[:actual_bs].clone()
            graph_q_out = q_out[:actual_bs].clone()
            # Snapshot KV cache pages touched by active tokens
            graph_kv_active = kv_cache[:actual_bs].clone()

            # --- Eager reference at actual_bs ---
            kv_cache_ref = kv_cache_clean.clone()
            kv_cache_ref_squeezed = kv_cache_ref.squeeze(1)
            ckv_ref = kv_cache_ref_squeezed[:, :, :self.KV_LORA_RANK]
            kpe_ref = kv_cache_ref_squeezed[:, :, self.KV_LORA_RANK:]

            q_out_ref = torch.empty(actual_bs, self.NUM_HEADS, self.HEAD_DIM, device=device, dtype=attn_dtype)

            ref_kv_indices = (active_cache_locs // self.PAGE_SIZE).to(torch.int32)
            ref_positions = (active_cache_locs % self.PAGE_SIZE).to(torch.int32)
            ref_kv_indptr = torch.arange(actual_bs + 1, dtype=torch.int32, device=device)
            ref_batch_indices = torch.arange(actual_bs, dtype=torch.int32, device=device)
            ref_pos_ids = torch.full((actual_bs,), seq_len - 1, device=device, dtype=torch.int32)

            flashinfer.rope.rope_quantize_fp8_append_paged_kv_cache(
                q_rope=q_rope_data, k_rope=k_rope_data, q_nope=q_nope_data, k_nope=k_nope_data,
                v=None, cos_sin_cache=cos_sin_cache, pos_ids=ref_pos_ids,
                paged_kv_cache=(ckv_ref, kpe_ref),
                kv_indices=ref_kv_indices, kv_indptr=ref_kv_indptr,
                batch_indices=ref_batch_indices, positions=ref_positions,
                is_neox=False, quantize_dtype=attn_dtype,
                quant_scale_q=1.0, quant_scale_kv=1.0,
                page_size=self.PAGE_SIZE, kv_layout="NHD",
                q_rope_out=q_out_ref[..., self.KV_LORA_RANK:],
                q_nope_out=q_out_ref[..., :self.KV_LORA_RANK],
            )
            kv_cache_ref_squeezed[:, :, :self.KV_LORA_RANK] = ckv_ref
            kv_cache_ref_squeezed[:, :, self.KV_LORA_RANK:] = kpe_ref
            torch.cuda.synchronize()

            # --- Exact checks: q_out and KV cache must be bit-identical ---
            # These use the same kernel with the same inputs, so padding must not
            # leak into the active slot computation.
            q_diff = torch.max(torch.abs(graph_q_out.float() - q_out_ref.float())).item()
            assert q_diff == 0.0, (
                f"q_out mismatch at actual_bs={actual_bs}! max_diff={q_diff:.6e}\n"
                "Padding metadata leaked into RoPE/quantize for active tokens."
            )

            kv_diff = torch.max(torch.abs(graph_kv_active.float() - kv_cache_ref[:actual_bs].float())).item()
            assert kv_diff == 0.0, (
                f"KV cache mismatch at actual_bs={actual_bs}! max_diff={kv_diff:.6e}\n"
                "Padding metadata corrupted KV cache writes for active tokens."
            )

            # --- Attention output: allow small tolerance ---
            # The trtllm attention kernel may use different internal tiling /
            # reduction strategies when total batch_size differs (128 in graph
            # vs actual_bs in eager), leading to minor FP accumulation diffs.
            # Observed diffs up to ~4e-3 for bf16 attention across batch sizes.
            # A tolerance of 1e-2 catches gross errors (e.g. reading wrong
            # pages would give diffs ~1.0) while accommodating kernel numerics.
            ref_block_tables = torch.full((actual_bs, max_blocks_per_seq), -1, dtype=torch.int32, device=device)
            for i in range(actual_bs):
                ref_block_tables[i, 0] = i
            ref_seq_lens = torch.full((actual_bs,), seq_len, dtype=torch.int32, device=device)

            query_ref = q_out_ref.unsqueeze(1)
            attn_out_ref = flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla(
                query=query_ref, kv_cache=kv_cache_ref, workspace_buffer=workspace_buffer,
                qk_nope_head_dim=self.QK_NOPE_HEAD_DIM, kv_lora_rank=self.KV_LORA_RANK,
                qk_rope_head_dim=self.QK_ROPE_HEAD_DIM, block_tables=ref_block_tables,
                seq_lens=ref_seq_lens, max_seq_len=max_seq_len,
                bmm1_scale=1.0 / (self.HEAD_DIM ** 0.5),
            )
            torch.cuda.synchronize()

            attn_diff = torch.max(torch.abs(graph_active_out.float() - attn_out_ref.float())).item()
            assert attn_diff < 1e-2, (
                f"CUDA graph padded batch attention mismatch! actual_bs={actual_bs}, "
                f"max_bs={max_batch_size}, max_diff={attn_diff:.6e}\n"
                "This likely indicates metadata for unused slots is leaking into "
                "active slot attention computation."
            )

    def test_cuda_graph_sacrificial_page0_contract(self, device, dtype, cos_sin_cache):
        """Verify padded replay writes only to sacrificial slot (page 0, pos 0).

        Contract under CUDA graph replay with actual_bs < capture_bs:
          - active tokens write to pages 1+
          - padded tokens are routed to page 0, position 0
          - page 0 positions 1..end remain untouched
        """
        max_batch_size = 64
        actual_bs = 16
        num_pages = 4
        attn_dtype = torch.float8_e4m3fn
        seq_len = 64

        # Capture-sized buffers
        q_nope = torch.zeros(max_batch_size, self.NUM_HEADS, self.KV_LORA_RANK, device=device, dtype=dtype)
        q_rope = torch.zeros(max_batch_size, self.NUM_HEADS, self.QK_ROPE_HEAD_DIM, device=device, dtype=dtype)
        k_nope = torch.zeros(max_batch_size, self.KV_LORA_RANK, device=device, dtype=dtype)
        k_rope = torch.zeros(max_batch_size, self.QK_ROPE_HEAD_DIM, device=device, dtype=dtype)
        pos_ids = torch.zeros(max_batch_size, device=device, dtype=torch.int32)
        kv_indices = torch.zeros(max_batch_size, device=device, dtype=torch.int32)
        positions_meta = torch.zeros(max_batch_size, device=device, dtype=torch.int32)
        kv_indptr = torch.arange(max_batch_size + 1, dtype=torch.int32, device=device)
        batch_indices = torch.arange(max_batch_size, dtype=torch.int32, device=device)
        q_out = torch.empty(max_batch_size, self.NUM_HEADS, self.HEAD_DIM, device=device, dtype=attn_dtype)

        ckv_cache = torch.zeros(num_pages, self.PAGE_SIZE, self.KV_LORA_RANK, device=device, dtype=attn_dtype)
        kpe_cache = torch.zeros(num_pages, self.PAGE_SIZE, self.QK_ROPE_HEAD_DIM, device=device, dtype=attn_dtype)

        # Warmup/capture with full-size valid metadata
        torch.manual_seed(66100)
        q_nope.normal_()
        q_rope.normal_()
        k_nope.normal_()
        k_rope.normal_()
        pos_ids[:] = seq_len - 1
        full_locs = torch.arange(max_batch_size, device=device, dtype=torch.int32) + self.PAGE_SIZE
        kv_indices[:] = (full_locs // self.PAGE_SIZE).to(torch.int32)
        positions_meta[:] = (full_locs % self.PAGE_SIZE).to(torch.int32)

        flashinfer.rope.rope_quantize_fp8_append_paged_kv_cache(
            q_rope=q_rope, k_rope=k_rope, q_nope=q_nope, k_nope=k_nope,
            v=None, cos_sin_cache=cos_sin_cache, pos_ids=pos_ids,
            paged_kv_cache=(ckv_cache, kpe_cache),
            kv_indices=kv_indices, kv_indptr=kv_indptr,
            batch_indices=batch_indices, positions=positions_meta,
            is_neox=False, quantize_dtype=attn_dtype,
            quant_scale_q=1.0, quant_scale_kv=1.0,
            page_size=self.PAGE_SIZE, kv_layout="NHD",
            q_rope_out=q_out[..., self.KV_LORA_RANK:],
            q_nope_out=q_out[..., :self.KV_LORA_RANK],
        )
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            flashinfer.rope.rope_quantize_fp8_append_paged_kv_cache(
                q_rope=q_rope, k_rope=k_rope, q_nope=q_nope, k_nope=k_nope,
                v=None, cos_sin_cache=cos_sin_cache, pos_ids=pos_ids,
                paged_kv_cache=(ckv_cache, kpe_cache),
                kv_indices=kv_indices, kv_indptr=kv_indptr,
                batch_indices=batch_indices, positions=positions_meta,
                is_neox=False, quantize_dtype=attn_dtype,
                quant_scale_q=1.0, quant_scale_kv=1.0,
                page_size=self.PAGE_SIZE, kv_layout="NHD",
                q_rope_out=q_out[..., self.KV_LORA_RANK:],
                q_nope_out=q_out[..., :self.KV_LORA_RANK],
            )

        # Runtime replay data: active tokens go to page 1, padded tokens -> page 0, pos 0.
        torch.manual_seed(66101)
        q_nope_data = torch.randn(actual_bs, self.NUM_HEADS, self.KV_LORA_RANK, device=device, dtype=dtype)
        q_rope_data = torch.randn(actual_bs, self.NUM_HEADS, self.QK_ROPE_HEAD_DIM, device=device, dtype=dtype)
        k_nope_data = torch.randn(actual_bs, self.KV_LORA_RANK, device=device, dtype=dtype)
        k_rope_data = torch.randn(actual_bs, self.QK_ROPE_HEAD_DIM, device=device, dtype=dtype)
        active_locs = torch.arange(actual_bs, device=device, dtype=torch.int32) + self.PAGE_SIZE  # page 1, pos 0..15

        q_nope[:actual_bs].copy_(q_nope_data)
        q_rope[:actual_bs].copy_(q_rope_data)
        k_nope[:actual_bs].copy_(k_nope_data)
        k_rope[:actual_bs].copy_(k_rope_data)
        q_nope[actual_bs:].zero_()
        q_rope[actual_bs:].zero_()
        k_nope[actual_bs:].zero_()
        k_rope[actual_bs:].zero_()

        pos_ids[:actual_bs] = seq_len - 1
        pos_ids[actual_bs:] = 0
        kv_indices[:actual_bs] = (active_locs // self.PAGE_SIZE).to(torch.int32)  # page 1
        kv_indices[actual_bs:] = 0  # sacrificial page
        positions_meta[:actual_bs] = (active_locs % self.PAGE_SIZE).to(torch.int32)
        positions_meta[actual_bs:] = 0  # sacrificial slot
        kv_indptr[:] = torch.arange(max_batch_size + 1, dtype=torch.int32, device=device)
        batch_indices[:] = torch.arange(max_batch_size, dtype=torch.int32, device=device)

        # Seed page 0 with sentinel values so unexpected writes are visible.
        torch.manual_seed(66102)
        ckv_cache.zero_()
        kpe_cache.zero_()
        ckv_cache[0] = torch.randn_like(ckv_cache[0].float()).to(attn_dtype)
        kpe_cache[0] = torch.randn_like(kpe_cache[0].float()).to(attn_dtype)
        ckv_page0_before = ckv_cache[0].clone()
        kpe_page0_before = kpe_cache[0].clone()

        graph.replay()
        torch.cuda.synchronize()
        q_graph_active = q_out[:actual_bs].clone()
        ckv_graph = ckv_cache.clone()
        kpe_graph = kpe_cache.clone()

        # Eager reference at actual_bs with correctly sized metadata
        ckv_ref = torch.zeros_like(ckv_cache)
        kpe_ref = torch.zeros_like(kpe_cache)
        q_out_ref = torch.empty(actual_bs, self.NUM_HEADS, self.HEAD_DIM, device=device, dtype=attn_dtype)
        ref_kv_indices = (active_locs // self.PAGE_SIZE).to(torch.int32)
        ref_positions = (active_locs % self.PAGE_SIZE).to(torch.int32)
        ref_kv_indptr = torch.arange(actual_bs + 1, dtype=torch.int32, device=device)
        ref_batch_indices = torch.arange(actual_bs, dtype=torch.int32, device=device)
        ref_pos_ids = torch.full((actual_bs,), seq_len - 1, device=device, dtype=torch.int32)

        flashinfer.rope.rope_quantize_fp8_append_paged_kv_cache(
            q_rope=q_rope_data, k_rope=k_rope_data, q_nope=q_nope_data, k_nope=k_nope_data,
            v=None, cos_sin_cache=cos_sin_cache, pos_ids=ref_pos_ids,
            paged_kv_cache=(ckv_ref, kpe_ref),
            kv_indices=ref_kv_indices, kv_indptr=ref_kv_indptr,
            batch_indices=ref_batch_indices, positions=ref_positions,
            is_neox=False, quantize_dtype=attn_dtype,
            quant_scale_q=1.0, quant_scale_kv=1.0,
            page_size=self.PAGE_SIZE, kv_layout="NHD",
            q_rope_out=q_out_ref[..., self.KV_LORA_RANK:],
            q_nope_out=q_out_ref[..., :self.KV_LORA_RANK],
        )
        torch.cuda.synchronize()

        # Active results must match eager reference exactly.
        q_diff = torch.max(torch.abs(q_graph_active.float() - q_out_ref.float())).item()
        assert q_diff == 0.0, f"Active q_out mismatch under sacrificial-page replay! max_diff={q_diff}"
        ckv_diff = torch.max(torch.abs(ckv_graph[1:].float() - ckv_ref[1:].float())).item()
        kpe_diff = torch.max(torch.abs(kpe_graph[1:].float() - kpe_ref[1:].float())).item()
        assert ckv_diff == 0.0, f"Active-page ckv mismatch under sacrificial-page replay! max_diff={ckv_diff}"
        assert kpe_diff == 0.0, f"Active-page kpe mismatch under sacrificial-page replay! max_diff={kpe_diff}"

        # Sacrificial page contract: only page0,pos0 may change; page0,pos1..end must stay untouched.
        ckv_page0_tail_diff = torch.max(torch.abs(ckv_graph[0, 1:].float() - ckv_page0_before[1:].float())).item()
        kpe_page0_tail_diff = torch.max(torch.abs(kpe_graph[0, 1:].float() - kpe_page0_before[1:].float())).item()
        assert ckv_page0_tail_diff == 0.0, (
            f"Unexpected writes outside sacrificial slot in ckv page 0 tail! max_diff={ckv_page0_tail_diff}"
        )
        assert kpe_page0_tail_diff == 0.0, (
            f"Unexpected writes outside sacrificial slot in kpe page 0 tail! max_diff={kpe_page0_tail_diff}"
        )

        ckv_page0_slot0_diff = torch.max(torch.abs(ckv_graph[0, 0].float() - ckv_page0_before[0].float())).item()
        kpe_page0_slot0_diff = torch.max(torch.abs(kpe_graph[0, 0].float() - kpe_page0_before[0].float())).item()
        assert ckv_page0_slot0_diff > 0.0 or kpe_page0_slot0_diff > 0.0, (
            "Expected sacrificial slot page0,pos0 to absorb padded writes, but it did not change."
        )

    def test_cuda_graph_fused_vs_eager_separated(self, device, dtype, cos_sin_cache, workspace_buffer):
        """Compare CUDA graph fused path against eager separated reference.

        The existing CUDA graph tests only verify replay consistency (same
        result across replays). This test compares the graph output against
        an independent eager separated-path reference, catching bugs where
        the graph consistently produces the wrong answer.
        """
        batch_size = 64
        attn_dtype = torch.float8_e4m3fn
        alignment = 128 // self.PAGE_SIZE

        seq_len = 64
        max_seq_len = seq_len
        num_pages = ((batch_size + alignment - 1) // alignment) * alignment

        seq_lens = torch.full((batch_size,), seq_len, dtype=torch.int32, device=device)
        max_blocks_per_seq = alignment
        block_tables = torch.full((batch_size, max_blocks_per_seq), -1, dtype=torch.int32, device=device)
        for i in range(batch_size):
            if i < num_pages:
                block_tables[i, 0] = i

        out_cache_loc = torch.arange(batch_size, device=device, dtype=torch.int32) * self.PAGE_SIZE + (seq_len - 1)
        out_cache_loc = torch.clamp(out_cache_loc, max=(num_pages - 1) * self.PAGE_SIZE + (seq_len - 1))
        pos_ids = (seq_lens - 1).to(torch.int32)

        kv_indices = (out_cache_loc // self.PAGE_SIZE).to(torch.int32)
        positions_meta = (out_cache_loc % self.PAGE_SIZE).to(torch.int32)
        kv_indptr = torch.arange(batch_size + 1, dtype=torch.int32, device=device)
        batch_indices = torch.arange(batch_size, dtype=torch.int32, device=device)

        torch.manual_seed(65000)
        q_nope = torch.randn(batch_size, self.NUM_HEADS, self.KV_LORA_RANK, device=device, dtype=dtype)
        q_rope = torch.randn(batch_size, self.NUM_HEADS, self.QK_ROPE_HEAD_DIM, device=device, dtype=dtype)
        k_nope = torch.randn(batch_size, self.KV_LORA_RANK, device=device, dtype=dtype)
        k_rope = torch.randn(batch_size, self.QK_ROPE_HEAD_DIM, device=device, dtype=dtype)

        q_out = torch.empty(batch_size, self.NUM_HEADS, self.HEAD_DIM, device=device, dtype=attn_dtype)
        kv_cache = self._create_combined_kv_cache(num_pages, device)
        attn_out = torch.empty(batch_size, 1, self.NUM_HEADS, self.KV_LORA_RANK, device=device, dtype=dtype)

        torch.manual_seed(65001)
        for page_idx in range(min(batch_size, num_pages)):
            kv_cache[page_idx, 0, :, :] = torch.randn(
                self.PAGE_SIZE, self.KV_LORA_RANK + self.QK_ROPE_HEAD_DIM,
                device=device, dtype=dtype,
            ).to(attn_dtype)
        kv_cache_initial = kv_cache.clone()

        # Capture and replay graph
        graph = self._warmup_and_capture_graph(
            q_nope, q_rope, k_nope, k_rope, q_out, kv_cache, attn_out,
            cos_sin_cache, pos_ids, kv_indices, positions_meta, kv_indptr, batch_indices,
            workspace_buffer, block_tables, seq_lens, max_seq_len,
        )
        kv_cache.copy_(kv_cache_initial)
        graph.replay()
        torch.cuda.synchronize()
        graph_attn_out = attn_out.clone()

        # --- Eager separated reference ---
        kv_cache_ref = kv_cache_initial.clone()
        kv_cache_dim = self.KV_LORA_RANK + self.QK_ROPE_HEAD_DIM

        q_out_sep = torch.empty(batch_size, self.NUM_HEADS, self.HEAD_DIM, device=device, dtype=attn_dtype)
        k_rope_out = torch.empty(batch_size, self.QK_ROPE_HEAD_DIM, device=device, dtype=attn_dtype)
        k_nope_out = torch.empty(batch_size, self.KV_LORA_RANK, device=device, dtype=attn_dtype)

        flashinfer.rope.mla_rope_quantize_fp8(
            q_rope=q_rope, k_rope=k_rope, q_nope=q_nope, k_nope=k_nope,
            cos_sin_cache=cos_sin_cache, pos_ids=pos_ids,
            is_neox=False, quantize_dtype=attn_dtype,
            q_rope_out=q_out_sep[..., self.KV_LORA_RANK:],
            k_rope_out=k_rope_out,
            q_nope_out=q_out_sep[..., :self.KV_LORA_RANK],
            k_nope_out=k_nope_out,
            quant_scale_q=1.0, quant_scale_kv=1.0,
        )

        k_combined = torch.cat([k_nope_out, k_rope_out], dim=-1)
        kv_cache_ref_squeezed = kv_cache_ref.squeeze(1)
        kv_cache_ref_flat = kv_cache_ref_squeezed.view(-1, kv_cache_dim)
        kv_cache_ref_flat[out_cache_loc] = k_combined

        query_sep = q_out_sep.unsqueeze(1)
        attn_out_sep = flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla(
            query=query_sep, kv_cache=kv_cache_ref, workspace_buffer=workspace_buffer,
            qk_nope_head_dim=self.QK_NOPE_HEAD_DIM, kv_lora_rank=self.KV_LORA_RANK,
            qk_rope_head_dim=self.QK_ROPE_HEAD_DIM, block_tables=block_tables,
            seq_lens=seq_lens, max_seq_len=max_seq_len,
            bmm1_scale=1.0 / (self.HEAD_DIM ** 0.5),
        )
        torch.cuda.synchronize()

        max_diff = torch.max(torch.abs(graph_attn_out.float() - attn_out_sep.float())).item()
        assert max_diff == 0.0, (
            f"CUDA graph fused vs eager separated mismatch! Max diff: {max_diff:.6e}\n"
            "The CUDA graph captures a consistently wrong answer compared to the eager separated path."
        )


###############################################################################
# Large batch eager-mode tests (no CUDA graphs)
###############################################################################


@pytest.mark.skipif(not _is_cuda, reason="CUDA not available")
@pytest.mark.skipif(not _has_flashinfer, reason="FlashInfer not available")
class TestMLAAttentionNonCUDAGraph:
    """
    Test MLA attention WITHOUT CUDA graphs (eager mode) at large batch sizes.

    This mimics production behavior when batch_size > cuda_graph_max_bs.
    Uses full DeepSeek V3/R1 dimensions (128 heads).
    """

    NUM_HEADS = 128
    KV_LORA_RANK = 512
    QK_ROPE_HEAD_DIM = 64
    QK_NOPE_HEAD_DIM = 128
    HEAD_DIM = KV_LORA_RANK + QK_ROPE_HEAD_DIM  # 576
    PAGE_SIZE = 64

    @pytest.fixture
    def device(self):
        return torch.device("cuda")

    @pytest.fixture
    def dtype(self):
        return torch.bfloat16

    @pytest.fixture
    def cos_sin_cache(self, device):
        return _create_cos_sin_cache_large(self.QK_ROPE_HEAD_DIM, device)

    @pytest.fixture
    def workspace_buffer(self, device):
        return torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=device)

    def _create_combined_kv_cache(self, num_pages: int, device: torch.device) -> torch.Tensor:
        return _create_combined_kv_cache_standalone(
            num_pages, self.PAGE_SIZE, self.KV_LORA_RANK, self.QK_ROPE_HEAD_DIM, device,
        )

    @pytest.mark.parametrize("batch_size", [512, 768, 1024, 1536, 2048, 3072, 4096])
    def test_large_batch_no_cuda_graph(self, device, dtype, cos_sin_cache, workspace_buffer, batch_size):
        """Test fused vs separated paths with large batch sizes (NO CUDA graphs)."""
        attn_dtype = torch.float8_e4m3fn
        alignment = 128 // self.PAGE_SIZE

        seq_len = 64
        max_seq_len = seq_len
        num_pages = ((batch_size + alignment - 1) // alignment) * alignment

        seq_lens = torch.full((batch_size,), seq_len, dtype=torch.int32, device=device)
        max_blocks_per_seq = alignment
        block_tables = torch.full((batch_size, max_blocks_per_seq), -1, dtype=torch.int32, device=device)
        for i in range(batch_size):
            if i < num_pages:
                block_tables[i, 0] = i

        out_cache_loc = torch.arange(batch_size, device=device, dtype=torch.int32) * self.PAGE_SIZE + 63
        out_cache_loc = torch.clamp(out_cache_loc, max=(num_pages - 1) * self.PAGE_SIZE + 63)
        pos_ids = (seq_lens - 1).to(torch.int32)

        torch.manual_seed(10000 + batch_size)
        q_nope = torch.randn(batch_size, self.NUM_HEADS, self.KV_LORA_RANK, device=device, dtype=dtype)
        q_rope = torch.randn(batch_size, self.NUM_HEADS, self.QK_ROPE_HEAD_DIM, device=device, dtype=dtype)
        k_nope = torch.randn(batch_size, self.KV_LORA_RANK, device=device, dtype=dtype)
        k_rope = torch.randn(batch_size, self.QK_ROPE_HEAD_DIM, device=device, dtype=dtype)

        kv_indices = (out_cache_loc // self.PAGE_SIZE).to(torch.int32)
        positions_meta = (out_cache_loc % self.PAGE_SIZE).to(torch.int32)
        kv_indptr = torch.arange(batch_size + 1, dtype=torch.int32, device=device)
        batch_indices = torch.arange(batch_size, dtype=torch.int32, device=device)

        kv_cache_fused = self._create_combined_kv_cache(num_pages, device)
        kv_cache_separated = self._create_combined_kv_cache(num_pages, device)

        torch.manual_seed(11000 + batch_size)
        for page_idx in range(min(batch_size, num_pages)):
            random_data = torch.randn(
                self.PAGE_SIZE, self.KV_LORA_RANK + self.QK_ROPE_HEAD_DIM,
                device=device, dtype=dtype,
            ).to(attn_dtype)
            kv_cache_fused[page_idx, 0, :, :] = random_data
            kv_cache_separated[page_idx, 0, :, :] = random_data.clone()

        q_out_fused = torch.empty(batch_size, self.NUM_HEADS, self.HEAD_DIM, device=device, dtype=attn_dtype)
        q_out_sep = torch.empty(batch_size, self.NUM_HEADS, self.HEAD_DIM, device=device, dtype=attn_dtype)

        # Fused path
        kv_cache_squeezed = kv_cache_fused.squeeze(1)
        ckv_cache = kv_cache_squeezed[:, :, :self.KV_LORA_RANK]
        kpe_cache = kv_cache_squeezed[:, :, self.KV_LORA_RANK:]
        flashinfer.rope.rope_quantize_fp8_append_paged_kv_cache(
            q_rope=q_rope, k_rope=k_rope, q_nope=q_nope, k_nope=k_nope,
            v=None, cos_sin_cache=cos_sin_cache, pos_ids=pos_ids,
            paged_kv_cache=(ckv_cache, kpe_cache),
            kv_indices=kv_indices, kv_indptr=kv_indptr,
            batch_indices=batch_indices, positions=positions_meta,
            is_neox=False, quantize_dtype=attn_dtype,
            quant_scale_q=1.0, quant_scale_kv=1.0,
            page_size=self.PAGE_SIZE, kv_layout="NHD",
            q_rope_out=q_out_fused[..., self.KV_LORA_RANK:],
            q_nope_out=q_out_fused[..., :self.KV_LORA_RANK],
        )
        kv_cache_squeezed[:, :, :self.KV_LORA_RANK] = ckv_cache
        kv_cache_squeezed[:, :, self.KV_LORA_RANK:] = kpe_cache

        query_fused = q_out_fused.unsqueeze(1)
        attn_out_fused = flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla(
            query=query_fused, kv_cache=kv_cache_fused, workspace_buffer=workspace_buffer,
            qk_nope_head_dim=self.QK_NOPE_HEAD_DIM, kv_lora_rank=self.KV_LORA_RANK,
            qk_rope_head_dim=self.QK_ROPE_HEAD_DIM, block_tables=block_tables,
            seq_lens=seq_lens, max_seq_len=max_seq_len,
            bmm1_scale=1.0 / (self.HEAD_DIM ** 0.5),
        )

        # Separated path
        kv_cache_dim = self.KV_LORA_RANK + self.QK_ROPE_HEAD_DIM
        k_rope_out = torch.empty(batch_size, self.QK_ROPE_HEAD_DIM, device=device, dtype=attn_dtype)
        k_nope_out = torch.empty(batch_size, self.KV_LORA_RANK, device=device, dtype=attn_dtype)

        flashinfer.rope.mla_rope_quantize_fp8(
            q_rope=q_rope, k_rope=k_rope, q_nope=q_nope, k_nope=k_nope,
            cos_sin_cache=cos_sin_cache, pos_ids=pos_ids,
            is_neox=False, quantize_dtype=attn_dtype,
            q_rope_out=q_out_sep[..., self.KV_LORA_RANK:],
            k_rope_out=k_rope_out,
            q_nope_out=q_out_sep[..., :self.KV_LORA_RANK],
            k_nope_out=k_nope_out,
            quant_scale_q=1.0, quant_scale_kv=1.0,
        )

        k_combined = torch.cat([k_nope_out, k_rope_out], dim=-1)
        kv_cache_sep_squeezed = kv_cache_separated.squeeze(1)
        kv_cache_sep_flat = kv_cache_sep_squeezed.view(-1, kv_cache_dim)
        kv_cache_sep_flat[out_cache_loc] = k_combined

        query_sep = q_out_sep.unsqueeze(1)
        attn_out_sep = flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla(
            query=query_sep, kv_cache=kv_cache_separated, workspace_buffer=workspace_buffer,
            qk_nope_head_dim=self.QK_NOPE_HEAD_DIM, kv_lora_rank=self.KV_LORA_RANK,
            qk_rope_head_dim=self.QK_ROPE_HEAD_DIM, block_tables=block_tables,
            seq_lens=seq_lens, max_seq_len=max_seq_len,
            bmm1_scale=1.0 / (self.HEAD_DIM ** 0.5),
        )

        torch.cuda.synchronize()

        q_diff = torch.max(torch.abs(q_out_fused.float() - q_out_sep.float())).item()
        kv_fused = kv_cache_fused.squeeze(1).view(-1, kv_cache_dim)[out_cache_loc]
        kv_sep = kv_cache_separated.squeeze(1).view(-1, kv_cache_dim)[out_cache_loc]
        kv_diff = torch.max(torch.abs(kv_fused.float() - kv_sep.float())).item()
        attn_diff = torch.max(torch.abs(attn_out_fused.float() - attn_out_sep.float())).item()

        assert q_diff == 0, f"Q mismatch at batch_size={batch_size}: max_diff={q_diff:.6e}"
        assert kv_diff == 0, f"KV cache mismatch at batch_size={batch_size}: max_diff={kv_diff:.6e}"
        assert attn_diff == 0, f"Attention output mismatch at batch_size={batch_size}: max_diff={attn_diff:.6e}"


###############################################################################
# Mixed CUDA graph / eager mode transition tests
###############################################################################


@pytest.mark.skipif(not _is_cuda, reason="CUDA not available")
@pytest.mark.skipif(not _has_flashinfer, reason="FlashInfer not available")
class TestMLAAttentionMixedModes:
    """
    Test MLA attention with MIXED CUDA graph and non-CUDA graph execution.

    This mimics production behavior where batch sizes vary during inference:
    - Small batches (<=512) use CUDA graphs
    - Large batches (>512) fall back to eager mode
    """

    NUM_HEADS = 128
    KV_LORA_RANK = 512
    QK_ROPE_HEAD_DIM = 64
    QK_NOPE_HEAD_DIM = 128
    HEAD_DIM = KV_LORA_RANK + QK_ROPE_HEAD_DIM  # 576
    PAGE_SIZE = 64

    @pytest.fixture
    def device(self):
        return torch.device("cuda")

    @pytest.fixture
    def dtype(self):
        return torch.bfloat16

    @pytest.fixture
    def cos_sin_cache(self, device):
        return _create_cos_sin_cache_large(self.QK_ROPE_HEAD_DIM, device)

    @pytest.fixture
    def workspace_buffer(self, device):
        return torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=device)

    def _create_combined_kv_cache(self, num_pages: int, device: torch.device) -> torch.Tensor:
        return _create_combined_kv_cache_standalone(
            num_pages, self.PAGE_SIZE, self.KV_LORA_RANK, self.QK_ROPE_HEAD_DIM, device,
        )

    def _run_fused_iteration(
        self, batch_size, device, dtype, cos_sin_cache, workspace_buffer,
        use_cuda_graph, seed,
    ) -> torch.Tensor:
        """Run a single iteration with the fused path, optionally using CUDA graph."""
        attn_dtype = torch.float8_e4m3fn
        alignment = 128 // self.PAGE_SIZE

        seq_len = 64
        max_seq_len = seq_len
        num_pages = ((batch_size + alignment - 1) // alignment) * alignment

        seq_lens = torch.full((batch_size,), seq_len, dtype=torch.int32, device=device)
        max_blocks_per_seq = alignment
        block_tables = torch.full((batch_size, max_blocks_per_seq), -1, dtype=torch.int32, device=device)
        for i in range(batch_size):
            if i < num_pages:
                block_tables[i, 0] = i

        out_cache_loc = torch.arange(batch_size, device=device, dtype=torch.int32) * self.PAGE_SIZE + 63
        out_cache_loc = torch.clamp(out_cache_loc, max=(num_pages - 1) * self.PAGE_SIZE + 63)
        pos_ids = (seq_lens - 1).to(torch.int32)

        kv_indices = (out_cache_loc // self.PAGE_SIZE).to(torch.int32)
        positions = (out_cache_loc % self.PAGE_SIZE).to(torch.int32)
        kv_indptr = torch.arange(batch_size + 1, dtype=torch.int32, device=device)
        batch_indices = torch.arange(batch_size, dtype=torch.int32, device=device)

        torch.manual_seed(seed)
        q_nope = torch.randn(batch_size, self.NUM_HEADS, self.KV_LORA_RANK, device=device, dtype=dtype)
        q_rope = torch.randn(batch_size, self.NUM_HEADS, self.QK_ROPE_HEAD_DIM, device=device, dtype=dtype)
        k_nope = torch.randn(batch_size, self.KV_LORA_RANK, device=device, dtype=dtype)
        k_rope = torch.randn(batch_size, self.QK_ROPE_HEAD_DIM, device=device, dtype=dtype)

        q_out = torch.empty(batch_size, self.NUM_HEADS, self.HEAD_DIM, device=device, dtype=attn_dtype)
        kv_cache = self._create_combined_kv_cache(num_pages, device)
        attn_out = torch.empty(batch_size, 1, self.NUM_HEADS, self.KV_LORA_RANK, device=device, dtype=dtype)

        torch.manual_seed(seed + 1000)
        for page_idx in range(min(batch_size, num_pages)):
            kv_cache[page_idx, 0, :, :] = torch.randn(
                self.PAGE_SIZE, self.KV_LORA_RANK + self.QK_ROPE_HEAD_DIM,
                device=device, dtype=dtype,
            ).to(attn_dtype)

        def run_ops():
            kv_cache_squeezed = kv_cache.squeeze(1)
            ckv_cache = kv_cache_squeezed[:, :, :self.KV_LORA_RANK]
            kpe_cache = kv_cache_squeezed[:, :, self.KV_LORA_RANK:]
            flashinfer.rope.rope_quantize_fp8_append_paged_kv_cache(
                q_rope=q_rope, k_rope=k_rope, q_nope=q_nope, k_nope=k_nope,
                v=None, cos_sin_cache=cos_sin_cache, pos_ids=pos_ids,
                paged_kv_cache=(ckv_cache, kpe_cache),
                kv_indices=kv_indices, kv_indptr=kv_indptr,
                batch_indices=batch_indices, positions=positions,
                is_neox=False, quantize_dtype=attn_dtype,
                quant_scale_q=1.0, quant_scale_kv=1.0,
                page_size=self.PAGE_SIZE, kv_layout="NHD",
                q_rope_out=q_out[..., self.KV_LORA_RANK:],
                q_nope_out=q_out[..., :self.KV_LORA_RANK],
            )
            kv_cache_squeezed[:, :, :self.KV_LORA_RANK] = ckv_cache
            kv_cache_squeezed[:, :, self.KV_LORA_RANK:] = kpe_cache
            query = q_out.unsqueeze(1)
            return flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla(
                query=query, kv_cache=kv_cache, workspace_buffer=workspace_buffer,
                qk_nope_head_dim=self.QK_NOPE_HEAD_DIM, kv_lora_rank=self.KV_LORA_RANK,
                qk_rope_head_dim=self.QK_ROPE_HEAD_DIM, block_tables=block_tables,
                seq_lens=seq_lens, max_seq_len=max_seq_len,
                bmm1_scale=1.0 / (self.HEAD_DIM ** 0.5),
            )

        if use_cuda_graph:
            for _ in range(3):
                run_ops()
            torch.cuda.synchronize()

            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    out = run_ops()
                    attn_out.copy_(out)
            torch.cuda.current_stream().wait_stream(stream)
            torch.cuda.synchronize()
            graph.replay()
            torch.cuda.synchronize()
        else:
            out = run_ops()
            attn_out.copy_(out)
            torch.cuda.synchronize()

        return attn_out.clone()

    def test_alternating_cuda_graph_and_eager(self, device, dtype, cos_sin_cache, workspace_buffer):
        """Test alternating between CUDA graph and eager mode execution."""
        batch_sizes = [64, 1024, 128, 2048, 256, 768, 512, 1536]
        use_graph = [True, False, True, False, True, False, True, False]

        results_first_pass = []
        results_second_pass = []

        for bs, use_g in zip(batch_sizes, use_graph):
            result = self._run_fused_iteration(
                batch_size=bs, device=device, dtype=dtype,
                cos_sin_cache=cos_sin_cache, workspace_buffer=workspace_buffer,
                use_cuda_graph=use_g, seed=20000 + bs,
            )
            results_first_pass.append(result)

        for bs, use_g in zip(batch_sizes, use_graph):
            result = self._run_fused_iteration(
                batch_size=bs, device=device, dtype=dtype,
                cos_sin_cache=cos_sin_cache, workspace_buffer=workspace_buffer,
                use_cuda_graph=use_g, seed=20000 + bs,
            )
            results_second_pass.append(result)

        for i, (r1, r2) in enumerate(zip(results_first_pass, results_second_pass)):
            diff = torch.max(torch.abs(r1.float() - r2.float())).item()
            assert diff == 0, (
                f"Results differ at iteration {i} (batch_size={batch_sizes[i]}, "
                f"use_graph={use_graph[i]}): max_diff={diff:.6e}"
            )

    def test_mode_transition_consistency(self, device, dtype, cos_sin_cache, workspace_buffer):
        """Test that the same batch size produces identical results via graph or eager."""
        for batch_size in [64, 128, 256, 512]:
            result_graph = self._run_fused_iteration(
                batch_size=batch_size, device=device, dtype=dtype,
                cos_sin_cache=cos_sin_cache, workspace_buffer=workspace_buffer,
                use_cuda_graph=True, seed=30000 + batch_size,
            )
            result_eager = self._run_fused_iteration(
                batch_size=batch_size, device=device, dtype=dtype,
                cos_sin_cache=cos_sin_cache, workspace_buffer=workspace_buffer,
                use_cuda_graph=False, seed=30000 + batch_size,
            )
            diff = torch.max(torch.abs(result_graph.float() - result_eager.float())).item()
            assert diff == 0, (
                f"CUDA graph vs eager mismatch at batch_size={batch_size}: max_diff={diff:.6e}"
            )

    def test_rapid_mode_switching(self, device, dtype, cos_sin_cache, workspace_buffer):
        """Stress test: 50 iterations of rapid switching between CUDA graph and eager modes."""
        num_iterations = 50
        results = []

        for i in range(num_iterations):
            if i % 2 == 0:
                batch_size = 128
                use_graph = True
            else:
                batch_size = 1024
                use_graph = False

            result = self._run_fused_iteration(
                batch_size=batch_size, device=device, dtype=dtype,
                cos_sin_cache=cos_sin_cache, workspace_buffer=workspace_buffer,
                use_cuda_graph=use_graph, seed=40000 + i,
            )
            results.append((batch_size, use_graph, result))

        for i, (batch_size, use_graph, expected) in enumerate(results):
            actual = self._run_fused_iteration(
                batch_size=batch_size, device=device, dtype=dtype,
                cos_sin_cache=cos_sin_cache, workspace_buffer=workspace_buffer,
                use_cuda_graph=use_graph, seed=40000 + i,
            )
            diff = torch.max(torch.abs(expected.float() - actual.float())).item()
            assert diff == 0, (
                f"Mismatch at iteration {i} (batch_size={batch_size}, "
                f"use_graph={use_graph}): max_diff={diff:.6e}"
            )


###############################################################################
# Draft extend / speculative decoding tests
###############################################################################


@pytest.mark.skipif(not _is_cuda, reason="CUDA not available")
@pytest.mark.skipif(not _has_flashinfer, reason="FlashInfer not available")
class TestMLAAttentionDraftExtend:
    """
    Test MLA attention in draft_extend mode (speculative decoding).

    draft_extend differs from regular decode:
    1. Multiple query tokens per sequence (accept_length + 1)
    2. Variable accept_lengths per sequence
    3. Query padding/unpadding for variable lengths

    The fused kernel writes multiple KV entries per sequence.
    """

    NUM_HEADS = 128
    KV_LORA_RANK = 512
    QK_ROPE_HEAD_DIM = 64
    QK_NOPE_HEAD_DIM = 128
    HEAD_DIM = KV_LORA_RANK + QK_ROPE_HEAD_DIM  # 576
    PAGE_SIZE = 64

    @pytest.fixture
    def device(self):
        return torch.device("cuda")

    @pytest.fixture
    def dtype(self):
        return torch.bfloat16

    @pytest.fixture
    def cos_sin_cache(self, device):
        return _create_cos_sin_cache_large(self.QK_ROPE_HEAD_DIM, device)

    @pytest.fixture
    def workspace_buffer(self, device):
        return torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=device)

    def _create_combined_kv_cache(self, num_pages: int, device: torch.device) -> torch.Tensor:
        return _create_combined_kv_cache_standalone(
            num_pages, self.PAGE_SIZE, self.KV_LORA_RANK, self.QK_ROPE_HEAD_DIM, device,
        )

    @pytest.mark.parametrize("batch_size", [4, 16, 64, 128])
    def test_draft_extend_uniform_accept_lengths(self, device, dtype, cos_sin_cache, workspace_buffer, batch_size):
        """Test draft_extend with uniform accept_lengths (all sequences accept same number)."""
        attn_dtype = torch.float8_e4m3fn
        alignment = 128 // self.PAGE_SIZE
        num_draft_tokens = 4

        total_tokens = batch_size * num_draft_tokens
        context_len = 128
        tokens_per_seq = context_len + num_draft_tokens
        pages_per_seq = (tokens_per_seq + self.PAGE_SIZE - 1) // self.PAGE_SIZE
        num_pages = batch_size * pages_per_seq
        num_pages = ((num_pages + alignment - 1) // alignment) * alignment

        max_blocks_per_seq = pages_per_seq
        max_blocks_per_seq = ((max_blocks_per_seq + alignment - 1) // alignment) * alignment
        block_tables = torch.full((batch_size, max_blocks_per_seq), -1, dtype=torch.int32, device=device)

        out_cache_locs = []
        page_counter = 0
        for seq_idx in range(batch_size):
            for block_idx in range(pages_per_seq):
                if page_counter < num_pages:
                    block_tables[seq_idx, block_idx] = page_counter
                    page_counter += 1
            for token_idx in range(num_draft_tokens):
                pos = context_len + token_idx
                page_idx = pos // self.PAGE_SIZE
                pos_in_page = pos % self.PAGE_SIZE
                block = block_tables[seq_idx, page_idx].item()
                out_cache_locs.append(block * self.PAGE_SIZE + pos_in_page)

        out_cache_loc = torch.tensor(out_cache_locs, device=device, dtype=torch.int32)

        pos_ids_list = []
        for seq_idx in range(batch_size):
            for token_idx in range(num_draft_tokens):
                pos_ids_list.append(context_len + token_idx)
        pos_ids = torch.tensor(pos_ids_list, device=device, dtype=torch.int32)

        seq_lens = torch.full((batch_size,), context_len + num_draft_tokens, dtype=torch.int32, device=device)
        max_seq_len = seq_lens.max().item()

        kv_indices = (out_cache_loc // self.PAGE_SIZE).to(torch.int32)
        positions = (out_cache_loc % self.PAGE_SIZE).to(torch.int32)
        kv_indptr = torch.arange(total_tokens + 1, dtype=torch.int32, device=device)
        batch_indices = torch.arange(total_tokens, dtype=torch.int32, device=device)

        torch.manual_seed(50000 + batch_size)
        q_nope = torch.randn(total_tokens, self.NUM_HEADS, self.KV_LORA_RANK, device=device, dtype=dtype)
        q_rope = torch.randn(total_tokens, self.NUM_HEADS, self.QK_ROPE_HEAD_DIM, device=device, dtype=dtype)
        k_nope = torch.randn(total_tokens, self.KV_LORA_RANK, device=device, dtype=dtype)
        k_rope = torch.randn(total_tokens, self.QK_ROPE_HEAD_DIM, device=device, dtype=dtype)

        kv_cache_fused = self._create_combined_kv_cache(num_pages, device)
        kv_cache_separated = self._create_combined_kv_cache(num_pages, device)

        torch.manual_seed(51000 + batch_size)
        for page_idx in range(num_pages):
            kv_cache_fused[page_idx, 0, :, :] = torch.randn(
                self.PAGE_SIZE, self.KV_LORA_RANK + self.QK_ROPE_HEAD_DIM,
                device=device, dtype=dtype,
            ).to(attn_dtype)
            kv_cache_separated[page_idx, 0, :, :] = kv_cache_fused[page_idx, 0, :, :].clone()

        q_out_fused = torch.empty(total_tokens, self.NUM_HEADS, self.HEAD_DIM, device=device, dtype=attn_dtype)
        q_out_sep = torch.empty(total_tokens, self.NUM_HEADS, self.HEAD_DIM, device=device, dtype=attn_dtype)

        # Fused path
        kv_cache_squeezed = kv_cache_fused.squeeze(1)
        ckv_cache = kv_cache_squeezed[:, :, :self.KV_LORA_RANK]
        kpe_cache = kv_cache_squeezed[:, :, self.KV_LORA_RANK:]
        flashinfer.rope.rope_quantize_fp8_append_paged_kv_cache(
            q_rope=q_rope, k_rope=k_rope, q_nope=q_nope, k_nope=k_nope,
            v=None, cos_sin_cache=cos_sin_cache, pos_ids=pos_ids,
            paged_kv_cache=(ckv_cache, kpe_cache),
            kv_indices=kv_indices, kv_indptr=kv_indptr,
            batch_indices=batch_indices, positions=positions,
            is_neox=False, quantize_dtype=attn_dtype,
            quant_scale_q=1.0, quant_scale_kv=1.0,
            page_size=self.PAGE_SIZE, kv_layout="NHD",
            q_rope_out=q_out_fused[..., self.KV_LORA_RANK:],
            q_nope_out=q_out_fused[..., :self.KV_LORA_RANK],
        )
        kv_cache_squeezed[:, :, :self.KV_LORA_RANK] = ckv_cache
        kv_cache_squeezed[:, :, self.KV_LORA_RANK:] = kpe_cache

        query_fused = q_out_fused.view(batch_size, num_draft_tokens, self.NUM_HEADS, self.HEAD_DIM)
        attn_out_fused = flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla(
            query=query_fused, kv_cache=kv_cache_fused, workspace_buffer=workspace_buffer,
            qk_nope_head_dim=self.QK_NOPE_HEAD_DIM, kv_lora_rank=self.KV_LORA_RANK,
            qk_rope_head_dim=self.QK_ROPE_HEAD_DIM, block_tables=block_tables,
            seq_lens=seq_lens, max_seq_len=max_seq_len,
            bmm1_scale=1.0 / (self.HEAD_DIM ** 0.5),
        )

        # Separated path
        kv_cache_dim = self.KV_LORA_RANK + self.QK_ROPE_HEAD_DIM
        k_rope_out = torch.empty(total_tokens, self.QK_ROPE_HEAD_DIM, device=device, dtype=attn_dtype)
        k_nope_out = torch.empty(total_tokens, self.KV_LORA_RANK, device=device, dtype=attn_dtype)

        flashinfer.rope.mla_rope_quantize_fp8(
            q_rope=q_rope, k_rope=k_rope, q_nope=q_nope, k_nope=k_nope,
            cos_sin_cache=cos_sin_cache, pos_ids=pos_ids,
            is_neox=False, quantize_dtype=attn_dtype,
            q_rope_out=q_out_sep[..., self.KV_LORA_RANK:],
            k_rope_out=k_rope_out,
            q_nope_out=q_out_sep[..., :self.KV_LORA_RANK],
            k_nope_out=k_nope_out,
            quant_scale_q=1.0, quant_scale_kv=1.0,
        )

        k_combined = torch.cat([k_nope_out, k_rope_out], dim=-1)
        kv_cache_sep_squeezed = kv_cache_separated.squeeze(1)
        kv_cache_sep_flat = kv_cache_sep_squeezed.view(-1, kv_cache_dim)
        kv_cache_sep_flat[out_cache_loc] = k_combined

        query_sep = q_out_sep.view(batch_size, num_draft_tokens, self.NUM_HEADS, self.HEAD_DIM)
        attn_out_sep = flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla(
            query=query_sep, kv_cache=kv_cache_separated, workspace_buffer=workspace_buffer,
            qk_nope_head_dim=self.QK_NOPE_HEAD_DIM, kv_lora_rank=self.KV_LORA_RANK,
            qk_rope_head_dim=self.QK_ROPE_HEAD_DIM, block_tables=block_tables,
            seq_lens=seq_lens, max_seq_len=max_seq_len,
            bmm1_scale=1.0 / (self.HEAD_DIM ** 0.5),
        )

        torch.cuda.synchronize()

        q_diff = torch.max(torch.abs(q_out_fused.float() - q_out_sep.float())).item()
        kv_fused = kv_cache_fused.squeeze(1).view(-1, kv_cache_dim)[out_cache_loc]
        kv_sep = kv_cache_separated.squeeze(1).view(-1, kv_cache_dim)[out_cache_loc]
        kv_diff = torch.max(torch.abs(kv_fused.float() - kv_sep.float())).item()
        attn_diff = torch.max(torch.abs(attn_out_fused.float() - attn_out_sep.float())).item()

        assert q_diff == 0, f"Q mismatch: max_diff={q_diff:.6e}"
        assert kv_diff == 0, f"KV cache mismatch: max_diff={kv_diff:.6e}"
        assert attn_diff == 0, f"Attention output mismatch: max_diff={attn_diff:.6e}"

    @pytest.mark.parametrize("batch_size", [4, 16, 64])
    def test_draft_extend_variable_accept_lengths(self, device, dtype, cos_sin_cache, workspace_buffer, batch_size):
        """Test draft_extend with variable accept_lengths per sequence."""
        attn_dtype = torch.float8_e4m3fn
        alignment = 128 // self.PAGE_SIZE

        torch.manual_seed(52000 + batch_size)
        accept_lengths = torch.randint(0, 8, (batch_size,), device=device, dtype=torch.int32)
        tokens_per_seq = accept_lengths + 1
        total_tokens = tokens_per_seq.sum().item()
        max_tokens_per_seq = tokens_per_seq.max().item()

        context_len = 128
        max_seq_len_with_draft = context_len + max_tokens_per_seq
        pages_per_seq = (max_seq_len_with_draft + self.PAGE_SIZE - 1) // self.PAGE_SIZE
        num_pages = batch_size * pages_per_seq
        num_pages = ((num_pages + alignment - 1) // alignment) * alignment

        max_blocks_per_seq = ((pages_per_seq + alignment - 1) // alignment) * alignment
        block_tables = torch.full((batch_size, max_blocks_per_seq), -1, dtype=torch.int32, device=device)

        out_cache_locs = []
        page_counter = 0
        for seq_idx in range(batch_size):
            for block_idx in range(pages_per_seq):
                if page_counter < num_pages:
                    block_tables[seq_idx, block_idx] = page_counter
                    page_counter += 1
            num_tokens_this_seq = tokens_per_seq[seq_idx].item()
            for token_idx in range(num_tokens_this_seq):
                pos = context_len + token_idx
                page_idx = pos // self.PAGE_SIZE
                pos_in_page = pos % self.PAGE_SIZE
                block = block_tables[seq_idx, page_idx].item()
                out_cache_locs.append(block * self.PAGE_SIZE + pos_in_page)

        out_cache_loc = torch.tensor(out_cache_locs, device=device, dtype=torch.int32)

        pos_ids_list = []
        for seq_idx in range(batch_size):
            num_tokens_this_seq = tokens_per_seq[seq_idx].item()
            for token_idx in range(num_tokens_this_seq):
                pos_ids_list.append(context_len + token_idx)
        pos_ids = torch.tensor(pos_ids_list, device=device, dtype=torch.int32)

        seq_lens = torch.full((batch_size,), context_len, dtype=torch.int32, device=device) + tokens_per_seq
        max_seq_len = seq_lens.max().item()

        kv_indices = (out_cache_loc // self.PAGE_SIZE).to(torch.int32)
        positions = (out_cache_loc % self.PAGE_SIZE).to(torch.int32)
        kv_indptr = torch.arange(total_tokens + 1, dtype=torch.int32, device=device)
        batch_indices = torch.arange(total_tokens, dtype=torch.int32, device=device)

        torch.manual_seed(53000 + batch_size)
        q_nope = torch.randn(total_tokens, self.NUM_HEADS, self.KV_LORA_RANK, device=device, dtype=dtype)
        q_rope = torch.randn(total_tokens, self.NUM_HEADS, self.QK_ROPE_HEAD_DIM, device=device, dtype=dtype)
        k_nope = torch.randn(total_tokens, self.KV_LORA_RANK, device=device, dtype=dtype)
        k_rope = torch.randn(total_tokens, self.QK_ROPE_HEAD_DIM, device=device, dtype=dtype)

        kv_cache_fused = self._create_combined_kv_cache(num_pages, device)
        kv_cache_separated = self._create_combined_kv_cache(num_pages, device)

        torch.manual_seed(54000 + batch_size)
        for page_idx in range(num_pages):
            kv_cache_fused[page_idx, 0, :, :] = torch.randn(
                self.PAGE_SIZE, self.KV_LORA_RANK + self.QK_ROPE_HEAD_DIM,
                device=device, dtype=dtype,
            ).to(attn_dtype)
            kv_cache_separated[page_idx, 0, :, :] = kv_cache_fused[page_idx, 0, :, :].clone()

        q_out_fused = torch.empty(total_tokens, self.NUM_HEADS, self.HEAD_DIM, device=device, dtype=attn_dtype)
        q_out_sep = torch.empty(total_tokens, self.NUM_HEADS, self.HEAD_DIM, device=device, dtype=attn_dtype)

        # Fused path
        kv_cache_squeezed = kv_cache_fused.squeeze(1)
        ckv_cache = kv_cache_squeezed[:, :, :self.KV_LORA_RANK]
        kpe_cache = kv_cache_squeezed[:, :, self.KV_LORA_RANK:]
        flashinfer.rope.rope_quantize_fp8_append_paged_kv_cache(
            q_rope=q_rope, k_rope=k_rope, q_nope=q_nope, k_nope=k_nope,
            v=None, cos_sin_cache=cos_sin_cache, pos_ids=pos_ids,
            paged_kv_cache=(ckv_cache, kpe_cache),
            kv_indices=kv_indices, kv_indptr=kv_indptr,
            batch_indices=batch_indices, positions=positions,
            is_neox=False, quantize_dtype=attn_dtype,
            quant_scale_q=1.0, quant_scale_kv=1.0,
            page_size=self.PAGE_SIZE, kv_layout="NHD",
            q_rope_out=q_out_fused[..., self.KV_LORA_RANK:],
            q_nope_out=q_out_fused[..., :self.KV_LORA_RANK],
        )
        kv_cache_squeezed[:, :, :self.KV_LORA_RANK] = ckv_cache
        kv_cache_squeezed[:, :, self.KV_LORA_RANK:] = kpe_cache

        # Separated path
        kv_cache_dim = self.KV_LORA_RANK + self.QK_ROPE_HEAD_DIM
        k_rope_out = torch.empty(total_tokens, self.QK_ROPE_HEAD_DIM, device=device, dtype=attn_dtype)
        k_nope_out = torch.empty(total_tokens, self.KV_LORA_RANK, device=device, dtype=attn_dtype)

        flashinfer.rope.mla_rope_quantize_fp8(
            q_rope=q_rope, k_rope=k_rope, q_nope=q_nope, k_nope=k_nope,
            cos_sin_cache=cos_sin_cache, pos_ids=pos_ids,
            is_neox=False, quantize_dtype=attn_dtype,
            q_rope_out=q_out_sep[..., self.KV_LORA_RANK:],
            k_rope_out=k_rope_out,
            q_nope_out=q_out_sep[..., :self.KV_LORA_RANK],
            k_nope_out=k_nope_out,
            quant_scale_q=1.0, quant_scale_kv=1.0,
        )

        k_combined = torch.cat([k_nope_out, k_rope_out], dim=-1)
        kv_cache_sep_squeezed = kv_cache_separated.squeeze(1)
        kv_cache_sep_flat = kv_cache_sep_squeezed.view(-1, kv_cache_dim)
        kv_cache_sep_flat[out_cache_loc] = k_combined

        torch.cuda.synchronize()

        q_diff = torch.max(torch.abs(q_out_fused.float() - q_out_sep.float())).item()
        kv_fused = kv_cache_fused.squeeze(1).view(-1, kv_cache_dim)[out_cache_loc]
        kv_sep = kv_cache_separated.squeeze(1).view(-1, kv_cache_dim)[out_cache_loc]
        kv_diff = torch.max(torch.abs(kv_fused.float() - kv_sep.float())).item()

        assert q_diff == 0, f"Q mismatch: max_diff={q_diff:.6e}"
        assert kv_diff == 0, f"KV cache mismatch: max_diff={kv_diff:.6e}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
