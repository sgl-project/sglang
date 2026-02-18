"""
Tests for flashinfer.rope.rope_quantize_fp8_append_paged_kv_cache.

This fused kernel replaces two separate calls (mla_rope_quantize_fp8 + set_mla_kv_buffer)
with a single call that performs RoPE, FP8 quantization, and paged KV cache append.
These tests verify the fused kernel is bit-identical to the separated path.

Test cases:
    - test_fused_correctness: Parametrized across DeepSeek V3/R1 production
      dimensions (batch_size, num_heads for TP=1/2/4/8, page_size=32/64).
      Each case checks fused-vs-separated equivalence, no-spurious-writes
      sentinel check, and combined non-contiguous buffer verification.
    - test_kv_indptr_contract_mismatch: Intentionally violates kv_indptr semantics
      and verifies KV cache writes are corrupted, then restored.
    - test_high_rope_positions_deepseek: High RoPE position indices near the
      cos_sin_cache upper bound (up to 8191).
    - test_permuted_out_cache_loc_order: Non-monotonic (shuffled) metadata
      ordering to verify the kernel handles arbitrary write order.
    - test_boundary_position_ids_deepseek: RoPE positions around power-of-2
      boundaries (1023/1024, 2047/2048, 4094/4095).
"""

import pytest
import torch

from sglang.srt.utils import is_cuda, is_flashinfer_available
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=2, suite="nightly-1-gpu", nightly=True)

_is_cuda = is_cuda()
_has_flashinfer = is_flashinfer_available()

if _has_flashinfer:
    import flashinfer


def create_cos_sin_cache(max_seq_len: int, rotary_dim: int, device: torch.device):
    """Create a cos/sin cache for RoPE.

    Returns a tensor of shape (max_seq_len, rotary_dim) containing concatenated
    cos and sin values: [cos(0), cos(1), ..., sin(0), sin(1), ...]
    Must be float32 for the FlashInfer RoPE kernel.
    """
    freqs = 1.0 / (
        10000 ** (torch.arange(0, rotary_dim, 2, device=device).float() / rotary_dim)
    )
    t = torch.arange(max_seq_len, device=device).float()
    freqs = torch.outer(t, freqs)
    cos = torch.cos(freqs)
    sin = torch.sin(freqs)
    # FlashInfer expects [cos_0..cos_{d/2-1}, sin_0..sin_{d/2-1}]
    cos_sin_cache = torch.cat([cos, sin], dim=-1)
    return cos_sin_cache


@pytest.mark.skipif(not _is_cuda, reason="CUDA not available")
@pytest.mark.skipif(not _has_flashinfer, reason="FlashInfer not available")
class TestFusedVsSeparatedPath:
    """
    Compare rope_quantize_fp8_append_paged_kv_cache vs mla_rope_quantize_fp8 + manual KV write.

    These two paths should produce identical results:
    1. Fused: rope_quantize_fp8_append_paged_kv_cache (RoPE + quantize + KV append in one call)
    2. Separated: mla_rope_quantize_fp8 + manual KV cache write (set_mla_kv_buffer equivalent)
    """
    
    # DeepSeek V3/R1 MLA dimensions (fixed by model architecture)
    KV_LORA_RANK = 512
    QK_ROPE_HEAD_DIM = 64
    
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
    
    @staticmethod
    def _generate_cache_locs(batch_size: int, page_size: int, device: torch.device):
        """Generate deterministic scattered cache locations across pages.

        Each token is assigned to its own page with a varying offset,
        ensuring writes hit different positions within pages.
        """
        pages = torch.arange(batch_size, device=device, dtype=torch.int32)
        offsets = (pages * 7 + 3) % page_size
        return pages * page_size + offsets

    def _create_paged_kv_cache(
        self, num_pages: int, page_size: int, device: torch.device,
    ) -> tuple:
        """Create a paged KV cache buffer for testing.

        Returns:
            tuple: (ckv_cache, kpe_cache) where:
                - ckv_cache: [num_pages, page_size, kv_lora_rank] for k_nope
                - kpe_cache: [num_pages, page_size, qk_rope_head_dim] for k_rope
        """
        attn_dtype = torch.float8_e4m3fn
        ckv_cache = torch.zeros(
            num_pages, page_size, self.KV_LORA_RANK,
            device=device, dtype=attn_dtype
        )
        kpe_cache = torch.zeros(
            num_pages, page_size, self.QK_ROPE_HEAD_DIM,
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
        page_size: int,
    ) -> torch.Tensor:
        """Run the fused path: rope_quantize_fp8_append_paged_kv_cache."""
        nnz = out_cache_loc.shape[0]
        device = out_cache_loc.device
        attn_dtype = torch.float8_e4m3fn
        assert q_nope.shape[0] == nnz, "q_nope batch must match out_cache_loc length"
        assert q_rope.shape[0] == nnz, "q_rope batch must match out_cache_loc length"
        assert k_nope.shape[0] == nnz, "k_nope batch must match out_cache_loc length"
        assert k_rope.shape[0] == nnz, "k_rope batch must match out_cache_loc length"
        assert pos_ids.shape[0] == nnz, "pos_ids length must match out_cache_loc length"

        q_out = torch.empty(
            nnz,
            q_rope.shape[1],
            self.KV_LORA_RANK + self.QK_ROPE_HEAD_DIM,
            device=device,
            dtype=attn_dtype,
        )

        kv_indices = (out_cache_loc // page_size).to(torch.int32)
        positions = (out_cache_loc % page_size).to(torch.int32)
        kv_indptr = torch.arange(nnz + 1, dtype=torch.int32, device=device)
        batch_indices = torch.arange(nnz, dtype=torch.int32, device=device)

        flashinfer.rope.rope_quantize_fp8_append_paged_kv_cache(
            q_rope=q_rope,
            k_rope=k_rope,
            q_nope=q_nope,
            k_nope=k_nope,
            v=None,
            cos_sin_cache=cos_sin_cache,
            pos_ids=pos_ids,
            paged_kv_cache=(ckv_cache, kpe_cache),
            kv_indices=kv_indices,
            kv_indptr=kv_indptr,
            batch_indices=batch_indices,
            positions=positions,
            is_neox=False,
            quantize_dtype=attn_dtype,
            quant_scale_q=1.0,
            quant_scale_kv=1.0,
            page_size=page_size,
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
        page_size: int,
    ) -> torch.Tensor:
        """Run the separated path: mla_rope_quantize_fp8 + manual KV cache write."""
        nnz = out_cache_loc.shape[0]
        device = q_rope.device
        attn_dtype = torch.float8_e4m3fn
        assert q_nope.shape[0] == nnz, "q_nope batch must match out_cache_loc length"
        assert q_rope.shape[0] == nnz, "q_rope batch must match out_cache_loc length"
        assert k_nope.shape[0] == nnz, "k_nope batch must match out_cache_loc length"
        assert k_rope.shape[0] == nnz, "k_rope batch must match out_cache_loc length"
        assert pos_ids.shape[0] == nnz, "pos_ids length must match out_cache_loc length"

        q_out = torch.empty(
            nnz,
            q_rope.shape[1],
            self.KV_LORA_RANK + self.QK_ROPE_HEAD_DIM,
            device=device,
            dtype=attn_dtype,
        )
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

        kv_indices = out_cache_loc // page_size
        positions = out_cache_loc % page_size

        for i in range(nnz):
            page_idx = kv_indices[i].item()
            pos_in_page = positions[i].item()
            ckv_cache[page_idx, pos_in_page, :] = k_nope_out[i]
            kpe_cache[page_idx, pos_in_page, :] = k_rope_out[i]

        return q_out

    def _assert_fused_separated_parity(
        self,
        *,
        q_nope: torch.Tensor,
        q_rope: torch.Tensor,
        k_nope: torch.Tensor,
        k_rope: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        pos_ids: torch.Tensor,
        out_cache_loc: torch.Tensor,
        page_size: int,
        num_pages: int,
        sentinel_val: torch.Tensor,
        label: str = "",
    ):
        """Run fused/separated paths and assert parity on outputs and KV writes."""
        attn_dtype = torch.float8_e4m3fn
        ckv_fused = torch.full(
            (num_pages, page_size, self.KV_LORA_RANK),
            sentinel_val.item(),
            device=q_nope.device,
            dtype=attn_dtype,
        )
        kpe_fused = torch.full(
            (num_pages, page_size, self.QK_ROPE_HEAD_DIM),
            sentinel_val.item(),
            device=q_nope.device,
            dtype=attn_dtype,
        )
        ckv_fused_snap = ckv_fused.clone()
        kpe_fused_snap = kpe_fused.clone()
        ckv_sep = torch.full_like(ckv_fused, sentinel_val.item())
        kpe_sep = torch.full_like(kpe_fused, sentinel_val.item())

        q_out_fused = self._run_fused_path(
            q_nope,
            q_rope,
            k_nope,
            k_rope,
            cos_sin_cache,
            pos_ids,
            out_cache_loc,
            ckv_fused,
            kpe_fused,
            page_size,
        )
        q_out_sep = self._run_separated_path(
            q_nope,
            q_rope,
            k_nope,
            k_rope,
            cos_sin_cache,
            pos_ids,
            out_cache_loc,
            ckv_sep,
            kpe_sep,
            page_size,
        )

        err = f"{label} " if label else ""
        q_diff = torch.max(torch.abs(q_out_fused.float() - q_out_sep.float())).item()
        assert q_diff == 0.0, f"{err}q_out differs between paths! Max diff: {q_diff}"

        kv_indices = out_cache_loc // page_size
        positions = out_cache_loc % page_size
        batch_size = out_cache_loc.shape[0]
        for i in range(batch_size):
            p, s = kv_indices[i].item(), positions[i].item()
            ckv_d = torch.max(
                torch.abs(ckv_fused[p, s].float() - ckv_sep[p, s].float())
            ).item()
            kpe_d = torch.max(
                torch.abs(kpe_fused[p, s].float() - kpe_sep[p, s].float())
            ).item()
            assert ckv_d == 0.0, f"{err}ckv differs at loc {i}! diff={ckv_d}"
            assert kpe_d == 0.0, f"{err}kpe differs at loc {i}! diff={kpe_d}"

        written = torch.zeros(num_pages, page_size, dtype=torch.bool, device=q_nope.device)
        for i in range(batch_size):
            written[kv_indices[i], positions[i]] = True
        unwritten = ~written
        assert torch.equal(ckv_fused[unwritten], ckv_fused_snap[unwritten]), (
            f"Spurious writes detected in {label}ckv_cache!"
        )
        assert torch.equal(kpe_fused[unwritten], kpe_fused_snap[unwritten]), (
            f"Spurious writes detected in {label}kpe_cache!"
        )

        return q_out_fused, ckv_fused, kpe_fused, unwritten
    
    @pytest.mark.parametrize("batch_size,num_heads,page_size", [
        (1, 128, 64),    # single-token decode, TP=1
        (1, 16, 64),     # single-token decode, TP=8
        (3, 128, 64),    # odd small batch, TP=1
        (8, 16, 64),     # medium batch, TP=8
        (4, 32, 32),     # small batch, TP=4, page_size=32
        (32, 64, 64),    # medium batch, TP=2
        (64, 128, 32),   # larger batch, TP=1, page_size=32
        (128, 128, 64),  # max cuda_graph_max_bs, TP=1
        (128, 16, 32),   # max batch, TP=8, page_size=32
    ])
    def test_fused_correctness(self, device, dtype, cos_sin_cache,
                               batch_size, num_heads, page_size):
        """Fused kernel must be bit-identical to separated path, write only
        targeted slots, and work with non-contiguous combined buffers.

        For each parametrized config this single test checks:
        1. q_out and KV cache match between fused and separated paths (bit-exact)
        2. No spurious writes: untouched slots retain a sentinel value
        3. Combined (non-contiguous) buffer produces the same result as separate buffers
        """
        attn_dtype = torch.float8_e4m3fn
        sentinel_val = torch.tensor(0.5, dtype=torch.float32).to(attn_dtype)
        num_pages = batch_size + 1  # extra page to detect out-of-bounds writes
        out_cache_loc = self._generate_cache_locs(batch_size, page_size, device)
        pos_ids = torch.arange(batch_size, device=device, dtype=torch.int32)

        torch.manual_seed(batch_size * 1000 + num_heads * 10 + page_size)
        q_nope = torch.randn(batch_size, num_heads, self.KV_LORA_RANK, device=device, dtype=dtype)
        q_rope = torch.randn(batch_size, num_heads, self.QK_ROPE_HEAD_DIM, device=device, dtype=dtype)
        k_nope = torch.randn(batch_size, self.KV_LORA_RANK, device=device, dtype=dtype)
        k_rope = torch.randn(batch_size, self.QK_ROPE_HEAD_DIM, device=device, dtype=dtype)

        q_out_fused, ckv_fused, kpe_fused, unwritten = self._assert_fused_separated_parity(
            q_nope=q_nope,
            q_rope=q_rope,
            k_nope=k_nope,
            k_rope=k_rope,
            cos_sin_cache=cos_sin_cache,
            pos_ids=pos_ids,
            out_cache_loc=out_cache_loc,
            page_size=page_size,
            num_pages=num_pages,
            sentinel_val=sentinel_val,
        )

        kv_indices = out_cache_loc // page_size
        positions = out_cache_loc % page_size

        # --- Check 3: combined non-contiguous buffer matches separate buffers ---
        kv_cache_dim = self.KV_LORA_RANK + self.QK_ROPE_HEAD_DIM
        combined = torch.full(
            (num_pages, page_size, kv_cache_dim),
            sentinel_val.item(), device=device, dtype=attn_dtype,
        )
        combined_snap = combined.clone()
        ckv_comb = combined[:, :, :self.KV_LORA_RANK]
        kpe_comb = combined[:, :, self.KV_LORA_RANK:]
        assert not ckv_comb.is_contiguous(), "ckv_cache should NOT be contiguous"
        assert not kpe_comb.is_contiguous(), "kpe_cache should NOT be contiguous"

        q_out_comb = self._run_fused_path(
            q_nope, q_rope, k_nope, k_rope,
            cos_sin_cache, pos_ids, out_cache_loc,
            ckv_comb, kpe_comb, page_size,
        )
        comb_q_diff = torch.max(torch.abs(q_out_comb.float() - q_out_fused.float())).item()
        assert comb_q_diff == 0.0, f"Combined buffer q_out mismatch! diff={comb_q_diff}"
        for i in range(batch_size):
            p, s = kv_indices[i].item(), positions[i].item()
            cd = torch.max(torch.abs(ckv_comb[p, s].float() - ckv_fused[p, s].float())).item()
            kd = torch.max(torch.abs(kpe_comb[p, s].float() - kpe_fused[p, s].float())).item()
            assert cd == 0.0, f"Combined ckv differs at loc {i}! diff={cd}"
            assert kd == 0.0, f"Combined kpe differs at loc {i}! diff={kd}"
        assert torch.equal(ckv_comb[unwritten], combined_snap[:, :, :self.KV_LORA_RANK][unwritten]), \
            "Spurious writes detected in combined ckv view!"
        assert torch.equal(kpe_comb[unwritten], combined_snap[:, :, self.KV_LORA_RANK:][unwritten]), \
            "Spurious writes detected in combined kpe view!"

    def test_kv_indptr_contract_mismatch(self, device, dtype, cos_sin_cache):
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
        num_heads = 16
        page_size = 64
        num_pages = 12
        attn_dtype = torch.float8_e4m3fn

        torch.manual_seed(7777)
        q_nope = torch.randn(batch_size, num_heads, self.KV_LORA_RANK, device=device, dtype=dtype)
        q_rope = torch.randn(batch_size, num_heads, self.QK_ROPE_HEAD_DIM, device=device, dtype=dtype)
        k_nope = torch.randn(batch_size, self.KV_LORA_RANK, device=device, dtype=dtype)
        k_rope = torch.randn(batch_size, self.QK_ROPE_HEAD_DIM, device=device, dtype=dtype)
        pos_ids = torch.arange(batch_size, device=device, dtype=torch.int32)

        pages = torch.arange(1, batch_size + 1, device=device, dtype=torch.int32)
        positions = torch.arange(batch_size, device=device, dtype=torch.int32)
        out_cache_loc = pages * page_size + positions
        kv_indices = (out_cache_loc // page_size).to(torch.int32)
        positions_meta = (out_cache_loc % page_size).to(torch.int32)
        batch_indices = torch.arange(batch_size, dtype=torch.int32, device=device)

        good_kv_indptr = torch.arange(batch_size + 1, dtype=torch.int32, device=device)
        bad_kv_indptr = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)

        ckv_good, kpe_good = self._create_paged_kv_cache(num_pages, page_size, device)
        q_out_good = torch.empty(
            batch_size, num_heads,
            self.KV_LORA_RANK + self.QK_ROPE_HEAD_DIM,
            device=device, dtype=attn_dtype,
        )
        flashinfer.rope.rope_quantize_fp8_append_paged_kv_cache(
            q_rope=q_rope, k_rope=k_rope, q_nope=q_nope, k_nope=k_nope,
            v=None, cos_sin_cache=cos_sin_cache, pos_ids=pos_ids,
            paged_kv_cache=(ckv_good, kpe_good),
            kv_indices=kv_indices, kv_indptr=good_kv_indptr,
            batch_indices=batch_indices, positions=positions_meta,
            is_neox=False, quantize_dtype=attn_dtype,
            quant_scale_q=1.0, quant_scale_kv=1.0,
            page_size=page_size, kv_layout="NHD",
            q_rope_out=q_out_good[..., self.KV_LORA_RANK:],
            q_nope_out=q_out_good[..., :self.KV_LORA_RANK],
        )

        ckv_bad, kpe_bad = self._create_paged_kv_cache(num_pages, page_size, device)
        q_out_bad = torch.empty_like(q_out_good)
        flashinfer.rope.rope_quantize_fp8_append_paged_kv_cache(
            q_rope=q_rope, k_rope=k_rope, q_nope=q_nope, k_nope=k_nope,
            v=None, cos_sin_cache=cos_sin_cache, pos_ids=pos_ids,
            paged_kv_cache=(ckv_bad, kpe_bad),
            kv_indices=kv_indices, kv_indptr=bad_kv_indptr,
            batch_indices=batch_indices, positions=positions_meta,
            is_neox=False, quantize_dtype=attn_dtype,
            quant_scale_q=1.0, quant_scale_kv=1.0,
            page_size=page_size, kv_layout="NHD",
            q_rope_out=q_out_bad[..., self.KV_LORA_RANK:],
            q_nope_out=q_out_bad[..., :self.KV_LORA_RANK],
        )
        torch.cuda.synchronize()

        q_diff = torch.max(torch.abs(q_out_good.float() - q_out_bad.float())).item()
        assert q_diff == 0.0, f"q_out unexpectedly changed by kv_indptr mismatch! max_diff={q_diff}"

        ckv_mismatch = torch.max(torch.abs(ckv_good.float() - ckv_bad.float())).item()
        kpe_mismatch = torch.max(torch.abs(kpe_good.float() - kpe_bad.float())).item()
        assert ckv_mismatch > 0.0 or kpe_mismatch > 0.0, (
            "Expected KV cache corruption with bad kv_indptr semantics, but saw no difference."
        )

        ckv_fixed, kpe_fixed = self._create_paged_kv_cache(num_pages, page_size, device)
        q_out_fixed = torch.empty_like(q_out_good)
        flashinfer.rope.rope_quantize_fp8_append_paged_kv_cache(
            q_rope=q_rope, k_rope=k_rope, q_nope=q_nope, k_nope=k_nope,
            v=None, cos_sin_cache=cos_sin_cache, pos_ids=pos_ids,
            paged_kv_cache=(ckv_fixed, kpe_fixed),
            kv_indices=kv_indices, kv_indptr=good_kv_indptr,
            batch_indices=batch_indices, positions=positions_meta,
            is_neox=False, quantize_dtype=attn_dtype,
            quant_scale_q=1.0, quant_scale_kv=1.0,
            page_size=page_size, kv_layout="NHD",
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

    def test_high_rope_positions_deepseek(self, device, dtype, cos_sin_cache):
        """DeepSeek-focused spot check for high RoPE positions near cache limit."""
        batch_size = 8
        num_heads = 128  # DeepSeek TP=1 head count
        page_size = 64
        num_pages = batch_size + 1
        sentinel_val = torch.tensor(0.5, dtype=torch.float32).to(torch.float8_e4m3fn)

        out_cache_loc = self._generate_cache_locs(batch_size, page_size, device)
        # Stress high-position RoPE indexing near the cache upper bound.
        pos_ids = torch.tensor(
            [0, 1, 127, 255, 1024, 4095, 8190, 8191],
            device=device,
            dtype=torch.int32,
        )

        torch.manual_seed(131072)
        q_nope = torch.randn(
            batch_size, num_heads, self.KV_LORA_RANK, device=device, dtype=dtype
        )
        q_rope = torch.randn(
            batch_size, num_heads, self.QK_ROPE_HEAD_DIM, device=device, dtype=dtype
        )
        k_nope = torch.randn(batch_size, self.KV_LORA_RANK, device=device, dtype=dtype)
        k_rope = torch.randn(batch_size, self.QK_ROPE_HEAD_DIM, device=device, dtype=dtype)

        self._assert_fused_separated_parity(
            q_nope=q_nope,
            q_rope=q_rope,
            k_nope=k_nope,
            k_rope=k_rope,
            cos_sin_cache=cos_sin_cache,
            pos_ids=pos_ids,
            out_cache_loc=out_cache_loc,
            page_size=page_size,
            num_pages=num_pages,
            sentinel_val=sentinel_val,
            label="high-pos ",
        )

    def test_permuted_out_cache_loc_order(self, device, dtype, cos_sin_cache):
        """DeepSeek-focused spot check for non-monotonic write order metadata."""
        batch_size = 8
        num_heads = 16  # DeepSeek TP=8 head count
        page_size = 64
        num_pages = batch_size + 1
        sentinel_val = torch.tensor(0.5, dtype=torch.float32).to(torch.float8_e4m3fn)

        out_cache_loc_base = self._generate_cache_locs(batch_size, page_size, device)
        perm = torch.tensor([5, 0, 7, 2, 6, 1, 4, 3], device=device, dtype=torch.int64)
        out_cache_loc = out_cache_loc_base[perm]
        pos_ids_base = torch.arange(batch_size, device=device, dtype=torch.int32)
        pos_ids = pos_ids_base[perm]
        assert not torch.equal(out_cache_loc, out_cache_loc_base), (
            "Expected a non-monotonic out_cache_loc permutation."
        )

        torch.manual_seed(131073)
        q_nope = torch.randn(
            batch_size, num_heads, self.KV_LORA_RANK, device=device, dtype=dtype
        )
        q_rope = torch.randn(
            batch_size, num_heads, self.QK_ROPE_HEAD_DIM, device=device, dtype=dtype
        )
        k_nope = torch.randn(batch_size, self.KV_LORA_RANK, device=device, dtype=dtype)
        k_rope = torch.randn(batch_size, self.QK_ROPE_HEAD_DIM, device=device, dtype=dtype)
        q_nope = q_nope[perm]
        q_rope = q_rope[perm]
        k_nope = k_nope[perm]
        k_rope = k_rope[perm]

        self._assert_fused_separated_parity(
            q_nope=q_nope,
            q_rope=q_rope,
            k_nope=k_nope,
            k_rope=k_rope,
            cos_sin_cache=cos_sin_cache,
            pos_ids=pos_ids,
            out_cache_loc=out_cache_loc,
            page_size=page_size,
            num_pages=num_pages,
            sentinel_val=sentinel_val,
            label="permuted-order ",
        )

    def test_boundary_position_ids_deepseek(self, device, dtype, cos_sin_cache):
        """DeepSeek-focused spot check around common RoPE position boundaries."""
        batch_size = 8
        num_heads = 128  # DeepSeek TP=1 head count
        page_size = 64
        num_pages = batch_size + 1
        sentinel_val = torch.tensor(0.5, dtype=torch.float32).to(torch.float8_e4m3fn)

        out_cache_loc = self._generate_cache_locs(batch_size, page_size, device)
        pos_ids = torch.tensor(
            [1022, 1023, 1024, 2046, 2047, 2048, 4094, 4095],
            device=device,
            dtype=torch.int32,
        )

        torch.manual_seed(131074)
        q_nope = torch.randn(
            batch_size, num_heads, self.KV_LORA_RANK, device=device, dtype=dtype
        )
        q_rope = torch.randn(
            batch_size, num_heads, self.QK_ROPE_HEAD_DIM, device=device, dtype=dtype
        )
        k_nope = torch.randn(batch_size, self.KV_LORA_RANK, device=device, dtype=dtype)
        k_rope = torch.randn(batch_size, self.QK_ROPE_HEAD_DIM, device=device, dtype=dtype)

        self._assert_fused_separated_parity(
            q_nope=q_nope,
            q_rope=q_rope,
            k_nope=k_nope,
            k_rope=k_rope,
            cos_sin_cache=cos_sin_cache,
            pos_ids=pos_ids,
            out_cache_loc=out_cache_loc,
            page_size=page_size,
            num_pages=num_pages,
            sentinel_val=sentinel_val,
            label="boundary-pos ",
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
