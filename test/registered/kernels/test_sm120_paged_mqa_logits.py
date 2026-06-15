"""SM120 fp8_paged_mqa_logits_torch_sm120 vectorized PyTorch fallback tests.

Validates that the vectorized SM120-specific implementation matches the loopy
reference (`fp8_paged_mqa_logits_torch`) and is CUDA-graph compatible.

Coverage:
- Numeric equivalence vs reference at small shapes
- Both KV-cache dtype views (uint8 raw / float8_e4m3fn) — guards against the
  historic garbled-output bug where Triton kernels treated uint8 bytes as raw
  integers instead of FP8 (dsv4_sm120_progress.md §4)
- Variable per-batch seq_lens with -inf masking semantics
- CUDA graph capture + replay equivalence (no .item() / data-dependent shapes)
- Shape-assertion guards

Pure-PyTorch implementations on any CUDA GPU — no SM120 hardware required.
"""

from __future__ import annotations

import unittest

import torch

from sglang.srt.layers.attention.dsv4.indexer import (
    FP8_DTYPE,
    fp8_paged_mqa_logits_torch,
    fp8_paged_mqa_logits_torch_sm120,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=9, stage="base-b", runner_config="1-gpu-small")


# DSv4 indexer cache layout (fixed by deepseek_v4_memory_pool.DeepSeekV4IndexerPool):
#   page_size = 64 tokens
#   head_dim = 128 (FP8 values per token)
#   quant_block_size = 128 -> num_scales_per_token = 1 (fp32 scale)
#   per-page memory: [page_size*head_dim FP8 bytes][page_size*4 scale bytes]
#                  = [8192][256] = 8448 bytes
# The (block_size, 1, head_dim+4) shape is a fiction for downstream consumers.
PAGE_SIZE = 64
HEAD_DIM = 128
SCALE_BYTES_PER_TOKEN = 4
HEAD_DIM_WITH_SF = HEAD_DIM + SCALE_BYTES_PER_TOKEN  # 132
PAGE_BYTES = PAGE_SIZE * HEAD_DIM + PAGE_SIZE * SCALE_BYTES_PER_TOKEN  # 8448


def _build_kvcache(
    num_pages: int,
    *,
    dtype_view: torch.dtype,
    device: torch.device,
    seed: int = 0,
) -> torch.Tensor:
    """Construct a paged KV cache matching the production layout.

    Returns a tensor shaped (num_pages, PAGE_SIZE, 1, HEAD_DIM_WITH_SF) whose
    underlying memory is the same regardless of `dtype_view`:
      - [0 : PAGE_SIZE*HEAD_DIM)        : random FP8 bit patterns (values)
      - [PAGE_SIZE*HEAD_DIM : PAGE_BYTES) : random positive fp32 scales (bytes)
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    raw = torch.empty(num_pages, PAGE_BYTES, dtype=torch.uint8, device=device)

    # Random FP8 byte pattern for the value section. Bias away from extreme
    # bit patterns that map to NaN/Inf in float8_e4m3fn (sign|exp4|mantissa3;
    # exp=0xF mantissa!=0 -> NaN). Restricting to [0, 0x6F] keeps |x| < 256.
    val_bytes = torch.randint(
        0, 0x70, (num_pages, PAGE_SIZE * HEAD_DIM), generator=g, dtype=torch.uint8
    ).to(device)
    raw[:, : PAGE_SIZE * HEAD_DIM] = val_bytes

    # Positive fp32 scales in [0.05, 0.55]. Byte-view into the trailing region.
    scales = (
        torch.rand((num_pages, PAGE_SIZE), generator=g, dtype=torch.float32).to(device)
        * 0.5
        + 0.05
    )
    raw[:, PAGE_SIZE * HEAD_DIM :] = scales.contiguous().view(torch.uint8)

    kv = raw.view(num_pages, PAGE_SIZE, 1, HEAD_DIM_WITH_SF)
    return kv if dtype_view == torch.uint8 else kv.view(dtype=dtype_view)


def _build_inputs(
    batch_size: int,
    seq_lens: list[int],
    *,
    kv_dtype_view: torch.dtype,
    num_heads: int = 32,
    device: torch.device = torch.device("cuda"),
    seed: int = 0,
):
    assert len(seq_lens) == batch_size
    max_seq_len = max(seq_lens)
    max_pages = (max_seq_len + PAGE_SIZE - 1) // PAGE_SIZE
    # One global page pool, batch picks its own page ids.
    num_pages_total = batch_size * max_pages + 1

    kvcache = _build_kvcache(
        num_pages_total, dtype_view=kv_dtype_view, device=device, seed=seed
    )

    g = torch.Generator(device="cpu").manual_seed(seed + 1)
    # Construct query as random bf16->fp8 to keep values inside fp8 range.
    q_bf16 = torch.randn(
        (batch_size, 1, num_heads, HEAD_DIM), generator=g, dtype=torch.float32
    ).to(device)
    q_bf16 = q_bf16.clamp_(-2.0, 2.0)
    q_fp8 = q_bf16.to(FP8_DTYPE)
    if kv_dtype_view == torch.uint8:
        # Query dtype isn't toggled — only kvcache. q always fp8.
        pass

    weight = (
        torch.rand((batch_size, num_heads), generator=g, dtype=torch.float32).to(device)
        * 0.5
    )

    seq_lens_t = torch.tensor(seq_lens, dtype=torch.int32, device=device)

    # Each batch occupies its own slice of pages, randomized for realism.
    page_table = torch.zeros((batch_size, max_pages), dtype=torch.int32, device=device)
    for i in range(batch_size):
        page_table[i] = torch.arange(
            1 + i * max_pages, 1 + (i + 1) * max_pages, dtype=torch.int32, device=device
        )

    return q_fp8, kvcache, weight, seq_lens_t, page_table, max_seq_len


def _compare(
    ref: torch.Tensor,
    sm120: torch.Tensor,
    seq_lens: torch.Tensor,
    *,
    atol: float = 1e-3,
    rtol: float = 1e-3,
):
    """Compare reference (uninitialized beyond seq_len) vs SM120 (-inf beyond)."""
    # Valid positions must match
    batch_size, max_seq_len = ref.shape
    for i in range(batch_size):
        sl = int(seq_lens[i].item())
        torch.testing.assert_close(
            ref[i, :sl], sm120[i, :sl], atol=atol, rtol=rtol, equal_nan=False
        )
    # Invalid positions in SM120 output must be -inf
    positions = torch.arange(max_seq_len, device=sm120.device)
    invalid = positions.unsqueeze(0) >= seq_lens.unsqueeze(1)
    assert torch.all(
        torch.isinf(sm120[invalid]) & (sm120[invalid] < 0)
    ), "SM120 output must fill invalid positions with -inf"


class TestSM120PagedMqaLogitsTorch(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA required")
        cls.device = torch.device("cuda")

    def _run_one(
        self,
        batch_size: int,
        seq_lens: list[int],
        kv_dtype_view: torch.dtype,
        num_heads: int = 32,
    ):
        q, kv, w, sl, pt, msl = _build_inputs(
            batch_size,
            seq_lens,
            kv_dtype_view=kv_dtype_view,
            num_heads=num_heads,
            device=self.device,
        )
        # Reference (loopy)
        ref = fp8_paged_mqa_logits_torch(
            q,
            kv,
            w,
            sl,
            pt,
            deep_gemm_metadata=None,
            max_seq_len=msl,
            clean_logits=False,
        )
        # SM120 vectorized
        sm120 = fp8_paged_mqa_logits_torch_sm120(
            q,
            kv,
            w,
            sl,
            pt,
            deep_gemm_metadata=None,
            max_seq_len=msl,
            clean_logits=False,
        )
        _compare(ref, sm120, sl)

    def test_equiv_fp8_view_bs1(self):
        self._run_one(1, [128], FP8_DTYPE)

    def test_equiv_fp8_view_bs4_uniform(self):
        self._run_one(4, [128, 128, 128, 128], FP8_DTYPE)

    def test_equiv_fp8_view_bs4_variable(self):
        self._run_one(4, [40, 96, 200, 256], FP8_DTYPE)

    def test_equiv_uint8_view_bs1(self):
        """Regression guard: KV cache viewed as raw uint8 (no FP8 dtype hint).

        Historic bug (progress doc §4): some kernels treated uint8 bytes as raw
        integers, producing garbled attention output. The vectorized impl must
        produce the same numbers as the loopy reference regardless of the
        caller's KV view dtype.
        """
        self._run_one(1, [128], torch.uint8)

    def test_equiv_uint8_view_bs4_variable(self):
        self._run_one(4, [40, 96, 200, 256], torch.uint8)

    def test_seq_lens_zero_remainder(self):
        """seq_len not aligned to page_size — last partial page must mask correctly."""
        self._run_one(2, [65, 129], FP8_DTYPE)  # 65 = 1 full page + 1 token

    def test_seq_lens_full_pages(self):
        self._run_one(2, [64, 192], FP8_DTYPE)

    def test_seq_lens_2d_input_accepted(self):
        """SM120 impl squeezes seq_lens if dim>1 (matches indexer.py call site)."""
        q, kv, w, sl, pt, msl = _build_inputs(
            2, [64, 128], kv_dtype_view=FP8_DTYPE, device=self.device
        )
        sl_2d = sl.unsqueeze(-1)  # (B, 1)
        ref = fp8_paged_mqa_logits_torch(
            q, kv, w, sl, pt, None, max_seq_len=msl, clean_logits=False
        )
        sm120 = fp8_paged_mqa_logits_torch_sm120(
            q, kv, w, sl_2d, pt, None, max_seq_len=msl, clean_logits=False
        )
        _compare(ref, sm120, sl)

    def test_cuda_graph_capture_and_replay(self):
        """No .item() / data-dependent control flow — must be CUDA-graph safe."""
        batch_size = 2
        seq_lens = [128, 192]
        q, kv, w, sl, pt, msl = _build_inputs(
            batch_size, seq_lens, kv_dtype_view=FP8_DTYPE, device=self.device
        )

        # Warmup outside graph
        for _ in range(2):
            _ = fp8_paged_mqa_logits_torch_sm120(
                q, kv, w, sl, pt, None, max_seq_len=msl, clean_logits=False
            )
        torch.cuda.synchronize()

        # Pre-allocated output (graph replay reuses this buffer)
        static_logits_holder = {}

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = fp8_paged_mqa_logits_torch_sm120(
                q, kv, w, sl, pt, None, max_seq_len=msl, clean_logits=False
            )
            static_logits_holder["out"] = out

        # Eager reference using the same inputs
        ref = fp8_paged_mqa_logits_torch_sm120(
            q, kv, w, sl, pt, None, max_seq_len=msl, clean_logits=False
        )

        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(
            static_logits_holder["out"], ref, atol=1e-5, rtol=1e-5
        )

        # Replay with a different seq_lens (in-place edit of the captured tensor)
        sl_new = torch.tensor([64, 256], dtype=torch.int32, device=self.device)
        sl.copy_(sl_new)
        graph.replay()
        torch.cuda.synchronize()
        ref_new = fp8_paged_mqa_logits_torch_sm120(
            q, kv, w, sl, pt, None, max_seq_len=msl, clean_logits=False
        )
        torch.testing.assert_close(
            static_logits_holder["out"], ref_new, atol=1e-5, rtol=1e-5
        )

    def test_shape_assertions(self):
        """Wrong head_dim or block_size must raise."""
        q, kv, w, sl, pt, msl = _build_inputs(
            1, [64], kv_dtype_view=FP8_DTYPE, device=self.device
        )
        # head_dim != 128
        bad_q = q[..., :64]
        with self.assertRaises(AssertionError):
            fp8_paged_mqa_logits_torch_sm120(
                bad_q, kv, w, sl, pt, None, max_seq_len=msl, clean_logits=False
            )
        # clean_logits=True not supported
        with self.assertRaises(AssertionError):
            fp8_paged_mqa_logits_torch_sm120(
                q, kv, w, sl, pt, None, max_seq_len=msl, clean_logits=True
            )


if __name__ == "__main__":
    import sys

    sys.exit(unittest.main())
