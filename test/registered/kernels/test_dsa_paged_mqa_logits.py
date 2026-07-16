# SPDX-License-Identifier: Apache-2.0
"""Tests for `dsa_paged_mqa_logits_backend="torch"` (+ its Triton fast path).

`fp8_paged_mqa_logits_torch_dsa` (`sglang.srt.layers.attention.dsa.
torch_paged_mqa_logits`) is a pure-torch, CUDA-graph-safe fallback for the DSA
indexer's paged-MQA-logits kernel, for CUDA archs that DeepGEMM/CuTe DSL don't
cover. It optionally dispatches to a fused Triton kernel
(`fp8_paged_mqa_logits_triton_dsa`, `dsa/triton_paged_mqa_logits.py`) that is
bit-exact with it but far cheaper under CUDA graph capture (see that module's
docstring). Modelled on the DSv4 SM120 precedent test,
`test/registered/kernels/test_sm120_paged_mqa_logits.py` (added by PR #24692,
"no SM120 hardware required" -- any CUDA GPU exercises this) -- same DSA cache
layout constants, same style of numeric-equivalence and CUDA-graph tests.

Coverage:
- torch fn vs. an independent, deliberately un-vectorized ("loopy") Python
  reference: no vectorized code is shared with either implementation under
  test, so a bug shared between the torch and Triton kernels (both written
  against the same design) would not also be present here.
- Triton vs. torch fn: bit-exact-tolerance equivalence, including identical
  -inf masks, across several (batch, seq_len, page-table-width) shapes.
- Both KV-cache dtype views (raw uint8 / float8_e4m3fn): guards the historic
  class of bug where a kernel treats raw uint8 bytes as integers instead of
  reinterpreting them as FP8 (see the SM120 precedent's docstring).
- Variable per-batch seq_lens, including a page_table with -1 ("no page here")
  entries in a request's invalid trailing pages -- exercises the defensive
  clamp in the torch fn and the Triton kernel's per-block early exit; neither
  must dereference the sentinel or let it affect a valid position.
- Per-token (flattened-batch) shapes matching the target-verify /
  draft-extend-v2 calling convention: seqlens already expanded to one row per
  token, page_table already repeat_interleaved to one row per token. A
  regression test also asserts that repeating the page_table a SECOND time
  (the double-expansion bug the live dgxarley bring-up hit at the MTP warmup)
  is caught by the batch-size shape assert rather than silently miscomputing.
- CUDA graph capture + replay, including a seq_lens change to the captured
  static buffer between capture and replay, for both the torch-only and the
  Triton dispatch path.
- The `num_heads < 16` fallback guard (Triton's `tl.dot` minimum): verifies
  the Triton kernel is not invoked below that threshold and the torch path
  alone still produces the correct result.
- Shape-assertion guards (wrong head_dim, unsupported clean_logits=True).

Pure-PyTorch + Triton on any CUDA GPU -- no specific SM arch required.
"""

from __future__ import annotations

import unittest
from unittest import mock

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.attention.dsa.torch_paged_mqa_logits import (
    FP8_DTYPE,
    fp8_paged_mqa_logits_torch_dsa,
)
from sglang.srt.layers.attention.dsa.triton_paged_mqa_logits import (
    fp8_paged_mqa_logits_triton_dsa,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=25, stage="base-b", runner_config="1-gpu-small")


# DSA indexer paged-KV cache layout (fixed by the DSA token-to-KV pool):
#   page_size = 64 tokens; head_dim = 128 FP8 values/token; 1 fp32 scale/token
#   per-page memory: [page_size*head_dim FP8 bytes][page_size*4 scale bytes]
#                   = [8192][256] = 8448 bytes
# The (block_size, 1, head_dim+4) shape callers pass this tensor around in is
# a fiction for shape-compatibility only: the real per-page layout is ALL
# value bytes for the whole page, THEN all scale bytes -- NOT interleaved
# per-token. `fp8_paged_mqa_logits_torch_dsa` and `..._triton_dsa` both
# reinterpret it that way (flatten -> split at the value/scale boundary ->
# reshape); the loopy reference below does the same independently.
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
      - [0 : PAGE_SIZE*HEAD_DIM)          : random FP8 bit patterns (values)
      - [PAGE_SIZE*HEAD_DIM : PAGE_BYTES) : random positive fp32 scales (bytes)
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    raw = torch.empty(num_pages, PAGE_BYTES, dtype=torch.uint8, device=device)

    # Bias away from bit patterns that map to NaN/Inf in float8_e4m3fn
    # (sign|exp4|mantissa3; exp=0xF mantissa!=0 -> NaN). [0, 0x6F] keeps |x| < 256.
    val_bytes = torch.randint(
        0, 0x70, (num_pages, PAGE_SIZE * HEAD_DIM), generator=g, dtype=torch.uint8
    ).to(device)
    raw[:, : PAGE_SIZE * HEAD_DIM] = val_bytes

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
    invalid_page_id: int | None = None,
):
    """Build (q_fp8, kvcache, weight, seq_lens, page_table, max_seq_len).

    When `invalid_page_id` is given (e.g. -1), every page slot beyond a
    request's own page count (but within the batch-wide `max_pages`) is set
    to that id, instead of a real page -- exercising the sentinel-handling
    path for batches with heterogeneous seq_lens.
    """
    assert len(seq_lens) == batch_size
    max_seq_len = max(seq_lens)
    max_pages = (max_seq_len + PAGE_SIZE - 1) // PAGE_SIZE
    num_pages_total = batch_size * max_pages + 1

    kvcache = _build_kvcache(
        num_pages_total, dtype_view=kv_dtype_view, device=device, seed=seed
    )

    g = torch.Generator(device="cpu").manual_seed(seed + 1)
    q_bf16 = torch.randn(
        (batch_size, 1, num_heads, HEAD_DIM), generator=g, dtype=torch.float32
    ).to(device)
    q_bf16 = q_bf16.clamp_(-2.0, 2.0)
    q_fp8 = q_bf16.to(FP8_DTYPE)

    weight = (
        torch.rand((batch_size, num_heads), generator=g, dtype=torch.float32).to(device)
        * 0.5
    )

    seq_lens_t = torch.tensor(seq_lens, dtype=torch.int32, device=device)

    page_table = torch.full(
        (batch_size, max_pages),
        invalid_page_id if invalid_page_id is not None else 0,
        dtype=torch.int32,
        device=device,
    )
    for i in range(batch_size):
        own_pages = (seq_lens[i] + PAGE_SIZE - 1) // PAGE_SIZE
        page_table[i, :own_pages] = torch.arange(
            1 + i * max_pages,
            1 + i * max_pages + own_pages,
            dtype=torch.int32,
            device=device,
        )
        if invalid_page_id is None:
            # No sentinel requested: fill the remaining (unused) columns with
            # real (if unused) page ids too, matching the original precedent
            # test's behaviour of never emitting a sentinel.
            remaining = max_pages - own_pages
            if remaining > 0:
                page_table[i, own_pages:] = torch.arange(
                    1 + i * max_pages + own_pages,
                    1 + i * max_pages + own_pages + remaining,
                    dtype=torch.int32,
                    device=device,
                )

    return q_fp8, kvcache, weight, seq_lens_t, page_table, max_seq_len


def _loopy_reference(
    q_fp8: torch.Tensor,
    kvcache_fp8: torch.Tensor,
    weight: torch.Tensor,
    seq_lens: torch.Tensor,
    page_table: torch.Tensor,
    max_seq_len: int,
) -> torch.Tensor:
    """Independent, deliberately un-vectorized per-(request, position, head)
    reference. Shares no computation path with the vectorized torch kernel
    (batched gather + bmm) or the Triton kernel (one program per KV block) --
    only the documented byte layout, which both must also honor.

    Runs entirely on CPU in float64 for numerical clarity; only meant for
    small test shapes.
    """
    batch_size, _, num_heads, head_dim = q_fp8.shape
    block_size = kvcache_fp8.shape[1]
    assert head_dim == HEAD_DIM and block_size == PAGE_SIZE

    kv_cpu = kvcache_fp8.reshape(kvcache_fp8.shape[0], -1).cpu()
    if kv_cpu.dtype == torch.uint8:
        value_bytes = kv_cpu[:, : block_size * head_dim].view(FP8_DTYPE)
    else:
        value_bytes = kv_cpu[:, : block_size * head_dim]
    scale_bytes = kv_cpu[:, block_size * head_dim :].contiguous().view(torch.float32)

    values = value_bytes.to(torch.float64).reshape(
        -1, block_size, head_dim
    )  # (pages, 64, 128)
    scales = scale_bytes.to(torch.float64).reshape(-1, block_size)  # (pages, 64)

    q = q_fp8[:, 0].to(torch.float64).cpu()  # (B, H, D)
    w = weight.to(torch.float64).cpu()  # (B, H)
    sl = (seq_lens.squeeze(-1) if seq_lens.dim() > 1 else seq_lens).cpu()
    pt = page_table.cpu()

    out = torch.full((batch_size, max_seq_len), float("-inf"), dtype=torch.float64)
    for b in range(batch_size):
        seq_len_b = min(int(sl[b].item()), max_seq_len)
        for pos in range(seq_len_b):
            page_idx = pos // block_size
            offset = pos % block_size
            page_id = int(pt[b, page_idx].item())
            kv_vec = values[page_id, offset]  # (D,)
            scale = scales[page_id, offset].item()
            acc = 0.0
            for h in range(num_heads):
                dot = torch.dot(kv_vec, q[b, h]).item()
                dot = max(dot, 0.0)
                acc += dot * w[b, h].item()
            out[b, pos] = acc * scale
    return out.to(q_fp8.device)


def _compare(
    ref: torch.Tensor,
    out: torch.Tensor,
    seq_lens: torch.Tensor,
    *,
    atol: float = 2e-2,
    rtol: float = 2e-2,
):
    """Valid positions ([:seq_len)) must match; invalid positions must be -inf."""
    batch_size, max_seq_len = ref.shape
    seq_lens = seq_lens.squeeze(-1) if seq_lens.dim() > 1 else seq_lens
    for i in range(batch_size):
        sl = int(seq_lens[i].item())
        torch.testing.assert_close(
            ref[i, :sl].to(torch.float32),
            out[i, :sl].to(torch.float32),
            atol=atol,
            rtol=rtol,
            equal_nan=False,
        )
    positions = torch.arange(max_seq_len, device=out.device)
    invalid = positions.unsqueeze(0) >= seq_lens.unsqueeze(1)
    assert torch.all(
        torch.isinf(out[invalid]) & (out[invalid] < 0)
    ), "output must fill invalid positions with -inf"


class TestDSAPagedMqaLogitsTorch(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA required")
        cls.device = torch.device("cuda")

    def _torch_only(self, *args, **kwargs):
        with envs.SGLANG_DSA_INDEXER_TRITON.override(False):
            return fp8_paged_mqa_logits_torch_dsa(*args, **kwargs)

    def _triton_dispatch(self, *args, **kwargs):
        with envs.SGLANG_DSA_INDEXER_TRITON.override(True):
            return fp8_paged_mqa_logits_torch_dsa(*args, **kwargs)

    # -- torch fn vs. independent loopy reference --------------------------

    def test_torch_vs_loopy_bs1(self):
        q, kv, w, sl, pt, msl = _build_inputs(
            1, [96], kv_dtype_view=FP8_DTYPE, num_heads=8, device=self.device
        )
        ref = _loopy_reference(q, kv, w, sl, pt, msl)
        out = self._torch_only(q, kv, w, sl, pt, None, msl, clean_logits=False)
        _compare(ref, out, sl)

    def test_torch_vs_loopy_bs4_variable(self):
        q, kv, w, sl, pt, msl = _build_inputs(
            4,
            [40, 96, 130, 192],
            kv_dtype_view=FP8_DTYPE,
            num_heads=8,
            device=self.device,
        )
        ref = _loopy_reference(q, kv, w, sl, pt, msl)
        out = self._torch_only(q, kv, w, sl, pt, None, msl, clean_logits=False)
        _compare(ref, out, sl)

    def test_torch_vs_loopy_partial_last_page(self):
        # 65 = 1 full page + 1 token -- exercises the partial trailing page.
        q, kv, w, sl, pt, msl = _build_inputs(
            2, [65, 129], kv_dtype_view=FP8_DTYPE, num_heads=8, device=self.device
        )
        ref = _loopy_reference(q, kv, w, sl, pt, msl)
        out = self._torch_only(q, kv, w, sl, pt, None, msl, clean_logits=False)
        _compare(ref, out, sl)

    def test_uint8_view_matches_fp8_view(self):
        """Regression guard: identical raw bytes, different dtype VIEW of the
        KV cache, must produce identical logits (historic garbled-output bug
        class: treating raw uint8 bytes as integers instead of FP8)."""
        q, kv_fp8, w, sl, pt, msl = _build_inputs(
            2,
            [40, 96],
            kv_dtype_view=FP8_DTYPE,
            num_heads=8,
            device=self.device,
            seed=7,
        )
        _, kv_u8, _, _, _, _ = _build_inputs(
            2,
            [40, 96],
            kv_dtype_view=torch.uint8,
            num_heads=8,
            device=self.device,
            seed=7,
        )
        out_fp8 = self._torch_only(q, kv_fp8, w, sl, pt, None, msl, clean_logits=False)
        out_u8 = self._torch_only(q, kv_u8, w, sl, pt, None, msl, clean_logits=False)
        torch.testing.assert_close(out_fp8, out_u8, atol=0, rtol=0, equal_nan=True)

    # -- -1 sentinel: heterogeneous seq_lens with invalid trailing pages ---

    def test_negative_sentinel_page_table_masked_and_no_crash(self):
        """A page_table with -1 ("no page here") entries in a shorter
        request's unused trailing columns must not crash the gather (the
        torch fn clamps defensively) and those columns' output must still be
        -inf regardless of what garbage got gathered at the clamped id."""
        q, kv, w, sl, pt, msl = _build_inputs(
            2,
            [64, 192],
            kv_dtype_view=FP8_DTYPE,
            num_heads=8,
            device=self.device,
            invalid_page_id=-1,
        )
        # Request 0 (seq_len=64, 1 page) has 2 unused page slots (max_pages=3)
        # filled with -1 by invalid_page_id; confirm the fixture did that.
        assert int(pt[0, 1].item()) == -1 and int(pt[0, 2].item()) == -1
        out = self._torch_only(q, kv, w, sl, pt, None, msl, clean_logits=False)
        ref = _loopy_reference(q, kv, w, sl, pt.clamp(min=0), msl)
        _compare(ref, out, sl)

    def test_negative_sentinel_triton_no_crash(self):
        """Same fixture through the Triton path: the -1 page id is only ever
        stored (not the version tested here -- see kernel docstring) when the
        WHOLE block is past seq_len, so the early exit means it is never
        dereferenced at all."""
        q, kv, w, sl, pt, msl = _build_inputs(
            2,
            [64, 192],
            kv_dtype_view=FP8_DTYPE,
            num_heads=16,
            device=self.device,
            invalid_page_id=-1,
        )
        out_triton = fp8_paged_mqa_logits_triton_dsa(q, kv, w, sl, pt, None, msl)
        out_torch = self._torch_only(q, kv, w, sl, pt, None, msl, clean_logits=False)
        _compare(out_torch, out_triton, sl, atol=1e-3, rtol=1e-3)

    # -- Triton vs. torch: bit-exact-tolerance equivalence ------------------

    def _assert_triton_matches_torch(self, batch_size, seq_lens, num_heads=32, seed=0):
        q, kv, w, sl, pt, msl = _build_inputs(
            batch_size,
            seq_lens,
            kv_dtype_view=FP8_DTYPE,
            num_heads=num_heads,
            device=self.device,
            seed=seed,
        )
        out_torch = self._torch_only(q, kv, w, sl, pt, None, msl, clean_logits=False)
        out_triton = self._triton_dispatch(
            q, kv, w, sl, pt, None, msl, clean_logits=False
        )
        # -inf masks must match exactly (this is a comparison, not floating
        # math, so there is no tolerance to allow here).
        torch.testing.assert_close(
            torch.isinf(out_torch) & (out_torch < 0),
            torch.isinf(out_triton) & (out_triton < 0),
            atol=0,
            rtol=0,
        )
        finite = ~(torch.isinf(out_torch) & (out_torch < 0))
        torch.testing.assert_close(
            out_torch[finite], out_triton[finite], atol=1e-3, rtol=1e-3, equal_nan=False
        )

    def test_triton_bitexact_vs_torch_bs1_short(self):
        self._assert_triton_matches_torch(1, [300], num_heads=32)

    def test_triton_bitexact_vs_torch_bs4_variable(self):
        self._assert_triton_matches_torch(4, [128, 512, 900, 2048], num_heads=32)

    def test_triton_bitexact_vs_torch_bs32_uniform(self):
        self._assert_triton_matches_torch(32, [300] * 32, num_heads=32)

    def test_triton_bitexact_vs_torch_dsv32_headcount(self):
        # DSv3.2-family head count (64) -- distinct code path in
        # _dsa_indexer_logits_kernel's per-head accumulation width.
        self._assert_triton_matches_torch(2, [300, 900], num_heads=64)

    # -- Per-token (flattened) shapes: target-verify / draft-extend-v2 -----

    def test_per_token_expanded_shapes_next_n(self):
        """Mirrors the dsa_indexer.py torch-backend dispatch calling
        convention for target_verify/draft_extend_v2 (next_n >= 2): q/weights
        already sliced to q_offset tokens, seqlens already expanded to one
        row PER TOKEN, page_table already repeat_interleaved to one row per
        token by dsa_backend.py's init_forward_metadata (`page_table =
        torch.repeat_interleave(page_table, repeats=next_n, dim=0)` before
        the metadata is built). The kernel itself must need no next_n
        awareness -- batch dim = flattened tokens."""
        num_requests = 3
        next_n = 4
        # >= 16 so the direct Triton call below is legal (tl.dot minimum);
        # realistic values are 32 (GLM DSA) / 64 (DeepSeek-V3.2). The < 16
        # fallback path has its own dedicated test.
        num_heads = 32
        ctx_lens = [64, 130, 300]  # per-request context length before verify
        device = self.device

        q_req, kv, w_req, _, pt_req, _ = _build_inputs(
            num_requests,
            [c + next_n for c in ctx_lens],
            kv_dtype_view=FP8_DTYPE,
            num_heads=num_heads,
            device=device,
        )
        # Expand to per-token rows (B*next_n), exactly like
        # seqlens_expand_triton + repeat_interleave(page_table, next_n).
        q_tok = q_req.repeat_interleave(next_n, dim=0)  # (B*next_n, 1, H, D)
        w_tok = w_req.repeat_interleave(next_n, dim=0)  # (B*next_n, H)
        pt_tok = pt_req.repeat_interleave(next_n, dim=0)  # (B*next_n, max_pages)
        # kv_len = c + next_n (full post-verify context); qo_len = next_n;
        # dsa_backend.py builds this as arange(kv_len - qo_len + 1, kv_len + 1)
        # = arange(c + 1, c + next_n + 1) -- context grows by one per draft
        # position within a request's verify window.
        seqlens_tok = torch.cat(
            [
                torch.arange(c + 1, c + next_n + 1, dtype=torch.int32, device=device)
                for c in ctx_lens
            ]
        )
        max_seq_len = max(c + next_n for c in ctx_lens)

        out = self._torch_only(
            q_tok, kv, w_tok, seqlens_tok, pt_tok, None, max_seq_len, clean_logits=False
        )
        ref = _loopy_reference(q_tok, kv, w_tok, seqlens_tok, pt_tok, max_seq_len)
        _compare(ref, out, seqlens_tok)

        out_triton = fp8_paged_mqa_logits_triton_dsa(
            q_tok, kv, w_tok, seqlens_tok, pt_tok, None, max_seq_len
        )
        _compare(ref, out_triton, seqlens_tok)

    def test_double_expansion_is_caught_by_shape_assert(self):
        """Regression test for the double-expansion bug hit live at the MTP
        warmup: repeating an already per-token page_table a second time must
        NOT silently miscompute -- it must trip the batch-size shape assert,
        because q_fp8's row count (still B*next_n) no longer matches the
        (B*next_n*next_n)-row page_table."""
        next_n = 4
        q, kv, w, sl, pt, msl = _build_inputs(
            2, [64, 130], kv_dtype_view=FP8_DTYPE, num_heads=8, device=self.device
        )
        q_tok = q.repeat_interleave(next_n, dim=0)
        w_tok = w.repeat_interleave(next_n, dim=0)
        sl_tok = sl.repeat_interleave(next_n, dim=0)
        pt_correct = pt.repeat_interleave(next_n, dim=0)  # correct: matches q_tok rows
        pt_double = pt_correct.repeat_interleave(next_n, dim=0)  # bug: double-expanded

        # Sanity: the correctly-expanded call must NOT raise.
        self._torch_only(
            q_tok, kv, w_tok, sl_tok, pt_correct, None, msl, clean_logits=False
        )

        with self.assertRaises(AssertionError):
            self._torch_only(
                q_tok, kv, w_tok, sl_tok, pt_double, None, msl, clean_logits=False
            )

    # -- num_heads < 16: Triton fallback guard ------------------------------

    def test_num_heads_below_16_skips_triton(self):
        q, kv, w, sl, pt, msl = _build_inputs(
            2, [64, 128], kv_dtype_view=FP8_DTYPE, num_heads=8, device=self.device
        )
        with mock.patch(
            "sglang.srt.layers.attention.dsa.torch_paged_mqa_logits."
            "fp8_paged_mqa_logits_triton_dsa",
            side_effect=AssertionError("triton must not be called for num_heads < 16"),
        ):
            out = self._triton_dispatch(q, kv, w, sl, pt, None, msl, clean_logits=False)
        ref = _loopy_reference(q, kv, w, sl, pt, msl)
        _compare(ref, out, sl)

    def test_num_heads_above_16_uses_triton(self):
        q, kv, w, sl, pt, msl = _build_inputs(
            2, [64, 128], kv_dtype_view=FP8_DTYPE, num_heads=16, device=self.device
        )
        with mock.patch(
            "sglang.srt.layers.attention.dsa.torch_paged_mqa_logits."
            "fp8_paged_mqa_logits_triton_dsa",
            wraps=fp8_paged_mqa_logits_triton_dsa,
        ) as spy:
            self._triton_dispatch(q, kv, w, sl, pt, None, msl, clean_logits=False)
        spy.assert_called_once()

    # -- CUDA graph capture + replay ----------------------------------------

    def _run_cuda_graph_case(self, use_triton: bool):
        batch_size = 2
        seq_lens = [128, 192]
        q, kv, w, sl, pt, msl = _build_inputs(
            batch_size,
            seq_lens,
            kv_dtype_view=FP8_DTYPE,
            num_heads=16,
            device=self.device,
        )
        call = self._triton_dispatch if use_triton else self._torch_only

        for _ in range(2):
            _ = call(q, kv, w, sl, pt, None, msl, clean_logits=False)
        torch.cuda.synchronize()

        static_holder = {}
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = call(q, kv, w, sl, pt, None, msl, clean_logits=False)
            static_holder["out"] = out

        ref = call(q, kv, w, sl, pt, None, msl, clean_logits=False)
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(static_holder["out"], ref, atol=1e-5, rtol=1e-5)

        # Replay with a DIFFERENT seq_lens, in-place edit of the captured buffer.
        sl_new = torch.tensor([64, 256], dtype=torch.int32, device=self.device)
        sl.copy_(sl_new)
        graph.replay()
        torch.cuda.synchronize()
        ref_new = call(q, kv, w, sl, pt, None, msl, clean_logits=False)
        torch.testing.assert_close(static_holder["out"], ref_new, atol=1e-5, rtol=1e-5)

    def test_cuda_graph_capture_and_replay_torch(self):
        self._run_cuda_graph_case(use_triton=False)

    def test_cuda_graph_capture_and_replay_triton(self):
        self._run_cuda_graph_case(use_triton=True)

    # -- Shape-assertion guards -----------------------------------------

    def test_shape_assertions(self):
        q, kv, w, sl, pt, msl = _build_inputs(
            1, [64], kv_dtype_view=FP8_DTYPE, num_heads=8, device=self.device
        )
        bad_q = q[..., :64]  # head_dim != 128
        with self.assertRaises(AssertionError):
            self._torch_only(bad_q, kv, w, sl, pt, None, msl, clean_logits=False)
        with self.assertRaises(AssertionError):
            self._torch_only(q, kv, w, sl, pt, None, msl, clean_logits=True)


if __name__ == "__main__":
    import sys

    sys.exit(unittest.main())
