"""Self-contained correctness test for the embedding KV-cache-skip fast path
integrated into TorchNativeAttnBackend.forward_extend
(python/sglang/srt/layers/attention/torch_native_backend.py).

Does NOT require a running SGLang server or the model weights -- constructs
synthetic q/k/v tensors + a minimal fake KV pool / forward_batch / layer that
satisfy exactly the attributes forward_extend reads, then compares, on the SAME
real forward_extend method, its two internal paths:

  (a) ORIGINAL path: writes q/k/v into a fake paged KV pool via set_kv_buffer,
      then gathers them back via req_to_token indices before calling SDPA
      (the production write-then-gather path). Forced by leaving the opt-in
      switch off (_skip_embed_kv_cache = False), against

  (b) SKIP path: skips the pool write/gather and calls SDPA directly on the
      local k/v (_forward_extend_kv_skip). Enabled by setting all guard flags.

Positive cases assert the two are BIT-IDENTICAL (single-seq, multi-seq, GQA,
bf16). Negative cases assert that when a guard fails (extend_prefix_len != 0,
sliding window enabled, or the switch off) the skip backend falls back to the
original path and is likewise bit-identical -- i.e. zero regression.

Usage:
  python3 -m unittest test.registered.attention.test_embedding_kv_cache_skip
  # or, from this directory:
  python3 test_embedding_kv_cache_skip.py
"""

import unittest

import torch

from sglang.srt.layers.attention.torch_native_backend import TorchNativeAttnBackend
from sglang.srt.layers.radix_attention import AttentionType


class FakeLayer:
    def __init__(
        self,
        tp_q_head_num,
        tp_k_head_num,
        tp_v_head_num,
        qk_head_dim,
        v_head_dim,
        layer_id=0,
        scaling=None,
        is_cross_attention=False,
        attn_type=AttentionType.DECODER,
        sliding_window_size=-1,
    ):
        self.tp_q_head_num = tp_q_head_num
        self.tp_k_head_num = tp_k_head_num
        self.tp_v_head_num = tp_v_head_num
        self.qk_head_dim = qk_head_dim
        self.v_head_dim = v_head_dim
        self.layer_id = layer_id
        self.scaling = scaling
        self.is_cross_attention = is_cross_attention
        self.attn_type = attn_type
        self.sliding_window_size = sliding_window_size


class FakePool:
    """Minimal stand-in for MHATokenToKVPool: plain bf16/fp32 storage, no
    quantization (store_dtype == dtype), matching the embedding baseline.
    set_kv_buffer accepts the fork's KVWriteLoc (extracts .loc) or a bare loc."""

    def __init__(self, max_total_tokens, num_kv_heads, qk_head_dim, v_head_dim, dtype):
        self.dtype = dtype
        self.store_dtype = dtype
        self.k_buffer = torch.zeros(
            max_total_tokens, num_kv_heads, qk_head_dim, dtype=dtype
        )
        self.v_buffer = torch.zeros(
            max_total_tokens, num_kv_heads, v_head_dim, dtype=dtype
        )

    def set_kv_buffer(self, layer, loc_info, cache_k, cache_v):
        loc = getattr(loc_info, "loc", loc_info)  # KVWriteLoc.loc, else bare tensor
        self.k_buffer[loc] = cache_k.to(self.dtype)
        self.v_buffer[loc] = cache_v.to(self.dtype)

    def get_key_buffer(self, layer_id):
        return self.k_buffer

    def get_value_buffer(self, layer_id):
        return self.v_buffer


class FakeReqToTokenPool:
    def __init__(self, req_to_token):
        self.req_to_token = req_to_token


class FakeForwardBatch:
    def __init__(
        self,
        extend_prefix_lens,
        extend_seq_lens,
        seq_lens,
        orig_seq_lens,
        req_pool_indices,
        out_cache_loc,
    ):
        self.extend_prefix_lens = extend_prefix_lens
        self.extend_seq_lens = extend_seq_lens
        self.seq_lens = seq_lens
        self.orig_seq_lens = orig_seq_lens
        self.req_pool_indices = req_pool_indices
        self.out_cache_loc = out_cache_loc
        self.encoder_out_cache_loc = None
        self.encoder_lens = None


def make_backend(skip_enabled, embed_no_decode=True, chunk_impossible=True):
    """Build a TorchNativeAttnBackend without running __init__ (which needs a
    real ModelRunner). Guard flags are set explicitly; pool / req_to_token
    are attached per-batch by the caller."""
    b = TorchNativeAttnBackend.__new__(TorchNativeAttnBackend)
    b.forward_metadata = None
    b.device = "cpu"
    b.swa_out_cache_loc = None
    b.use_sliding_window_kv_pool = False
    b._skip_embed_kv_cache = skip_enabled
    b._embed_no_decode = embed_no_decode
    b._kvskip_chunk_impossible = chunk_impossible
    return b


def build_batch(
    seq_lens_list, num_kv_heads, qk_head_dim, v_head_dim, dtype, prefix_lens_list=None
):
    """Contiguous KV-pool allocation: request i's newly-extended tokens occupy
    pool rows [offset_i : offset_i + seq_len_i). If prefix_lens_list is given
    (non-zero for some request), extra rows are reserved *before* the extend
    rows to stand in for an already-cached prefix; zero-filled since only the
    fallback code path (not attention numerics) is under test in that case.
    Returns (pool, req_to_token, forward_batch, total_extend_tokens)."""
    num_seqs = len(seq_lens_list)
    if prefix_lens_list is None:
        prefix_lens_list = [0] * num_seqs
    total_extend_tokens = sum(seq_lens_list)
    total_prefix_tokens = sum(prefix_lens_list)
    max_seq_len = max(p + s for p, s in zip(prefix_lens_list, seq_lens_list))

    pool = FakePool(
        total_prefix_tokens + total_extend_tokens,
        num_kv_heads,
        qk_head_dim,
        v_head_dim,
        dtype,
    )

    req_to_token = torch.zeros(num_seqs, max_seq_len, dtype=torch.long)
    out_cache_loc = torch.zeros(total_extend_tokens, dtype=torch.long)
    prefix_offset = 0
    extend_offset = total_prefix_tokens  # extend rows come after all prefix rows
    q_offset = 0
    for i, (pl, sl) in enumerate(zip(prefix_lens_list, seq_lens_list)):
        if pl > 0:
            prefix_loc = torch.arange(
                prefix_offset, prefix_offset + pl, dtype=torch.long
            )
            req_to_token[i, :pl] = prefix_loc
            prefix_offset += pl
        extend_loc = torch.arange(extend_offset, extend_offset + sl, dtype=torch.long)
        req_to_token[i, pl : pl + sl] = extend_loc
        out_cache_loc[q_offset : q_offset + sl] = extend_loc
        extend_offset += sl
        q_offset += sl

    extend_prefix_lens = torch.tensor(prefix_lens_list, dtype=torch.long)
    extend_seq_lens = torch.tensor(seq_lens_list, dtype=torch.long)
    seq_lens = torch.tensor(
        [p + s for p, s in zip(prefix_lens_list, seq_lens_list)], dtype=torch.long
    )
    orig_seq_lens = seq_lens.clone()
    req_pool_indices = torch.arange(num_seqs, dtype=torch.long)

    fb = FakeForwardBatch(
        extend_prefix_lens,
        extend_seq_lens,
        seq_lens,
        orig_seq_lens,
        req_pool_indices,
        out_cache_loc,
    )
    return pool, req_to_token, fb, total_extend_tokens


def _run_one(backend, seq_lens_list, q, k, v, layer, dtype, prefix_lens_list):
    pool, req_to_token, fb, _ = build_batch(
        seq_lens_list,
        layer.tp_k_head_num,
        layer.qk_head_dim,
        layer.v_head_dim,
        dtype,
        prefix_lens_list=prefix_lens_list,
    )
    backend.token_to_kv_pool = pool
    backend.req_to_token_pool = FakeReqToTokenPool(req_to_token)
    return backend.forward_extend(
        q.clone(), k.clone(), v.clone(), layer, fb, save_kv_cache=True
    )


def compare_paths(
    seq_lens_list,
    tp_q_head_num=16,
    tp_k_head_num=8,
    tp_v_head_num=8,
    qk_head_dim=128,
    v_head_dim=128,
    dtype=torch.float32,
    prefix_lens_list=None,
    sliding_window_size=-1,
    skip_enabled=True,
):
    """Runs the same q/k/v/layer through (a) a backend forced onto the ORIGINAL
    path (switch off) and (b) a backend with the skip switch on (subject to
    guards). Returns (bit_identical, max_abs_diff)."""
    torch.manual_seed(0)
    total_tokens = sum(seq_lens_list)
    # q stays flat [T, Hq*D] (forward_extend reshapes it); k/v arrive already 3D
    # [T, Hk, D] (radix_attention reshapes before the backend call).
    q = torch.randn(total_tokens, tp_q_head_num * qk_head_dim, dtype=dtype)
    k = torch.randn(total_tokens, tp_k_head_num, qk_head_dim, dtype=dtype)
    v = torch.randn(total_tokens, tp_v_head_num, v_head_dim, dtype=dtype)

    layer = FakeLayer(
        tp_q_head_num,
        tp_k_head_num,
        tp_v_head_num,
        qk_head_dim,
        v_head_dim,
        scaling=1.0 / (qk_head_dim**0.5),
        sliding_window_size=sliding_window_size,
    )

    backend_orig = make_backend(skip_enabled=False)  # always original path
    out_orig = _run_one(
        backend_orig, seq_lens_list, q, k, v, layer, dtype, prefix_lens_list
    )

    backend_skip = make_backend(skip_enabled=skip_enabled)
    out_skip = _run_one(
        backend_skip, seq_lens_list, q, k, v, layer, dtype, prefix_lens_list
    )

    max_abs_diff = (out_orig - out_skip).abs().max().item()
    return torch.equal(out_orig, out_skip), max_abs_diff


class TestEmbeddingKVCacheSkip(unittest.TestCase):
    # ---- Positive: skip path is bit-exact vs. the original gather path -------
    def test_single_seq_fp32_gqa(self):
        ident, diff = compare_paths([37])
        self.assertTrue(ident, f"not bit-identical, max_abs_diff={diff:.3e}")

    def test_multi_seq_fp32_gqa(self):
        ident, diff = compare_paths([30, 300, 5])
        self.assertTrue(ident, f"not bit-identical, max_abs_diff={diff:.3e}")

    def test_multi_seq_bf16_gqa(self):
        ident, diff = compare_paths([30, 300, 5], dtype=torch.bfloat16)
        self.assertTrue(ident, f"not bit-identical, max_abs_diff={diff:.3e}")

    def test_multi_seq_fp32_mha(self):
        ident, diff = compare_paths([12, 40], tp_q_head_num=8)
        self.assertTrue(ident, f"not bit-identical, max_abs_diff={diff:.3e}")

    # ---- Negative: guard failure => fall back to original path (zero regr) ---
    def test_fallback_nonzero_prefix(self):
        # guard (5) fails for one seq -> both backends take the original path.
        ident, diff = compare_paths([30, 20], prefix_lens_list=[5, 0])
        self.assertTrue(ident, f"fallback not bit-identical, max_abs_diff={diff:.3e}")

    def test_fallback_sliding_window(self):
        # guard (8) fails (SWA layer) -> both backends take the original path.
        ident, diff = compare_paths([30, 300, 5], sliding_window_size=64)
        self.assertTrue(ident, f"fallback not bit-identical, max_abs_diff={diff:.3e}")

    def test_fallback_switch_off(self):
        # guard (1) fails (switch off) -> skip backend == original backend.
        ident, diff = compare_paths([30, 300, 5], skip_enabled=False)
        self.assertTrue(ident, f"switch-off not bit-identical, max_abs_diff={diff:.3e}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
