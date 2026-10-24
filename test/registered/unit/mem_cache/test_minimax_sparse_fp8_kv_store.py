"""Route A: fused bf16->fp8 main-KV store for MiniMax-M3 sparse layers.

`MiniMaxSparseKVPool.set_fused_kv_index_buffer` can write the main K/V through
the Triton `reshape_and_cache_flash` kernel -- the same kernel the dense layers
already use via `AiterAttnBackend._use_fused_fp8_kv_write` -- folding the
bf16->fp8 cast into the paged scatter instead of running it as a separate
`div_` + `.to()` + scatter chain.

Two groups here:

* gating (CPU) -- every precondition in `_can_fuse_fp8_main_kv_store`. These are
  the cases where taking the fast path would be *silently* wrong rather than an
  error, so each one gets its own test.
* equivalence (GPU) -- the fused writer against the `set_kv_buffer` fallback it
  replaces, driven through the real `set_fused_kv_index_buffer` entry point.
"""

import unittest

import torch

from sglang.srt.environ import envs
from sglang.srt.mem_cache.memory_pool import MiniMaxSparseKVPool
from sglang.test.ci.ci_register import (
    register_amd_ci,
    register_cpu_ci,
    register_cuda_ci,
)

# The gating tests below need no accelerator, so register CPU too -- otherwise
# they only ever run on the GPU runners and a gating regression goes unnoticed
# on CPU-only CI.
register_cpu_ci(est_time=9, suite="base-a-test-cpu")
register_cuda_ci(est_time=9, stage="base-b", runner_config="1-gpu-small")
register_amd_ci(est_time=9, stage="stage-b", runner_config="1-gpu-small-amd")

# Layer layout shared by every pool built here: 4 local layers, two dense, one
# sparse with an index V, one sparse K-only (the "block selector" layers).
START_LAYER = 0
DENSE_LAYER_IDS = [0, 2]
SPARSE_KV_LAYER_ID = 1
SPARSE_K_ONLY_LAYER_ID = 3
SPARSE_LAYER_IDS = [SPARSE_KV_LAYER_ID, SPARSE_K_ONLY_LAYER_ID]
END_LAYER = 4

HEAD_NUM = 2
HEAD_DIM = 16
IDX_HEAD_DIM = 16


class _StubLayer:
    """The attribute surface the pool touches: layer_id plus the scale tensors.

    `set_fused_kv_index_buffer` is handed the `*_scale_float` mirrors by the
    backend, but the Triton kernel does `tl.load(k_scale_ptr)` and needs the
    device tensors -- `BaseKVCacheMethod.process_weights_after_loading` writes
    both from the same value, which is what makes the substitution sound.
    """

    def __init__(self, layer_id: int, scale: float = None, device: str = "cpu"):
        self.layer_id = layer_id
        if scale is None:
            self.k_scale = None
            self.v_scale = None
            self.k_scale_float = None
            self.v_scale_float = None
        else:
            self.k_scale = torch.tensor(scale, dtype=torch.float32, device=device)
            self.v_scale = torch.tensor(scale, dtype=torch.float32, device=device)
            self.k_scale_float = scale
            self.v_scale_float = scale


def _make_pool(
    *,
    dtype: torch.dtype,
    device: str = "cpu",
    size: int = 64,
    page_size: int = 16,
    index_dtype: torch.dtype = torch.bfloat16,
) -> MiniMaxSparseKVPool:
    # The *other* fused path (raw-byte kv+index store) is disqualified for an
    # fp8 main pool by its own dtype checks; pin it off so these tests do not
    # depend on that and always exercise the fallback branch.
    with envs.SGLANG_OPT_USE_MINIMAX_FUSED_KV_INDEX_STORE.override(False):
        return MiniMaxSparseKVPool(
            size=size,
            page_size=page_size,
            dtype=dtype,
            head_num=HEAD_NUM,
            head_dim=HEAD_DIM,
            idx_head_dim=IDX_HEAD_DIM,
            dense_layer_ids=DENSE_LAYER_IDS,
            sparse_layer_ids=SPARSE_LAYER_IDS,
            disable_value_sparse_layer_ids=[SPARSE_K_ONLY_LAYER_ID],
            device=device,
            index_dtype=index_dtype,
            start_layer=START_LAYER,
            end_layer=END_LAYER,
        )


class TestSparseFp8KvStoreGating(unittest.TestCase):
    """`_can_fuse_fp8_main_kv_store` -- CPU-only, no kernel launched."""

    def _src(self, T=8, dtype=torch.bfloat16, head_num=HEAD_NUM, head_dim=HEAD_DIM):
        return torch.randn(T, head_num, head_dim, dtype=dtype)

    def _accepts(self, pool, layer, k, v, k_scale=None, v_scale=None) -> bool:
        return pool._can_fuse_fp8_main_kv_store(layer, k, v, k_scale, v_scale)

    def test_accepts_fp8_pool_with_model_dtype_source(self):
        pool = _make_pool(dtype=torch.float8_e4m3fn)
        layer = _StubLayer(SPARSE_KV_LAYER_ID)
        self.assertTrue(self._accepts(pool, layer, self._src(), self._src()))

    def test_model_dtype_pool_rejected(self):
        # No cast to fuse: set_kv_buffer's scatter is already the whole operation.
        pool = _make_pool(dtype=torch.bfloat16)
        layer = _StubLayer(SPARSE_KV_LAYER_ID)
        self.assertFalse(self._accepts(pool, layer, self._src(), self._src()))

    def test_already_fp8_source_rejected(self):
        pool = _make_pool(dtype=torch.float8_e4m3fn)
        layer = _StubLayer(SPARSE_KV_LAYER_ID)
        src = self._src().to(torch.float8_e4m3fn)
        self.assertFalse(self._accepts(pool, layer, src, src))

    def test_one_sided_scale_rejected(self):
        # The launcher passes `key` as the stand-in for a missing scale pointer
        # while USE_SCALE is a single constexpr covering both K and V -- a
        # one-sided scale would make the kernel load the key tensor as v_scale.
        pool = _make_pool(dtype=torch.float8_e4m3fn)
        layer = _StubLayer(SPARSE_KV_LAYER_ID, scale=0.5)
        k, v = self._src(), self._src()
        self.assertFalse(self._accepts(pool, layer, k, v, k_scale=0.5, v_scale=None))
        self.assertFalse(self._accepts(pool, layer, k, v, k_scale=None, v_scale=0.5))
        self.assertTrue(self._accepts(pool, layer, k, v, k_scale=0.5, v_scale=0.5))

    def test_missing_scale_tensor_rejected(self):
        # Float scales given but the layer never got its tensor mirrors: the
        # kernel has nothing to tl.load, so fall back rather than launch.
        pool = _make_pool(dtype=torch.float8_e4m3fn)
        layer = _StubLayer(SPARSE_KV_LAYER_ID, scale=None)
        k, v = self._src(), self._src()
        self.assertFalse(self._accepts(pool, layer, k, v, k_scale=0.5, v_scale=0.5))

    def test_non_token_major_source_rejected(self):
        # The kernel honours an arbitrary stride(0) but assumes heads/dims are
        # packed behind it; a head-major view would scatter to wrong offsets.
        pool = _make_pool(dtype=torch.float8_e4m3fn)
        layer = _StubLayer(SPARSE_KV_LAYER_ID)
        good = self._src()
        bad = torch.randn(HEAD_NUM, 8, HEAD_DIM, dtype=torch.bfloat16).transpose(0, 1)
        self.assertEqual(bad.shape[1:], (HEAD_NUM, HEAD_DIM))
        self.assertFalse(bad.is_contiguous())
        self.assertFalse(self._accepts(pool, layer, bad, good))
        self.assertFalse(self._accepts(pool, layer, good, bad))

    def test_row_slice_of_a_larger_source_accepted(self):
        # Converse of the above: an outer-dimension slice keeps each token
        # contiguous, which is exactly what stride(0) is for.
        pool = _make_pool(dtype=torch.float8_e4m3fn)
        layer = _StubLayer(SPARSE_KV_LAYER_ID)
        big = torch.randn(16, HEAD_NUM, HEAD_DIM, dtype=torch.bfloat16)
        sliced = big[::2]
        self.assertFalse(sliced.is_contiguous())
        self.assertTrue(self._accepts(pool, layer, sliced, sliced))

    def test_head_shape_mismatch_rejected(self):
        pool = _make_pool(dtype=torch.float8_e4m3fn)
        layer = _StubLayer(SPARSE_KV_LAYER_ID)
        good = self._src()
        self.assertFalse(
            self._accepts(pool, layer, self._src(head_num=HEAD_NUM + 1), good)
        )
        self.assertFalse(
            self._accepts(pool, layer, good, self._src(head_dim=HEAD_DIM * 2))
        )

    def test_unaligned_pool_size_rejected(self):
        # Buffer is [size + page_size, heads, dim]; the 4-D page view needs the
        # row count to divide by page_size.
        pool = _make_pool(dtype=torch.float8_e4m3fn, size=20, page_size=16)
        layer = _StubLayer(SPARSE_KV_LAYER_ID)
        self.assertFalse(self._accepts(pool, layer, self._src(), self._src()))

    def test_hnd_layout_rejected(self):
        pool = _make_pool(dtype=torch.float8_e4m3fn)
        layer = _StubLayer(SPARSE_KV_LAYER_ID)
        pool.main_pool.use_hnd = True
        pool.main_pool.kv_cache_layout = "hnd"
        self.assertFalse(self._accepts(pool, layer, self._src(), self._src()))

    def test_quantized_pool_rejected(self):
        # `is_quantized_kv_cache` is a read-only property derived from
        # `quant_method`, so set the attribute it reads
        # rather than the property -- assigning the property raises
        # AttributeError and the gate is never reached. Any object that is not
        # an UnquantizedKVCacheMethod makes it True; the assert below pins that
        # so the case cannot start passing for the wrong reason.
        pool = _make_pool(dtype=torch.float8_e4m3fn)
        layer = _StubLayer(SPARSE_KV_LAYER_ID)
        pool.main_pool.quant_method = object()
        self.assertTrue(pool.main_pool.is_quantized_kv_cache)
        self.assertFalse(self._accepts(pool, layer, self._src(), self._src()))


@unittest.skipUnless(torch.cuda.is_available(), "fused fp8 KV store needs a GPU")
class TestSparseFp8KvStoreEquivalence(unittest.TestCase):
    """Fused writer vs the `set_kv_buffer` fallback, through the real entry point."""

    SIZE = 64
    PAGE_SIZE = 16
    T = 9

    @classmethod
    def setUpClass(cls):
        from sglang.kernels.ops.quantization.fp8_kernel import fp8_dtype

        cls.fp8_dtype = fp8_dtype
        cls.dev = "cuda"

    def _inputs(self, seed: int):
        torch.manual_seed(seed)
        dev = self.dev
        # Keep magnitudes well inside fp8 e4m3 range (max 448 / 224): this test
        # is about scatter placement and cast agreement, not saturation, and the
        # two paths saturate through different code (tl.store vs torch .to()).
        k = torch.randn(self.T, HEAD_NUM, HEAD_DIM, dtype=torch.bfloat16, device=dev)
        v = torch.randn(self.T, HEAD_NUM, HEAD_DIM, dtype=torch.bfloat16, device=dev)
        idx_k = torch.randn(self.T, 1, IDX_HEAD_DIM, dtype=torch.bfloat16, device=dev)
        idx_v = torch.randn(self.T, 1, IDX_HEAD_DIM, dtype=torch.bfloat16, device=dev)
        # Distinct, non-monotonic rows so a page/offset mix-up cannot pass.
        loc = torch.randperm(self.SIZE, device=dev)[: self.T].to(torch.int64)
        return k, v, idx_k, idx_v, loc

    def _store(self, *, fused, layer_id, scale, seed):
        """Run one store into a fresh pool; return (pool, raw k/v store buffers)."""
        pool = _make_pool(
            dtype=self.fp8_dtype,
            device=self.dev,
            size=self.SIZE,
            page_size=self.PAGE_SIZE,
        )
        if not fused:
            # The precondition is the only thing that routes a store to the
            # fused kernel, so refusing it exercises exactly the set_kv_buffer
            # fallback these tests compare against.
            pool._can_fuse_fp8_main_kv_store = lambda *a, **kw: False
        layer = _StubLayer(layer_id, scale=scale, device=self.dev)
        k, v, idx_k, idx_v, loc = self._inputs(seed)
        disable_value = layer_id == SPARSE_K_ONLY_LAYER_ID
        pool.set_fused_kv_index_buffer(
            layer,
            loc,
            k,
            v,
            idx_k,
            None if disable_value else idx_v,
            layer.k_scale_float,
            layer.v_scale_float,
            None,
            None,
        )
        local = layer_id - START_LAYER
        return (
            pool,
            k,
            v,
            loc,
            (
                pool.main_pool.k_buffer[local].clone(),
                pool.main_pool.v_buffer[local].clone(),
            ),
        )

    def _assert_took_fused_path(self, layer_id, scale):
        pool = _make_pool(
            dtype=self.fp8_dtype,
            device=self.dev,
            size=self.SIZE,
            page_size=self.PAGE_SIZE,
        )
        layer = _StubLayer(layer_id, scale=scale, device=self.dev)
        k, v, _, _, _ = self._inputs(0)
        self.assertTrue(
            pool._can_fuse_fp8_main_kv_store(
                layer, k, v, layer.k_scale_float, layer.v_scale_float
            ),
            "precondition rejected the case this test claims to cover",
        )

    def test_unit_scale_is_bitwise_identical_to_fallback(self):
        # No scale on either side, so both paths are a plain bf16->fp8 cast:
        # the numerics must not move at all.
        self._assert_took_fused_path(SPARSE_KV_LAYER_ID, None)
        _, _, _, _, fused = self._store(
            fused=True, layer_id=SPARSE_KV_LAYER_ID, scale=None, seed=7
        )
        _, _, _, _, ref = self._store(
            fused=False, layer_id=SPARSE_KV_LAYER_ID, scale=None, seed=7
        )
        self.assertTrue(torch.equal(fused[0], ref[0]), "K store differs")
        self.assertTrue(torch.equal(fused[1], ref[1]), "V store differs")

    def test_scaled_store_agrees_with_fallback_within_one_fp8_step(self):
        # The fallback divides in bf16 (`cache_k.div_(k_scale)`) then casts; the
        # kernel divides in fp32 (the scale is an fp32 tensor) then casts. That
        # double rounding is the only difference, so agreement is to one fp8
        # step, not bitwise. e4m3 has 3 mantissa bits -> relative step 2^-3.
        scale = 0.5
        self._assert_took_fused_path(SPARSE_KV_LAYER_ID, scale)
        _, _, _, _, fused = self._store(
            fused=True, layer_id=SPARSE_KV_LAYER_ID, scale=scale, seed=11
        )
        _, _, _, _, ref = self._store(
            fused=False, layer_id=SPARSE_KV_LAYER_ID, scale=scale, seed=11
        )
        for name, a, b in (("K", fused[0], ref[0]), ("V", fused[1], ref[1])):
            af = a.view(self.fp8_dtype).float()
            bf = b.view(self.fp8_dtype).float()
            torch.testing.assert_close(
                af, bf, rtol=2**-3, atol=2**-9, msg=f"{name} store diverged"
            )

    def test_scaled_store_matches_fp32_reference_bitwise(self):
        # Pins down what the fused path actually computes, independent of the
        # fallback: divide in fp32, then one cast.
        scale = 0.5
        _, k, v, loc, (kb, vb) = self._store(
            fused=True, layer_id=SPARSE_KV_LAYER_ID, scale=scale, seed=13
        )
        want_k = (k.float() / scale).to(self.fp8_dtype)
        want_v = (v.float() / scale).to(self.fp8_dtype)
        self.assertTrue(torch.equal(kb.view(self.fp8_dtype)[loc], want_k))
        self.assertTrue(torch.equal(vb.view(self.fp8_dtype)[loc], want_v))

    def test_fused_path_does_not_mutate_the_source(self):
        # set_kv_buffer's `div_` is in place and mutates the caller's activation
        # tensor; the fused path must not, or a later reader of k/v would see
        # pre-divided values on one arm and not the other.
        scale = 0.5
        pool = _make_pool(
            dtype=self.fp8_dtype,
            device=self.dev,
            size=self.SIZE,
            page_size=self.PAGE_SIZE,
        )
        layer = _StubLayer(SPARSE_KV_LAYER_ID, scale=scale, device=self.dev)
        k, v, idx_k, idx_v, loc = self._inputs(17)
        k_before, v_before = k.clone(), v.clone()
        pool.set_fused_kv_index_buffer(
            layer, loc, k, v, idx_k, idx_v, scale, scale, None, None
        )
        self.assertTrue(torch.equal(k, k_before))
        self.assertTrue(torch.equal(v, v_before))

    def test_rows_outside_loc_are_untouched(self):
        _, _, _, loc, (kb, vb) = self._store(
            fused=True, layer_id=SPARSE_KV_LAYER_ID, scale=None, seed=23
        )
        untouched = torch.ones(kb.shape[0], dtype=torch.bool, device=self.dev)
        untouched[loc] = False
        self.assertTrue((kb[untouched] == 0).all())
        self.assertTrue((vb[untouched] == 0).all())

    def test_index_caches_still_written_on_the_fused_path(self):
        # The fast path covers main K/V only; the index caches are model-dtype
        # (no cast to fuse) and must keep their own scatter.
        pool = _make_pool(
            dtype=self.fp8_dtype,
            device=self.dev,
            size=self.SIZE,
            page_size=self.PAGE_SIZE,
        )
        layer = _StubLayer(SPARSE_KV_LAYER_ID, device=self.dev)
        k, v, idx_k, idx_v, loc = self._inputs(29)
        pool.set_fused_kv_index_buffer(
            layer, loc, k, v, idx_k, idx_v, None, None, None, None
        )
        got_k, got_v = pool.get_index_kv_buffer(SPARSE_KV_LAYER_ID)
        self.assertTrue(torch.equal(got_k[loc], idx_k))
        self.assertTrue(torch.equal(got_v[loc], idx_v))

    def test_k_only_sparse_layer(self):
        # The block-selector layers have no index V at all; the main-KV fast
        # path must still apply and the K-only index scatter still run.
        pool = _make_pool(
            dtype=self.fp8_dtype,
            device=self.dev,
            size=self.SIZE,
            page_size=self.PAGE_SIZE,
        )
        layer = _StubLayer(SPARSE_K_ONLY_LAYER_ID, device=self.dev)
        k, v, idx_k, _, loc = self._inputs(31)
        pool.set_fused_kv_index_buffer(
            layer, loc, k, v, idx_k, None, None, None, None, None
        )
        local = SPARSE_K_ONLY_LAYER_ID - START_LAYER
        got = pool.main_pool.k_buffer[local].view(self.fp8_dtype)
        self.assertTrue(torch.equal(got[loc], k.float().to(self.fp8_dtype)))
        self.assertTrue(
            torch.equal(pool.get_index_k_buffer(SPARSE_K_ONLY_LAYER_ID)[loc], idx_k)
        )


if __name__ == "__main__":
    unittest.main()
