"""The two-part gate on MiniMax-M3's aiter fused qk-norm + rope + KV-store path.

The fused kernel replaces qk-norm, rope, the bf16->fp8 cast and both cache
stores with one launch. It reads the caches by raw pointer and asserts tensor
properties it cannot recover from, so every case it does not implement has to be
turned away *before* the launch. That screening is split in two:

* ``_aiter_fused_store_cache_views`` -- the pool's physical layout. aiter's
  ``asm_layout=False`` insert mode addresses the main cache as
  ``[num_blocks, page, num_kv_heads, 128]``; only SGLang's plain NHD buffer
  reshapes onto that for free. HND and vectorized_5d are different byte layouts
  and a blind reshape would write the wrong slots -- silently, with plausible
  numbers.
* ``_aiter_fused_store_runtime_reason`` -- the per-call tensor properties, which
  a static check cannot see (``cos_sin_cache`` is re-cast by
  ``SGLANG_ROPE_CACHE_FP32`` and ``_match_cos_sin_cache_dtype``, and the scales
  are filled at weight load).

Both are pure predicates over a small attribute surface, so they are driven here
with stub objects rather than a real attention module: no accelerator, and every
rejection reason gets its own case. A gate term quietly dropped is exactly the
regression this file exists to catch, and it would not show up as a crash.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.models import minimax_m3
from sglang.srt.models.minimax_m3 import MiniMaxM3Attention
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_CACHE_VIEWS = MiniMaxM3Attention._aiter_fused_store_cache_views
_RUNTIME_REASON = MiniMaxM3Attention._aiter_fused_store_runtime_reason

# Production geometry: page_size 128 is what the M3 configs run, and head_dim
# 128 is compile-time in the kernel (enforced by the static half of the gate).
PAGE_SIZE = 128
NUM_BLOCKS = 2
ROWS = NUM_BLOCKS * PAGE_SIZE
NUM_KV_HEADS = 2
HEAD_DIM = 128
ROTARY_DIM = 64
# [q|k|v|idx_q|idx_k] for 4 q heads, 2 kv heads and 1 index head pair.
FUSED_ROW = (4 + 2 * NUM_KV_HEADS + 1 + 1) * HEAD_DIM


class _Pool:
    """The three accessors ``_aiter_fused_store_cache_views`` reaches for."""

    def __init__(self, k, v, index_cache, page_size):
        self._k, self._v = k, v
        self._index_cache = index_cache
        self.page_size = page_size

    def get_kv_buffer(self, layer_id):
        assert layer_id == 3, "the views must be taken for the layer that asked"
        return self._k, self._v

    def get_index_k_buffer(self, layer_id):
        assert layer_id == 3
        return self._index_cache


def _pool(
    *,
    dtype=torch.uint8,
    v_dtype=None,
    rows=ROWS,
    n_kv=NUM_KV_HEADS,
    dim=HEAD_DIM,
    page_size=PAGE_SIZE,
    k_dims=3,
    index_numel=None,
):
    shape = (rows, n_kv, dim) if k_dims == 3 else (rows, n_kv, dim, 1)
    k = torch.zeros(shape, dtype=dtype)
    v = torch.zeros(shape, dtype=v_dtype or dtype)
    numel = rows * dim if index_numel is None else index_numel
    return _Pool(k, v, torch.zeros(numel, dtype=torch.bfloat16), page_size)


def _layer(**overrides):
    """A stub ``self`` whose every term the gate accepts, before a caller breaks one."""
    attrs = {
        "attn": SimpleNamespace(layer_id=3, idx_k_scale_float=1.0),
        "num_kv_heads": NUM_KV_HEADS,
        "head_dim": HEAD_DIM,
        "rotary_dim": ROTARY_DIM,
        "rotary_emb": SimpleNamespace(
            cos_sin_cache=torch.zeros(64, ROTARY_DIM, dtype=torch.bfloat16)
        ),
        "_aiter_fused_row": FUSED_ROW,
        "q_norm": SimpleNamespace(weight=torch.zeros(HEAD_DIM, dtype=torch.bfloat16)),
        "index_q_norm": SimpleNamespace(
            weight=torch.zeros(HEAD_DIM, dtype=torch.bfloat16)
        ),
    }
    attrs.update(overrides)
    return SimpleNamespace(**attrs)


class TestAiterFusedStoreStaticGate(CustomTestCase):
    """Without aiter the path must be unreachable, not merely unused.

    ``_use_aiter`` is ``SGLANG_USE_AITER and _is_hip``: an explicit operator
    opt-in, the same one every other aiter fusion in the tree takes, rather than
    a consequence of running on ROCm. This pins the closed end of that -- on a
    CPU or CUDA build the kernel handle and the accepted storage dtypes must
    both stay empty, so ``_aiter_fused_store_cache_views`` cannot return views
    even for a pool that is otherwise perfectly shaped.
    """

    @unittest.skipIf(minimax_m3._use_aiter, "aiter is in use; nothing to close")
    def test_gate_is_closed_without_aiter(self):
        self.assertIsNone(minimax_m3._aiter_fused_qknorm_idxrqknorm)
        self.assertEqual(minimax_m3._AITER_FP8_CACHE_DTYPES, ())
        # The dtype screen is what turns "no aiter" into "no fused store".
        self.assertIsNone(_CACHE_VIEWS(_layer(), _pool()))


class TestAiterFusedStoreCacheViews(CustomTestCase):
    """The pool-layout half of the gate."""

    def setUp(self):
        super().setUp()
        # aiter resolves exactly one torch fp8 variant per arch, so the real
        # tuple is arch-dependent and empty off ROCm. Pin a value so the
        # accepting path is reachable here; the unpatched empty tuple is covered
        # by TestAiterFusedStoreStaticGate above.
        patcher = patch.object(
            minimax_m3, "_AITER_FP8_CACHE_DTYPES", (torch.uint8, torch.float8_e4m3fn)
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_plain_nhd_pool_reshapes_onto_the_kernel_layout(self):
        pool = _pool()
        views = _CACHE_VIEWS(_layer(), pool)
        self.assertIsNotNone(views)
        k, v, index_cache, page_size = views
        self.assertEqual(page_size, PAGE_SIZE)
        self.assertEqual(k.shape, (NUM_BLOCKS, PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM))
        self.assertEqual(v.shape, k.shape)
        self.assertIs(index_cache, pool._index_cache)
        # A view, not a copy: the kernel writes through these into the pool.
        self.assertEqual(k.data_ptr(), pool._k.data_ptr())
        self.assertEqual(v.data_ptr(), pool._v.data_ptr())

    def test_float8_storage_is_accepted(self):
        self.assertIsNotNone(_CACHE_VIEWS(_layer(), _pool(dtype=torch.float8_e4m3fn)))

    def test_model_dtype_cache_is_rejected(self):
        # A bf16 cache has no fp8 insert mode; aiter's dtype-id lookup raises
        # rather than falling back, so it must never reach the launch.
        self.assertIsNone(_CACHE_VIEWS(_layer(), _pool(dtype=torch.bfloat16)))

    def test_non_nhd_rank_is_rejected(self):
        # Stands in for vectorized_5d: any buffer that is not rank 3 has a
        # different physical layout, and reshaping it would address wrong slots.
        self.assertIsNone(_CACHE_VIEWS(_layer(), _pool(k_dims=4)))

    def test_mismatched_k_and_v_shapes_are_rejected(self):
        pool = _pool()
        pool._v = torch.zeros(ROWS, NUM_KV_HEADS + 1, HEAD_DIM, dtype=torch.uint8)
        self.assertIsNone(_CACHE_VIEWS(_layer(), pool))

    def test_mismatched_k_and_v_dtypes_are_rejected(self):
        self.assertIsNone(
            _CACHE_VIEWS(
                _layer(), _pool(dtype=torch.uint8, v_dtype=torch.float8_e4m3fn)
            )
        )

    def test_rows_must_be_a_whole_number_of_pages(self):
        # (slot // page, slot % page) only addresses the buffer if it tiles it.
        for rows in (ROWS + 1, PAGE_SIZE // 2):
            with self.subTest(rows=rows):
                self.assertIsNone(_CACHE_VIEWS(_layer(), _pool(rows=rows)))

    def test_absent_page_size_is_rejected(self):
        for page_size in (0, -1):
            with self.subTest(page_size=page_size):
                self.assertIsNone(_CACHE_VIEWS(_layer(), _pool(page_size=page_size)))

    def test_pool_geometry_must_match_the_layer(self):
        with self.subTest(term="num_kv_heads"):
            self.assertIsNone(_CACHE_VIEWS(_layer(), _pool(n_kv=NUM_KV_HEADS + 1)))
        with self.subTest(term="head_dim"):
            self.assertIsNone(_CACHE_VIEWS(_layer(), _pool(dim=HEAD_DIM // 2)))

    def test_non_contiguous_buffers_are_rejected(self):
        # The kernel takes raw pointers and assumes packed rows; a strided view
        # would be read as if it were packed.
        for name in ("_k", "_v", "_index_cache"):
            with self.subTest(buffer=name):
                pool = _pool()
                packed = getattr(pool, name)
                doubled = torch.zeros(
                    (packed.shape[0] * 2, *packed.shape[1:]), dtype=packed.dtype
                )
                setattr(pool, name, doubled[::2])
                self.assertFalse(getattr(pool, name).is_contiguous())
                self.assertIsNone(_CACHE_VIEWS(_layer(), pool))

    def test_index_cache_must_cover_the_slot_space(self):
        # Indexed as slot * head_dim + d over the same slots as the main cache.
        short = _pool(index_numel=ROWS * HEAD_DIM - 1)
        self.assertIsNone(_CACHE_VIEWS(_layer(), short))
        exact = _pool(index_numel=ROWS * HEAD_DIM)
        self.assertIsNotNone(_CACHE_VIEWS(_layer(), exact))


class TestAiterFusedStoreRuntimeReason(CustomTestCase):
    """The per-call half: ``None`` to launch, else the first failing term.

    The reason is asserted by substring rather than in full -- the contract is
    that a rejection names *which* term failed, so the one-time log says why the
    path went dark, not that the wording is frozen.
    """

    def _call(self, layer=None, *, positions=None, combined=None, loc=None, rows=8):
        if positions is None:
            positions = torch.zeros(rows, dtype=torch.int64)
        if combined is None:
            combined = torch.zeros(rows, FUSED_ROW, dtype=torch.bfloat16)
        if loc is None:
            loc = torch.zeros(rows, dtype=torch.int64)
        return _RUNTIME_REASON(layer or _layer(), positions, combined, loc)

    def test_accepts_when_every_term_holds(self):
        self.assertIsNone(self._call())

    def test_accepts_fp16(self):
        layer = _layer(
            rotary_emb=SimpleNamespace(
                cos_sin_cache=torch.zeros(64, ROTARY_DIM, dtype=torch.float16)
            ),
            q_norm=SimpleNamespace(weight=torch.zeros(HEAD_DIM, dtype=torch.float16)),
            index_q_norm=SimpleNamespace(
                weight=torch.zeros(HEAD_DIM, dtype=torch.float16)
            ),
        )
        combined = torch.zeros(8, FUSED_ROW, dtype=torch.float16)
        self.assertIsNone(self._call(layer, combined=combined))

    def test_qkv_row_must_be_contiguous_2d_of_the_fused_width(self):
        cases = {
            "rank": torch.zeros(FUSED_ROW, dtype=torch.bfloat16),
            "width": torch.zeros(8, FUSED_ROW - HEAD_DIM, dtype=torch.bfloat16),
            "dtype": torch.zeros(8, FUSED_ROW, dtype=torch.float32),
            "strided": torch.zeros(16, FUSED_ROW, dtype=torch.bfloat16)[::2],
        }
        for name, combined in cases.items():
            with self.subTest(term=name):
                self.assertIn("qkv row", self._call(combined=combined))

    def test_cos_sin_cache_must_match_dtype_and_rotary_dim(self):
        cases = {
            # SGLANG_ROPE_CACHE_FP32 leaves it fp32 against a bf16 qkv row.
            "dtype": torch.zeros(64, ROTARY_DIM, dtype=torch.float32),
            "rotary_dim": torch.zeros(64, ROTARY_DIM * 2, dtype=torch.bfloat16),
            "rank": torch.zeros(64 * ROTARY_DIM, dtype=torch.bfloat16),
            "strided": torch.zeros(64, ROTARY_DIM * 2, dtype=torch.bfloat16)[:, ::2],
        }
        for name, cos_sin in cases.items():
            with self.subTest(term=name):
                layer = _layer(rotary_emb=SimpleNamespace(cos_sin_cache=cos_sin))
                self.assertIn("cos_sin_cache", self._call(layer))

    def test_norm_weights_must_match_the_qkv_dtype(self):
        fp32 = SimpleNamespace(weight=torch.zeros(HEAD_DIM, dtype=torch.float32))
        with self.subTest(term="q_norm"):
            self.assertIn("qk-norm weights", self._call(_layer(q_norm=fp32)))
        with self.subTest(term="index_q_norm"):
            self.assertIn("qk-norm weights", self._call(_layer(index_q_norm=fp32)))

    def test_positions_must_be_a_long_enough_contiguous_int64_vector(self):
        cases = {
            "rank": torch.zeros(8, 1, dtype=torch.int64),
            "dtype": torch.zeros(8, dtype=torch.int32),
            "short": torch.zeros(7, dtype=torch.int64),
            "strided": torch.zeros(16, dtype=torch.int64)[::2],
        }
        for name, positions in cases.items():
            with self.subTest(term=name):
                self.assertIn("positions", self._call(positions=positions))

    def test_out_cache_loc_must_be_a_long_enough_contiguous_int64_vector(self):
        cases = {
            "none": None,
            "rank": torch.zeros(8, 1, dtype=torch.int64),
            "dtype": torch.zeros(8, dtype=torch.int32),
            "short": torch.zeros(7, dtype=torch.int64),
            "strided": torch.zeros(16, dtype=torch.int64)[::2],
        }
        for name, loc in cases.items():
            with self.subTest(term=name):
                # `loc=None` cannot go through the default, so call directly.
                reason = _RUNTIME_REASON(
                    _layer(),
                    torch.zeros(8, dtype=torch.int64),
                    torch.zeros(8, FUSED_ROW, dtype=torch.bfloat16),
                    loc,
                )
                self.assertIn("out_cache_loc", reason)

    def test_non_unit_index_k_scale_is_rejected(self):
        # The kernel's index-cache store applies a hardcoded unit scale, so a
        # non-unit scale would be dropped and the stored index-K silently wrong.
        for scale in (0.5, 2.0):
            with self.subTest(scale=scale):
                layer = _layer(
                    attn=SimpleNamespace(layer_id=3, idx_k_scale_float=scale)
                )
                self.assertIn("index-K scale", self._call(layer))

    def test_unit_and_absent_index_k_scale_are_accepted(self):
        for scale in (None, 1.0):
            with self.subTest(scale=scale):
                layer = _layer(
                    attn=SimpleNamespace(layer_id=3, idx_k_scale_float=scale)
                )
                self.assertIsNone(self._call(layer))

    def test_reports_the_first_failing_term(self):
        """Ordering matters: the log names one reason, so it must be the earliest.

        Break the qkv row and the index-K scale at once -- the row check runs
        first, so a reason mentioning the scale would mean the checks had been
        reordered and the cheap screen no longer runs first.
        """
        layer = _layer(attn=SimpleNamespace(layer_id=3, idx_k_scale_float=0.5))
        reason = self._call(layer, combined=torch.zeros(8, 1, dtype=torch.bfloat16))
        self.assertIn("qkv row", reason)


if __name__ == "__main__":
    unittest.main()
