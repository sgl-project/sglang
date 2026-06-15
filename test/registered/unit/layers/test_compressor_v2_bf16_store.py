"""Unit tests for bf16 kv-cache store dispatch in CompressorBackendMixin.

Covers the _forward_compress_all_in_one change in compressor_v2.py:
  - is_bf16_kv_cache=True  + not is_indexer  → compress_norm_rope_store_bf16
  - is_bf16_kv_cache=True  + is_indexer      → compress_norm_rope_store  (FP8)
  - is_bf16_kv_cache=False                   → compress_norm_rope_store

The jit_kernel.internal.dsv4 package requires a built C extension and a
utils.py that only exist in the full build.  We stub those modules before
importing compressor_v2 so the test runs on any GPU CI runner.
"""

import sys
import types
import unittest
from unittest.mock import MagicMock, patch

import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b", runner_config="1-gpu-small")

# ---------------------------------------------------------------------------
# Stub unbuilt internal JIT modules before any sglang.srt import touches them.
# ---------------------------------------------------------------------------
_STUB_MODS = [
    "sglang.jit_kernel.internal",
    "sglang.jit_kernel.internal.dsv4",
    "sglang.jit_kernel.internal.dsv4.utils",
    "sglang.jit_kernel.internal.dsv4.compress",
    "sglang.jit_kernel.internal.dsv4.elementwise",
]
_stubs = {}
for _m in _STUB_MODS:
    if _m not in sys.modules:
        _stubs[_m] = types.ModuleType(_m)
        sys.modules[_m] = _stubs[_m]

# Provide the symbol that compressor_v2 imports from the internal package.
_mock_bf16_store = MagicMock(name="compress_norm_rope_store_bf16")
sys.modules["sglang.jit_kernel.internal.dsv4"].compress_norm_rope_store_bf16 = (
    _mock_bf16_store
)

# Now it is safe to import the module under test.
from sglang.srt.layers.attention.dsv4.compressor_v2 import (  # noqa: E402
    CompressorBackendMixin,
)

_MODULE = "sglang.srt.layers.attention.dsv4.compressor_v2"


def _make_backend(is_bf16: bool) -> CompressorBackendMixin:
    """Return a bare CompressorBackendMixin instance (no super().__init__)."""

    class _Backend(CompressorBackendMixin):
        pass

    obj = _Backend.__new__(_Backend)
    obj.is_bf16_kv_cache = is_bf16
    obj.token_to_kv_pool = MagicMock()
    return obj


def _call_forward(obj: CompressorBackendMixin, is_indexer: bool) -> None:
    """Call _forward_compress_all_in_one with a minimal fake plan.

    Assertion in _forward_compress_all_in_one:
        rotate == is_indexer == (head_dim == 128)
    So:
      - is_indexer=True  → rotate=True,  head_dim=128
      - is_indexer=False → rotate=False, head_dim=512 (core attention)
    """
    head_dim = 128 if is_indexer else 512
    compress_ratio = 4
    device = torch.device("cuda")
    coff = 2  # overlap=True for c4
    last_dim = 2 * head_dim * coff

    fake_plan = MagicMock()
    fake_plan.is_decode = True
    obj._get_paged_compress_metadata = lambda _: fake_plan

    fake_compressed = torch.zeros(1, head_dim, device=device, dtype=torch.bfloat16)

    with patch(f"{_MODULE}.compress_forward", return_value=fake_compressed), patch(
        f"{_MODULE}._use_online_compress", return_value=False
    ):
        obj._forward_compress_all_in_one(
            kv_score_buffer=torch.zeros(
                compress_ratio * 2, last_dim, device=device, dtype=torch.bfloat16
            ),
            kv_score_input=torch.zeros(
                1, last_dim, device=device, dtype=torch.bfloat16
            ),
            ape=torch.zeros(compress_ratio, head_dim, device=device),
            head_dim=head_dim,
            norm=MagicMock(
                weight=torch.ones(head_dim, device=device),
                variance_epsilon=1e-6,
            ),
            freqs_cis_cache=torch.zeros(16, head_dim // 2, 2, device=device),
            kv_cache=torch.zeros(256, device=device, dtype=torch.uint8),
            is_indexer=is_indexer,
            rotate=is_indexer,
            compress_ratio=compress_ratio,
            page_size=compress_ratio,
            out_loc=torch.zeros(1, device=device, dtype=torch.int32),
            use_fp4_indexer=False,
        )


class TestBf16StoreDispatch(CustomTestCase):
    """Verify store function selection based on is_bf16_kv_cache / is_indexer."""

    def setUp(self):
        import sglang.srt.layers.attention.dsv4.compressor_v2 as _mod

        self._fp8_mock = MagicMock(name="compress_norm_rope_store")
        self._bf16_mock = MagicMock(name="compress_norm_rope_store_bf16")
        self._patches = [
            patch.object(_mod, "compress_norm_rope_store", self._fp8_mock),
            patch.object(_mod, "compress_norm_rope_store_bf16", self._bf16_mock),
        ]
        for p in self._patches:
            p.start()

    def tearDown(self):
        for p in self._patches:
            p.stop()

    def test_bf16_core_uses_bf16_store(self):
        """BF16 kvcache + core compressor must call bf16 store."""
        _call_forward(_make_backend(is_bf16=True), is_indexer=False)
        self.assertTrue(
            self._bf16_mock.called, "expected compress_norm_rope_store_bf16"
        )
        self.assertFalse(
            self._fp8_mock.called, "must not call compress_norm_rope_store"
        )

    def test_bf16_indexer_uses_fp8_store(self):
        """BF16 kvcache + indexer must still use FP8 store."""
        _call_forward(_make_backend(is_bf16=True), is_indexer=True)
        self.assertTrue(self._fp8_mock.called, "expected compress_norm_rope_store")
        self.assertFalse(self._bf16_mock.called, "indexer must not call bf16 store")

    def test_fp8_core_uses_fp8_store(self):
        """FP8 kvcache + core compressor must use FP8 store."""
        _call_forward(_make_backend(is_bf16=False), is_indexer=False)
        self.assertTrue(self._fp8_mock.called, "expected compress_norm_rope_store")
        self.assertFalse(self._bf16_mock.called)

    def test_missing_attr_defaults_to_fp8(self):
        """Object without is_bf16_kv_cache attribute defaults to FP8 store."""
        obj = _make_backend(is_bf16=False)
        del obj.is_bf16_kv_cache
        _call_forward(obj, is_indexer=False)
        self.assertTrue(self._fp8_mock.called)
        self.assertFalse(self._bf16_mock.called)


if __name__ == "__main__":
    unittest.main(verbosity=2)
