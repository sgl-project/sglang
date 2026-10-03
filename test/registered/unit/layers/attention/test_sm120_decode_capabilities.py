"""CPU-only capability tests; requires the CUDA Python dependency stack."""

import sys
import unittest
from types import ModuleType, SimpleNamespace
from unittest import mock

from sglang.kernels.ops.attention import flash_mla_sm120 as fmod
from sglang.kernels.ops.attention.flash_mla_sm120 import (
    flashinfer_dsv4_decode_supports_num_heads,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=1, stage="base-b", runner_config="1-gpu-small")


class TestFlashInferDecodeCapabilities(unittest.TestCase):
    """Capability probing needs neither a GPU nor an installed FlashInfer."""

    def setUp(self):
        super().setUp()
        fmod._flashinfer_dsv4_decode_capabilities.cache_clear()
        self.addCleanup(fmod._flashinfer_dsv4_decode_capabilities.cache_clear)
        package = ModuleType("flashinfer")
        package.__path__ = []
        self.mla = ModuleType("flashinfer.mla")
        self.mla.__path__ = []
        self.legacy = ModuleType("flashinfer.mla._sparse_mla_sm120")
        self.legacy._DECODE_MAX_TOKENS = 64
        self.legacy._DECODE_DSV4_DISPATCH = frozenset({(16, 128), (32, 128), (64, 256)})
        modules = mock.patch.dict(
            sys.modules,
            {
                "flashinfer": package,
                "flashinfer.mla": self.mla,
                "flashinfer.mla._sparse_mla_sm120": self.legacy,
            },
        )
        modules.start()
        self.addCleanup(modules.stop)

    def test_public_runtime_envelope_and_cache(self):
        # The new private envelope is intentionally not iterable.
        self.legacy._DECODE_DSV4_DISPATCH = object()
        config = SimpleNamespace(
            max_num_tokens=64,
            supported_num_heads=mock.Mock(return_value=tuple(range(1, 129))),
        )
        query = mock.Mock(return_value={"dsv4": config})
        self.mla.supported_sparse_mla_sm120_configs = query
        for heads in (1, 8, 16, 32, 64, 128):
            self.assertTrue(flashinfer_dsv4_decode_supports_num_heads(heads, 64))
        for heads, tokens in ((0, 1), (129, 1), (8, 65)):
            self.assertFalse(flashinfer_dsv4_decode_supports_num_heads(heads, tokens))
        query.assert_called_once_with()
        config.supported_num_heads.assert_called_once_with()

    def test_public_finite_heads_and_token_limit(self):
        self.mla.supported_sparse_mla_sm120_configs = mock.Mock(
            return_value={
                "dsv4": SimpleNamespace(
                    max_num_tokens=32,
                    supported_num_heads=lambda: (16, 64, 128),
                )
            }
        )
        self.assertTrue(flashinfer_dsv4_decode_supports_num_heads(16, 32))
        self.assertFalse(flashinfer_dsv4_decode_supports_num_heads(32, 1))
        self.assertFalse(flashinfer_dsv4_decode_supports_num_heads(16, 33))

    def test_public_missing_family_fails_closed(self):
        self.mla.supported_sparse_mla_sm120_configs = mock.Mock(return_value={})
        self.assertFalse(flashinfer_dsv4_decode_supports_num_heads(16, 1))

    def test_legacy_enumerated_dispatch(self):
        for heads in (16, 32, 64):
            self.assertTrue(flashinfer_dsv4_decode_supports_num_heads(heads, 64))
        self.assertFalse(flashinfer_dsv4_decode_supports_num_heads(8, 1))
        self.assertFalse(flashinfer_dsv4_decode_supports_num_heads(16, 65))

    def test_missing_legacy_capabilities_fails_closed(self):
        del self.legacy._DECODE_DSV4_DISPATCH
        self.assertFalse(flashinfer_dsv4_decode_supports_num_heads(16, 1))

    def test_unenumerable_legacy_dispatch_fails_closed(self):
        self.legacy._DECODE_DSV4_DISPATCH = object()
        self.assertFalse(flashinfer_dsv4_decode_supports_num_heads(16, 1))

    def test_missing_flashinfer_fails_closed(self):
        with mock.patch.dict(
            sys.modules,
            {"flashinfer.mla": None, "flashinfer.mla._sparse_mla_sm120": None},
        ):
            self.assertFalse(flashinfer_dsv4_decode_supports_num_heads(16, 1))


if __name__ == "__main__":
    unittest.main()
