"""CPU guard tests for FA3's shared MLA FP8 shadow.

The GPU suite validates the Triton conversion and CUDA Graph replay. This file
pins every backend/source eligibility condition independently, so unsupported
configurations cannot accidentally leave the legacy full-pool cast path.
"""

from types import SimpleNamespace

import torch

from sglang.kernels.ops.kvcache.mla_buffer import (
    FA3MLAFP8KVShadow,
    is_fa3_mla_fp8_shadow_enabled,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestBackendEligibilityGuard(CustomTestCase):
    QUALIFYING = dict(
        fa_impl_ver=3,
        use_mla=True,
        page_size=1,
        unified_dense=False,
        attn_cp_size=1,
        is_draft_runner=False,
        dcp_enabled=False,
        dsa_kv_cache_store_fp8=False,
    )

    def test_qualifying_config_enables_shadow(self):
        self.assertTrue(is_fa3_mla_fp8_shadow_enabled(**self.QUALIFYING))

    def test_each_disqualifier_disables_shadow(self):
        overrides = (
            ("FA4", dict(fa_impl_ver=4)),
            ("non-MLA model", dict(use_mla=False)),
            ("page size greater than one", dict(page_size=64)),
            ("unified dense pool", dict(unified_dense=True)),
            ("attention CP", dict(attn_cp_size=2)),
            ("draft runner", dict(is_draft_runner=True)),
            ("decode context parallelism", dict(dcp_enabled=True)),
            ("DSA FP8 storage", dict(dsa_kv_cache_store_fp8=True)),
        )
        for name, override in overrides:
            with self.subTest(name):
                config = {**self.QUALIFYING, **override}
                self.assertFalse(is_fa3_mla_fp8_shadow_enabled(**config))


class TestSourceEligibilityGuard(CustomTestCase):
    @staticmethod
    def _source(**overrides):
        attrs = dict(
            dtype=torch.float8_e4m3fn,
            ndim=3,
            is_cuda=True,
            is_contiguous=lambda: True,
        )
        return SimpleNamespace(**{**attrs, **overrides})

    def test_supported_dtype_pairs(self):
        for source_dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            for output_dtype in (torch.bfloat16, torch.float16):
                with self.subTest(source_dtype=source_dtype, output_dtype=output_dtype):
                    source = self._source(dtype=source_dtype)
                    self.assertTrue(
                        FA3MLAFP8KVShadow.is_supported_source(source, output_dtype)
                    )

    def test_each_source_disqualifier_is_rejected(self):
        cases = (
            (
                "non-FP8 source",
                self._source(dtype=torch.bfloat16),
                torch.bfloat16,
            ),
            (
                "unsupported output dtype",
                self._source(),
                torch.float32,
            ),
            (
                "one-dimensional source",
                self._source(ndim=1),
                torch.bfloat16,
            ),
            (
                "CPU source",
                self._source(is_cuda=False),
                torch.bfloat16,
            ),
            (
                "non-contiguous source",
                self._source(is_contiguous=lambda: False),
                torch.bfloat16,
            ),
        )
        for name, source, output_dtype in cases:
            with self.subTest(name):
                self.assertFalse(
                    FA3MLAFP8KVShadow.is_supported_source(source, output_dtype)
                )

    def test_factory_does_not_allocate_for_cpu_tensor(self):
        source = torch.zeros(8, 1, 576, dtype=torch.uint8).view(torch.float8_e4m3fn)
        self.assertIsNone(FA3MLAFP8KVShadow.maybe_create(source, torch.bfloat16))


class TestShadowStateLayout(CustomTestCase):
    def test_persistent_state_matches_source_pool(self):
        source = torch.zeros(8, 1, 576, dtype=torch.uint8).view(torch.float8_e4m3fn)
        shadow = FA3MLAFP8KVShadow(source, torch.bfloat16)

        self.assertEqual(shadow.buffer.shape, source.shape)
        self.assertEqual(shadow.buffer.dtype, torch.bfloat16)
        self.assertEqual(shadow.buffer.device, source.device)
        self.assertEqual(shadow.page_epochs.shape, (source.shape[0],))
        self.assertEqual(shadow.page_epochs.dtype, torch.int32)
        self.assertEqual(shadow.epoch.shape, torch.Size([]))
        self.assertEqual(shadow.epoch.dtype, torch.int32)
        self.assertEqual(shadow.epoch.item(), 0)


if __name__ == "__main__":
    import unittest

    unittest.main()
