"""DSV4.1 indexer arch gates: SM120/SM121 take the dense DeepGEMM path.

Regression test for sgl-project/sglang#49173: #48399 replaced
``is_sm100_or_newer()`` (capability major >= 10) with a strict ``is_sm(10)``
gate, which silently demoted GB10 (SM121) to the torch prefill fallback whose
per-layer ``tolist()`` costs ~125 ms/layer. The dense DeepGEMM fp8_fp4
mqa-logits kernels support SM100+ (sgl-deep-gemm 0.2.6 dispatches arch 12
with the fp32 weights/logits contract), so only the *sparse* candidate path
stays strict SM100: DeepGEMM's sparse mqa-logits kernels assert arch == 10.

Runs on CPU: the arch is faked by patching ``current_arch()`` in the
v41_indexer / index_logits module namespaces.
"""

import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.kernels.ops.attention.dsv4 import index_logits
from sglang.srt.environ import envs
from sglang.srt.layers.attention.dsv4 import v41_indexer as indexer
from sglang.srt.layers.attention.dsv4.v41_indexer.dense_blocks import (
    DenseBlocksBackend,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


class _FakeCudaArch:
    """Mimics sglang.kernels.arch._cuda.CudaArch for one capability."""

    def __init__(self, capability):
        self.capability = capability
        self.major, self.minor = capability

    @property
    def cuda(self):
        return self

    def is_sm(self, major, minor=None):
        return self.major == major and (minor is None or self.minor == minor)

    def is_sm_at_least(self, major, minor=0):
        return self.capability >= (major, minor)


class _FakeNonCudaArch:
    """Mimics the plain DeviceArch returned off CUDA: every gate is False."""

    @property
    def cuda(self):
        return None

    def is_sm(self, major, minor=None):
        return False

    def is_sm_at_least(self, major, minor=0):
        return False


def _clear_gate_caches():
    indexer.is_sm100.cache_clear()
    indexer.is_sm100_or_newer.cache_clear()
    indexer._use_deep_gemm_prefill.cache_clear()
    indexer.has_dense_fp4_indexer.cache_clear()
    index_logits.index_logits_dtype.cache_clear()
    index_logits.index_weights_dtype.cache_clear()


class TestV41IndexerArchGates(CustomTestCase):
    def setUp(self):
        super().setUp()
        self.addCleanup(_clear_gate_caches)
        _clear_gate_caches()

    def _route(self, arch, *, deep_gemm_present=True):
        """Patch the arch (and deep_gemm presence) and reset the gate caches."""
        stack = ExitStack()
        stack.enter_context(
            mock.patch.object(indexer, "current_arch", return_value=arch)
        )
        stack.enter_context(
            mock.patch.object(index_logits, "current_arch", return_value=arch)
        )
        stack.enter_context(
            mock.patch.object(
                indexer, "has_dense_fp4_indexer", return_value=deep_gemm_present
            )
        )
        _clear_gate_caches()
        return stack

    def test_is_sm100_stays_strict(self):
        for capability, expected in (
            ((10, 0), True),
            ((10, 3), True),
            ((9, 0), False),
            ((12, 0), False),
            ((12, 1), False),
        ):
            with self.subTest(capability=capability):
                with self._route(_FakeCudaArch(capability)):
                    self.assertEqual(indexer.is_sm100(), expected)
        with self._route(_FakeNonCudaArch()):
            self.assertFalse(indexer.is_sm100())

    def test_is_sm100_or_newer_includes_sm120(self):
        # #49173: SM120/SM121 must take the dense DeepGEMM path again.
        for capability, expected in (
            ((10, 0), True),
            ((10, 3), True),
            ((12, 0), True),
            ((12, 1), True),
            ((9, 0), False),
        ):
            with self.subTest(capability=capability):
                with self._route(_FakeCudaArch(capability)):
                    self.assertEqual(indexer.is_sm100_or_newer(), expected)
        with self._route(_FakeNonCudaArch()):
            self.assertFalse(indexer.is_sm100_or_newer())

    def test_full_topk_indexer_routes_sm120_to_deep_gemm(self):
        with self._route(_FakeCudaArch((12, 1))):
            full = indexer.make_full_topk_indexer(
                token_to_kv_pool=SimpleNamespace(),
                req_to_token=torch.empty(0),
                index_heads=64,
                index_topk=512,
            )
            self.assertTrue(full.use_deep_gemm_prefill)
            self.assertTrue(full.use_deep_gemm_decode)

    def test_full_topk_indexer_keeps_hopper_on_torch(self):
        with self._route(_FakeCudaArch((9, 0))):
            full = indexer.make_full_topk_indexer(
                token_to_kv_pool=SimpleNamespace(),
                req_to_token=torch.empty(0),
                index_heads=64,
                index_topk=512,
            )
            self.assertFalse(full.use_deep_gemm_prefill)
            self.assertFalse(full.use_deep_gemm_decode)

    def test_deep_gemm_prefill_still_needs_deep_gemm(self):
        with self._route(_FakeCudaArch((12, 1)), deep_gemm_present=False):
            self.assertFalse(indexer._use_deep_gemm_prefill())

    def test_candidate_indexer_keeps_sm120_on_dense_blocks(self):
        # The sparse-table path is SM100-only in deep-gemm (its kernels
        # assert arch == 10), so SM120 must keep the dense-blocks fallback
        # instead of reaching the paged-sparse RuntimeError.
        for capability in ((12, 0), (12, 1), (9, 0)):
            with self.subTest(capability=capability):
                with self._route(_FakeCudaArch(capability)):
                    prefill, decode = indexer.make_candidate_indexer(
                        token_to_kv_pool=SimpleNamespace(),
                        req_to_token=torch.empty(0),
                        page_size=64,
                        candidate_topk_blocks=8,
                        candidate_block_size=8,
                    )
                    self.assertIsInstance(prefill, DenseBlocksBackend)
                    self.assertIs(prefill, decode)

    def test_index_logits_dtype_stays_fp32_off_sm100(self):
        # The dtype gates stay strict: deep-gemm 0.2.6 asserts fp32 weights
        # off SM100, so the restored path gate must not drag the dtypes
        # along (a naive revert would crash on GB10).
        with envs.SGLANG_OPT_DSV41_BF16_INDEX_LOGITS.override(None):
            for capability, expected in (
                ((10, 0), torch.bfloat16),
                ((12, 0), torch.float32),
                ((12, 1), torch.float32),
                ((9, 0), torch.float32),
            ):
                with self.subTest(capability=capability):
                    with self._route(_FakeCudaArch(capability)):
                        self.assertEqual(index_logits.index_logits_dtype(), expected)
                        self.assertEqual(index_logits.index_weights_dtype(), expected)


if __name__ == "__main__":
    unittest.main()
