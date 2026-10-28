"""PD peers must agree on the index-K scale bytes: FP32 or DeepGEMM's packed UE8M0."""

import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.disaggregation.utils import get_pd_kv_layout, pd_kv_layouts_match
from sglang.srt.environ import envs
from sglang.srt.layers.attention.dsa.utils import dsa_index_k_cache_is_ue8m0
from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool
from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_UTILS = "sglang.srt.layers.attention.dsa.utils"


def _kernel_args(backend="auto", fp4_indexer=False):
    return SimpleNamespace(
        kernel=SimpleNamespace(
            dsa_paged_mqa_logits_backend=backend,
            enable_deepseek_v4_fp4_indexer=fp4_indexer,
        )
    )


def _device(stack, *, sm100=True, backend="auto", fp4_indexer=False):
    exec_args = _kernel_args(backend, fp4_indexer)
    stack.enter_context(patch(f"{_UTILS}.is_cuda", return_value=True))
    stack.enter_context(patch(f"{_UTILS}.uses_mx_mqa_logits", return_value=sm100))
    stack.enter_context(patch(f"{_UTILS}.get_exec", return_value=exec_args))
    stack.enter_context(
        patch(
            "sglang.srt.layers.attention.dsa.paged_mqa_logits_backend.get_platform",
            return_value=SimpleNamespace(is_sm100=sm100),
        )
    )
    stack.enter_context(
        patch("sglang.srt.disaggregation.utils.get_exec", return_value=exec_args)
    )


def _dsa_pool(index_kpool=1):
    pool = DSATokenToKVPool.__new__(DSATokenToKVPool)
    pool.index_kpool = index_kpool
    return pool


class TestPdIndexKScaleLayout(CustomTestCase):
    def _dsa_layout(self, pool=None, **device):
        with ExitStack() as stack:
            _device(stack, **device)
            return get_pd_kv_layout(pool if pool is not None else _dsa_pool())

    def test_dsa_cache_format_follows_its_sm100_reader(self):
        cases = [
            (dict(backend="deepgemm"), 1, True),
            (dict(backend="cutedsl"), 1, False),
            (dict(backend="cutedsl"), 4, True),
            (dict(sm100=False), 1, False),
        ]
        for device, index_kpool, ue8m0 in cases:
            with self.subTest(device=device, index_kpool=index_kpool):
                with ExitStack() as stack:
                    _device(stack, **device)
                    self.assertEqual(
                        dsa_index_k_cache_is_ue8m0(kpool=index_kpool > 1), ue8m0
                    )
        with ExitStack() as stack:
            _device(stack)
            stack.enter_context(envs.SGLANG_FP8_PAGED_MQA_LOGITS_TORCH.override(True))
            self.assertFalse(dsa_index_k_cache_is_ue8m0(kpool=False))

    def test_dsa_peers_with_different_readers_report_different_layouts(self):
        deepgemm = self._dsa_layout(backend="deepgemm")
        self.assertEqual(deepgemm, "index_k:ue8m0")
        self.assertEqual(self._dsa_layout(backend="cutedsl"), "index_k:fp32")
        self.assertEqual(self._dsa_layout(sm100=False), "index_k:fp32")
        hybrid = SimpleNamespace(full_kv_pool=_dsa_pool())
        self.assertEqual(self._dsa_layout(hybrid, backend="deepgemm"), deepgemm)

    def test_a_peer_without_a_layout_only_matches_fp32_index_scales(self):
        self.assertTrue(pd_kv_layouts_match(None, None))
        self.assertTrue(pd_kv_layouts_match(None, "index_k:fp32"))
        self.assertTrue(pd_kv_layouts_match("v4", None))
        self.assertFalse(pd_kv_layouts_match(None, "index_k:ue8m0"))
        self.assertFalse(pd_kv_layouts_match("v4,index_k:ue8m0", None))
        self.assertFalse(pd_kv_layouts_match("index_k:fp32", "index_k:ue8m0"))
        self.assertTrue(pd_kv_layouts_match("index_k:ue8m0", "index_k:ue8m0"))

    def test_dsv4_layout_names_a_ue8m0_fp8_index_cache(self):
        pool = DeepSeekV4TokenToKVPool.__new__(DeepSeekV4TokenToKVPool)
        cases = [
            (True, False, "v4,index_k:ue8m0"),
            (False, False, "v4"),
            (True, True, "v4"),
        ]
        for ue8m0, fp4_indexer, want in cases:
            with (
                self.subTest(ue8m0=ue8m0, fp4_indexer=fp4_indexer),
                ExitStack() as stack,
            ):
                _device(stack, fp4_indexer=fp4_indexer)
                stack.enter_context(
                    patch.object(
                        DeepSeekV4TokenToKVPool, "pd_kv_layout", return_value="v4"
                    )
                )
                stack.enter_context(
                    patch(
                        "sglang.srt.layers.attention.deep_gemm_mqa_logits."
                        "dsv4_fp8_index_cache_is_ue8m0",
                        return_value=ue8m0,
                    )
                )
                self.assertEqual(get_pd_kv_layout(pool), want)


if __name__ == "__main__":
    unittest.main()
