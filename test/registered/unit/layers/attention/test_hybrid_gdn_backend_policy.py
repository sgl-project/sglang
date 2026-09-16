import sys
import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import MagicMock, patch, sentinel

import torch

from sglang.srt.layers.attention import attention_registry
from sglang.srt.runtime_context import get_platform
from sglang.srt.utils import common
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestHybridGDNBackendPolicy(CustomTestCase):
    def setUp(self):
        super().setUp()
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        # Exercise the startup wrapper without constructing GPU backends.
        self.gdn_backend = MagicMock()
        self.hybrid_backend = MagicMock()
        modules = {
            "sglang.srt.configs.model_config": MagicMock(
                is_minimax_sparse=MagicMock(return_value=False)
            ),
            "sglang.kernels.ops.attention.fla.utils": MagicMock(),
            "sglang.srt.layers.attention.linear.kda_backend": MagicMock(),
            "sglang.srt.layers.attention.linear.lightning_backend": MagicMock(),
            "sglang.srt.layers.attention.linear.utils": MagicMock(),
            "sglang.srt.layers.attention.hybrid_linear_attn_backend": MagicMock(
                HybridLinearAttnBackend=self.hybrid_backend
            ),
            "sglang.srt.layers.attention.linear.gdn_backend": MagicMock(
                GDNAttnBackend=self.gdn_backend
            ),
            "sglang.srt.layers.attention.qsa.config": MagicMock(
                is_qwen_qsa=MagicMock(return_value=False)
            ),
        }
        self.stack.enter_context(patch.dict(sys.modules, modules))
        cfg = SimpleNamespace(full_attention_layer_ids=[3, 7])
        self.stack.enter_context(
            patch.object(attention_registry, "hybrid_gdn_config", return_value=cfg)
        )
        self.stack.enter_context(
            patch.object(attention_registry, "mambaish_config", return_value=cfg)
        )
        self.stack.enter_context(
            patch("sglang.srt.utils.is_blackwell", return_value=True)
        )
        self.stack.enter_context(patch("sglang.srt.utils.is_npu", return_value=False))
        self.stack.enter_context(patch("sglang.srt.utils.is_xpu", return_value=False))

    def wrap(self, sm, prefill, decode):
        runner = SimpleNamespace(
            model_config=SimpleNamespace(
                hf_config=SimpleNamespace(), hf_text_config=SimpleNamespace()
            ),
            use_mla_backend=False,
            is_draft_worker=False,
            prefill_attention_backend_str=prefill,
            decode_attention_backend_str=decode,
        )
        with patch.object(
            attention_registry,
            "get_platform",
            return_value=SimpleNamespace(is_sm110=sm == 110, is_sm120=sm == 120),
        ):
            result = attention_registry.attn_backend_wrapper(runner, sentinel.full_attn)
        self.gdn_backend.assert_called_with(runner)
        self.hybrid_backend.assert_called_with(
            sentinel.full_attn, self.gdn_backend.return_value, [3, 7]
        )
        self.assertIs(result, self.hybrid_backend.return_value)

    def test_sm110_accepts_flashinfer_for_prefill_and_decode(self):
        for prefill, decode in (
            ("flashinfer", "flashinfer"),
            ("flashinfer", "triton"),
            ("triton", "flashinfer"),
        ):
            with self.subTest(prefill=prefill, decode=decode):
                self.wrap(110, prefill, decode)

    def test_sm110_preserves_existing_backend_choices(self):
        for backend in ("triton", "trtllm_mha", "fa4"):
            with self.subTest(backend=backend):
                self.wrap(110, backend, backend)

    def test_datacenter_blackwell_still_rejects_flashinfer(self):
        for sm in (100, 103):
            for prefill, decode in (
                ("flashinfer", "triton"),
                ("triton", "flashinfer"),
            ):
                with self.subTest(sm=sm, prefill=prefill, decode=decode):
                    with self.assertRaisesRegex(AssertionError, "hybrid GDN models"):
                        self.wrap(sm, prefill, decode)
        self.gdn_backend.assert_not_called()

    def test_sm120_backend_choices_are_unchanged(self):
        for backend in ("triton", "trtllm_mha", "flashinfer"):
            with self.subTest(backend=backend):
                self.wrap(120, backend, backend)
        with self.assertRaisesRegex(AssertionError, "hybrid GDN models"):
            self.wrap(120, "fa4", "fa4")

    def test_sm110_rejects_invalid_prefill_and_decode(self):
        for prefill, decode in (
            ("unsupported", "flashinfer"),
            ("flashinfer", "unsupported"),
        ):
            with self.subTest(prefill=prefill, decode=decode):
                with self.assertRaisesRegex(
                    AssertionError, f"Got prefill={prefill}, decode={decode}"
                ):
                    self.wrap(110, prefill, decode)
        self.gdn_backend.assert_not_called()


class TestSM110PlatformProbe(CustomTestCase):
    def test_probe_checks_device_and_cuda_version(self):
        cases = (
            (True, (11, 0), "12.8", True),
            (True, (11, 0), "13.0", True),
            (True, (11, 0), "12.7", False),
            (True, (10, 0), "13.0", False),
            (True, (10, 3), "13.0", False),
            (True, (12, 0), "13.0", False),
            (True, (9, 0), "13.0", False),
            (False, None, None, False),
        )
        self.addCleanup(common.is_sm110_supported.cache_clear)
        for cuda, capability, version, expected in cases:
            with self.subTest(cuda=cuda, capability=capability, version=version):
                common.is_sm110_supported.cache_clear()
                with (
                    patch.object(common, "is_cuda", return_value=cuda),
                    patch.object(
                        torch.cuda, "get_device_capability", return_value=capability
                    ) as get_capability,
                    patch.object(torch.version, "cuda", version),
                ):
                    self.assertEqual(get_platform().is_sm110, expected)
                    if not cuda:
                        get_capability.assert_not_called()


if __name__ == "__main__":
    unittest.main()
