import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.arg_groups import overrides as overrides_module
from sglang.srt.arg_groups.model_override_base import ResolvedView
from sglang.srt.arg_groups.overrides import (
    _intel_xpu_page_constraint,
    _mamba_radix_cache_resolution,
)
from sglang.srt.runtime_context import override_platform
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _view(**fields):
    defaults = dict(
        attention_backend="intel_xpu",
        prefill_attention_backend=None,
        decode_attention_backend=None,
        page_size=None,
        uses_mamba_radix_cache=False,
        mamba_radix_cache_strategy="auto",
        disable_radix_cache=False,
        disable_overlap_schedule=True,
        linear_attn_backend="triton",
    )
    defaults.update(fields)
    return ResolvedView(SimpleNamespace(**defaults))


def _model(arch):
    return SimpleNamespace(hf_config=SimpleNamespace(architectures=[arch]))


class TestIntelXpuMambaPageSize(CustomTestCase):
    def _strategy(self, arch, **fields):
        with (
            override_platform(is_xpu=True, is_cuda=False, is_hip=False),
            patch.object(
                overrides_module, "model_config_of", return_value=_model(arch)
            ),
        ):
            return _mamba_radix_cache_resolution(_view(**fields))

    def _page(self, *, mla=False, **fields):
        with (
            override_platform(is_xpu=True, is_cuda=False, is_hip=False),
            patch.object(overrides_module, "use_mla_backend", return_value=mla),
        ):
            return _intel_xpu_page_constraint(_view(**fields))

    def test_extra_buffer_when_intel_xpu_will_page(self):
        for arch in (
            "Qwen3_5ForConditionalGeneration",
            "MiniCPMV4_6ForConditionalGeneration",
            "NemotronHForCausalLM",
        ):
            with self.subTest(arch=arch):
                self.assertEqual(
                    self._strategy(arch),
                    {
                        "uses_mamba_radix_cache": True,
                        "mamba_radix_cache_strategy": "extra_buffer",
                    },
                )

    def test_no_buffer_without_extra_buffer_support(self):
        for arch in ("Lfm2ForCausalLM", "Lfm2VlForConditionalGeneration"):
            with self.subTest(arch=arch):
                self.assertEqual(
                    self._strategy(arch),
                    {
                        "uses_mamba_radix_cache": True,
                        "mamba_radix_cache_strategy": "no_buffer",
                        "disable_overlap_schedule": True,
                    },
                )
        self.assertEqual(
            self._strategy(
                "Qwen3_5ForConditionalGeneration", linear_attn_backend="intel_xpu"
            )["mamba_radix_cache_strategy"],
            "no_buffer",
        )

    def test_no_buffer_when_backend_does_not_page(self):
        self.assertEqual(
            self._strategy(
                "Qwen3_5ForConditionalGeneration", attention_backend="triton"
            )["mamba_radix_cache_strategy"],
            "no_buffer",
        )

    def test_explicit_page_size_unchanged(self):
        self.assertEqual(
            self._strategy("Qwen3_5ForConditionalGeneration", page_size=1)[
                "mamba_radix_cache_strategy"
            ],
            "no_buffer",
        )

    def test_no_buffer_hybrid_gets_page_size_one(self):
        self.assertEqual(
            self._page(
                uses_mamba_radix_cache=True, mamba_radix_cache_strategy="no_buffer"
            ),
            {"page_size": 1},
        )

    def test_no_buffer_hybrid_on_mla_is_refused(self):
        with self.assertRaisesRegex(ValueError, "--disable-radix-cache"):
            self._page(
                mla=True,
                uses_mamba_radix_cache=True,
                mamba_radix_cache_strategy="no_buffer",
            )

    def test_other_models_keep_page_size_128(self):
        for fields in (
            {},
            {
                "uses_mamba_radix_cache": True,
                "mamba_radix_cache_strategy": "extra_buffer",
            },
            {"uses_mamba_radix_cache": False, "disable_radix_cache": True},
        ):
            with self.subTest(**fields):
                self.assertEqual(self._page(**fields), {"page_size": 128})

    def test_explicit_supported_page_size_kept(self):
        self.assertEqual(
            self._page(
                page_size=64,
                uses_mamba_radix_cache=True,
                mamba_radix_cache_strategy="extra_buffer",
            ),
            {},
        )


if __name__ == "__main__":
    unittest.main()
