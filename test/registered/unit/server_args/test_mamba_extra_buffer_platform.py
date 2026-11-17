import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.arg_groups.mamba_hook import validate_mamba_extra_buffer
from sglang.srt.arg_groups.overrides import supports_mamba_cache_extra_buffer
from sglang.srt.runtime_context import override_platform
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


def _validate():
    validate_mamba_extra_buffer(
        SimpleNamespace(
            mamba_radix_cache_strategy="extra_buffer",
            speculative_num_draft_tokens=None,
            page_size=None,
        ),
        SimpleNamespace(architectures=["MambaForCausalLM"]),
        mamba_cache_chunk_size_of=lambda: None,
    )


class TestMambaExtraBufferPlatform(CustomTestCase):
    @patch(
        "sglang.srt.arg_groups.mamba_hook.supports_mamba_cache_extra_buffer",
        return_value=True,
    )
    @patch("sglang.srt.arg_groups.mamba_hook.current_platform")
    def test_in_tree_validation_uses_platform_context(self, platform, _supports_model):
        platform.is_out_of_tree.return_value = False
        platform_facts = dict(
            is_cuda=False,
            is_musa=False,
            is_npu=False,
            is_hip=False,
            is_xpu=False,
        )
        with override_platform(**platform_facts):
            with self.assertRaisesRegex(AssertionError, "platform support"):
                _validate()
            for fact in platform_facts:
                with self.subTest(fact=fact), override_platform(**{fact: True}):
                    _validate()
        platform.support_mamba_cache_extra_buffer.assert_not_called()

    @patch(
        "sglang.srt.arg_groups.mamba_hook.supports_mamba_cache_extra_buffer",
        return_value=True,
    )
    @patch("sglang.srt.arg_groups.mamba_hook.current_platform")
    def test_out_of_tree_validation_uses_platform_capability(
        self, platform, _supports_model
    ):
        platform.is_out_of_tree.return_value = True
        with override_platform(is_cuda=True):
            platform.support_mamba_cache_extra_buffer.return_value = False
            with self.assertRaisesRegex(AssertionError, "platform support"):
                _validate()

            platform.support_mamba_cache_extra_buffer.return_value = True
            _validate()

    def test_xpu_keeps_model_extra_buffer_support(self):
        """intel_xpu runs MLA at page_size 128 only, and no_buffer needs page_size 1,
        so a hybrid MLA + linear-attention model on XPU starts only with extra_buffer.
        A model-support check that answered no on XPU left Ling-2.6/3.0 with no valid
        radix-cache strategy (MambaComponent page_size assertion at startup)."""
        view = SimpleNamespace(linear_attn_backend="triton")
        for arch in ("BailingMoeV2_5ForCausalLM", "BailingMoeV3ForCausalLM"):
            with self.subTest(arch=arch), override_platform(is_xpu=True):
                self.assertTrue(
                    supports_mamba_cache_extra_buffer(
                        view, SimpleNamespace(architectures=[arch])
                    )
                )


if __name__ == "__main__":
    unittest.main()
