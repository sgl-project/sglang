import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.arg_groups.mamba_hook import validate_mamba_extra_buffer
from sglang.srt.arg_groups.overrides import supports_mamba_cache_extra_buffer
from sglang.srt.runtime_context import override_platform
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


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


def _supports(arch: str) -> bool:
    return supports_mamba_cache_extra_buffer(
        SimpleNamespace(linear_attn_backend="triton"),
        SimpleNamespace(architectures=[arch]),
    )


class TestMambaExtraBufferXpuArchs(CustomTestCase):
    def test_inkling_is_admitted_on_xpu(self):
        """Inkling has no no_buffer path, so rejecting it on XPU fails every launch."""
        with override_platform(is_xpu=True):
            for arch in (
                "InklingForConditionalGeneration",
                "InklingForConditionalGenerationMTP",
            ):
                with self.subTest(arch=arch):
                    self.assertTrue(_supports(arch))

    def test_other_archs_stay_rejected_on_xpu(self):
        for arch in ("KimiK3ForConditionalGeneration", "NemotronHForCausalLM"):
            with self.subTest(arch=arch):
                with override_platform(is_xpu=False):
                    self.assertTrue(_supports(arch))
                with override_platform(is_xpu=True):
                    self.assertFalse(_supports(arch))


if __name__ == "__main__":
    unittest.main()
