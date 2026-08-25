import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.arg_groups.mamba_hook import validate_mamba_extra_buffer
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestMambaExtraBufferPlatform(CustomTestCase):
    @patch(
        "sglang.srt.arg_groups.mamba_hook.supports_mamba_cache_extra_buffer",
        return_value=True,
    )
    @patch("sglang.srt.arg_groups.mamba_hook.current_platform")
    def test_validation_uses_platform_capability(self, platform, _supports_model):
        view = SimpleNamespace(
            mamba_radix_cache_strategy="extra_buffer",
            speculative_num_draft_tokens=None,
            page_size=None,
        )

        platform.support_mamba_cache_extra_buffer.return_value = False
        with self.assertRaisesRegex(AssertionError, "platform support"):
            validate_mamba_extra_buffer(
                view,
                "MambaForCausalLM",
                mamba_cache_chunk_size_of=lambda: None,
            )

        platform.support_mamba_cache_extra_buffer.return_value = True
        validate_mamba_extra_buffer(
            view,
            "MambaForCausalLM",
            mamba_cache_chunk_size_of=lambda: None,
        )


if __name__ == "__main__":
    unittest.main()
