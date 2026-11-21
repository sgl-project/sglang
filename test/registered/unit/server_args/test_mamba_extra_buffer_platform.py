import unittest
from importlib.metadata import PackageNotFoundError
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.arg_groups.mamba_hook import (
    handle_mamba_backend,
    validate_mamba_extra_buffer,
)
from sglang.srt.runtime_context import override_platform
from sglang.srt.server_args import ServerArgs
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


class TestFlashInferSSDPrefillVersion(CustomTestCase):
    def test_requires_released_exact_scan_implementation(self):
        args = ServerArgs(model_path="dummy", mamba_prefill_backend="flashinfer")
        for installed, accepted in (
            ("0.7.1", False),
            ("0.7.2rc1", False),
            ("0.7.2", True),
            ("0.7.2+local", True),
            ("0.7.3", True),
            (None, False),
        ):
            with (
                self.subTest(installed=installed),
                override_platform(is_sm100=True),
                patch(
                    "sglang.srt.utils.common.version",
                    return_value=installed,
                    side_effect=PackageNotFoundError if installed is None else None,
                ),
            ):
                if accepted:
                    handle_mamba_backend(args)
                else:
                    with self.assertRaisesRegex(
                        ValueError, "flashinfer-python >= 0.7.2"
                    ):
                        handle_mamba_backend(args)


if __name__ == "__main__":
    unittest.main()
