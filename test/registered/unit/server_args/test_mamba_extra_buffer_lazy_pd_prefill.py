"""--mamba-radix-cache-strategy extra_buffer_lazy on PD workers: accepted on a unified or
PD prefill worker, refused on a PD decode worker; extra_buffer is accepted everywhere."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.arg_groups import mamba_hook
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _validate(strategy, mode):
    view = SimpleNamespace(
        mamba_radix_cache_strategy=strategy,
        disaggregation_mode=mode,
        speculative_num_draft_tokens=None,
        page_size=None,
    )
    with (
        patch.object(
            mamba_hook, "supports_mamba_cache_extra_buffer", return_value=True
        ),
        patch.object(
            mamba_hook, "_platform_supports_mamba_cache_extra_buffer", return_value=True
        ),
    ):
        mamba_hook.validate_mamba_extra_buffer(
            view,
            SimpleNamespace(architectures=["KimiK3ForConditionalGeneration"]),
            mamba_cache_chunk_size_of=lambda: None,
        )


class TestMambaExtraBufferLazyPD(CustomTestCase):
    def test_lazy_allowed_on_unified_and_prefill(self):
        _validate("extra_buffer_lazy", "null")
        _validate("extra_buffer_lazy", "prefill")

    def test_lazy_refused_on_decode(self):
        with self.assertRaisesRegex(AssertionError, "PD decode worker"):
            _validate("extra_buffer_lazy", "decode")

    def test_extra_buffer_everywhere(self):
        for mode in ("null", "prefill", "decode"):
            with self.subTest(mode=mode):
                _validate("extra_buffer", mode)


if __name__ == "__main__":
    unittest.main()
