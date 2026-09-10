"""CPU-only regression tests for DFLASH and PD disaggregation validation."""

import unittest
from types import SimpleNamespace
from unittest import mock

from sglang.srt.arg_groups.speculative_hook import _handle_dflash
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestDflashPdDisaggregationGuard(CustomTestCase):
    @staticmethod
    def _server_args(disaggregation_mode):
        return SimpleNamespace(
            device="cuda",
            disaggregation_mode=disaggregation_mode,
            pp_size=1,
            speculative_draft_model_path="dummy-draft",
            speculative_num_steps=1,
            speculative_eagle_topk=1,
            speculative_dflash_block_size=None,
            speculative_num_draft_tokens=8,
            speculative_draft_window_size=None,
            max_running_requests=48,
            enable_mixed_chunk=False,
        )

    def test_rejects_pd_prefill_and_decode_before_model_loading(self):
        for mode in ("prefill", "decode"):
            with self.subTest(disaggregation_mode=mode):
                with self.assertRaisesRegex(
                    ValueError, "does not transfer DFLASH draft state"
                ):
                    _handle_dflash(self._server_args(mode))

    def test_keeps_non_pd_dflash_available(self):
        with (
            mock.patch(
                "sglang.srt.arg_groups.overrides.resolved_view",
                return_value=SimpleNamespace(enable_dp_attention=False),
            ),
            mock.patch(
                "sglang.srt.arg_groups.speculative_hook._resolve_dflash_draft_attention_backend"
            ),
        ):
            _handle_dflash(self._server_args("null"))


if __name__ == "__main__":
    unittest.main()
