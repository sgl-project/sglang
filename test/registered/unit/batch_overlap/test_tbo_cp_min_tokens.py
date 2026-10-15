import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.batch_overlap import two_batch_overlap as tbo
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestCpTboMinTokens(CustomTestCase):
    @staticmethod
    def _extend_batch(num_tokens: int):
        left = num_tokens // 2
        return SimpleNamespace(
            forward_mode=ForwardMode.EXTEND,
            spec_info=None,
            extend_num_tokens=num_tokens,
            extend_lens=[left, num_tokens - left],
            is_extend_in_batch=True,
        )

    def _prepare(self, num_tokens: int, *, is_cp_tbo: bool):
        no_a2a = SimpleNamespace(is_none=lambda: True)
        normal_mode = SimpleNamespace(is_low_latency=lambda: False)
        deepep_mode = SimpleNamespace(resolve=lambda _: normal_mode)
        with (
            patch.object(tbo, "is_tbo_enabled", return_value=True),
            patch.object(tbo, "get_moe_a2a_backend", return_value=no_a2a),
            patch.object(tbo, "get_deepep_mode", return_value=deepep_mode),
            patch.object(
                tbo, "get_tbo_token_distribution_threshold", return_value=0.48
            ),
            patch.object(tbo, "_is_cp_tbo_configuration", return_value=is_cp_tbo),
        ):
            preparer = tbo.TboDPAttentionPreparer()
            can_run, mode = preparer.prepare_all_gather(self._extend_batch(num_tokens))
        return preparer, can_run, mode

    def test_threshold_boundary(self):
        threshold = tbo._CP_TBO_MIN_EXTEND_TOKENS
        self.assertFalse(tbo._cp_tbo_has_sufficient_work(threshold - 1))
        self.assertTrue(tbo._cp_tbo_has_sufficient_work(threshold))

    def test_small_cp_extend_falls_back_to_cp_only(self):
        preparer, can_run, mode = self._prepare(
            tbo._CP_TBO_MIN_EXTEND_TOKENS - 1,
            is_cp_tbo=True,
        )
        self.assertFalse(can_run)
        self.assertIsNone(preparer.local_tbo_split_seq_index)
        self.assertEqual(mode, ForwardMode.EXTEND)

    def test_threshold_does_not_disable_non_cp_tbo(self):
        preparer, can_run, _ = self._prepare(
            tbo._CP_TBO_MIN_EXTEND_TOKENS - 1,
            is_cp_tbo=False,
        )
        self.assertTrue(can_run)
        self.assertEqual(preparer.local_tbo_split_seq_index, 1)

    def test_cp_tbo_runs_at_threshold(self):
        preparer, can_run, _ = self._prepare(
            tbo._CP_TBO_MIN_EXTEND_TOKENS,
            is_cp_tbo=True,
        )
        self.assertTrue(can_run)
        self.assertEqual(preparer.local_tbo_split_seq_index, 1)


if __name__ == "__main__":
    unittest.main()
