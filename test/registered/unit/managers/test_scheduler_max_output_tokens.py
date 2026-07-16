import unittest
from types import SimpleNamespace

from sglang.srt.environ import envs
from sglang.srt.managers.scheduler import Scheduler
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="stage-a-test-cpu")


class TestSchedulerMaxOutputTokens(unittest.TestCase):
    def _new_scheduler(
        self,
        max_req_len: int = 128,
        max_total_num_tokens: int = 1024,
        page_size: int = 1,
    ) -> Scheduler:
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.max_req_len = max_req_len
        scheduler.max_total_num_tokens = max_total_num_tokens
        scheduler.page_size = page_size
        scheduler.max_output_tokens = envs.SGLANG_MAX_OUTPUT_TOKENS.get()
        return scheduler

    def _new_req(self, max_new_tokens, input_len: int = 8, min_new_tokens: int = 0):
        return SimpleNamespace(
            origin_input_ids=[0] * input_len,
            sampling_params=SimpleNamespace(
                max_new_tokens=max_new_tokens, min_new_tokens=min_new_tokens
            ),
        )

    def test_env_limit_is_disabled_by_default(self):
        req = self._new_req(max_new_tokens=64, input_len=8)

        with envs.SGLANG_MAX_OUTPUT_TOKENS.override(None):
            scheduler = self._new_scheduler(max_req_len=128)
            scheduler.init_req_max_new_tokens(req)

        self.assertEqual(req.sampling_params.max_new_tokens, 64)

    def test_env_limit_clips_request_max_new_tokens(self):
        req = self._new_req(max_new_tokens=64, input_len=8)

        with envs.SGLANG_MAX_OUTPUT_TOKENS.override(16):
            scheduler = self._new_scheduler(max_req_len=128)
            scheduler.init_req_max_new_tokens(req)

        self.assertEqual(req.sampling_params.max_new_tokens, 16)

    def test_env_limit_applies_when_request_limit_is_not_set(self):
        req = self._new_req(max_new_tokens=None, input_len=8)

        with envs.SGLANG_MAX_OUTPUT_TOKENS.override(16):
            scheduler = self._new_scheduler(max_req_len=128)
            scheduler.init_req_max_new_tokens(req)

        self.assertEqual(req.sampling_params.max_new_tokens, 16)

    def test_context_limit_still_applies_after_env_limit(self):
        req = self._new_req(max_new_tokens=64, input_len=20)

        with envs.SGLANG_MAX_OUTPUT_TOKENS.override(16):
            scheduler = self._new_scheduler(max_req_len=32)
            scheduler.init_req_max_new_tokens(req)

        self.assertEqual(req.sampling_params.max_new_tokens, 11)

    def test_non_positive_env_limit_is_ignored(self):
        req = self._new_req(max_new_tokens=64, input_len=8)

        with envs.SGLANG_MAX_OUTPUT_TOKENS.override(0):
            scheduler = self._new_scheduler(max_req_len=128)
            scheduler.init_req_max_new_tokens(req)

        self.assertEqual(req.sampling_params.max_new_tokens, 64)

    def test_admission_budget_limit_still_applies_after_env_limit(self):
        req = self._new_req(max_new_tokens=64, input_len=8)

        with envs.SGLANG_MAX_OUTPUT_TOKENS.override(32):
            scheduler = self._new_scheduler(
                max_req_len=128, max_total_num_tokens=24, page_size=4
            )
            scheduler.init_req_max_new_tokens(req)

        self.assertEqual(req.sampling_params.max_new_tokens, 11)

    def test_min_new_tokens_is_clipped_with_env_limit(self):
        req = self._new_req(max_new_tokens=64, min_new_tokens=32, input_len=8)

        with envs.SGLANG_MAX_OUTPUT_TOKENS.override(16):
            scheduler = self._new_scheduler(max_req_len=128)
            scheduler.init_req_max_new_tokens(req)

        self.assertEqual(req.sampling_params.max_new_tokens, 16)
        self.assertEqual(req.sampling_params.min_new_tokens, 16)


if __name__ == "__main__":
    unittest.main()
