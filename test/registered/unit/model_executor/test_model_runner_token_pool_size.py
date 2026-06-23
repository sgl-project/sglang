from types import SimpleNamespace

from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestModelRunnerTokenPoolSize(CustomTestCase):
    def _make_runner(
        self,
        *,
        enable_hisparse=False,
        is_hybrid_swa=False,
        size_full=None,
    ):
        runner = object.__new__(ModelRunner)
        runner.enable_hisparse = enable_hisparse
        runner.is_hybrid_swa = is_hybrid_swa
        runner.max_total_num_tokens = 1024
        runner.full_max_total_num_tokens = 2048

        allocator = SimpleNamespace()
        if size_full is not None:
            allocator.size_full = size_full
        runner.token_to_kv_pool_allocator = allocator
        return runner

    def test_hisparse_uses_allocator_size_full_when_available(self):
        runner = self._make_runner(enable_hisparse=True, size_full=4096)

        self.assertEqual(runner.max_token_pool_size, 4096)

    def test_hisparse_without_size_full_keeps_non_hybrid_fallback(self):
        runner = self._make_runner(enable_hisparse=True)

        self.assertEqual(runner.max_token_pool_size, 1024)

    def test_hisparse_without_size_full_preserves_hybrid_swa_fallback(self):
        runner = self._make_runner(enable_hisparse=True, is_hybrid_swa=True)

        self.assertEqual(runner.max_token_pool_size, 2048)

    def test_non_hisparse_preserves_hybrid_swa_behavior(self):
        runner = self._make_runner(enable_hisparse=False, is_hybrid_swa=True)

        self.assertEqual(runner.max_token_pool_size, 2048)

    def test_non_hisparse_preserves_default_behavior(self):
        runner = self._make_runner()

        self.assertEqual(runner.max_token_pool_size, 1024)
