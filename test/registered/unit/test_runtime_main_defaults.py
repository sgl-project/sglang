"""Keep the runtime PR's speculative TP synchronization default equal to main."""

import unittest

from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestRuntimeMainDefaults(unittest.TestCase):
    def test_spec_tp_sync_retains_main_init_default_and_explicit_override(self):
        sync = envs.SGLANG_SPEC_TP_SYNC
        for master in (False, True):
            with envs.SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION.override(master):
                with sync.override("init"):
                    sync.clear()
                    self.assertEqual(sync.get(), "init")
                    with sync.override("all"):
                        self.assertEqual(sync.get(), "all")
                    self.assertEqual(sync.get(), "init")


if __name__ == "__main__":
    unittest.main()
