import os
import unittest

from sglang.srt.environ import envs, temp_set_env
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestTempSetEnv(unittest.TestCase):
    def tearDown(self):
        envs.SGLANG_HICACHE_MOONCAKE_CONFIG_PATH.clear()
        envs.MOONCAKE_MASTER.clear()
        envs.SGLANG_TEST_RETRACT.clear()
        os.environ.pop("SGLANG_TEMP_SET_ENV_TEST", None)

    def test_descriptor_backed_envs_are_set_and_restored(self):
        with temp_set_env(
            SGLANG_HICACHE_MOONCAKE_CONFIG_PATH="/tmp/mooncake.json",
            MOONCAKE_MASTER="127.0.0.1:50051",
        ):
            self.assertEqual(
                envs.SGLANG_HICACHE_MOONCAKE_CONFIG_PATH.get(),
                "/tmp/mooncake.json",
            )
            self.assertEqual(envs.MOONCAKE_MASTER.get(), "127.0.0.1:50051")

        self.assertFalse(envs.SGLANG_HICACHE_MOONCAKE_CONFIG_PATH.is_set())
        self.assertFalse(envs.MOONCAKE_MASTER.is_set())

    def test_restores_descriptor_none_state_and_plain_env(self):
        envs.SGLANG_TEST_RETRACT.set(None)

        with temp_set_env(
            SGLANG_TEST_RETRACT=True,
            SGLANG_TEMP_SET_ENV_TEST="value",
        ):
            self.assertTrue(envs.SGLANG_TEST_RETRACT.get())
            self.assertEqual(os.environ["SGLANG_TEMP_SET_ENV_TEST"], "value")

        self.assertTrue(envs.SGLANG_TEST_RETRACT.is_set())
        self.assertIsNone(envs.SGLANG_TEST_RETRACT.get())
        self.assertNotIn("SGLANG_TEMP_SET_ENV_TEST", os.environ)


if __name__ == "__main__":
    unittest.main(verbosity=2)
