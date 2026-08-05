"""Unit tests for the CP Cache LayerSplit CLI surface."""

import argparse
import unittest

from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestCpCacheLayerSplitCli(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.parser = argparse.ArgumentParser()
        ServerArgs.add_cli_args(cls.parser)

    def test_canonical_and_dsa_alias_enable_the_same_field(self):
        for option in (
            "--enable-cp-cache-layer-split",
            "--enable-dsa-cache-layer-split",
        ):
            with self.subTest(option=option):
                parsed = self.parser.parse_args(["--model", "dummy", option])
                self.assertTrue(parsed.enable_cp_cache_layer_split)


if __name__ == "__main__":
    unittest.main()
