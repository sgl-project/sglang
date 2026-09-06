# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import os
import tempfile
import unittest

from sglang.srt.utils.server_args_config_parser import ConfigArgumentMerger
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestConfigArgumentMerger(unittest.TestCase):
    def _merge(self, parser, config):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(config)
            config_file = f.name

        try:
            return ConfigArgumentMerger(parser).merge_config_with_args(
                ["--config", config_file]
            )
        finally:
            os.unlink(config_file)

    def test_yaml_list_is_json_for_single_value_action(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--forward-hooks", type=json.loads)

        merged = self._merge(
            parser,
            "forward-hooks:\n  - type: capture\n    layers: [1, 2]\n",
        )

        self.assertEqual(
            parser.parse_args(merged).forward_hooks,
            [{"type": "capture", "layers": [1, 2]}],
        )

    def test_empty_yaml_list_is_preserved_for_single_value_action(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--sidecar-args", type=json.loads)

        merged = self._merge(parser, "sidecar-args: []\n")

        self.assertEqual(parser.parse_args(merged).sidecar_args, [])

    def test_yaml_list_is_expanded_for_multi_value_action(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--cuda-graph-bs", type=int, nargs="+")

        merged = self._merge(parser, "cuda-graph-bs: [1, 4]\n")

        self.assertEqual(parser.parse_args(merged).cuda_graph_bs, [1, 4])


if __name__ == "__main__":
    unittest.main()
