"""Unit test: target_layers field filters which MoE layers get swapped."""
import json
import os
import tempfile
import unittest

from sglang.srt.layers.moe.heter_moe import _parse_heter_config


class TestTargetLayers(unittest.TestCase):
    def _write_cfg(self, d, extra):
        cfg = {
            "groups": [
                {"name": "cold", "num_bits": 4, "group_size": 128,
                 "checkpoint": "/fake/path"},
                {"name": "hot", "num_bits": 16},
            ],
            "policy": "expert_batch",
            "policy_params": {"threshold": 128},
        }
        cfg.update(extra)
        p = os.path.join(d, "heter.json")
        with open(p, "w") as f:
            json.dump(cfg, f)
        return p

    def test_target_layers_parsed_when_present(self):
        with tempfile.TemporaryDirectory() as d:
            p = self._write_cfg(d, {"target_layers": [3, 7, 11]})
            cfg = _parse_heter_config(p)
            self.assertEqual(cfg.get("target_layers"), [3, 7, 11])

    def test_target_layers_defaults_to_none(self):
        with tempfile.TemporaryDirectory() as d:
            p = self._write_cfg(d, {})
            cfg = _parse_heter_config(p)
            self.assertIsNone(cfg.get("target_layers"))


if __name__ == "__main__":
    unittest.main()
