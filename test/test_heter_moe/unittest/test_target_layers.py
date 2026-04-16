"""Unit test: bf16_only_experts_file parsing in heter config."""
import json
import os
import tempfile
import unittest

from sglang.srt.layers.moe.heter_moe import _parse_heter_config


class TestBf16OnlyExperts(unittest.TestCase):
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

    def test_bf16_only_parsed_when_present(self):
        with tempfile.TemporaryDirectory() as d:
            bf16_file = os.path.join(d, "bf16_only.json")
            with open(bf16_file, "w") as f:
                json.dump({"0": [0, 1, 2], "1": [3, 4, 5]}, f)
            p = self._write_cfg(d, {"bf16_only_experts_file": bf16_file})
            cfg = _parse_heter_config(p)
            self.assertEqual(cfg["_bf16_only_by_layer"], {"0": [0, 1, 2], "1": [3, 4, 5]})

    def test_bf16_only_defaults_to_empty(self):
        with tempfile.TemporaryDirectory() as d:
            p = self._write_cfg(d, {})
            cfg = _parse_heter_config(p)
            self.assertEqual(cfg.get("_bf16_only_by_layer"), {})

    def test_int4_only_still_works(self):
        with tempfile.TemporaryDirectory() as d:
            int4_file = os.path.join(d, "int4_only.json")
            with open(int4_file, "w") as f:
                json.dump({"0": [10, 20]}, f)
            p = self._write_cfg(d, {"int4_only_experts_file": int4_file})
            cfg = _parse_heter_config(p)
            self.assertEqual(cfg["_int4_only_by_layer"], {"0": [10, 20]})


if __name__ == "__main__":
    unittest.main()
