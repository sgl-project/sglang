"""transformers 5 Nemotron-H layer names load as the legacy names."""

import unittest

from sglang.srt.configs.nemotron_h import NemotronHConfig
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestNemotronHLayerAliases(unittest.TestCase):
    def test_transformers_5_names_become_legacy_names(self):
        config = NemotronHConfig(
            layers_block_type=["linear_attention", "full_attention", "mlp", "moe"],
            n_routed_experts=2,
            n_group=1,
        )
        self.assertEqual(config.layers_block_type, ["mamba", "attention", "mlp", "moe"])

    def test_legacy_names_are_unchanged(self):
        config = NemotronHConfig(
            layers_block_type=["mamba", "attention"],
        )
        self.assertEqual(config.layers_block_type, ["mamba", "attention"])


if __name__ == "__main__":
    unittest.main()
