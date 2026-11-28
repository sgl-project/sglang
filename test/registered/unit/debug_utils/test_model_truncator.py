import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import torch
from huggingface_hub import save_torch_state_dict
from safetensors.torch import save_file

from sglang.srt.debug_utils.model_truncator import main
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestModelTruncator(CustomTestCase):
    def test_index_size_counts_only_retained_weights(self):
        """The index should report the bytes of the weights it still references."""
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder)
            output = source / "truncated"
            (source / "config.json").write_text('{"num_hidden_layers": 3}')
            for name in ("generation_config.json", "tokenizer.json", "modeling.py"):
                (source / name).touch()
            weights = {
                "model.layers.0.weight": torch.zeros(3, dtype=torch.float32),
                "model.layers.1.weight": torch.zeros(4, dtype=torch.int16),
                "model.layers.2.weight": torch.zeros(7, dtype=torch.uint8),
            }
            save_torch_state_dict(weights, source, max_shard_size=12)
            save_file(
                {"model.layers.0.weight": weights["model.layers.0.weight"]},
                source / "other.safetensors",
            )

            main(Namespace(input=source, output=output, keep_num_layers=2))

            index = json.loads((output / "model.safetensors.index.json").read_text())
            self.assertEqual(
                set(index["weight_map"]),
                {"model.layers.0.weight", "model.layers.1.weight"},
            )
            self.assertEqual(index["metadata"]["total_size"], 20)


if __name__ == "__main__":
    unittest.main()
