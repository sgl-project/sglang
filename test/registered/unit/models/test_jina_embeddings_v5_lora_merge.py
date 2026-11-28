"""Tests for merging a jina-embeddings-v5 task adapter into the base weights."""

import json
import os
import tempfile
import unittest

import torch
from safetensors.torch import save_file

from sglang.srt.models.jina_embeddings_v5 import _load_lora_deltas
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_Q_PROJ = "layers.0.self_attn.q_proj"


def _write_adapter(adapter_dir, *, lora_a, lora_b, **config):
    save_file(
        {
            f"base_model.model.{_Q_PROJ}.lora_A.weight": lora_a,
            f"base_model.model.{_Q_PROJ}.lora_B.weight": lora_b,
        },
        os.path.join(adapter_dir, "adapter_model.safetensors"),
    )
    with open(os.path.join(adapter_dir, "adapter_config.json"), "w") as f:
        json.dump({"r": 2, "lora_alpha": 8, **config}, f)


class TestJinaEmbeddingsV5LoraMerge(CustomTestCase):
    def test_delta_matches_peft_lora_forward(self):
        """The merged delta must reproduce PEFT's LoRA output, (alpha / r) * B(A(x)),
        under the base checkpoint's weight name."""
        torch.manual_seed(0)
        lora_a = torch.randn(2, 4)  # [r, in_features]
        lora_b = torch.randn(3, 2)  # [out_features, r]
        with tempfile.TemporaryDirectory() as adapter_dir:
            _write_adapter(adapter_dir, lora_a=lora_a, lora_b=lora_b)
            deltas = _load_lora_deltas(adapter_dir)

        self.assertEqual(list(deltas), [f"{_Q_PROJ}.weight"])
        x = torch.randn(5, 4)
        torch.testing.assert_close(
            x @ deltas[f"{_Q_PROJ}.weight"].T, (8 / 2) * (x @ lora_a.T @ lora_b.T)
        )

    def test_rejects_lora_variants_a_plain_merge_gets_wrong(self):
        """DoRA, rsLoRA and fan-in-fan-out adapters need other merge formulas."""
        for flag in ("use_dora", "use_rslora", "fan_in_fan_out"):
            with self.subTest(flag=flag), tempfile.TemporaryDirectory() as adapter_dir:
                _write_adapter(
                    adapter_dir,
                    lora_a=torch.zeros(2, 4),
                    lora_b=torch.zeros(3, 2),
                    **{flag: True},
                )
                with self.assertRaisesRegex(ValueError, flag):
                    _load_lora_deltas(adapter_dir)


if __name__ == "__main__":
    unittest.main()
