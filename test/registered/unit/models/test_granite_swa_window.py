"""Granite SWA: the runner's sliding window must equal the sliding layers' window,
or backends that bound SWA KV reads by the runner attend to one extra token."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace

from sglang.srt.model_executor.model_runner_components.load_model_utils import (
    resolve_sliding_window_size,
)
from sglang.srt.models.granite import GraniteSWAForCausalLM, granite_layer_attn_params
from sglang.srt.models.granitemoe import GraniteMoeSWAForCausalLM
from sglang.test.test_utils import CustomTestCase


class TestGraniteSWAWindow(CustomTestCase):
    def test_runner_window_matches_the_sliding_layers(self):
        for cls, model_type in (
            (GraniteSWAForCausalLM, "granite_swa"),
            (GraniteMoeSWAForCausalLM, "granitemoe_swa"),
        ):
            with self.subTest(cls=cls.__name__):
                config = SimpleNamespace(
                    model_type=model_type,
                    sliding_window=129,
                    layer_types=["full_attention", "sliding_attention"],
                    layer_rope_theta=[10000.0, 10000.0],
                )
                model = object.__new__(cls)
                object.__setattr__(model, "config", config)
                model_config = SimpleNamespace(
                    is_hybrid_swa=True,
                    sliding_window_size=config.sliding_window,
                    attention_chunk_size=None,
                )
                full, sliding = (
                    granite_layer_attn_params(config, layer_id)[0]
                    for layer_id in range(2)
                )
                self.assertEqual(full, -1)
                self.assertEqual(sliding, 128)
                self.assertEqual(
                    resolve_sliding_window_size(model, model_config), sliding
                )


if __name__ == "__main__":
    unittest.main()
