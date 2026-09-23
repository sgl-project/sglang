"""Unit tests for model-runner layer discovery."""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.distributed.utils import get_pp_indices
from sglang.srt.model_executor.model_runner_components.layer_setup import (
    adjust_hybrid_swa_layer_ids,
    compute_attention_and_moe_layers,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=6, suite="base-a-test-cpu")


class TestComputeAttentionAndMoeLayers(unittest.TestCase):
    def test_deepseek_mla_registers_mha_companion(self):
        attn_mqa = SimpleNamespace()
        attn_mha = SimpleNamespace()
        layer_model = SimpleNamespace(
            layers=[
                SimpleNamespace(
                    self_attn=SimpleNamespace(attn_mqa=attn_mqa, attn_mha=attn_mha)
                )
            ]
        )

        attention_layers, _, _, _, mha_companion_layers = (
            compute_attention_and_moe_layers(layer_model)
        )

        self.assertEqual(attention_layers, [attn_mqa])
        self.assertEqual(mha_companion_layers, [attn_mha])
        self.assertNotIn("_pcg_mha_companion", vars(attn_mqa))

    def test_pipeline_placeholders_preserve_global_layer_ids(self):
        local_attention = SimpleNamespace()
        layer_model = SimpleNamespace(
            layers=[SimpleNamespace(), SimpleNamespace()]
            + [SimpleNamespace(self_attn=SimpleNamespace(attn=local_attention))]
        )

        attention_layers, _, _, _, mha_companion_layers = (
            compute_attention_and_moe_layers(layer_model)
        )

        self.assertEqual(attention_layers, [None, None, local_attention])
        self.assertEqual(mha_companion_layers, [None, None, None])


class TestAdjustHybridSWALayerIds(unittest.TestCase):
    def test_pipeline_stages_own_each_hybrid_layer_once(self):
        """Each pipeline stage lists only its own layers as full or sliding-window
        attention layers, so every layer belongs to exactly one stage."""
        for pp_size in (1, 8):
            with self.subTest(pp_size=pp_size), patch.dict(os.environ):
                os.environ.pop("SGLANG_PP_LAYER_PARTITION", None)
                full_ids = []
                swa_ids = []
                for rank in range(pp_size):
                    start, end = get_pp_indices(36, rank, pp_size)
                    config = SimpleNamespace(
                        full_attention_layer_ids=list(range(1, 36, 2)),
                        swa_attention_layer_ids=list(range(0, 36, 2)),
                        is_deepseek_v4_arch=False,
                    )
                    adjust_hybrid_swa_layer_ids(
                        model_config=config,
                        start_layer=start,
                        end_layer=end,
                        is_hybrid_swa=True,
                    )
                    self.assertEqual(
                        sorted(
                            config.full_attention_layer_ids
                            + config.swa_attention_layer_ids
                        ),
                        list(range(start, end)),
                    )
                    full_ids.extend(config.full_attention_layer_ids)
                    swa_ids.extend(config.swa_attention_layer_ids)

                self.assertEqual(full_ids, list(range(1, 36, 2)))
                self.assertEqual(swa_ids, list(range(0, 36, 2)))


if __name__ == "__main__":
    unittest.main()
