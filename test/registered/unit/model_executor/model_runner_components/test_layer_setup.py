"""Unit tests for model-runner layer discovery."""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.distributed.utils import get_pp_indices
from sglang.srt.model_executor.model_runner_components.layer_setup import (
    compute_attention_and_moe_layers,
    resolve_layer_indices,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=6, suite="base-a-test-cpu")


class TestComputeAttentionAndMoeLayers(CustomTestCase):
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


NUM_LAYERS = 36
GLOBAL_FULL_IDS = list(range(1, NUM_LAYERS, 2))
GLOBAL_SWA_IDS = list(range(0, NUM_LAYERS, 2))


def _hybrid_swa_model_config(*, architecture="Hybrid"):
    return SimpleNamespace(
        hf_config=SimpleNamespace(architectures=[architecture]),
        num_hidden_layers=NUM_LAYERS,
        num_attention_layers=NUM_LAYERS,
        num_nextn_predict_layers=None,
        is_hybrid_swa=True,
        full_attention_layer_ids=list(GLOBAL_FULL_IDS),
        swa_attention_layer_ids=list(GLOBAL_SWA_IDS),
    )


class TestResolveHybridSWALayerIds(CustomTestCase):
    def test_pipeline_stages_own_each_hybrid_layer_once(self):
        """Every hybrid layer belongs to exactly one pipeline stage."""
        for pp_size in (1, 8):
            with self.subTest(pp_size=pp_size), patch.dict(os.environ):
                os.environ.pop("SGLANG_PP_LAYER_PARTITION", None)
                full_ids = []
                swa_ids = []
                for rank in range(pp_size):
                    start, end = get_pp_indices(NUM_LAYERS, rank, pp_size)
                    model = SimpleNamespace(start_layer=start, end_layer=end)
                    layer_info = resolve_layer_indices(
                        model=model,
                        model_config=_hybrid_swa_model_config(),
                        is_draft_worker=False,
                    )
                    self.assertEqual(
                        sorted(
                            layer_info.full_attention_layer_ids
                            + layer_info.swa_attention_layer_ids
                        ),
                        list(range(start, end)),
                    )
                    full_ids.extend(layer_info.full_attention_layer_ids)
                    swa_ids.extend(layer_info.swa_attention_layer_ids)

                self.assertEqual(full_ids, GLOBAL_FULL_IDS)
                self.assertEqual(swa_ids, GLOBAL_SWA_IDS)

    def test_model_config_is_shared_and_never_modified(self):
        """Resolving one runner's view must not change the shared ModelConfig."""
        model_config = _hybrid_swa_model_config()
        for start, end in ((0, 4), (4, 9), (0, 1)):
            resolve_layer_indices(
                model=SimpleNamespace(start_layer=start, end_layer=end),
                model_config=model_config,
                is_draft_worker=False,
            )
            self.assertEqual(model_config.full_attention_layer_ids, GLOBAL_FULL_IDS)
            self.assertEqual(model_config.swa_attention_layer_ids, GLOBAL_SWA_IDS)

    def test_multi_layer_mtp_draft_owns_its_depth(self):
        """Each MTP draft runner owns the depth it serves, not layer 0."""
        model_config = _hybrid_swa_model_config(
            architecture="InklingForConditionalGenerationMTP"
        )
        model_config.swa_attention_layer_ids = [0, 2]
        model_config.full_attention_layer_ids = [1]
        model_config.num_nextn_predict_layers = 3

        for depth, swa, full in ((0, [0], []), (1, [], [1]), (2, [2], [])):
            with self.subTest(depth=depth):
                layer_info = resolve_layer_indices(
                    model=SimpleNamespace(mtp_layer_id_is_depth=True),
                    model_config=model_config,
                    is_draft_worker=True,
                    draft_model_idx=depth,
                )
                self.assertTrue(layer_info.is_hybrid_swa_mtp_draft)
                self.assertEqual(layer_info.swa_attention_layer_ids, swa)
                self.assertEqual(layer_info.full_attention_layer_ids, full)

    def test_single_block_mtp_draft_keeps_layer_range_view(self):
        """A draft whose block is always layer 0 is an ordinary [0, 1) slice."""
        model_config = _hybrid_swa_model_config(architecture="MiMoV2MTP")
        model_config.swa_attention_layer_ids = [0]
        model_config.full_attention_layer_ids = []

        layer_info = resolve_layer_indices(
            model=SimpleNamespace(),
            model_config=model_config,
            is_draft_worker=True,
            draft_model_idx=1,
        )

        self.assertFalse(layer_info.is_hybrid_swa_mtp_draft)
        self.assertEqual(layer_info.swa_attention_layer_ids, [0])
        self.assertEqual(layer_info.full_attention_layer_ids, [])


if __name__ == "__main__":
    unittest.main()
