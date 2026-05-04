# SPDX-License-Identifier: Apache-2.0

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from sglang.srt.ug.bagel_checkpoint import (
    BAGEL_KEY_CATEGORY_BAGEL_OUTER,
    BAGEL_KEY_CATEGORY_MOT_GEN_BRANCH,
    BAGEL_KEY_CATEGORY_QWEN2_SHARED,
    BAGEL_KEY_CATEGORY_UNKNOWN,
    BAGEL_KEY_CATEGORY_VIT_VAE,
    classify_bagel_checkpoint_key,
    format_bagel_checkpoint_key_summary,
    load_bagel_checkpoint_keys,
    summarize_bagel_checkpoint_keys,
)


class TestBAGELCheckpointKeyClassifier(unittest.TestCase):
    def test_classifies_bagel_qwen2_mot_keys(self):
        cases = {
            "language_model.model.embed_tokens.weight": (
                BAGEL_KEY_CATEGORY_QWEN2_SHARED
            ),
            "language_model.model.layers.0.self_attn.q_proj.weight": (
                BAGEL_KEY_CATEGORY_QWEN2_SHARED
            ),
            "language_model.model.layers.0.self_attn.q_norm.weight": (
                BAGEL_KEY_CATEGORY_QWEN2_SHARED
            ),
            "language_model.model.layers.0.self_attn.q_proj_moe_gen.weight": (
                BAGEL_KEY_CATEGORY_MOT_GEN_BRANCH
            ),
            "language_model.model.layers.0.mlp_moe_gen.down_proj.weight": (
                BAGEL_KEY_CATEGORY_MOT_GEN_BRANCH
            ),
            "language_model.model.norm_moe_gen.weight": (
                BAGEL_KEY_CATEGORY_MOT_GEN_BRANCH
            ),
            "time_embedder.mlp.0.weight": BAGEL_KEY_CATEGORY_BAGEL_OUTER,
            "vae2llm.weight": BAGEL_KEY_CATEGORY_BAGEL_OUTER,
            "connector.fc1.weight": BAGEL_KEY_CATEGORY_BAGEL_OUTER,
            "vit_model.vision_model.encoder.layers.0.self_attn.q_proj.weight": (
                BAGEL_KEY_CATEGORY_VIT_VAE
            ),
            "decoder.conv_in.weight": BAGEL_KEY_CATEGORY_VIT_VAE,
            "quant_conv.weight": BAGEL_KEY_CATEGORY_VIT_VAE,
            "some_future_module.weight": BAGEL_KEY_CATEGORY_UNKNOWN,
        }

        for key, expected in cases.items():
            with self.subTest(key=key):
                self.assertEqual(classify_bagel_checkpoint_key(key), expected)

    def test_summarizes_counts_and_examples(self):
        summary = summarize_bagel_checkpoint_keys(
            [
                "language_model.model.layers.0.self_attn.q_proj.weight",
                "language_model.model.layers.0.self_attn.q_proj_moe_gen.weight",
                "time_embedder.mlp.0.weight",
                "vit_model.vision_model.post_layernorm.weight",
                "unknown.weight",
                "another_unknown.weight",
            ],
            max_examples_per_category=1,
        )

        self.assertEqual(summary.total, 6)
        self.assertEqual(summary.counts[BAGEL_KEY_CATEGORY_QWEN2_SHARED], 1)
        self.assertEqual(summary.counts[BAGEL_KEY_CATEGORY_MOT_GEN_BRANCH], 1)
        self.assertEqual(summary.counts[BAGEL_KEY_CATEGORY_BAGEL_OUTER], 1)
        self.assertEqual(summary.counts[BAGEL_KEY_CATEGORY_VIT_VAE], 1)
        self.assertEqual(summary.counts[BAGEL_KEY_CATEGORY_UNKNOWN], 2)
        self.assertEqual(
            summary.examples[BAGEL_KEY_CATEGORY_UNKNOWN],
            ("unknown.weight",),
        )
        self.assertAlmostEqual(summary.ratio(BAGEL_KEY_CATEGORY_UNKNOWN), 2 / 6)

    def test_formats_summary(self):
        summary = summarize_bagel_checkpoint_keys(
            ["language_model.model.layers.0.self_attn.q_proj.weight"]
        )

        formatted = format_bagel_checkpoint_key_summary(summary)

        self.assertIn("total: 1", formatted)
        self.assertIn("qwen2_shared: 1 (100.00%)", formatted)
        self.assertIn(
            "language_model.model.layers.0.self_attn.q_proj.weight",
            formatted,
        )

    def test_loads_keys_from_hf_safetensors_index(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = Path(tmpdir) / "model.safetensors.index.json"
            index_path.write_text(
                json.dumps(
                    {
                        "metadata": {"total_size": 1},
                        "weight_map": {
                            "language_model.model.embed_tokens.weight": "a.safetensors",
                            "time_embedder.mlp.0.weight": "b.safetensors",
                        },
                    }
                ),
                encoding="utf-8",
            )

            keys = load_bagel_checkpoint_keys(index_path)

        self.assertEqual(
            keys,
            (
                "language_model.model.embed_tokens.weight",
                "time_embedder.mlp.0.weight",
            ),
        )

    def test_loads_index_keys_from_checkpoint_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir)
            index_path = checkpoint_dir / "ema.safetensors.index.json"
            index_path.write_text(
                json.dumps(
                    {
                        "weight_map": {
                            "vit_model.vision_model.head.weight": "ema-1.safetensors"
                        }
                    }
                ),
                encoding="utf-8",
            )

            keys = load_bagel_checkpoint_keys(checkpoint_dir)

        self.assertEqual(keys, ("vit_model.vision_model.head.weight",))

    @unittest.skipUnless(
        importlib.util.find_spec("safetensors") and importlib.util.find_spec("torch"),
        "requires safetensors and torch",
    )
    def test_loads_keys_from_single_safetensors_file(self):
        import torch
        from safetensors.torch import save_file

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "ema.safetensors"
            save_file(
                {
                    "language_model.lm_head.weight": torch.zeros(1, 1),
                    "connector.fc1.weight": torch.zeros(1, 1),
                },
                checkpoint_path,
            )

            keys = load_bagel_checkpoint_keys(checkpoint_path)

        self.assertEqual(
            keys,
            ("connector.fc1.weight", "language_model.lm_head.weight"),
        )


if __name__ == "__main__":
    unittest.main()
