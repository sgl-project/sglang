"""K3 wrappers must expose the local layer range to runner buffer sizing."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.model_executor.model_runner_components import misc_utils
from sglang.srt.model_executor.model_runner_components.layer_setup import (
    resolve_layer_indices,
)
from sglang.srt.model_executor.runner_utils.buffers import allocate_pp_proxy_tensors
from sglang.srt.models import kimi_k3
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


class TestKimiK3PipelineLayers(CustomTestCase):
    def _build_text_model(self, start, end):
        with (
            patch.object(
                kimi_k3,
                "KimiK3LinearModel",
                return_value=SimpleNamespace(start_layer=start, end_layer=end),
            ),
            patch.object(
                kimi_k3,
                "get_parallel",
                return_value=SimpleNamespace(
                    pp_group=SimpleNamespace(is_last_rank=False)
                ),
            ),
            patch.object(kimi_k3, "LogitsProcessor"),
        ):
            return kimi_k3.KimiK3LinearForCausalLM(
                SimpleNamespace(vocab_size=16, hidden_size=8)
            )

    def _assert_runner_bank(self, model, arch, start, end):
        text_config = SimpleNamespace(attn_res_block_size=4)
        model_config = SimpleNamespace(
            hf_config=SimpleNamespace(architectures=[arch]),
            hf_text_config=text_config,
            num_hidden_layers=6,
            num_attention_layers=6,
            num_nextn_predict_layers=None,
            swa_attention_layer_ids=None,
        )
        info = resolve_layer_indices(
            model=model, model_config=model_config, is_draft_worker=False
        )
        self.assertEqual((info.start_layer, info.end_layer), (start, end))
        self.assertEqual(info.num_effective_layers, end - start)
        with patch.object(
            misc_utils,
            "get_parallel",
            return_value=SimpleNamespace(pp_size=2, pp_rank=1),
        ):
            blocks = misc_utils.resolve_pp_proxy_residual_num_blocks(
                model_config=model_config, start_layer=info.start_layer
            )
        self.assertEqual(blocks, (start + 3) // 4)
        buffers = allocate_pp_proxy_tensors(
            max_num_tokens=3,
            max_hidden_tokens=3,
            hidden_size=8,
            dtype=torch.float32,
            pp_proxy_residual_num_blocks=blocks,
        )
        self.assertEqual(buffers["attn_res_bank"].shape, (3, blocks, 8))

    def test_text_model_sizes_inherited_bank_at_both_cuts(self):
        for start in (3, 4):
            with self.subTest(start=start):
                self._assert_runner_bank(
                    self._build_text_model(start, 6),
                    "KimiK3LinearForCausalLM",
                    start,
                    6,
                )

    def test_multimodal_model_exposes_language_layer_range(self):
        for start in (3, 4):
            with (
                self.subTest(start=start),
                patch.object(kimi_k3, "KimiK3VisionTower"),
                patch.object(kimi_k3, "KimiK3MultiModalProjector"),
                patch.object(
                    kimi_k3,
                    "KimiK3LinearForCausalLM",
                    return_value=self._build_text_model(start, 6),
                ),
            ):
                model = kimi_k3.KimiK3ForConditionalGeneration(
                    SimpleNamespace(
                        encoder_only=False,
                        vision_config=SimpleNamespace(),
                        text_config=SimpleNamespace(),
                    )
                )
                self._assert_runner_bank(
                    model, "KimiK3ForConditionalGeneration", start, 6
                )

    def test_encoder_only_keeps_existing_layer_range(self):
        with (
            patch.object(kimi_k3, "KimiK3VisionTower"),
            patch.object(kimi_k3, "KimiK3MultiModalProjector"),
        ):
            model = kimi_k3.KimiK3ForConditionalGeneration(
                SimpleNamespace(
                    encoder_only=True,
                    vision_config=SimpleNamespace(),
                    text_config=SimpleNamespace(num_hidden_layers=6),
                )
            )
        self.assertIsNone(model.language_model)
        self.assertEqual((model.start_layer, model.end_layer), (0, 6))


if __name__ == "__main__":
    unittest.main()
