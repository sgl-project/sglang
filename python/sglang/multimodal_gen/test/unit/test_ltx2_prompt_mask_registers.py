import unittest
from types import SimpleNamespace

import torch

from sglang.multimodal_gen.runtime.models.adapter.ltx_2_connector import (
    LTX2TextConnectors,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.ltx_2.text_connector import (
    LTX2TextConnectorStage,
)


class _Connectors:
    def __init__(self, *, registers: bool):
        self.output_mask_is_all_valid = registers

    def __call__(self, embeds, mask, additive_mask):
        return embeds, embeds * 2, torch.ones(embeds.shape[:2], dtype=torch.int64)


def _run(*, quality: str, registers: bool, cfg: bool = True):
    stage = object.__new__(LTX2TextConnectorStage)
    stage.connectors = _Connectors(registers=registers)
    mask = torch.ones(1, 4, dtype=torch.int64)
    mask[:, 3:] = 0
    batch = SimpleNamespace(
        prompt_embeds=[torch.randn(1, 4, 8)],
        prompt_attention_mask=mask,
        negative_prompt_embeds=[torch.randn(1, 4, 8)],
        negative_attention_mask=mask,
        do_classifier_free_guidance=cfg,
        quality=quality,
    )
    return stage.forward(batch, server_args=None)


class TestLTX2RegisteredPromptMask(unittest.TestCase):
    def test_lossless_drops_the_all_valid_mask_of_a_registered_connector(self):
        for cfg in (True, False):
            batch = _run(quality="lossless", registers=True, cfg=cfg)
            self.assertIsNone(batch.prompt_attention_mask)
            self.assertIsNone(batch.negative_attention_mask)

    def test_exact_keeps_the_reference_mask(self):
        batch = _run(quality="exact", registers=True)
        all_valid = torch.ones(1, 4, dtype=torch.int64)
        self.assertTrue(torch.equal(batch.prompt_attention_mask, all_valid))
        self.assertTrue(torch.equal(batch.negative_attention_mask, all_valid))

    def test_registers_make_the_connector_mask_all_valid(self):
        for registers, all_valid in ((torch.zeros(2, 8), True), (None, False)):
            connectors = SimpleNamespace(
                video_connector=SimpleNamespace(learnable_registers=registers)
            )
            self.assertEqual(
                LTX2TextConnectors.output_mask_is_all_valid.fget(connectors), all_valid
            )

    def test_connector_without_registers_keeps_its_mask(self):
        batch = _run(quality="lossless", registers=False)
        self.assertIsNotNone(batch.prompt_attention_mask)
        self.assertIsNotNone(batch.negative_attention_mask)


if __name__ == "__main__":
    unittest.main()
