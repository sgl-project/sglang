import unittest
from types import SimpleNamespace

from sglang.multimodal_gen.runtime.pipelines.ltx_2_pipeline import (
    LTX2TwoStageResidencyController,
)


def _controller(
    mode: str, *, distilled_lora: str | None, stage1_lora: str | None = None
) -> LTX2TwoStageResidencyController:
    controller = object.__new__(LTX2TwoStageResidencyController)
    controller.mode = mode
    controller.pipeline = SimpleNamespace(
        _distilled_lora_path=distilled_lora, _stage1_lora_path=stage1_lora
    )
    return controller


class TestLTX2PremergedStage2(unittest.TestCase):
    def test_resident_mode_premerges_the_distilled_lora(self):
        self.assertTrue(
            _controller("resident", distilled_lora="lora").should_use_premerged
        )

    def test_original_mode_keeps_one_dit(self):
        self.assertFalse(
            _controller("original", distilled_lora="lora").should_use_premerged
        )

    def test_no_distilled_lora_or_a_user_lora_keeps_one_dit(self):
        # a pre-distilled DiT (LTX-2.5) has nothing to merge; a user stage-1 LoRA
        # still switches per request
        self.assertFalse(
            _controller("resident", distilled_lora=None).should_use_premerged
        )
        self.assertFalse(
            _controller(
                "resident", distilled_lora="lora", stage1_lora="user"
            ).should_use_premerged
        )


if __name__ == "__main__":
    unittest.main()
