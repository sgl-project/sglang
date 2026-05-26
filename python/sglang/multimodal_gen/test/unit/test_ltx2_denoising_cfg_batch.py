import pytest
import torch

from sglang.multimodal_gen.runtime.pipelines_core.stages.ltx_2_denoising import (
    LTX2DenoisingStage,
)


def test_ltx2_condition_batch_expands_per_prompt_outputs_in_order():
    condition = torch.tensor([[[1.0]], [[2.0]]])

    expanded = LTX2DenoisingStage._expand_ltx2_condition_batch(
        condition,
        6,
        name="encoder_hidden_states",
    )

    expected = torch.tensor([[[1.0]], [[1.0]], [[1.0]], [[2.0]], [[2.0]], [[2.0]]])
    assert torch.equal(expanded, expected)


def test_ltx2_cfg_condition_batch_expands_each_branch_before_concat():
    negative = torch.tensor([[[10.0]], [[20.0]]])
    positive = torch.tensor([[[1.0]], [[2.0]]])

    cfg_condition = LTX2DenoisingStage._cat_ltx2_cfg_condition_batch(
        negative,
        positive,
        6,
        name="encoder_hidden_states",
    )

    expected = torch.cat(
        [
            negative.repeat_interleave(3, dim=0),
            positive.repeat_interleave(3, dim=0),
        ],
        dim=0,
    )
    assert torch.equal(cfg_condition, expected)
    assert cfg_condition.shape[0] == 12


def test_ltx2_optional_cfg_condition_batch_handles_absent_masks():
    assert (
        LTX2DenoisingStage._cat_ltx2_optional_cfg_condition_batch(
            None,
            None,
            4,
            name="encoder_attention_mask",
        )
        is None
    )


def test_ltx2_condition_batch_rejects_unaligned_prompt_count():
    condition = torch.zeros(3, 2)

    with pytest.raises(ValueError, match="encoder_hidden_states batch dimension"):
        LTX2DenoisingStage._expand_ltx2_condition_batch(
            condition,
            4,
            name="encoder_hidden_states",
        )
