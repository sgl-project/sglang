import torch

from sglang.multimodal_gen.runtime.models.dits.qwen_image import (
    _repeat_conditioning_to_batch_size,
)


def test_repeat_conditioning_to_multi_output_batch():
    conditioning = torch.tensor([[[1.0], [2.0]]])

    repeated = _repeat_conditioning_to_batch_size(
        conditioning, target_batch_size=3, name="encoder_hidden_states"
    )

    assert repeated.shape == (3, 2, 1)
    assert torch.equal(repeated[0], conditioning[0])
    assert torch.equal(repeated[1], conditioning[0])
    assert torch.equal(repeated[2], conditioning[0])


def test_repeat_conditioning_preserves_per_prompt_output_order():
    conditioning = torch.tensor([[[1.0]], [[2.0]]])

    repeated = _repeat_conditioning_to_batch_size(
        conditioning, target_batch_size=4, name="encoder_hidden_states"
    )

    assert torch.equal(repeated[:, 0, 0], torch.tensor([1.0, 1.0, 2.0, 2.0]))


def test_repeat_conditioning_rejects_incompatible_batch_size():
    conditioning = torch.zeros(2, 1, 1)

    try:
        _repeat_conditioning_to_batch_size(
            conditioning, target_batch_size=3, name="encoder_hidden_states"
        )
    except ValueError as e:
        assert "cannot expand" in str(e)
    else:
        raise AssertionError("Expected ValueError for incompatible batch size")
