from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sglang.srt.managers.mm_utils import _scatter_mm_embedding, embed_mm_inputs
from sglang.srt.managers.schedule_batch import Modality
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-test-cpu-intel")

NUM_TOKENS = 64


def _make_mask(pattern: str) -> torch.Tensor:
    mask = torch.zeros(NUM_TOKENS, dtype=torch.bool)
    if pattern == "interleaved":
        mask[::3] = True
    elif pattern == "blocks":
        mask[5:20] = True
        mask[40:41] = True
    elif pattern == "all_true":
        mask[:] = True
    return mask.unsqueeze(-1)


@pytest.mark.parametrize("width", [8, 24])
@pytest.mark.parametrize("src_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize(
    "mask_pattern", ["interleaved", "blocks", "all_true", "all_false"]
)
def test_scatter_matches_masked_scatter_bitwise(width, src_dtype, mask_pattern):
    """The row-index mm embedding merge must stay bitwise identical to
    masked_scatter_ semantics, whose internal transients it avoids."""
    torch.manual_seed(0)
    mask = _make_mask(mask_pattern)
    dest = torch.randn(NUM_TOKENS, width).to(torch.bfloat16)
    src = torch.randn(int(mask.sum()), width, dtype=src_dtype)

    expected = dest.clone()
    expected.masked_scatter_(mask.expand_as(expected), src.to(expected.dtype))

    actual = dest.clone()
    _scatter_mm_embedding(dest=actual, mask=mask, src=src)
    assert torch.equal(actual, expected)


def test_scatter_row_count_mismatch_fails_loud():
    """A mask/src row-count mismatch must raise, not silently corrupt rows."""
    dest = torch.zeros(8, 4)
    src_short_mask = _make_mask("all_false")[:8]
    src_short_mask[1] = True
    with pytest.raises((RuntimeError, IndexError)):
        _scatter_mm_embedding(dest=dest, mask=src_short_mask, src=torch.ones(3, 4))
    mask_heavy = src_short_mask.clone()
    mask_heavy[2:6] = True
    with pytest.raises((RuntimeError, IndexError)):
        _scatter_mm_embedding(dest=dest, mask=mask_heavy, src=torch.ones(1, 4))


@pytest.mark.parametrize("as_view", [False, True], ids=["owned", "view"])
def test_embedding_preserves_reusable_multimodal_input_ids(as_view):
    pad = 1_000_001
    original = torch.tensor([3, 1, pad, pad, 2, 4])
    original_before = original.clone()
    ids = original[1:5] if as_view else original[1:5].clone()
    before = ids.clone()
    item = SimpleNamespace(
        pad_value=pad,
        offsets=[(1, 2)],
        is_modality=lambda modality: modality == Modality.IMAGE,
    )
    embedding = torch.nn.Embedding(16, 4)

    def image_embedding(**kwargs):
        return (
            torch.full((2, 4), 42.0),
            kwargs["input_ids"].eq(pad).unsqueeze(-1),
            kwargs["input_ids"],
        )

    with patch(
        "sglang.srt.managers.mm_utils.get_embedding_and_mask",
        side_effect=image_embedding,
    ):
        for _ in range(2):
            result, _ = embed_mm_inputs(
                mm_inputs_list=[SimpleNamespace(mm_items=[item])],
                extend_prefix_lens=[0],
                extend_seq_lens=[4],
                input_ids=ids,
                input_embedding=embedding,
                multimodal_model=SimpleNamespace(get_image_feature=lambda items: None),
            )
            assert torch.equal(ids, before)
            assert torch.equal(original, original_before)
            torch.testing.assert_close(result[1:3], torch.full((2, 4), 42.0))
            torch.testing.assert_close(result[[0, 3]], embedding(before[[0, 3]]))


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
