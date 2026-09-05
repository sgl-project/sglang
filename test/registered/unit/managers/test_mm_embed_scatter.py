import pytest
import torch
from torch import nn

from sglang.srt.environ import envs
from sglang.srt.managers.mm_utils import _scatter_mm_embedding, embed_mm_inputs
from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
)
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


def test_embed_mm_inputs_isolates_offset_mask_count_mismatch():
    bad_embedding = torch.tensor(
        [[10.0, 11.0, 12.0, 13.0], [20.0, 21.0, 22.0, 23.0], [30.0, 31.0, 32.0, 33.0]]
    )
    good_embedding = torch.tensor([[40.0, 41.0, 42.0, 43.0], [50.0, 51.0, 52.0, 53.0]])
    mm_inputs = [
        MultimodalInputs(
            mm_items=[
                MultimodalDataItem(
                    modality=Modality.IMAGE,
                    pad_value=100,
                    offsets=[(0, 2)],
                    precomputed_embeddings=bad_embedding,
                )
            ]
        ),
        MultimodalInputs(
            mm_items=[
                MultimodalDataItem(
                    modality=Modality.IMAGE,
                    pad_value=101,
                    offsets=[(0, 1)],
                    precomputed_embeddings=good_embedding,
                )
            ]
        ),
    ]
    input_ids = torch.tensor([100, 100, 0, 101, 101, 0])
    input_embedding = nn.Embedding(128, 4)
    input_embedding.weight.data.zero_()

    with envs.SGLANG_ENABLE_ASYNC_ASSERT.override(True):
        actual, other_info = embed_mm_inputs(
            mm_inputs_list=mm_inputs,
            extend_prefix_lens=[0, 0],
            extend_seq_lens=[3, 3],
            input_ids=input_ids,
            input_embedding=input_embedding,
            data_embedding_func_mapping={Modality.IMAGE: lambda _items: None},
        )

    assert other_info["mm_embedding_errors"] == [(0, 3, 2)]
    assert torch.equal(actual[3:5], good_embedding)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
