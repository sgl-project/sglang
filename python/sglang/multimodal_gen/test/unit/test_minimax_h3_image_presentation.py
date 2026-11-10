# SPDX-License-Identifier: Apache-2.0
"""Image presentations and native CPU mRoPE, without downloading weights."""

from types import SimpleNamespace

import pytest
import torch

from sglang.multimodal_gen.runtime.models.encoders.qwen3vl import Qwen3VLModel
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.presentation import (
    minimax_h3_image_presentation,
)


class CharacterTokenizer:
    """One id per character plus the three vision delimiters, enough to check layout."""

    SPECIAL = {"<|vision_start|>": 10, "<|image_pad|>": 11, "<|vision_end|>": 12}

    def __call__(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        return {"input_ids": [100 + ord(c) for c in text]}

    def convert_tokens_to_ids(self, token):
        return self.SPECIAL[token]


class NativeRopeOnly:
    """Reuse the real native methods, with their required config and no weights."""

    get_rope_index = Qwen3VLModel.get_rope_index
    _get_text_rope_index = Qwen3VLModel._get_text_rope_index

    def __init__(self):
        self.config = SimpleNamespace(
            vision_config=SimpleNamespace(spatial_merge_size=2),
            image_token_id=11,
            video_token_id=13,
            vision_start_token_id=10,
        )


def _present(prompt, counts=(), grid=None):
    return minimax_h3_image_presentation(
        CharacterTokenizer(),
        prompt=prompt,
        image_token_counts=list(counts),
        get_rope_index=NativeRopeOnly().get_rope_index,
        image_grid_thw=grid,
    )


@pytest.mark.parametrize(
    "text",
    [" ", "  abc\n", "x" * 4096, "x" * 5000],
    ids=["space", "whitespace", "limit", "truncated"],
)
def test_plain_text_verbatim_truncation_and_padding(text):
    ids, tags, pos = _present(text)
    n = min(len(text), 4096)
    assert ids.dtype == torch.int64 and ids.device.type == "cpu"
    assert ids.tolist() == [100 + ord(c) for c in text[:n]] + [0] * (-n % 8)
    assert tags.tolist() == [1] * n
    assert pos.dtype == torch.int64 and pos.shape == (3, 1, len(ids))
    assert torch.equal(pos, torch.arange(len(ids)).view(1, 1, -1).expand(3, 1, -1))


def test_empty_text_has_no_rows():
    ids, tags, pos = _present("")
    assert ids.numel() == 0 and tags.numel() == 0 and pos is None


@pytest.mark.parametrize(
    "prompt", ["", "The picture\n", "z" * 5000], ids=["empty", "plain", "truncated"]
)
def test_multi_image_order_labels_delimiters_tags_and_padding(prompt):
    grid = torch.tensor([[1, 2, 4], [1, 4, 6]], dtype=torch.int32)
    ids, tags, pos = _present(prompt, [2, 6], grid)
    expected, expected_tags = [], []
    for i, count in enumerate([2, 6], 1):
        label = [100 + ord(c) for c in f"Picture {i}: "]
        expected += label + [10] + [11] * count + [12]
        expected_tags += [1] * len(label) + [0] * (count + 2)
    expected += [100 + ord(c) for c in prompt[:4096]]
    expected_tags += [1] * min(len(prompt), 4096)
    text_len = len(expected)
    expected += [0] * (-len(expected) % 8)
    assert ids.tolist() == expected
    assert tags.tolist() == expected_tags
    raw, _ = NativeRopeOnly().get_rope_index(ids[None, :text_len], image_grid_thw=grid)
    assert torch.equal(pos[..., :text_len], raw)
    assert torch.equal(
        pos[..., text_len:], raw[..., -1:].expand(3, 1, len(expected) - text_len)
    )
    # First image starts after 11 label characters and the start delimiter.
    assert pos[:, 0, 12:14].tolist() == [[12, 12], [12, 12], [12, 13]]
