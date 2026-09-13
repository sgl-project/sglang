# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.multimodal_gen.runtime.pipelines_core.stages.text_encoding import (
    TextEncodingStage,
)

_GLOBAL_ARGS_PATCH = (
    "sglang.multimodal_gen.runtime.pipelines_core.stages.base.get_global_server_args"
)

BATCH = 2
SEQ_POS = 45
SEQ_NEG = 7
DIM = 16


def make_stage() -> TextEncodingStage:
    with patch(_GLOBAL_ARGS_PATCH) as mock_global_args:
        mock_global_args.return_value = MagicMock()
        return TextEncodingStage(text_encoders=[], tokenizers=[])


def make_batch() -> SimpleNamespace:
    return SimpleNamespace(
        negative_prompt_embeds=[],
        negative_attention_mask=None,
        negative_prompt_embeds_mask=None,
        negative_prompt_seq_lens=None,
        neg_pooled_embeds=[],
    )


def append_negative(stage, *, positive_embeds, neg_embeds, neg_mask):
    batch = make_batch()
    stage._append_negative_text_outputs(
        batch,
        [positive_embeds],
        [neg_embeds],
        [neg_mask],
        [],
        [neg_mask],
        [[SEQ_NEG]],
    )
    return batch


def test_shared_negative_mask_stays_two_dimensional():
    stage = make_stage()
    batch = append_negative(
        stage,
        positive_embeds=torch.zeros(BATCH, SEQ_POS, DIM),
        neg_embeds=torch.zeros(1, SEQ_NEG, DIM),
        neg_mask=torch.ones(1, SEQ_NEG, dtype=torch.int64),
    )

    for name in ("negative_attention_mask", "negative_prompt_embeds_mask"):
        mask = getattr(batch, name)[0]
        assert mask.shape == (BATCH, SEQ_NEG), f"{name} became {tuple(mask.shape)}"
        assert mask.sum(dim=1).tolist() == [SEQ_NEG] * BATCH


def test_batchless_negative_embeddings_still_gain_a_batch_axis():
    stage = make_stage()
    batch = append_negative(
        stage,
        positive_embeds=torch.zeros(BATCH, SEQ_POS, DIM),
        neg_embeds=torch.zeros(SEQ_NEG, DIM),
        neg_mask=torch.ones(1, SEQ_NEG, dtype=torch.int64),
    )
    assert batch.negative_prompt_embeds[0].shape == (BATCH, SEQ_NEG, DIM)


def test_single_request_negative_conditioning_is_unchanged():
    stage = make_stage()
    batch = append_negative(
        stage,
        positive_embeds=torch.zeros(1, SEQ_POS, DIM),
        neg_embeds=torch.zeros(1, SEQ_NEG, DIM),
        neg_mask=torch.ones(1, SEQ_NEG, dtype=torch.int64),
    )
    assert batch.negative_prompt_embeds[0].shape == (1, SEQ_NEG, DIM)
    assert batch.negative_attention_mask[0].shape == (1, SEQ_NEG)
