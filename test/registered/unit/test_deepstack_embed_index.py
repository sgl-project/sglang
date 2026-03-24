"""Unit test for deepstack_embeddings indexing in embed_mm_inputs.

Reproduces the IndexError when prefix caching causes one modality's
embedding to be None while another's is not, and use_deepstack is
enabled for both.
"""

from unittest.mock import patch

import torch
import torch.nn as nn

from sglang.srt.managers.mm_utils import embed_mm_inputs
from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
)

# Pad values for image and video (must be > vocab_size to avoid collision)
IMAGE_PAD = 1_000_100
VIDEO_PAD = 1_000_200
HIDDEN = 8
VOCAB = 32


def _make_mm_inputs(
    image_offsets: list[tuple[int, int]],
    video_offsets: list[tuple[int, int]],
) -> MultimodalInputs:
    """Create a MultimodalInputs with one image item and one video item."""
    image_item = MultimodalDataItem(
        modality=Modality.IMAGE,
        pad_value=IMAGE_PAD,
        offsets=image_offsets,
        feature=torch.zeros(1),  # dummy
    )
    video_item = MultimodalDataItem(
        modality=Modality.VIDEO,
        pad_value=VIDEO_PAD,
        offsets=video_offsets,
        feature=torch.zeros(1),  # dummy
    )
    return MultimodalInputs(
        mm_items=[image_item, video_item],
        im_token_id=IMAGE_PAD,
        video_token_id=VIDEO_PAD,
    )


def _mock_get_embedding_and_mask(cached_modality: Modality, embed_width: int = HIDDEN):
    """Return a mock that returns None for the cached modality, real tensors for others.

    embed_width should be HIDDEN * (1 + num_deepstack_layers) when deepstack
    is active, since separate_deepstack_embeds splits the wide embedding.
    """

    def mock_fn(
        data_embedding_func,
        embedding_items,
        placeholder_tensor,
        input_ids,
        items_size,
        prefix_length,
        extend_length,
        items_offset_list,
    ):
        modality = embedding_items[0].modality
        if modality == cached_modality:
            return None, None, input_ids

        # Return a real embedding + mask for the non-cached modality.
        # Width must match what the real embedder would produce (wide when
        # deepstack is active, since separate_deepstack_embeds splits it).
        mask = (input_ids == placeholder_tensor[0]).unsqueeze(-1)
        n_tokens = mask.sum().item()
        embedding = torch.ones(n_tokens, embed_width)
        return embedding, mask, input_ids

    return mock_fn


def _make_multimodal_model(num_deepstack_layers: int):
    """Create a mock multimodal model with deepstack support."""
    model = nn.Module()
    model.deepstack_visual_indexes = list(range(num_deepstack_layers))

    def separate_deepstack_embeds(embedding):
        base = embedding[:, :HIDDEN]
        deepstack = embedding[:, HIDDEN:]
        return base, deepstack

    # When deepstack is active, the embedding func returns wider tensors
    width = HIDDEN * (1 + num_deepstack_layers)

    def get_image_feature(items):
        total = sum(1 for item in items for _ in item.offsets)
        return torch.ones(total, width)

    def get_video_feature(items):
        total = sum(1 for item in items for _ in item.offsets)
        return torch.ones(total, width)

    model.separate_deepstack_embeds = separate_deepstack_embeds
    model.get_image_feature = get_image_feature
    model.get_video_feature = get_video_feature
    return model


class TestDeepstackEmbedIndex:
    """Tests for deepstack_embeddings indexing in embed_mm_inputs."""

    def _build_input_ids(self, img_tokens: int, vid_tokens: int) -> torch.Tensor:
        """Build input_ids: [text...] [IMAGE_PAD × img] [text...] [VIDEO_PAD × vid] [text...]"""
        text = list(range(10))  # 10 text tokens (within vocab)
        ids = text + [IMAGE_PAD] * img_tokens + text + [VIDEO_PAD] * vid_tokens + text
        return torch.tensor(ids, dtype=torch.long)

    @patch("sglang.srt.managers.mm_utils.get_embedding_and_mask")
    def test_image_cached_video_not_cached(self, mock_get_emb):
        """IMAGE fully cached (None), VIDEO not cached — should not crash."""
        num_ds = 3
        mock_get_emb.side_effect = _mock_get_embedding_and_mask(
            cached_modality=Modality.IMAGE,
            embed_width=HIDDEN * (1 + num_ds),
        )

        input_ids = self._build_input_ids(img_tokens=4, vid_tokens=4)
        embedding_layer = nn.Embedding(VOCAB, HIDDEN)
        model = _make_multimodal_model(num_deepstack_layers=num_ds)
        mm_inputs = _make_mm_inputs(
            image_offsets=[(10, 13)],
            video_offsets=[(24, 27)],
        )

        result, _ = embed_mm_inputs(
            mm_inputs_list=[mm_inputs],
            extend_prefix_lens=[0],
            extend_seq_lens=[len(input_ids)],
            input_ids=input_ids,
            input_embedding=embedding_layer,
            multimodal_model=model,
            use_deepstack={Modality.IMAGE: True, Modality.VIDEO: True},
        )

        assert result is not None
        assert result.shape[0] == len(input_ids)

    @patch("sglang.srt.managers.mm_utils.get_embedding_and_mask")
    def test_video_cached_image_not_cached(self, mock_get_emb):
        """VIDEO fully cached (None), IMAGE not cached — should not crash."""
        num_ds = 3
        mock_get_emb.side_effect = _mock_get_embedding_and_mask(
            cached_modality=Modality.VIDEO,
            embed_width=HIDDEN * (1 + num_ds),
        )

        input_ids = self._build_input_ids(img_tokens=4, vid_tokens=4)
        embedding_layer = nn.Embedding(VOCAB, HIDDEN)
        model = _make_multimodal_model(num_deepstack_layers=3)
        mm_inputs = _make_mm_inputs(
            image_offsets=[(10, 13)],
            video_offsets=[(24, 27)],
        )

        result, _ = embed_mm_inputs(
            mm_inputs_list=[mm_inputs],
            extend_prefix_lens=[0],
            extend_seq_lens=[len(input_ids)],
            input_ids=input_ids,
            input_embedding=embedding_layer,
            multimodal_model=model,
            use_deepstack={Modality.IMAGE: True, Modality.VIDEO: True},
        )

        assert result is not None
        assert result.shape[0] == len(input_ids)

    @patch("sglang.srt.managers.mm_utils.get_embedding_and_mask")
    def test_both_not_cached(self, mock_get_emb):
        """Neither cached — both get embeddings, should work."""
        num_ds = 3
        mock_get_emb.side_effect = _mock_get_embedding_and_mask(
            cached_modality=None,  # nothing cached
            embed_width=HIDDEN * (1 + num_ds),
        )

        input_ids = self._build_input_ids(img_tokens=4, vid_tokens=4)
        embedding_layer = nn.Embedding(VOCAB, HIDDEN)
        model = _make_multimodal_model(num_deepstack_layers=num_ds)
        mm_inputs = _make_mm_inputs(
            image_offsets=[(10, 13)],
            video_offsets=[(24, 27)],
        )

        result, _ = embed_mm_inputs(
            mm_inputs_list=[mm_inputs],
            extend_prefix_lens=[0],
            extend_seq_lens=[len(input_ids)],
            input_ids=input_ids,
            input_embedding=embedding_layer,
            multimodal_model=model,
            use_deepstack={Modality.IMAGE: True, Modality.VIDEO: True},
        )

        assert result is not None

    @patch("sglang.srt.managers.mm_utils.get_embedding_and_mask")
    def test_no_deepstack(self, mock_get_emb):
        """Without deepstack, mixed cached/non-cached should always work."""
        mock_get_emb.side_effect = _mock_get_embedding_and_mask(
            cached_modality=Modality.IMAGE
        )

        input_ids = self._build_input_ids(img_tokens=4, vid_tokens=4)
        embedding_layer = nn.Embedding(VOCAB, HIDDEN)
        mm_inputs = _make_mm_inputs(
            image_offsets=[(10, 13)],
            video_offsets=[(24, 27)],
        )

        result, _ = embed_mm_inputs(
            mm_inputs_list=[mm_inputs],
            extend_prefix_lens=[0],
            extend_seq_lens=[len(input_ids)],
            input_ids=input_ids,
            input_embedding=embedding_layer,
            multimodal_model=None,
            data_embedding_func_mapping={
                Modality.IMAGE: lambda items: torch.ones(
                    sum(1 for item in items for _ in item.offsets), HIDDEN
                ),
                Modality.VIDEO: lambda items: torch.ones(
                    sum(1 for item in items for _ in item.offsets), HIDDEN
                ),
            },
            use_deepstack={},
        )

        assert result is not None
