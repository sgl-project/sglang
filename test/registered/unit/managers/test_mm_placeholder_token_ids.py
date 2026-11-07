"""embed_mm_inputs must leave each modality's placeholder token id, not a clamped
pad value, at multimodal positions of input_ids, on both the plain and the split
path. CPU-only: no engine, GPU or weights.
"""

import unittest
from types import SimpleNamespace

import torch
from torch import nn

from sglang.srt.managers import mm_schedule, mm_utils
from sglang.srt.managers.req_types import (
    MM_PAD_SHIFT_VALUE,
    MultimodalDataItem,
    MultimodalInputs,
)
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.srt.multimodal.modality import Modality
from sglang.srt.runtime_context import get_context, get_parallel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

VOCAB = 64
HIDDEN = 8
IMAGE_TOKEN_ID = 41
VIDEO_TOKEN_ID = 42
# Two items per request; (start, end) offsets are inclusive, as processors emit them.
OFFSETS = [(3, 6), (10, 12)]
SEQ_LEN = 16


def _feature(item: MultimodalDataItem) -> torch.Tensor:
    start, end = item.offsets[0]
    gen = torch.Generator().manual_seed(item.hash)
    return torch.randn(end - start + 1, HIDDEN, generator=gen)


class _FakeVLM:
    def get_image_feature(self, items):
        return torch.cat([_feature(item) for item in items], dim=0)

    def get_video_feature(self, items):
        return torch.cat([_feature(item) for item in items], dim=0)


def _request(
    req_seed,
    im_token_id,
    modalities=(Modality.IMAGE, Modality.IMAGE),
    video_token_id=None,
    precomputed=False,
):
    """One request: text ids < VOCAB, item runs filled with their items' pad values.

    Returns (mm_inputs, host key ids, multimodal positions per modality).
    """
    items = []
    for i, (offset, modality) in enumerate(zip(OFFSETS, modalities)):
        item = MultimodalDataItem(
            modality=modality, feature=torch.zeros(1), offsets=[offset]
        )
        item.set_hash(1_000 * req_seed + 7 * i + 1)
        if precomputed:
            item.precomputed_embeddings = _feature(item)
        items.append(item)
    gen = torch.Generator().manual_seed(req_seed)
    ids = torch.randint(0, VOCAB, (SEQ_LEN,), generator=gen)
    positions = {m: torch.zeros(SEQ_LEN, dtype=torch.bool) for m in Modality.all()}
    for item in items:
        start, end = item.offsets[0]
        ids[start : end + 1] = torch.tensor(item.padding_sequence(end - start + 1))
        positions[item.modality][start : end + 1] = True
    mm_inputs = MultimodalInputs(
        mm_items=items, im_token_id=im_token_id, video_token_id=video_token_id
    )
    return mm_inputs, ids, positions


def _embedding():
    torch.manual_seed(0)
    return nn.Embedding(VOCAB, HIDDEN)


def _embed(mm_inputs_list, input_ids, prefix_lens, seq_lens):
    embedding = _embedding()
    embeds, _ = mm_utils.embed_mm_inputs(
        mm_inputs_list=mm_inputs_list,
        extend_prefix_lens=prefix_lens,
        extend_seq_lens=seq_lens,
        input_ids=input_ids,
        input_embedding=embedding,
        multimodal_model=_FakeVLM(),
    )
    return embeds, embedding


def _expected_embeds(mm_inputs_list, host_ids, embedding):
    """Token embeddings, with each item's feature over its run."""
    expected = embedding(host_ids.clamp(0, VOCAB - 1)).clone()
    for r, mm in enumerate(mm_inputs_list):
        for item in mm.mm_items:
            start, end = item.offsets[0]
            expected[r * SEQ_LEN + start : r * SEQ_LEN + end + 1] = _feature(item)
    return expected


def _case_placeholder_token_id_written_in_place(num_requests):
    reqs = [_request(seed, IMAGE_TOKEN_ID) for seed in range(1, num_requests + 1)]
    mm_inputs_list = [mm for mm, _, _ in reqs]
    host_ids = torch.cat([ids for _, ids, _ in reqs])
    image_positions = torch.cat([pos[Modality.IMAGE] for _, _, pos in reqs])
    input_ids = host_ids.clone()  # forward_batch.input_ids, copied from the host ids

    embeds, embedding = _embed(
        mm_inputs_list, input_ids, [0] * num_requests, [SEQ_LEN] * num_requests
    )

    assert bool((input_ids[image_positions] == IMAGE_TOKEN_ID).all())
    assert torch.equal(input_ids[~image_positions], host_ids[~image_positions])
    # The host key ids (radix / prefix keys) keep their pad values.
    assert bool((host_ids[image_positions] >= MM_PAD_SHIFT_VALUE).all())
    torch.testing.assert_close(
        embeds, _expected_embeds(mm_inputs_list, host_ids, embedding), rtol=0, atol=0
    )


def _case_image_and_video_get_their_own_placeholder():
    mm, host_ids, positions = _request(
        5,
        IMAGE_TOKEN_ID,
        modalities=(Modality.IMAGE, Modality.VIDEO),
        video_token_id=VIDEO_TOKEN_ID,
    )
    input_ids = host_ids.clone()

    embeds, embedding = _embed([mm], input_ids, [0], [SEQ_LEN])

    assert bool((input_ids[positions[Modality.IMAGE]] == IMAGE_TOKEN_ID).all())
    assert bool((input_ids[positions[Modality.VIDEO]] == VIDEO_TOKEN_ID).all())
    mm_positions = positions[Modality.IMAGE] | positions[Modality.VIDEO]
    assert torch.equal(input_ids[~mm_positions], host_ids[~mm_positions])
    torch.testing.assert_close(
        embeds, _expected_embeds([mm], host_ids, embedding), rtol=0, atol=0
    )


def _case_only_a_later_request_carries_the_video_id():
    # Request 0 has images only and no video id; request 1 brings the video.
    reqs = [
        _request(1, IMAGE_TOKEN_ID),
        _request(
            2,
            IMAGE_TOKEN_ID,
            modalities=(Modality.IMAGE, Modality.VIDEO),
            video_token_id=VIDEO_TOKEN_ID,
        ),
    ]
    mm_inputs_list = [mm for mm, _, _ in reqs]
    host_ids = torch.cat([ids for _, ids, _ in reqs])
    image = torch.cat([pos[Modality.IMAGE] for _, _, pos in reqs])
    video = torch.cat([pos[Modality.VIDEO] for _, _, pos in reqs])
    input_ids = host_ids.clone()

    embeds, embedding = _embed(mm_inputs_list, input_ids, [0, 0], [SEQ_LEN] * 2)

    assert bool((input_ids[image] == IMAGE_TOKEN_ID).all())
    assert bool((input_ids[video] == VIDEO_TOKEN_ID).all())
    assert torch.equal(input_ids[~(image | video)], host_ids[~(image | video)])
    torch.testing.assert_close(
        embeds, _expected_embeds(mm_inputs_list, host_ids, embedding), rtol=0, atol=0
    )


def _case_chunk_boundary_inside_an_image():
    # Second chunk of a chunked prefill: it starts inside the first image.
    mm, ids, positions = _request(3, IMAGE_TOKEN_ID)
    prefix = 5
    chunk = ids[prefix:].clone()
    chunk_positions = positions[Modality.IMAGE][prefix:]

    embeds, _ = _embed([mm], chunk, [prefix], [SEQ_LEN - prefix])

    assert bool((chunk[chunk_positions] == IMAGE_TOKEN_ID).all())
    assert torch.equal(chunk[~chunk_positions], ids[prefix:][~chunk_positions])
    full = _FakeVLM().get_image_feature(mm.mm_items)
    torch.testing.assert_close(
        embeds[chunk_positions], full[prefix - OFFSETS[0][0] :], rtol=0, atol=0
    )


def _case_without_placeholder_token_id_falls_back_to_clamp():
    mm, ids, positions = _request(4, None)
    image_positions = positions[Modality.IMAGE]
    input_ids = ids.clone()

    embeds, _ = _embed([mm], input_ids, [0], [SEQ_LEN])

    assert bool((input_ids[image_positions] == VOCAB - 1).all())
    assert torch.equal(input_ids[~image_positions], ids[~image_positions])
    torch.testing.assert_close(
        embeds[image_positions],
        _FakeVLM().get_image_feature(mm.mm_items),
        rtol=0,
        atol=0,
    )


def _case_split_path_mixed_precomputed():
    # Adaptive encoder dispatch embeds a torch.cat copy per group; input_ids must
    # still end up with the placeholder ids.
    reqs = [
        _request(6, IMAGE_TOKEN_ID, precomputed=True),
        _request(7, IMAGE_TOKEN_ID, precomputed=False),
    ]
    mm_inputs_list = [mm for mm, _, _ in reqs]
    host_ids = torch.cat([ids for _, ids, _ in reqs])
    image_positions = torch.cat([pos[Modality.IMAGE] for _, _, pos in reqs])
    input_ids = host_ids.clone()
    forward_batch = SimpleNamespace(
        extend_seq_lens_cpu=[SEQ_LEN, SEQ_LEN], mm_inputs=mm_inputs_list
    )
    embedding = _embedding()

    embeds, _ = mm_utils._embed_mm_inputs_with_split(
        mm_inputs_list=mm_inputs_list,
        extend_prefix_lens=[0, 0],
        extend_seq_lens=[SEQ_LEN, SEQ_LEN],
        input_ids=input_ids,
        forward_batch=forward_batch,
        input_embedding=embedding,
        multimodal_model=_FakeVLM(),
    )

    assert bool((input_ids[image_positions] == IMAGE_TOKEN_ID).all())
    assert torch.equal(input_ids[~image_positions], host_ids[~image_positions])
    assert bool((host_ids[image_positions] >= MM_PAD_SHIFT_VALUE).all())
    torch.testing.assert_close(
        embeds, _expected_embeds(mm_inputs_list, host_ids, embedding), rtol=0, atol=0
    )


def _case_later_pipeline_stage():
    # Later PP stages skip the encoder; input_ids must still get the placeholder ids.
    mm_a, ids_a, pos_a = _request(
        8,
        IMAGE_TOKEN_ID,
        modalities=(Modality.IMAGE, Modality.VIDEO),
        video_token_id=VIDEO_TOKEN_ID,
    )
    mm_b, ids_b, pos_b = _request(9, IMAGE_TOKEN_ID)
    gen = torch.Generator().manual_seed(10)
    text_ids = torch.randint(0, VOCAB, (SEQ_LEN,), generator=gen)
    no_mm = torch.zeros(SEQ_LEN, dtype=torch.bool)
    host_ids = torch.cat([ids_a, text_ids, ids_b])
    image_positions = torch.cat([pos_a[Modality.IMAGE], no_mm, pos_b[Modality.IMAGE]])
    video_positions = torch.cat([pos_a[Modality.VIDEO], no_mm, no_mm])
    input_ids = host_ids.clone()
    forward_batch = SimpleNamespace(
        forward_mode=ForwardMode.EXTEND,
        contains_mm_inputs=lambda: True,
        mm_inputs=[mm_a, None, mm_b],
    )
    language_model = SimpleNamespace(
        get_input_embeddings=_embedding,
        pp_group=SimpleNamespace(is_first_rank=False),
    )

    input_embeds = mm_utils.prepare_mm_inputs(
        input_ids=input_ids,
        forward_batch=forward_batch,
        language_model=language_model,
        kwargs={},
        multimodal_model=_FakeVLM(),
    )

    assert input_embeds is None
    assert bool((input_ids[image_positions] == IMAGE_TOKEN_ID).all())
    assert bool((input_ids[video_positions] == VIDEO_TOKEN_ID).all())
    mm_positions = image_positions | video_positions
    assert torch.equal(input_ids[~mm_positions], host_ids[~mm_positions])
    assert bool((host_ids[mm_positions] >= MM_PAD_SHIFT_VALUE).all())


class TestMMPlaceholderTokenIds(CustomTestCase):
    def setUp(self):
        # The embedding path reads the config namespaces and the attention-TP rank.
        self._server_args = get_context().override_server_args(tp_size=1)
        self._server_args.install()
        self._parallel = get_parallel().override(
            attn_tp_rank=0, attn_tp_size=1, tp_size=1
        )
        self._parallel.__enter__()
        mm_schedule.init_mm_embedding_cache(1 << 30)

    def tearDown(self):
        self._parallel.__exit__(None, None, None)
        self._server_args.restore()

    def test_placeholder_token_id_written_in_place(self):
        for num_requests in (1, 2):
            with self.subTest(num_requests=num_requests):
                _case_placeholder_token_id_written_in_place(num_requests)

    def test_image_and_video_get_their_own_placeholder(self):
        _case_image_and_video_get_their_own_placeholder()

    def test_only_a_later_request_carries_the_video_id(self):
        _case_only_a_later_request_carries_the_video_id()

    def test_chunk_boundary_inside_an_image(self):
        _case_chunk_boundary_inside_an_image()

    def test_without_placeholder_token_id_falls_back_to_clamp(self):
        _case_without_placeholder_token_id_falls_back_to_clamp()

    def test_split_path_mixed_precomputed(self):
        _case_split_path_mixed_precomputed()

    def test_later_pipeline_stage(self):
        _case_later_pipeline_stage()


if __name__ == "__main__":
    unittest.main()
