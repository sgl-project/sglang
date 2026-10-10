"""Tests for encoder SWA replay with image inputs."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.managers import mm_schedule
from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.managers.schedule_batch import (
    MM_PAD_SHIFT_VALUE,
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
    check_encoder_swa_replay_request,
)
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.mem_cache.multimodal_cache import MultiModalStaticCache
from sglang.srt.model_executor.encoder_swa_replay import run_encoder_swa_replay
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.models.deepseek_v4 import DeepseekV4ForCausalLM
from sglang.srt.runtime_context import get_context
from sglang.test.test_utils import CustomTestCase

WINDOW = 128
HIDDEN = 8
VOCAB = 64
IMAGE_TOKEN_ID = VOCAB - 1
SEQ_LEN = 320
IMAGE_A = (40, 79, MM_PAD_SHIFT_VALUE + 7)
IMAGE_B = (150, 189, MM_PAD_SHIFT_VALUE + 9)


def _image(first, last, pad_value=MM_PAD_SHIFT_VALUE + 7, modality=Modality.IMAGE):
    return MultimodalDataItem(
        modality=modality,
        hash=pad_value - MM_PAD_SHIFT_VALUE,
        pad_value=pad_value,
        offsets=[(first, last)],
    )


def _mm(*items):
    return MultimodalInputs(mm_items=list(items))


def _text_ids():
    return (torch.arange(SEQ_LEN) % IMAGE_TOKEN_ID).tolist()


def _image_req():
    fill_ids = _text_ids()
    for first, last, pad_value in (IMAGE_A, IMAGE_B):
        fill_ids[first : last + 1] = [pad_value] * (last - first + 1)
    mm = _mm(*(_image(*image) for image in (IMAGE_A, IMAGE_B)))
    return SimpleNamespace(full_untruncated_fill_ids=fill_ids, multimodal_inputs=mm)


def _text_req():
    return SimpleNamespace(
        full_untruncated_fill_ids=_text_ids(), multimodal_inputs=None
    )


class _Window:
    def __init__(self):
        self.reset_slots = []

    def reset(self, slot):
        self.reset_slots.append(int(slot[0]))


class _Model:
    """DeepSeek-V4.1 multimodal entry point over a stub vision encoder and decoder."""

    forward = DeepseekV4ForCausalLM.forward
    _prepare_mm_embeddings = DeepseekV4ForCausalLM._prepare_mm_embeddings
    get_input_embeddings = DeepseekV4ForCausalLM.get_input_embeddings

    def __init__(self):
        torch.manual_seed(0)
        embedding = torch.nn.Embedding(VOCAB, HIDDEN)
        self.vision = object()
        self.config = SimpleNamespace(image_token_id=IMAGE_TOKEN_ID)
        self.pp_group = SimpleNamespace(is_last_rank=False)
        self.model = SimpleNamespace(
            engram_hasher=None,
            get_input_embeddings=lambda: embedding,
            forward=self._decoder,
        )
        self.encoded = []
        self.calls = []

    def get_image_feature(self, items):
        self.encoded.extend(item.hash for item in items)
        spans = []
        for item in items:
            first, last = item.offsets[0]
            rows = torch.arange((last - first + 1) * HIDDEN, dtype=torch.float32)
            spans.append(1000.0 * item.hash + rows.reshape(-1, HIDDEN))
        return spans

    def _decoder(self, input_ids, positions, forward_batch, input_embeds, pp_proxy):
        self.calls.append((forward_batch, input_ids, input_embeds))


def _forward_batch(input_ids, mm_inputs, prefix_lens, extend_lens):
    fb = ForwardBatch.__new__(ForwardBatch)
    fb.forward_mode = ForwardMode.EXTEND
    fb.input_ids = input_ids
    fb.mm_inputs = mm_inputs
    fb.extend_prefix_lens_cpu = list(prefix_lens)
    fb.extend_seq_lens_cpu = list(extend_lens)
    return fb


class TestEncoderSwaReplayImages(CustomTestCase):
    def setUp(self):
        cache = patch.object(
            mm_schedule, "embedding_cache", MultiModalStaticCache(1 << 30)
        )
        cache.start()
        self.addCleanup(cache.stop)
        self.model = _Model()

    def _replay(self, reqs, prefix_lens, reset):
        slots = [3, 5, 6][: len(reqs)]
        window = _Window()
        replays = []

        def init_new(replay, *args, **kwargs):
            replays.append(replay)
            return _forward_batch(
                replay.input_ids,
                replay.multimodal_inputs,
                replay.prefix_lens,
                replay.extend_lens,
            )

        runner = SimpleNamespace(
            device="cpu",
            token_to_kv_pool=SimpleNamespace(request_window=window),
            req_to_token_pool=SimpleNamespace(
                req_to_token=torch.arange(8 * 1024).reshape(8, 1024)
            ),
            model=self.model,
            forward=lambda fb: self.model.forward(fb.input_ids, None, fb),
        )
        batch = SimpleNamespace(
            forward_mode=ForwardMode.EXTEND,
            encoder_swa_reset=reset,
            req_pool_indices=torch.tensor(slots),
            req_pool_indices_cpu=torch.tensor(slots),
            prefix_lens=prefix_lens,
            reqs=reqs,
            multimodal_inputs=[r.multimodal_inputs for r in reqs],
        )
        self.model.calls.clear()
        with patch.object(ForwardBatch, "init_new", side_effect=init_new):
            run_encoder_swa_replay(SimpleNamespace(model_runner=runner), batch)
        return window, replays, list(self.model.calls)

    def _prefill_embeds(self, req):
        ids = torch.tensor(req.full_untruncated_fill_ids)
        fb = _forward_batch(ids, [req.multimodal_inputs], [0], [len(ids)])
        with torch.no_grad():
            self.model.forward(ids, None, fb)
        return self.model.calls.pop()[2]

    def test_replay_window_matches_prefill_rows(self):
        # Clamped start ending inside A, start inside A and end inside B, after both.
        a, b = IMAGE_A[2] - MM_PAD_SHIFT_VALUE, IMAGE_B[2] - MM_PAD_SHIFT_VALUE
        for end, overlapping in ((60, [a]), (170, [a, b]), (SEQ_LEN, [])):
            for cached in (True, False):
                with self.subTest(end=end, cached=cached):
                    mm_schedule.embedding_cache.clear()
                    req = _image_req()
                    prefill = self._prefill_embeds(req)
                    if not cached:
                        mm_schedule.embedding_cache.clear()
                    self.model.encoded.clear()

                    with torch.no_grad():
                        _, _, calls = self._replay([req], [end], [True])

                    start = max(0, end - WINDOW)
                    [(fb, model_ids, embeds)] = calls
                    torch.testing.assert_close(
                        embeds, prefill[start:end], rtol=0, atol=0
                    )
                    self.assertEqual(self.model.encoded, [] if cached else overlapping)
                    fill_ids = req.full_untruncated_fill_ids[start:end]
                    self.assertEqual(fb.input_ids.tolist(), fill_ids)
                    self.assertEqual(
                        (model_ids == IMAGE_TOKEN_ID).tolist(),
                        [i >= MM_PAD_SHIFT_VALUE for i in fill_ids],
                    )
                    self.assertTrue(fb.contains_mm_inputs())

    def test_each_replay_carries_its_request_items(self):
        reqs = [_text_req(), _image_req()]
        window, replays, calls = self._replay(reqs, [300, 170], [True, True])
        self.assertEqual(window.reset_slots, [3, 5])
        for replay, req, slot, end in zip(replays, reqs, (3, 5), (300, 170)):
            self.assertEqual(replay.reqs, [req])
            self.assertEqual(replay.multimodal_inputs, [req.multimodal_inputs])
            self.assertEqual(replay.prefix_lens, [end - WINDOW])
            self.assertEqual(replay.extend_lens, [WINDOW])
            self.assertEqual(
                replay.out_cache_loc.tolist(),
                list(range(slot * 1024 + end - WINDOW, slot * 1024 + end)),
            )
            self.assertTrue(replay.is_prefill_only)
        (text_fb, _, text_embeds), (image_fb, _, image_embeds) = calls
        self.assertIsNone(text_embeds)
        self.assertFalse(text_fb.contains_mm_inputs())
        self.assertIsNotNone(image_embeds)
        self.assertTrue(image_fb.contains_mm_inputs())

    def test_requests_without_reset_or_prefix_do_not_replay(self):
        window, replays, _ = self._replay(
            [_image_req(), _image_req()], [4, 0], [False, True]
        )
        self.assertEqual(window.reset_slots, [5])
        self.assertEqual(replays, [])


class TestEncoderReplayRequestValidation(CustomTestCase):
    def setUp(self):
        override = get_context().override_server_args(
            enable_encoder_swa_bounded_replay=True
        )
        override.install()
        self.addCleanup(override.restore)
        self.manager = TokenizerManager.__new__(TokenizerManager)
        self.manager.context_len = 128
        self.manager.num_reserved_tokens = 0
        self.manager.allow_auto_truncate = False
        self.manager.validate_total_tokens = False
        self.manager.is_generation = True
        self.manager.model_config = SimpleNamespace(joint_head_config=None)

    def _validate(self, **kwargs):
        req = GenerateReqInput(input_ids=[1, 2, 3], sampling_params={}, **kwargs)
        self.manager._validate_one_request(req, req.input_ids)

    def test_accepts_text_and_image_requests(self):
        self._validate()
        self._validate(image_data=["data:image/png;base64,AAAA"])
        self._validate(return_logprob=True, logprob_start_len=-1)

    def test_rejects_inputs_replay_cannot_rebuild(self):
        for field, value in (
            ("video_data", ["video.mp4"]),
            ("audio_data", ["audio.wav"]),
            ("input_embeds", [[0.0] * 4]),
            ("positional_embed_overrides", {"0": [0.0]}),
        ):
            with (
                self.subTest(field),
                self.assertRaisesRegex(
                    ValueError, "supports text and image requests only"
                ),
            ):
                self._validate(**{field: value})

    def test_rejects_cached_prompt_logprobs(self):
        with self.assertRaisesRegex(ValueError, "cached prompt logprobs"):
            self._validate(return_logprob=True, logprob_start_len=0)


class TestSchedulerReplayRequestCheck(CustomTestCase):
    def _req(self, **overrides):
        fields = dict(
            multimodal_inputs=None,
            input_embeds=None,
            positional_embed_overrides=None,
            return_logprob=False,
            logprob_start_len=-1,
            origin_input_ids=[1, 2, 3],
        )
        fields.update(overrides)
        return SimpleNamespace(**fields)

    def test_accepts_text_and_image_requests(self):
        check_encoder_swa_replay_request(self._req())
        check_encoder_swa_replay_request(
            self._req(multimodal_inputs=_mm(_image(0, 3), _image(5, 9)))
        )
        check_encoder_swa_replay_request(
            self._req(return_logprob=True, logprob_start_len=3)
        )

    def test_rejects_other_embedding_sources(self):
        for name, overrides in (
            (
                "video",
                dict(multimodal_inputs=_mm(_image(0, 3, modality=Modality.VIDEO))),
            ),
            (
                "audio",
                dict(multimodal_inputs=_mm(_image(0, 3, modality=Modality.AUDIO))),
            ),
            ("input embeds", dict(input_embeds=[[0.0]])),
            ("positional overrides", dict(positional_embed_overrides=object())),
        ):
            with (
                self.subTest(name),
                self.assertRaisesRegex(
                    ValueError, "supports text and image requests only"
                ),
            ):
                check_encoder_swa_replay_request(self._req(**overrides))

    def test_rejects_cached_prompt_logprobs(self):
        with self.assertRaisesRegex(ValueError, "cached prompt logprobs"):
            check_encoder_swa_replay_request(
                self._req(return_logprob=True, logprob_start_len=1)
            )


if __name__ == "__main__":
    unittest.main()
