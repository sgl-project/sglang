"""Unit tests for GigaChat3.5-Omni: config, conformer encoder and video token reduction."""

import math
import tempfile
import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from sglang.srt.configs.gigachat35 import GigaChat35Config
from sglang.srt.configs.gigachat35_omni import (
    GigaChat35OmniAudioConfig,
    GigaChat35OmniConfig,
)
from sglang.srt.configs.linear_attn_model_registry import get_linear_attn_config
from sglang.srt.hf_transformers.config import get_config
from sglang.srt.managers import mm_schedule, mm_utils
from sglang.srt.managers.req_types import MultimodalDataItem, MultimodalInputs
from sglang.srt.mem_cache.multimodal_cache import MultiModalStaticCache
from sglang.srt.models.gigachat35_omni import (
    GigaChat35OmniConformerEncoder,
    GigaChat35OmniConformerLayer,
    GigaChat35OmniForConditionalGeneration,
)
from sglang.srt.multimodal.gigachat35_omni import GigaChat35OmniVideoEmbedding
from sglang.srt.multimodal.modality import Modality
from sglang.srt.runtime_context import reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.parallel_groups import publish
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="base-a-test-cpu")


class TestGigaChat35OmniConfig(CustomTestCase):
    def test_nested_configs_survive_a_checkpoint_round_trip(self):
        """The served config must come back with typed sub-configs and the hybrid LM spec."""
        config = GigaChat35OmniConfig(
            text_config={"num_hidden_layers": 2, "full_attention_layers": [1]},
            vision_config={"image_size": 448, "patch_size": 14},
            audio_config={"hidden_size": 32},
            architectures=["GigaChat35OmniForConditionalGeneration"],
        )

        with tempfile.TemporaryDirectory() as path:
            config.save_pretrained(path)
            loaded = get_config(path, trust_remote_code=False)

        self.assertIsInstance(loaded, GigaChat35OmniConfig)
        self.assertIsInstance(loaded.get_text_config(), GigaChat35Config)
        self.assertEqual(loaded.get_text_config().full_attention_layer_ids, [1])
        self.assertIsInstance(loaded.audio_config, GigaChat35OmniAudioConfig)
        self.assertEqual(loaded.audio_config.hidden_size, 32)
        self.assertEqual(loaded.vision_config.image_size, 448)

        spec, text_config = get_linear_attn_config(loaded)
        self.assertIn("GigaChat35OmniForConditionalGeneration", spec.arch_names)
        self.assertIs(text_config, loaded.text_config)
        self.assertEqual(loaded.video_tokens_per_frame, 77)


class TestGigaChat35OmniAudio(CustomTestCase):
    def setUp(self):
        reset_context()
        self.addCleanup(reset_context)
        publish(
            ServerArgs(
                model_path="dummy", device="cpu", tp_size=1, mm_attention_backend="sdpa"
            ),
            role="test",
        )
        torch.manual_seed(0)
        config = GigaChat35OmniAudioConfig(
            hidden_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            intermediate_size=64,
            num_mel_bins=16,
            subsampling_conv_channels=24,
            attention_chunk_size=8,
        )
        self.model = SimpleNamespace(
            config=SimpleNamespace(audio_stack_factor=4),
            audio_tower=GigaChat35OmniConformerEncoder(config).eval(),
            audio_projector=nn.Linear(4 * 32, 8),
        )

    def _clip(self, num_mel_frames, num_tokens):
        """Builds an audio item with random mel frames.

        Args:
            num_mel_frames: Mel frames of the clip.
            num_tokens: Placeholders of the clip in the prompt.

        Returns:
            The item.
        """
        return MultimodalDataItem(
            modality=Modality.AUDIO,
            feature=torch.randn(num_mel_frames, 16),
            offsets=[(0, num_tokens - 1)],
        )

    def _encode(self, items):
        """Runs `get_audio_feature` on the tiny encoder.

        Args:
            items: Audio items.

        Returns:
            Embeddings of every item.
        """
        with torch.no_grad():
            return GigaChat35OmniForConditionalGeneration.get_audio_feature(
                self.model, items
            )

    def test_token_count_matches_the_checkpoint_processor(self):
        """Two stride-2 convolutions, then 4 frames per token: ceil(T_mel / 16)."""
        for num_mel_frames in (1, 15, 16, 17, 64, 333, 801):
            with self.subTest(num_mel_frames=num_mel_frames):
                num_tokens = math.ceil(num_mel_frames / 16)
                embeds = self._encode([self._clip(num_mel_frames, num_tokens)])
                self.assertEqual(tuple(embeds.shape), (num_tokens, 8))

        with self.assertRaisesRegex(AssertionError, "placeholders"):
            self._encode([self._clip(64, 5)])

    def test_clips_are_encoded_independently(self):
        clips = [self._clip(333, 21), self._clip(97, 7)]
        together = self._encode(clips)
        alone = torch.cat([self._encode([clip]) for clip in clips])
        torch.testing.assert_close(together, alone, rtol=0, atol=0)

    def test_attention_blocks_and_rope_restart_every_chunk(self):
        calls = []
        cases = [(76, [0, 8, 16, 19], 8), (64, [0, 8, 16], 8), (20, [0, 5], 5)]

        for num_mel_frames, cu_seqlens, max_seqlen in cases:
            with self.subTest(num_mel_frames=num_mel_frames):
                calls.clear()

                with patch.object(
                    GigaChat35OmniConformerLayer,
                    "forward",
                    autospec=True,
                    side_effect=lambda layer, x, *args: calls.append(args) or x,
                ):
                    clip = self._clip(num_mel_frames, math.ceil(num_mel_frames / 16))
                    self._encode([clip])

                cos, _, metadata = calls[0]
                self.assertEqual(metadata.cu_seqlens.tolist(), cu_seqlens)
                self.assertEqual(metadata.max_seqlen, max_seqlen)

                for start in cu_seqlens[1:-1]:
                    length = min(8, len(cos) - start)
                    torch.testing.assert_close(
                        cos[start : start + length], cos[:length]
                    )

                self.assertFalse(torch.equal(cos[1], cos[0]))

    def test_windowed_long_clips_match_the_whole_clip(self):
        """A halo of one block per layer makes windows match the whole clip."""
        encoder = self.model.audio_tower
        cases = {"just above one window": 41, "short last window": 83, "one window": 40}

        for name, num_frames in cases.items():
            with self.subTest(name):
                mel = torch.randn(4 * num_frames, 16)

                with torch.no_grad(), patch.object(encoder, "_MAX_WINDOW_FRAMES", 40):
                    whole = encoder.encode_frames(encoder.pre_encode(mel))
                    windowed = encoder(mel)

                    with patch.object(encoder, "halo", encoder.chunk_size):
                        short_halo = encoder(mel)

                torch.testing.assert_close(windowed, whole)
                gap = (short_halo - whole).abs().max().item()
                self.assertEqual(gap > 1e-4, num_frames > 40)

    def test_window_boundaries_at_the_served_window_size(self):
        """Up to 48,000 frames take one pass; longer clips split with the halo."""
        encoder = self.model.audio_tower
        cases = [(48000, [48000]), (48001, [48001, 17]), (96005, [48016, 48021, 21])]
        calls = []

        with patch.object(
            GigaChat35OmniConformerEncoder,
            "encode_frames",
            autospec=True,
            side_effect=lambda encoder, x: calls.append(len(x)) or x,
        ):
            for num_frames, window_lengths in cases:
                with self.subTest(num_frames=num_frames):
                    calls.clear()
                    mel = torch.randn(4 * num_frames, 16)

                    with torch.no_grad():
                        frames = encoder.pre_encode(mel)
                        torch.testing.assert_close(encoder(mel), frames, rtol=0, atol=0)

                    self.assertEqual(calls, window_lengths)


class TestGigaChat35OmniWeightLoading(CustomTestCase):
    def _model(self):
        """Builds a wrapper with one encoder weight and a stub LM.

        Returns:
            The model.
        """
        model = nn.Module()
        model.language_model = SimpleNamespace(
            load_weights=lambda weights: list(weights), post_load_weights=lambda: None
        )
        model.audio_projector = nn.Linear(2, 2)
        model.check_missing_weights = True
        return model

    def test_only_the_checkpoint_load_needs_every_encoder_weight(self):
        """Online weight updates may send only the LM."""
        lm_only = [("language_model.model.norm.weight", torch.zeros(2))]
        checkpoint = lm_only + [
            ("audio_projector.weight", torch.ones(2, 2)),
            ("audio_projector.bias", torch.ones(2)),
        ]
        model = self._model()

        with self.assertRaisesRegex(AssertionError, "audio_projector.bias"):
            GigaChat35OmniForConditionalGeneration.load_weights(model, lm_only)

        GigaChat35OmniForConditionalGeneration.load_weights(model, checkpoint)
        GigaChat35OmniForConditionalGeneration.load_weights(model, lm_only)
        self.assertTrue(torch.equal(model.audio_projector.bias, torch.ones(2)))

        # Dummy and remote loaders write the weights without `load_weights`.
        model = self._model()
        GigaChat35OmniForConditionalGeneration.post_load_weights(model)
        GigaChat35OmniForConditionalGeneration.load_weights(model, lm_only)


class TestGigaChat35OmniVideoReduction(CustomTestCase):
    """Video token reduction and the layout rewrite of every prefill chunk."""

    V, W = 1_000_001, 1_000_002

    @staticmethod
    def _pad(ids, items, **token_ids):
        """Runs the model's `pad_input_ids` on one request.

        Args:
            ids: The request's prompt.
            items: The request's items.
            **token_ids: Placeholder ids for `MultimodalInputs`.

        Returns:
            The padded prompt.
        """
        return GigaChat35OmniForConditionalGeneration.pad_input_ids(
            None, array("q", ids), MultimodalInputs(mm_items=items, **token_ids)
        )

    @staticmethod
    def _video(pad_value, offsets):
        """Builds a video item with one pad value.

        Args:
            pad_value: The video's pad value.
            offsets: Frame runs in the request's prompt.

        Returns:
            The item.
        """
        return MultimodalDataItem(
            modality=Modality.VIDEO,
            pad_value=pad_value,
            offsets=offsets,
            model_specific_data={"thw_grids": [(len(offsets), 2, 2)]},
        )

    def test_budget_is_first_frame_plus_fixed_tokens_per_later_frame(self):
        """256 + 77 * (T - 1) tokens per video, never more than the video has."""
        for num_frames, expected in [(1, 256), (2, 333), (5, 564)]:
            with self.subTest(num_frames=num_frames):
                self.assertEqual(
                    GigaChat35OmniVideoEmbedding.num_kept_tokens(num_frames, 256, 77),
                    expected,
                )
                split = GigaChat35OmniVideoEmbedding.placeholder_counts(
                    num_frames, 256, 77
                )
                self.assertEqual(sum(split), expected)

        self.assertEqual(
            GigaChat35OmniVideoEmbedding.placeholder_counts(5, 256, 77),
            [112] * 4 + [116],
        )
        self.assertEqual(GigaChat35OmniVideoEmbedding.num_kept_tokens(3, 4, 8), 12)

    def test_reduction_keeps_the_most_changed_tokens(self):
        """A scaled token has no cosine distance but the largest mean |diff|."""
        first = torch.arange(16.0).reshape(4, 4) / 10
        second = first.clone()
        second[2] += 5
        third = second.clone()
        third[0] += 3
        third[3] *= 10
        video = torch.cat([first, second, third])

        kept, counts = GigaChat35OmniVideoEmbedding.reduce(video, (3, 2, 2), 1)

        self.assertEqual(counts, [4, 1, 1])
        torch.testing.assert_close(kept, video[[0, 1, 2, 3, 6, 11]], rtol=0, atol=0)

        # Ties keep the earliest tokens.
        video = torch.cat([first] * 3)
        kept, counts = GigaChat35OmniVideoEmbedding.reduce(video, (3, 2, 2), 1)
        self.assertEqual(counts, [4, 2, 0])
        torch.testing.assert_close(kept, video[:6], rtol=0, atol=0)

    def test_reduction_does_not_depend_on_frame_chunking(self):
        video = torch.randn(24, 8, generator=torch.Generator().manual_seed(0))
        expected, expected_counts = GigaChat35OmniVideoEmbedding.reduce(
            video, (6, 2, 2), 2
        )

        for frame_chunk in (1, 2, 4):
            with (
                self.subTest(frame_chunk=frame_chunk),
                patch.object(GigaChat35OmniVideoEmbedding, "FRAME_CHUNK", frame_chunk),
            ):
                kept, counts = GigaChat35OmniVideoEmbedding.reduce(video, (6, 2, 2), 2)
                self.assertEqual(counts, expected_counts)
                torch.testing.assert_close(kept, expected, rtol=0, atol=0)

    def test_one_frame_videos_keep_every_token(self):
        """A one-frame video is kept whole, alone or next to a longer video."""
        features = torch.randn(16, 8, generator=torch.Generator().manual_seed(1))
        model = SimpleNamespace(
            extract_feature=lambda pixels: pixels,
            config=SimpleNamespace(video_tokens_per_frame=1),
        )
        items = [
            MultimodalDataItem(
                modality=Modality.VIDEO,
                feature=feature,
                model_specific_data={"thw_grids": [(len(feature) // 4, 2, 2)]},
            )
            for feature in features.split([4, 12])
        ]

        result = GigaChat35OmniForConditionalGeneration.get_video_feature(model, items)

        self.assertEqual(result.frame_counts[0], [4])
        self.assertEqual(result.frame_counts[1][0], 4)
        self.assertEqual(result.embedding.shape, (10, 8))
        torch.testing.assert_close(result.embedding[:8], features[:8], rtol=0, atol=0)

        # One-frame videos only: one encoder call, nothing to reduce.
        result = GigaChat35OmniForConditionalGeneration.get_video_feature(
            model, [items[0], items[0]]
        )
        torch.testing.assert_close(
            result, torch.cat([features[:4]] * 2), rtol=0, atol=0
        )

    def test_rewrite_changes_only_its_request_in_a_batch(self):
        """A prefill chunk of the second of three requests in a batch.

        After a retraction the re-prefill chunk ends with generated tokens,
        which stay as given.
        """
        V = self.V
        ids = [1, 2, V, V, V, V, 7, V, V, V, V, 8, V, V, V, V, 9]
        video = self._video(V, [(2, 5), (7, 10), (12, 15)])
        self._pad(ids, [video], video_token_id=V)
        result = GigaChat35OmniVideoEmbedding(
            embedding=torch.zeros(12, 1), frame_counts=[[6, 2, 4]]
        )
        before, after = [100, 101, 102, 103, 104], [200, 201]
        rewritten = [1, 2, V, V, V, V, V, V, 7, V, V, 8, V, V, V, V, 9]

        for prefix, length, generated in [(3, 10, []), (5, 15, [300, 301, 302])]:
            with self.subTest(prefix=prefix, length=length):
                prompt = slice(prefix, prefix + length - len(generated))
                batch = torch.tensor(before + ids[prompt] + generated + after)

                new_batch, offsets = result.rewrite_layout(
                    batch,
                    items=[video],
                    extend_prefix_len=prefix,
                    extend_seq_len=length,
                    extend_start=len(before),
                )

                self.assertEqual(
                    new_batch.tolist(), before + rewritten[prompt] + generated + after
                )
                self.assertEqual(offsets, [(2, 7), (9, 10), (12, 15)])

    def test_rewrite_handles_several_videos_per_request(self):
        V, W = self.V, self.W
        ids = [1, V, V, V, 2, V, V, V, 3, W, W, 4, W, W, 5]
        result = GigaChat35OmniVideoEmbedding(
            embedding=torch.zeros(10, 1), frame_counts=[[5, 1], [1, 3]]
        )

        # In a session, an earlier turn's video was padded on that turn's prompt.
        for first_prompt in (ids, ids[:9]):
            with self.subTest(first_prompt_len=len(first_prompt)):
                videos = [
                    self._video(V, [(1, 3), (5, 7)]),
                    self._video(W, [(9, 10), (12, 13)]),
                ]
                self._pad(first_prompt, videos[:1], video_token_id=V)
                self._pad(ids, videos[1:], video_token_id=W)

                new_ids, offsets = result.rewrite_layout(
                    torch.tensor(ids),
                    items=videos,
                    extend_prefix_len=0,
                    extend_seq_len=15,
                    extend_start=0,
                )

                self.assertEqual(
                    new_ids.tolist(), [1, V, V, V, V, V, 2, V, 3, W, 4, W, W, W, 5]
                )
                self.assertEqual(offsets, [(1, 5), (7, 7), (9, 9), (11, 13)])

        # The first video kept 7 tokens for its 6 placeholders.
        result.frame_counts = [[5, 2], [1, 2]]

        with self.assertRaises(AssertionError):
            result.rewrite_layout(
                torch.tensor(ids),
                items=videos,
                extend_prefix_len=0,
                extend_seq_len=15,
                extend_start=0,
            )

    def test_rewrite_moves_soundtracks_of_identical_videos_per_chunk(self):
        """Soundtrack runs between the frames move with them, on every chunk.

        Identical videos share pad values, which also cycle within a run, so
        offsets cannot be found by searching the prompt for a pad value.
        """
        V, S, T0, T1 = 5, 6, 7, 8
        video_ids = [T0, V, V, V, V, S, S, T1, V, V, V, V, S, S]
        ids = [9] + video_ids + [9] + video_ids + [9]
        items = []

        for index, start in enumerate((1, 16)):
            video = MultimodalDataItem(
                modality=Modality.VIDEO,
                feature=torch.zeros(2, 1),
                offsets=[(start + 1, start + 4), (start + 8, start + 11)],
                model_specific_data={"thw_grids": [(2, 2, 2)]},
            )
            soundtrack = MultimodalDataItem(
                modality=Modality.AUDIO,
                feature=torch.ones(4, 1),
                offsets=[(start + 5, start + 6), (start + 12, start + 13)],
                model_specific_data={
                    GigaChat35OmniVideoEmbedding.SOUNDTRACK_OF_KEY: index
                },
            )
            items += [video, soundtrack]

        for item in items:
            item.set_pad_value()

        padded = self._pad(ids, items, video_token_id=V, audio_token_id=S)
        videos, soundtracks = items[::2], items[1::2]
        result = GigaChat35OmniVideoEmbedding(
            embedding=torch.zeros(16, 1), frame_counts=[[6, 2], [6, 2]]
        )
        chunks = []

        for prefix, length in [(0, 12), (12, 19)]:
            chunk, offsets = result.rewrite_layout(
                torch.tensor(padded[prefix : prefix + length]),
                items=videos,
                extend_prefix_len=prefix,
                extend_seq_len=length,
                extend_start=0,
            )
            chunks += chunk.tolist()
            self.assertEqual(offsets, [(2, 7), (11, 12), (17, 22), (26, 27)])
            self.assertEqual(soundtracks[0].offsets, [(8, 9), (13, 14)])
            self.assertEqual(soundtracks[1].offsets, [(23, 24), (28, 29)])

        video_values = set(videos[0].padding_values())
        self.assertEqual(
            [i for i, token in enumerate(chunks) if token in video_values],
            [2, 3, 4, 5, 6, 7, 11, 12, 17, 18, 19, 20, 21, 22, 26, 27],
        )

        for soundtrack, positions in zip(
            soundtracks, ([8, 9, 13, 14], [23, 24, 28, 29])
        ):
            values = set(soundtrack.padding_values())
            self.assertEqual(
                [i for i, token in enumerate(chunks) if token in values], positions
            )
        self.assertEqual(
            [(i, chunks[i]) for i in (0, 1, 10, 15, 16, 25, 30)],
            [(0, 9), (1, T0), (10, T1), (15, 9), (16, T0), (25, T1), (30, 9)],
        )

    def test_prefill_chunks_encode_a_video_once_when_the_cache_holds_it(self):
        """Chunks outside a video's span run no encoder.

        A chunk holding only a stamp and soundtrack between two frames is
        rewritten, so it takes the video path; the last soundtrack run never
        moves, so its chunk does not. A cache too small for the embedding
        re-encodes the video on every chunk that meets its span.
        """
        V, S, T0, T1 = 5, 6, 7, 8
        ids = [9, 9, 9, T0, V, V, V, V, S, S, T1, V, V, V, V, S, S, 9, 9, 9]
        video = MultimodalDataItem(
            modality=Modality.VIDEO,
            feature=torch.zeros(2, 1),
            offsets=[(4, 7), (11, 14)],
            model_specific_data={"thw_grids": [(2, 2, 2)]},
        )
        soundtrack = MultimodalDataItem(
            modality=Modality.AUDIO,
            feature=torch.ones(4, 1),
            offsets=[(8, 9), (15, 16)],
            model_specific_data={GigaChat35OmniVideoEmbedding.SOUNDTRACK_OF_KEY: 0},
        )

        for item in (video, soundtrack):
            item.set_pad_value()

        padded = self._pad(ids, [video, soundtrack], video_token_id=V, audio_token_id=S)
        calls = []

        def encode(items):
            calls.append(items)
            return GigaChat35OmniVideoEmbedding(
                embedding=torch.zeros(8, 1), frame_counts=[[4, 4]]
            )

        # (prefix, length, video rows in the chunk); None means no encoder run.
        chunks = [
            (0, 3, None),
            (3, 5, 4),
            (8, 3, 0),
            (11, 4, 4),
            (15, 2, None),
            (17, 3, None),
        ]
        video_values = set(video.padding_values())

        for cache_size, expected_calls in [(1 << 20, 1), (16, 3)]:
            calls.clear()

            with (
                self.subTest(cache_size=cache_size),
                patch.object(
                    mm_schedule, "embedding_cache", MultiModalStaticCache(cache_size)
                ),
            ):
                for prefix, length, rows in chunks:
                    input_ids = torch.tensor(padded[prefix : prefix + length])
                    embedding, new_ids = mm_schedule._get_chunked_prefill_embedding(
                        encode,
                        [video],
                        [0, 1],
                        [prefix],
                        [length],
                        [video.offsets],
                        input_ids,
                    )
                    # The rewrite changes the pad values, not their positions.
                    self.assertEqual(
                        [token in video_values for token in new_ids.tolist()],
                        [token in video_values for token in input_ids.tolist()],
                    )
                    self.assertEqual(
                        None if embedding is None else len(embedding), rows
                    )

                self.assertEqual(len(calls), expected_calls)

    def test_soundtrack_offsets_move_before_audio_is_embedded(self):
        """`embed_mm_inputs` embeds videos first, so audio reads moved offsets.

        The first frame keeps 6 tokens, so the soundtrack run after it moves
        from (6, 7) to (8, 9): a chunk ending at 8 holds one soundtrack token.
        """
        V, S, T0, T1 = 5, 6, 7, 8
        ids = [9, T0, V, V, V, V, S, S, T1, V, V, V, V, S, S, 9]
        video = MultimodalDataItem(
            modality=Modality.VIDEO,
            feature=torch.zeros(2, 1),
            offsets=[(2, 5), (9, 12)],
            model_specific_data={"thw_grids": [(2, 2, 2)]},
        )
        soundtrack = MultimodalDataItem(
            modality=Modality.AUDIO,
            feature=torch.ones(4, 1),
            offsets=[(6, 7), (13, 14)],
            model_specific_data={GigaChat35OmniVideoEmbedding.SOUNDTRACK_OF_KEY: 0},
        )

        for item in (video, soundtrack):
            item.set_pad_value()

        items = [video, soundtrack]
        padded = self._pad(ids, items, video_token_id=V, audio_token_id=S)
        embedders = {
            Modality.VIDEO: lambda _: GigaChat35OmniVideoEmbedding(
                embedding=100 + torch.arange(8.0).unsqueeze(1), frame_counts=[[6, 2]]
            ),
            Modality.AUDIO: lambda _: 200 + torch.arange(4.0).unsqueeze(1),
        }
        text = nn.Embedding(10, 1)
        nn.init.zeros_(text.weight)

        with patch.object(
            mm_schedule, "embedding_cache", MultiModalStaticCache(1 << 20)
        ):
            embeds, _ = mm_utils.embed_mm_inputs(
                mm_inputs_list=[
                    MultimodalInputs(mm_items=items, video_token_id=V, audio_token_id=S)
                ],
                extend_prefix_lens=[0],
                extend_seq_lens=[9],
                input_ids=torch.tensor(padded[:9]),
                input_embedding=text,
                data_embedding_func_mapping=embedders,
                extend_starts=[0],
            )

        self.assertEqual(
            embeds.squeeze(1).tolist(), [0, 0, 100, 101, 102, 103, 104, 105, 200]
        )

    def test_pad_input_ids_saves_each_video_layout(self):
        """Every video is laid out after encoding; a soundtrack moves with its video only."""
        V, A = 5, 6
        items = [
            MultimodalDataItem(
                modality=Modality.VIDEO, feature=torch.zeros(1), offsets=[(0, 1)]
            ),
            MultimodalDataItem(
                modality=Modality.VIDEO, feature=torch.ones(1), offsets=[(3, 4)]
            ),
            MultimodalDataItem(
                modality=Modality.AUDIO,
                feature=torch.zeros(2),
                offsets=[(5, 5)],
                model_specific_data={GigaChat35OmniVideoEmbedding.SOUNDTRACK_OF_KEY: 1},
            ),
            MultimodalDataItem(
                modality=Modality.AUDIO, feature=torch.ones(2), offsets=[(7, 8)]
            ),
        ]

        for item in items:
            item.set_pad_value()

        padded = self._pad(
            [V, V, 0, V, V, A, 0, A, A], items, video_token_id=V, audio_token_id=A
        )

        self.assertEqual(padded[5], items[2].pad_value)

        for item, span in zip(items[:2], (padded[0:2], padded[3:5])):
            self.assertTrue(
                item.model_specific_data[
                    GigaChat35OmniVideoEmbedding.REWRITES_LAYOUT_KEY
                ]
            )
            self.assertEqual(item.span_ids, span)

        self.assertNotIn(
            GigaChat35OmniVideoEmbedding.SOUNDTRACK_KEY, items[0].model_specific_data
        )
        self.assertIs(items[1].soundtrack, items[2])
        self.assertEqual(items[1].soundtrack_offsets, [(5, 5)])


if __name__ == "__main__":
    unittest.main()
