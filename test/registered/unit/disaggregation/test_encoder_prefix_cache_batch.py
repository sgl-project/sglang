"""CPU regressions through the actual encoder item builder and embedding cache.

Image/video forwards and the audio tower are synthetic; Qwen3-Omni audio
packing, model-facing item construction, hashes, slicing and cache accounting
use their real owners. This does not load weights or qualify GPU/EPD behavior.
"""

import asyncio
import hashlib
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

import torch

import sglang.srt.disaggregation.encoder.server as encoder_server
from sglang.srt.disaggregation.encoder.preprocessor import EncoderPreprocessResult
from sglang.srt.disaggregation.encoder.server import EncodeContext, MMEncoder
from sglang.srt.managers.req_types import MultimodalDataItem
from sglang.srt.mem_cache.multimodal_cache import EmbeddingResult
from sglang.srt.models.qwen3_omni_moe import (
    Qwen3OmniMoeThinkerForConditionalGeneration,
)
from sglang.srt.multimodal.encoder_preprocessing import EncoderPreprocessOutput
from sglang.srt.multimodal.modality import Modality
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _FeatureFn:
    def __init__(self, *, as_list=False):
        self.calls = []
        self.as_list = as_list

    def __call__(self, items):
        features = [item.feature for item in items]
        self.calls.append([feature.clone() for feature in features])
        return features if self.as_list else torch.cat(features, dim=0)


class _PackedAudioTower(torch.nn.Module):
    """Synthetic neural boundary: sum each clip packed by the real consumer."""

    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.empty(0), requires_grad=False)
        self.calls = []

    @property
    def dtype(self):
        return self.anchor.dtype

    def forward(self, features, *, feature_lens):
        lengths = feature_lens.tolist()
        self.calls.append(lengths)
        clips = features.split(lengths, dim=1)
        return SimpleNamespace(
            last_hidden_state=torch.stack([clip.sum(dim=1) for clip in clips])
        )


def _ctx(
    features,
    feature_fn,
    *,
    itemwise=False,
    items_per_req=None,
    modality=Modality.IMAGE,
    metadata=None,
):
    grid = torch.tensor([[1, 1, len(feature)] for feature in features], device="cpu")
    joined = torch.cat(features, dim=0)
    feature_key, grid_key = {
        Modality.IMAGE: ("pixel_values", "image_grid_thw"),
        Modality.VIDEO: ("pixel_values_videos", "video_grid_thw"),
    }[modality]
    inputs = {feature_key: joined, grid_key: grid, **(metadata or {})}
    if itemwise:
        inputs = EncoderPreprocessOutput(
            inputs,
            mm_items=[
                MultimodalDataItem.from_dict(
                    {
                        "modality": modality,
                        "feature": feature,
                        "model_specific_data": {
                            grid_key: grid[index : index + 1],
                        },
                    }
                )
                for index, feature in enumerate(features)
            ],
        )
    return EncodeContext(
        req_id="encoder-prefix-cache-cpu",
        modality=modality,
        preprocess_result=EncoderPreprocessResult(
            mm_inputs=inputs,
            grid_thw=grid,
            token_counts=[len(feature) for feature in features],
        ),
        get_feature_fn=feature_fn,
        mm_feature=joined,
        num_items=len(features),
        items_per_req=items_per_req or [1] * len(features),
        aux_data={},
        str_mm_hashes=None,
        use_global_cache=False,
        is_health_check=False,
    )


def _encoder(*, capacity=1 << 20):
    encoder = MMEncoder.__new__(MMEncoder)
    encoder.model_type = "qwen2_5_vl"
    encoder.cache_namespace = (
        "sha256:" + hashlib.sha256(b"encoder-prefix-cache-cpu").hexdigest()
    )
    encoder.preprocessor = SimpleNamespace(
        get_num_patches=lambda grid, modality: int(grid.prod().item())
    )
    encoder.mm_cache = encoder_server.MultiModalStaticCache(capacity)
    encoder.mm_cache_lock = asyncio.Lock()
    return encoder


class TestEncoderPrefixCacheBatch(CustomTestCase):
    def setUp(self):
        self.a = torch.tensor([[1.0, 1.1], [1.2, 1.3]], device="cpu")
        self.b = torch.tensor([[2.0, 2.1]], device="cpu")
        self.c = torch.tensor([[3.0, 3.1]], device="cpu")
        self.metrics = Mock()
        self.mm_patch = patch.object(
            encoder_server,
            "get_mm",
            return_value=SimpleNamespace(enable_prefix_mm_cache=True),
        )
        self.metrics_patch = patch.object(
            encoder_server, "encoder_metrics_collector", self.metrics
        )
        self.mm_patch.start()
        self.metrics_patch.start()
        self.addCleanup(self.mm_patch.stop)
        self.addCleanup(self.metrics_patch.stop)

    def _run(self, encoder, ctx, *, keep_on_gpu=False):
        return asyncio.run(
            encoder._compute_direct_embedding(ctx, keep_on_gpu=keep_on_gpu)
        )

    def test_real_legacy_builder_cold_repeat_reorder_and_partial_hit(self):
        encoder, forward = _encoder(), _FeatureFn()
        cold = _ctx([self.a, self.b], forward, items_per_req=[1, 1])
        # Legacy processors aggregate two grids into one model-facing item.
        built = encoder._build_model_mm_items(
            cold.mm_feature, cold.preprocess_result, [0, 1], Modality.IMAGE
        )
        self.assertEqual(len(built), 1)
        first = self._run(encoder, cold)
        keys = encoder._calculate_hashes_from_features(
            cold.mm_feature, cold.preprocess_result, Modality.IMAGE
        )
        self.assertEqual(
            [
                encoder.mm_cache.get_single(key).embedding.untyped_storage().nbytes()
                for key in keys
            ],
            [
                self.a.numel() * self.a.element_size(),
                self.b.numel() * self.b.element_size(),
            ],
        )
        again = self._run(encoder, cold)
        reordered = self._run(encoder, _ctx([self.b, self.a], forward))
        partial = self._run(encoder, _ctx([self.a, self.c], forward))

        self.assertTrue(torch.equal(first, torch.cat([self.a, self.b])))
        self.assertTrue(torch.equal(again, first))
        self.assertTrue(torch.equal(reordered, torch.cat([self.b, self.a])))
        self.assertTrue(torch.equal(partial, torch.cat([self.a, self.c])))
        self.assertEqual(len(forward.calls), 2)
        self.assertEqual(len(forward.calls[0]), 1)
        self.assertTrue(torch.equal(forward.calls[-1][0], self.c))
        self.assertEqual(
            self.metrics.record_cache_tokens.call_args_list,
            [
                call(0, 3, modality="image"),
                call(3, 3, modality="image"),
                call(3, 3, modality="image"),
                call(2, 3, modality="image"),
            ],
        )
        self.assertEqual(
            self.metrics.record_cache_files.call_args_list,
            [
                call(0, 2, modality="image"),
                call(2, 2, modality="image"),
                call(2, 2, modality="image"),
                call(1, 2, modality="image"),
            ],
        )

    def test_real_itemwise_builder_reuses_each_grid(self):
        encoder, forward = _encoder(), _FeatureFn()
        ctx = _ctx([self.a, self.b], forward, itemwise=True)
        built = encoder._build_model_mm_items(
            ctx.mm_feature, ctx.preprocess_result, [0, 1], Modality.IMAGE
        )
        self.assertEqual(len(built), 2)
        self._run(encoder, ctx)
        reordered = self._run(encoder, _ctx([self.b, self.a], forward, itemwise=True))
        partial = self._run(encoder, _ctx([self.a, self.c], forward, itemwise=True))
        self.assertTrue(torch.equal(reordered, torch.cat([self.b, self.a])))
        self.assertTrue(torch.equal(partial, torch.cat([self.a, self.c])))
        self.assertEqual([len(items) for items in forward.calls], [2, 1])
        self.assertTrue(torch.equal(forward.calls[-1][0], self.c))

    def test_partial_audio_hit_selects_masks_for_the_real_qwen_consumer(self):
        """A cached clip must not leave its mask beside only the missing clip.

        Qwen3-Omni packs features using the selected clips' attention masks.
        Keeping a full-batch mask on a partial cache miss raises IndexError.
        Only the neural tower is synthetic; packing, building and cache are real.
        """
        tower = _PackedAudioTower()
        thinker = Qwen3OmniMoeThinkerForConditionalGeneration.__new__(
            Qwen3OmniMoeThinkerForConditionalGeneration
        )
        torch.nn.Module.__init__(thinker)
        thinker.audio_tower = tower
        encoder = _encoder()
        encoder.model_type = "qwen3_omni_moe"
        mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]], device="cpu")
        lengths = torch.tensor([2, 3], device="cpu")

        def context(second_value):
            features = torch.stack([torch.ones(2, 4), torch.full((2, 4), second_value)])
            return EncodeContext(
                req_id="encoder-prefix-cache-audio-cpu",
                modality=Modality.AUDIO,
                preprocess_result=EncoderPreprocessResult(
                    mm_inputs={
                        "input_features": features,
                        "feature_attention_mask": mask,
                        "audio_feature_lens_raw": lengths,
                        "audio_feature_lens": lengths,
                    },
                    grid_thw=lengths,
                    token_counts=[1, 1],
                ),
                get_feature_fn=thinker.get_audio_feature,
                mm_feature=features,
                num_items=2,
                items_per_req=[1, 1],
                aux_data={},
                str_mm_hashes=None,
                use_global_cache=False,
                is_health_check=False,
            )

        cold = self._run(encoder, context(2.0))
        partial = self._run(encoder, context(3.0))
        self.assertTrue(torch.equal(cold, torch.tensor([[2.0, 2.0], [6.0, 6.0]])))
        self.assertTrue(torch.equal(partial, torch.tensor([[2.0, 2.0], [9.0, 9.0]])))
        self.assertEqual(tower.calls, [[2, 3], [3]])
        self.assertTrue(torch.equal(mask, torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]])))

    def test_video_identity_uses_only_its_own_timing_metadata(self):
        """Another clip's timing must not invalidate an unchanged video key.

        Timing remains part of identity, but an item's key must be independent
        of its batch position and its companions' metadata.
        """
        for model_type, metadata_key, metadata_values in (
            ("qwen3_vl", "video_timestamps", [[0.0, 0.5], [0.0, 0.75]]),
            ("qwen3_omni_moe", "second_per_grid_ts", torch.tensor([0.5, 0.5])),
        ):
            with self.subTest(model_type=model_type):
                encoder = _encoder()
                encoder.model_type = model_type

                def key(features, values, index=0):
                    ctx = _ctx(
                        features,
                        _FeatureFn(),
                        modality=Modality.VIDEO,
                        metadata={metadata_key: values},
                    )
                    return encoder._calculate_hashes_from_features(
                        ctx.mm_feature, ctx.preprocess_result, Modality.VIDEO
                    )[index]

                alone_key = key([self.a], metadata_values[:1])
                batched_key = key([self.a, self.b], metadata_values)
                reversed_values = (
                    metadata_values.flip(0)
                    if isinstance(metadata_values, torch.Tensor)
                    else metadata_values[::-1]
                )
                reordered_key = key([self.b, self.a], reversed_values, index=1)
                self.assertEqual(alone_key, batched_key)
                self.assertEqual(alone_key, reordered_key)
                changed = (
                    torch.tensor([1.0])
                    if isinstance(metadata_values, torch.Tensor)
                    else [[0.0, 1.0]]
                )
                self.assertNotEqual(alone_key, key([self.a], changed))

    def test_rank_zero_stages_cached_grids_for_each_request(self):
        """Cache reordering must preserve each request's rows and grid metadata.

        A legacy aggregate forward item must not collapse request boundaries
        when cached grids and newly encoded grids are combined.
        """
        encoder, forward = _encoder(), _FeatureFn()
        encoder.rank = 0
        encoder.req_states = {}
        encoder.abandoned_req_ids = set()
        d = torch.tensor([[4.0, 4.1], [4.2, 4.3], [4.4, 4.5]], device="cpu")
        a_size, b_size, c_size, d_size = (101, 103), (211, 223), (307, 311), (401, 409)
        rounds = (
            (
                "cold",
                [self.a, self.b, self.c],
                [a_size, b_size, c_size],
                [
                    (
                        torch.cat([self.a, self.b]),
                        [[1, 1, 2], [1, 1, 1]],
                        [a_size, b_size],
                    ),
                    (self.c, [[1, 1, 1]], [c_size]),
                ],
            ),
            (
                "reordered",
                [self.b, self.c, self.a],
                [b_size, c_size, a_size],
                [
                    (
                        torch.cat([self.b, self.c]),
                        [[1, 1, 1], [1, 1, 1]],
                        [b_size, c_size],
                    ),
                    (self.a, [[1, 1, 2]], [a_size]),
                ],
            ),
            (
                "partial",
                [self.c, d, self.a],
                [c_size, d_size, a_size],
                [
                    (torch.cat([self.c, d]), [[1, 1, 1], [1, 1, 3]], [c_size, d_size]),
                    (self.a, [[1, 1, 2]], [a_size]),
                ],
            ),
        )

        async def exercise():
            for name, features, original_sizes, expected in rounds:
                with self.subTest(round=name):
                    ctx = _ctx(features, forward, items_per_req=[2, 1])
                    ctx.aux_data["original_image_sizes"] = original_sizes
                    requests = [
                        {"req_id": f"{name}-multi", "num_parts": 3, "part_idx": 1},
                        {"req_id": f"{name}-single", "num_parts": 4, "part_idx": 2},
                    ]
                    ctx.req_id = requests[0]["req_id"]
                    states = [
                        encoder._acquire_encode_ref(req["req_id"]) for req in requests
                    ]
                    try:
                        embedding = await encoder._compute_embedding(
                            ctx, keep_on_gpu=False
                        )
                        results = encoder._stage_embeddings(
                            ctx, requests, embedding, keep_on_gpu=False
                        )
                        self.assertEqual(
                            results,
                            [
                                (rows.nbytes, len(rows), 2, None, None)
                                for rows, _, _ in expected
                            ],
                        )
                        for req, state, (rows, grid, sizes) in zip(
                            requests, states, expected
                        ):
                            self.assertTrue(state.embedding_ready.is_set())
                            staged = state.embedding_data
                            self.assertTrue(torch.equal(staged.embedding, rows))
                            self.assertTrue(
                                torch.equal(staged.grid_dim, torch.tensor(grid))
                            )
                            self.assertEqual(staged.original_image_sizes, sizes)
                            self.assertEqual(
                                (staged.req_id, staged.num_parts, staged.part_idx),
                                (req["req_id"], req["num_parts"], req["part_idx"]),
                            )
                    finally:
                        for state in states:
                            await encoder._release_encode_ref(state)
                        self.assertEqual(
                            [state.active_encodes for state in states], [0, 0]
                        )

        asyncio.run(exercise())
        self.assertEqual([len(items) for items in forward.calls], [1, 1])
        self.assertTrue(torch.equal(forward.calls[-1][0], d))
        self.assertEqual(
            self.metrics.record_cache_tokens.call_args_list,
            [
                call(0, 4, modality="image"),
                call(4, 4, modality="image"),
                call(3, 6, modality="image"),
            ],
        )
        self.assertEqual(
            self.metrics.record_cache_files.call_args_list,
            [
                call(0, 3, modality="image"),
                call(3, 3, modality="image"),
                call(2, 3, modality="image"),
            ],
        )

    def test_stale_grid_entry_reencodes_only_that_grid(self):
        encoder, forward = _encoder(), _FeatureFn()
        ctx = _ctx([self.a, self.b], forward)
        self._run(encoder, ctx)
        key = encoder._calculate_hashes_from_features(
            ctx.mm_feature, ctx.preprocess_result, Modality.IMAGE
        )[0]
        encoder.mm_cache.free(key, None)
        encoder.mm_cache.set(
            key, EmbeddingResult(embedding=torch.zeros(3, 2, device="cpu"))
        )
        result = self._run(encoder, ctx)
        self.assertTrue(torch.equal(result, torch.cat([self.a, self.b])))
        self.assertEqual(len(forward.calls), 2)
        self.assertTrue(torch.equal(forward.calls[-1][0], self.a))
        self.metrics.record_cache_tokens.assert_called_with(1, 3, modality="image")
        self.metrics.record_cache_files.assert_called_with(1, 2, modality="image")

    def test_admitted_entries_count_actual_capacity_evictions(self):
        encoder, forward = _encoder(capacity=16), _FeatureFn()
        self._run(encoder, _ctx([self.a, self.b], forward))
        self.assertEqual(len(encoder.mm_cache), 1)
        self.assertEqual(encoder.mm_cache.current_size, 8)
        self.metrics.inc_cache_evictions.assert_called_once_with(
            modality="image", count=1
        )
        result = self._run(encoder, _ctx([self.b, self.c], forward))
        self.assertTrue(torch.equal(result, torch.cat([self.b, self.c])))
        self.assertTrue(torch.equal(forward.calls[-1][0], self.c))
        self.assertEqual(len(encoder.mm_cache), 2)
        self.assertEqual(encoder.mm_cache.current_size, 16)
        self.metrics.record_cache_tokens.assert_called_with(1, 2, modality="image")
        self.metrics.record_cache_files.assert_called_with(1, 2, modality="image")
        self.assertEqual(self.metrics.inc_cache_evictions.call_count, 1)

    def test_uncacheable_grid_is_not_a_hit_or_an_eviction(self):
        encoder, forward = _encoder(capacity=3), _FeatureFn()
        ctx = _ctx([self.a, self.b], forward)
        first = self._run(encoder, ctx)
        again = self._run(encoder, ctx)
        self.assertTrue(torch.equal(first, again))
        self.assertEqual(len(forward.calls), 2)
        self.assertEqual(len(encoder.mm_cache), 0)
        self.assertEqual(encoder.mm_cache.current_size, 0)
        self.assertEqual(
            self.metrics.record_cache_tokens.call_args_list,
            [call(0, 3, modality="image"), call(0, 3, modality="image")],
        )
        self.metrics.inc_cache_evictions.assert_not_called()

    def test_list_forward_results_are_sliced_by_grid_counts(self):
        for itemwise in (False, True):
            with self.subTest(itemwise=itemwise):
                encoder, forward = _encoder(), _FeatureFn(as_list=True)
                ctx = _ctx([self.a, self.b], forward, itemwise=itemwise)
                first = self._run(encoder, ctx)
                again = self._run(encoder, ctx)
                self.assertTrue(torch.equal(first, torch.cat([self.a, self.b])))
                self.assertTrue(torch.equal(again, first))
                self.assertEqual(len(forward.calls), 1)

    def test_cache_bypass_preserves_the_forward_tensor(self):
        """Bypassing the cache must not duplicate the full model output."""
        output = torch.cat([self.a, self.b])
        for enabled, health_check, keep_on_gpu in (
            (True, False, True),
            (False, False, False),
            (True, True, False),
        ):
            with self.subTest(
                enabled=enabled,
                health_check=health_check,
                keep_on_gpu=keep_on_gpu,
            ):
                encoder = _encoder()
                ctx = _ctx([self.a, self.b], lambda items: output)
                ctx.is_health_check = health_check
                with patch.object(
                    encoder_server,
                    "get_mm",
                    return_value=SimpleNamespace(enable_prefix_mm_cache=enabled),
                ):
                    result = self._run(encoder, ctx, keep_on_gpu=keep_on_gpu)
                self.assertIs(result, output)

    def test_health_checks_and_gpu_output_bypass_prefix_cache(self):
        for health_check, keep_on_gpu in ((True, False), (False, True)):
            with self.subTest(health_check=health_check, keep_on_gpu=keep_on_gpu):
                encoder, forward = _encoder(), _FeatureFn()
                ctx = _ctx([self.a, self.b], forward)
                ctx.is_health_check = health_check
                self.metrics.reset_mock()
                result = self._run(encoder, ctx, keep_on_gpu=keep_on_gpu)
                self.assertTrue(torch.equal(result, torch.cat([self.a, self.b])))
                self.assertEqual(len(encoder.mm_cache), 0)
                self.metrics.record_cache_tokens.assert_not_called()
                if health_check:
                    self.metrics.observe_model_forward.assert_not_called()


if __name__ == "__main__":
    unittest.main()
