# SPDX-License-Identifier: Apache-2.0

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.multimodal_gen.configs.pipeline_configs.minimax_h3 import (
    MiniMaxH3PipelineConfig,
)
from sglang.multimodal_gen.runtime.layers.attention.backends.subblock_sparse_attn import (
    SubBlockSparseAttentionImpl,
    _get_subblock_sparse_attention_runner,
    _sm90_sparse_attention,
    _sm100_sparse_attention,
    _sm120_sparse_attention,
)
from sglang.multimodal_gen.runtime.models.dits.minimax_h3 import (
    _minimax_h3_attention_core_impl,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.denoise_loop import (
    MiniMaxH3DenoiseBranch,
    _minimax_h3_subblock_sparse_query_block_mask,
    _minimax_h3_subblock_video_query_indices,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.packed_sequence import (
    minimax_h3_packed_sequence,
    minimax_h3_packed_sequence_ref2va_blocks,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.presentation import (
    IMAGE_PAD,
    VIDEO_PAD,
    minimax_h3_ref2va_video_presentation,
)
from sglang.multimodal_gen.runtime.platforms import (
    AttentionBackendEnum,
    current_platform,
)
from sglang.multimodal_gen.runtime.platforms.cuda import (
    _SubBlockSparseAttentionBackendResolver,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=9, suite="base-b-test-cpu")


class TestSubBlockSparseAttentionDispatch(CustomTestCase):
    def setUp(self):
        _get_subblock_sparse_attention_runner.cache_clear()
        self.addCleanup(_get_subblock_sparse_attention_runner.cache_clear)

    def test_dispatch_is_resolved_once_per_device(self):
        device = torch.device("cuda:0")
        with patch(
            "torch.cuda.get_device_capability", return_value=(9, 0)
        ) as get_capability:
            first = _get_subblock_sparse_attention_runner(device)
            second = _get_subblock_sparse_attention_runner(device)

        self.assertIs(first, _sm90_sparse_attention)
        self.assertIs(second, first)
        get_capability.assert_called_once_with(device)

    def test_dispatches_sm100(self):
        device = torch.device("cuda:0")
        with patch("torch.cuda.get_device_capability", return_value=(10, 0)):
            runner = _get_subblock_sparse_attention_runner(device)

        self.assertIs(runner, _sm100_sparse_attention)

    def test_dispatches_sm120(self):
        device = torch.device("cuda:0")
        with patch("torch.cuda.get_device_capability", return_value=(12, 0)):
            runner = _get_subblock_sparse_attention_runner(device)

        self.assertIs(runner, _sm120_sparse_attention)

    def test_platform_resolver_loads_sm120_dependency(self):
        capability = Mock(major=12, minor=0)
        capability.as_version_str.return_value = "12.0"
        platform = Mock()
        platform.get_device_capability.return_value = capability

        with patch(
            "sglang.multimodal_gen.runtime.layers.attention.backends."
            "subblock_sparse.load_bsa_attn_sm120_blk64_fwd"
        ) as load_sm120:
            resolved = _SubBlockSparseAttentionBackendResolver.resolve(platform)

        self.assertEqual(
            resolved,
            "sglang.multimodal_gen.runtime.layers.attention.backends."
            "subblock_sparse_attn.SubBlockSparseAttentionBackend",
        )
        load_sm120.assert_called_once_with()

    def test_sm120_adapter_forwards_subblock_plan(self):
        q = torch.empty((1, 64, 2, 128), dtype=torch.bfloat16)
        k = torch.empty((1, 65, 2, 128), dtype=torch.bfloat16)
        v = torch.empty_like(k)
        q2k_block_index = torch.zeros((1, 2, 1, 2), dtype=torch.int32)
        block_counts = torch.tensor([[[2], [1]]], dtype=torch.int32)
        expected = torch.empty_like(q)
        kernel = Mock(return_value=(expected, None))

        with patch(
            "sglang.multimodal_gen.runtime.layers.attention.backends."
            "subblock_sparse_attn.load_bsa_attn_sm120_blk64_fwd",
            return_value=kernel,
        ):
            result = _sm120_sparse_attention(
                q,
                k,
                v,
                q2k_block_index,
                topk=2,
                softmax_scale=0.125,
                block_counts=block_counts,
            )

        self.assertIs(result, expected)
        kernel.assert_called_once()
        args, kwargs = kernel.call_args
        self.assertIs(args[0], q)
        self.assertIs(args[1], k)
        self.assertIs(args[2], v)
        self.assertIs(args[3], q2k_block_index)
        self.assertEqual(args[4], 2)
        torch.testing.assert_close(
            kwargs["block_sizes"], torch.tensor([64, 1], dtype=torch.int32)
        )
        self.assertIs(kwargs["q2k_block_nums"], block_counts)
        self.assertEqual(kwargs["softmax_scale"], 0.125)

    def test_rejects_unsupported_compute_capability(self):
        device = torch.device("cuda:0")
        with patch("torch.cuda.get_device_capability", return_value=(10, 3)):
            with self.assertRaisesRegex(
                RuntimeError,
                "supports compute capability 9.0, 10.0, or 12.0;.*10.3 device",
            ):
                _get_subblock_sparse_attention_runner(device)


class TestSubBlockSparseAttentionModalities(CustomTestCase):
    def test_transformer_subblock_with_ring_fails_admission(self):
        config = MiniMaxH3PipelineConfig()
        server_args = SimpleNamespace(
            attention_backend="fa",
            ring_degree=2,
            resolve_component_attention_backend=lambda *_names: (
                AttentionBackendEnum.SUBBLOCK_SPARSE_ATTN,
                "transformer",
            ),
        )

        with (
            patch.object(current_platform, "is_mps", return_value=False),
            self.assertRaisesRegex(ValueError, "ring parallelism requires"),
        ):
            config.validate_server_args(server_args)

    @staticmethod
    def _run_attention_core_without_query_mask(
        *,
        sparse_ready: bool,
        min_seq_len: int,
    ) -> Mock:
        impl = Mock()
        impl._sparse_ready.return_value = sparse_ready
        impl.schedule = SimpleNamespace(min_seq_len=min_seq_len)
        q = torch.zeros(4, 1, 2)
        impl.forward_varlen.return_value = torch.zeros_like(q)
        attention = SimpleNamespace(
            _attention_impl=impl,
            _attention_backend_enum=AttentionBackendEnum.SUBBLOCK_SPARSE_ATTN,
        )
        _minimax_h3_attention_core_impl(
            attention,
            q,
            q,
            q,
            cu_seqlens=torch.tensor([0, 4], dtype=torch.int32),
            cu_seqlens_host=(0, 4),
            max_seqlen=4,
            ulysses_active=False,
            subblock_sparse_query_block_mask=None,
        )
        return impl

    def test_missing_query_mask_is_allowed_for_dense_fallback(self):
        impl = self._run_attention_core_without_query_mask(
            sparse_ready=False,
            min_seq_len=4,
        )

        impl.forward_varlen.assert_called_once()

    def test_missing_query_mask_is_allowed_when_segments_are_short(self):
        impl = self._run_attention_core_without_query_mask(
            sparse_ready=True,
            min_seq_len=8,
        )

        impl.forward_varlen.assert_called_once()

    def test_missing_query_mask_fails_only_when_sparse_attention_will_run(self):
        with self.assertRaisesRegex(
            ValueError,
            "when SubBlock sparse attention is active",
        ):
            self._run_attention_core_without_query_mask(
                sparse_ready=True,
                min_seq_len=4,
            )

    def test_fl2va_keyframe_images_remain_dense(self):
        packed = minimax_h3_packed_sequence(
            text_len=5,
            latent_t=2,
            latent_h=4,
            latent_w=4,
            audio_t=5,
            include_keyframe_cond=True,
            keyframe_frame_indices=[0, -1],
            frame_count=5,
            include_video_pos=True,
        )
        video_indices = _minimax_h3_subblock_video_query_indices(
            packed,
            None,
        )
        condition_image_indices = set(
            packed["img_pos"][~packed["update_mask"]].tolist()
        )

        torch.testing.assert_close(video_indices, packed["video_pos"])
        self.assertTrue(condition_image_indices.isdisjoint(video_indices.tolist()))

    def test_ref2va_images_are_dense_but_reference_and_target_video_are_sparse(self):
        packed = minimax_h3_packed_sequence_ref2va_blocks(
            text_len=5,
            latent_t=2,
            latent_h=4,
            latent_w=4,
            audio_t=5,
            ref_blocks=[
                {"kind": "image", "latent_h": 4, "latent_w": 4},
                {
                    "kind": "video_audio",
                    "ref_audio_t": 3,
                    "latent_t": 2,
                    "latent_h": 4,
                    "latent_w": 4,
                },
            ],
            include_video_pos=True,
        )
        text_video_mask = torch.tensor([False, True, False, True, False])
        video_indices = _minimax_h3_subblock_video_query_indices(
            packed,
            text_video_mask,
        )

        image_indices = set(packed["img_pos"].tolist()) - set(
            packed["video_pos"].tolist()
        )
        text_video_indices = set(packed["text_pos"][text_video_mask].tolist())
        text_non_video_indices = set(packed["text_pos"][~text_video_mask].tolist())

        video_index_set = set(video_indices.tolist())
        self.assertTrue(image_indices.isdisjoint(video_index_set))
        self.assertTrue(set(packed["audio_pos"].tolist()).isdisjoint(video_index_set))
        self.assertTrue(text_non_video_indices.isdisjoint(video_index_set))
        self.assertTrue(text_video_indices.issubset(video_index_set))
        self.assertTrue(set(packed["video_pos"].tolist()).issubset(video_index_set))

    def test_ref2va_presentation_marks_only_video_vision_blocks_sparse(self):
        class FakeTokenizer:
            _special_ids = {
                "<|vision_start|>": 10,
                "<|vision_end|>": 11,
                IMAGE_PAD: 12,
                VIDEO_PAD: 13,
            }

            def __call__(self, text, *, add_special_tokens):
                del add_special_tokens
                return {"input_ids": [100 + len(text)]}

            def convert_tokens_to_ids(self, token):
                return self._special_ids[token]

        ids, tags, video_mask = minimax_h3_ref2va_video_presentation(
            FakeTokenizer(),
            prompt="prompt",
            condition_labels=[("image", 1), ("video", 1)],
            image_token_count=2,
            video_block_token_counts=[[2]],
            video_block_timestamps=[[0.0]],
            return_video_mask=True,
        )

        self.assertFalse(video_mask[ids == 12].any())
        self.assertTrue(video_mask[ids == 13].all())
        self.assertFalse(video_mask[ids == 10].any())
        self.assertFalse(video_mask[ids == 11].any())
        self.assertEqual(int(video_mask.sum()), 2)
        self.assertEqual(tags[ids == 12].unique().tolist(), [0])
        self.assertEqual(tags[ids == 13].unique().tolist(), [0])

        default_result = minimax_h3_ref2va_video_presentation(
            FakeTokenizer(),
            prompt="prompt",
            condition_labels=[("video", 1)],
            image_token_count=None,
            video_block_token_counts=[[1]],
            video_block_timestamps=[[0.0]],
        )
        self.assertEqual(len(default_result), 2)

    def test_video_query_indices_validate_first_segment_bounds(self):
        for invalid in (
            torch.tensor([-1]),
            torch.tensor([5]),
            torch.tensor([2, 2]),
        ):
            with self.subTest(indices=invalid.tolist()), self.assertRaises(ValueError):
                _minimax_h3_subblock_sparse_query_block_mask(invalid, used_len=5)

    def test_ref2va_video_positions_are_subblock_only_metadata(self):
        kwargs = dict(
            text_len=3,
            latent_t=2,
            latent_h=4,
            latent_w=4,
            audio_t=3,
            ref_blocks=[
                {
                    "kind": "video",
                    "ref_audio_t": 0,
                    "latent_t": 2,
                    "latent_h": 4,
                    "latent_w": 4,
                }
            ],
        )

        ordinary = minimax_h3_packed_sequence_ref2va_blocks(**kwargs)
        subblock = minimax_h3_packed_sequence_ref2va_blocks(
            **kwargs,
            include_video_pos=True,
        )

        self.assertNotIn("video_pos", ordinary)
        self.assertIn("video_pos", subblock)
        self.assertTrue(
            set(subblock["video_pos"].tolist()).issubset(
                set(subblock["img_pos"].tolist())
            )
        )

    def test_non_subblock_branch_does_not_retain_dense_query_metadata(self):
        packed = minimax_h3_packed_sequence(
            text_len=3,
            latent_t=2,
            latent_h=4,
            latent_w=4,
            audio_t=3,
            include_keyframe_cond=False,
        )

        self.assertNotIn("video_pos", packed)
        branch = MiniMaxH3DenoiseBranch(
            packed=packed,
            text_embeddings=torch.zeros(3, 5120),
            token_tags=packed["token_tags"],
            video_query_indices=None,
            device=torch.device("cpu"),
        )

        self.assertNotIn(
            "subblock_sparse_query_block_mask",
            branch.static_kwargs,
        )

    def test_query_mask_marks_only_pure_video_blocks_sparse(self):
        sparse_query_block_mask = _minimax_h3_subblock_sparse_query_block_mask(
            torch.cat([torch.arange(64), torch.arange(128, 140)]),
            used_len=140,
        )
        torch.testing.assert_close(
            sparse_query_block_mask,
            torch.tensor([True, False, True]),
        )

    def test_hybrid_query_routing_uses_one_heterogeneous_bsa_call(self):
        impl = object.__new__(SubBlockSparseAttentionImpl)
        impl.softmax_scale = 2**-0.5
        impl.causal = False
        impl.schedule = SimpleNamespace(sparsity=0.75)
        plan = SimpleNamespace(
            index=torch.tensor(
                [[[[7, 1, 4], [6, 2, 0], [5, 0, 3]]]], dtype=torch.int32
            ),
            topk=3,
            num_blocks=8,
            density=3 / 8,
        )
        impl.router = Mock(route=Mock(return_value=plan))
        q = torch.zeros(1, 3 * 64, 1, 2)
        k = torch.zeros(1, 8 * 64, 1, 2)
        v = torch.zeros_like(k)
        sparse_query_block_mask = torch.tensor([True, False, True])
        impl.dense_impl = Mock()
        sparse_out = torch.ones_like(q)

        for runner, sparse_rows in (
            (_sm90_sparse_attention, ([1, 4, 7], [0, 3, 5])),
            (_sm100_sparse_attention, ([7, 1, 4], [5, 0, 3])),
            (_sm120_sparse_attention, ([7, 1, 4], [5, 0, 3])),
        ):
            with (
                self.subTest(runner=runner.__name__),
                patch(
                    "sglang.multimodal_gen.runtime.layers.attention.backends."
                    "subblock_sparse_attn._run_subblock_sparse_attention",
                    return_value=sparse_out,
                ) as run_sparse,
                patch(
                    "sglang.multimodal_gen.runtime.layers.attention.backends."
                    "subblock_sparse_attn._get_subblock_sparse_attention_runner",
                    return_value=runner,
                ),
            ):
                out = impl._sparse_attention(
                    q,
                    k,
                    v,
                    sparse_query_block_mask=sparse_query_block_mask,
                )

            impl.dense_impl.forward.assert_not_called()
            torch.testing.assert_close(out, sparse_out)
            routing_q = impl.router.route.call_args.args[0]
            self.assertEqual(routing_q.shape[1], 3 * 64)
            sparse_call = run_sparse.call_args.args
            self.assertEqual(sparse_call[0].shape[1], 3 * 64)
            self.assertIs(sparse_call[1], k)
            self.assertIs(sparse_call[2], v)
            self.assertEqual(sparse_call[4], 8)
            torch.testing.assert_close(
                sparse_call[6], torch.tensor([[[3, 8, 3]]], dtype=torch.int32)
            )
            block_index = sparse_call[3]
            self.assertEqual(block_index[0, 0, 0, :3].tolist(), sparse_rows[0])
            self.assertEqual(block_index[0, 0, 1].tolist(), list(range(8)))
            self.assertEqual(block_index[0, 0, 2, :3].tolist(), sparse_rows[1])


if __name__ == "__main__":
    unittest.main(verbosity=3)
