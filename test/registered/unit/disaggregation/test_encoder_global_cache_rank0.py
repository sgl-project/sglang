"""Regression tests for encoder rank-zero embedding cache ownership."""

import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import torch

import sglang.srt.disaggregation.encoder.server as encoder_server
from sglang.srt.disaggregation.encoder.server import MMEncoder
from sglang.srt.managers.schedule_batch import Modality
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestEncoderGlobalCache(CustomTestCase):
    def test_initialization_owns_storage_only_on_rank_zero(self):
        for enabled in (False, True):
            for tp_size, rank in ((1, 0), (2, 0), (2, 1)):
                with self.subTest(enabled=enabled, tp_size=tp_size, rank=rank):
                    encoder = MMEncoder.__new__(MMEncoder)
                    encoder.rank = rank
                    encoder._embedding_dims = {Modality.IMAGE: 4}
                    encoder._embedding_dtype = torch.float16
                    cpu_group = object()
                    parallel = SimpleNamespace(
                        tp_size=tp_size,
                        tp_group=SimpleNamespace(cpu_group=cpu_group),
                    )
                    with (
                        patch.object(
                            encoder_server,
                            "get_mm",
                            return_value=SimpleNamespace(
                                enable_mm_global_cache=enabled,
                                mm_global_cache_backend="mooncake",
                            ),
                        ),
                        patch.object(
                            encoder_server, "get_parallel", return_value=parallel
                        ),
                        patch.object(
                            torch.distributed,
                            "get_process_group_ranks",
                            return_value=list(range(tp_size)),
                        ) as get_ranks,
                        patch.object(
                            encoder_server, "create_custom_parallel_group"
                        ) as create_group,
                        patch(
                            "sglang.srt.mem_cache.embedding_store.EmbeddingStoreFactory.create_backend"
                        ) as create_backend,
                        patch(
                            "sglang.srt.mem_cache.embedding_cache_controller.EmbeddingCacheController"
                        ) as controller,
                    ):
                        encoder._init_global_cache()

                    self.assertEqual(encoder.mm_global_cache_enabled, enabled)
                    if enabled and tp_size > 1:
                        get_ranks.assert_called_once_with(cpu_group)
                        create_group.assert_called_once_with(
                            group_ranks=list(range(tp_size)), backend="gloo"
                        )
                        self.assertIs(
                            encoder.global_cache_tp_group, create_group.return_value
                        )
                    else:
                        get_ranks.assert_not_called()
                        create_group.assert_not_called()
                        self.assertIsNone(encoder.global_cache_tp_group)
                    if enabled and rank == 0:
                        create_backend.assert_called_once_with("mooncake")
                        controller.assert_called_once_with(
                            embedding_store=create_backend.return_value,
                            hidden_dims=encoder._embedding_dims,
                            all_rank_get=False,
                            dtype=torch.float16,
                        )
                        self.assertIs(encoder.mm_global_cache, controller.return_value)
                    else:
                        create_backend.assert_not_called()
                        controller.assert_not_called()
                        self.assertIsNone(encoder.mm_global_cache)

    def test_mask_broadcast_uses_encoder_group_without_controller(self):
        encoder = MMEncoder.__new__(MMEncoder)
        encoder.mm_global_cache = None
        encoder.global_cache_tp_group = object()
        mask = torch.zeros(2, dtype=torch.int32)
        for tp_size in (1, 2):
            with (
                self.subTest(tp_size=tp_size),
                patch.object(
                    encoder_server,
                    "get_parallel",
                    return_value=SimpleNamespace(tp_size=tp_size),
                ),
                patch.object(torch.distributed, "broadcast") as broadcast,
            ):
                encoder._broadcast_global_cache_mask(mask)
                if tp_size > 1:
                    broadcast.assert_called_once_with(
                        mask, src=0, group=encoder.global_cache_tp_group
                    )
                else:
                    broadcast.assert_not_called()

    def test_nonzero_rank_enters_cache_path_without_controller(self):
        async def run(enabled, health_check, use_mooncake):
            encoder = MMEncoder.__new__(MMEncoder)
            encoder.rank = 1
            encoder.mm_global_cache = None
            encoder.mm_global_cache_enabled = enabled
            encoder.use_mooncake = use_mooncake
            encoder.profiler = None
            ctx = object()
            encoder._prepare_encode_context_on_all_ranks = AsyncMock(return_value=ctx)
            encoder._compute_embedding = AsyncMock(return_value=None)
            encoder._stage_errors = Mock(side_effect=AssertionError("unexpected error"))
            req_id = (
                f"{encoder_server.HEALTH_CHECK_RID_PREFIX}-probe"
                if health_check
                else "req"
            )
            requests = [{"req_id": req_id}]

            result = await encoder.batch_encode(requests, Modality.IMAGE)

            encoder._prepare_encode_context_on_all_ranks.assert_awaited_once_with(
                requests,
                Modality.IMAGE,
                use_global_cache=enabled and not health_check,
                is_health_check=health_check,
            )
            encoder._compute_embedding.assert_awaited_once_with(
                ctx, keep_on_gpu=use_mooncake and not health_check
            )
            self.assertEqual(result, [(0, 0, 0, None, None)])

        for enabled in (False, True):
            for health_check in (False, True):
                for use_mooncake in (False, True):
                    with self.subTest(
                        enabled=enabled,
                        health_check=health_check,
                        use_mooncake=use_mooncake,
                    ):
                        asyncio.run(run(enabled, health_check, use_mooncake))

    def test_nonzero_rank_cache_pipeline_without_controller(self):
        scenarios = (
            ([1, 1], [0, 0], []),
            ([1, 0], [0, 0], [[1]]),
            ([0, 0], [0, 0], [[0, 1]]),
            ([1, 0], [1, 0], [[1], [0]]),
        )

        async def run(hit_mask, fallback_mask, expected, keep_on_gpu, modality):
            encoder = MMEncoder.__new__(MMEncoder)
            encoder.rank = 1
            encoder.mm_global_cache = None
            encoder._encode_missing = Mock(return_value=[torch.ones((2, 4))])
            masks = iter([hit_mask, fallback_mask])

            def receive_mask(tensor):
                tensor.copy_(torch.tensor(next(masks), dtype=tensor.dtype))

            encoder._broadcast_global_cache_mask = Mock(side_effect=receive_mask)
            ctx = SimpleNamespace(
                req_id="req",
                num_items=2,
                str_mm_hashes=None,
                modality=modality,
                mm_feature=None,
                get_feature_fn=Mock(),
                preprocess_result=SimpleNamespace(token_counts=[2, 2]),
            )
            result = await encoder._compute_global_cache_embedding(
                ctx, keep_on_gpu=keep_on_gpu
            )
            self.assertIsNone(result)
            self.assertEqual(encoder._broadcast_global_cache_mask.call_count, 2)
            self.assertEqual(
                [call.args[2] for call in encoder._encode_missing.call_args_list],
                expected,
            )

        for keep_on_gpu in (False, True):
            for modality in (Modality.IMAGE, Modality.VIDEO, Modality.AUDIO):
                for hit_mask, fallback_mask, expected in scenarios:
                    with self.subTest(
                        hit_mask=hit_mask,
                        fallback_mask=fallback_mask,
                        keep_on_gpu=keep_on_gpu,
                        modality=modality,
                    ):
                        asyncio.run(
                            run(
                                hit_mask, fallback_mask, expected, keep_on_gpu, modality
                            )
                        )


if __name__ == "__main__":
    unittest.main()
