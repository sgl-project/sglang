"""Unit tests for MLA host-dedup draft-cache planning."""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.managers.cache_controller import CacheOperation, HiCacheController
from sglang.srt.mem_cache import kv_cache_builder, mla_host_dedup
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.speculative import base_spec_worker
from sglang.srt.speculative.base_spec_worker import (
    BaseSpecWorker,
    HiCacheDraftMode,
    HiCacheDraftPlan,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _nextn_worker(*, enable_dedup: bool):
    spec_algorithm = SimpleNamespace(
        is_eagle=lambda: True,
        is_eagle3=lambda: False,
        is_dspark=lambda: False,
    )
    target_runner = SimpleNamespace(
        mtp_draft_device_pools=(),
        spec_algorithm=spec_algorithm,
    )
    draft_pool = object()
    draft_runner = SimpleNamespace(
        token_to_kv_pool=draft_pool,
        model_config=SimpleNamespace(
            num_nextn_predict_layers=1,
            hf_config=SimpleNamespace(architectures=["Glm4MoeForCausalLM"]),
        ),
    )
    worker = SimpleNamespace(
        target_worker=SimpleNamespace(model_runner=target_runner),
        server_args=SimpleNamespace(
            enable_hierarchical_cache=True,
            enable_mla_hicache_host_dedup=enable_dedup,
        ),
        _draft_model_runners=lambda: (draft_runner,),
    )
    return worker, target_runner, draft_pool


class TestHiCacheMLADedupDraftPlan(unittest.TestCase):
    def test_disabled_flag_does_not_create_dedup_context(self):
        with (
            mock.patch.object(mla_host_dedup, "mla_host_dedup_eligible") as eligible,
            mock.patch.object(mla_host_dedup.MLAHostDedupBroadcaster, "build") as build,
        ):
            context = mla_host_dedup.maybe_create_mla_host_dedup_context(
                kv_cache=object(),
                tp_group=object(),
                attn_cp_group=None,
                attn_tp_group=None,
                storage_backend=None,
                enabled=False,
            )

        self.assertIsNone(context)
        eligible.assert_not_called()
        build.assert_not_called()

    def test_disabled_flag_keeps_original_write_path(self):
        op = CacheOperation(torch.arange(2), torch.arange(2), node_id=1)
        completion = SimpleNamespace(
            start_event=object(), finish_event=object(), timing_enabled=False
        )
        controller = HiCacheController.__new__(HiCacheController)
        controller.mla_dedup = None
        controller.write_queue = [op]
        controller.ack_write_queue = []
        controller._move_write_operation = mock.Mock(
            return_value=(op.host_indices, op.device_indices, None)
        )
        controller._move_mla_write_operation = mock.Mock()
        controller._l2_transfers = mock.Mock(return_value=[mock.sentinel.transfer])
        controller._mla_l2_transfers = mock.Mock()
        controller._num_tokens_by_pool = mock.Mock(return_value={})
        controller._transfer_num_bytes = mock.Mock(return_value=0)
        controller._mla_transfer_num_bytes = mock.Mock()
        controller.l2_transfer_engine = SimpleNamespace(
            submit_device_to_host=mock.Mock(return_value=completion)
        )

        controller.start_writing()

        controller._move_write_operation.assert_called_once_with(op)
        controller._l2_transfers.assert_called_once()
        controller._move_mla_write_operation.assert_not_called()
        controller._mla_l2_transfers.assert_not_called()
        controller._mla_transfer_num_bytes.assert_not_called()

    def test_disabled_flag_keeps_original_load_path(self):
        op = CacheOperation(torch.arange(2), torch.arange(2), node_id=1)
        start_event = mock.Mock()
        producer_event = SimpleNamespace(start_event=start_event, complete=mock.Mock())
        completion = SimpleNamespace(
            start_event=object(), finish_event=object(), timing_enabled=False
        )
        controller = HiCacheController.__new__(HiCacheController)
        controller.mla_dedup = None
        controller.load_queue = [op]
        controller.ack_load_queue = []
        controller.layer_num = 2
        controller.layer_done_counter = SimpleNamespace(
            update_producer=mock.Mock(return_value=0), events=[producer_event]
        )
        controller._move_op_indices = mock.Mock(
            return_value=(op.host_indices, op.device_indices, None)
        )
        controller._l2_load_transfers = mock.Mock(return_value=[mock.sentinel.transfer])
        controller._start_loading_mla = mock.Mock()
        controller._num_tokens_by_pool = mock.Mock(return_value={})
        controller._transfer_num_bytes = mock.Mock(return_value=0)
        controller.l2_transfer_engine = SimpleNamespace(
            submit_host_to_device=mock.Mock(return_value=completion)
        )

        self.assertEqual(controller.start_loading(), 0)

        controller._move_op_indices.assert_called_once_with(op)
        controller._l2_load_transfers.assert_called_once()
        controller._start_loading_mla.assert_not_called()

    def test_dedup_keeps_nextn_draft_rank_local(self):
        worker, target_runner, draft_pool = _nextn_worker(enable_dedup=True)

        with mock.patch.object(
            base_spec_worker,
            "get_memory",
            return_value=SimpleNamespace(enable_hierarchical_cache=True),
        ):
            plan = BaseSpecWorker._build_hicache_draft_plan(worker)

        self.assertEqual(plan.mode, HiCacheDraftMode.SIDECAR)
        self.assertEqual(plan.device_pools, (draft_pool,))
        self.assertEqual(target_runner.mtp_draft_device_pools, ())

    def test_normal_hicache_still_packs_nextn_draft(self):
        worker, target_runner, draft_pool = _nextn_worker(enable_dedup=False)

        with mock.patch.object(
            base_spec_worker,
            "get_memory",
            return_value=SimpleNamespace(enable_hierarchical_cache=True),
        ):
            plan = BaseSpecWorker._build_hicache_draft_plan(worker)

        self.assertEqual(plan.mode, HiCacheDraftMode.PACKED)
        self.assertEqual(plan.device_pools, (draft_pool,))
        self.assertEqual(target_runner.mtp_draft_device_pools, (draft_pool,))

    def test_unified_cache_uses_independent_draft_pool_with_dedup(self):
        tree_cache = UnifiedRadixCache.__new__(UnifiedRadixCache)
        draft_pool = object()
        plan = HiCacheDraftPlan(
            mode=HiCacheDraftMode.SIDECAR,
            device_pools=(draft_pool,),
        )
        server_args = SimpleNamespace(enable_mla_hicache_host_dedup=True)

        with mock.patch.object(
            kv_cache_builder, "_register_legacy_hicache_draft"
        ) as register_legacy:
            kv_cache_builder.maybe_register_hicache_draft(
                tree_cache=tree_cache,
                draft_plan=plan,
                server_args=server_args,
                page_size=64,
            )

        register_legacy.assert_called_once_with(
            tree_cache=tree_cache,
            draft_pool=draft_pool,
            server_args=server_args,
            page_size=64,
        )


if __name__ == "__main__":
    unittest.main()
