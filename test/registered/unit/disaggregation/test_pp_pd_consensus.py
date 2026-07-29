import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.disaggregation.base import KVPoll  # noqa: E402
from sglang.srt.disaggregation.prefill import PrefillBootstrapQueue  # noqa: E402
from sglang.srt.disaggregation.utils import (  # noqa: E402
    _DRAFT_KV_LAYER_ID_BASE,
    build_transfer_entry_pairs,
)
from sglang.srt.managers.scheduler_pp_mixin import (  # noqa: E402
    _pp_merge_transfer_status,
)

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestPPPDConsensus(CustomTestCase):
    @staticmethod
    def _make_prefill_queue(pp_rank):
        class FakeKVArgs:
            pass

        class FakeManager:
            def __init__(self, kv_args, *_args):
                self.kv_args = kv_args

        class FakePool:
            start_layer = 4
            end_layer = 6
            head_num = 1
            page_size = 64

            @staticmethod
            def get_contiguous_buf_infos():
                return [10, 11], [100, 100], [10, 10]

        class FakeDraftPool:
            start_layer = 0
            end_layer = 1

            @staticmethod
            def get_contiguous_buf_infos():
                return [20], [100], [10]

        queue = PrefillBootstrapQueue.__new__(PrefillBootstrapQueue)
        queue.transfer_backend = "mooncake"
        queue.tp_rank = 0
        queue.pp_rank = pp_rank
        queue.pp_size = 2
        queue.scheduler = SimpleNamespace(
            ps=SimpleNamespace(dp_rank=0, gpu_id=0),
            server_args=SimpleNamespace(disaggregation_ib_device=None),
            model_config=SimpleNamespace(
                num_hidden_layers=8,
                get_total_num_kv_heads=lambda: 1,
            ),
            req_to_token_pool=None,
        )
        queue.token_to_kv_pool = FakePool()
        queue.draft_token_to_kv_pool = FakeDraftPool()
        queue.metadata_buffers = SimpleNamespace(
            get_buf_infos=lambda: ([], [], [])
        )
        queue.is_mla_backend = False
        return queue, FakeKVArgs, FakeManager

    def test_transfer_failure_overrides_ordered_success_intersection(self):
        """A failure on one PP rank must terminate an otherwise successful rid."""
        status = _pp_merge_transfer_status(
            previous=(["req-a", "req-b", "req-c"], ["req-x"]),
            current=(["req-c", "req-a", "req-b"], ["req-b", "req-y"]),
        )

        self.assertEqual(
            status,
            (["req-a", "req-c"], ["req-x", "req-b", "req-y"]),
        )

    def test_bootstrap_probe_respects_local_metadata_credit_prefix(self):
        """A slower PP rank must not advertise requests it cannot admit."""
        queue = PrefillBootstrapQueue.__new__(PrefillBootstrapQueue)
        queue.queue = [
            SimpleNamespace(
                rid="req-failed",
                metadata_buffer_index=-1,
                disagg_kv_sender=object(),
            ),
            SimpleNamespace(
                rid="req-ready",
                metadata_buffer_index=-1,
                disagg_kv_sender=object(),
            ),
            SimpleNamespace(
                rid="req-blocked",
                metadata_buffer_index=-1,
                disagg_kv_sender=object(),
            ),
        ]
        queue.scheduler = SimpleNamespace(
            attn_cp_cpu_group=object(),
            attn_tp_cpu_group=object(),
        )
        queue.req_to_metadata_buffer_idx_allocator = SimpleNamespace(
            available_size=lambda: 1
        )

        with patch(
            "sglang.srt.disaggregation.prefill."
            "poll_and_all_reduce_attn_cp_tp_group",
            return_value=[
                KVPoll.Failed,
                KVPoll.WaitingForInput,
                KVPoll.WaitingForInput,
            ],
        ):
            good_rids, failed_rids = queue.get_ready_bootstrapped_rids_for_pp()

        self.assertEqual(good_rids, ["req-ready"])
        self.assertEqual(failed_rids, ["req-failed"])
        self.assertEqual(
            [req.metadata_buffer_index for req in queue.queue],
            [-1, -1, -1],
        )

    def test_only_last_pp_registers_draft_kv_for_transfer(self):
        """Draft KV has one prefill owner even though every PP rank has a worker."""
        layer_ids_by_rank = []
        for pp_rank in (0, 1):
            queue, fake_args, fake_manager = self._make_prefill_queue(pp_rank)

            def get_kv_class(_backend, class_type):
                return fake_args if class_type.value == "kvargs" else fake_manager

            with (
                patch(
                    "sglang.srt.disaggregation.prefill.get_kv_class",
                    side_effect=get_kv_class,
                ),
                patch(
                    "sglang.srt.disaggregation.prefill.setup_state_kv_args",
                ),
            ):
                manager = queue._init_kv_manager()
            layer_ids_by_rank.append(manager.kv_args.kv_layer_ids)

        self.assertEqual(layer_ids_by_rank[0], [4, 5])
        self.assertEqual(
            layer_ids_by_rank[1],
            [4, 5, _DRAFT_KV_LAYER_ID_BASE],
        )

    def test_pp_prefill_entries_pair_with_pp1_decode_entries(self):
        """Each PP source maps target layers plus the unique draft entry by id."""
        decode_layer_ids = [
            0,
            1,
            2,
            3,
            4,
            5,
            _DRAFT_KV_LAYER_ID_BASE,
        ]

        rank_0_pairs = build_transfer_entry_pairs(
            src_layer_ids=[0, 1, 2, 3],
            dst_layer_ids=decode_layer_ids,
            n_src=4,
            n_dst=7,
        )
        rank_1_pairs = build_transfer_entry_pairs(
            src_layer_ids=[4, 5, _DRAFT_KV_LAYER_ID_BASE],
            dst_layer_ids=decode_layer_ids,
            n_src=3,
            n_dst=7,
        )

        self.assertEqual(rank_0_pairs, [(0, 0), (1, 1), (2, 2), (3, 3)])
        self.assertEqual(rank_1_pairs, [(0, 4), (1, 5), (2, 6)])


if __name__ == "__main__":
    unittest.main()
