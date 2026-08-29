import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.disaggregation.base import KVPoll  # noqa: E402
from sglang.srt.disaggregation.prefill import (  # noqa: E402
    PrefillBootstrapQueue,
    SchedulerDisaggregationPrefillMixin,
)
from sglang.srt.disaggregation.utils import (  # noqa: E402
    build_transfer_entry_pairs,
)
from sglang.srt.managers.schedule_batch import FINISH_ABORT  # noqa: E402
from sglang.srt.managers.scheduler_pp_mixin import (  # noqa: E402
    _pp_merge_transfer_status,
)
from sglang.srt.runtime_context import get_context  # noqa: E402

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
            layer_shard_enabled = False

            @staticmethod
            def get_contiguous_buf_infos():
                return [10, 11], [100, 100], [10, 10]

            @staticmethod
            def get_kv_layer_ids():
                return [4, 5]

        class FakeDraftPool:
            start_layer = 0
            end_layer = 1
            layer_num = 1

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
            tp_worker=SimpleNamespace(
                model_runner=SimpleNamespace(kv_cache_dtype_str="auto")
            ),
            model_config=SimpleNamespace(
                num_hidden_layers=8,
                get_total_num_kv_heads=lambda: 1,
            ),
            req_to_token_pool=None,
        )
        queue.token_to_kv_pool = FakePool()
        queue.draft_token_to_kv_pool = FakeDraftPool()
        queue.metadata_buffers = SimpleNamespace(get_buf_infos=lambda: ([], [], []))
        queue.is_mla_backend = False
        return queue, FakeKVArgs, FakeManager

    def test_transfer_failure_overrides_ordered_success_intersection(self):
        status = _pp_merge_transfer_status(
            previous=(["req-a", "req-b", "req-c"], ["req-x"]),
            current=(["req-c", "req-a", "req-b"], ["req-b", "req-y"]),
        )

        self.assertEqual(
            status,
            (["req-a", "req-c"], ["req-x", "req-b", "req-y"]),
        )

    def test_bootstrap_probe_respects_local_metadata_credit_prefix(self):
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
            "sglang.srt.disaggregation.prefill." "poll_and_all_reduce_attn_cp_tp_group",
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

    def test_bootstrap_probe_reports_failures_after_metadata_backpressure(self):
        queue = PrefillBootstrapQueue.__new__(PrefillBootstrapQueue)
        queue.queue = [
            SimpleNamespace(
                rid="req-blocked",
                metadata_buffer_index=-1,
                disagg_kv_sender=object(),
            ),
            SimpleNamespace(
                rid="req-failed",
                metadata_buffer_index=-1,
                disagg_kv_sender=object(),
            ),
            SimpleNamespace(
                rid="req-ready-after-block",
                metadata_buffer_index=0,
                disagg_kv_sender=object(),
            ),
        ]
        queue.scheduler = SimpleNamespace(
            attn_cp_cpu_group=object(),
            attn_tp_cpu_group=object(),
        )
        queue.req_to_metadata_buffer_idx_allocator = SimpleNamespace(
            available_size=lambda: 0
        )

        with patch(
            "sglang.srt.disaggregation.prefill." "poll_and_all_reduce_attn_cp_tp_group",
            return_value=[
                KVPoll.WaitingForInput,
                KVPoll.Failed,
                KVPoll.WaitingForInput,
            ],
        ):
            good_rids, failed_rids = queue.get_ready_bootstrapped_rids_for_pp()

        self.assertEqual(good_rids, [])
        self.assertEqual(failed_rids, ["req-failed"])

    def test_remote_failure_waits_for_local_transfer_terminal_state(self):
        sender = SimpleNamespace()
        req = SimpleNamespace(
            rid="req-race",
            disagg_kv_sender=sender,
            finished_reason=None,
            pending_bootstrap=False,
            return_logprob=False,
            time_stats=SimpleNamespace(set_completion_time=Mock()),
        )
        handle_failure = Mock()
        scheduler = SimpleNamespace(
            disagg_prefill_inflight_queue=[req],
            attn_cp_cpu_group=object(),
            attn_tp_cpu_group=object(),
            ps=SimpleNamespace(pp_rank=1),
            handle_inflight_transfer_failure=handle_failure,
            output_streamer=SimpleNamespace(stream_output=Mock()),
            req_to_metadata_buffer_idx_allocator=object(),
        )

        def mark_abort(target_req, message, status_code):
            del message, status_code
            target_req.finished_reason = FINISH_ABORT("remote PP failure")

        with (
            patch(
                "sglang.srt.disaggregation.prefill."
                "poll_and_all_reduce_attn_cp_tp_group",
                return_value=[KVPoll.Transferring],
            ),
            patch(
                "sglang.srt.disaggregation.prefill.prepare_abort",
                side_effect=mark_abort,
            ),
        ):
            done_reqs = SchedulerDisaggregationPrefillMixin.process_disagg_prefill_inflight_queue(
                scheduler,
                transfer_status=([], ["req-race"]),
            )

        self.assertEqual(done_reqs, [])
        self.assertEqual(scheduler.disagg_prefill_inflight_queue, [req])
        handle_failure.assert_not_called()

        with (
            patch(
                "sglang.srt.disaggregation.prefill."
                "poll_and_all_reduce_attn_cp_tp_group",
                return_value=[KVPoll.Success],
            ),
            patch("sglang.srt.disaggregation.prefill.maybe_release_metadata_buffer"),
        ):
            done_reqs = SchedulerDisaggregationPrefillMixin.process_disagg_prefill_inflight_queue(
                scheduler,
                transfer_status=([], []),
            )

        self.assertEqual(done_reqs, [req])
        self.assertEqual(scheduler.disagg_prefill_inflight_queue, [])
        handle_failure.assert_called_once_with(req)

    def test_only_last_pp_registers_draft_kv_for_transfer(self):
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
                get_context().override_server_args(disaggregation_ib_device=None),
            ):
                manager = queue._init_kv_manager()
            layer_ids_by_rank.append(manager.kv_args.kv_layer_ids)

        self.assertEqual(layer_ids_by_rank[0], [4, 5])
        self.assertEqual(
            layer_ids_by_rank[1],
            [4, 5, 8],
        )

    def test_pp_prefill_entries_pair_with_pp1_decode_entries(self):
        decode_layer_ids = [
            0,
            1,
            2,
            3,
            4,
            5,
            8,
        ]

        rank_0_pairs = build_transfer_entry_pairs(
            src_layer_ids=[0, 1, 2, 3],
            dst_layer_ids=decode_layer_ids,
            n_src=4,
            n_dst=7,
        )
        rank_1_pairs = build_transfer_entry_pairs(
            src_layer_ids=[4, 5, 8],
            dst_layer_ids=decode_layer_ids,
            n_src=3,
            n_dst=7,
        )

        self.assertEqual(rank_0_pairs, [(0, 0), (1, 1), (2, 2), (3, 3)])
        self.assertEqual(rank_1_pairs, [(0, 4), (1, 5), (2, 6)])


if __name__ == "__main__":
    unittest.main()
