"""Fail closed when the API cannot expose GEN replay accounting."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.srt.arg_groups.pd_disaggregation_hook import handle_pd_disaggregation
from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestBootstrapAccountingGate(unittest.TestCase):
    def test_native_egress_without_replay_counter_is_rejected(self):
        cfg = SimpleNamespace(
            disaggregation_decode_draft_bootstrap=True,
        )
        with (
            envs.SGLANG_RUST_SERVER.override(True),
            patch(
                "sglang.srt.arg_groups.pd_disaggregation_hook.resolving_view",
                return_value=cfg,
            ),
            self.assertRaisesRegex(ValueError, "native egress.*replay accounting"),
        ):
            handle_pd_disaggregation(object())


class TestRejectedPrebuiltCleanup(unittest.TestCase):
    def test_overbudget_request_released_once_while_ready_request_proceeds(self):
        from sglang.srt.disaggregation.decode import SchedulerDisaggregationDecodeMixin
        from sglang.srt.disaggregation.utils import DisaggregationMode
        from sglang.srt.managers.scheduler_components.batch_result_processor import (
            SchedulerBatchResultProcessor,
        )

        def make_req(rid, length):
            req = SimpleNamespace(
                rid=rid,
                origin_input_ids=list(range(length)),
                output_ids=[20],
                multimodal_inputs=None,
                input_embeds=None,
                grammar=None,
                kv=SimpleNamespace(kv_committed_len=None),
                init_next_round_input=Mock(),
                time_stats=Mock(),
                return_logprob=False,
                finished_reason=None,
                update_finish_state=Mock(),
                pd_draft_bootstrap_pending=True,
            )
            req.finished = lambda: req.finished_reason is not None
            return req

        rejected, ready = make_req("overbudget", 5), make_req("ready", 3)
        cfg = SimpleNamespace(
            disaggregation_decode_draft_bootstrap=True,
            disaggregation_decode_draft_bootstrap_max_tokens=4,
            disaggregation_decode_enable_radix_cache=False,
        )
        processor = SimpleNamespace(
            disaggregation_mode=DisaggregationMode.DECODE,
            tree_cache=object(),
            output_streamer=Mock(),
        )
        scheduler = SimpleNamespace(
            waiting_queue=[rejected, ready],
            grammar_manager=SimpleNamespace(has_waiting_grammars=lambda: False),
            enable_priority_scheduling=False,
            req_to_token_pool=SimpleNamespace(size=4),
            max_running_requests=4,
            tree_cache=processor.tree_cache,
            token_to_kv_pool_allocator=object(),
            model_config=object(),
            enable_overlap=False,
            spec_algorithm=object(),
            future_map=object(),
            ngram_embedding_manager=Mock(),
            chunked_req=None,
            batch_result_processor=SimpleNamespace(
                process_batch_result_prebuilt=lambda batch: (
                    SchedulerBatchResultProcessor.process_batch_result_prebuilt(
                        processor, batch
                    )
                )
            ),
        )

        def make_batch(reqs, *args):
            return SimpleNamespace(
                reqs=reqs,
                return_logprob=False,
                prepare_for_prebuilt=Mock(),
                process_prebuilt=Mock(),
            )

        module = "sglang.srt.disaggregation.decode"
        result_module = (
            "sglang.srt.managers.scheduler_components.batch_result_processor"
        )
        with (
            patch(f"{module}.get_disagg", return_value=cfg),
            patch(f"{result_module}.get_disagg", return_value=cfg),
            patch(
                f"{result_module}.get_memory",
                return_value=SimpleNamespace(enable_hisparse=False),
            ),
            patch(f"{result_module}.release_kv_cache") as release,
            patch(f"{module}.ScheduleBatch.init_new", side_effect=make_batch),
            patch(
                "sglang.srt.disaggregation.draft_bootstrap.bootstrap_decode_draft"
            ) as bootstrap,
        ):
            batch = SchedulerDisaggregationDecodeMixin.get_new_prebuilt_batch(
                scheduler, SimpleNamespace(batch_size=lambda: 0)
            )
        release.assert_called_once_with(rejected, processor.tree_cache)
        processor.output_streamer.stream_output.assert_called_once_with(
            [rejected], False
        )
        self.assertEqual(rejected.finished_reason.to_json()["status_code"], 400)
        bootstrap.assert_called_once_with(scheduler, ready)
        self.assertEqual(batch.reqs, [ready])
        batch.prepare_for_prebuilt.assert_called_once()
        batch.process_prebuilt.assert_called_once_with(scheduler.future_map)
        self.assertEqual(scheduler.waiting_queue, [])


if __name__ == "__main__":
    unittest.main()
