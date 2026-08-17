import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from sglang.srt.environ import envs
from sglang.srt.managers.io_struct import ListExternalCorporaReqOutput
from sglang.srt.managers.scheduler import Scheduler
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase


register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestSchedulerControlOutput(CustomTestCase):
    def _scheduler(self):
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.server_args = SimpleNamespace()
        scheduler.tp_worker = object()
        scheduler.ps = SimpleNamespace(gpu_id=0)
        scheduler.nccl_port = 0

        class _DraftWorker:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
                self.ngram_corpus = SimpleNamespace(cancel_external_corpus_load=Mock())

        scheduler.spec_algorithm = SimpleNamespace(
            is_none=lambda: False,
            is_ngram=lambda: True,
            create_worker=lambda server_args: _DraftWorker,
        )
        scheduler.ipc_channels = SimpleNamespace(
            send_to_tokenizer=SimpleNamespace(send_output=Mock())
        )
        scheduler.rust_server = None
        scheduler.maybe_init_draft_worker()
        return scheduler

    def test_deferred_corpus_reply_uses_scheduler_router(self):
        scheduler = self._scheduler()
        callback = scheduler.external_corpus_manager._send_response
        self.assertIs(callback.__self__, scheduler)
        self.assertIs(callback.__func__, Scheduler._send_control_output)
        synchronizer = scheduler.external_corpus_manager._synchronize_load_result
        self.assertIs(synchronizer.__self__, scheduler)
        self.assertIs(
            synchronizer.__func__,
            Scheduler._synchronize_external_corpus_load_result,
        )
        self.assertIs(
            scheduler.external_corpus_manager._cancel_load,
            scheduler.draft_worker.ngram_corpus.cancel_external_corpus_load,
        )

    def test_router_uses_rust_egress_when_present(self):
        scheduler = self._scheduler()
        rust_server = Mock()
        scheduler.rust_server = rust_server
        request = object()
        output = object()

        scheduler._send_control_output(output, request)

        rust_server.push_control_output.assert_called_once_with(request, output)
        scheduler.ipc_channels.send_to_tokenizer.send_output.assert_not_called()

    def test_router_uses_tokenizer_socket_without_rust_server(self):
        scheduler = self._scheduler()
        request = object()
        output = object()

        with envs.SGLANG_RUST_SERVER.override(False):
            scheduler._send_control_output(output, request)

        scheduler.ipc_channels.send_to_tokenizer.send_output.assert_called_once_with(
            output, request
        )

    def test_non_entry_rust_rank_drops_control_reply(self):
        scheduler = self._scheduler()

        with envs.SGLANG_RUST_SERVER.override(True):
            scheduler._send_control_output(object(), object())

        scheduler.ipc_channels.send_to_tokenizer.send_output.assert_not_called()

    def test_list_control_output_detects_cross_rank_divergence(self):
        scheduler = self._scheduler()
        group = Mock()
        group.all_gather_object.return_value = [
            (True, "", (("a", 3),)),
            (True, "", (("a", 4),)),
        ]
        scheduler._external_corpus_control_group = Mock(return_value=group)

        output = scheduler._merge_external_corpus_control_output(
            ListExternalCorporaReqOutput(success=True, corpus_token_counts={"a": 3})
        )

        self.assertFalse(output.success)
        self.assertIn("inconsistent", output.message)


if __name__ == "__main__":
    unittest.main()
