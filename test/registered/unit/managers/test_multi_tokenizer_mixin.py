import unittest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.io_struct import BatchStrOutput
from sglang.srt.managers.multi_tokenizer_mixin import (
    TokenizerWorker,
    _handle_output_by_index,
    get_tokenizer_worker_class,
)

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class CustomTokenizerWorker(TokenizerWorker):
    pass


class NotAWorker:
    pass


class DefaultServerArgs:
    def get_tokenizer_worker_class(self):
        return TokenizerWorker


class CustomServerArgs:
    def get_tokenizer_worker_class(self):
        return CustomTokenizerWorker


class InvalidServerArgs:
    def get_tokenizer_worker_class(self):
        return NotAWorker


def _make_batch_str_output() -> BatchStrOutput:
    return BatchStrOutput(
        rids=["rid-0", "rid-1"],
        spec_verify_ct=[0, 0],
        spec_num_correct_drafts=[0, 0],
        spec_correct_drafts_histogram=[[], []],
        finished_reasons=[None, {"type": "length"}],
        output_strs=["first", "second"],
        output_ids=[[1], [2]],
        prompt_tokens=[10, 20],
        completion_tokens=[1, 2],
        reasoning_tokens=[0, 0],
        cached_tokens=[3, 4],
        cached_tokens_details=[
            {"device": 3, "host": 0},
            {"device": 1, "host": 3},
        ],
        input_token_logprobs_val=[[], []],
        input_token_logprobs_idx=[[], []],
        output_token_logprobs_val=[[], []],
        output_token_logprobs_idx=[[], []],
        input_top_logprobs_val=[[], []],
        input_top_logprobs_idx=[[], []],
        output_top_logprobs_val=[[], []],
        output_top_logprobs_idx=[[], []],
        input_token_ids_logprobs_val=[[], []],
        input_token_ids_logprobs_idx=[[], []],
        output_token_ids_logprobs_val=[[], []],
        output_token_ids_logprobs_idx=[[], []],
        output_token_entropy_val=[0.0, 0.0],
        output_token_sampling_mask=[[], []],
        output_token_sampling_logprobs=[[], []],
        output_hidden_states=[None, None],
        routed_experts=[None, None],
        indexer_topk=[None, None],
        placeholder_tokens_idx=[None, None],
        placeholder_tokens_val=[None, None],
        retraction_counts=[0, 0],
    )


class TestAwaitReporterStartup(unittest.TestCase):
    """M1: the router reporter startup future must not leak on timeout/failure."""

    def setUp(self):
        import asyncio
        import threading

        from sglang.srt.managers.multi_tokenizer_mixin import MultiTokenizerRouter

        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()
        # Skip the heavy MultiTokenizerRouter.__init__ (zmq sockets, router
        # thread, disagg service): _await_reporter_startup only reads self._loop.
        self._router = MultiTokenizerRouter.__new__(MultiTokenizerRouter)
        self._router._loop = self._loop

    def tearDown(self):
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5.0)
        self._loop.close()

    def test_timeout_cancels_pending_startup(self):
        import asyncio
        import threading

        started = threading.Event()
        cancelled = threading.Event()

        async def slow_start():
            started.set()
            try:
                await asyncio.sleep(30)
                return object()
            except asyncio.CancelledError:
                cancelled.set()
                raise

        future = asyncio.run_coroutine_threadsafe(slow_start(), self._loop)
        self.assertTrue(started.wait(timeout=1.0))

        with self.assertRaises(TimeoutError):
            self._router._await_reporter_startup(future, timeout=0.1)

        self.assertTrue(cancelled.wait(timeout=1.0))
        self.assertTrue(future.cancelled())

    def test_handle_closed_when_produced_after_timeout(self):
        """Cancel-vs-complete race: when the startup future already produced a
        handle at timeout (``cancel()`` returns False), that handle must be
        closed, not leaked."""
        import concurrent.futures
        import threading

        closed = threading.Event()

        class FakeHandle:
            async def close(self):
                closed.set()

        # A RUNNING future: result(timeout) times out and cancel() returns False,
        # exactly the branch where a late handle would leak.
        future: concurrent.futures.Future = concurrent.futures.Future()
        future.set_running_or_notify_cancel()

        with self.assertRaises(TimeoutError):
            self._router._await_reporter_startup(future, timeout=0.05)

        # The startup coroutine settles with a real handle just after timeout.
        future.set_result(FakeHandle())
        # The helper's done-callback must close that late handle.
        self.assertTrue(closed.wait(timeout=2.0))


class TestMultiTokenizerMixin(unittest.TestCase):
    def test_batch_str_output_preserves_cached_tokens_details(self):
        output = _make_batch_str_output()

        single_output = _handle_output_by_index(output, 1)

        self.assertEqual(single_output.rids, ["rid-1"])
        self.assertEqual(single_output.cached_tokens, [4])
        self.assertEqual(
            single_output.cached_tokens_details,
            [{"device": 1, "host": 3}],
        )

    def test_get_tokenizer_worker_class_uses_default(self):
        self.assertIs(get_tokenizer_worker_class(DefaultServerArgs()), TokenizerWorker)

    def test_get_tokenizer_worker_class_resolves_custom_class(self):
        self.assertIs(
            get_tokenizer_worker_class(CustomServerArgs()),
            CustomTokenizerWorker,
        )

    def test_get_tokenizer_worker_class_rejects_non_worker(self):
        with self.assertRaisesRegex(TypeError, "TokenizerWorker"):
            get_tokenizer_worker_class(InvalidServerArgs())


if __name__ == "__main__":
    unittest.main()
