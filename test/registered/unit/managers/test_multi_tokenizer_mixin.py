import asyncio
import unittest
from unittest.mock import Mock

from sglang.srt.utils.weight_versions import WeightVersionSpan
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.io_struct import (
    BatchStrOutput,
    ContinueGenerationReqInput,
    PauseContinueBroadcastReq,
    PauseGenerationReqInput,
)
from sglang.srt.managers.multi_tokenizer_mixin import (
    TokenizerWorker,
    _handle_output_by_index,
    get_tokenizer_worker_class,
)

register_cpu_ci(est_time=12, suite="base-a-test-cpu")


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
        weight_versions=[
            [
                WeightVersionSpan(version="v1", start=0, end=3),
                WeightVersionSpan(version="v2", start=3, end=5),
            ],
            [WeightVersionSpan(version="v2", start=0, end=2)],
        ],
    )


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

    def test_batch_str_output_keeps_weight_versions_nested_per_request(self):
        """Per-request segment lists stay one level nested after the split."""
        output = _make_batch_str_output()

        self.assertEqual(
            _handle_output_by_index(output, 0).weight_versions,
            [
                [
                    WeightVersionSpan(version="v1", start=0, end=3),
                    WeightVersionSpan(version="v2", start=3, end=5),
                ]
            ],
        )
        self.assertEqual(
            _handle_output_by_index(output, 1).weight_versions,
            [[WeightVersionSpan(version="v2", start=0, end=2)]],
        )

    def test_batch_str_output_without_weight_versions_stays_none(self):
        """An output from an older server without the field splits into None."""
        output = _make_batch_str_output()
        output.weight_versions = None

        self.assertIsNone(_handle_output_by_index(output, 0).weight_versions)

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


class TestColdTokenizerWorkerControl(unittest.IsolatedAsyncioTestCase):
    async def _check_cold_control(self, *, pause):
        """Deliver a router acknowledgment only after receive-loop initialization."""
        worker = TokenizerWorker.__new__(TokenizerWorker)
        worker.event_loop = None
        worker.is_pause = not pause
        worker.is_pause_cond = asyncio.Condition()
        worker._pause_continue_future = None
        inbox = asyncio.Queue()
        receiver = None

        async def receive():
            """Apply queued broadcasts through the upstream worker implementation."""
            while True:
                await worker._apply_pause_continue_broadcast(await inbox.get())

        def initialize():
            """Start the simulated receive loop once, like the native initializer."""
            nonlocal receiver
            if worker.event_loop is None:
                worker.event_loop = asyncio.get_running_loop()
                receiver = asyncio.create_task(receive(), name="test-control-receiver")

        worker.auto_create_handle_loop = Mock(side_effect=initialize)
        worker._dispatch_to_scheduler = Mock(
            side_effect=lambda obj: inbox.put_nowait(
                PauseContinueBroadcastReq(is_pause=pause)
            )
        )
        request = (
            PauseGenerationReqInput(mode="retract")
            if pause
            else ContinueGenerationReqInput()
        )
        operation = worker.pause_generation if pause else worker.continue_generation
        try:
            await asyncio.wait_for(operation(request), timeout=1)
            worker.auto_create_handle_loop.assert_called_once_with()
            worker._dispatch_to_scheduler.assert_called_once_with(request)
            self.assertEqual(worker.is_pause, pause)
            self.assertTrue(inbox.empty())
            self.assertIsNone(worker._pause_continue_future)
        finally:
            if receiver is not None:
                receiver.cancel()
                await asyncio.gather(receiver, return_exceptions=True)

    async def test_pause_starts_receive_loop_on_cold_worker(self):
        """A first pause request must consume its broadcast without prior generation."""
        await self._check_cold_control(pause=True)

    async def test_continue_starts_receive_loop_on_cold_worker(self):
        """A first continue request must consume its broadcast without prior generation."""
        await self._check_cold_control(pause=False)


if __name__ == "__main__":
    unittest.main()
