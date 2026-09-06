import asyncio
import unittest
from unittest.mock import patch

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
    MultiTokenizerRouter,
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


class TestPauseContinueCompletion(unittest.IsolatedAsyncioTestCase):
    def _make_worker(self):
        worker = TokenizerWorker.__new__(TokenizerWorker)
        worker.is_pause = False
        worker.is_pause_cond = asyncio.Condition()
        worker._pause_continue_futures = {}
        dispatched = []
        worker._dispatch_to_scheduler = dispatched.append
        return worker, dispatched

    @staticmethod
    def _broadcast_for(request, *, is_pause):
        # The fallback keeps this regression test on the pre-fix boolean-only
        # protocol long enough to demonstrate the orphaned waiter.
        kwargs = {"is_pause": is_pause}
        operation_id = getattr(request, "operation_id", None)
        if operation_id is not None:
            kwargs["operation_id"] = operation_id
        return PauseContinueBroadcastReq(**kwargs)

    async def test_concurrent_operations_complete_independently_when_reordered(self):
        worker, dispatched = self._make_worker()
        pause_task = asyncio.create_task(
            worker.pause_generation(PauseGenerationReqInput(mode="retract"))
        )
        continue_task = asyncio.create_task(
            worker.continue_generation(
                ContinueGenerationReqInput(torch_empty_cache=False)
            )
        )

        try:
            await asyncio.sleep(0)
            self.assertEqual(len(dispatched), 2)
            pause_request, continue_request = dispatched

            await worker._apply_pause_continue_broadcast(
                self._broadcast_for(continue_request, is_pause=False)
            )
            await asyncio.sleep(0)
            self.assertTrue(continue_task.done())
            self.assertFalse(pause_task.done())

            await worker._apply_pause_continue_broadcast(
                self._broadcast_for(pause_request, is_pause=True)
            )
            await asyncio.sleep(0)
            self.assertTrue(pause_task.done())
            await asyncio.gather(pause_task, continue_task)
        finally:
            for task in (pause_task, continue_task):
                if not task.done():
                    task.cancel()
            await asyncio.gather(pause_task, continue_task, return_exceptions=True)

    async def test_cancelled_operation_is_removed_before_a_late_broadcast(self):
        worker, dispatched = self._make_worker()
        pause_task = asyncio.create_task(
            worker.pause_generation(PauseGenerationReqInput(mode="retract"))
        )
        await asyncio.sleep(0)
        pause_request = dispatched[0]
        self.assertIn(pause_request.operation_id, worker._pause_continue_futures)

        pause_task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await pause_task
        self.assertEqual(worker._pause_continue_futures, {})

        await worker._apply_pause_continue_broadcast(
            self._broadcast_for(pause_request, is_pause=True)
        )
        self.assertTrue(worker.is_pause)
        self.assertEqual(worker._pause_continue_futures, {})

    async def test_sequential_pause_and_continue_preserve_state_transitions(self):
        worker, dispatched = self._make_worker()
        pause_task = asyncio.create_task(
            worker.pause_generation(PauseGenerationReqInput(mode="retract"))
        )
        await asyncio.sleep(0)
        await worker._apply_pause_continue_broadcast(
            self._broadcast_for(dispatched[0], is_pause=True)
        )
        await pause_task
        self.assertTrue(worker.is_pause)

        continue_task = asyncio.create_task(
            worker.continue_generation(
                ContinueGenerationReqInput(torch_empty_cache=False)
            )
        )
        await asyncio.sleep(0)
        await worker._apply_pause_continue_broadcast(
            self._broadcast_for(dispatched[1], is_pause=False)
        )
        await continue_task

        self.assertFalse(worker.is_pause)
        self.assertNotEqual(
            dispatched[0].operation_id,
            dispatched[1].operation_id,
        )
        self.assertEqual(worker._pause_continue_futures, {})

    async def test_abort_still_polls_after_its_matching_broadcast(self):
        class UnlockedModelUpdateLock:
            async def is_locked(self):
                return False

        worker, dispatched = self._make_worker()
        abort_calls = []
        worker.abort_request = lambda *, abort_all: abort_calls.append(abort_all)
        worker.model_update_lock = UnlockedModelUpdateLock()
        abort_task = asyncio.create_task(
            worker.pause_generation(PauseGenerationReqInput(mode="abort"))
        )
        await asyncio.sleep(0)

        await worker._apply_pause_continue_broadcast(
            self._broadcast_for(dispatched[0], is_pause=True)
        )
        await abort_task

        self.assertEqual(abort_calls, [True])
        self.assertTrue(worker.is_pause)
        self.assertEqual(worker._pause_continue_futures, {})


class TestPauseContinueRouting(unittest.IsolatedAsyncioTestCase):
    async def test_abort_broadcast_preserves_operation_id_for_every_worker(self):
        class RecordingSocketMapping:
            def __init__(self):
                self.outputs = []

            def send_output(self, ipc_name, output):
                self.outputs.append((ipc_name, output))

        router = MultiTokenizerRouter.__new__(MultiTokenizerRouter)
        router.receive_from_worker = object()
        router.send_to_scheduler = object()
        router.all_worker_ipcs = {"worker-0", "worker-1"}
        router.socket_mapping = RecordingSocketMapping()
        request = PauseGenerationReqInput(mode="abort", operation_id="pause-operation")
        received = False

        async def receive_once(_socket):
            nonlocal received
            if not received:
                received = True
                return request
            raise asyncio.CancelledError

        with patch(
            "sglang.srt.managers.multi_tokenizer_mixin.async_sock_recv",
            side_effect=receive_once,
        ):
            with self.assertRaises(asyncio.CancelledError):
                await router.router_worker_obj()

        self.assertEqual(
            {ipc_name for ipc_name, _ in router.socket_mapping.outputs},
            router.all_worker_ipcs,
        )
        self.assertTrue(
            all(
                output.is_pause and output.operation_id == request.operation_id
                for _, output in router.socket_mapping.outputs
            )
        )


if __name__ == "__main__":
    unittest.main()
