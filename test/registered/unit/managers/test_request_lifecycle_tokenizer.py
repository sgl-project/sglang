import unittest
import uuid
from types import SimpleNamespace

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.io_struct import AbortReq, GenerateReqInput
from sglang.srt.managers.request_lifecycle import RequestLifecycle
from sglang.srt.managers.tokenizer_manager import TokenizerManager

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestLifecycleTokenizer(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.manager = TokenizerManager.__new__(TokenizerManager)
        self.manager.enable_trace = False
        self.manager.disaggregation_mode = DisaggregationMode.NULL
        self.manager.rid_to_state = {}
        self.manager.request_lifecycle = RequestLifecycle()
        self.attempt = uuid.uuid4().hex
        self.manager.request_lifecycle.claim(self.attempt, "null")

    def test_exact_child_cancellation_keeps_native_prefix_compatibility(self):
        a = SimpleNamespace(rid="client", lifecycle_id="first")
        b = SimpleNamespace(rid="client-1", lifecycle_id="second")
        reused = SimpleNamespace(rid="client", lifecycle_id="third")
        exact = AbortReq(lifecycle_id="first")
        self.assertEqual(
            [exact.matches(req) for req in (a, b, reused)], [True, False, False]
        )
        legacy = AbortReq(rid="client")
        self.assertTrue(all(legacy.matches(req) for req in (a, b, reused)))

    async def test_batch_sampling_tracks_only_executed_children(self):
        manager = self.manager
        obj = GenerateReqInput(
            input_ids=[[1, 2], [3, 4]],
            sampling_params={"n": 3, "max_new_tokens": 1},
            rid="native",
        )
        obj._lifecycle_attempt_id = self.attempt
        obj.normalize_batch_and_arguments()
        manager._init_req_state(obj)
        self.assertEqual(len(manager.rid_to_state), 2)
        self.assertEqual(
            manager.request_lifecycle.snapshot(self.attempt)["children"], []
        )

        async def tokenize(item):
            return SimpleNamespace(
                rid=item.rid,
                input_ids=item.input_ids,
                sampling_params=SimpleNamespace(max_new_tokens=1),
                mm_inputs=None,
                stream=False,
            )

        async def send(item):
            manager._prepare_lifecycle_dispatch(item)
            manager.request_lifecycle.scheduler_event(item.lifecycle_id, 0, "prefill")
            manager.request_lifecycle.scheduler_event(item.lifecycle_id, 0, "terminal")

        async def response(item, request):
            manager.rid_to_state.pop(item.rid)
            yield {"meta_info": {"id": item.rid}}

        manager._tokenize_one_request = tokenize
        manager._send_one_request = send
        manager._wait_one_response = response
        generator = manager._handle_batch_request(obj)
        result = await generator.__anext__()
        await generator.aclose()
        self.assertEqual(len(result), 6)
        self.assertEqual(manager.rid_to_state, {})
        snapshot = manager.request_lifecycle.snapshot(self.attempt)
        self.assertTrue(snapshot["terminal"])
        self.assertEqual(len(snapshot["children"]), 8)
        self.assertEqual(
            sum(child["kind"] == "warmup" for child in snapshot["children"]), 2
        )

    def test_duplicate_batch_rejection_is_atomic(self):
        obj = GenerateReqInput(input_ids=[[1], [2]], rid=["a", "b"])
        obj.normalize_batch_and_arguments()
        obj._lifecycle_attempt_id = self.attempt
        old_state = object()
        self.manager.rid_to_state["b"] = old_state
        with self.assertRaises(ValueError):
            self.manager._init_req_state(obj)
        self.assertEqual(self.manager.rid_to_state, {"b": old_state})
        self.assertEqual(
            self.manager.request_lifecycle.snapshot(self.attempt)["children"], []
        )


if __name__ == "__main__":
    unittest.main()
