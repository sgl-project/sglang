import unittest
from types import SimpleNamespace

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.io_struct import GenerateReqInput  # noqa: E402
from sglang.srt.managers.tokenizer_manager import TokenizerManager  # noqa: E402

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestTokenizerManagerSessionParams(unittest.IsolatedAsyncioTestCase):
    async def test_parallel_sampling_preserves_per_prompt_session_params(self):
        batch = GenerateReqInput(
            text=["first", "second"],
            rid=["prompt-1", "prompt-2"],
            sampling_params={"n": 3},
            session_params=[{"id": "s1"}, {"id": "s2"}],
        )
        batch.normalize_batch_and_arguments()

        manager = object.__new__(TokenizerManager)
        manager.rid_to_state = {
            rid: SimpleNamespace(
                time_stats=SimpleNamespace(set_finished_time=lambda: None)
            )
            for rid in batch.rid
        }

        async def tokenize(request):
            return SimpleNamespace(
                rid=request.rid,
                input_ids=[],
                mm_inputs=None,
                sampling_params=SimpleNamespace(max_new_tokens=16),
                session_params=request.session_params,
            )

        sent = []

        def init_state(request):
            manager.rid_to_state[request.rid] = SimpleNamespace(
                time_stats=SimpleNamespace()
            )

        async def wait_one_response(_request, _http_request):
            yield {}

        async def collect_batch_responses(_generators):
            return []

        manager._tokenize_one_request = tokenize
        manager._init_req_state = init_state
        manager._send_one_request = sent.append
        manager._wait_one_response = wait_one_response
        manager._collect_batch_responses = collect_batch_responses

        outputs = [output async for output in manager._handle_batch_request(batch)]

        self.assertEqual(outputs, [[]])
        self.assertEqual(
            [request.session_params for request in sent[:2]],
            [{"id": "s1"}, {"id": "s2"}],
        )
        self.assertEqual(
            [request.session_params for request in sent[2:]],
            [
                {"id": "s1"},
                {"id": "s1"},
                {"id": "s1"},
                {"id": "s2"},
                {"id": "s2"},
                {"id": "s2"},
            ],
        )


if __name__ == "__main__":
    unittest.main()
