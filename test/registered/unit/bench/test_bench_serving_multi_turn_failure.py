import unittest
from argparse import Namespace
from unittest.mock import Mock, patch

from sglang.benchmark import serving
from sglang.benchmark.serving import RequestFuncInput, RequestFuncOutput
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestMultiTurnFailure(unittest.IsolatedAsyncioTestCase):
    async def test_failed_round_ends_session_without_sending_later_turns(self):
        sent = []

        async def request(request_input, pbar=None):
            sent.append(request_input.prompt)
            if pbar:
                pbar.update(1)
            if len(sent) == 2:
                return RequestFuncOutput(error="Request rejected")
            return RequestFuncOutput(success=True, generated_text="reply")

        run = serving.wrap_multi_turn_request_func(request, "sglang-oai-chat")
        progress = Mock()
        request_input = RequestFuncInput(
            prompt=["first", "second", "third"],
            api_url="",
            prompt_len=1,
            output_len=2,
            model="test-model",
            lora_name=None,
            image_data=None,
            extra_request_body={},
        )
        with patch.object(serving, "args", Namespace(disable_stream=True), create=True):
            outputs = await run(request_input, pbar=progress)

        first = {"role": "user", "content": "first"}
        reply = {"role": "assistant", "content": "reply"}
        second = {"role": "user", "content": "second"}
        self.assertEqual(sent, [[first], [first, reply, second]])
        self.assertEqual([output.success for output in outputs], [True, False])
        self.assertEqual(outputs[-1].error, "Request rejected")
        progress.update.assert_called_once_with(1)


if __name__ == "__main__":
    unittest.main()
