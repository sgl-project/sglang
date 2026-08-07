import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from sglang.srt.disaggregation.decode import DecodePreallocQueue
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def make_model_runner(
    *,
    enable_hisparse: bool,
    disaggregation_mode: str,
    device_capacity: int,
    logical_capacity: int,
):
    runner = object.__new__(ModelRunner)
    runner.enable_hisparse = enable_hisparse
    runner.server_args = SimpleNamespace(disaggregation_mode=disaggregation_mode)
    runner.token_to_kv_pool_allocator = SimpleNamespace(size_full=logical_capacity)
    runner.is_hybrid_swa = False
    runner.max_total_num_tokens = device_capacity
    return runner


class TestHiSparseRequestCapacity(CustomTestCase):
    def test_decode_uses_logical_capacity(self):
        runner = make_model_runner(
            enable_hisparse=True,
            disaggregation_mode="decode",
            device_capacity=1024,
            logical_capacity=4096,
        )
        self.assertEqual(runner.effective_max_total_num_tokens, 4096)

    def test_aggregate_uses_device_capacity(self):
        runner = make_model_runner(
            enable_hisparse=True,
            disaggregation_mode="null",
            device_capacity=1024,
            logical_capacity=4096,
        )
        self.assertEqual(runner.effective_max_total_num_tokens, 1024)

    def test_non_hisparse_uses_device_capacity(self):
        runner = make_model_runner(
            enable_hisparse=False,
            disaggregation_mode="decode",
            device_capacity=1024,
            logical_capacity=4096,
        )
        self.assertEqual(runner.effective_max_total_num_tokens, 1024)


def make_decode_queue(*, runner: ModelRunner, enable_hisparse: bool):
    queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
    queue.max_total_num_tokens = runner.max_total_num_tokens
    queue.token_to_kv_pool_allocator = SimpleNamespace(size_swa=10**9)
    queue._uses_swa_tail_prealloc = MagicMock(return_value=False)
    queue.scheduler = SimpleNamespace(
        enable_hisparse=enable_hisparse,
        max_req_token_capacity=runner.effective_max_total_num_tokens,
        tp_worker=SimpleNamespace(model_runner=runner),
        output_streamer=MagicMock(),
    )
    return queue


def make_request(prompt_len: int):
    return SimpleNamespace(
        rid="test",
        origin_input_ids=[0] * prompt_len,
        output_ids=[],
        return_logprob=False,
        pd_rebootstrap_in_progress=False,
        finished_reason=None,
    )


class TestDecodePreallocCapacity(CustomTestCase):
    def setUp(self):
        self.runner = make_model_runner(
            enable_hisparse=True,
            disaggregation_mode="decode",
            device_capacity=1024,
            logical_capacity=4096,
        )
        self.queue = make_decode_queue(
            runner=self.runner,
            enable_hisparse=True,
        )

    def test_admits_request_larger_than_device_pool(self):
        self.assertFalse(
            self.queue._check_if_req_exceed_kv_capacity(make_request(2048))
        )
        self.queue.scheduler.output_streamer.stream_output.assert_not_called()

    def test_rejects_request_larger_than_logical_pool(self):
        req = make_request(4097)
        self.assertTrue(self.queue._check_if_req_exceed_kv_capacity(req))
        self.assertIsNotNone(req.finished_reason)
        self.queue.scheduler.output_streamer.stream_output.assert_called_once()


if __name__ == "__main__":
    unittest.main()
