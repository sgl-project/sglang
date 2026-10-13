"""The dLLM worker around the target worker."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from sglang.srt.dllm.worker import DllmWorker
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _worker(*, fdfo):
    worker = object.__new__(DllmWorker)
    worker.target_worker = SimpleNamespace(
        get_worker_info=lambda: tuple(range(100, 112)),
        init_forward_batch=lambda batch: (batch, "forward_batch"),
        model_runner="runner",
    )
    worker.dllm_algorithm = SimpleNamespace(
        block_size=32,
        fdfo=fdfo,
        run=Mock(return_value=("logits", [[1]], [32], [None], False)),
    )
    return worker


class TestDllmWorker(unittest.TestCase):
    def test_worker_info_leaves_one_block_past_the_input(self):
        info = _worker(fdfo=False).get_worker_info()
        expected = list(range(100, 112))
        expected[5] -= 32  # max_req_input_len
        self.assertEqual(list(info), expected)

    def test_fdfo_forward_carries_each_request_algorithm_state(self):
        reqs = [SimpleNamespace(dllm=SimpleNamespace(algo_state=s)) for s in "ab"]
        for fdfo, algo_states in ((True, ["a", "b"]), (False, None)):
            with self.subTest(fdfo=fdfo):
                worker = _worker(fdfo=fdfo)
                result = worker.forward_model(SimpleNamespace(reqs=reqs))
                worker.dllm_algorithm.run.assert_called_once_with(
                    "runner", "forward_batch", algo_states
                )
                self.assertEqual(result.accept_length_per_req_cpu, [32])
                self.assertEqual(result.dllm_algo_state, [None])


if __name__ == "__main__":
    unittest.main()
