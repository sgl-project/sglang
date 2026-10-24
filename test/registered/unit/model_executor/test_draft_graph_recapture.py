"""A weight-update recapture only redoes a decode graph the runner captured."""

import unittest

from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

_UNSET = object()


def _runner(*, is_draft_worker, decode_graph=_UNSET):
    """A runner whose decode slot holds what capture actually leaves there.

    `capture_cuda_graphs(capture_decode_cuda_graph=False)` — what a spec worker
    passes for its draft — puts the runner's own eager runner in the decode
    slot, so `None` is not the state a draft is really in.
    """
    runner = ModelRunner.__new__(ModelRunner)
    runner.is_draft_worker = is_draft_worker
    runner.eager_runner = object()
    runner.decode_cuda_graph_runner = (
        runner.eager_runner if decode_graph is _UNSET else decode_graph
    )
    runner.captures = []
    runner.init_decode_cuda_graph = lambda: runner.captures.append("decode")
    return runner


class TestDraftGraphRecapture(unittest.TestCase):
    def test_a_draft_without_its_own_decode_graph_skips_it(self):
        # The spec worker skipped the draft runner's decode capture, so the
        # slot holds the eager runner; or it asked for no graphs at all.
        for runner in (
            _runner(is_draft_worker=True),
            _runner(is_draft_worker=True, decode_graph=None),
        ):
            runner.recapture_decode_cuda_graph()
            self.assertEqual(runner.captures, [])

    def test_a_runner_that_owns_a_decode_graph_recaptures_it(self):
        for is_draft_worker in (False, True):
            with self.subTest(is_draft_worker=is_draft_worker):
                runner = _runner(is_draft_worker=is_draft_worker, decode_graph=object())
                runner.recapture_decode_cuda_graph()
                self.assertEqual(runner.captures, ["decode"])

    def test_a_target_recaptures_as_before(self):
        runner = _runner(is_draft_worker=False, decode_graph=None)
        runner.recapture_decode_cuda_graph()
        self.assertEqual(runner.captures, ["decode"])


if __name__ == "__main__":
    unittest.main()
