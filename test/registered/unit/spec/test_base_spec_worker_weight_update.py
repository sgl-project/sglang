"""CPU tests for speculative draft weight-update graph recapture ordering."""

import unittest
from types import SimpleNamespace

from sglang.srt.speculative.base_spec_worker import BaseSpecWorker
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

_SUCCESS = (True, "Succeeded to update model weights.")


class _RecordingWeightUpdater:
    def __init__(self, name, events, result=_SUCCESS):
        self.name = name
        self.events = events
        self.result = result

    def update_weights_from_disk(
        self, model_path, load_format, *, recapture_cuda_graph
    ):
        self.events.append(
            (
                "load",
                self.name,
                model_path,
                load_format,
                recapture_cuda_graph,
            )
        )
        if recapture_cuda_graph:
            return False, f"{self.name} received inner CUDA graph recapture"
        return self.result


class _RecordingSpecWorker(BaseSpecWorker):
    def __init__(self, runner_results):
        self.events = []
        self._draft_worker = SimpleNamespace(
            draft_runners=[
                SimpleNamespace(
                    weight_updater=_RecordingWeightUpdater(
                        f"draft-{index}", self.events, result
                    )
                )
                for index, result in enumerate(runner_results)
            ]
        )

    def init_cuda_graphs(self):
        self.events.append(("capture",))


def _request(*, recapture_cuda_graph):
    return SimpleNamespace(
        model_path="/models/mimo-v2",
        load_format="safetensors",
        recapture_cuda_graph=recapture_cuda_graph,
    )


class TestBaseSpecWorkerWeightUpdate(CustomTestCase):
    def assertLoads(self, worker, names, *, captured):
        expected = [
            (
                "load",
                name,
                "/models/mimo-v2",
                "safetensors",
                False,
            )
            for name in names
        ]
        if captured:
            expected.append(("capture",))
        self.assertEqual(worker.events, expected)

    def test_recapture_updates_single_runner_before_coordinator_capture(self):
        worker = _RecordingSpecWorker([_SUCCESS])

        result = worker.update_weights_from_disk(_request(recapture_cuda_graph=True))

        self.assertEqual(result, _SUCCESS)
        self.assertLoads(worker, ["draft-0"], captured=True)

    def test_recapture_updates_all_runners_before_one_coordinator_capture(self):
        worker = _RecordingSpecWorker([_SUCCESS, _SUCCESS, _SUCCESS])

        result = worker.update_weights_from_disk(_request(recapture_cuda_graph=True))

        self.assertEqual(result, _SUCCESS)
        self.assertLoads(
            worker,
            ["draft-0", "draft-1", "draft-2"],
            captured=True,
        )

    def test_without_recapture_updates_all_runners_without_capture(self):
        worker = _RecordingSpecWorker([_SUCCESS, _SUCCESS])

        result = worker.update_weights_from_disk(_request(recapture_cuda_graph=False))

        self.assertEqual(result, _SUCCESS)
        self.assertLoads(worker, ["draft-0", "draft-1"], captured=False)

    def test_failure_stops_before_later_runner_and_capture(self):
        failure = (False, "draft-1 load failed")
        worker = _RecordingSpecWorker([_SUCCESS, failure, _SUCCESS])

        result = worker.update_weights_from_disk(_request(recapture_cuda_graph=True))

        self.assertEqual(result, failure)
        self.assertLoads(worker, ["draft-0", "draft-1"], captured=False)


if __name__ == "__main__":
    unittest.main()
