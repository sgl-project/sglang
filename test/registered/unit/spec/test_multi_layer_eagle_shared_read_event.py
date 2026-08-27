from types import SimpleNamespace

from sglang.srt.speculative.multi_layer_eagle_draft_extend_cuda_graph_runner import (
    MultiLayerEagleMultiStepDraftExtendCudaGraphRunner,
)
from sglang.srt.speculative.multi_layer_eagle_worker_v2 import MultiLayerEagleWorkerV2
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def test_last_shared_read_runner_is_final_draft_runner():
    worker = MultiLayerEagleWorkerV2.__new__(MultiLayerEagleWorkerV2)
    draft_runners = [object(), object(), object()]
    worker._draft_worker = SimpleNamespace(draft_runner_list=draft_runners)

    assert worker.last_shared_read_runner is draft_runners[-1]


def test_only_final_draft_step_publishes_shared_read_event():
    recorded = []
    event = SimpleNamespace(record=lambda: recorded.append("record"))
    final_runner = SimpleNamespace(
        device_module=SimpleNamespace(Event=lambda: event),
        model_runner=SimpleNamespace(shared_read_done_event=None),
    )
    runner = MultiLayerEagleMultiStepDraftExtendCudaGraphRunner.__new__(
        MultiLayerEagleMultiStepDraftExtendCudaGraphRunner
    )
    runner.runners = [SimpleNamespace(), final_runner]
    runner.speculative_num_steps = 2

    runner._publish_shared_read_done(0)
    assert recorded == []
    assert final_runner.model_runner.shared_read_done_event is None

    runner._publish_shared_read_done(1)
    assert recorded == ["record"]
    assert final_runner.model_runner.shared_read_done_event is event
