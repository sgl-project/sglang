"""CPU tests for NPU graph update handling, with real background threads."""

import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from sglang.srt.hardware_backend.npu.graph_runner.npu_cudagraph_backend import (
    NPUCudaGraphBackend,
)
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


@pytest.fixture
def make_backend(request):
    def make(graph):
        runner = SimpleNamespace(
            device_module=SimpleNamespace(
                current_device=Mock(return_value=0),
                set_device=Mock(),
                Event=Mock(side_effect=lambda: Mock(spec=["record", "synchronize"])),
            ),
        )
        with get_parallel().override(tp_group=Mock()):
            backend = NPUCudaGraphBackend(runner)
        request.addfinalizer(backend.cleanup)
        backend._graphs = {1: graph}
        backend._outputs = {1: object()}
        return backend

    return make


@pytest.mark.parametrize("legacy", [False, True])
def test_npu_graph_update_success(legacy, make_backend):
    graph = SimpleNamespace(update=Mock(), replay=Mock())
    backend = make_backend(graph)
    if legacy:
        output = backend.replay_with_input_update(
            1, [3, 5], attr_name="actual_seq_lengths_kv", attr_type=torch.empty(0)
        )
        values = graph.update.call_args.kwargs["cpu_update_input"][0]
        torch.testing.assert_close(
            values["actual_seq_lengths_kv"], torch.tensor([3, 5], dtype=torch.int32)
        )
    else:
        inputs = [{"actual_seq_lengths_kv": [3]}, {"actual_seq_lengths_kv": [4]}]
        output = backend.replay_with_input_update(1, None, cpu_update_input=inputs)
        graph.update.assert_called_once_with(cpu_update_input=inputs)
    assert output is backend._outputs[1]
    graph.replay.assert_called_once_with()
    backend._device_module.set_device.assert_called_once_with(0)
    backend._rebind_fence.record.assert_called_once_with()
    backend._rebind_fence.synchronize.assert_not_called()


def test_npu_graph_overlaps_update_and_replay_and_reuses_worker(make_backend):
    caller_thread = threading.get_ident()
    update_threads = []
    events = []
    replay_started = threading.Event()

    def update(**kwargs):
        backend._device_module.set_device.assert_called_once_with(0)
        update_threads.append(threading.get_ident())
        events.append("update")
        assert replay_started.wait(timeout=5), (
            "replay must not wait for update to finish"
        )
        events.append("update_done")

    def replay():
        events.append("replay")
        replay_started.set()

    graph = SimpleNamespace(update=update, replay=replay)
    backend = make_backend(graph)
    for _ in range(2):
        replay_started.clear()
        events.clear()
        previous_fence = backend._rebind_fence
        if previous_fence is not None:
            previous_fence.synchronize.side_effect = lambda: events.append(
                "synchronize"
            )
        result = backend.replay_with_input_update(1, None, cpu_update_input=[{}, {}])
        assert result is backend._outputs[1]
        assert events.count("update") == events.count("replay") == 1
        assert events[-1] == "update_done"
        if previous_fence is not None:
            previous_fence.synchronize.assert_called_once_with()
            assert events[0] == "synchronize"
        assert backend._rebind_fence is not previous_fence
        backend._rebind_fence.record.assert_called_once_with()
    assert update_threads[0] == update_threads[1] != caller_thread
    backend._device_module.set_device.assert_called_once_with(0)


@pytest.mark.parametrize("failure", ["update", "replay"])
def test_npu_graph_propagates_failures(failure, make_backend):
    graph = SimpleNamespace(update=Mock(), replay=Mock())
    error = RuntimeError(f"{failure} failed")
    getattr(graph, failure).side_effect = error
    backend = make_backend(graph)
    with pytest.raises(RuntimeError) as exc_info:
        backend.replay_with_input_update(1, None, cpu_update_input=[{}])
    assert exc_info.value is error
    graph.replay.assert_called_once_with()
    assert backend._rebind_fence is None


def test_npu_graph_cleanup_stops_update_worker(make_backend):
    graph = SimpleNamespace(update=Mock(), replay=Mock())
    backend = make_backend(graph)
    backend.replay_with_input_update(1, None, cpu_update_input=[{}])
    backend.cleanup()
    assert backend._graphs == {} and backend._outputs == {}
    assert backend._pool is None
    with pytest.raises(RuntimeError, match="shutdown"):
        backend._update_executor.submit(lambda: None)


@pytest.mark.parametrize(
    "runner_kind", ["target_decode", "target_verify", "draft", "draft_extend"]
)
@pytest.mark.parametrize(
    "architecture,qsa,dsa,skip_update",
    [
        ("Qwen4ExpForConditionalGeneration", True, False, True),
        ("Qwen4ExpForCausalLMMTP", True, False, True),
        ("Qwen4ExpForConditionalGeneration", False, False, False),
        ("Qwen4ExpForCausalLMMTP", False, False, False),
        ("Qwen3ForCausalLM", False, False, False),
        ("Qwen3ForCausalLM", True, False, False),
        ("DeepseekV32ForCausalLM", False, True, True),
        ("GlmMoeDsaForCausalLM", False, True, True),
        ("DeepseekV4ForCausalLM", False, False, True),
    ],
)
def test_npu_graph_qwen_qsa_replay_dispatch(
    runner_kind, architecture, qsa, dsa, skip_update
):
    from sglang.srt.layers.attention.qwen_sparse_attn_backend import QwenSparseAttnBackend
    from sglang.srt.model_executor.runner.decode_cuda_graph_runner import DecodeCudaGraphRunner
    from sglang.srt.speculative.eagle_draft_cuda_graph_runner import EAGLEDraftCudaGraphRunner
    from sglang.srt.speculative.eagle_draft_extend_cuda_graph_runner import EAGLEDraftExtendCudaGraphRunner

    runner_type = {
        "target_decode": DecodeCudaGraphRunner,
        "target_verify": DecodeCudaGraphRunner,
        "draft": EAGLEDraftCudaGraphRunner,
        "draft_extend": EAGLEDraftExtendCudaGraphRunner,
    }[runner_kind]
    runner = object.__new__(runner_type)
    runner.model_runner = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=SimpleNamespace(
            architectures=[architecture], index_topk=2048 if dsa else None,
        )),
        spec_algorithm=SimpleNamespace(is_dflash=lambda: False, is_dflash_family=lambda: False),
    )
    # The merged backend uses the attention capability, replacing the deleted
    # NPU runner subclasses. Use the actual QSA declaration for this model.
    runner.attn_backend = SimpleNamespace(needs_cpu_seq_lens=(
        QwenSparseAttnBackend.needs_cpu_seq_lens
        if architecture.startswith("Qwen4Exp") and qsa else True
    ))
    runner.bs, runner.raw_bs = 2, 1
    runner.captured_req_width = 4
    runner.speculative_num_steps = 3
    graph, output = SimpleNamespace(replay=Mock()), object()
    backend = object.__new__(NPUCudaGraphBackend)
    backend._decode_runner = runner
    backend._graphs, backend._outputs = {2: graph}, {2: output}
    backend.replay_with_input_update = Mock(return_value=output)
    backend.get_update_attr_name = lambda: "actual_seq_lengths_kv"
    lengths = torch.tensor([7], dtype=torch.int32)
    if skip_update:
        lengths = Mock()
        lengths.tolist.side_effect = AssertionError("Unexpected CPU length preparation")
    batch = SimpleNamespace(
        seq_lens=lengths, seq_lens_cpu=lengths,
        forward_mode=SimpleNamespace(is_target_verify=lambda: runner_kind == "target_verify"),
    )
    assert backend.replay(2, batch) is output
    if skip_update:
        graph.replay.assert_called_once_with()
        backend.replay_with_input_update.assert_not_called()
    else:
        graph.replay.assert_not_called()
        update = backend.replay_with_input_update
        assert update.call_count == 1
        if runner_kind == "draft":
            assert update.call_args.kwargs == {
                "seq_lens": None,
                "cpu_update_input": [
                    {"actual_seq_lengths_kv": [8, 0]},
                    {"actual_seq_lengths_kv": [9, 0]},
                ],
            }
        elif runner_kind == "draft_extend":
            assert update.call_args.kwargs == {
                "seq_lens": None,
                "cpu_update_input": [{"actual_seq_lengths_kv": [7, 0]}],
            }
        else:
            assert update.call_args.kwargs == {
                "seq_lens": [11, 0] if runner_kind == "target_verify" else [7, 0],
                "attr_name": "actual_seq_lengths_kv", "attr_type": [],
            }


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
