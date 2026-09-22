"""CPU tests for NPU graph update handling, with real background threads."""

import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from sglang.srt.hardware_backend.npu.graph_runner.npu_cudagraph_backend import (
    NPUCudaGraphBackend,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


@pytest.fixture
def make_backend(request):
    def make(graph):
        runner = SimpleNamespace(
            device_module=SimpleNamespace(
                current_device=Mock(return_value=0), set_device=Mock()
            ),
            model_runner=SimpleNamespace(tp_group=Mock()),
        )
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


def test_npu_graph_waits_for_update_before_replay_and_reuses_worker(make_backend):
    caller_thread = threading.get_ident()
    update_threads = []
    events = []

    def update(**kwargs):
        backend._device_module.set_device.assert_called_once_with(0)
        update_threads.append(threading.get_ident())
        events.append("update")

    graph = SimpleNamespace(update=update, replay=lambda: events.append("replay"))
    backend = make_backend(graph)
    for _ in range(2):
        result = backend.replay_with_input_update(1, None, cpu_update_input=[{}, {}])
        assert result is backend._outputs[1]
    assert events == ["update", "replay", "update", "replay"]
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
    if failure == "update":
        graph.replay.assert_not_called()


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
    from sglang.srt.hardware_backend.npu.graph_runner.eagle_draft_extend_npu_graph_runner import (
        EAGLEDraftExtendNpuGraphRunner,
    )
    from sglang.srt.hardware_backend.npu.graph_runner.eagle_draft_npu_graph_runner import (
        EAGLEDraftNpuGraphRunner,
    )
    from sglang.srt.hardware_backend.npu.graph_runner.npu_graph_runner import (
        NPUGraphRunner,
    )
    from sglang.srt.layers.logits_processor import LogitsProcessorOutput

    text_config = SimpleNamespace()
    if qsa:
        text_config = SimpleNamespace(
            indexer_n_heads=3,
            indexer_kv_heads=1,
            indexer_head_dim=256,
            indexer_budget=2048,
            indexer_compress_ratio=4,
        )
    config = SimpleNamespace(
        architectures=[architecture],
        text_config=text_config,
        index_topk=2048 if dsa else None,
    )
    output = LogitsProcessorOutput(next_token_logits=torch.zeros(2, 3))
    backend = SimpleNamespace(
        replay=Mock(return_value=output),
        replay_with_input_update=Mock(return_value=output),
    )
    runner = SimpleNamespace(
        model_runner=SimpleNamespace(
            model_config=SimpleNamespace(hf_config=config),
            spec_algorithm=SimpleNamespace(is_dflash=lambda: False),
        ),
        backend=backend,
        bs=2,
        raw_bs=1,
        raw_num_token=1,
        captured_req_width=4,
        speculative_num_steps=3,
        is_dllm=False,
        load_batch=Mock(),
        _make_graph_key=lambda bs: bs,
        _get_update_attr_name=lambda: "actual_seq_lengths_kv",
        _get_update_attr_type=lambda: [],
        _replay_attn_backend=lambda: SimpleNamespace(forward_metadata=None),
    )
    lengths = torch.tensor([7], dtype=torch.int32)
    if skip_update:
        # Direct replay must not prepare CPU lengths, including for draft steps.
        lengths = Mock()
        lengths.cpu.side_effect = AssertionError("Unexpected CPU length preparation")
        lengths.tolist.side_effect = AssertionError("Unexpected CPU length preparation")
    forward_batch = SimpleNamespace(
        seq_lens=lengths,
        seq_lens_cpu=lengths,
        needs_forward_metadata_init=lambda: True,
        forward_mode=SimpleNamespace(
            is_target_verify=lambda: runner_kind == "target_verify"
        ),
    )
    if runner_kind.startswith("target"):
        NPUGraphRunner.execute(runner, forward_batch)
        runner.load_batch.assert_called_once_with(forward_batch, None)
    elif runner_kind == "draft":
        EAGLEDraftNpuGraphRunner._replay_graph(runner, 2, forward_batch)
    else:
        EAGLEDraftExtendNpuGraphRunner._replay_graph(runner, 2, forward_batch)

    if skip_update:
        backend.replay.assert_called_once_with(2, forward_batch)
        backend.replay_with_input_update.assert_not_called()
    else:
        backend.replay.assert_not_called()
        update = backend.replay_with_input_update
        assert update.call_count == 1
        assert update.call_args.args == (2,)
        if runner_kind == "draft":
            assert update.call_args.kwargs == {
                "seq_lens": None,
                "cpu_update_input": [
                    {"actual_seq_lengths_kv": [8, 0]},
                    {"actual_seq_lengths_kv": [9, 0]},
                ],
            }
        else:
            expected = [11, 0] if runner_kind == "target_verify" else [7, 0]
            assert update.call_args.kwargs == {
                "seq_lens": expected,
                "attr_name": "actual_seq_lengths_kv",
                "attr_type": [],
            }
