import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, call

import pytest

from sglang.srt.model_executor.model_runner_components import cuda_graph_setup
from sglang.srt.model_executor.model_runner_components.cuda_graph_setup import (
    capture_decode_graph,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def test_model_runner_can_override_decode_graph_runner(monkeypatch):
    class CustomGraphRunner:
        def __init__(self, model_runner):
            self.model_runner = model_runner

    class TestModelRunner:
        is_generation = True
        device = "cuda"
        gpu_id = 0
        is_draft_worker = False
        spec_algorithm = SimpleNamespace(is_speculative=lambda: False)
        server_args = SimpleNamespace(
            model_impl="auto",
            cuda_graph_config=SimpleNamespace(
                decode=SimpleNamespace(backend="default")
            ),
        )

        def _decode_cuda_graph_runner_cls(self):
            return CustomGraphRunner

    model_runner = TestModelRunner()
    monkeypatch.setattr(cuda_graph_setup, "check_cuda_graph_backend", lambda *_: False)
    monkeypatch.setattr(cuda_graph_setup, "get_available_gpu_memory", lambda *_: 10.0)
    monkeypatch.setattr(
        cuda_graph_setup, "get_batch_sizes_to_capture", lambda *_: ([1], None)
    )
    monkeypatch.setattr(
        cuda_graph_setup.current_platform, "is_out_of_tree", lambda: False
    )

    capture = capture_decode_graph(model_runner=model_runner)

    assert isinstance(capture.runner, CustomGraphRunner)
    assert capture.runner.model_runner is model_runner


def test_cuda_graph_prewarm_delegates_to_the_language_model(monkeypatch):
    prewarm = MagicMock(name="prewarm_cuda_graphs")
    language_model = SimpleNamespace(prewarm_cuda_graphs=prewarm)
    runner = SimpleNamespace(
        device="cuda",
        model=object(),
        server_args=SimpleNamespace(
            cuda_graph_config=SimpleNamespace(
                prefill=SimpleNamespace(backend="full"),
                decode=SimpleNamespace(backend="piecewise"),
            )
        ),
    )
    monkeypatch.setattr(
        cuda_graph_setup, "resolve_language_model", lambda _: language_model
    )
    monkeypatch.setattr(
        cuda_graph_setup, "is_sm120_supported", lambda: True, raising=False
    )

    cuda_graph_setup._prewarm_model_cuda_graphs(runner, capture_decode_cuda_graph=True)

    prewarm.assert_called_once_with(runner, capture_decode_cuda_graph=True)


def test_cuda_graph_prewarm_is_required_for_ple_offload(monkeypatch):
    runner = SimpleNamespace(
        device="cuda",
        model=object(),
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(ple_offload_embedding=True)
        ),
        server_args=SimpleNamespace(
            cuda_graph_config=SimpleNamespace(
                prefill=SimpleNamespace(backend="full"),
                decode=SimpleNamespace(backend="piecewise"),
            )
        ),
    )
    monkeypatch.setattr(
        cuda_graph_setup,
        "resolve_language_model",
        lambda _: SimpleNamespace(),
    )
    monkeypatch.setattr(
        cuda_graph_setup, "is_sm120_supported", lambda: True, raising=False
    )

    with pytest.raises(RuntimeError, match="PLE offload.*prewarm_cuda_graphs"):
        cuda_graph_setup._prewarm_model_cuda_graphs(
            runner, capture_decode_cuda_graph=True
        )


def test_cuda_graph_prewarm_does_not_reach_non_sm120_models(monkeypatch):
    prewarm = MagicMock(name="prewarm_cuda_graphs")
    runner = SimpleNamespace(
        device="cuda",
        model=SimpleNamespace(prewarm_cuda_graphs=prewarm),
        server_args=SimpleNamespace(
            cuda_graph_config=SimpleNamespace(
                prefill=SimpleNamespace(backend="full"),
                decode=SimpleNamespace(backend="piecewise"),
            )
        ),
    )
    monkeypatch.setattr(cuda_graph_setup, "resolve_language_model", lambda model: model)
    monkeypatch.setattr(
        cuda_graph_setup, "is_sm120_supported", lambda: False, raising=False
    )

    cuda_graph_setup._prewarm_model_cuda_graphs(runner, capture_decode_cuda_graph=True)

    prewarm.assert_not_called()


def test_cuda_graph_prewarm_does_not_reach_sm121_models(monkeypatch):
    prewarm = MagicMock(name="prewarm_cuda_graphs")
    runner = SimpleNamespace(
        device="cuda",
        model=SimpleNamespace(prewarm_cuda_graphs=prewarm),
        server_args=SimpleNamespace(
            cuda_graph_config=SimpleNamespace(
                prefill=SimpleNamespace(backend="full"),
                decode=SimpleNamespace(backend="piecewise"),
            )
        ),
    )
    monkeypatch.setattr(cuda_graph_setup, "resolve_language_model", lambda model: model)
    monkeypatch.setattr(
        cuda_graph_setup, "is_sm120_supported", lambda: True, raising=False
    )
    monkeypatch.setattr(cuda_graph_setup, "is_sm121", lambda: True, raising=False)

    cuda_graph_setup._prewarm_model_cuda_graphs(runner, capture_decode_cuda_graph=True)

    prewarm.assert_not_called()


def test_capture_cuda_graphs_prewarms_before_prefill_capture(monkeypatch):
    runner = SimpleNamespace(
        device="cpu",
        model=object(),
        model_config=SimpleNamespace(quantization=None),
        is_draft_worker=False,
        server_args=SimpleNamespace(
            moe_runner_backend="cutlass",
            moe_a2a_backend="none",
            forward_hooks=None,
            enable_symm_mem=False,
        ),
        forward_stream=None,
        canary_manager=None,
    )
    eager_runner = object()
    calls = MagicMock()
    prewarm = calls.prewarm
    capture_prefill = calls.capture_prefill
    prefill = cuda_graph_setup.GraphCapture(
        runner=eager_runner,
        memory_phase="prefill",
        memory_usage_gb=0,
        capture_time=0,
    )
    monkeypatch.setattr(
        cuda_graph_setup.GraphSharedOutput,
        "create_for_model_runner",
        lambda _: object(),
    )
    monkeypatch.setattr(cuda_graph_setup, "EagerRunner", lambda _: eager_runner)
    monkeypatch.setattr(cuda_graph_setup, "_prewarm_model_cuda_graphs", prewarm)
    capture_prefill.return_value = prefill
    monkeypatch.setattr(cuda_graph_setup, "capture_prefill_graph", capture_prefill)
    monkeypatch.setattr(
        cuda_graph_setup, "prealloc_symmetric_memory_pool", lambda **_: None
    )

    cuda_graph_setup.capture_cuda_graphs(
        model_runner=runner, capture_decode_cuda_graph=False
    )

    assert calls.mock_calls[:2] == [
        call.prewarm(runner, capture_decode_cuda_graph=False),
        call.capture_prefill(model_runner=runner, eager_runner=eager_runner),
    ]


def test_cuda_graph_prewarm_skips_when_both_phases_are_disabled(monkeypatch):
    prewarm = MagicMock(name="prewarm_cuda_graphs")
    runner = SimpleNamespace(
        device="cuda",
        model=SimpleNamespace(prewarm_cuda_graphs=prewarm),
        server_args=SimpleNamespace(
            cuda_graph_config=SimpleNamespace(
                prefill=SimpleNamespace(backend="disabled"),
                decode=SimpleNamespace(backend="disabled"),
            )
        ),
    )
    monkeypatch.setattr(cuda_graph_setup, "resolve_language_model", lambda model: model)
    monkeypatch.setattr(
        cuda_graph_setup, "is_sm120_supported", lambda: True, raising=False
    )

    cuda_graph_setup._prewarm_model_cuda_graphs(runner, capture_decode_cuda_graph=True)

    prewarm.assert_not_called()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
