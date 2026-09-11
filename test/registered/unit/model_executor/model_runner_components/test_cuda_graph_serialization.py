"""CPU unit tests for the graph-serialization lifecycle component.

``plan_graph_serialization`` reads the published ``exec.graph`` bag (the
``override_server_args(...).install()`` idiom publishes a partial config on
CPU) and forces the plan off for non-CUDA devices and draft workers;
``finalize_graph_serialization`` is a no-op for a disabled plan and a loud
stub past the per-runner ``finish()`` otherwise; ``cuda_graph_setup``
delegates to the two functions and threads the resolved plan into the runner
constructors (an enabled plan only), so the runners never re-read the config.
"""

import ast
import pathlib
import sys
from types import SimpleNamespace
from unittest import mock

import pytest

from sglang.srt.model_executor.graph_serialization.plan import (
    CacheMode,
    GraphSerializationPlan,
    Placement,
)
from sglang.srt.model_executor.model_runner_components import (
    cuda_graph_serialization as component,
)
from sglang.srt.model_executor.model_runner_components import (
    cuda_graph_setup,
)
from sglang.srt.model_executor.model_runner_components.cuda_graph_serialization import (
    finalize_graph_serialization,
    force_memory_pool_config,
    plan_graph_serialization,
)
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=8, suite="base-a-test-cpu")

_CACHE_MODE_DECLARED = "cuda_graph_cache_mode" in ServerArgs.__dataclass_fields__
needs_cache_mode_leaf = pytest.mark.skipif(
    not _CACHE_MODE_DECLARED,
    reason="ServerArgs.cuda_graph_cache_mode is not declared in this tree",
)


@pytest.fixture
def publish():
    """Publish a partial config for the test body and restore it after."""
    from sglang.srt.runtime_context import get_context

    installed = []

    def _publish(**fields):
        override = get_context().override_server_args(**fields)
        override.install()
        installed.append(override)
        return override

    yield _publish
    for override in reversed(installed):
        override.restore()


def test_default_config_is_off_and_returned_unchanged(publish):
    publish(
        cuda_graph_config=SimpleNamespace(decode=SimpleNamespace(backend="default"))
    )
    plan = plan_graph_serialization(device="cuda", is_draft_worker=False)
    assert isinstance(plan, GraphSerializationPlan)
    assert not plan.enabled
    assert plan.mode is CacheMode.OFF
    assert plan.disabled_reason == ""


@needs_cache_mode_leaf
def test_non_cuda_device_is_forced_off_with_reason(publish, tmp_path):
    publish(cuda_graph_cache_mode="save", cuda_graph_cache_dir=str(tmp_path))
    plan = plan_graph_serialization(device="cpu", is_draft_worker=False)
    assert not plan.enabled
    assert plan.mode is CacheMode.OFF
    assert "'cpu'" in plan.disabled_reason
    # The rest of the plan survives the demotion for diagnostics.
    assert plan.cache_dir == str(tmp_path)


@needs_cache_mode_leaf
def test_draft_worker_is_forced_off_with_reason(publish, tmp_path):
    publish(cuda_graph_cache_mode="load", cuda_graph_cache_dir=str(tmp_path))
    plan = plan_graph_serialization(device="cuda", is_draft_worker=True)
    assert not plan.enabled
    assert "capture-only" in plan.disabled_reason
    assert "9.3" in plan.disabled_reason


@needs_cache_mode_leaf
def test_enabled_when_the_bag_carries_save(publish, tmp_path, monkeypatch):
    logged = []
    monkeypatch.setattr(
        component, "log_info_on_rank0", lambda _logger, msg: logged.append(msg)
    )
    publish(cuda_graph_cache_mode="save", cuda_graph_cache_dir=str(tmp_path))

    plan = plan_graph_serialization(device="cuda", is_draft_worker=False)

    assert plan.enabled
    assert plan.mode is CacheMode.SAVE
    assert plan.saves and not plan.loads
    assert plan.cache_dir == str(tmp_path)
    assert plan.placement is Placement.RELOCATE
    assert plan.disabled_reason == ""
    assert len(logged) == 1
    assert "mode=save" in logged[0]
    assert str(tmp_path) in logged[0]


def test_plan_never_reads_a_server_args_instance():
    # The component must go through the bag (read_plan_from_config), never a
    # ServerArgs record: the ratchet tests enforce the same at package level.
    source = pathlib.Path(component.__file__).read_text(encoding="utf-8")
    assert "server_args" not in source
    assert "read_plan_from_config" in source


def test_finalize_is_a_noop_for_a_disabled_plan():
    finish = mock.Mock(name="finish")
    runner = SimpleNamespace(materializer=SimpleNamespace(finish=finish))
    disabled = GraphSerializationPlan(mode=CacheMode.SAVE).disabled("test")
    assert not disabled.enabled

    assert (
        finalize_graph_serialization(
            disabled, prefill_runner=runner, decode_runner=runner
        )
        is None
    )
    assert (
        finalize_graph_serialization(
            GraphSerializationPlan(), prefill_runner=None, decode_runner=None
        )
        is None
    )
    finish.assert_not_called()


def test_finalize_collects_finish_then_stops_at_the_lifecycle_stub():
    prefill_finish = mock.Mock(name="prefill_finish", return_value=None)
    prefill_runner = SimpleNamespace(
        materializer=SimpleNamespace(finish=prefill_finish)
    )
    # An EagerRunner (or a None phase) owns no materializer and is skipped.
    decode_runner = SimpleNamespace()

    with pytest.raises(NotImplementedError) as caught:
        finalize_graph_serialization(
            GraphSerializationPlan(mode=CacheMode.SAVE, cache_dir="/nonexistent"),
            prefill_runner=prefill_runner,
            decode_runner=decode_runner,
        )

    prefill_finish.assert_called_once_with()
    assert "section 12" in str(caught.value)
    assert "prefill" in str(caught.value)


def test_finalize_propagates_a_materializer_finish_stub():
    # An ArtifactGraphMaterializer.finish() that raises (exports pending) is
    # not swallowed.
    runner = SimpleNamespace(
        materializer=SimpleNamespace(
            finish=mock.Mock(side_effect=NotImplementedError("finish stub"))
        )
    )
    with pytest.raises(NotImplementedError, match="finish stub"):
        finalize_graph_serialization(
            GraphSerializationPlan(mode=CacheMode.LOAD),
            prefill_runner=None,
            decode_runner=runner,
        )


def test_force_memory_pool_config_is_a_documented_stub():
    with pytest.raises(NotImplementedError) as caught:
        force_memory_pool_config(GraphSerializationPlan(mode=CacheMode.LOAD))
    assert "section 10" in str(caught.value)
    assert "section 12" in str(caught.value)


def test_cuda_graph_setup_delegates_to_the_component():
    assert cuda_graph_setup.plan_graph_serialization is plan_graph_serialization
    assert cuda_graph_setup.finalize_graph_serialization is finalize_graph_serialization

    tree = ast.parse(
        pathlib.Path(cuda_graph_setup.__file__).read_text(encoding="utf-8")
    )
    capture = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "capture_cuda_graphs"
    )
    calls = {}
    for node in ast.walk(capture):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            calls.setdefault(node.func.id, []).append(node)

    (plan_call,) = calls["plan_graph_serialization"]
    (finalize_call,) = calls["finalize_graph_serialization"]
    assert {kw.arg for kw in plan_call.keywords} == {"device", "is_draft_worker"}
    assert {kw.arg for kw in finalize_call.keywords} == {
        "prefill_runner",
        "decode_runner",
    }
    # Order: after the eager runner, before the forward-hook registration.
    (eager_call,) = calls["EagerRunner"]
    (hooks_call,) = calls["register_forward_hooks"]
    assert eager_call.lineno < plan_call.lineno < finalize_call.lineno
    assert finalize_call.lineno < hooks_call.lineno

    # The resolved plan reaches every runner construction site: the runners
    # consume it instead of re-reading the config (design section 6.8).
    (prefill_call,) = calls["capture_prefill_graph"]
    decode_calls = calls["capture_decode_graph"]
    assert len(decode_calls) == 2
    for call in (prefill_call, *decode_calls):
        keyword = next(
            kw for kw in call.keywords if kw.arg == "graph_serialization_plan"
        )
        assert ast.unparse(keyword.value) == "plan"
        assert plan_call.lineno < call.lineno < finalize_call.lineno


def test_runner_kwargs_carry_only_an_enabled_plan():
    enabled = GraphSerializationPlan(mode=CacheMode.SAVE, cache_dir="/nonexistent")
    assert cuda_graph_setup._runner_kwargs(None) == {}
    assert cuda_graph_setup._runner_kwargs(GraphSerializationPlan()) == {}
    assert cuda_graph_setup._runner_kwargs(enabled.disabled("device 'npu'")) == {}
    assert cuda_graph_setup._runner_kwargs(enabled) == {
        "graph_serialization_plan": enabled
    }


def _decode_capture_model_runner(runner_cls):
    class TestModelRunner:
        is_generation = True
        device = "cuda"
        gpu_id = 0
        is_draft_worker = False
        spec_algorithm = SimpleNamespace(is_speculative=lambda: False)
        server_args = SimpleNamespace(model_impl="auto")

        def _decode_cuda_graph_runner_cls(self):
            return runner_cls

    return TestModelRunner()


def test_capture_decode_graph_hands_an_enabled_plan_to_the_runner(publish, monkeypatch):
    publish(
        cuda_graph_config=SimpleNamespace(decode=SimpleNamespace(backend="default"))
    )
    monkeypatch.setattr(cuda_graph_setup, "check_cuda_graph_backend", lambda *_: False)
    monkeypatch.setattr(cuda_graph_setup, "get_available_gpu_memory", lambda *_: 10.0)
    monkeypatch.setattr(
        cuda_graph_setup, "get_batch_sizes_to_capture", lambda *_: ([1], None)
    )
    monkeypatch.setattr(
        cuda_graph_setup.current_platform, "is_out_of_tree", lambda: False
    )

    class PlanAwareGraphRunner:
        def __init__(self, model_runner, *, graph_serialization_plan=None):
            self.model_runner = model_runner
            self.graph_serialization_plan = graph_serialization_plan

    class BareGraphRunner:
        def __init__(self, model_runner):
            self.model_runner = model_runner

    enabled = GraphSerializationPlan(mode=CacheMode.LOAD, cache_dir="/nonexistent")
    capture = cuda_graph_setup.capture_decode_graph(
        model_runner=_decode_capture_model_runner(PlanAwareGraphRunner),
        graph_serialization_plan=enabled,
    )
    assert isinstance(capture.runner, PlanAwareGraphRunner)
    assert capture.runner.graph_serialization_plan is enabled

    # A disabled plan (the component's device / draft-worker gate) or no plan
    # passes nothing, so a bare (model_runner) constructor keeps working.
    for plan in (None, enabled.disabled("device 'npu' is not cuda")):
        capture = cuda_graph_setup.capture_decode_graph(
            model_runner=_decode_capture_model_runner(BareGraphRunner),
            graph_serialization_plan=plan,
        )
        assert isinstance(capture.runner, BareGraphRunner)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
