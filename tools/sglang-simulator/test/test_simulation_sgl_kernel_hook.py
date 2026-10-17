"""Regression tests for the simulator's import-time CPU-op stand-ins."""

import os
import subprocess
import sys
import textwrap

import pytest
import torch
from sglang_simulator.simulation.sglang.sgl_kernel_hook import (
    install_cpu_op_standins,
)

# Keep these independent of the implementation: dropping a required name must
# fail the tests. qwen3_5 binds all four; muse_glimmer binds the first.
CPU_OP_NAMES = (
    "fused_sigmoid_mul_cpu",
    "fused_qk_gemma_rmsnorm_cpu",
    "fused_qk_gemma_rmsnorm_with_gate_cpu",
    "fused_qkvzba_split_reshape_cat_contiguous_cpu",
)
FEATURE_PROBE_NAMES = (
    "convert_weight_packed",
    "fused_qk_norm_rope_cpu",
    "fused_qk_norm_cpu",
)


@pytest.fixture
def cpu_op_namespace(monkeypatch):
    # Exercise PyTorch's real attribute/dispatcher lookup without deleting or
    # replacing any operators already installed in the test process.
    namespace = torch._ops._OpNamespace("sglang_simulator_cpu_hook_test")
    monkeypatch.setattr(torch.ops, "sgl_kernel", namespace)
    return namespace


@pytest.mark.parametrize("cpu_engine", [None, "0", "true"])
def test_install_is_a_noop_without_cpu_engine(
    monkeypatch, cpu_op_namespace, cpu_engine
):
    if cpu_engine is None:
        monkeypatch.delenv("SGLANG_USE_CPU_ENGINE", raising=False)
    else:
        monkeypatch.setenv("SGLANG_USE_CPU_ENGINE", cpu_engine)

    install_cpu_op_standins()

    for op_name in CPU_OP_NAMES:
        assert not hasattr(cpu_op_namespace, op_name)


def test_standins_resolve_import_time_names_and_raise_when_called(
    monkeypatch, cpu_op_namespace
):
    monkeypatch.setenv("SGLANG_USE_CPU_ENGINE", "1")

    install_cpu_op_standins()

    for op_name in CPU_OP_NAMES:
        op = getattr(cpu_op_namespace, op_name)
        with pytest.raises(RuntimeError, match=rf"{op_name}.*import-time stand-in"):
            op()


def test_install_preserves_registered_ops_and_is_idempotent(
    monkeypatch, cpu_op_namespace
):
    # A real dispatcher registration exercises _OpNamespace.__getattr__, unlike
    # a Python sentinel. Keep the Library alive until after the assertions.
    library = torch.library.Library(cpu_op_namespace.name, "FRAGMENT")
    library.define("fused_sigmoid_mul_cpu(Tensor(a!) input, Tensor gate) -> ()")
    real_op = cpu_op_namespace.fused_sigmoid_mul_cpu
    real_overload = real_op.default
    monkeypatch.setenv("SGLANG_USE_CPU_ENGINE", "1")

    install_cpu_op_standins()
    installed = {name: getattr(cpu_op_namespace, name) for name in CPU_OP_NAMES}
    install_cpu_op_standins()

    assert cpu_op_namespace.fused_sigmoid_mul_cpu is real_op
    assert real_op.default is real_overload
    for name, op in installed.items():
        assert getattr(cpu_op_namespace, name) is op


@pytest.mark.parametrize("available", [False, True])
def test_install_preserves_feature_probes(monkeypatch, cpu_op_namespace, available):
    sentinel = object()
    if available:
        for name in FEATURE_PROBE_NAMES:
            monkeypatch.setattr(cpu_op_namespace, name, sentinel, raising=False)
    monkeypatch.setenv("SGLANG_USE_CPU_ENGINE", "1")

    install_cpu_op_standins()

    for name in FEATURE_PROBE_NAMES:
        assert hasattr(cpu_op_namespace, name) is available
        if available:
            assert getattr(cpu_op_namespace, name) is sentinel


def test_bootstrap_installs_ops_before_spawned_arguments_are_unpickled(tmp_path):
    # Use the actual bootstrap and its dependencies in fresh interpreters. A
    # worker's target is imported before its arguments are unpickled; a model
    # config among those arguments can already resolve CPU ops at that point.
    script = tmp_path / "check_cpu_op_bootstrap.py"
    script.write_text(
        textwrap.dedent(
            f"""\
            import multiprocessing
            import sys

            import torch

            CPU_OP_NAMES = {CPU_OP_NAMES!r}

            def check_import_time_bindings():
                # The target wrapper has not run yet.
                assert (
                    "sglang_simulator.simulation.sglang.hook_bootstrap"
                    in sys.modules
                )
                assert "sglang" not in sys.modules
                for name in CPU_OP_NAMES:
                    op = getattr(torch.ops.sgl_kernel, name)
                    try:
                        op()
                    except RuntimeError as exc:
                        assert "import-time stand-in" in str(exc)
                    else:
                        raise AssertionError(name)
                return True

            class ImportTimeBindings:
                def __reduce__(self):
                    return check_import_time_bindings, ()

            def worker(target_and_argument):
                target, bindings_ready = target_and_argument
                assert target.__name__ == "run_simulator_scheduler_process"
                assert bindings_ready

            if __name__ == "__main__":
                from sglang_simulator.simulation.sglang.hook_bootstrap import (
                    run_simulator_scheduler_process,
                )

                assert check_import_time_bindings()
                context = multiprocessing.get_context("spawn")
                process = context.Process(
                    target=worker,
                    args=((run_simulator_scheduler_process, ImportTimeBindings()),),
                )
                process.start()
                try:
                    process.join(timeout=60)
                    assert process.exitcode == 0, process.exitcode
                finally:
                    if process.is_alive():
                        process.terminate()
                        process.join(timeout=10)
            """
        ),
        encoding="utf-8",
    )
    env = os.environ.copy()
    env["SGLANG_USE_CPU_ENGINE"] = "1"
    result = subprocess.run(
        [sys.executable, str(script)],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
