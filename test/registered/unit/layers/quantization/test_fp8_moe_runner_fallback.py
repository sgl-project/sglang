import sys
from types import SimpleNamespace

from sglang.srt.layers.moe import MoeRunnerBackend, MoeRunnerConfig
from sglang.srt.layers.quantization import fp8
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-c-test-cpu")


def test_flashinfer_cutlass_falls_back_to_triton_for_native_fp8(monkeypatch):
    created = []
    warnings = []

    class StubRunner:
        def __init__(self, backend, config):
            self.runner_backend = backend
            self.config = config
            created.append(self)

    monkeypatch.setattr(
        fp8,
        "get_moe_runner_backend",
        lambda: MoeRunnerBackend.FLASHINFER_CUTLASS,
    )
    monkeypatch.setattr(fp8, "MoeRunner", StubRunner)
    monkeypatch.setattr(fp8, "print_warning_once", warnings.append)

    method = object.__new__(fp8.Fp8MoEMethod)
    method.quant_config = SimpleNamespace(force_triton_moe_runner=True)
    config = MoeRunnerConfig()
    method.create_moe_runner(layer=None, moe_runner_config=config)

    assert len(created) == 1
    assert method.runner is created[0]
    assert method.runner.runner_backend is MoeRunnerBackend.TRITON
    assert method.runner.config is config
    assert warnings and "native FP8 experts" in warnings[0]


def test_stock_fp8_does_not_enable_mtp_runner_fallback(monkeypatch):
    created = []

    class StubRunner:
        def __init__(self, backend, config):
            created.append((backend, config))

    monkeypatch.setattr(
        fp8,
        "get_moe_runner_backend",
        lambda: MoeRunnerBackend.FLASHINFER_CUTLASS,
    )
    monkeypatch.setattr(fp8, "MoeRunner", StubRunner)

    method = object.__new__(fp8.Fp8MoEMethod)
    method.quant_config = SimpleNamespace()
    method.create_moe_runner(layer=None, moe_runner_config=MoeRunnerConfig())

    assert not created
    assert not hasattr(method, "runner")


if __name__ == "__main__":
    sys.exit(__import__("pytest").main([__file__, "-v"]))
