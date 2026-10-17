import importlib.util
from pathlib import Path

import pytest

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


@pytest.fixture
def tuner():
    path = (
        Path(__file__).resolve().parents[5]
        / "benchmark/kernels/fused_moe_triton/tune_down_moe.py"
    )
    spec = importlib.util.spec_from_file_location("down_moe_tuning", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module._PIN.update(gate_up={"BLOCK_SIZE_M": 16}, down={"BLOCK_SIZE_M": 16})
    return module


def test_config_pin_uses_the_runtime_return_argument(tuner):
    args = ((), (), 2, None, 8, False, [128, 128], False)
    assert tuner._patched_try_get_optimal(*args) == {"BLOCK_SIZE_M": 16}
    for invoke in (
        lambda: tuner._patched_try_get_optimal(*args, True),
        lambda: tuner._patched_try_get_optimal(*args, return_down_config=True),
    ):
        up, (down, block_m) = invoke()
        assert up == down == {"BLOCK_SIZE_M": block_m} == {"BLOCK_SIZE_M": 16}
        up["BLOCK_SIZE_M"] = 64
        down["BLOCK_SIZE_M"] = 32
        assert tuner._PIN == {
            "gate_up": {"BLOCK_SIZE_M": 16},
            "down": {"BLOCK_SIZE_M": 16},
        }


def test_candidate_keeps_the_benchmarks_microseconds(tuner):
    assert tuner._benchmark_candidate(lambda: 12.5) == (12.5, None)


@pytest.mark.parametrize("fatal", [False, True])
@pytest.mark.parametrize("message", ["illegal memory access", "device-side assert"])
def test_candidate_does_not_swallow_fatal_cuda_errors(tuner, fatal, message):
    error = RuntimeError(f"CUDA error: {message}" if fatal else "out of resources")

    def fail():
        raise error

    if fatal:
        with pytest.raises(RuntimeError) as caught:
            tuner._benchmark_candidate(fail)
        assert caught.value is error
    else:
        assert tuner._benchmark_candidate(fail) == (None, repr(error))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
