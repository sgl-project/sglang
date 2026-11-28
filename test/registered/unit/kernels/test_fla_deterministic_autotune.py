"""CPU tests for the fixed-config autotune of the KDA prefill kernels.

Under deterministic inference (or SGLANG_KDA_FIXED_AUTOTUNE=1) each kernel's
``early_config_prune`` must keep exactly one of its own configs, so every
process runs the same one; otherwise the list must reach Triton unchanged.
"""

import sys

import pytest
import triton

from sglang.kernels.ops.attention.fla import cumsum, kda
from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _autotuner(kernel):
    # The gate kernel wraps its autotuner in triton.heuristics.
    while not isinstance(kernel, triton.runtime.Autotuner):
        kernel = kernel.fn
    return kernel


def _bs_configs(bs_list):
    return [
        triton.Config({"BS": bs}, num_warps=num_warps)
        for bs in bs_list
        for num_warps in [2, 4, 8]
    ]


def _cases():
    kernels = {
        "gate_cumsum": kda.kda_gate_chunk_cumsum_vector_kernel,
        "local_cumsum": cumsum.chunk_local_cumsum_vector_kernel,
        "gla_fwd_o": kda.chunk_gla_fwd_kernel_o,
    }
    for name, kernel in kernels.items():
        tuner = _autotuner(kernel)
        yield pytest.param(tuner, list(tuner.configs), id=name)
        if "BS" in tuner.configs[0].kwargs:
            # BS_LIST depends on the device's shared memory: cover both lists.
            for bs_list in ([16, 32], [32, 64]):
                yield pytest.param(
                    tuner, _bs_configs(bs_list), id=f"{name}-BS{bs_list[0]}"
                )


CASES = list(_cases())

SWITCHES_ON = [
    (envs.SGLANG_KDA_FIXED_AUTOTUNE, "1"),
    (envs.SGLANG_ENABLE_DETERMINISTIC_INFERENCE, "1"),
]


@pytest.fixture(autouse=True)
def _switches_off():
    with (
        envs.SGLANG_KDA_FIXED_AUTOTUNE.override(False),
        envs.SGLANG_ENABLE_DETERMINISTIC_INFERENCE.override(False),
    ):
        yield


@pytest.mark.parametrize("tuner,configs", CASES)
def test_switch_off_keeps_every_config(tuner, configs):
    assert tuner.early_config_prune(configs, {}) == configs
    with envs.SGLANG_KDA_FIXED_AUTOTUNE.override("0"):
        assert tuner.early_config_prune(configs, {}) == configs


@pytest.mark.parametrize("tuner,configs", CASES)
@pytest.mark.parametrize("switch,value", SWITCHES_ON)
def test_switch_on_keeps_one_listed_config(tuner, configs, switch, value):
    with switch.override(value):
        kept = tuner.early_config_prune(configs, {})
    assert len(kept) == 1
    assert kept[0] in configs


@pytest.mark.parametrize("switch,value", SWITCHES_ON)
def test_pinned_configs(switch, value):
    gate = _autotuner(kda.kda_gate_chunk_cumsum_vector_kernel)
    out = _autotuner(kda.chunk_gla_fwd_kernel_o)
    with switch.override(value):
        for bs_list in ([16, 32], [32, 64]):
            (kept,) = gate.early_config_prune(_bs_configs(bs_list), {})
            assert (kept.kwargs["BS"], kept.num_warps) == (bs_list[0], 4)
        # Through Triton's own pruning step, as at the first launch.
        out.nargs = {}
        (kept,) = out.prune_configs({})
    assert kept.kwargs == {"BK": 64, "BV": 64}
    assert (kept.num_warps, kept.num_stages) == (4, 2)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
