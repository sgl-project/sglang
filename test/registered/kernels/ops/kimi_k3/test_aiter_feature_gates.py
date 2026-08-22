import importlib

import pytest

from sglang.srt.utils import is_hip
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=5, suite="stage-b-test-1-gpu-small-amd-mi35x")

pytestmark = pytest.mark.skipif(not is_hip(), reason="Kimi-K3 AITER gates require HIP")


@pytest.mark.parametrize(
    ("module_name", "env_name"),
    [
        (
            "sglang.kernels.ops.kimi_k3.mla_gate_aiter_hip",
            "SGLANG_K3_AITER_MLA_GATE",
        ),
        (
            "sglang.kernels.ops.kimi_k3.kda_group64_aiter_hip",
            "SGLANG_K3_AITER_KDA_GROUP64",
        ),
        (
            "sglang.kernels.ops.kimi_k3.moe_preroute_aiter_hip",
            "SGLANG_K3_AITER_MOE_PREROUTE_FP8",
        ),
        (
            "sglang.kernels.ops.kimi_k3.latent_tail_aiter_hip",
            "SGLANG_K3_AITER_LATENT_TAIL_FP8",
        ),
    ],
)
def test_aiter_feature_is_opt_in(monkeypatch, module_name, env_name):
    module = importlib.import_module(module_name)
    monkeypatch.delenv(env_name, raising=False)
    assert not module.enabled()
    monkeypatch.setenv(env_name, "1")
    assert module.enabled()


def test_moe_preroute_cooperative_feature_is_opt_in(monkeypatch):
    module = importlib.import_module(
        "sglang.kernels.ops.kimi_k3.moe_preroute_aiter_hip"
    )
    monkeypatch.setenv("SGLANG_K3_AITER_MOE_PREROUTE_FP8", "1")
    monkeypatch.setenv("SGLANG_K3_FLYDSL_SOURCE", "sglang")
    monkeypatch.delenv("SGLANG_K3_PREROUTE_PREACTIVATED_SHARED", raising=False)
    assert not module.cooperative_preactivated_enabled()
    monkeypatch.setenv("SGLANG_K3_PREROUTE_PREACTIVATED_SHARED", "1")
    assert module.cooperative_preactivated_enabled()


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
