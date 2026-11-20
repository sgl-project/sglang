from __future__ import annotations

import json
from pathlib import Path

import pytest

from sglang.kernels.ops.lora.moe.activation import ActivationFn
from sglang.srt.lora.moe import plan as ep
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


@pytest.fixture(autouse=True)
def _packaged_plans(monkeypatch):
    monkeypatch.delenv("SGLANG_LORA_MOE_CONFIG_DIR", raising=False)
    ep._load_plans.cache_clear()
    try:
        yield
    finally:
        ep._load_plans.cache_clear()


@pytest.fixture
def override_table(tmp_path, monkeypatch):
    path = tmp_path / "sm100.plans.json"
    data = json.loads((Path(ep._CONFIG_DIR) / path.name).read_text())
    monkeypatch.setenv("SGLANG_LORA_MOE_CONFIG_DIR", str(tmp_path))

    def write():
        path.write_text(json.dumps(data))
        ep._load_plans.cache_clear()

    return data, write


def _select(family, vendor, rank, *, shared=False, experts=256, act_scale="static"):
    return ep.resolve_plans(
        quant_family=family,
        vendor=vendor,
        architecture="sm100",
        is_shared_outer=shared,
        physical_rank=rank,
        activation=ActivationFn.SILU,
        hidden_size=4096,
        num_local_experts=experts,
        act_scale=act_scale,
    )


def test_shared_decode_plan_respects_vendor_and_family():
    def row(family, vendor, experts=256):
        return _select(family, vendor, 16, shared=True, experts=experts)[
            ep.Phase.DECODE
        ]

    for family in ("nvfp4", "mxfp4"):
        expected = (
            "decode.shared.nvfp4" if family == "nvfp4" else "decode.shared.marlin"
        )
        assert row(family, "marlin").name == expected, family
    assert row("fp8", "cutedsl").name == "decode.shared"
    large_fp8 = row("fp8", "cutedsl", experts=512)
    assert (large_fp8.name, large_fp8.base_gemm_rows) == (
        "decode.shared.fp8.e_ge512",
        "expert_major",
    )


@pytest.mark.parametrize(
    "defect",
    (
        "wildcard",
        "token_dense_b_act",
        "epilogue_with_shared_finalize",
        "per_expert_token_a",
    ),
)
def test_invalid_unselected_rows_fail_at_load(override_table, defect):
    data, write = override_table
    rows = {row["name"]: row for row in (*data["scenarios"], *data["fallback"])}
    if defect == "wildcard":
        row = rows["decode.shared"]
        row["layout"] = None
        error = "requires shared-outer down-B ownership"
    elif defect == "token_dense_b_act":
        row = rows["prefill.shared"]
        row["plan"]["gate_up_a_family"] = "token_dense"
        error = "only a standalone per_row gate/up B"
    elif defect == "epilogue_with_shared_finalize":
        rows["prefill.shared"]["plan"]["down_b_family"] = "epilogue"
        error = "epilogue down-B requires a materialized finalize"
    else:
        row = rows["fallback.prefill.per_expert"]
        row["plan"]["gate_up_a_family"] = "token_grouped"
        error = "requires shared-outer A ownership"
    write()
    with pytest.raises(ValueError, match=error):
        _select("bf16", "cutedsl", 16)  # selects none of the rows above


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
