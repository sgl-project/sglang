from __future__ import annotations

import json
from dataclasses import replace
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


_EPILOGUE_ROW = "prefill.per_expert.cutedsl_blockscaled"


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

    for family in ("nvfp4", "mxfp4", "fp8", "fp8_channel", "fp8_tensor", "mxfp8"):
        assert row(family, "marlin").name == "decode.shared.marlin", family
    assert row("fp8", "cutedsl").name == "decode.shared"
    large_fp8 = row("fp8", "marlin", experts=512)
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
        "rank_multiple",
        "act_scale",
    ),
)
def test_invalid_unselected_rows_fail_at_load(override_table, defect):
    data, write = override_table
    rows = {row["name"]: row for row in (*data["scenarios"], *data["fallback"])}
    if defect == "rank_multiple":
        rows[_EPILOGUE_ROW]["rank_multiple"] = 0
        error = "rank_multiple"
    elif defect == "act_scale":
        rows[_EPILOGUE_ROW]["act_scale"] = ["per_block"]
        error = "act_scale"
    elif defect == "wildcard":
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


@pytest.mark.parametrize(
    "family,vendor,rank,scale_mode,fused",
    [
        ("nvfp4", "cutedsl", 0, "static", False),
        ("nvfp4", "cutedsl", 8, "static", True),
        ("nvfp4", "cutedsl", 9, "static", False),
        ("nvfp4", "cutedsl", 16, "per_token", True),
        ("nvfp4", "cutedsl", 24, "per_token", False),
        ("nvfp4", "marlin", 16, "static", False),
        ("mxfp4", "cutedsl", 16, "static", True),
        ("mxfp8", "cutedsl", 16, "static", True),
    ],
)
def test_epilogue_selection(family, vendor, rank, scale_mode, fused):
    prefill = _select(family, vendor, rank, act_scale=scale_mode)[ep.Phase.PREFILL]
    assert prefill.name == (_EPILOGUE_ROW if fused else "prefill.per_expert")
    assert prefill.plan.down_b.family is (
        ep.BFamily.EPILOGUE if fused else ep.BFamily.GROUPED
    )


def test_override_tables_may_set_a_rank_floor(override_table):
    data, write = override_table
    next(r for r in data["scenarios"] if r["name"] == _EPILOGUE_ROW)["min_rank"] = 16
    write()
    assert _select("nvfp4", "cutedsl", 8)[ep.Phase.PREFILL].name == "prefill.per_expert"
    assert _select("nvfp4", "cutedsl", 16)[ep.Phase.PREFILL].name == _EPILOGUE_ROW


def test_epilogue_down_b_plans_need_the_bridge_and_own_the_delta():
    plan = _select("nvfp4", "cutedsl", 16)[ep.Phase.PREFILL].plan
    for late in (dict(down_overlap=ep.DownOverlap.DOWN_A), dict(down_b_into_base=True)):
        with pytest.raises(ValueError, match="epilogue down-B"):
            replace(plan, **late)
    for site, is_shared_outer in ((ep.Site.GATE_UP, False), (ep.Site.DOWN, True)):
        with pytest.raises(ValueError, match="per-expert down delta"):
            ep.BSpec(site, ep.BFamily.EPILOGUE, is_shared_outer=is_shared_outer)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
