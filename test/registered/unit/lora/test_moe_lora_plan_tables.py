"""Invalid plan-table rows fail at load, even rows the selector never picks."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sglang.srt.lora.moe import plan as ep
from sglang.srt.lora.moe.plan import ActivationFn
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


def _resolve():
    """Per-expert sm100 selection; it never picks the rows the test mutates."""
    return ep.resolve_plans(
        quant_family="bf16",
        architecture="sm100",
        is_shared_outer=False,
        physical_rank=16,
        activation=ActivationFn.SILU,
        hidden_size=4096,
        num_local_experts=256,
    )


@pytest.mark.parametrize(
    "defect", ("wildcard", "token_dense_b_act", "per_expert_token_a")
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
    else:
        row = rows["fallback.prefill.per_expert"]
        row["plan"]["gate_up_a_family"] = "token_grouped"
        error = "requires shared-outer A ownership"
    write()
    with pytest.raises(ValueError, match=error):
        _resolve()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
