import importlib.util
import json
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


@pytest.fixture
def tuner(monkeypatch):
    root = Path(__file__).resolve().parents[4]
    path = root / "benchmark/kernels/lora_dense/tune_plans.py"
    spec = importlib.util.spec_from_file_location("dense_lora_tuning", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def case_data(**updates):
    return dict(
        name="qkv",
        in_features=64,
        slices=[32, 16],
        tokens=2,
        rank=4,
        pool_rank=8,
        slots=2,
        request_lengths=[1, 1],
        request_slots=[0, -1],
        **updates,
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("rank", 9),
        ("tokens", 3),
        ("slices", []),
        ("in_features", True),
        ("request_slots", [2, -1]),
        ("request_slots", [0]),
        ("request_lengths", [2]),
        ("kind", "sink_down"),
        ("mode", "profile"),
    ],
)
def test_invalid_workload_is_rejected_before_cuda(tuner, field, value):
    value_dict = case_data()
    value_dict[field] = value
    with pytest.raises(ValueError):
        tuner.Case.from_dict(value_dict)


def test_candidates_are_unique_preserve_incumbent_and_respect_family_contract(tuner):
    incumbent = dict(
        a_family="grouped",
        b_family="grouped",
        overlap="none",
        block_size=16,
        a_tiles={
            "SPLIT_K": 4,
            "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 128,
            "num_warps": 4,
            "num_stages": 3,
        },
        b_tiles={"BLOCK_SIZE_N": 64, "num_warps": 4},
    )
    candidates = tuner.candidate_specs(incumbent)
    assert candidates.count(incumbent) == 1
    assert any(p["a_tiles"].get("SPLIT_MODE") == "planes" for p in candidates)
    for plan in candidates:
        if plan["a_family"] == "all_slots":
            assert plan["b_family"] == "per_row"
        if plan["a_family"] != "grouped":
            assert plan["a_tiles"]["SPLIT_K"] == 1


def test_selector_cannot_silently_merge_conflicting_occupancy_or_graph_modes(tuner):
    case = asdict(tuner.Case.from_dict(case_data()))
    rows = [
        dict(case=case, selected={"block_size": 16}),
        dict(
            case={**case, "name": "other", "slots": 4, "mode": "eager"},
            selected={"block_size": 64},
        ),
    ]
    groups = tuner.compatible_selections(rows)
    assert len(groups) == 1 and groups[0]["status"] == "CONFLICT"
    assert groups[0]["spec"] is None
    rows[1]["selected"] = rows[0]["selected"]
    assert tuner.compatible_selections(rows)[0]["status"] == "CONSISTENT"
    rows[1]["case"]["pool_rank"] = 16
    assert len(tuner.compatible_selections(rows)) == 2


def test_reference_uses_packed_a_slices_pool_stride_and_base_only_rows(tuner):
    import torch

    case = tuner.Case.from_dict(case_data())
    data = tuner.Workload(case, 7, torch.device("cpu"))
    base = data.x.float() @ data.weight.float().T
    torch.testing.assert_close(data.reference[1], base[1])
    for index, (lo, hi) in enumerate(zip(data.offsets, data.offsets[1:])):
        a = data.a[0, index * case.rank : (index + 1) * case.rank].float()
        b = data.b[0, lo:hi, : case.rank].float()
        expected = base[0, lo:hi] + b @ (a @ data.x[0].float())
        torch.testing.assert_close(data.reference[0, lo:hi], expected)
    assert torch.count_nonzero(data.b[:, :, case.rank :]) == 0
    assert (data.reference[0] - base[0]).abs().max() > 0.02


def test_effective_candidates_exclude_ignored_knobs_and_clamped_rank_tiles(tuner):
    spec = dict(
        a_family="all_slots",
        b_family="per_row",
        overlap="none",
        block_size=16,
        a_tiles={"BLOCK_SIZE_N": 16},
        b_tiles={"BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8},
    )
    alias = {
        **spec,
        "block_size": 64,
        "a_tiles": {"BLOCK_SIZE_N": 128},
        "b_tiles": {"BLOCK_SIZE_K": 256, "GROUP_SIZE_M": 1},
    }
    assert tuner.execution_key(spec, 16) == tuner.execution_key(alias, 16)
    assert tuner.execution_key(spec, 64) != tuner.execution_key(alias, 64)
    assert spec["a_tiles"] == {"BLOCK_SIZE_N": 16}


@pytest.mark.parametrize("failure", ["fatal", "all_missing", "one_missing"])
def test_candidate_errors_preserve_ledger_and_reject_all_invalid(
    tuner, monkeypatch, failure
):
    import torch

    incumbent = dict(
        a_family="grouped",
        b_family="grouped",
        overlap="none",
        block_size=16,
        a_tiles={"BLOCK_SIZE_N": 64},
        b_tiles={"BLOCK_SIZE_K": 32},
    )
    specs = [{**incumbent, "a_tiles": {"BLOCK_SIZE_N": n}} for n in (32, 16)]
    plan = SimpleNamespace(
        DenseLoraKind=lambda x: x,
        DensePlan=lambda **kw: kw,
        DensePlanTable=lambda *a: SimpleNamespace(plan_for=lambda *a: incumbent),
        _PlanSpecModel=SimpleNamespace(
            model_validate=lambda value: SimpleNamespace(model_dump=lambda: value)
        ),
    )
    monkeypatch.setitem(sys.modules, "sglang.srt.lora.dense.plan", plan)
    monkeypatch.setitem(
        sys.modules, "sglang.srt.lora.utils", SimpleNamespace(Phase=lambda x: x)
    )
    monkeypatch.setattr(tuner, "plan_dict", lambda value: value)
    monkeypatch.setattr(
        tuner,
        "Workload",
        lambda *args: SimpleNamespace(check=lambda fn: 0, runner=lambda spec: None),
    )
    monkeypatch.setattr(
        tuner.random,
        "Random",
        lambda seed: SimpleNamespace(shuffle=lambda values: None),
    )
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: None)

    def paired(workload, plans, args):
        if (
            failure == "all_missing"
            or plans["candidate"]["a_tiles"]["BLOCK_SIZE_N"] == 16
        ):
            if failure != "fatal":
                raise KeyError("BLOCK_SIZE_N")
            raise RuntimeError("CUDA illegal memory access")
        return {"baseline": [100] * 3, "candidate": [100] * 3}, {}

    monkeypatch.setattr(tuner, "paired", paired)
    args = SimpleNamespace(
        candidates=SimpleNamespace(read_text=lambda: json.dumps(specs)),
        seed=0,
        min_gain=0.02,
        max_regression=0.02,
    )
    if failure == "one_missing":
        result = tuner.tune_case(tuner.Case.from_dict(case_data()), args, None, "sm100")
        assert result["validation"]["status"] == "RETAIN"
        records = result["search"]
    else:
        with pytest.raises((RuntimeError, ValueError)) as error:
            tuner.tune_case(tuner.Case.from_dict(case_data()), args, None, "sm100")
        records = error.value.tuning_partial["search"]
        assert (
            "illegal memory" in str(error.value)
            if failure == "fatal"
            else "no valid candidate" in str(error.value)
        )
    assert len(records) == 2
    if failure != "all_missing":
        assert records[0]["comparison"]["status"] == "TIE"
    else:
        assert all(row["status"] == "INVALID" for row in records)
    assert records[1]["status"] == "INVALID"
    if failure != "fatal":
        assert records[1]["error"] == "KeyError: 'BLOCK_SIZE_N'"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
