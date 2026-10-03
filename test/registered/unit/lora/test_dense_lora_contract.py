from __future__ import annotations

import ast
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from sglang.srt.environ import envs
from sglang.srt.lora.dense import plan as plan_module
from sglang.srt.lora.dense.plan import (
    AFamily,
    BFamily,
    DenseLoraKind,
    DensePlan,
    DensePlanTable,
    Overlap,
)
from sglang.srt.lora.utils import Phase
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def test_plan_rejects_incompatible_families():
    with pytest.raises(ValueError, match="all_slots"):
        DensePlan(a_family=AFamily.ALL_SLOTS, b_family=BFamily.GROUPED)


def test_plan_rejects_unknown_split_mode():
    with pytest.raises(ValueError, match="SPLIT_MODE"):
        DensePlan(a_tiles={**DensePlan().a_tiles, "SPLIT_K": 4, "SPLIT_MODE": "atomic"})


@pytest.mark.parametrize("family", ["a_family", "b_family"])
def test_plan_spec_rejects_removed_sgemm_family(family):
    with pytest.raises(ValueError, match=family):
        plan_module._PlanSpecModel.model_validate({family: "sgemm"})


def test_defaults_without_a_table(tmp_path, monkeypatch):
    # An empty config directory exercises the in-code defaults.
    monkeypatch.setattr(plan_module, "_CONFIG_DIR", str(tmp_path))
    plan_module._load_plans.cache_clear()
    try:
        assert plan_module._load_plans("sm90") is None
        table = DensePlanTable("sm90", max_rank=64)
        decode = table.plan_for(DenseLoraKind.LINEAR, Phase.DECODE, 8)
        prefill = table.plan_for(DenseLoraKind.LINEAR, Phase.PREFILL, 2048)
        assert decode.a_tiles["SPLIT_K"] == 4
        assert decode.block_size == 16 and prefill.block_size == 64
        assert prefill.overlap is Overlap.NONE
        assert prefill.b_tiles["BLOCK_SIZE_N"] == 64
    finally:
        plan_module._load_plans.cache_clear()


def test_table_rows_match_in_order(tmp_path):
    rows = {
        "rows": [
            {
                "name": "tiny decode lm_head",
                "phase": "decode",
                "kinds": ["lm_head"],
                "max_tokens": 16,
                "spec": {
                    "a_family": "per_row",
                    "b_family": "per_row",
                    "overlap": "ab_delta",
                },
            },
            {
                "name": "small ranks decode",
                "phase": "decode",
                "max_tokens": 64,
                "max_rank": 32,
                "spec": {"a_family": "all_slots", "b_family": "per_row"},
            },
            {
                "name": "prefill",
                "phase": "prefill",
                "spec": {
                    "block_size": 32,
                    "a_tiles": {
                        "BLOCK_SIZE_N": 32,
                        "BLOCK_SIZE_K": 64,
                        "GROUP_SIZE_M": 4,
                        "num_warps": 4,
                        "num_stages": 2,
                    },
                },
            },
        ],
    }
    (tmp_path / "default.plans.json").write_text(json.dumps(rows))
    plan_module._load_plans.cache_clear()
    try:
        with envs.SGLANG_LORA_DENSE_CONFIG_DIR.override(str(tmp_path)):
            table = DensePlanTable("default", max_rank=64)
            assert (
                table.plan_for(DenseLoraKind.LM_HEAD, Phase.DECODE, 8).a_family
                is AFamily.PER_ROW
            )
            # rank 64 exceeds the second row's max_rank: in-code default decode plan
            fallback = table.plan_for(DenseLoraKind.LINEAR, Phase.DECODE, 8)
            assert fallback.a_family is AFamily.GROUPED
            assert fallback.a_tiles["SPLIT_K"] == 4
            prefill = table.plan_for(DenseLoraKind.LINEAR, Phase.PREFILL, 4096)
            assert prefill.block_size == 32 and prefill.a_tiles["BLOCK_SIZE_N"] == 32

            small = DensePlanTable("default", max_rank=16)
            assert (
                small.plan_for(DenseLoraKind.LINEAR, Phase.DECODE, 8).a_family
                is AFamily.ALL_SLOTS
            )
            assert (
                small.plan_for(DenseLoraKind.LINEAR, Phase.DECODE, 65).a_family
                is AFamily.GROUPED
            )
    finally:
        plan_module._load_plans.cache_clear()


def test_routed_rows_of_the_blackwell_table():
    # Only grouped shrink/expand reads routes; all_slots/per_row block values
    # do not affect the route set built for a batch.
    assert not DensePlan(
        a_family=AFamily.ALL_SLOTS, b_family=BFamily.PER_ROW
    ).needs_aligned_route
    assert not DensePlan(
        a_family=AFamily.PER_ROW, b_family=BFamily.PER_ROW
    ).needs_aligned_route
    plan_module._load_plans.cache_clear()
    try:
        table = plan_module._load_plans("sm100")
        routed = {}
        for row in table.rows:
            plan = DensePlan(**row.spec.model_dump())
            if plan.needs_aligned_route:
                routed.setdefault(row.phase, set()).add(plan.block_size)
        assert routed[Phase.DECODE] == {16}
        # the route pads per slot, not per request, so a captured plan's
        # capacity ignores the live request count at every block size
        assert routed[Phase.PREFILL] == {16, 32, 64, 128}
    finally:
        plan_module._load_plans.cache_clear()


def test_geometry_rows_do_not_leak_to_neighboring_sites(tmp_path):
    rows = {
        "rows": [
            {
                "name": "bounded geometry",
                "phase": "decode",
                "max_tokens": 8,
                "min_in_features": 256,
                "max_in_features": 512,
                "min_out_features": 128,
                "max_out_features": 256,
                "spec": {"block_size": 32},
            }
        ]
    }
    (tmp_path / "default.plans.json").write_text(json.dumps(rows))
    plan_module._load_plans.cache_clear()
    try:
        with envs.SGLANG_LORA_DENSE_CONFIG_DIR.override(str(tmp_path)):
            table = DensePlanTable("default", max_rank=64)
            for k, n, expected in (
                (256, 128, 32),
                (512, 256, 32),
                (255, 128, 16),
                (513, 128, 16),
                (256, 127, 16),
                (256, 257, 16),
            ):
                # Reusing one table also catches cache keys that omit K or N.
                assert (
                    table.plan_for(
                        DenseLoraKind.LINEAR, Phase.DECODE, 8, k, n
                    ).block_size
                    == expected
                )
    finally:
        plan_module._load_plans.cache_clear()


@pytest.mark.parametrize("architecture", ["default", "sm90", "sm100"])
@pytest.mark.parametrize("phase", [Phase.DECODE, Phase.PREFILL])
@pytest.mark.parametrize("kind", [DenseLoraKind.EMBEDDING, DenseLoraKind.LM_HEAD])
@pytest.mark.parametrize("rank", [8, 32, 64, 128])
def test_shipped_vocab_plans_keep_the_supported_execution_contract(
    monkeypatch, architecture, phase, kind, rank
):
    # This is a serving contract, independent of any table-generation script.
    monkeypatch.delenv("SGLANG_LORA_DENSE_CONFIG_DIR", raising=False)
    plan_module._load_plans.cache_clear()
    try:
        table = DensePlanTable(architecture, max_rank=rank)
        for tokens in (1, 16, 17, 1024):
            for k, n in ((256, 131), (2048, 65536)):
                plan = table.plan_for(kind, phase, tokens, k, n)
                assert plan.a_family is AFamily.GROUPED
                assert plan.b_family is BFamily.GROUPED
                assert plan.overlap is Overlap.NONE
                assert plan.block_size == (16 if phase is Phase.DECODE else 64)
                assert plan.a_tiles.get("SPLIT_K", 1) == (
                    4 if phase is Phase.DECODE else 1
                )
                assert plan == table.plan_for(kind, phase, tokens, k, 0)
                assert plan == table.plan_for(kind, phase, tokens, 0, n)
    finally:
        plan_module._load_plans.cache_clear()


def test_table_rejects_all_slots_rows_that_name_sink_down():
    row = {
        "name": "decode.sink_down",
        "kinds": ["sink_down"],
        "spec": {"a_family": "all_slots", "b_family": "per_row"},
    }
    with pytest.raises(ValueError, match="sink_down"):
        plan_module._PlanRowModel.model_validate(row)
    plan_module._PlanRowModel.model_validate({**row, "kinds": ["linear"]})
    plan_module._PlanRowModel.model_validate({**row, "spec": {"a_family": "grouped"}})


def test_windowed_sites_skip_all_slots_rows(tmp_path):
    """Windowed shrink skips all_slots rows; non-windowed kinds may select them."""
    rows = {
        "rows": [
            {
                "name": "decode.tokens_le16",
                "phase": "decode",
                "max_tokens": 16,
                "spec": {
                    "a_family": "all_slots",
                    "b_family": "per_row",
                    "overlap": "a",
                },
            },
            {
                "name": "decode",
                "phase": "decode",
                "spec": {"block_size": 32, "overlap": "ab_delta"},
            },
        ],
    }
    (tmp_path / "default.plans.json").write_text(json.dumps(rows))
    plan_module._load_plans.cache_clear()
    try:
        with envs.SGLANG_LORA_DENSE_CONFIG_DIR.override(str(tmp_path)):
            table = DensePlanTable("default", max_rank=64)
            linear = table.plan_for(DenseLoraKind.LINEAR, Phase.DECODE, 8)
            sink = table.plan_for(DenseLoraKind.SINK_DOWN, Phase.DECODE, 8)
            assert linear.a_family is AFamily.ALL_SLOTS and linear.overlap is Overlap.A
            assert sink.a_family is AFamily.GROUPED
            assert sink.block_size == 32 and sink.overlap is Overlap.AB_DELTA
            # beyond the all_slots row's token bound both kinds take the same row
            assert table.plan_for(DenseLoraKind.LINEAR, Phase.DECODE, 64) == sink
    finally:
        plan_module._load_plans.cache_clear()


def test_shipped_h200_table_serial_in_proj_exceptions():
    """H200 in_proj_qkvz (K=2048, N=12288), pool ranks 33-64, uses serial block 16
    through 2048 prefill tokens; other kinds and pool-16 selections are unchanged.
    """
    plan_module._load_plans.cache_clear()
    try:
        table = DensePlanTable("sm90", max_rank=64)
        short = table.plan_for(DenseLoraKind.LINEAR, Phase.PREFILL, 512, 2048, 12288)
        assert short.block_size == 16 and short.overlap is Overlap.NONE
        longer = table.plan_for(DenseLoraKind.LINEAR, Phase.PREFILL, 2048, 2048, 12288)
        assert longer.block_size == 16
        assert (
            table.plan_for(
                DenseLoraKind.LINEAR, Phase.PREFILL, 4096, 2048, 12288
            ).block_size
            == 64
        )
        assert (
            table.plan_for(
                DenseLoraKind.LINEAR, Phase.PREFILL, 512, 2048, 9216
            ).block_size
            == 64
        )
    finally:
        plan_module._load_plans.cache_clear()


ROOT = Path(__file__).resolve().parents[4] / "python/sglang/srt"


def _tree(path):
    return ast.parse((ROOT / path).read_text())


def _runner_constructor(cuda):
    tree = _tree("lora/dense/runner.py")
    runner = next(
        node for node in tree.body if getattr(node, "name", None) == "DenseLoraRunner"
    )
    init = next(
        node for node in runner.body if getattr(node, "name", None) == "__init__"
    )
    mapping = _tree("lora/utils.py")
    capability = next(
        node
        for node in mapping.body
        if getattr(node, "name", None) == "architecture_for_capability"
    )
    namespace = {"torch": SimpleNamespace(cuda=cuda)}
    module = ast.Module(
        body=[*ast.parse("from __future__ import annotations").body, capability, init],
        type_ignores=[],
    )
    exec(compile(module, "runner_constructor", "exec"), namespace)
    return namespace["__init__"]


@pytest.mark.parametrize("major,physical", [(8, "default"), (9, "sm90"), (10, "sm100")])
@pytest.mark.parametrize("override", [None, "default", "sm90", "sm100"])
def test_runner_resolves_architecture_from_the_device_unless_overridden(
    major, physical, override
):
    cuda = SimpleNamespace(
        get_device_name=Mock(return_value="device"),
        get_device_capability=Mock(return_value=(major, 0)),
    )
    init = _runner_constructor(cuda)
    device = SimpleNamespace(type="cuda", index=0)
    runner = SimpleNamespace(reset=Mock())
    init(runner, object(), max_loras=4, device=device, architecture=override)
    if override is None:
        cuda.get_device_capability.assert_called_once_with(device)
    else:
        cuda.get_device_capability.assert_not_called()
    assert runner.architecture == (physical if override is None else override)


@pytest.mark.parametrize("override", [None, "default", "sm90", "sm100"])
def test_cpu_runner_does_not_query_cuda_metadata(override):
    cuda = SimpleNamespace(get_device_name=Mock(), get_device_capability=Mock())
    init = _runner_constructor(cuda)
    runner = SimpleNamespace(reset=Mock())
    init(
        runner,
        object(),
        max_loras=4,
        device=SimpleNamespace(type="cpu"),
        architecture=override,
    )
    cuda.get_device_name.assert_not_called()
    cuda.get_device_capability.assert_not_called()
    assert runner.architecture == ("default" if override is None else override)


def test_explicit_experimental_mla_selection_precedes_v2():
    tree = _tree("models/deepseek_common/attention_forward_methods/forward_mla.py")
    # Explicit experimental selection still takes precedence over V2.
    experimental_fallbacks = {
        id(child)
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Name)
        and node.test.id == "_SGLANG_EXPERIMENTAL_LORA_OPTI"
        for branch in node.orelse
        for child in ast.walk(branch)
    }
    dense_imports = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module == "sglang.srt.lora.dense"
    ]
    assert len(dense_imports) == 2
    assert all(id(node) in experimental_fallbacks for node in dense_imports)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
