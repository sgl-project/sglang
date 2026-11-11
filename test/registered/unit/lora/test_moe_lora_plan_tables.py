"""Invalid plan-table rows fail at load, even rows the selector never picks;
rank and activation-scale selectors route every pool to a row that can serve
it, and adding a selector leaves the static selection unchanged."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sglang.kernels.ops.lora.moe.activation import ActivationFn
from sglang.srt.lora.moe import plan as ep
from sglang.srt.lora.moe.runner import _epilogue_down_b
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
        vendor="cutedsl",
    )


def test_vendor_rows_serve_only_their_vendor():
    """The Marlin decode row serves the FP4 families and the FP8 families the
    W8A16 provider repacks; other vendors skip it, as does a family Marlin
    never prepares tiles for."""

    def decode_row(quant_family, vendor):
        return ep.resolve_plans(
            quant_family=quant_family,
            vendor=vendor,
            architecture="sm100",
            is_shared_outer=True,
            physical_rank=16,
            activation=ActivationFn.SILU,
            hidden_size=4096,
            num_local_experts=256,
        )[ep.Phase.DECODE].name

    for family in ("nvfp4", "mxfp4", "fp8", "fp8_channel", "fp8_tensor", "mxfp8"):
        assert decode_row(family, "marlin") == "decode.shared.marlin", family
    assert decode_row("nvfp4", "cutedsl") == "decode.shared"
    assert decode_row("fp8", "cutedsl") == "decode.shared"
    assert decode_row("bf16", "marlin") == "decode.shared"


_FP8_MARLIN_FAMILIES = ("fp8", "fp8_channel", "fp8_tensor", "mxfp8")


@pytest.mark.parametrize("architecture", ("sm90", "sm100"))
@pytest.mark.parametrize("is_shared_outer", (False, True))
@pytest.mark.parametrize("rank", (8, 16, 32))
def test_fp8_marlin_families_select_the_fp4_marlin_rows(
    architecture, is_shared_outer, rank
):
    """The W8A16 FP8 families on the Marlin vendor resolve exactly the rows the
    W4A16 FP4 families resolve: the same raw pair-major provider serves both,
    so a row that admits one Marlin family admits them all, and no other row
    singles an FP8 family out on either architecture."""

    def rows(family):
        selected = ep.resolve_plans(
            quant_family=family,
            vendor="marlin",
            architecture=architecture,
            is_shared_outer=is_shared_outer,
            physical_rank=rank,
            activation=ActivationFn.SILU,
            hidden_size=4096,
            num_local_experts=256,
        )
        return {
            phase: (sel.name, sel.base_gemm_rows) for phase, sel in selected.items()
        }

    fp4 = rows("nvfp4")
    for family in _FP8_MARLIN_FAMILIES:
        assert rows(family) == fp4, (architecture, is_shared_outer, rank, family)


@pytest.mark.parametrize("architecture", ("sm90", "sm100"))
@pytest.mark.parametrize("family", _FP8_MARLIN_FAMILIES)
def test_fp8_families_default_to_w8a16_on_both_architectures(architecture, family):
    """Each FP8 family's default vendor is Marlin on SM90 and SM100: for both
    adapter layouts at ranks 8 and 16, the rows selected for both phases bind
    the W8A16 provider, and none asks for the in-GEMM down-B, which only the
    block-scaled CuTeDSL providers implement."""
    from sglang.srt.lora.moe.base_gemm_provider import select_provider_cls
    from sglang.srt.lora.moe.base_gemm_provider.marlin_fp8 import (
        MarlinFp8ContiguousProvider,
    )
    from sglang.srt.lora.moe.families import default_vendor

    vendor = default_vendor(family)
    assert vendor == "marlin"
    for is_shared_outer in (False, True):
        for rank in (8, 16):
            selected = ep.resolve_plans(
                quant_family=family,
                vendor=vendor,
                architecture=architecture,
                is_shared_outer=is_shared_outer,
                physical_rank=rank,
                activation=ActivationFn.SILU,
                hidden_size=4096,
                num_local_experts=256,
            )
            assert set(selected) == set(ep.Phase)
            for sel in selected.values():
                provider_cls = select_provider_cls(sel.base_gemm_rows, family, vendor)
                assert provider_cls is MarlinFp8ContiguousProvider
                assert not _epilogue_down_b(sel.plan)


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


@pytest.mark.parametrize("quant_family", ("nvfp4", "mxfp4", "mxfp8"))
@pytest.mark.parametrize(
    "rank,row,family",
    (
        (8, "prefill.per_expert.cutedsl_blockscaled", ep.BFamily.EPILOGUE),
        (9, "prefill.per_expert", ep.BFamily.GROUPED),
        (12, "prefill.per_expert", ep.BFamily.GROUPED),
        (15, "prefill.per_expert", ep.BFamily.GROUPED),
        (16, "prefill.per_expert.cutedsl_blockscaled", ep.BFamily.EPILOGUE),
        (17, "prefill.per_expert", ep.BFamily.GROUPED),
    ),
)
def test_epilogue_row_serves_only_rank_multiples_of_eight(
    quant_family, rank, row, family
):
    """The epilogue down-B loads the adapter's B rows as TMA tiles of eight,
    so its row takes ranks 8 and 16 of its 8..16 window and no other; the
    ranks between fall through to the separate down-B row instead of
    reaching the epilogue's initialization, and ranks past the window keep
    the separate row as before."""
    selected = ep.resolve_plans(
        quant_family=quant_family,
        architecture="sm100",
        is_shared_outer=False,
        physical_rank=rank,
        activation=ActivationFn.SILU,
        hidden_size=4096,
        num_local_experts=256,
        vendor="cutedsl",
    )[ep.Phase.PREFILL]
    assert selected.name == row
    assert selected.plan.down_b is not None
    assert selected.plan.down_b.family is family
    # The runner enables the epilogue on its provider only for this family.
    assert _epilogue_down_b(selected.plan) is (family is ep.BFamily.EPILOGUE)


def test_rank_multiple_must_be_positive(override_table):
    data, write = override_table
    row = next(
        row
        for row in data["scenarios"]
        if row["name"] == "prefill.per_expert.cutedsl_blockscaled"
    )
    row["rank_multiple"] = 0
    write()
    with pytest.raises(ValueError, match="rank_multiple"):
        _resolve()


def _comparable(selected):
    """Selections by content: a TileTable compares by identity."""
    return {
        phase: (sel.name, sel.base_gemm_rows, sel.plan, sel.tiles._rules)
        for phase, sel in selected.items()
    }


def _select(family, vendor, rank, act_scale, *, is_shared_outer=False):
    return ep.resolve_plans(
        quant_family=family,
        architecture="sm100",
        is_shared_outer=is_shared_outer,
        physical_rank=rank,
        activation=ActivationFn.SILU,
        hidden_size=4096,
        num_local_experts=256,
        vendor=vendor,
        act_scale=act_scale,
    )


@pytest.mark.parametrize("rank", (8, 16))
@pytest.mark.parametrize("quant_family", ("nvfp4", "mxfp4", "mxfp8"))
def test_epilogue_row_serves_both_activation_scale_modes(quant_family, rank):
    """The epilogue down-B row serves the static mode of every block-scaled
    family and, through the separate accumulator, per-token NVFP4 as well
    (the MX families only ever resolve static; the row's list is what the
    selector checks), at both physical ranks of its window; the whole
    selection, decode included, does not depend on the mode."""
    static = _select(quant_family, "cutedsl", rank, "static")
    per_token = _select(quant_family, "cutedsl", rank, "per_token")
    for selected in (static, per_token):
        prefill = selected[ep.Phase.PREFILL]
        assert prefill.name == "prefill.per_expert.cutedsl_blockscaled"
        assert prefill.plan.down_b.family is ep.BFamily.EPILOGUE
        assert selected[ep.Phase.DECODE].name == "decode.per_expert"
    assert _comparable(per_token) == _comparable(static)


@pytest.mark.parametrize(
    "rank, vendor, is_shared_outer, prefill",
    [
        (12, "cutedsl", False, "prefill.per_expert"),  # not a multiple of eight
        (32, "cutedsl", False, "prefill.per_expert"),  # past the row's rank window
        (16, "marlin", False, "prefill.per_expert"),  # another vendor
        (16, "cutedsl", True, "prefill.shared"),  # the shared-outer layout
    ],
)
def test_per_token_falls_back_like_static_outside_the_epilogue_row(
    rank, vendor, is_shared_outer, prefill
):
    """Outside the epilogue row's rank, vendor and layout restrictions the
    per-token mode resolves exactly what the static mode resolves: the
    separate grouped down-B row for per-expert pools, the shared rows for
    shared-outer pools, decode unchanged."""
    static = _select("nvfp4", vendor, rank, "static", is_shared_outer=is_shared_outer)
    per_token = _select(
        "nvfp4", vendor, rank, "per_token", is_shared_outer=is_shared_outer
    )
    assert per_token[ep.Phase.PREFILL].name == prefill
    if not is_shared_outer:
        assert per_token[ep.Phase.PREFILL].plan.down_b.family is ep.BFamily.GROUPED
    assert _comparable(per_token) == _comparable(static)


def test_act_scale_defaults_to_static():
    """A caller that names no mode gets the static selection."""
    kwargs = dict(
        quant_family="nvfp4",
        architecture="sm100",
        is_shared_outer=False,
        physical_rank=16,
        activation=ActivationFn.SILU,
        hidden_size=4096,
        num_local_experts=256,
        vendor="cutedsl",
    )
    assert _comparable(ep.resolve_plans(**kwargs)) == _comparable(
        ep.resolve_plans(**kwargs, act_scale="static")
    )


# The per-expert sm100 rows static block-scaled pools selected before the
# act_scale selector existed: (family, vendor, rank) -> (prefill, decode).
_STATIC_ROWS = {
    (family, "cutedsl", 8): (
        "prefill.per_expert.cutedsl_blockscaled",
        "decode.per_expert",
    )
    for family in ("nvfp4", "mxfp4", "mxfp8")
}
_STATIC_ROWS |= {
    (family, "cutedsl", 16): (
        "prefill.per_expert.cutedsl_blockscaled",
        "decode.per_expert",
    )
    for family in ("nvfp4", "mxfp4", "mxfp8")
}
_STATIC_ROWS |= {
    (family, "cutedsl", rank): ("prefill.per_expert", "decode.per_expert")
    for family in ("nvfp4", "mxfp4", "mxfp8")
    for rank in (12, 32)
}
_STATIC_ROWS |= {
    (family, "marlin", 16): ("prefill.per_expert", "decode.per_expert")
    for family in ("nvfp4", "mxfp4", *_FP8_MARLIN_FAMILIES)
}


@pytest.mark.parametrize("key", sorted(_STATIC_ROWS))
def test_static_block_scaled_rows_are_pinned(key):
    family, vendor, rank = key
    selected = _select(family, vendor, rank, "static")
    names = (selected[ep.Phase.PREFILL].name, selected[ep.Phase.DECODE].name)
    assert names == _STATIC_ROWS[key]


@pytest.mark.parametrize("is_shared_outer", (False, True))
@pytest.mark.parametrize("rank", (8, 12, 16, 32, 64))
@pytest.mark.parametrize(
    "family,vendor",
    (
        ("nvfp4", "cutedsl"),
        ("mxfp4", "cutedsl"),
        ("mxfp8", "cutedsl"),
        ("nvfp4", "marlin"),
        ("mxfp4", "marlin"),
        ("fp8", "marlin"),
        ("fp8_channel", "marlin"),
        ("fp8_tensor", "marlin"),
        ("mxfp8", "marlin"),
        ("bf16", "cutedsl"),
        ("fp8", "cutedsl"),
    ),
)
def test_static_selection_is_unchanged_by_the_act_scale_field(
    override_table, tmp_path, monkeypatch, family, vendor, rank, is_shared_outer
):
    """The packaged table with `act_scale` stripped from every row selects
    the same rows, plans and tiles for static modes: the field lists the
    modes the epilogue row serves (static and per-token) and never changes a
    static selection."""
    data, write = override_table
    monkeypatch.delenv("SGLANG_LORA_MOE_CONFIG_DIR")
    ep._load_plans.cache_clear()
    packaged = _select(family, vendor, rank, "static", is_shared_outer=is_shared_outer)
    for section in ("scenarios", "fallback"):
        for row in data[section]:
            row.pop("act_scale", None)
    monkeypatch.setenv("SGLANG_LORA_MOE_CONFIG_DIR", str(tmp_path))
    write()
    stripped = _select(family, vendor, rank, "static", is_shared_outer=is_shared_outer)
    assert _comparable(stripped) == _comparable(packaged)


def test_act_scale_must_name_a_mode(override_table):
    data, write = override_table
    row = next(
        row
        for row in data["scenarios"]
        if row["name"] == "prefill.per_expert.cutedsl_blockscaled"
    )
    row["act_scale"] = ["per_block"]
    write()
    with pytest.raises(ValueError, match="act_scale"):
        _resolve()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
