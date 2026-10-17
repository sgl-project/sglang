"""Check shipped plan and tile selection and reject malformed overrides."""

from __future__ import annotations

import json

import pytest

from sglang.srt.environ import envs
from sglang.srt.lora.moe import plan as ep
from sglang.srt.lora.moe.plan import (
    ActFamily,
    ActivationFn,
    AFamily,
    BFamily,
    DownOverlap,
    FinalizeFamily,
    GateUpOverlap,
    MoeLoraLaunchConfig,
    RouteBuilderFamily,
    _load_plans,
    _tile_table,
    resolve_plans,
)
from sglang.srt.lora.utils import Phase, architecture_for_capability
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_SWIGLU = ActivationFn.SILU
_GB300 = "sm100"
_H200 = "sm90"


def _clear_caches():
    ep._load_plans.cache_clear()


@pytest.fixture(autouse=True)
def _fresh_caches():
    _clear_caches()
    yield
    _clear_caches()


def _resolve(
    architecture=_GB300,
    layout=False,
    rank=16,
    act=_SWIGLU,
    hidden=4096,
    experts=256,
):
    return resolve_plans(
        quant_family="bf16",
        architecture=architecture,
        is_shared_outer=layout,
        physical_rank=rank,
        activation=act,
        hidden_size=hidden,
        num_local_experts=experts,
    )


class TestSm100PerExpert:
    def test_decode_ships_per_row_b_grouped_a_wide_windows(self):
        c = _resolve(rank=32, experts=512)[Phase.DECODE]
        assert c.base_gemm_rows == "expert_major"
        assert c.plan.gate_up_a.family is AFamily.GROUPED
        assert c.plan.gate_up_b.family is BFamily.PER_ROW
        assert c.plan.down_a.family is AFamily.GROUPED
        assert c.plan.down_b.family is BFamily.PER_ROW
        assert c.plan.act.family is ActFamily.MATERIALIZED
        assert c.plan.finalize.family is FinalizeFamily.MATERIALIZED
        assert c.plan.gate_up_overlap is GateUpOverlap.GATE_UP_A_B
        assert c.plan.down_overlap is DownOverlap.DOWN_A_B
        assert c.plan.route_builder is RouteBuilderFamily.STANDARD

    def test_prefill_ships_serial_route_major_b_activation(self):
        c = _resolve()[Phase.PREFILL]
        assert c.base_gemm_rows == "route_major"
        assert c.plan.act.family is ActFamily.B_ACTIVATION
        assert c.plan.gate_up_b is None  # the b_activation kernel does this work
        assert not c.plan.down_b_into_base  # SM90 and default-table rows enable it
        assert c.plan.gate_up_overlap is GateUpOverlap.NONE
        assert c.plan.down_overlap is DownOverlap.NONE

    def test_decode_tile_ladder_is_rank_then_token_bucketed(self):
        # gate_up_b BLOCK_SIZE_N names the tile set the table chose. The rank
        # rules come first in the ladder, then the token rules.
        def block_n(rank: int, tokens: int) -> int:
            selected = _resolve(rank=rank)[Phase.DECODE]
            assert selected.name == "decode.per_expert"
            return selected.tiles.config_for(tokens).gate_up_b["BLOCK_SIZE_N"]

        assert block_n(16, 4) == 128
        assert block_n(16, 4096) == 128
        assert [block_n(32, m) for m in (4, 16, 32, 33, 128, 4096)] == [
            128,
            512,
            512,
            512,
            512,
            512,
        ]
        assert [block_n(64, m) for m in (4, 16, 17)] == [128, 512, 256]

    def test_rank_filter_without_a_surviving_rule_serves_the_default_launch_config(
        self,
    ):
        wide = {**MoeLoraLaunchConfig().gate_up_a, "BLOCK_SIZE_N": 64}
        rules = [ep._TileRuleModel(max_rank=8, sites={"gate_up_a": wide})]
        assert _tile_table(rules, physical_rank=8).config_for(4).gate_up_a == wide
        assert (
            _tile_table(rules, physical_rank=16).config_for(4) == MoeLoraLaunchConfig()
        )


class TestSm100Shared:
    def test_decode_ships_one_pass_with_the_down_a_window(self):
        # The one-pass finalize owns down-B, so only down-A is left to overlap.
        c = _resolve(layout=True, rank=32)[Phase.DECODE]
        assert c.base_gemm_rows == "expert_major"
        assert c.plan.gate_up_overlap is GateUpOverlap.GATE_UP_A_B
        assert c.plan.down_overlap is DownOverlap.DOWN_A
        assert c.plan.act.family is ActFamily.MATERIALIZED
        assert c.plan.finalize.family is FinalizeFamily.SHARED_ONE_PASS
        assert c.plan.gate_up_b.family is BFamily.GROUPED
        assert c.plan.down_b is None
        assert c.plan.route_builder is RouteBuilderFamily.PARALLEL_SHARED_OUTER

    def test_prefill_ships_token_grouped_serial(self):
        c = _resolve(layout=True, rank=32)[Phase.PREFILL]
        assert c.base_gemm_rows == "route_major"
        assert c.plan.gate_up_a.family is AFamily.TOKEN_GROUPED
        assert c.plan.act.family is ActFamily.B_ACTIVATION
        assert c.plan.route_builder is RouteBuilderFamily.PARALLEL_SHARED_OUTER
        assert c.plan.gate_up_overlap is GateUpOverlap.NONE
        assert c.plan.down_overlap is DownOverlap.NONE


class TestH200:
    def test_decode_ships_indexed_down_a_at_every_rank(self):
        for rank in (8, 32, 320):
            c = _resolve(architecture=_H200, rank=rank)[Phase.DECODE]
            assert c.plan.down_a.family is AFamily.PER_ROW
            assert c.base_gemm_rows == "expert_major"

    def test_shared_prefill_rank_band(self):
        # All rank bands use shared_token_delta; ranks above 16 use the unbanded row.
        for rank in (8, 16):
            small = _resolve(architecture=_H200, layout=True, rank=rank)[Phase.PREFILL]
            assert small.name == "prefill.shared.rank_le16"
            assert small.plan.finalize.family is FinalizeFamily.SHARED_TOKEN_DELTA
            assert small.plan.route_builder is RouteBuilderFamily.STANDARD
        for rank in (32, 64, 128):
            wide = _resolve(architecture=_H200, layout=True, rank=rank)[Phase.PREFILL]
            assert wide.name == "prefill.shared"
            assert wide.plan.finalize.family is FinalizeFamily.SHARED_TOKEN_DELTA
            assert wide.plan.down_b is None


class TestResolution:
    def test_activation_does_not_select_a_row_but_is_injected(self):
        for architecture in (_GB300, _H200):
            for layout in (False, True):
                swiglu = _resolve(architecture=architecture, layout=layout)
                relu2 = _resolve(
                    architecture=architecture, layout=layout, act=ActivationFn.RELU2
                )
                assert {p: sel.name for p, sel in relu2.items()} == {
                    p: sel.name for p, sel in swiglu.items()
                }
                for phase, sel in relu2.items():
                    assert sel.plan.act.activation is ActivationFn.RELU2
                    assert swiglu[phase].plan.act.activation is ActivationFn.SILU

    def test_out_of_domain_serves_the_tuned_table_fallback(self):
        # Out-of-domain geometry keeps the architecture's decode fallback;
        # DEFAULT remains serial.
        selected = _resolve(hidden=8192, experts=1024)
        decode = selected[Phase.DECODE]
        assert decode.name == "fallback.decode"
        assert decode.base_gemm_rows == "expert_major"
        assert decode.plan.gate_up_overlap is GateUpOverlap.GATE_UP_A_B
        assert decode.plan.down_overlap is DownOverlap.DOWN_A_B
        assert selected[Phase.PREFILL].name == "fallback.prefill.per_expert"

    def test_default_table_fallback_stays_serial(self):
        # Unknown architectures use the conservative no-overlap fallback.
        decode = _resolve(architecture="default")[Phase.DECODE]
        assert decode.plan.gate_up_overlap is GateUpOverlap.NONE
        assert decode.plan.down_overlap is DownOverlap.NONE

    def test_unknown_architecture_serves_the_default_table(self):
        assert architecture_for_capability(8) == "default"
        assert architecture_for_capability(9) == _H200
        assert architecture_for_capability(10) == _GB300
        table = _load_plans("default")
        assert table.scenarios == []
        selected = _resolve(architecture="default")
        assert selected[Phase.DECODE].name == "fallback.decode"
        assert selected[Phase.PREFILL].name == "fallback.prefill.per_expert"

    def test_every_resolvable_plan_is_a_declared_row(self):
        # The test reads the raw table. A shared helper repeats the filter,
        # so the test misses a filter bug.
        for architecture in (_GB300, _H200):
            for layout in (False, True):
                table = _load_plans(architecture)
                layout_name = "shared" if layout else "per_expert"
                declared = {
                    row.name
                    for row in (*table.scenarios, *table.fallback)
                    if row.layout in (None, layout_name)
                }
                for rank in (8, 16, 64, 320):
                    for hidden in (4096, 8192):
                        for sel in _resolve(
                            architecture=architecture,
                            layout=layout,
                            rank=rank,
                            hidden=hidden,
                            experts=512,
                        ).values():
                            assert sel.name in declared, (sel.name, declared)

    def test_row_names_are_unique_within_a_table(self):
        # Row names must identify report and override selections uniquely.
        for architecture in (_GB300, _H200, "default"):
            table = _load_plans(architecture)
            names = [row.name for row in (*table.scenarios, *table.fallback)]
            assert len(names) == len(set(names)), (architecture, names)

    def test_tile_rules_carry_only_the_shared_finalize_their_row_selects(self):
        # A row's shared finalize reads its own section. A section for the
        # other family would be dead weight; a missing one silently serves
        # the built-in default tile under a row that looks tuned.
        section = {
            FinalizeFamily.SHARED_TOKEN_DELTA: "shared_token_delta",
            FinalizeFamily.SHARED_ONE_PASS: "shared_one_pass",
        }
        for architecture in (_GB300, _H200, "default"):
            table = _load_plans(architecture)
            for row in (*table.scenarios, *table.fallback):
                family = row.plan.finalize_family
                expected = {section[family]} if family in section else set()
                for rule in row.tiles:
                    present = set(rule.sites) & set(section.values())
                    assert present == expected, (architecture, row.name, present)

    def test_unknown_domain_key_fails_closed(self, tmp_path):
        # Unknown domain bounds must fail closed rather than widen tuned geometry.
        packaged = json.load(open(f"{ep._CONFIG_DIR}/sm100.plans.json"))
        json.dump(packaged, open(tmp_path / "sm100.plans.json", "w"))
        with envs.SGLANG_LORA_MOE_CONFIG_DIR.override(str(tmp_path)):
            _clear_caches()
            beyond = _resolve(hidden=packaged["domain"]["max_hidden"] * 2)
            assert beyond[Phase.DECODE].name == "fallback.decode"

            _clear_caches()
            typo = json.loads(json.dumps(packaged))
            typo["domain"]["max_hidden_size"] = typo["domain"].pop("max_hidden")
            json.dump(typo, open(tmp_path / "sm100.plans.json", "w"))
            with pytest.raises(ValueError):
                _resolve()

    def test_override_dir_wins(self, tmp_path):
        packaged = json.load(open(f"{ep._CONFIG_DIR}/sm100.plans.json"))
        row = packaged["scenarios"][0]
        assert row["name"] == "decode.per_expert"
        row["tiles"][0]["sites"]["gate_up_a"]["num_warps"] = 8
        json.dump(packaged, open(tmp_path / "sm100.plans.json", "w"))

        def _tiny():
            return _resolve(rank=16)[Phase.DECODE].tiles.config_for(4)

        with envs.SGLANG_LORA_MOE_CONFIG_DIR.override(str(tmp_path)):
            _clear_caches()
            assert _tiny().gate_up_a["num_warps"] == 8
        _clear_caches()
        assert _tiny().gate_up_a["num_warps"] != 8

    def test_malformed_plan_family_fails_closed(self, tmp_path):
        packaged = json.load(open(f"{ep._CONFIG_DIR}/sm100.plans.json"))
        packaged["scenarios"][0]["plan"]["gate_up_b_family"] = "no_such_kernel"
        json.dump(packaged, open(tmp_path / "sm100.plans.json", "w"))
        with envs.SGLANG_LORA_MOE_CONFIG_DIR.override(str(tmp_path)):
            _clear_caches()
            with pytest.raises(ValueError):
                _resolve()

    def test_unknown_plan_field_fails_closed(self, tmp_path):
        # An old file can hold a retired "when" key. The loader must stop
        # with an error, because an ignored key changes which rows match.
        packaged = json.load(open(f"{ep._CONFIG_DIR}/sm100.plans.json"))
        packaged["scenarios"][0]["when"] = {"activation": "swiglu"}
        json.dump(packaged, open(tmp_path / "sm100.plans.json", "w"))
        with envs.SGLANG_LORA_MOE_CONFIG_DIR.override(str(tmp_path)):
            _clear_caches()
            with pytest.raises(ValueError):
                _resolve()

    def test_unknown_row_keys_are_rejected(self, tmp_path):
        # A table with a misspelled key is a config error, not a silent skip.
        packaged = json.load(open(f"{ep._CONFIG_DIR}/sm90.plans.json"))
        packaged["scenarios"][0]["provenence"] = "typo"
        json.dump(packaged, open(tmp_path / "sm90.plans.json", "w"))
        with envs.SGLANG_LORA_MOE_CONFIG_DIR.override(str(tmp_path)):
            _clear_caches()
            with pytest.raises(ValueError, match="provenence"):
                _resolve(architecture=_H200, layout=True, rank=64)

    def test_row_without_tiles_is_rejected(self, tmp_path):
        packaged = json.load(open(f"{ep._CONFIG_DIR}/sm100.plans.json"))
        del packaged["scenarios"][0]["tiles"]
        json.dump(packaged, open(tmp_path / "sm100.plans.json", "w"))
        with envs.SGLANG_LORA_MOE_CONFIG_DIR.override(str(tmp_path)):
            _clear_caches()
            with pytest.raises(ValueError, match="tiles"):
                _resolve()

    def test_unknown_tile_field_fails_closed(self, tmp_path):
        packaged = json.load(open(f"{ep._CONFIG_DIR}/sm100.plans.json"))
        packaged["scenarios"][0]["tiles"][0]["min_tokens"] = 1
        json.dump(packaged, open(tmp_path / "sm100.plans.json", "w"))
        with envs.SGLANG_LORA_MOE_CONFIG_DIR.override(str(tmp_path)):
            _clear_caches()
            with pytest.raises(ValueError):
                _resolve()

    def test_unknown_tile_site_key_fails_closed(self, tmp_path):
        # The rule-level extra="forbid" does not reach inside "sites". The
        # launch-config class must reject an unknown key on its own.
        packaged = json.load(open(f"{ep._CONFIG_DIR}/sm100.plans.json"))
        rule = packaged["scenarios"][0]["tiles"][0]
        rule["sites"]["gate_up_bee"] = rule["sites"].pop("gate_up_b")
        json.dump(packaged, open(tmp_path / "sm100.plans.json", "w"))
        with envs.SGLANG_LORA_MOE_CONFIG_DIR.override(str(tmp_path)):
            _clear_caches()
            with pytest.raises(ValueError, match="gate_up_bee"):
                _resolve()

    @pytest.mark.parametrize("excluded", ("layout", "rank", "shadowed", "domain"))
    def test_all_tile_rules_are_validated_before_selection(self, tmp_path, excluded):
        packaged = json.load(open(f"{ep._CONFIG_DIR}/sm100.plans.json"))
        row = packaged["scenarios"][0]
        rule = row["tiles"][0]
        if excluded == "layout":
            row["layout"] = "shared"
        elif excluded == "rank":
            rule["max_rank"] = 1
        elif excluded == "shadowed":
            row["tiles"].insert(0, {"sites": {}})
        else:
            packaged["domain"]["max_hidden"] = 1
        rule["sites"]["gate_up_a"]["BLOCK_SIZE_N"] = 3
        json.dump(packaged, open(tmp_path / "sm100.plans.json", "w"))
        with envs.SGLANG_LORA_MOE_CONFIG_DIR.override(str(tmp_path)):
            with pytest.raises(ValueError, match="gate_up_a.BLOCK_SIZE_N"):
                _resolve()

    @pytest.mark.parametrize(
        "excluded",
        ("layout", "quant", "rank", "shadowed", "fallback", "domain", "wildcard"),
    )
    def test_semantic_plan_errors_fail_before_selection(self, tmp_path, excluded):
        packaged = json.load(open(f"{ep._CONFIG_DIR}/sm100.plans.json"))
        row = packaged["scenarios"][0]
        request_layout = False
        if excluded == "layout":
            row["layout"] = "shared"
            row["plan"]["gate_up_a_family"] = "token_dense"
            row["plan"]["gate_up_b_family"] = "grouped"
        elif excluded == "quant":
            row["quant"] = ["nvfp4"]
            row["plan"]["act_family"] = "b_activation"
            row["plan"]["gate_up_overlap"] = "gate_up_a_b"
        elif excluded == "rank":
            row["max_rank"] = 1
            row["plan"]["finalize_family"] = "shared_one_pass"
        elif excluded == "wildcard":
            row = next(r for r in packaged["scenarios"] if r["name"] == "decode.shared")
            row["layout"] = None
            request_layout = True
        else:
            if excluded == "shadowed":
                row = json.loads(json.dumps(row))
                row["name"] += ".shadowed"
                packaged["scenarios"].insert(1, row)
            elif excluded == "fallback":
                row = packaged["fallback"][0]
            else:
                packaged["domain"]["max_hidden"] = 1
            row["plan"]["down_a_family"] = "token_grouped"
        json.dump(packaged, open(tmp_path / "sm100.plans.json", "w"))
        with envs.SGLANG_LORA_MOE_CONFIG_DIR.override(str(tmp_path)):
            with pytest.raises(ValueError):
                _resolve(layout=request_layout)

    @pytest.mark.parametrize("layout", ("per_expert", "shared", None))
    def test_load_validates_only_declared_layouts_without_binding_a_model(
        self, monkeypatch, layout
    ):
        seen = []
        build_plan = ep._build_plan

        def record(spec, **kwargs):
            seen.append(kwargs["is_shared_outer"])
            return build_plan(spec, **kwargs)

        monkeypatch.setattr(ep, "_build_plan", record)
        monkeypatch.setattr(
            ep,
            "_read_table",
            lambda _: {
                "fallback": [
                    {
                        "name": "generic",
                        "layout": layout,
                        "base_gemm_rows": "expert_major",
                        "plan": {},
                        "tiles": [{"sites": {}}],
                    }
                ],
            },
        )
        _load_plans("generic")
        ownerships = (False, True) if layout is None else (layout == "shared",)
        assert seen == list(ownerships)


@pytest.mark.parametrize(
    "site,key,value",
    (
        ("gate_up_a", "BLOCK_SIZE_NN", 32),
        ("gate_up_b", "SPLIT_K", 4),
        ("down_a", "num_stages", 0),
        ("down_b", "GROUP_SIZE_M", -1),
        ("b_activation", "BLOCK_SIZE_N", 64),
        ("gate_up_a", "BLOCK_SIZE_K", 3),
        ("gate_up_a", "SPLIT_K", 3),
        ("gate_up_b", "BLOCK_SIZE_N", 3),
        ("down_a", "SPLIT_K", 0),
        ("down_b", "num_warps", 3),
        ("b_activation", "BLOCK_SIZE_W", 3),
        ("shared_one_pass", "BLOCK_SIZE_H", 3),
    ),
)
def test_invalid_launch_config_is_rejected(site, key, value):
    config = {**getattr(MoeLoraLaunchConfig(), site), key: value}
    with pytest.raises(ValueError, match=key):
        MoeLoraLaunchConfig(**{site: config})


@pytest.mark.parametrize(
    "section,key,value",
    (
        ("reduce", "BLOCK_SIZE_H", 64),
        ("reduce", "BLOCK_SIZE_T", 3),
        ("tail", "num_stages", 0),
        ("tail", "num_warps", 3),
    ),
)
def test_invalid_shared_token_launch_config_is_rejected(section, key, value):
    config = {
        name: dict(values)
        for name, values in MoeLoraLaunchConfig().shared_token_delta.items()
    }
    config[section][key] = value
    with pytest.raises(ValueError, match=key):
        MoeLoraLaunchConfig(shared_token_delta=config)


@pytest.mark.parametrize(
    "site,tile_keys",
    (
        ("gate_up_a", ("BLOCK_SIZE_N", "BLOCK_SIZE_K")),
        ("gate_up_b", ("BLOCK_SIZE_N", "BLOCK_SIZE_K")),
        ("down_a", ("BLOCK_SIZE_N", "BLOCK_SIZE_K")),
        ("down_b", ("BLOCK_SIZE_N", "BLOCK_SIZE_K")),
        ("b_activation", ("BLOCK_SIZE_W", "BLOCK_SIZE_K")),
        ("shared_one_pass", ("BLOCK_SIZE_H",)),
    ),
)
def test_required_launch_keys_are_rejected_when_missing(site, tile_keys):
    for key in (*tile_keys, "num_warps", "num_stages"):
        config = dict(getattr(MoeLoraLaunchConfig(), site))
        del config[key]
        with pytest.raises(ValueError, match=key):
            MoeLoraLaunchConfig(**{site: config})


@pytest.mark.parametrize(
    "section,tile_key", (("reduce", "BLOCK_SIZE_T"), ("tail", "BLOCK_SIZE_H"))
)
def test_shared_token_launch_keys_are_required(section, tile_key):
    for key in (tile_key, "num_warps", "num_stages"):
        config = {
            name: dict(values)
            for name, values in MoeLoraLaunchConfig().shared_token_delta.items()
        }
        del config[section][key]
        with pytest.raises(ValueError, match=key):
            MoeLoraLaunchConfig(shared_token_delta=config)


@pytest.mark.parametrize(
    "defect", ("duplicate", "descending", "shadowed", "nonpositive")
)
def test_invalid_token_ladders_fail_before_selection(tmp_path, defect):
    packaged = json.load(open(f"{ep._CONFIG_DIR}/sm100.plans.json"))
    rules = packaged["scenarios"][0]["tiles"]
    if defect == "shadowed":
        rules.append({"sites": {}})
    else:
        rules[1]["max_tokens"] = {
            "duplicate": rules[0]["max_tokens"],
            "descending": 2,
            "nonpositive": 0,
        }[defect]
    json.dump(packaged, open(tmp_path / "sm100.plans.json", "w"))
    with envs.SGLANG_LORA_MOE_CONFIG_DIR.override(str(tmp_path)):
        with pytest.raises(ValueError, match="tile"):
            _resolve()


def test_non_dot_launch_sizes_remain_positive_not_dot_minimum():
    config = MoeLoraLaunchConfig(
        gate_up_a={**MoeLoraLaunchConfig().gate_up_a, "GROUP_SIZE_M": 3},
        shared_one_pass={"BLOCK_SIZE_H": 8, "num_warps": 1, "num_stages": 1},
    )
    assert config.gate_up_a["GROUP_SIZE_M"] == 3
    assert config.shared_one_pass["BLOCK_SIZE_H"] == 8


def test_per_row_launch_configs_keep_optional_group_and_row_tiles():
    base = MoeLoraLaunchConfig()
    configs = {}
    for site in ("gate_up_a", "gate_up_b", "down_a", "down_b"):
        configs[site] = {
            key: value
            for key, value in getattr(base, site).items()
            if key not in ("GROUP_SIZE_M", "BLOCK_SIZE_M")
        }
    assert MoeLoraLaunchConfig(**configs).gate_up_b == configs["gate_up_b"]


def _load_consumer_plan(monkeypatch, layout, plan, sites, container="fallback"):
    row = {
        "name": "context",
        "layout": layout,
        "max_rank": 1,
        "quant": ["nvfp4"],
        "base_gemm_rows": "expert_major",
        "plan": plan,
        "tiles": [{"sites": sites}],
    }
    raw = {"fallback": []}
    raw[container] = [row]
    monkeypatch.setattr(ep, "_read_table", lambda _: raw)
    _load_plans.cache_clear()
    return _load_plans("context")


@pytest.mark.parametrize(
    "layout,plan,site,tile_keys",
    (
        ("per_expert", {}, "gate_up_a", ("BLOCK_SIZE_N", "BLOCK_SIZE_K")),
        ("per_expert", {}, "down_a", ("BLOCK_SIZE_N", "BLOCK_SIZE_K")),
        ("per_expert", {}, "gate_up_b", ("BLOCK_SIZE_N",)),
        ("per_expert", {}, "down_b", ("BLOCK_SIZE_N",)),
        (
            "shared",
            {"gate_up_a_family": "token_grouped"},
            "gate_up_a",
            ("BLOCK_SIZE_N", "BLOCK_SIZE_K"),
        ),
        (
            "per_expert",
            {"act_family": "b_activation"},
            "b_activation",
            ("BLOCK_SIZE_W", "BLOCK_SIZE_K"),
        ),
        (
            "per_expert",
            {"down_b_family": "per_row", "down_b_into_base": True},
            "down_b",
            ("BLOCK_SIZE_N", "BLOCK_SIZE_K"),
        ),
        (
            "shared",
            {"finalize_family": "shared_token_delta"},
            "down_b",
            ("BLOCK_SIZE_N",),
        ),
    ),
)
@pytest.mark.parametrize("container", ("scenarios", "fallback"))
def test_dot_consumers_validate_group_size_and_tile_minimum(
    monkeypatch, layout, plan, site, tile_keys, container
):
    config = dict(getattr(MoeLoraLaunchConfig(), site))
    del config["GROUP_SIZE_M"]
    with pytest.raises(ValueError, match=f"{site} requires GROUP_SIZE_M"):
        _load_consumer_plan(monkeypatch, layout, plan, {site: config}, container)
    for key in tile_keys:
        for value in (8, 16):
            config = {**getattr(MoeLoraLaunchConfig(), site), key: value}
            if value < 16:
                with pytest.raises(ValueError, match=f"{site}.{key}"):
                    _load_consumer_plan(
                        monkeypatch, layout, plan, {site: config}, container
                    )
            else:
                _load_consumer_plan(
                    monkeypatch, layout, plan, {site: config}, container
                )


@pytest.mark.parametrize(
    "layout,plan,sites",
    (
        (
            "per_expert",
            {
                "gate_up_a_family": "per_row",
                "down_a_family": "per_row",
                "gate_up_b_family": "per_row",
                "down_b_family": "per_row",
            },
            ("gate_up_a", "gate_up_b", "down_a", "down_b", "b_activation"),
        ),
        (
            "shared",
            {"gate_up_a_family": "token_dense", "gate_up_b_family": "per_row"},
            ("gate_up_a", "gate_up_b"),
        ),
        ("per_expert", {"act_family": "b_activation"}, ("gate_up_b",)),
        ("shared", {"finalize_family": "shared_one_pass"}, ("down_b",)),
    ),
)
def test_vector_and_unused_tiles_keep_small_tiles_and_optional_group_size(
    monkeypatch, layout, plan, sites
):
    default = MoeLoraLaunchConfig()
    configs = {
        site: {
            key: 8 if key.startswith("BLOCK_SIZE_") and key != "BLOCK_SIZE_M" else value
            for key, value in getattr(default, site).items()
            if key != "GROUP_SIZE_M"
        }
        for site in sites
    }
    _load_consumer_plan(monkeypatch, layout, plan, configs)


@pytest.mark.parametrize(
    "layout,plan",
    (("per_expert", {}), ("shared", {"finalize_family": "shared_token_delta"})),
)
def test_grouped_b_accepts_k_clamped_by_the_launcher(monkeypatch, layout, plan):
    default = MoeLoraLaunchConfig()
    sites = {
        site: {**getattr(default, site), "BLOCK_SIZE_K": 8}
        for site in ("gate_up_b", "down_b")
    }
    _load_consumer_plan(monkeypatch, layout, plan, sites)


@pytest.mark.parametrize(
    "family,split_k",
    (
        ("grouped", 8),
        ("grouped", 4),
        ("token_grouped", 1),
        ("per_row", 8),
        ("token_dense", 8),
    ),
)
def test_contextual_split_k_preserves_supported_and_unused_settings(
    monkeypatch, family, split_k
):
    plan = {"gate_up_a_family": family}
    if family == "token_dense":
        plan["gate_up_b_family"] = "per_row"
    sites = {"gate_up_a": {**MoeLoraLaunchConfig().gate_up_a, "SPLIT_K": split_k}}
    _load_consumer_plan(monkeypatch, "shared", plan, sites)


def test_token_grouped_split_k_fails_before_selection(monkeypatch):
    sites = {"gate_up_a": {**MoeLoraLaunchConfig().gate_up_a, "SPLIT_K": 2}}
    with pytest.raises(
        ValueError, match="token_grouped does not provide split-K scratch"
    ):
        _load_consumer_plan(
            monkeypatch, "shared", {"gate_up_a_family": "token_grouped"}, sites
        )


class TestLoraMoeRunnerBackend:
    def test_lora_backend_predicates_are_exclusive(self):
        # LoRA vendors must not enter the plain vendor dispatch paths.
        from sglang.srt.layers.moe.utils import MoeRunnerBackend

        for backend in (
            MoeRunnerBackend.LORA_CUTEDSL,
            MoeRunnerBackend.LORA_TRITON,
            MoeRunnerBackend.LORA_MARLIN,
        ):
            assert backend.is_lora()
        assert not MoeRunnerBackend.LORA_TRITON.is_triton()
        assert not MoeRunnerBackend.LORA_MARLIN.is_marlin()
        assert MoeRunnerBackend.LORA_MARLIN.is_lora_marlin()
        assert not MoeRunnerBackend.MARLIN.is_lora_marlin()
        assert not MoeRunnerBackend.DEEP_GEMM.is_lora()
        assert not MoeRunnerBackend.TRITON.is_lora()

    def test_backend_values_are_valid_cli_choices(self):
        from sglang.srt.layers.moe.utils import MoeRunnerBackend
        from sglang.srt.server_args import MOE_RUNNER_BACKEND_CHOICES

        for backend in (
            MoeRunnerBackend.LORA_CUTEDSL,
            MoeRunnerBackend.LORA_TRITON,
            MoeRunnerBackend.LORA_MARLIN,
        ):
            assert backend.value in MOE_RUNNER_BACKEND_CHOICES


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
