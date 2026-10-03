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
        assert not c.plan.down_b_into_base  # H200-only; see configs/README.md
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


class TestLoraMoeRunnerBackend:
    def test_lora_backend_predicates_are_exclusive(self):
        # LoRA vendors must not enter the plain vendor dispatch paths.
        from sglang.srt.layers.moe.utils import MoeRunnerBackend

        for backend in (MoeRunnerBackend.LORA_TRITON,):
            assert backend.is_lora()
        assert not MoeRunnerBackend.LORA_TRITON.is_triton()
        assert not MoeRunnerBackend.DEEP_GEMM.is_lora()
        assert not MoeRunnerBackend.TRITON.is_lora()

    def test_backend_values_are_valid_cli_choices(self):
        from sglang.srt.layers.moe.utils import MoeRunnerBackend
        from sglang.srt.server_args import MOE_RUNNER_BACKEND_CHOICES

        for backend in (MoeRunnerBackend.LORA_TRITON,):
            assert backend.value in MOE_RUNNER_BACKEND_CHOICES


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
