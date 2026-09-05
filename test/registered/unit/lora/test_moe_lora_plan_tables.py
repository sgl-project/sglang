"""Check shipped plan/tile selection and reject malformed overrides."""

from __future__ import annotations

import json

import pytest

from sglang.srt.environ import envs
from sglang.srt.lora.moe import execution_plan as ep
from sglang.srt.lora.moe import launch_config as lc
from sglang.srt.lora.moe.execution_plan import (
    ActFamily,
    ActivationFn,
    DeviceArchitecture,
    DownOverlap,
    FinalizeFamily,
    GateUpOverlap,
    LoraAFamily,
    LoraBFamily,
    Phase,
    RouteBuilderFamily,
    architecture_for_capability,
    load_plans,
    resolve_plans,
)
from sglang.srt.lora.moe.launch_config import MoeLoraLaunchConfig, resolve_tiles
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=5, stage="base-b", runner_config="1-gpu-large")

_SWIGLU = ActivationFn.SILU
_GB300 = DeviceArchitecture.GB300
_H200 = DeviceArchitecture.H200


def _clear_caches():
    ep.load_plans.cache_clear()
    lc._load_tiles.cache_clear()


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
    def test_decode_ships_per_pair_b_grouped_a_wide_windows(self):
        c = _resolve(rank=32, experts=512)[Phase.DECODE]
        assert c.base_gemm_rows == "expert_major"
        assert c.plan.gate_up_a.family is LoraAFamily.GROUPED
        assert c.plan.gate_up_b.family is LoraBFamily.PER_PAIR
        assert c.plan.down_a.family is LoraAFamily.GROUPED
        assert c.plan.down_b.family is LoraBFamily.PER_PAIR
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
        assert c.plan.down_b_into_base
        assert c.plan.gate_up_overlap is GateUpOverlap.NONE
        assert c.plan.down_overlap is DownOverlap.NONE

    def test_decode_tile_ladder_is_rank_then_token_bucketed(self):
        # gate_up_b BLOCK_SIZE_N names the tile set the table chose. The rank
        # rules come first in the ladder, then the token rules.
        def block_n(rank: int, tokens: int) -> int:
            table = resolve_tiles(
                architecture_value="gb300",
                plan_key_name="decode.per_expert",
                physical_rank=rank,
            )
            return table.config_for(tokens).gate_up_b["BLOCK_SIZE_N"]

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

    def test_unknown_row_serves_the_default_launch_config(self):
        table = resolve_tiles(
            architecture_value="gb300",
            plan_key_name="no.such.row",
            physical_rank=16,
        )
        assert table.config_for(4) == MoeLoraLaunchConfig()


class TestSm100Shared:
    def test_decode_ships_one_pass_with_the_down_a_window(self):
        # The one-pass finalize owns down-B, so only down-A is left to overlap.
        c = _resolve(layout=True, rank=32)[Phase.DECODE]
        assert c.base_gemm_rows == "expert_major"
        assert c.plan.gate_up_overlap is GateUpOverlap.GATE_UP_A_B
        assert c.plan.down_overlap is DownOverlap.DOWN_A
        assert c.plan.act.family is ActFamily.MATERIALIZED
        assert c.plan.finalize.family is FinalizeFamily.SHARED_ONE_PASS
        assert c.plan.gate_up_b.family is LoraBFamily.GROUPED
        assert c.plan.down_b is None
        assert c.plan.route_builder is RouteBuilderFamily.PARALLEL_SHARED_OUTER

    def test_prefill_ships_token_grouped_serial(self):
        c = _resolve(layout=True, rank=32)[Phase.PREFILL]
        assert c.base_gemm_rows == "route_major"
        assert c.plan.gate_up_a.family is LoraAFamily.TOKEN_GROUPED
        assert c.plan.act.family is ActFamily.B_ACTIVATION
        assert c.plan.route_builder is RouteBuilderFamily.PARALLEL_SHARED_OUTER
        assert c.plan.gate_up_overlap is GateUpOverlap.NONE
        assert c.plan.down_overlap is DownOverlap.NONE


class TestH200:
    def test_decode_ships_indexed_down_a_at_every_rank(self):
        # On a real model, the indexed down-A row was faster than the retired
        # step10 pdl_down split.
        for rank in (8, 32, 320):
            c = _resolve(architecture=_H200, rank=rank)[Phase.DECODE]
            assert c.plan.down_a.family is LoraAFamily.PER_PAIR
            assert c.base_gemm_rows == "expert_major"

    def test_shared_prefill_rank_band(self):
        # The bands differ in their A kernels and overlap windows; every band
        # finalizes through shared_token_delta (measured at ranks 8 to 128).
        small = _resolve(architecture=_H200, layout=True, rank=8)[Phase.PREFILL]
        assert small.name == "prefill.shared.rank_le8"
        assert small.plan.finalize.family is FinalizeFamily.SHARED_TOKEN_DELTA
        fused = _resolve(architecture=_H200, layout=True, rank=64)[Phase.PREFILL]
        assert fused.name == "prefill.shared.rank_le64"
        assert fused.plan.finalize.family is FinalizeFamily.SHARED_TOKEN_DELTA
        assert fused.plan.route_builder is RouteBuilderFamily.STANDARD
        wide = _resolve(architecture=_H200, layout=True, rank=128)[Phase.PREFILL]
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
        # A geometry outside the domain still runs on tuned silicon, so the
        # table's own fallback keeps the decode overlap the sweep measured.
        # Only the untuned DEFAULT table stays serial.
        selected = _resolve(hidden=8192, experts=1024)
        decode = selected[Phase.DECODE]
        assert decode.name == "fallback.decode"
        assert decode.base_gemm_rows == "expert_major"
        assert decode.plan.gate_up_overlap is GateUpOverlap.GATE_UP_A_B
        assert decode.plan.down_overlap is DownOverlap.DOWN_A_B
        assert selected[Phase.PREFILL].name == "fallback.prefill.per_expert"

    def test_default_table_fallback_stays_serial(self):
        # Unknown silicon gets the conservative plan: no overlap window is
        # claimed on hardware nothing was ever measured on.
        decode = _resolve(architecture=DeviceArchitecture.DEFAULT)[Phase.DECODE]
        assert decode.plan.gate_up_overlap is GateUpOverlap.NONE
        assert decode.plan.down_overlap is DownOverlap.NONE

    def test_unknown_architecture_serves_the_default_table(self):
        assert architecture_for_capability(8, 0) is DeviceArchitecture.DEFAULT
        assert architecture_for_capability(9, 0) is _H200
        assert architecture_for_capability(10, 3) is _GB300
        table = load_plans(DeviceArchitecture.DEFAULT)
        assert table.scenarios == []
        selected = _resolve(architecture=DeviceArchitecture.DEFAULT)
        assert selected[Phase.DECODE].name == "fallback.decode"
        assert selected[Phase.PREFILL].name == "fallback.prefill.per_expert"

    def test_every_resolvable_plan_is_a_declared_row(self):
        # The test reads the raw table. A shared helper repeats the filter,
        # so the test misses a filter bug.
        for architecture in (_GB300, _H200):
            for layout in (False, True):
                table = load_plans(architecture)
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

    def test_row_names_state_their_own_selection_criteria(self):
        # A row named after its plan contents goes stale the moment a sweep
        # rewrites those contents, and nothing fails. "fallback.serial" kept
        # its name after the decode fallback gained both overlap windows, and
        # "prefill.serial" kept its after every prefill row became serial, so
        # the word stopped distinguishing anything. Naming a row for what
        # selects it - phase, layout, rank bound - cannot drift from the plan,
        # because the plan is not what the name describes.
        for architecture in (_GB300, _H200, DeviceArchitecture.DEFAULT):
            table = load_plans(architecture)
            for section, is_fallback in (
                (table.scenarios, False),
                (table.fallback, True),
            ):
                for row in section:
                    parts = ["fallback"] if is_fallback else []
                    parts.append(row.phase.value if row.phase else "any_phase")
                    if row.layout:
                        parts.append(row.layout)
                    if row.max_rank is not None:
                        parts.append(f"rank_le{row.max_rank}")
                    if row.quant is not None:
                        parts.append("_".join(row.quant))
                    if row.min_local_experts is not None:
                        parts.append(f"e_ge{row.min_local_experts}")
                    if row.max_local_experts is not None:
                        parts.append(f"e_le{row.max_local_experts}")
                    assert row.name == ".".join(parts), (
                        architecture.value,
                        row.name,
                    )

    def test_row_names_are_unique_within_a_table(self):
        # The name is the tiles lookup key (resolve_tiles(plan_key_name=...)),
        # so two rows sharing one name silently share one set of tuned tiles.
        # Two rows did: the per-expert/route_major and shared/expert_major
        # prefill fallbacks were both "fallback.serial_prefill".
        for architecture in (_GB300, _H200, DeviceArchitecture.DEFAULT):
            table = load_plans(architecture)
            names = [row.name for row in (*table.scenarios, *table.fallback)]
            assert len(names) == len(set(names)), (architecture.value, names)

    def test_tile_rule_keys_and_plan_rows_match(self):
        # resolve_tiles returns the built-in defaults when a key is missing,
        # so a renamed plan row loses its tuned tiles and raises no error, and
        # a new plan row without a rule runs the defaults under a name that
        # looks tuned. Only the speed drops, in silence, either way.
        for architecture in (_GB300, _H200, DeviceArchitecture.DEFAULT):
            plans = load_plans(architecture)
            tiles = lc._load_tiles(architecture.value)
            if tiles is None:
                continue
            declared = {row.name for row in (*plans.scenarios, *plans.fallback)}
            orphans = sorted(set(tiles.rules) - declared)
            assert not orphans, (architecture.value, orphans)
            missing = sorted(declared - set(tiles.rules))
            assert not missing, (architecture.value, missing)

    def test_tile_rules_carry_only_the_shared_finalize_their_row_selects(self):
        # A row's shared finalize reads its own section. A section for the
        # other family would be dead weight; a missing one silently serves
        # the built-in default tile under a row that looks tuned.
        section = {
            FinalizeFamily.SHARED_TOKEN_DELTA: "shared_token_delta",
            FinalizeFamily.SHARED_ONE_PASS: "shared_one_pass",
        }
        for architecture in DeviceArchitecture:
            tiles = lc._load_tiles(architecture.value)
            if tiles is None:
                continue
            table = load_plans(architecture)
            family = {
                row.name: row.plan.finalize_family
                for row in (*table.scenarios, *table.fallback)
            }
            for name, rules in tiles.rules.items():
                expected = {section[family[name]]} if family[name] in section else set()
                for rule in rules:
                    present = set(rule.sites) & set(section.values())
                    assert present == expected, (architecture.value, name, present)

    def test_unknown_domain_key_fails_closed(self, tmp_path):
        # The loader once read the domain with .get(key, 1 << 30). A typo in
        # a bound then let tuned rows serve a geometry nobody measured.
        packaged = json.load(open(f"{ep._CONFIG_DIR}/gb300.plans.json"))
        json.dump(packaged, open(tmp_path / "gb300.plans.json", "w"))
        with envs.SGLANG_LORA_MOE_CONFIG_DIR.override(str(tmp_path)):
            _clear_caches()
            beyond = _resolve(hidden=packaged["domain"]["max_hidden"] * 2)
            assert beyond[Phase.DECODE].name == "fallback.decode"

            _clear_caches()
            typo = json.loads(json.dumps(packaged))
            typo["domain"]["max_hidden_size"] = typo["domain"].pop("max_hidden")
            json.dump(typo, open(tmp_path / "gb300.plans.json", "w"))
            with pytest.raises(ValueError):
                _resolve()

    def test_override_dir_wins(self, tmp_path):
        packaged = json.load(open(f"{ep._CONFIG_DIR}/gb300.tiles.json"))
        packaged["rules"]["decode.per_expert"][0]["sites"]["gate_up_a"]["num_warps"] = 8
        json.dump(packaged, open(tmp_path / "gb300.tiles.json", "w"))

        def _tiny():
            return resolve_tiles(
                architecture_value="gb300",
                plan_key_name="decode.per_expert",
                physical_rank=16,
            ).config_for(4)

        with envs.SGLANG_LORA_MOE_CONFIG_DIR.override(str(tmp_path)):
            _clear_caches()
            assert _tiny().gate_up_a["num_warps"] == 8
        _clear_caches()
        assert _tiny().gate_up_a["num_warps"] != 8

    def test_malformed_plan_family_fails_closed(self, tmp_path):
        packaged = json.load(open(f"{ep._CONFIG_DIR}/gb300.plans.json"))
        packaged["scenarios"][0]["plan"]["gate_up_b_family"] = "no_such_kernel"
        json.dump(packaged, open(tmp_path / "gb300.plans.json", "w"))
        with envs.SGLANG_LORA_MOE_CONFIG_DIR.override(str(tmp_path)):
            _clear_caches()
            with pytest.raises(ValueError):
                _resolve()

    def test_unknown_plan_field_fails_closed(self, tmp_path):
        # An old file can hold a retired "when" key. The loader must stop
        # with an error, because an ignored key changes which rows match.
        packaged = json.load(open(f"{ep._CONFIG_DIR}/gb300.plans.json"))
        packaged["scenarios"][0]["when"] = {"activation": "swiglu"}
        json.dump(packaged, open(tmp_path / "gb300.plans.json", "w"))
        with envs.SGLANG_LORA_MOE_CONFIG_DIR.override(str(tmp_path)):
            _clear_caches()
            with pytest.raises(ValueError):
                _resolve()

    def test_unknown_row_keys_are_rejected(self, tmp_path):
        # A table with a misspelled key is a config error, not a silent skip.
        packaged = json.load(open(f"{ep._CONFIG_DIR}/h200.plans.json"))
        packaged["scenarios"][0]["provenence"] = "typo"
        json.dump(packaged, open(tmp_path / "h200.plans.json", "w"))
        with envs.SGLANG_LORA_MOE_CONFIG_DIR.override(str(tmp_path)):
            _clear_caches()
            with pytest.raises(ValueError, match="provenence"):
                _resolve(architecture=_H200, layout=True, rank=64)

    def test_unknown_tile_field_fails_closed(self, tmp_path):
        packaged = json.load(open(f"{ep._CONFIG_DIR}/gb300.tiles.json"))
        packaged["rules"]["decode.per_expert"][0]["min_tokens"] = 1
        json.dump(packaged, open(tmp_path / "gb300.tiles.json", "w"))
        with envs.SGLANG_LORA_MOE_CONFIG_DIR.override(str(tmp_path)):
            _clear_caches()
            with pytest.raises(ValueError):
                resolve_tiles(
                    architecture_value="gb300",
                    plan_key_name="decode.per_expert",
                    physical_rank=16,
                )

    def test_unknown_tile_site_key_fails_closed(self, tmp_path):
        # The rule-level extra="forbid" does not reach inside "sites". The
        # launch-config class must reject an unknown key on its own.
        packaged = json.load(open(f"{ep._CONFIG_DIR}/gb300.tiles.json"))
        rule = packaged["rules"]["decode.per_expert"][0]
        rule["sites"]["gate_up_bee"] = rule["sites"].pop("gate_up_b")
        json.dump(packaged, open(tmp_path / "gb300.tiles.json", "w"))
        with envs.SGLANG_LORA_MOE_CONFIG_DIR.override(str(tmp_path)):
            _clear_caches()
            with pytest.raises(ValueError, match="gate_up_bee"):
                resolve_tiles(
                    architecture_value="gb300",
                    plan_key_name="decode.per_expert",
                    physical_rank=16,
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

    def test_lora_backend_gets_deep_gemm_resident_weight_prep(self):
        # LoRA providers require resident [E, N, K] weights.
        import inspect

        from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE

        src = inspect.getsource(FusedMoE.__init__)
        assert "use_deep_gemm" in src and "is_lora()" in src

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
