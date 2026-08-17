"""Plan and tile table resolution for the MoE LoRA MoE engine.

Pins the shipped best-config tables (2026-08 campaign, Appendix B of the
best-config document): per-row kernel families, fusion shape, overlap
windows, route builder, the H200 shared-prefill rank band, and the GB300
decode tile ladder — resolved from the packaged JSON files. Also pins the
resolution machinery itself: bind-time plan selection, the M-bucket tile
pick, the override directory, out-of-domain fallback, the unknown-
architecture default, and fail-closed behavior on malformed files.
"""

from __future__ import annotations

import json

import pytest

from sglang.srt.environ import envs
from sglang.srt.lora.moe import execution_plan as ep
from sglang.srt.lora.moe import launch_config as lc
from sglang.srt.lora.moe.execution_plan import (
    ActivationFamily,
    DeviceArchitecture,
    EarlyOverlap,
    FinalizeFamily,
    LateOverlap,
    LoraAFamily,
    LoraBFamily,
    MiddleFamily,
    Phase,
    RouteBuilderFamily,
    architecture_for_capability,
    load_plans,
    resolve_plans,
)
from sglang.srt.lora.moe.launch_config import MoeLoraLaunchConfig, resolve_tiles
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=5, stage="base-b", runner_config="1-gpu-small")

_SWIGLU = ActivationFamily.SWIGLU
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
        architecture=architecture,
        is_shared_outer=layout,
        physical_rank=rank,
        activation=act,
        hidden_size=hidden,
        num_local_experts=experts,
    )


class TestSm100PerExpert:
    def test_decode_ships_indexed_pairs_b_grouped_a_wide_windows(self):
        c = _resolve(rank=32, experts=512)[Phase.DECODE]
        assert c.provider == "cutedsl"
        assert c.plan.gate_up_a.family is LoraAFamily.GROUPED
        assert c.plan.gate_up_b.family is LoraBFamily.INDEXED_PAIRS
        assert c.plan.down_a.family is LoraAFamily.GROUPED
        assert c.plan.down_b.family is LoraBFamily.INDEXED_PAIRS
        assert c.plan.middle.family is MiddleFamily.MATERIALIZED
        assert c.plan.finalize.family is FinalizeFamily.MATERIALIZED
        assert c.plan.early_overlap is EarlyOverlap.GATE_UP_A_B
        assert c.plan.late_overlap is LateOverlap.DOWN_A_B
        assert c.plan.route_builder is RouteBuilderFamily.STANDARD

    def test_prefill_ships_serial_route_major_b_activation(self):
        c = _resolve()[Phase.PREFILL]
        assert c.provider == "cutedsl_contiguous"
        assert c.plan.middle.family is MiddleFamily.B_ACTIVATION
        assert c.plan.gate_up_b is None  # consumed by the b_activation middle
        assert c.plan.down_b_scatter
        assert c.plan.early_overlap is EarlyOverlap.NONE
        assert c.plan.late_overlap is LateOverlap.NONE

    def test_decode_tile_ladder_is_rank_then_token_bucketed(self):
        # rank <= 16 pins the tiny row for EVERY batch size (the ladder
        # terminates at the unconditional-M rank rule); larger ranks walk
        # the M ladder: <=4 tiny-shape, <=16 mse, then the large row.
        tiny = resolve_tiles(
            architecture_value="gb300",
            plan_key_name="decode.per_expert",
            physical_rank=16,
        )
        assert tiny.config_for(4).gate_up_b["BLOCK_SIZE_N"] == 128
        assert tiny.config_for(4096).gate_up_b["BLOCK_SIZE_N"] == 128
        ladder = resolve_tiles(
            architecture_value="gb300",
            plan_key_name="decode.per_expert",
            physical_rank=64,
        )
        assert ladder.config_for(4).gate_up_b["BLOCK_SIZE_N"] == 128
        assert ladder.config_for(16).gate_up_b["BLOCK_SIZE_N"] == 512
        assert ladder.config_for(17).gate_up_b["BLOCK_SIZE_N"] == 256

    def test_unknown_row_serves_the_default_launch_config(self):
        table = resolve_tiles(
            architecture_value="gb300",
            plan_key_name="no.such.row",
            physical_rank=16,
        )
        assert table.config_for(4) == MoeLoraLaunchConfig()


class TestSm100Shared:
    def test_decode_ships_wide_window_materialized_joint(self):
        c = _resolve(layout=True, rank=32)[Phase.DECODE]
        assert c.provider == "cutedsl"
        assert c.plan.early_overlap is EarlyOverlap.GATE_UP_A_B
        assert c.plan.late_overlap is LateOverlap.DOWN_A_B
        assert c.plan.middle.family is MiddleFamily.MATERIALIZED
        assert c.plan.finalize.family is FinalizeFamily.MATERIALIZED
        assert c.plan.gate_up_b.family is LoraBFamily.ONE_LAUNCH_SLICED
        assert c.plan.down_b.family is LoraBFamily.ONE_LAUNCH_SLICED
        assert c.plan.route_builder is RouteBuilderFamily.JOINT_SHARED_OUTER

    def test_prefill_ships_token_dedup_serial(self):
        c = _resolve(layout=True, rank=32)[Phase.PREFILL]
        assert c.provider == "cutedsl_contiguous"
        assert c.plan.gate_up_a.family is LoraAFamily.TOKEN_DEDUP_GROUPED
        assert c.plan.middle.family is MiddleFamily.B_ACTIVATION
        assert c.plan.route_builder is RouteBuilderFamily.JOINT_SHARED_OUTER
        assert c.plan.early_overlap is EarlyOverlap.NONE
        assert c.plan.late_overlap is LateOverlap.NONE


class TestH200:
    def test_decode_ships_indexed_down_a_at_every_rank(self):
        # Re-validated on real Qwen3.5-35B 2026-08-13: indexed down-A beats
        # the retired step10 pdl_down split at every bs (+13% dec at bs32).
        for rank in (8, 32, 320):
            c = _resolve(architecture=_H200, rank=rank)[Phase.DECODE]
            assert c.plan.down_a.family is LoraAFamily.INDEXED
            assert c.provider == "cutedsl"

    def test_shared_prefill_rank_band(self):
        # The one plan-level rank band, bound once at bind time: <=8 the
        # materialized small-rank twin, <=64 the shared-rank reduce (the
        # Inkling r128 replication set the ceiling), above it materialized.
        small = _resolve(architecture=_H200, layout=True, rank=8)[Phase.PREFILL]
        assert small.name == "prefill.materialized.small_rank"
        assert small.plan.finalize.family is FinalizeFamily.MATERIALIZED
        fused = _resolve(architecture=_H200, layout=True, rank=64)[Phase.PREFILL]
        assert fused.name == "prefill.shared_rank"
        assert fused.plan.finalize.family is FinalizeFamily.SHARED_RANK_REDUCE
        assert fused.plan.route_builder is RouteBuilderFamily.STANDARD
        wide = _resolve(architecture=_H200, layout=True, rank=128)[Phase.PREFILL]
        assert wide.name == "prefill.materialized"
        assert wide.plan.finalize.family is FinalizeFamily.MATERIALIZED


class TestResolution:
    def test_out_of_domain_serves_the_serial_deepgemm_fallback(self):
        selected = _resolve(hidden=8192, experts=1024)
        decode = selected[Phase.DECODE]
        assert decode.name == "fallback.serial"
        assert decode.provider == "deepgemm"
        assert decode.plan.early_overlap is EarlyOverlap.NONE
        assert decode.plan.late_overlap is LateOverlap.NONE
        assert selected[Phase.PREFILL].name == "fallback.serial_prefill"

    def test_unknown_architecture_serves_the_default_table(self):
        assert architecture_for_capability(8, 0) is DeviceArchitecture.DEFAULT
        assert architecture_for_capability(9, 0) is _H200
        assert architecture_for_capability(10, 3) is _GB300
        table = load_plans(DeviceArchitecture.DEFAULT)
        assert table.scenarios == []
        selected = _resolve(architecture=DeviceArchitecture.DEFAULT)
        assert selected[Phase.DECODE].name == "fallback.serial"
        assert selected[Phase.PREFILL].name == "fallback.serial_prefill"

    def test_every_resolvable_plan_is_a_declared_row(self):
        # resolve_plans may only ever return a row the table declares for
        # that layout — checked against the raw table, not against another
        # helper that shares its filtering code.
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

    def test_override_dir_wins(self, tmp_path):
        packaged = json.load(open(f"{ep._CONFIG_DIR}/gb300.tiles.json"))
        # flip one tile value; the override must be what resolves
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
        # A field this build does not understand (e.g. a retired "when"
        # predicate from an older file) must abort at load, not silently
        # widen or narrow the match.
        packaged = json.load(open(f"{ep._CONFIG_DIR}/gb300.plans.json"))
        packaged["scenarios"][0]["when"] = {"activation": "swiglu"}
        json.dump(packaged, open(tmp_path / "gb300.plans.json", "w"))
        with envs.SGLANG_LORA_MOE_CONFIG_DIR.override(str(tmp_path)):
            _clear_caches()
            with pytest.raises(ValueError):
                _resolve()

    def test_tuner_annotations_load_but_near_misses_do_not(self, tmp_path):
        # tune_lora_config.py --emit-seed stamps row "provenance" and a
        # file-level "seeded_for"; those must load (the whole onboarding
        # flow serves through SGLANG_LORA_MOE_CONFIG_DIR) while a typo of
        # either still fails closed.
        packaged = json.load(open(f"{ep._CONFIG_DIR}/h200.plans.json"))
        packaged["seeded_for"] = {"model": "acme/moe", "hidden": 6144}
        for row in packaged["scenarios"]:
            row["provenance"] = "campaign-2026-08"
        json.dump(packaged, open(tmp_path / "h200.plans.json", "w"))
        with envs.SGLANG_LORA_MOE_CONFIG_DIR.override(str(tmp_path)):
            _clear_caches()
            served = _resolve(architecture=_H200, layout=True, rank=64)
            assert served[Phase.PREFILL].name == "prefill.shared_rank"

            _clear_caches()
            typo = json.loads(json.dumps(packaged))
            typo["scenarios"][0]["provenence"] = "typo"
            json.dump(typo, open(tmp_path / "h200.plans.json", "w"))
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
        # A typo'd SITE key must abort bind, not silently serve the
        # built-in default tiles for that site (extra="forbid" on the
        # launch-config dataclass — rule-level extra="forbid" alone does
        # not reach inside "sites").
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
    """The MoE LoRA engine is selected by --moe-runner-backend lora."""

    def test_lora_backend_is_distinct_from_deep_gemm(self):
        # The backends stay separate values: the one place that must treat
        # them alike is the resident weight prep, which spells it out (see
        # FusedMoE.__init__ use_deep_gemm).
        from sglang.srt.layers.moe.utils import MoeRunnerBackend

        assert MoeRunnerBackend.LORA.is_lora()
        assert not MoeRunnerBackend.LORA.is_deep_gemm()
        assert not MoeRunnerBackend.DEEP_GEMM.is_lora()
        assert not MoeRunnerBackend.TRITON.is_lora()

    def test_lora_backend_gets_deep_gemm_resident_weight_prep(self):
        # Guards the single site the two backends share: without it the layer
        # would build triton-layout experts and the engine would refuse to
        # attach (or bind a provider to the wrong resident layout).
        import inspect

        from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE

        src = inspect.getsource(FusedMoE.__init__)
        assert "use_deep_gemm" in src and "is_lora()" in src

    def test_backend_value_is_a_valid_cli_choice(self):
        from sglang.srt.layers.moe.utils import MoeRunnerBackend
        from sglang.srt.server_args import MOE_RUNNER_BACKEND_CHOICES

        assert MoeRunnerBackend.LORA.value == "lora"
        assert "lora" in MOE_RUNNER_BACKEND_CHOICES


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
