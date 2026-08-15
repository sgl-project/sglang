"""JSON config resolution for the MoE LoRA MoE engine.

Pins the shipped best-config tables (2026-08 campaign, Appendix B of the
best-config document): per-scenario kernel families, fusion shape, overlap
windows, route builder, and the tier/fallback boundaries — resolved from the
packaged JSON files. Also pins the resolution machinery itself: the override
directory, out-of-domain fallback, unknown-architecture default, and
fail-closed behavior on malformed files.
"""

from __future__ import annotations

import json

import pytest

from sglang.srt.environ import envs
from sglang.srt.lora.moe import config as cm
from sglang.srt.lora.moe.execution_plan import (
    ActivationFamily,
    EarlyOverlap,
    FinalizeFamily,
    LateOverlap,
    LoraAFamily,
    LoraBFamily,
    MiddleFamily,
    RouteBuilderFamily,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=5, stage="base-b", runner_config="1-gpu-small")

_SWIGLU = ActivationFamily.SWIGLU


def _clear_caches():
    cm._config_table.cache_clear()
    cm._choices_for.cache_clear()


@pytest.fixture(autouse=True)
def _fresh_caches():
    _clear_caches()
    yield
    _clear_caches()


def _select(
    cap=10,
    layout=False,
    act=_SWIGLU,
    mode=cm.Phase.DECODE,
    tokens=16,
    rank=16,
    hidden=4096,
    experts=256,
):
    return cm.select_config(
        cm.ConfigInput(
            capability_major=cap,
            capability_minor=0,
            is_shared_outer=layout,
            activation=act,
            mode=mode,
            num_tokens=tokens,
            active_rank=rank,
            hidden_size=hidden,
            num_local_experts=experts,
            has_active_lora=True,
            use_cuda_graph=False,
        )
    )


class TestSm100PerExpert:
    def test_decode_ships_indexed_pairs_b_grouped_a_wide_windows(self):
        c = _select(tokens=32, rank=32, experts=512)
        assert c.provider == "cutedsl"
        assert c.plan.gate_a.family is LoraAFamily.GROUPED
        assert c.plan.gate_b.family is LoraBFamily.INDEXED_PAIRS
        assert c.plan.down_a.family is LoraAFamily.GROUPED
        assert c.plan.down_b.family is LoraBFamily.INDEXED_PAIRS
        assert c.plan.middle.family is MiddleFamily.MATERIALIZED
        assert c.plan.finalize.family is FinalizeFamily.MATERIALIZED
        assert c.plan.early_overlap is EarlyOverlap.GATE_A_B
        assert c.plan.late_overlap is LateOverlap.DOWN_A_B
        assert c.plan.route_builder is RouteBuilderFamily.STANDARD

    def test_decode_tier_boundaries(self):
        # tiny: tokens<=4 AND rank<=16
        assert "gab.tiny" in _select(tokens=4, rank=16).key
        assert "gab.tiny" not in _select(tokens=4, rank=32).key
        assert "gab.tiny" not in _select(tokens=5, rank=16).key
        # medium ceiling is geometry-aware: 32 iff local experts >= 512
        assert "gab.medium" in _select(tokens=32, rank=16, experts=512).key
        assert "gab.large" in _select(tokens=32, rank=16, experts=256).key
        assert "gab.medium" in _select(tokens=16, rank=16, experts=256).key
        assert "gab.large" in _select(tokens=33, rank=16, experts=512).key

    def test_prefill_ships_serial_route_major_b_activation(self):
        c = _select(mode=cm.Phase.PREFILL, tokens=4096)
        assert c.provider == "cutedsl_contiguous"
        assert c.plan.middle.family is MiddleFamily.B_ACTIVATION
        assert c.plan.gate_b is None  # consumed by the b_activation middle
        assert c.plan.down_b_scatter
        assert c.plan.early_overlap is EarlyOverlap.NONE
        assert c.plan.late_overlap is LateOverlap.NONE


class TestSm100Shared:
    def test_decode_ships_wide_window_materialized_joint_pdl(self):
        c = _select(layout=True, tokens=32, rank=32)
        assert c.provider == "cutedsl"
        assert c.plan.early_overlap is EarlyOverlap.GATE_A_B
        assert c.plan.late_overlap is LateOverlap.DOWN_A_B
        assert c.plan.middle.family is MiddleFamily.MATERIALIZED
        assert c.plan.finalize.family is FinalizeFamily.MATERIALIZED
        assert c.plan.gate_b.family is LoraBFamily.ONE_LAUNCH_SLICED
        assert c.plan.down_b.family is LoraBFamily.ONE_LAUNCH_SLICED
        assert c.plan.route_builder is RouteBuilderFamily.JOINT_SHARED_OUTER
        assert c.plan.route_pdl is True

    def test_prefill_ships_token_dedup_serial(self):
        c = _select(layout=True, mode=cm.Phase.PREFILL, tokens=4096, rank=32)
        assert c.provider == "cutedsl_contiguous"
        assert c.plan.gate_a.family is LoraAFamily.TOKEN_DEDUP_GROUPED
        assert c.plan.middle.family is MiddleFamily.B_ACTIVATION
        assert c.plan.route_builder is RouteBuilderFamily.JOINT_SHARED_OUTER
        assert c.plan.early_overlap is EarlyOverlap.NONE
        assert c.plan.late_overlap is LateOverlap.NONE


class TestH200:
    def test_decode_ships_indexed_down_a_at_every_batch_size(self):
        # Re-validated on real Qwen3.5-35B 2026-08-13: indexed down-A beats
        # the retired step10 pdl_down split at every bs (+13% dec at bs32).
        for tokens in (1, 16, 17, 64):
            c = _select(cap=9, tokens=tokens, rank=32)
            assert c.plan.down_a.family is LoraAFamily.INDEXED
            assert not c.plan.down_a_to_b_pdl
            assert c.provider == "cutedsl"

    def test_shared_prefill_rank_split(self):
        fused = _select(
            cap=9,
            layout=True,
            mode=cm.Phase.PREFILL,
            tokens=4096,
            rank=256,
        )
        assert fused.plan.finalize.family is FinalizeFamily.SHARED_RANK_REDUCE
        assert fused.plan.route_builder is RouteBuilderFamily.STANDARD
        wide = _select(
            cap=9,
            layout=True,
            mode=cm.Phase.PREFILL,
            tokens=4096,
            rank=320,
        )
        assert wide.plan.finalize.family is FinalizeFamily.MATERIALIZED


class TestResolution:
    def test_out_of_domain_serves_the_serial_deepgemm_fallback(self):
        c = _select(tokens=32, rank=16, hidden=8192, experts=1024)
        assert "fallback" in c.key
        assert c.provider == "deepgemm"
        assert c.plan.early_overlap is EarlyOverlap.NONE
        assert c.plan.late_overlap is LateOverlap.NONE

    def test_every_selectable_choice_is_bindable(self):
        # choices_for must cover everything select_config can return, for
        # every in-domain geometry class and both layouts.
        for cap in (10, 9):
            for layout in (False, True):
                arch = cm.architecture_for_capability(cap, 0)
                keys = {
                    c.key
                    for c in cm.choices_for(
                        arch,
                        layout,
                        _SWIGLU,
                        hidden_size=4096,
                        num_local_experts=512,
                    )
                }
                for tokens in (1, 4, 16, 32, 64):
                    for rank in (16, 64, 320):
                        got = _select(
                            cap=cap,
                            layout=layout,
                            tokens=tokens,
                            rank=rank,
                            experts=512,
                            mode=cm.Phase.DECODE,
                        )
                        assert got.key in keys
                        got = _select(
                            cap=cap,
                            layout=layout,
                            tokens=2048,
                            rank=rank,
                            experts=512,
                            mode=cm.Phase.PREFILL,
                        )
                        assert got.key in keys

    def test_override_dir_wins(self, tmp_path):
        packaged = json.load(open(f"{cm._CONFIG_DIR}/gb300.json"))
        # flip one tile value; the override must be what resolves
        packaged["scenarios"][0]["config"]["gate_a"]["num_warps"] = 8
        json.dump(packaged, open(tmp_path / "gb300.json", "w"))
        with envs.SGLANG_LORA_MOE_CONFIG_DIR.override(str(tmp_path)):
            _clear_caches()
            c = _select(tokens=4, rank=16)
            assert c.launch_config.gate_a["num_warps"] == 8
        _clear_caches()
        assert _select(tokens=4, rank=16).launch_config.gate_a["num_warps"] != 8

    def test_malformed_scenario_fails_closed(self, tmp_path):
        packaged = json.load(open(f"{cm._CONFIG_DIR}/gb300.json"))
        packaged["scenarios"][0]["plan"]["gate_b_family"] = "no_such_kernel"
        json.dump(packaged, open(tmp_path / "gb300.json", "w"))
        with envs.SGLANG_LORA_MOE_CONFIG_DIR.override(str(tmp_path)):
            _clear_caches()
            with pytest.raises(ValueError):
                _select(tokens=4, rank=16)

    def test_unknown_plan_field_fails_closed(self, tmp_path):
        packaged = json.load(open(f"{cm._CONFIG_DIR}/gb300.json"))
        packaged["scenarios"][0]["plan"]["not_a_field"] = True
        json.dump(packaged, open(tmp_path / "gb300.json", "w"))
        with envs.SGLANG_LORA_MOE_CONFIG_DIR.override(str(tmp_path)):
            _clear_caches()
            with pytest.raises(ValueError, match="unknown plan fields"):
                _select(tokens=4, rank=16)

    def test_unknown_when_key_fails_closed(self, tmp_path):
        # A predicate key this build does not understand (e.g. "quant" from a
        # newer config file) must abort at load, not silently widen the match.
        packaged = json.load(open(f"{cm._CONFIG_DIR}/gb300.json"))
        packaged["scenarios"][0]["when"]["quant"] = "fp8"
        json.dump(packaged, open(tmp_path / "gb300.json", "w"))
        with envs.SGLANG_LORA_MOE_CONFIG_DIR.override(str(tmp_path)):
            _clear_caches()
            with pytest.raises(ValueError, match="predicate keys"):
                _select(tokens=4, rank=16)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


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
