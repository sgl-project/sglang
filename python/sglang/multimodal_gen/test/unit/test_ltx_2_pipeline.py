from types import SimpleNamespace

import pytest

pytest.importorskip("pybase64")

from sglang.multimodal_gen.runtime.models.dits.ltx_2 import LTX2VideoTransformer3DModel
from sglang.multimodal_gen.runtime.pipelines.ltx_2_pipeline import (
    LTX2TwoStagePipeline,
)


def _make_server_args(ltx_variant: str):
    return SimpleNamespace(
        pipeline_config=SimpleNamespace(
            vae_config=SimpleNamespace(
                arch_config=SimpleNamespace(ltx_variant=ltx_variant)
            )
        )
    )


def test_ltx23_two_stage_merges_stage2_distilled_lora():
    assert (
        LTX2TwoStagePipeline._should_merge_stage2_distilled_lora(
            _make_server_args("ltx_2_3")
        )
        is True
    )


def test_ltx2_two_stage_keeps_stage2_distilled_lora_unmerged():
    assert (
        LTX2TwoStagePipeline._should_merge_stage2_distilled_lora(
            _make_server_args("ltx_2")
        )
        is False
    )


def test_ltx23_av_ca_gate_timestep_factor_matches_official_scaling():
    model = object.__new__(LTX2VideoTransformer3DModel)
    model.config = SimpleNamespace(arch_config=SimpleNamespace(ltx_variant="ltx_2_3"))
    model.av_ca_timestep_scale_multiplier = 1000
    model.timestep_scale_multiplier = 1000
    assert model._get_av_ca_gate_timestep_factor() == 1.0


def test_ltx2_av_ca_gate_timestep_factor_preserves_legacy_scaling():
    model = object.__new__(LTX2VideoTransformer3DModel)
    model.config = SimpleNamespace(arch_config=SimpleNamespace(ltx_variant="ltx_2"))
    model.av_ca_timestep_scale_multiplier = 1
    model.timestep_scale_multiplier = 1000
    assert model._get_av_ca_gate_timestep_factor() == 1.0
