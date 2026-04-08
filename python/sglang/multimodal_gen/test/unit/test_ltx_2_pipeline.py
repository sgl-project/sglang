from types import SimpleNamespace

import pytest

pytest.importorskip("pybase64")

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
