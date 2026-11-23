# SPDX-License-Identifier: Apache-2.0
"""Where the automatic memory policy puts the two DiTs of --model-variant
fl2va+ref2va: real ServerArgs normalization with simulated CUDA capacity, never
GPU execution."""

import pytest

from sglang.multimodal_gen.configs.pipeline_configs import minimax_h3 as h3_config
from sglang.multimodal_gen.configs.pipeline_configs.minimax_h3 import (
    FastH3PipelineConfig,
    MiniMaxH3PipelineConfig,
    PikaH3FlexPipelineConfig,
)
from sglang.multimodal_gen.runtime.managers.memory_managers.component_residency import (
    COMPONENT_OFFLOAD,
    LAYERWISE_OFFLOAD,
    RESIDENT,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.server_args import auto_tune as auto_tune_module
from sglang.multimodal_gen.runtime.server_args import server_args as args_module
from sglang.multimodal_gen.test.unit.test_minimax_h3_image_server_args import (
    SimulatedCUDA,
)

DITS = ("transformer", "transformer_2")
DUAL = "fl2va+ref2va"


@pytest.fixture
def make_server(monkeypatch, tmp_path):
    def make(memory_gib, *, config=None, model_variant=DUAL, **kwargs):
        platform = SimulatedCUDA(memory_gib)
        for module in (args_module, auto_tune_module, h3_config):
            monkeypatch.setattr(module, "current_platform", platform)
        return ServerArgs(
            model_path=str(tmp_path / "not-a-model-no-weights"),
            pipeline_config=MiniMaxH3PipelineConfig() if config is None else config,
            model_variant=model_variant,
            num_gpus=2,
            log_level="error",
            output_path=str(tmp_path / "outputs"),
            input_save_path=str(tmp_path / "inputs"),
            **kwargs,
        )

    return make


def placement(args):
    return [args.residency_mode(name) for name in DITS]


@pytest.mark.parametrize(
    "config_cls", [MiniMaxH3PipelineConfig, PikaH3FlexPipelineConfig]
)
def test_both_dits_stay_resident_when_both_fit(make_server, config_cls):
    # 120 GiB for one DiT with the rest of the pipeline, 61.73 GiB more for the second
    for memory_gib in (268, 181.8):
        args = make_server(memory_gib, config=config_cls())
        assert placement(args) == [RESIDENT, RESIDENT]
        assert args.residency_mode("text_encoder") == RESIDENT


@pytest.mark.parametrize("memory_gib", [181.7, 140, 120])
def test_one_dit_on_the_gpu_at_a_time_when_only_one_fits(make_server, memory_gib):
    args = make_server(memory_gib)
    # both are moved whole; the denoising stage keeps the one that ran last on
    # the GPU because the placement is automatic, not explicit
    assert placement(args) == [COMPONENT_OFFLOAD, COMPONENT_OFFLOAD]
    assert [args.explicit_residency_mode(name) for name in DITS] == [None, None]
    assert args.residency_mode("text_encoder") == RESIDENT


@pytest.mark.parametrize("memory_gib", [119.9, 80])
def test_startup_fails_when_no_dit_fits(make_server, memory_gib):
    with pytest.raises(ValueError, match="cannot keep even one of them") as error:
        make_server(memory_gib)
    message = str(error.value)
    for text in (
        f"--model-variant {DUAL}",
        "transformer, transformer_2",
        f"{memory_gib:.1f} GiB is free",
        "120.0 GiB to keep one DiT resident",
        "(181.7 GiB for all of them)",
        "--component-residency dit=layerwise-offload",
        "--dit-cpu-offload true",
    ):
        assert text in message


@pytest.mark.parametrize(
    "memory_gib,kwargs,expected",
    [
        (140, {"dit_cpu_offload": False}, [RESIDENT, RESIDENT]),
        (268, {"dit_cpu_offload": True}, [COMPONENT_OFFLOAD, COMPONENT_OFFLOAD]),
        (
            80,
            {"component_residency": "dit=layerwise-offload"},
            [LAYERWISE_OFFLOAD, LAYERWISE_OFFLOAD],
        ),
        (
            140,
            {"component_residency": "transformer_2=component-offload"},
            [RESIDENT, COMPONENT_OFFLOAD],
        ),
    ],
)
def test_explicit_placement_wins(make_server, memory_gib, kwargs, expected):
    assert placement(make_server(memory_gib, **kwargs)) == expected


@pytest.mark.parametrize("memory_gib", [80, 140, 268])
def test_single_partition_server_keeps_its_policy(make_server, memory_gib):
    args = make_server(memory_gib, model_variant="fl2va")
    resident = args.residency_mode("transformer") == RESIDENT
    assert resident == (memory_gib >= 120)


def test_other_modes_and_single_partition_releases_are_not_planned(make_server):
    # memory mode offloads by itself; FastH3 reports its own --model-variant error
    assert RESIDENT not in placement(make_server(80, performance_mode="memory"))
    with pytest.raises(ValueError, match="FastH3 ships one"):
        make_server(80, config=FastH3PipelineConfig())
