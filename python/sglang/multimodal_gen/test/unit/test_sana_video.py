# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch
from sglang.multimodal_gen.configs.models.dits.sana_video import (
    SanaVideoArchConfig,
    SanaVideoConfig,
)
from sglang.multimodal_gen.configs.pipeline_configs.sana_video import (
    SanaVideoOptimizedPipelineConfig,
    SanaVideoPipelineConfig,
)
from sglang.multimodal_gen.configs.sample.easycache import EasyCacheParams
from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams
from sglang.multimodal_gen.configs.sample.sana_video import SanaVideoSamplingParams
from sglang.multimodal_gen.runtime.cache.easycache import EasyCacheController
from sglang.multimodal_gen.runtime.loader.utils import get_param_names_mapping
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context


def _begin_cache_forward(batch, step_index, block_input):
    with set_forward_context(
        current_timestep=step_index,
        attn_metadata=None,
        forward_batch=batch,
    ):
        return EasyCacheController.begin_forward(block_input)


def test_sana_video_weight_mapping_packs_self_and_cross_attention():
    mapping = get_param_names_mapping(SanaVideoArchConfig().param_names_mapping)

    assert mapping("transformer_blocks.3.attn1.to_q.weight") == (
        "transformer_blocks.3.attn1.to_qkv.weight",
        0,
        3,
    )
    assert mapping("transformer_blocks.3.attn1.to_v.weight") == (
        "transformer_blocks.3.attn1.to_qkv.weight",
        2,
        3,
    )
    assert mapping("transformer_blocks.3.attn2.to_k.bias") == (
        "transformer_blocks.3.attn2.to_kv.bias",
        0,
        2,
    )
    assert mapping("transformer_blocks.3.attn2.to_q.weight") == (
        "transformer_blocks.3.attn2.to_q.weight",
        None,
        None,
    )
    assert mapping("transformer.transformer_blocks.3.attn1.to_k.weight") == (
        "transformer_blocks.3.attn1.to_qkv.weight",
        1,
        3,
    )


def test_sana_video_config_rejects_unknown_aggregation_precision():
    with pytest.raises(ValueError, match="linear_attention_aggregation_precision"):
        SanaVideoConfig(linear_attention_aggregation_precision="fp16")


def test_sana_video_optimized_profile_enables_sol_engine_paths():
    config = SanaVideoOptimizedPipelineConfig()

    assert config.dit_config.enable_easycache
    assert config.dit_config.linear_attention_aggregation_precision == "bf16"
    assert config.dit_config.torch_compile_mode == "default"


@pytest.mark.parametrize(
    ("kwargs", "exception"),
    [
        ({"threshold": float("nan")}, ValueError),
        ({"warmup_steps": -1}, ValueError),
        ({"subsample_stride": 0}, ValueError),
    ],
)
def test_easycache_params_validation(kwargs, exception):
    with pytest.raises(exception):
        EasyCacheParams(**kwargs)


def test_sana_video_sampling_params_accepts_easycache_dict():
    params = SanaVideoSamplingParams(
        easycache_params={
            "threshold": 0.2,
            "warmup_steps": 2,
            "subsample_stride": 4,
        }
    )

    assert isinstance(params.easycache_params, EasyCacheParams)
    assert params.easycache_params.threshold == 0.2
    assert params.easycache_params.warmup_steps == 2
    assert params.easycache_params.subsample_stride == 4


def test_sana_video_sampling_params_inherits_optimized_profile(monkeypatch):
    monkeypatch.setattr(SamplingParams, "_adjust", lambda _self, _args: None)
    params = SanaVideoSamplingParams()
    server_args = SimpleNamespace(
        pipeline_config=SanaVideoOptimizedPipelineConfig(),
        enable_cfg_parallel=False,
    )

    params._adjust(server_args)

    assert params.enable_easycache is True


def test_sana_video_sampling_params_rejects_invalid_easycache_deployment(monkeypatch):
    monkeypatch.setattr(SamplingParams, "_adjust", lambda _self, _args: None)
    baseline_args = SimpleNamespace(
        pipeline_config=SanaVideoPipelineConfig(),
        enable_cfg_parallel=False,
    )
    params = SanaVideoSamplingParams(enable_easycache=True)

    with pytest.raises(ValueError, match="must be enabled when the DiT is built"):
        params._adjust(baseline_args)

    cfg_parallel_args = SimpleNamespace(
        pipeline_config=SanaVideoOptimizedPipelineConfig(),
        enable_cfg_parallel=True,
    )
    params = SanaVideoSamplingParams(enable_easycache=True)

    with pytest.raises(ValueError, match="requires serial CFG"):
        params._adjust(cfg_parallel_args)


def test_easycache_serial_cfg_shares_the_latest_residual():
    batch = SimpleNamespace(
        enable_easycache=True,
        easycache_params=EasyCacheParams(
            threshold=100.0,
            warmup_steps=1,
            subsample_stride=1,
        ),
        extra={},
        is_cfg_negative=False,
    )

    step0_input = torch.zeros(1, 4, 2)
    cond0 = _begin_cache_forward(batch, 0, step0_input)
    assert cond0 is not None and cond0.should_compute
    EasyCacheController.after_compute(cond0, step0_input, step0_input + 1)

    batch.is_cfg_negative = True
    uncond0 = _begin_cache_forward(batch, 0, step0_input)
    assert uncond0 is not None and uncond0.should_compute
    EasyCacheController.after_compute(uncond0, step0_input, step0_input + 2)

    step1_input = torch.ones(1, 4, 2)
    batch.is_cfg_negative = False
    cond1 = _begin_cache_forward(batch, 1, step1_input)
    assert cond1 is not None and cond1.should_compute
    EasyCacheController.after_compute(cond1, step1_input, step1_input + 3)

    batch.is_cfg_negative = True
    uncond1 = _begin_cache_forward(batch, 1, step1_input)
    assert uncond1 is not None and uncond1.should_compute
    EasyCacheController.after_compute(uncond1, step1_input, step1_input + 4)

    step2_input = torch.full((1, 4, 2), 2.0)
    batch.is_cfg_negative = False
    cond2 = _begin_cache_forward(batch, 2, step2_input)
    assert cond2 is not None and not cond2.should_compute
    torch.testing.assert_close(
        EasyCacheController.reuse(cond2, step2_input),
        step2_input + 4,
    )

    batch.is_cfg_negative = True
    uncond2 = _begin_cache_forward(batch, 2, step2_input)
    assert uncond2 is not None and not uncond2.should_compute
    torch.testing.assert_close(
        EasyCacheController.reuse(uncond2, step2_input),
        step2_input + 4,
    )


def test_easycache_state_is_request_local():
    params = EasyCacheParams(threshold=100.0, warmup_steps=0)
    first = SimpleNamespace(
        enable_easycache=True,
        easycache_params=params,
        extra={},
        is_cfg_negative=False,
    )
    second = SimpleNamespace(
        enable_easycache=True,
        easycache_params=params,
        extra={},
        is_cfg_negative=False,
    )
    block_input = torch.zeros(1, 2, 2)

    first_decision = _begin_cache_forward(first, 0, block_input)
    second_decision = _begin_cache_forward(second, 0, block_input)

    assert first_decision is not None
    assert second_decision is not None
    assert first_decision.state is not second_decision.state
