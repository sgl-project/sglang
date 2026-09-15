# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sglang.multimodal_gen.configs.pipeline_configs.llada_image import (
    LLaDAImagePipelineConfig,
)
from sglang.multimodal_gen.runtime.loader.utils import get_param_names_mapping
from sglang.multimodal_gen.runtime.models.dits.llada_image import (
    _LLaDAImageTransformer2DModel,
)
from sglang.multimodal_gen.runtime.models.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.llada_image.conditioning import (
    format_llada_image_prompt,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.timestep_preparation import (
    TimestepPreparationStage,
)
from sglang.multimodal_gen.runtime.server_args import (
    get_global_server_args,
    set_global_server_args,
)


@pytest.fixture(autouse=True)
def server_args():
    try:
        previous = get_global_server_args()
    except ValueError:
        previous = None
    set_global_server_args(SimpleNamespace(kv_gather_degree=1, sp_split_auto=False))
    yield
    set_global_server_args(previous)


@pytest.mark.parametrize(
    "source,target,shard,total",
    [
        ("attention.to_k", "attention.to_qkv", 1, 3),
        ("feed_forward.w1", "feed_forward.w13", 0, 2),
        ("feed_forward.w3", "feed_forward.w13", 1, 2),
    ],
)
def test_dit_weight_mapping(source, target, shard, total):
    mapping = get_param_names_mapping(
        LLaDAImagePipelineConfig().dit_config.arch_config.param_names_mapping
    )
    assert mapping(f"layers.0.{source}.weight") == (
        f"layers.0.{target}.weight",
        shard,
        total,
    )


@pytest.mark.parametrize(
    "prompt,instruction",
    [("a red car", "Generate an image: a red car"), (None, "Generate an image.")],
)
def test_prompt_format(prompt, instruction):
    assert (
        format_llada_image_prompt(prompt)
        == f"<role>HUMAN</role> {instruction}\n<role>ASSISTANT</role>\n<IMAGE1>"
    )


def test_uniform_scheduler_schedule():
    scheduler = FlowMatchEulerDiscreteScheduler(shift=3.0, use_uniform_sigmas=True)
    batch = SimpleNamespace(
        scheduler=None,
        timesteps=None,
        sigmas=None,
        num_inference_steps=4,
        n_tokens=None,
        extra={},
        is_warmup=True,
        rollout=False,
    )
    module = "sglang.multimodal_gen.runtime.pipelines_core.stages.timestep_preparation"
    with (
        patch(f"{module}.get_local_torch_device", return_value=torch.device("cpu")),
        patch(f"{module}.get_or_create_request_scheduler", return_value=scheduler),
    ):
        TimestepPreparationStage(scheduler).forward(
            batch, SimpleNamespace(pipeline_config=LLaDAImagePipelineConfig())
        )
    assert batch.sigmas is None
    torch.testing.assert_close(
        scheduler.sigmas, torch.tensor([1.0, 0.9, 0.75, 0.5, 0.0])
    )


@pytest.mark.parametrize("seeds", [(11,), (11, 29)])
def test_stochastic_scheduler_per_sample_generators(seeds):
    scheduler = FlowMatchEulerDiscreteScheduler(
        shift=3.0, use_uniform_sigmas=True, stochastic_sampling=True
    )
    scheduler.set_timesteps(4, device="cpu")
    sample = torch.arange(len(seeds) * 8, dtype=torch.float32).reshape(
        len(seeds), 2, 2, 2
    )
    prediction = sample / 16
    generators = [torch.Generator().manual_seed(seed) for seed in seeds]
    noise = torch.cat(
        [
            torch.randn((1, 2, 2, 2), generator=torch.Generator().manual_seed(seed))
            for seed in seeds
        ]
    )
    sigma, next_sigma = scheduler.sigmas[:2]
    expected = (1 - next_sigma) * (sample - sigma * prediction) + next_sigma * noise
    actual = scheduler.step(
        prediction,
        scheduler.timesteps[0],
        sample,
        generator=generators,
        return_dict=False,
    )[0]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_edit_skips_empty_semantic_refiner():
    class CaptureBlock(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, hidden_states, *args, **kwargs):
            self.calls += 1
            return hidden_states

    model = _LLaDAImageTransformer2DModel(
        in_channels=4,
        dim=64,
        n_layers=0,
        n_refiner_layers=0,
        n_heads=2,
        cap_feat_dim=8,
        semantic_feat_dim=10,
        axes_dims=(8, 12, 12),
        axes_lens=(256, 32, 32),
    )
    blocks = [CaptureBlock() for _ in range(4)]
    for name, block in zip(
        ("noise_refiner", "context_refiner", "sigvq_refiner", "layers"),
        blocks,
        strict=True,
    ):
        setattr(model, name, torch.nn.ModuleList([block]))
    with torch.no_grad():
        model(
            x=[torch.randn(4, 1, 4, 4)],
            t=torch.tensor([0.5]),
            cap_feats=[torch.randn(3, 8)],
            glm_cap_feats=[torch.empty(0, 10)],
            source_latents=[torch.randn(4, 1, 4, 4)],
        )
    assert [block.calls for block in blocks] == [1, 1, 0, 1]
