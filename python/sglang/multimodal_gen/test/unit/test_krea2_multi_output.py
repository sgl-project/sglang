from types import SimpleNamespace

import pytest
import torch

from sglang.multimodal_gen.configs.pipeline_configs.krea2 import Krea2PipelineConfig
from sglang.multimodal_gen.configs.sample.krea2 import Krea2SamplingParams
from sglang.multimodal_gen.runtime.models.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.krea2 import (
    Krea2BeforeDenoisingStage,
)


def _encode(prompts, device, dtype):
    rows = [
        torch.full((3, 12, 8), float(len(prompt)), device=device) for prompt in prompts
    ]
    mask = torch.ones(len(prompts), 3, dtype=torch.bool, device=device)
    return torch.stack(rows).to(dtype), mask


def _before_denoising(prompt, seed, num_outputs=1, dynamic_batch_seeds=None):
    scheduler = FlowMatchEulerDiscreteScheduler(
        use_dynamic_shifting=True, time_shift_type="exponential"
    )
    stage = Krea2BeforeDenoisingStage(None, None, None, None, None, scheduler)
    stage._encode = _encode
    batch = Req(
        sampling_params=Krea2SamplingParams(
            prompt=prompt,
            height=64,
            width=64,
            seed=seed,
            num_inference_steps=2,
            num_outputs_per_prompt=num_outputs,
        )
    )
    if dynamic_batch_seeds is not None:
        batch.extra["dynamic_batch_seeds"] = dynamic_batch_seeds
    server_args = SimpleNamespace(pipeline_config=Krea2PipelineConfig())
    return stage.forward(batch, server_args)


@pytest.mark.parametrize(
    "prompt, seed, dynamic_batch_seeds, row_prompts, row_seeds",
    [
        ("a red fox", 42, None, ["a red fox"] * 2, [42, 43]),
        ("a red fox", [42, 900], None, ["a red fox"] * 2, [42, 900]),
        (
            ["a red fox", "a small blue bird"],
            42,
            [42, 900],
            ["a red fox"] * 2 + ["a small blue bird"] * 2,
            [42, 43, 900, 901],
        ),
    ],
)
def test_two_outputs_per_prompt_prepare_one_row_per_output(
    prompt, seed, dynamic_batch_seeds, row_prompts, row_seeds
):
    batch = _before_denoising(prompt, seed, 2, dynamic_batch_seeds)

    assert batch.latents.shape[0] == batch.batch_size == len(row_seeds)
    expected_embeds, expected_mask = _encode(row_prompts, "cpu", torch.bfloat16)
    for embeds, expected in (
        (batch.prompt_embeds, expected_embeds),
        (batch.prompt_embeds_mask, expected_mask),
    ):
        torch.testing.assert_close(embeds[0].cpu(), expected, atol=0, rtol=0)
    for neg in (batch.negative_prompt_embeds, batch.negative_prompt_embeds_mask):
        assert neg[0].shape[0] == len(row_seeds)
    assert [g.initial_seed() for g in batch.generator] == row_seeds
    for row, row_seed in enumerate(row_seeds):
        single = _before_denoising("a red fox", row_seed)
        torch.testing.assert_close(
            batch.latents[row : row + 1], single.latents, atol=0, rtol=0
        )
