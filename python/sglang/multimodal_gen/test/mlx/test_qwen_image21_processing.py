# SPDX-License-Identifier: Apache-2.0

import math
import sys

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
torch = pytest.importorskip("torch")
hf = pytest.importorskip("transformers.models.qwen3_vl.modeling_qwen3_vl")
diffusers = pytest.importorskip("diffusers")

from sglang.multimodal_gen.runtime.hardware_backend.mlx.qwen_image21 import (
    TimeEmbedding,
)
from sglang.multimodal_gen.runtime.hardware_backend.mlx.qwen_image21_processing import (
    decode_latents,
    flow_schedule,
    flow_step,
    image_position_ids,
)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_timestep_rounding_matches_torch(dtype):
    torch.manual_seed(31)
    first = torch.nn.Linear(256, 64, bias=False).to(dtype)
    second = torch.nn.Linear(64, 64, bias=False).to(dtype)
    model = TimeEmbedding(64)
    mlx_dtype = mx.float32 if dtype == torch.float32 else mx.bfloat16
    model.load_weights(
        [
            (
                "timestep_embedder.linear_1.weight",
                mx.array(first.weight.detach().float().numpy()).astype(mlx_dtype),
            ),
            (
                "timestep_embedder.linear_2.weight",
                mx.array(second.weight.detach().float().numpy()).astype(mlx_dtype),
            ),
        ]
    )
    timestep = torch.tensor([0, 7.231, 234.234, 789.13, 989.3, 1000])
    time = (timestep.to(dtype) / 1000).float()
    freq = torch.exp(-math.log(10000) * torch.arange(128).float() / 128)
    angles = time[:, None] * 1000 * freq
    with torch.no_grad():
        expected = second(
            torch.nn.functional.silu(
                first(torch.cat((angles.cos(), angles.sin()), -1).to(dtype))
            )
        )
    actual = model(mx.array(timestep.numpy()), mlx_dtype)
    np.testing.assert_allclose(
        np.array(actual.astype(mx.float32)),
        expected.float().numpy(),
        # Metal and CPU exp differ by one FP32 ulp, amplified by t <= 1000
        atol=5e-6 if dtype == torch.float32 else 0.002,
        rtol=2e-4 if dtype == torch.float32 else 0.01,
    )


@pytest.mark.parametrize("steps", [1, 4, 40])
@pytest.mark.parametrize("tokens", [256, 4096, 8192])
def test_flow_schedule_and_step_match_reference(steps, tokens):
    config = dict(
        num_train_timesteps=1000,
        shift=1,
        base_shift=0.5,
        max_shift=0.9,
        base_image_seq_len=256,
        max_image_seq_len=8192,
        use_dynamic_shifting=True,
        shift_terminal=0.02,
        time_shift_type="exponential",
    )
    reference = diffusers.FlowMatchEulerDiscreteScheduler(**config)
    mu = 0.5 + (tokens - 256) * 0.4 / (8192 - 256)
    # upstream diffusers has no single-step terminal-stretch guard
    if steps == 1:
        reference.register_to_config(shift_terminal=None)
    reference.set_timesteps(sigmas=np.linspace(1, 1 / steps, steps), mu=mu)
    sigmas, timesteps = flow_schedule(config, steps, tokens)
    np.testing.assert_allclose(
        np.array(sigmas), reference.sigmas.numpy(), atol=1e-7, rtol=1e-6
    )
    np.testing.assert_allclose(
        np.array(timesteps), reference.timesteps.numpy(), atol=1e-4, rtol=1e-6
    )
    torch.manual_seed(29)
    latent = torch.randn(1, 13, 64).bfloat16()
    noise = torch.randn_like(latent)
    for index, timestep in enumerate(reference.timesteps):
        # isolate arithmetic from the exp implementation's FP32 ulp differences
        delta = reference.sigmas[index + 1] - reference.sigmas[index]
        actual = flow_step(
            mx.array(latent.float().numpy()).astype(mx.bfloat16),
            mx.array(noise.float().numpy()).astype(mx.bfloat16),
            mx.array(delta.numpy()),
        )
        latent = reference.step(noise, timestep, latent, return_dict=False)[0]
        np.testing.assert_array_equal(
            np.array(actual.astype(mx.float32)), latent.float().numpy()
        )


def test_vae_inverse_scaling_matches_bfloat16_reference():
    torch.manual_seed(22)
    x = torch.randn(1, 3, 5, 64).bfloat16()
    mean = torch.randn(64).bfloat16()
    std = (torch.rand(64) + 2).bfloat16()
    expected = x / std.reciprocal() + mean
    actual = decode_latents(
        mx.array(x.float().numpy()).astype(mx.bfloat16),
        dict(latents_mean=mean.float().tolist(), latents_std=std.float().tolist()),
    )
    np.testing.assert_array_equal(
        np.array(actual.astype(mx.float32)), expected.float().numpy()
    )


@pytest.mark.parametrize("grids", [[], [(1, 4, 6)], [(1, 4, 6), (1, 8, 4)]])
def test_image_positions_match_transformers(grids):
    config = hf.Qwen3VLConfig(
        text_config=dict(
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            vocab_size=128,
            rope_scaling=dict(rope_type="default", mrope_section=[3, 3, 2]),
        ),
        vision_config=dict(
            hidden_size=144,
            intermediate_size=288,
            depth=1,
            num_heads=2,
            out_hidden_size=64,
            deepstack_visual_indexes=[],
        ),
        image_token_id=100,
        vision_start_token_id=101,
    )
    with torch.device("meta"):
        reference = hf.Qwen3VLModel(config)
    ids = [3, 4, 5]
    for _, height, width in grids:
        ids += [101] + [100] * (height * width // 4) + [102, 6, 7]
    ids += [8, 9]
    expected, _ = reference.get_rope_index(
        torch.tensor([ids]),
        image_grid_thw=torch.tensor(grids) if grids else None,
    )
    actual = image_position_ids(np.array(ids), grids, 100, 2)
    np.testing.assert_array_equal(np.array(actual), expected.numpy())


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, *sys.argv[1:]]))
