# SPDX-License-Identifier: Apache-2.0
"""Image noise: one generator per request.

Draw order (image_noise.py): target noise, then one anchor draw per
reference, then the reference VAE posterior samples; nothing touches the
process-global RNG.
"""

import pytest
import torch
from PIL import Image

from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae.klvae import (
    DiagonalGaussianDistribution,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3 import (
    denoise_loop,
    reference_encoding,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.constants import (
    MINIMAX_H3_DENOISE_STATE_EXTRA_KEY as STATE,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.constants import (
    MINIMAX_H3_REFERENCE_IMAGE_ROWS_EXTRA_KEY as REFS,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.image_noise import (
    MINIMAX_H3_IMAGE_NOISE_EXTRA_KEY as NOISE,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.packed_tokens import (
    minimax_h3_patchify_video_latent,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.stages.denoising import (
    MiniMaxH3DenoisingStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.stages.latent_preparation import (
    MiniMaxH3LatentPreparationStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.stages.timestep_preparation import (
    MiniMaxH3TimestepPreparationStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.stages.visual_encoding import (
    MiniMaxH3VisualEncodingStage,
)
from sglang.multimodal_gen.test.unit.test_minimax_h3_image_pipeline import (
    ZeroVelocityModel,
    make_req,
)
from sglang.multimodal_gen.test.unit.test_minimax_h3_image_pipeline import (
    server as server,
)

T = torch.tensor(denoise_loop.MINIMAX_H3_IMGVID_COND_TIMESTEP, dtype=torch.float32)
# make_req: 64x32 target -> latent grid (2, 4). _encode_references replaces
# make_req's two stub references with ones encoded at the given pixel sizes.
TARGET_GRID = (2, 4)


def _rows(generator, grid):
    noise = torch.randn((1, 24, 1, *grid), generator=generator)
    return minimax_h3_patchify_video_latent(noise, patch_size=[1, 2, 2])


class SamplingVAE(torch.nn.Module):
    """encode_images double: the posterior sample is the only random draw."""

    parallel_tiling = False

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.generators = []

    def encode_images(self, image, use_fp16_latent, generator=None):
        self.generators.append(generator)
        width, height = image.size
        return [torch.randn(24, 1, height // 16, width // 16, generator=generator)]


def _encode_references(req, server, monkeypatch, sizes):
    monkeypatch.setattr(
        reference_encoding,
        "minimax_h3_prepared_reference_image",
        lambda batch, plan: {
            "images": [
                {"image": Image.new("RGB", size), "condition_index": index}
                for index, size in enumerate(sizes)
            ]
        },
    )
    vae = SamplingVAE()
    req.extra.pop(REFS)
    MiniMaxH3VisualEncodingStage(
        vae, server.pipeline_config.vae_config.arch_config
    ).forward(req, server)
    return vae


def _denoise(req, server, monkeypatch):
    captured = {}

    def loop(**kwargs):
        captured.update(kwargs)
        return kwargs["initial_video_rows"].clone()

    monkeypatch.setattr(denoise_loop, "minimax_h3_image_denoise_loop", loop)
    MiniMaxH3TimestepPreparationStage().forward(req, server)
    MiniMaxH3LatentPreparationStage().forward(req, server)
    MiniMaxH3DenoisingStage(ZeroVelocityModel()).forward(req, server)
    return captured["keyframe_cond_rows"]


@pytest.mark.skipif(
    torch.cuda.is_available(),
    reason="CPU-only weight-free test; it mixes CPU and CUDA tensors on GPU hosts",
)
def test_draws_target_then_anchors_then_posteriors(server, monkeypatch):
    req = make_req(server, "edit")
    before = torch.random.get_rng_state().clone()
    vae = _encode_references(req, server, monkeypatch, [(64, 32), (96, 32)])
    anchors = _denoise(req, server, monkeypatch)
    assert torch.equal(before, torch.random.get_rng_state())

    expected = torch.Generator().manual_seed(17)
    target = _rows(expected, TARGET_GRID)
    anchor_noise = [_rows(expected, (2, 4)), _rows(expected, (2, 6))]
    assert torch.equal(req.extra[STATE]["initial_video_rows"], target)
    clean = [ref["rows"] for ref in req.extra[REFS]["images"]]
    assert torch.equal(
        anchors,
        torch.cat([T * c + (1.0 - T) * n for c, n in zip(clean, anchor_noise)]),
    )
    # Both references sampled from the request generator, after the anchors.
    assert vae.generators == [req.extra[NOISE].generator] * 2
    torch.randn(24, 1, 2, 4, generator=expected)
    torch.randn(24, 1, 2, 6, generator=expected)
    assert torch.equal(req.extra[NOISE].generator.get_state(), expected.get_state())


def test_noise_is_a_function_of_the_seed(server, monkeypatch):
    def noise(seed):
        req = make_req(server, "edit", seed=seed)
        anchors = _denoise(req, server, monkeypatch)
        return req.extra[STATE]["initial_video_rows"], anchors

    same_target, same_anchors = noise(17)
    target, anchors = noise(17)
    other_target, other_anchors = noise(18)
    assert torch.equal(target, same_target) and torch.equal(anchors, same_anchors)
    assert not torch.equal(target, other_target)
    assert not torch.equal(anchors, other_anchors)


@pytest.mark.skipif(
    torch.cuda.is_available(),
    reason="CPU-only weight-free test; it mixes CPU and CUDA tensors on GPU hosts",
)
def test_ranks_without_the_encode_draw_get_the_same_noise(server, monkeypatch):
    # Rank 0 draws in visual encoding; a receiving rank (or dedup sibling) only
    # sees the published rows and draws in latent preparation.
    owner = make_req(server, "edit")
    _encode_references(owner, server, monkeypatch, [(32, 32), (32, 64)])
    receiver = make_req(server, "edit")
    receiver.extra[REFS] = owner.extra[REFS]
    owner_anchors = _denoise(owner, server, monkeypatch)
    receiver_anchors = _denoise(receiver, server, monkeypatch)
    assert torch.equal(owner_anchors, receiver_anchors)
    assert torch.equal(
        owner.extra[STATE]["initial_video_rows"],
        receiver.extra[STATE]["initial_video_rows"],
    )


def test_posterior_sample_draws_from_the_given_generator():
    # mean 1, logvar 0: the sample is exactly 1 + the generator's normal draw.
    parameters = torch.cat([torch.ones(1, 24, 1, 2, 2), torch.zeros(1, 24, 1, 2, 2)], 1)
    sample = DiagonalGaussianDistribution(parameters).sample(
        torch.Generator().manual_seed(3)
    )
    noise = torch.randn(1, 24, 1, 2, 2, generator=torch.Generator().manual_seed(3))
    assert torch.equal(sample, 1.0 + noise)
