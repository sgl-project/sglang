"""Pixtral vision tower on several images at once.

The tower packs every image's patches into one sequence. Each image must attend
only over its own patches, on every CUDA vision attention backend, so encoding
images together gives the same features as encoding them one at a time.
"""

import pytest
import torch
from transformers import PixtralVisionConfig

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b", runner_config="1-gpu-large")

if not torch.cuda.is_available():
    pytest.skip("requires CUDA", allow_module_level=True)

from sglang.srt.models import pixtral  # noqa: E402
from sglang.srt.runtime_context import reset_context  # noqa: E402
from sglang.srt.server_args import ServerArgs  # noqa: E402
from sglang.test.parallel_groups import publish  # noqa: E402

PATCH = 4
# Different shapes, so a leak across images cannot cancel out.
IMAGE_SIZES = [(24, 40), (36, 20), (16, 16)]

BACKENDS = [
    "triton_attn",
    pytest.param(
        "fa3",
        marks=pytest.mark.skipif(
            torch.cuda.get_device_capability()[0] != 9, reason="FA3 requires sm_90"
        ),
    ),
    "sdpa",
]


@pytest.fixture
def tower(request):
    reset_context()
    request.addfinalizer(reset_context)
    publish(
        ServerArgs(
            model_path="dummy",
            device="cuda",
            tp_size=1,
            mm_attention_backend=request.param,
        ),
        role="test",
    )
    config = PixtralVisionConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        head_dim=16,
        patch_size=PATCH,
        image_size=64,
    )
    torch.manual_seed(0)
    with torch.device("cuda"):
        model = pixtral.PixtralHFVisionModel(config, quant_config=None)
    model = model.to(torch.bfloat16).eval()
    with torch.no_grad():
        for param in model.parameters():
            param.normal_(std=0.2)
    return model


def _images():
    torch.manual_seed(1)
    return [torch.randn(3, h, w, device="cuda") for h, w in IMAGE_SIZES]


def _batch(images):
    height = max(img.shape[1] for img in images)
    width = max(img.shape[2] for img in images)
    batch = torch.zeros(len(images), 3, height, width, device="cuda")
    for i, img in enumerate(images):
        batch[i, :, : img.shape[1], : img.shape[2]] = img
    return batch


@pytest.mark.parametrize("tower", BACKENDS, indirect=True)
def test_images_encoded_together_match_one_at_a_time(tower, monkeypatch):
    images = _images()

    def fail(*args, **kwargs):
        raise AssertionError("the tower must not build a dense attention mask")

    monkeypatch.setattr(pixtral, "_get_pixtral_attention_mask", fail)
    with torch.inference_mode():
        together = tower(_batch(images), image_sizes=IMAGE_SIZES)[0]
        alone = [
            tower(img[None], image_sizes=[size])[0]
            for img, size in zip(images, IMAGE_SIZES)
        ]

    counts = [(h // PATCH) * (w // PATCH) for h, w in IMAGE_SIZES]
    assert together.shape[0] == sum(counts)
    for got, want in zip(together.split(counts), alone):
        torch.testing.assert_close(got, want, atol=1e-2, rtol=1e-2)
