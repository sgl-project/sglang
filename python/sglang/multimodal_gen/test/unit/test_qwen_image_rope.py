# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch

import pytest
import torch

from sglang.multimodal_gen.configs.models.dits.qwenimage import QwenImageDitConfig
from sglang.multimodal_gen.configs.pipeline_configs.qwen_image import (
    QwenImagePipelineConfig,
)
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.models.dits import qwen_image
from sglang.multimodal_gen.runtime.models.dits.qwen_image import (
    QwenEmbedLayer3DRope,
    QwenEmbedRope,
    QwenImageTransformer2DModel,
)

# the DiT's fused kernels need a GPU on a GPU platform; elsewhere run on the CPU
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _reference_freqs(coordinates, axes, device):
    frequencies = []
    for index, dim in zip(coordinates, axes):
        index = index.to(device)
        scale = 1.0 / torch.pow(
            10000, torch.arange(0, dim, 2, device=device).float().div(dim)
        )
        phase = index.unsqueeze(-1) * scale
        frequencies.append(torch.polar(torch.ones_like(phase), phase))
    return torch.cat(frequencies, dim=-1)


@pytest.mark.parametrize("layered", [False, True])
@pytest.mark.parametrize("scale", [False, True])
@pytest.mark.parametrize("init_device", ["cpu", "meta"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
@torch.inference_mode()
def test_qwen_image_rope_coordinates_and_device_lifecycle(
    layered, scale, init_device, dtype, device
):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    device = torch.device(device)
    axes = [4, 6, 6]
    with torch.device(init_device):
        rope = (QwenEmbedLayer3DRope if layered else QwenEmbedRope)(10000, axes, scale)
    rope.to(dtype=dtype)
    reference_device = device if init_device == "meta" else torch.device("cpu")
    assert not rope.state_dict() and not list(rope.buffers())
    try:
        for shapes in [[(2, 3, 4), (1, 4, 3)], [(1, 1, 1)] * 5]:
            expected_images = []
            text_start = len(shapes) - 1 if layered else 0
            for index, (frames, height, width) in enumerate(shapes):
                frame_index = (
                    torch.tensor([-1])
                    if layered and index == len(shapes) - 1
                    else torch.arange(frames) + index
                )
                coordinates = torch.meshgrid(
                    frame_index,
                    torch.arange(height) - ((height + 1) // 2 if scale else 0),
                    torch.arange(width) - ((width + 1) // 2 if scale else 0),
                    indexing="ij",
                )
                expected_images.append(
                    _reference_freqs(coordinates, axes, reference_device).flatten(0, 2)
                )
                text_start = max(
                    text_start,
                    height // 2 if scale else height,
                    width // 2 if scale else width,
                )
            text_index = torch.arange(text_start, text_start + 7)
            expected = (
                torch.cat(expected_images).to(device),
                _reference_freqs([text_index] * 3, axes, reference_device).to(device),
            )
            for _ in range(2):
                actual = rope([shapes], [3, 7], device)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            assert rope.pos_freqs.dtype == rope.neg_freqs.dtype == torch.complex64
            assert rope.pos_freqs.device.type == device.type
        assert rope._compute_video_freqs.cache_info().maxsize == (
            None if layered else 128
        )
    finally:
        rope._compute_video_freqs.cache_clear()
        if layered:
            rope._compute_condition_freqs.cache_clear()


def _tiny_qwen_image_dit(num_layers):
    config = QwenImageDitConfig()
    for key, value in dict(
        in_channels=8,
        out_channels=8,
        num_layers=num_layers,
        attention_head_dim=128,
        num_attention_heads=2,
        joint_attention_dim=12,
    ).items():
        setattr(config.arch_config, key, value)
    model = QwenImageTransformer2DModel(config, {})
    generator = torch.Generator().manual_seed(0)
    for param in model.parameters():
        param.copy_(0.2 * torch.randn(param.shape, generator=generator))
    return model.to(device=_DEVICE, dtype=torch.bfloat16)


def _tiny_qwen_image_inputs(model, txt_seq_lens):
    batch_size, txt_len = len(txt_seq_lens), max(txt_seq_lens)
    img_shapes = [[(1, 2, 2)]] * batch_size
    freqs_cis, freqs_complex = QwenImagePipelineConfig.get_freqs_cis(
        img_shapes, txt_seq_lens, model.rotary_emb, _DEVICE, torch.bfloat16
    )
    generator = torch.Generator().manual_seed(1)
    bf16 = dict(device=_DEVICE, dtype=torch.bfloat16)
    return dict(
        hidden_states=torch.randn(batch_size, 4, 8, generator=generator).to(**bf16),
        encoder_hidden_states=torch.randn(
            batch_size, txt_len, 12, generator=generator
        ).to(**bf16),
        timestep=torch.full((batch_size,), 500.0, **bf16),
        img_shapes=img_shapes,
        txt_seq_lens=txt_seq_lens,
        freqs_cis=freqs_cis,
        freqs_complex=freqs_complex,
    )


@torch.no_grad()
def test_batched_text_varlen_metadata_is_built_once(single_process_model_parallel):
    """Batched prompts of different lengths reuse their varlen metadata: building
    it uploads the lengths from pageable memory, a host sync per step."""
    model = _tiny_qwen_image_dit(num_layers=2)
    inputs = _tiny_qwen_image_inputs(model, [3, 2])
    inputs["encoder_hidden_states_mask"] = torch.tensor(
        [[True, True, True], [True, True, False]], device=_DEVICE
    )
    qwen_image._text_image_varlen_meta.cache_clear()
    with (
        set_forward_context(current_timestep=0, attn_metadata=None),
        patch.object(
            qwen_image,
            "build_varlen_mask_meta_from_ranges",
            wraps=qwen_image.build_varlen_mask_meta_from_ranges,
        ) as build,
    ):
        first, second = model(**inputs), model(**inputs)
    assert build.call_count == 1
    torch.testing.assert_close(second, first, atol=0, rtol=0)
