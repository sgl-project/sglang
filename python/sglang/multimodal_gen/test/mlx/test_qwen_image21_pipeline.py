# SPDX-License-Identifier: Apache-2.0

import json
import sys

import numpy as np
import pytest
from PIL import Image

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")
mlx_utils = pytest.importorskip("mlx.utils")
transformers = pytest.importorskip("transformers")
tokenizers = pytest.importorskip("tokenizers")
torch = pytest.importorskip("torch")

from sglang.multimodal_gen.runtime.hardware_backend.mlx.qwen3vl_text import (
    Qwen3VLTextEncoder,
)
from sglang.multimodal_gen.runtime.hardware_backend.mlx.qwen3vl_vision import (
    Qwen3VLVisionEncoder,
)
from sglang.multimodal_gen.runtime.hardware_backend.mlx.qwen_image21 import (
    QwenImage21Transformer,
)
from sglang.multimodal_gen.runtime.hardware_backend.mlx.qwen_image21_pipeline import (
    QwenImage21MLXPipeline,
)
from sglang.multimodal_gen.runtime.hardware_backend.mlx.qwen_image21_processing import (
    decode_latents,
)
from sglang.multimodal_gen.runtime.hardware_backend.mlx.qwen_image21_vae import (
    QwenImage21VAE,
)
from sglang.multimodal_gen.runtime.hardware_backend.mlx.weights import load_vae


@pytest.fixture
def pipeline(tmp_path):
    special = [
        "[UNK]",
        "[PAD]",
        "<|im_start|>",
        "<|im_end|>",
        "<|vision_start|>",
        "<|vision_end|>",
        "<|image_pad|>",
        "<|video_pad|>",
    ]
    tokenizer = tokenizers.Tokenizer(
        tokenizers.models.WordLevel(
            dict(zip(special, range(len(special)))), unk_token="[UNK]"
        )
    )
    tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    tokenizer = transformers.Qwen2TokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="[UNK]",
        pad_token="[PAD]",
        additional_special_tokens=special[2:],
    )
    template = "{% for m in messages %}{{ '<|im_start|>' + m['role'] + '\\n' }}{% for c in m['content'] %}{{ c['text'] }}{% endfor %}{{ '<|im_end|>\\n' }}{% endfor %}"
    processor = transformers.Qwen3VLProcessor(
        tokenizer=tokenizer,
        image_processor=transformers.Qwen2VLImageProcessorFast(
            patch_size=16,
            temporal_patch_size=2,
            merge_size=2,
            size=dict(shortest_edge=1024, longest_edge=4096),
        ),
        video_processor=transformers.Qwen3VLVideoProcessor(),
        chat_template=template,
    )
    processor.save_pretrained(tmp_path / "processor")
    for folder in ["configs", "scheduler", "vae", "text_encoders", "diffusion_models"]:
        (tmp_path / folder).mkdir()
    text_config = dict(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
    )
    vision_config = dict(
        hidden_size=128,
        intermediate_size=272,
        depth=3,
        num_heads=2,
        out_hidden_size=64,
        deepstack_visual_indexes=[0, 1, 2],
    )
    dit_config = dict(
        num_layers=2,
        attention_head_dim=16,
        num_attention_heads=4,
        context_in_dim=64,
        in_channels=64,
    )
    vae_config = dict(base_dim=8, decoder_base_dim=8)
    mx.random.seed(67)
    text = Qwen3VLTextEncoder(**text_config, mrope_section=(3, 3, 2))
    vision = Qwen3VLVisionEncoder(**vision_config)
    dit = QwenImage21Transformer(**dit_config)
    vae = QwenImage21VAE(**vae_config)
    for model in (text, vision, dit, vae):
        model.set_dtype(mx.bfloat16)
    nn.quantize(
        dit,
        bits=4,
        group_size=64,
        class_predicate=lambda p, m: isinstance(m, nn.Linear),
    )
    for model in (text, vision):
        nn.quantize(
            model,
            bits=4,
            group_size=64,
            class_predicate=lambda p, m: (
                isinstance(m, (nn.Linear, nn.Embedding))
                and m.weight.shape[1] % 64 == 0
                and p != "embed_tokens"
            ),
        )
    encoder_weights = {
        f"model.{k}": v for k, v in mlx_utils.tree_flatten(text.parameters())
    }
    encoder_weights.update(
        {f"model.visual.{k}": v for k, v in mlx_utils.tree_flatten(vision.parameters())}
    )
    key = "model.visual.patch_embed.proj.weight"
    encoder_weights[key] = encoder_weights[key].transpose(0, 4, 1, 2, 3)
    mx.save_safetensors(
        str(tmp_path / "text_encoders/qwen3vl_8b_mlx_q4.safetensors"), encoder_weights
    )
    mx.save_safetensors(
        str(tmp_path / "diffusion_models/qwen_image_2.1_mlx_q4.safetensors"),
        dict(mlx_utils.tree_flatten(dit.parameters())),
    )
    vae_weights = {
        k: v.transpose(0, 3, 1, 2) if k.endswith(".weight") else v
        for k, v in mlx_utils.tree_flatten(vae.parameters())
    }
    mx.save_safetensors(
        str(tmp_path / "vae/qwen_image_2.1_vae_bf16.safetensors"), vae_weights
    )
    text_config["rope_scaling"] = dict(mrope_section=[3, 3, 2])
    dit_config["axes_dims_rope"] = [4, 6, 6]
    vae_config.update(latents_mean=[0.5] * 64, latents_std=[3.3] * 64)
    for name, config in [
        ("text_encoder", dict(text_config=text_config, vision_config=vision_config)),
        ("transformer", dit_config),
        ("vae", vae_config),
    ]:
        (tmp_path / "configs" / f"{name}_config.json").write_text(json.dumps(config))
    (tmp_path / "quantization_config.json").write_text(
        json.dumps(dict(method="affine", bits=4, group_size=64))
    )
    (tmp_path / "scheduler/scheduler_config.json").write_text(
        json.dumps(
            dict(
                num_train_timesteps=1000,
                use_dynamic_shifting=True,
                base_image_seq_len=256,
                max_image_seq_len=8192,
                base_shift=0.5,
                max_shift=0.9,
                shift_terminal=0.02,
            )
        )
    )
    return QwenImage21MLXPipeline(tmp_path)


@pytest.mark.parametrize("image_count", [0, 1, 2])
def test_pipeline_runs_all_phases_and_preserves_rgba_and_seed(pipeline, image_count):
    image = Image.new("RGBA", (64, 96), (70, 120, 200, 96))
    kwargs = dict(
        prompt="A red panda",
        width=64,
        height=64,
        num_inference_steps=3,
        seed=17,
        images=[image] * image_count,
        guidance_scale=3.71 if image_count == 2 else 1,
    )
    steps = []
    first, timings = pipeline.generate(
        **kwargs, progress=lambda i, n, seconds: steps.append((i, n))
    )
    second, _ = pipeline.generate(**kwargs)
    assert first.mode == "RGBA" and first.size == (64, 64)
    assert steps == [(1, 3), (2, 3), (3, 3)]
    assert timings["peak_memory_bytes"] > 0
    np.testing.assert_array_equal(np.array(first), np.array(second))
    different, _ = pipeline.generate(**(kwargs | {"seed": 18}))
    assert not np.array_equal(np.array(first), np.array(different))


def test_decode_preserves_sglang_bfloat16_pixel_rounding(pipeline):
    mx.random.seed(41)
    latents = mx.random.normal((1, 4, 4, 64)).astype(mx.bfloat16)
    vae = load_vae(
        str(pipeline.model_path / "vae/qwen_image_2.1_vae_bf16.safetensors"),
        pipeline.configs["vae"],
    )
    decoded = vae.compile_decode()(decode_latents(latents, pipeline.configs["vae"]))
    pixels = torch.from_numpy(np.array(decoded.astype(mx.float32))).bfloat16()
    # decoding and output materialization both retain the native tensor dtype
    expected = ((pixels / 2 + 0.5).clamp(0, 1) * 255).to(torch.uint8).numpy()[0]
    actual = pipeline.decode(latents.reshape(1, 16, 64), width=64, height=64)
    assert actual.mode == "RGBA"
    np.testing.assert_array_equal(np.array(actual), expected)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, *sys.argv[1:]]))
