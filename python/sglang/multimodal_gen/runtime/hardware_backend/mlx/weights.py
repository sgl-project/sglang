# SPDX-License-Identifier: Apache-2.0

import inspect
import re

import mlx.core as mx
import mlx.nn as nn

from .qwen3vl_text import Qwen3VLTextEncoder
from .qwen3vl_vision import Qwen3VLVisionEncoder
from .qwen_image21 import QwenImage21Transformer
from .qwen_image21_vae import QwenImage21VAE


def create_model(model_class, config):
    fields = inspect.signature(model_class).parameters
    return model_class(**{key: value for key, value in config.items() if key in fields})


def load_quantized_weights(model, weights, quantization):
    if quantization["method"] != "affine":
        raise ValueError("Qwen-Image MLX requires affine quantized weights")
    nn.quantize(
        model,
        bits=quantization["bits"],
        group_size=quantization["group_size"],
        class_predicate=lambda path, module: (
            isinstance(module, (nn.Linear, nn.Embedding))
            and f"{path}.scales" in weights
        ),
    )
    model.load_weights(list(weights.items()), strict=True)
    model.eval()
    mx.eval(model.parameters())
    return model


def load_transformer(path, config, quantization):
    if config.get("patch_size", 1) != 1 or not config.get("causal_condition", True):
        raise ValueError("Qwen-Image 2.1 requires patch_size=1 and causal_condition")
    weights = mx.load(path)
    converted = {}
    for name, value in weights.items():
        if ".img_mlp.gate_up." in name:
            gate, up = mx.split(value, 2, axis=0)
            converted[name.replace(".gate_up.", ".gate_layer.")] = gate
            converted[name.replace(".gate_up.", ".proj.")] = up
        else:
            converted[name] = value
    model = create_model(QwenImage21Transformer, config)
    return load_quantized_weights(model, converted, quantization)


def load_encoders(path, config, quantization, with_images):
    weights = mx.load(path)
    text_weights, vision_weights = {}, {}
    for name, value in weights.items():
        if name.startswith("model.visual."):
            vision_weights[name.removeprefix("model.visual.")] = value
        elif name.startswith("model.language_model."):
            text_weights[name.removeprefix("model.language_model.")] = value
        elif name.startswith("model."):
            text_weights[name.removeprefix("model.")] = value
        elif not name.startswith("lm_head."):
            raise ValueError(f"unrecognized Qwen3-VL weight: {name}")
    text_config = dict(config["text_config"])
    text_config["mrope_section"] = text_config["rope_scaling"]["mrope_section"]
    text_encoder = load_quantized_weights(
        create_model(Qwen3VLTextEncoder, text_config), text_weights, quantization
    )
    vision_encoder = None
    if with_images:
        name = "patch_embed.proj.weight"
        vision_weights[name] = vision_weights[name].transpose(0, 2, 3, 4, 1)
        vision_encoder = load_quantized_weights(
            create_model(Qwen3VLVisionEncoder, config["vision_config"]),
            vision_weights,
            quantization,
        )
    return text_encoder, vision_encoder


def convert_vae_name(name):
    if name.startswith("conv1."):
        return name.replace("conv1.", "quant_conv.", 1)
    if name.startswith("conv2."):
        return name.replace("conv2.", "post_quant_conv.", 1)
    name = re.sub(r"^(encoder|decoder)\.conv1\.", r"\1.conv_in.", name)
    name = name.replace(".head.0.", ".norm_out.").replace(".head.2.", ".conv_out.")
    for index, target in [(0, "resnets.0"), (1, "attentions.0"), (2, "resnets.1")]:
        name = name.replace(f".middle.{index}.", f".mid_block.{target}.")
    for source, target, count, sampler in [
        ("downsamples", "down_blocks", 2, "downsampler"),
        ("upsamples", "up_blocks", 3, "upsampler"),
    ]:

        def block_name(match):
            block, child = match.groups()
            layer = f"resnets.{child}" if int(child) < count else sampler
            return f".{target}.{block}.{layer}."

        name = re.sub(rf"\.{source}\.(\d+)\.{source}\.(\d+)\.", block_name, name)
    for source, target in [
        ("residual.0.", "norm1."),
        ("residual.2.", "conv1."),
        ("residual.3.", "norm2."),
        ("residual.6.", "conv2."),
        (".shortcut.", ".conv_shortcut."),
        ("resample.1.", "conv."),
    ]:
        name = name.replace(source, target)
    return name


def load_vae(path, config):
    if not config.get("is_residual", True) or config.get("patch_size") not in (None, 1):
        raise ValueError("unsupported Qwen-Image 2.1 VAE architecture")
    weights = mx.load(path)
    converted = {}
    for name, value in weights.items():
        # temporal convolutions are unused in the single-image checkpoint path
        if ".time_conv." in name:
            continue
        name = convert_vae_name(name)
        if name.endswith(".gamma"):
            value = value.reshape(-1)
        elif name.endswith(".weight"):
            if value.ndim == 5:
                value = mx.squeeze(value, axis=2)
            value = value.transpose(0, 2, 3, 1)
        converted[name] = value
    model = create_model(QwenImage21VAE, config)
    model.load_weights(list(converted.items()), strict=True)
    model.eval()
    mx.eval(model.parameters())
    return model
