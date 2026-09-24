# SPDX-License-Identifier: Apache-2.0

import math

import mlx.core as mx
import numpy as np
from PIL import Image

SYSTEM_PROMPT = "Comprehend and analyze the provided prompt."


def resize_images(images, width, height):
    resized = []
    area = width * height
    for image in images:
        image_width = max(
            32, round(math.sqrt(area * image.width / image.height) / 32) * 32
        )
        image_height = max(
            32, round(math.sqrt(area * image.height / image.width) / 32) * 32
        )
        resized.append(
            image.convert("RGBA").resize(
                (image_width, image_height), Image.Resampling.LANCZOS
            )
        )
    return resized


def image_position_ids(input_ids, image_grid_thw, image_token_id, merge_size):
    """Build Qwen3-VL positions for one unpadded text/image prompt."""
    positions = []
    cursor = offset = 0
    tokens = input_ids.tolist()
    for frames, height, width in image_grid_thw:
        start = tokens.index(image_token_id, cursor)
        length = start - cursor
        positions.append(np.broadcast_to(np.arange(length) + offset, (3, length)))
        grid = np.indices((frames, height // merge_size, width // merge_size)).reshape(
            3, -1
        )
        positions.append(grid + offset + length)
        offset = int(positions[-1].max()) + 1
        cursor = start + grid.shape[1]
    length = len(tokens) - cursor
    positions.append(np.broadcast_to(np.arange(length) + offset, (3, length)))
    return mx.array(np.concatenate(positions, axis=1)[:, None], dtype=mx.int32)


def encode_prompt(processor, text_encoder, vision_encoder, prompt, images):
    """Return pre-norm conditioning with each image run collapsed to one slot."""
    prefix = " ".join(
        f"<image{i + 1}><|vision_start|><|image_pad|><|vision_end|>"
        for i in range(len(images))
    )
    text = (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{prefix}{prompt or ' '}<|im_end|>\n<|im_start|>assistant\n"
    )
    kwargs = dict(text=[text], padding=True, padding_side="left", return_tensors="pt")
    if images:
        vision_images = []
        for image in images:
            white = Image.new("RGB", image.size, (255, 255, 255))
            white.paste(image, mask=image.getchannel("A"))
            vision_images.append(white)
        kwargs["images"] = vision_images
    inputs = processor(**kwargs)
    valid = inputs.attention_mask[0].bool()
    ids = inputs.input_ids[0, valid].numpy()
    input_ids = mx.array(ids[None])
    image_token_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    grid = inputs.image_grid_thw.tolist() if images else []
    positions = image_position_ids(
        ids, grid, image_token_id, processor.image_processor.merge_size
    )
    encoder_kwargs = dict(input_ids=input_ids, position_ids=positions)
    if images:
        pixels = mx.array(inputs.pixel_values.numpy()).astype(mx.bfloat16)
        pooled, deepstack = vision_encoder(pixels, grid)
        visual_positions = mx.array(np.flatnonzero(ids == image_token_id))
        embeddings = text_encoder.embed_tokens(input_ids)
        embeddings[0, visual_positions] = pooled
        encoder_kwargs = dict(
            inputs_embeds=embeddings,
            position_ids=positions,
            visual_positions=visual_positions,
            deepstack_visual_embeds=deepstack,
        )
    hidden = text_encoder(**encoder_kwargs)
    system_message = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]}
    ]
    drop = len(
        processor.apply_chat_template(system_message, tokenize=True, return_dict=False)[
            0
        ]
    )
    hidden, ids = hidden[:, drop:], ids[drop:]
    image_mask = ids == image_token_id
    keep = ~image_mask
    keep[0] = True
    keep[1:] |= image_mask[1:] & ~image_mask[:-1]
    return hidden[:, mx.array(np.flatnonzero(keep))], image_mask[keep].tolist()


def flow_schedule(config, steps, image_seq_len):
    if steps < 1:
        raise ValueError("num_inference_steps must be positive")
    if (
        not config["use_dynamic_shifting"]
        or config.get("time_shift_type", "exponential") != "exponential"
    ):
        raise ValueError("Qwen-Image 2.1 MLX requires dynamic exponential shifting")
    if any(
        config.get(key, False)
        for key in (
            "invert_sigmas",
            "stochastic_sampling",
            "use_beta_sigmas",
            "use_exponential_sigmas",
            "use_karras_sigmas",
        )
    ):
        raise ValueError("unsupported Qwen-Image 2.1 MLX scheduler configuration")
    slope = (config["max_shift"] - config["base_shift"]) / (
        config["max_image_seq_len"] - config["base_image_seq_len"]
    )
    mu = (
        image_seq_len * slope
        + config["base_shift"]
        - config["base_image_seq_len"] * slope
    )
    sigmas = np.linspace(1, 1 / steps, steps).astype(np.float32)
    sigmas = (math.exp(mu) / (math.exp(mu) + (1 / sigmas - 1))).astype(np.float32)
    terminal = config.get("shift_terminal")
    if terminal is not None and sigmas[-1] != 1:
        one_minus = 1 - sigmas
        sigmas = 1 - one_minus / (one_minus[-1] / (1 - terminal))
    timesteps = sigmas * config["num_train_timesteps"]
    return mx.array(np.append(sigmas, np.float32(0))), mx.array(timesteps)


def flow_step(latents, prediction, delta):
    # torch's zero-dimensional FP32 delta is converted to the BF16 output dtype
    update = delta.astype(prediction.dtype) * prediction
    return (latents.astype(mx.float32) + update.astype(mx.float32)).astype(
        prediction.dtype
    )


def decode_latents(latents, config):
    mean = mx.array(config["latents_mean"]).astype(latents.dtype)
    std = mx.array(config["latents_std"]).astype(latents.dtype)
    scale = mx.reciprocal(std.astype(mx.float32)).astype(latents.dtype)
    return latents / scale + mean
