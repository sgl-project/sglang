# SPDX-License-Identifier: Apache-2.0
"""Native Ming conditioning and independent RGBA layer decoding."""

import math

import numpy as np
import torch
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor, to_tensor
from transformers.models.qwen2_vl.image_processing_pil_qwen2_vl import (
    Qwen2VLImageProcessorPil,
    smart_resize,
)

from sglang.multimodal_gen.configs.pipeline_configs.ming_image import (
    MingImageLayerPipelineConfig,
)
from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType
from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.managers.memory_managers.component_manager import (
    ComponentUse,
)
from sglang.multimodal_gen.runtime.models.encoders.ming_image import ming_position_ids
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.decoding import DecodingStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.text_encoding import (
    TextEncodingStage,
)
from sglang.multimodal_gen.runtime.utils.precision import resolve_precision
from sglang.multimodal_gen.runtime.utils.vision import load_image


class MingImageEncodingStage(TextEncodingStage):
    def __init__(self, text_encoders, tokenizers):
        super().__init__(text_encoders, tokenizers)
        self.image_processor = Qwen2VLImageProcessorPil.from_pretrained(
            tokenizers[0].name_or_path
        )

    @torch.no_grad()
    def forward(self, batch, server_args):
        if batch.batch_size != 1:
            raise ValueError(
                "Ming conditioning expects one prompt per expanded request"
            )
        if batch.negative_prompt and str(batch.negative_prompt).strip():
            raise ValueError(
                "Ming uses zero unconditional embeddings, not a negative prompt"
            )
        layered = isinstance(server_args.pipeline_config, MingImageLayerPipelineConfig)
        expected_contract = ("learned", True) if layered else ("zero_masked", False)
        arch = server_args.pipeline_config.dit_config.arch_config
        if (arch.alignment_padding_mode, arch.multi_frame_output) != expected_contract:
            raise ValueError(
                "Ming checkpoint padding/output contract does not match the selected pipeline"
            )
        count = batch.extra.get("ming_num_layers", 1)
        batch.extra["ming_frames"] = count + 1 if layered else 1
        prompt = batch.prompt if isinstance(batch.prompt, str) else batch.prompt[0]
        if layered and not prompt.strip():
            prompt = f"Decompose this image into {count} layers."

        image_paths = batch.image_path
        if not isinstance(image_paths, list):
            image_paths = [image_paths] if image_paths else []
        if len(image_paths) > 1 or (layered and not image_paths):
            raise ValueError(
                "Ming accepts one reference image; layer decomposition requires it"
            )
        device = get_local_torch_device()
        encoder = self.text_encoders[0]
        tokenizer = self.tokenizers[0]
        inputs = {}
        grids = []
        image_text = ""
        if image_paths:
            image = load_image(
                image_paths[0], convert_method=lambda value: value.convert("RGBA")
            )
            # Reference VAE pixels and VLM pixels use separate official transforms.
            batch.extra["ming_reference_image"] = image
            h, w = ming_reference_size(
                image.height, image.width, min(batch.height, batch.width)
            )
            batch.height, batch.width = h, w
            ratio = max(h / image.height, w / image.width)
            batch.extra["ming_output_size"] = (
                int(image.width * ratio),
                int(image.height * ratio),
            )
            h0, w0 = smart_resize(
                image.height, image.width, min_pixels=56 * 56, max_pixels=1024 * 28 * 28
            )
            processed = self.image_processor(
                images=image.convert("RGB").resize((w0, h0)), return_tensors="pt"
            )
            grids = [tuple(grid) for grid in processed.image_grid_thw.tolist()]
            patches = math.prod(grids[0]) // 4
            image_text = "<image>" + "<imagePatch>" * patches + "</image>\n"
            inputs = {
                "pixel_values": processed.pixel_values.to(
                    device=device, dtype=encoder.dtype
                ),
                "image_grid_thw": processed.image_grid_thw,
            }
        if batch.height % 16 or batch.width % 16:
            raise ValueError("Ming height and width must be divisible by 16")
        # Fixed training template, not the tokenizer's generic chat template.
        system = "<role>SYSTEM</role>\u4f60\u662f\u4e00\u4e2a\u53cb\u597d\u7684AI\u52a9\u624b\u3002\n\ndetailed thinking off"
        text = (
            system
            + tokenizer.eos_token
            + "<role>HUMAN</role>"
            + image_text
            + prompt
            + tokenizer.eos_token
            + "<role>ASSISTANT</role>"
        )
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        query_count = sum(
            scale * scale
            for scale in encoder.config.projection_config["img_gen_scales"]
        )
        ids += (
            [tokenizer.convert_tokens_to_ids("<image>")]
            + [encoder.image_token] * query_count
            + [tokenizer.convert_tokens_to_ids("</image>")]
        )
        grids.append((1, 2, query_count * 2))
        inputs.update(
            input_ids=torch.tensor([ids], device=device),
            position_ids=ming_position_ids(ids, grids, encoder.image_token).to(device),
        )
        self._begin_text_encoder_use(0)
        # the official MoE router runs under BF16 autocast, including its logits
        with torch.autocast(device.type, dtype=encoder.dtype):
            encoded = self._forward_text_encoder(encoder, inputs)
        batch.prompt_embeds = [encoded.last_hidden_state]
        batch.negative_prompt_embeds = [torch.zeros_like(encoded.last_hidden_state)]
        batch.extra["ming_direct"] = encoded.hidden_states[0]
        return batch


class MingImageReferenceStage(PipelineStage):
    role_affinity = RoleType.ENCODER

    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def component_uses(self, server_args, stage_name=None):
        return [
            ComponentUse(
                stage_name=self._component_stage_name(stage_name),
                component_name="vae",
                target_dtype=resolve_precision(
                    server_args, "vae", precision_attr="vae_precision"
                ),
            )
        ]

    @torch.no_grad()
    def forward(self, batch, server_args):
        image = batch.extra.pop("ming_reference_image", None)
        if image is None:
            return batch
        pixels = to_tensor(
            image.resize((batch.width, batch.height), Image.Resampling.BILINEAR)
        )
        pixels = ((pixels - 0.5) * 2).unsqueeze(0).unsqueeze(2)
        with self.use_declared_component(component_name="vae", module=self.vae) as vae:
            if server_args.pipeline_config.vae_tiling:
                vae.enable_tiling()
            pixels = pixels.to(
                device=get_local_torch_device(), dtype=next(vae.parameters()).dtype
            )
            batch.extra["ming_reference_latents"] = vae.encode(pixels).mode() * 8.0064
        return batch


class MingImageDecodingStage(DecodingStage):
    def scale_and_shift(self, latents, server_args):
        return latents / 8.0064

    def decode(self, latents, server_args, *, vae_dtype):
        # Rounding before scaling is part of the official BF16 decode contract.
        latents = latents.to(vae_dtype)
        batch, channels, frames, height, width = latents.shape
        flat = latents.transpose(1, 2).reshape(
            batch * frames, channels, 1, height, width
        )
        decoded = super().decode(flat, server_args, vae_dtype=vae_dtype)
        return decoded.reshape(batch, frames, 4, *decoded.shape[-2:]).transpose(1, 2)

    def forward(self, batch, server_args):
        output = super().forward(batch, server_args)
        images = output.output
        if isinstance(server_args.pipeline_config, MingImageLayerPipelineConfig):
            images = images[:, :, 1:]
        images = images.transpose(1, 2).flatten(0, 1)
        size = batch.extra.get("ming_output_size")
        if size is not None and images.shape[-2:] != size[::-1]:
            # Match the public CLI's uint8 conversion before LANCZOS resizing.
            pixels = (
                (images.float().permute(0, 2, 3, 1).cpu().numpy() * 255)
                .round()
                .astype(np.uint8)
            )
            images = (
                torch.stack(
                    [
                        pil_to_tensor(
                            Image.fromarray(pixel).resize(
                                size, Image.Resampling.LANCZOS
                            )
                        )
                        for pixel in pixels
                    ]
                ).float()
                / 255
            )
        output.output = images
        return output


def ming_reference_size(height, width, resolution):
    """The published 512/1024 aspect buckets; keys preserve official rounding."""
    buckets = {
        512: {
            0.25: (256, 1024),
            0.26: (256, 992),
            0.27: (256, 960),
            0.28: (256, 928),
            0.32: (288, 896),
            0.33: (288, 864),
            0.35: (288, 832),
            0.4: (320, 800),
            0.42: (320, 768),
            0.48: (352, 736),
            0.5: (352, 704),
            0.52: (352, 672),
            0.5455: (384, 704),
            0.57: (384, 672),
            0.6: (384, 640),
            0.65: (416, 640),
            0.68: (416, 608),
            0.72: (416, 576),
            0.78: (448, 576),
            0.82: (448, 544),
            0.88: (480, 544),
            0.94: (480, 512),
            1.0: (512, 512),
            1.07: (512, 480),
            1.13: (544, 480),
            1.21: (544, 448),
            1.29: (576, 448),
            1.38: (576, 416),
            1.46: (608, 416),
            1.5385: (640, 416),
            1.67: (640, 384),
            1.75: (672, 384),
            1.8333: (704, 384),
            2.0: (704, 352),
            2.09: (736, 352),
            2.4: (768, 320),
            2.5: (800, 320),
            2.89: (832, 288),
            3.0: (864, 288),
            3.11: (896, 288),
            3.62: (928, 256),
            3.75: (960, 256),
            3.88: (992, 256),
            4.0: (1024, 256),
        },
        1024: {
            0.25: (512, 2048),
            0.26: (512, 1984),
            0.27: (512, 1920),
            0.28: (512, 1856),
            0.32: (576, 1792),
            0.33: (576, 1728),
            0.35: (576, 1664),
            0.4: (640, 1600),
            0.42: (640, 1536),
            0.48: (704, 1472),
            0.5: (704, 1408),
            0.52: (704, 1344),
            0.5581: (768, 1376),
            0.5625: (720, 1280),
            0.5647: (768, 1360),
            0.57: (768, 1344),
            0.6: (768, 1280),
            0.622: (816, 1312),
            0.625: (800, 1280),
            0.65: (832, 1280),
            0.6582: (832, 1264),
            0.6667: (832, 1248),
            0.6709: (848, 1264),
            0.68: (832, 1216),
            0.7013: (864, 1232),
            0.72: (832, 1152),
            0.7467: (896, 1200),
            0.75: (864, 1152),
            0.7568: (896, 1184),
            0.78: (896, 1152),
            0.8: (896, 1120),
            0.8056: (928, 1152),
            0.82: (896, 1088),
            0.88: (960, 1088),
            0.94: (960, 1024),
            0.9846: (1024, 1040),
            1.0: (1024, 1024),
            1.07: (1024, 960),
            1.13: (1088, 960),
            1.21: (1088, 896),
            1.2414: (1152, 928),
            1.25: (1120, 896),
            1.2807: (1168, 912),
            1.29: (1152, 896),
            1.3333: (1152, 864),
            1.3393: (1200, 896),
            1.38: (1152, 832),
            1.46: (1216, 832),
            1.4906: (1264, 848),
            1.5: (1248, 832),
            1.6: (1280, 800),
            1.67: (1280, 768),
            1.75: (1344, 768),
            1.7708: (1360, 768),
            1.7778: (1280, 720),
            2.0: (1408, 704),
            2.09: (1472, 704),
            2.4: (1536, 640),
            2.5: (1600, 640),
            2.89: (1664, 576),
            3.0: (1728, 576),
            3.11: (1792, 576),
            3.62: (1856, 512),
            3.75: (1920, 512),
            3.88: (1984, 512),
            4.0: (2048, 512),
        },
    }
    resolution = min(buckets, key=lambda value: (abs(value - resolution), value))
    table = buckets[resolution]
    return table[min(table, key=lambda ratio: abs(ratio - height / width))]
