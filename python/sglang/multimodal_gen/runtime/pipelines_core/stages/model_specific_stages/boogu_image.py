# SPDX-License-Identifier: Apache-2.0
from typing import Any

import PIL.Image
import torch

from sglang.multimodal_gen.configs.pipeline_configs.boogu_image import (
    BOOGU_SYSTEM_PROMPT_DROP,
)
from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.managers.memory_managers.component_manager import (
    ComponentUse,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.boogu_image_processor import (
    BooguImageProcessor,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.text_encoding import (
    TextEncodingStage,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.runtime.utils.precision import (
    get_module_dtype,
    resolve_precision,
)

logger = init_logger(__name__)

_MAX_VLM_INPUT_PIL_PIXELS = 384 * 384
_MAX_VLM_INPUT_PIL_SIDE_LENGTH = 384 * 2
_MAX_INPUT_IMAGE_PIXELS = 2048 * 2048
_MAX_INPUT_IMAGE_SIDE_LENGTH = 2048 * 2


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.retrieve_latents
def _retrieve_latents(
    encoder_output: torch.Tensor,
    generator: torch.Generator | None = None,
    sample_mode: str = "sample",
):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    raise AttributeError("Could not access latents of provided encoder_output")


def _reference_vae_generator(batch: Req) -> torch.Generator | None:
    """Pick the generator that seeds the reference-image VAE sample.

    Req.generator is typed torch.Generator | list[torch.Generator] | None, and
    input validation always replaces it with a one-entry-per-sample list. The
    reference image is encoded once per request, so take the first entry -- the
    request seed then reaches latent_dist.sample() instead of falling back to
    the global RNG, which would also let CFG-parallel ranks disagree.
    """
    generator = batch.generator
    if isinstance(generator, torch.Generator):
        return generator
    if isinstance(generator, list) and generator:
        return generator[0]
    return None


class BooguImageEncodingStage(TextEncodingStage):
    def __init__(self, text_encoders, tokenizers, vae) -> None:
        super().__init__(text_encoders, tokenizers)
        self.vae = vae
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = BooguImageProcessor(
            vae_scale_factor=self.vae_scale_factor * 2, do_resize=True
        )

    def _forward_text_encoder(self, text_encoder, encoder_forward_kwargs):
        inner = text_encoder.model
        with set_forward_context(current_timestep=0, attn_metadata=None):
            return inner(**encoder_forward_kwargs)

    def component_uses(
        self, server_args: ServerArgs, stage_name: str | None = None
    ) -> list[ComponentUse]:
        uses = super().component_uses(server_args, stage_name)
        vae_dtype = resolve_precision(
            server_args, "vae", precision_attr="vae_precision"
        )
        uses.append(
            ComponentUse(
                stage_name=self._component_stage_name(stage_name),
                component_name="vae",
                target_dtype=vae_dtype,
            )
        )
        return uses

    @torch.no_grad()
    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        ref_images = batch.condition_image
        if not ref_images:
            return super().forward(batch, server_args)
        if len(ref_images) != 1:
            raise ValueError(
                "Boogu-Image conditions on exactly one reference image, got "
                f"{len(ref_images)}"
            )

        raw_pil = ref_images[0].convert("RGB")
        prompts = batch.prompt if isinstance(batch.prompt, list) else [batch.prompt]
        device = get_local_torch_device()

        ref_latent = self._encode_reference_vae(raw_pil, batch, server_args, device)

        feats, mask = self._encode_reference_vlm(raw_pil, prompts, batch, device)
        ref_latent = ref_latent.to(feats.dtype)

        batch.prompt_embeds = [feats]
        batch.prompt_attention_mask = [mask]
        batch.prompt_embeds_mask = [mask]
        batch.ref_image_hidden_states = [[ref_latent] for _ in prompts]
        batch.image_latent = None

        if batch.do_classifier_free_guidance:
            assert isinstance(batch.negative_prompt, str)
            (
                neg_embeds_list,
                neg_masks_list,
                neg_pooler_embeds_list,
                neg_embeds_masks_list,
                neg_seq_lens_list,
            ) = self.get_or_compute_negative_text_embedding(batch, server_args, [0])
            self._append_negative_text_outputs(
                batch,
                [feats],
                neg_embeds_list,
                neg_masks_list,
                neg_pooler_embeds_list,
                neg_embeds_masks_list,
                neg_seq_lens_list,
            )

        return batch

    def _encode_reference_vae(
        self, raw_pil: PIL.Image.Image, batch: Req, server_args: ServerArgs, device
    ) -> torch.Tensor:
        preprocessed = self.image_processor.preprocess(
            raw_pil,
            max_pixels=_MAX_INPUT_IMAGE_PIXELS,
            max_side_length=_MAX_INPUT_IMAGE_SIDE_LENGTH,
        )
        batch.height = int(preprocessed.shape[-2])
        batch.width = int(preprocessed.shape[-1])

        scaling_factor, shift_factor = (
            server_args.pipeline_config.get_decode_scale_and_shift(
                device, None, self.vae
            )
        )
        vae_generator = _reference_vae_generator(batch)

        with self.use_declared_component(component_name="vae", module=self.vae) as vae:
            assert vae is not None
            self.vae = vae
            vae_dtype = get_module_dtype(vae, torch.float32)
            image = preprocessed.to(device=device, dtype=vae_dtype)
            latent = _retrieve_latents(
                vae.encode(image), generator=vae_generator, sample_mode="sample"
            )

        latent = self._apply_encode_scale_shift(latent, scaling_factor, shift_factor)
        return latent.squeeze(0)

    @staticmethod
    def _apply_encode_scale_shift(
        latent: torch.Tensor,
        scaling_factor: float | torch.Tensor | None,
        shift_factor: float | torch.Tensor | None,
    ) -> torch.Tensor:
        if shift_factor is not None:
            latent = latent - shift_factor
        if scaling_factor is not None:
            latent = latent * scaling_factor
        return latent

    def _encode_reference_vlm(
        self, raw_pil: PIL.Image.Image, prompts: list[str], batch: Req, device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        new_h, new_w = self.image_processor.get_new_height_width(
            raw_pil,
            None,
            None,
            _MAX_VLM_INPUT_PIL_PIXELS,
            _MAX_VLM_INPUT_PIL_SIDE_LENGTH,
        )
        vlm_pil = raw_pil.resize((new_w, new_h))

        messages_batch = [self._build_edit_messages(p, vlm_pil) for p in prompts]
        vlm_inputs = self.tokenizers[0].apply_chat_template(
            messages_batch,
            padding="longest",
            max_length=batch.max_sequence_length,
            truncation=False,
            padding_side="right",
            return_tensors="pt",
            tokenize=True,
            return_dict=True,
        )
        vlm_inputs = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in vlm_inputs.items()
        }
        vlm_inputs["output_hidden_states"] = True
        vlm_inputs["return_dict"] = True

        outputs = self._forward_text_encoder(self.text_encoders[0], vlm_inputs)
        feats = outputs.last_hidden_state
        mask = vlm_inputs["attention_mask"]
        return feats, mask

    def _build_edit_messages(
        self, instruction: str, pil_image: PIL.Image.Image
    ) -> list[dict[str, Any]]:
        return [
            {
                "role": "system",
                "content": [{"type": "text", "text": BOOGU_SYSTEM_PROMPT_DROP}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_image},
                    {"type": "text", "text": instruction or ""},
                ],
            },
        ]
