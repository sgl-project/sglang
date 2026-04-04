from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

DEFAULT_LTX_REPO_ROOTS = ["/tmp/LTX-2-official", "/tmp/LTX-2"]
LTX_REPO_ROOT = Path(
    os.environ.get(
        "LTX_REPO_ROOT",
        next(
            (p for p in DEFAULT_LTX_REPO_ROOTS if Path(p).exists()),
            DEFAULT_LTX_REPO_ROOTS[0],
        ),
    )
)
CHECKPOINT_GLOB = (
    "/root/.cache/huggingface/hub/models--Lightricks--LTX-2.3/snapshots/*/"
    "ltx-2.3-22b-dev.safetensors"
)
GEMMA_ROOT_GLOB = (
    "/root/.cache/huggingface/hub/models--Lightricks--LTX-2/snapshots/*/"
    "tokenizer/tokenizer.model"
)
PROMPT = os.environ.get("LTX23_PROMPT", "A beautiful sunset over the ocean")
IMAGE_PATH = os.environ.get("LTX23_IMAGE_PATH", "/tmp/ltx23_i2v_input_sunset.png")
DUMP_PATH = Path(
    os.environ.get("LTX23_OFFICIAL_FINAL_LATENTS", "/tmp/ltx23_official_i2v_final.pt")
)
STREAMING_PREFETCH_COUNT = int(os.environ.get("LTX23_STREAMING_PREFETCH_COUNT", "1"))


def resolve_single_path(pattern: str) -> Path:
    matches = sorted(Path("/").glob(pattern.lstrip("/")))
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one match for {pattern}, got {matches}")
    return matches[0]


@torch.inference_mode()
def main() -> None:
    checkpoint_path = resolve_single_path(CHECKPOINT_GLOB)
    gemma_root = resolve_single_path(GEMMA_ROOT_GLOB).parent.parent

    sys.path.insert(0, str(LTX_REPO_ROOT / "packages/ltx-core/src"))
    sys.path.insert(0, str(LTX_REPO_ROOT / "packages/ltx-pipelines/src"))

    from ltx_core.components.guiders import (
        MultiModalGuiderParams,
        create_multimodal_guider_factory,
    )
    from ltx_core.components.noisers import GaussianNoiser
    from ltx_core.components.schedulers import LTX2Scheduler
    from ltx_pipelines.utils.args import ImageConditioningInput
    from ltx_pipelines.utils.blocks import (
        DiffusionStage,
        ImageConditioner,
        PromptEncoder,
        generate_enhanced_prompt,
        gpu_model,
    )
    from ltx_pipelines.utils.constants import DEFAULT_NEGATIVE_PROMPT, LTX_2_3_PARAMS
    from ltx_pipelines.utils.denoisers import FactoryGuidedDenoiser
    from ltx_pipelines.utils.helpers import (
        combined_image_conditionings,
        post_process_latent,
    )
    from ltx_pipelines.utils.samplers import _step_state
    from ltx_pipelines.utils.types import ModalitySpec

    def _patched_prompt_encoder_call(
        self,
        prompts,
        *,
        enhance_first_prompt=False,
        enhance_prompt_image=None,
        enhance_prompt_seed=42,
        streaming_prefetch_count=None,
    ):
        text_encoder = self._text_encoder_builder.build(
            device=torch.device("cpu"),
            dtype=self._dtype,
        ).eval()
        if enhance_first_prompt:
            prompts = list(prompts)
            prompts[0] = generate_enhanced_prompt(
                text_encoder,
                prompts[0],
                enhance_prompt_image,
                seed=enhance_prompt_seed,
            )
        raw_outputs = [text_encoder.encode(p) for p in prompts]
        del text_encoder

        torch.cuda.empty_cache()

        with gpu_model(
            self._embeddings_processor_builder.build(
                device=self._device,
                dtype=self._dtype,
            )
            .to(self._device)
            .eval()
        ) as embeddings_processor:
            return [
                embeddings_processor.process_hidden_states(
                    [h.to(device=self._device, dtype=self._dtype) for h in hs],
                    mask.to(device=self._device),
                )
                for hs, mask in raw_outputs
            ]

    PromptEncoder.__call__ = _patched_prompt_encoder_call

    dtype = torch.bfloat16
    device = torch.device("cuda")
    params = LTX_2_3_PARAMS

    generator = torch.Generator(device=device).manual_seed(params.seed)
    noiser = GaussianNoiser(generator=generator)

    prompt_encoder = PromptEncoder(
        checkpoint_path=str(checkpoint_path),
        gemma_root=str(gemma_root),
        dtype=dtype,
        device=device,
    )
    image_conditioner = ImageConditioner(
        checkpoint_path=str(checkpoint_path),
        dtype=dtype,
        device=device,
    )
    diffusion_stage = DiffusionStage(
        checkpoint_path=str(checkpoint_path),
        dtype=dtype,
        device=device,
        loras=tuple(),
        torch_compile=False,
    )

    ctx_p, ctx_n = prompt_encoder(
        [PROMPT, DEFAULT_NEGATIVE_PROMPT],
        streaming_prefetch_count=STREAMING_PREFETCH_COUNT,
    )

    images = [ImageConditioningInput(path=IMAGE_PATH, frame_idx=0, strength=1.0)]
    stage_1_conditionings = image_conditioner(
        lambda enc: combined_image_conditionings(
            images=images,
            height=params.stage_1_height,
            width=params.stage_1_width,
            video_encoder=enc,
            dtype=dtype,
            device=device,
        )
    )

    sigmas = LTX2Scheduler().execute(steps=params.num_inference_steps).to(
        dtype=torch.float32, device=device
    )
    video_guider_factory = create_multimodal_guider_factory(
        params=MultiModalGuiderParams(
            cfg_scale=params.video_guider_params.cfg_scale,
            stg_scale=params.video_guider_params.stg_scale,
            rescale_scale=params.video_guider_params.rescale_scale,
            modality_scale=params.video_guider_params.modality_scale,
            skip_step=params.video_guider_params.skip_step,
            stg_blocks=list(params.video_guider_params.stg_blocks),
        ),
        negative_context=ctx_n.video_encoding,
    )
    audio_guider_factory = create_multimodal_guider_factory(
        params=MultiModalGuiderParams(
            cfg_scale=params.audio_guider_params.cfg_scale,
            stg_scale=params.audio_guider_params.stg_scale,
            rescale_scale=params.audio_guider_params.rescale_scale,
            modality_scale=params.audio_guider_params.modality_scale,
            skip_step=params.audio_guider_params.skip_step,
            stg_blocks=list(params.audio_guider_params.stg_blocks),
        ),
        negative_context=ctx_n.audio_encoding,
    )

    denoiser = FactoryGuidedDenoiser(
        v_context=ctx_p.video_encoding,
        a_context=ctx_p.audio_encoding,
        video_guider_factory=video_guider_factory,
        audio_guider_factory=audio_guider_factory,
    )

    final_video_state = None
    final_audio_state = None

    def dump_loop(sigmas, video_state, audio_state, stepper, transformer, denoiser):
        nonlocal final_video_state, final_audio_state
        for step_idx, _ in enumerate(sigmas[:-1]):
            denoised_video, denoised_audio = denoiser(
                transformer, video_state, audio_state, sigmas, step_idx
            )
            if video_state is not None and denoised_video is not None:
                denoised_video = post_process_latent(
                    denoised_video,
                    video_state.denoise_mask,
                    video_state.clean_latent,
                )
            if audio_state is not None and denoised_audio is not None:
                denoised_audio = post_process_latent(
                    denoised_audio,
                    audio_state.denoise_mask,
                    audio_state.clean_latent,
                )
            video_state = _step_state(
                video_state, denoised_video, stepper, sigmas, step_idx
            )
            audio_state = _step_state(
                audio_state, denoised_audio, stepper, sigmas, step_idx
            )

        final_video_state = video_state
        final_audio_state = audio_state
        return video_state, audio_state

    diffusion_stage(
        denoiser=denoiser,
        sigmas=sigmas,
        noiser=noiser,
        width=params.stage_1_width,
        height=params.stage_1_height,
        frames=params.num_frames,
        fps=params.frame_rate,
        video=ModalitySpec(
            context=ctx_p.video_encoding,
            conditionings=stage_1_conditionings,
        ),
        audio=ModalitySpec(context=ctx_p.audio_encoding),
        loop=dump_loop,
        streaming_prefetch_count=STREAMING_PREFETCH_COUNT,
        max_batch_size=1,
    )

    if final_video_state is None:
        raise RuntimeError("Official TI2V run did not produce final video latents")

    DUMP_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "video_latent_after": final_video_state.latent.detach().cpu(),
            "audio_latent_after": (
                None
                if final_audio_state is None
                else final_audio_state.latent.detach().cpu()
            ),
        },
        DUMP_PATH,
    )
    print(DUMP_PATH)


if __name__ == "__main__":
    main()
