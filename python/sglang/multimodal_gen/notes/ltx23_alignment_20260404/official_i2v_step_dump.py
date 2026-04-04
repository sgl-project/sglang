from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

DEFAULT_LTX_REPO_ROOTS = ["/tmp/LTX-2-official", "/tmp/LTX-2"]
LTX_REPO_ROOT = Path(
    os.environ.get(
        "LTX_REPO_ROOT",
        next((p for p in DEFAULT_LTX_REPO_ROOTS if Path(p).exists()), DEFAULT_LTX_REPO_ROOTS[0]),
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
DUMP_PATH = Path(os.environ.get("LTX23_OFFICIAL_STEP_DUMP", "/tmp/ltx23_official_i2v_step0.pt"))
TARGET_STEP = int(os.environ.get("LTX23_TARGET_STEP", "0"))
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

    from ltx_core.components.guiders import MultiModalGuiderParams, create_multimodal_guider_factory
    from ltx_core.components.noisers import GaussianNoiser
    from ltx_core.components.schedulers import LTX2Scheduler
    from ltx_core.guidance.perturbations import (
        BatchedPerturbationConfig,
        Perturbation,
        PerturbationConfig,
        PerturbationType,
    )
    from ltx_pipelines.utils.args import ImageConditioningInput
    from ltx_pipelines.utils.blocks import DiffusionStage, ImageConditioner, PromptEncoder
    from ltx_pipelines.utils.constants import DEFAULT_NEGATIVE_PROMPT, LTX_2_3_PARAMS
    from ltx_pipelines.utils.helpers import (
        combined_image_conditionings,
        modality_from_latent_state,
        post_process_latent,
    )
    from ltx_pipelines.utils.blocks import generate_enhanced_prompt, gpu_model
    from ltx_pipelines.utils import denoisers as denoisers_mod
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

    pass_dump: dict[str, torch.Tensor | None] = {}

    def _guided_denoise_with_dump(  # noqa: PLR0913
        transformer,
        video_state,
        audio_state,
        sigma,
        video_guider,
        audio_guider,
        v_context,
        a_context,
        *,
        last_denoised_video,
        last_denoised_audio,
        step_index,
    ):
        v_skip = video_guider.should_skip_step(step_index)
        a_skip = audio_guider.should_skip_step(step_index)

        if v_skip and a_skip:
            return last_denoised_video, last_denoised_audio

        _pass = tuple[str, torch.Tensor | None, torch.Tensor | None, PerturbationConfig]
        passes: list[_pass] = [("cond", v_context, a_context, PerturbationConfig.empty())]

        if video_guider.do_unconditional_generation() or audio_guider.do_unconditional_generation():
            v_neg = video_guider.negative_context if video_guider.negative_context is not None else v_context
            a_neg = audio_guider.negative_context if audio_guider.negative_context is not None else a_context
            passes.append(("uncond", v_neg, a_neg, PerturbationConfig.empty()))

        stg_perturbations: list[Perturbation] = []
        if video_guider.do_perturbed_generation():
            stg_perturbations.append(
                Perturbation(
                    type=PerturbationType.SKIP_VIDEO_SELF_ATTN,
                    blocks=video_guider.params.stg_blocks,
                )
            )
        if audio_guider.do_perturbed_generation():
            stg_perturbations.append(
                Perturbation(
                    type=PerturbationType.SKIP_AUDIO_SELF_ATTN,
                    blocks=audio_guider.params.stg_blocks,
                )
            )
        if stg_perturbations:
            passes.append(("ptb", v_context, a_context, PerturbationConfig(stg_perturbations)))

        if video_guider.do_isolated_modality_generation() or audio_guider.do_isolated_modality_generation():
            passes.append(
                (
                    "mod",
                    v_context,
                    a_context,
                    PerturbationConfig(
                        [
                            Perturbation(type=PerturbationType.SKIP_A2V_CROSS_ATTN, blocks=None),
                            Perturbation(type=PerturbationType.SKIP_V2A_CROSS_ATTN, blocks=None),
                        ]
                    ),
                )
            )

        pass_names = [name for name, _, _, _ in passes]
        ptb_configs = [ptb for _, _, _, ptb in passes]
        n = len(passes)

        def _batched_sigma(state):
            return sigma.expand(state.latent.shape[0] * n)

        def _repeat_state(state, repeats: int):
            def _repeat(t: torch.Tensor) -> torch.Tensor:
                repeat_shape = [1] * t.dim()
                repeat_shape[0] = repeats
                return t.repeat(repeat_shape)

            return state.__class__(
                latent=_repeat(state.latent),
                denoise_mask=_repeat(state.denoise_mask),
                positions=_repeat(state.positions),
                clean_latent=_repeat(state.clean_latent),
                attention_mask=(
                    _repeat(state.attention_mask)
                    if state.attention_mask is not None
                    else None
                ),
            )

        batched_video = None
        if video_state is not None:
            batched_video = modality_from_latent_state(
                _repeat_state(video_state, n),
                torch.cat([vc for _, vc, _, _ in passes], dim=0),
                _batched_sigma(video_state),
                enabled=not v_skip,
            )

        batched_audio = None
        if audio_state is not None:
            batched_audio = modality_from_latent_state(
                _repeat_state(audio_state, n),
                torch.cat([ac for _, _, ac, _ in passes], dim=0),
                _batched_sigma(audio_state),
                enabled=not a_skip,
            )

        all_v, all_a = transformer(
            video=batched_video,
            audio=batched_audio,
            perturbations=BatchedPerturbationConfig(ptb_configs),
        )

        splits_v = list(all_v.chunk(n)) if all_v is not None else [0.0] * n
        splits_a = list(all_a.chunk(n)) if all_a is not None else [0.0] * n
        result = dict(zip(pass_names, zip(splits_v, splits_a, strict=True), strict=True))

        cond_v, cond_a = result["cond"]
        uncond_v, uncond_a = result.get("uncond", (None, None))
        ptb_v, ptb_a = result.get("ptb", (None, None))
        mod_v, mod_a = result.get("mod", (None, None))

        if step_index == TARGET_STEP:
            pass_dump.clear()
            pass_dump.update(
                {
                    "video_cond": cond_v,
                    "video_uncond": uncond_v,
                    "video_ptb": ptb_v,
                    "video_mod": mod_v,
                    "audio_cond": cond_a,
                    "audio_uncond": uncond_a,
                    "audio_ptb": ptb_a,
                    "audio_mod": mod_a,
                }
            )

        video_uncond_input = 0.0 if uncond_v is None else uncond_v
        video_ptb_input = 0.0 if ptb_v is None else ptb_v
        video_mod_input = 0.0 if mod_v is None else mod_v
        audio_uncond_input = 0.0 if uncond_a is None else uncond_a
        audio_ptb_input = 0.0 if ptb_a is None else ptb_a
        audio_mod_input = 0.0 if mod_a is None else mod_a

        denoised_video = (
            last_denoised_video
            if v_skip
            else video_guider.calculate(
                cond_v, video_uncond_input, video_ptb_input, video_mod_input
            )
        )
        denoised_audio = (
            last_denoised_audio
            if a_skip
            else audio_guider.calculate(
                cond_a, audio_uncond_input, audio_ptb_input, audio_mod_input
            )
        )
        return denoised_video, denoised_audio

    denoisers_mod._guided_denoise = _guided_denoise_with_dump

    from ltx_pipelines.utils.denoisers import FactoryGuidedDenoiser

    denoiser = FactoryGuidedDenoiser(
        v_context=ctx_p.video_encoding,
        a_context=ctx_p.audio_encoding,
        video_guider_factory=video_guider_factory,
        audio_guider_factory=audio_guider_factory,
    )

    dumped = False

    def dump_loop(sigmas, video_state, audio_state, stepper, transformer, denoiser):
        nonlocal dumped
        for step_idx, _ in enumerate(sigmas[:-1]):
            denoised_video, denoised_audio = denoiser(
                transformer, video_state, audio_state, sigmas, step_idx
            )
            if video_state is not None and denoised_video is not None:
                denoised_video = post_process_latent(
                    denoised_video, video_state.denoise_mask, video_state.clean_latent
                )
            if audio_state is not None and denoised_audio is not None:
                denoised_audio = post_process_latent(
                    denoised_audio, audio_state.denoise_mask, audio_state.clean_latent
                )

            video_next = _step_state(video_state, denoised_video, stepper, sigmas, step_idx)
            audio_next = _step_state(audio_state, denoised_audio, stepper, sigmas, step_idx)

            if step_idx == TARGET_STEP:
                image_latent = None
                if video_state is not None:
                    cond_mask = (video_state.denoise_mask.squeeze(-1) < 1.0)
                    num_img_tokens = int(cond_mask[0].sum().item())
                    image_latent = video_state.clean_latent[:, :num_img_tokens]
                DUMP_PATH.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "step_index": step_idx,
                        "sigma": float(sigmas[step_idx].item()),
                        "prompt_embeds": ctx_p.video_encoding.detach().cpu(),
                        "audio_prompt_embeds": ctx_p.audio_encoding.detach().cpu(),
                        "negative_prompt_embeds": ctx_n.video_encoding.detach().cpu(),
                        "negative_audio_prompt_embeds": ctx_n.audio_encoding.detach().cpu(),
                        "prompt_attention_mask": ctx_p.attention_mask.detach().cpu(),
                        "negative_attention_mask": ctx_n.attention_mask.detach().cpu(),
                        "video_latent_before": None if video_state is None else video_state.latent.detach().cpu(),
                        "video_denoised": None if denoised_video is None else denoised_video.detach().cpu(),
                        "video_latent_after": None if video_next is None else video_next.latent.detach().cpu(),
                        "audio_latent_before": None if audio_state is None else audio_state.latent.detach().cpu(),
                        "audio_denoised": None if denoised_audio is None else denoised_audio.detach().cpu(),
                        "audio_latent_after": None if audio_next is None else audio_next.latent.detach().cpu(),
                        "video_cond": None if pass_dump.get("video_cond") is None else pass_dump["video_cond"].detach().cpu(),
                        "video_uncond": None if pass_dump.get("video_uncond") is None else pass_dump["video_uncond"].detach().cpu(),
                        "video_ptb": None if pass_dump.get("video_ptb") is None else pass_dump["video_ptb"].detach().cpu(),
                        "video_mod": None if pass_dump.get("video_mod") is None else pass_dump["video_mod"].detach().cpu(),
                        "audio_cond": None if pass_dump.get("audio_cond") is None else pass_dump["audio_cond"].detach().cpu(),
                        "audio_uncond": None if pass_dump.get("audio_uncond") is None else pass_dump["audio_uncond"].detach().cpu(),
                        "audio_ptb": None if pass_dump.get("audio_ptb") is None else pass_dump["audio_ptb"].detach().cpu(),
                        "audio_mod": None if pass_dump.get("audio_mod") is None else pass_dump["audio_mod"].detach().cpu(),
                        "video_clean_latent": None if video_state is None else video_state.clean_latent.detach().cpu(),
                        "video_denoise_mask": None if video_state is None else video_state.denoise_mask.detach().cpu(),
                        "image_latent": None if image_latent is None else image_latent.detach().cpu(),
                    },
                    DUMP_PATH,
                )
                dumped = True
                return video_next, audio_next

            video_state, audio_state = video_next, audio_next

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

    if not dumped:
        raise RuntimeError(f"Target step {TARGET_STEP} was not dumped")
    print(DUMP_PATH)


if __name__ == "__main__":
    main()
