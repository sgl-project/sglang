import dataclasses

from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams


@dataclasses.dataclass
class LTX2SamplingParams(SamplingParams):
    """Sampling parameters for LTX-2."""

    # Match the reference defaults used by ltx-pipelines (one-stage).
    # See: LTX-2/packages/ltx-pipelines/src/ltx_pipelines/utils/constants.py
    seed: int = 10

    # Video parameters
    height: int = 512
    width: int = 768
    num_frames: int = 121
    fps: int = 24

    # Audio specific
    generate_audio: bool = True

    # Denoising parameters
    guidance_scale: float = 4.0
    num_inference_steps: int = 40

    # Match ltx-pipelines default negative prompt (covers video + audio artifacts).
    negative_prompt: str = (
        "blurry, out of focus, overexposed, underexposed, low contrast, washed out colors, excessive noise, "
        "grainy texture, poor lighting, flickering, motion blur, distorted proportions, unnatural skin tones, "
        "deformed facial features, asymmetrical face, missing facial features, extra limbs, disfigured hands, "
        "wrong hand count, artifacts around text, inconsistent perspective, camera shake, incorrect depth of "
        "field, background too sharp, background clutter, distracting reflections, harsh shadows, inconsistent "
        "lighting direction, color banding, cartoonish rendering, 3D CGI look, unrealistic materials, uncanny "
        "valley effect, incorrect ethnicity, wrong gender, exaggerated expressions, wrong gaze direction, "
        "mismatched lip sync, silent or muted audio, distorted voice, robotic voice, echo, background noise, "
        "off-sync audio, incorrect dialogue, added dialogue, repetitive speech, jittery movement, awkward "
        "pauses, incorrect timing, unnatural transitions, inconsistent framing, tilted camera, flat lighting, "
        "inconsistent tone, cinematic oversaturation, stylized filters, or AI artifacts."
    )
