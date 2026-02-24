"""
Minimal demo: real-time diffusion streaming.

Intermediate latents are published via ZMQ PUB/SUB after every N denoising steps,
so clients receive progress updates *during* generation rather than waiting for the
final output.

Usage:
    # Latents-only (progress monitoring, no decode overhead):
    python streaming_demo.py --model Wan-AI/Wan2.2-T2V-A14B-Diffusers \\
        --prompt "A cat playing in the garden" \\
        --stream-every-n-steps 5

    # Decode pixel-space frames at each step (image models / few-step video models):
    python streaming_demo.py --model black-forest-labs/FLUX.1-dev \\
        --prompt "A cat" --stream-every-n-steps 5 --decode-frames

    # Few-step video model (TurboWan):
    python streaming_demo.py --model Wan-AI/Wan2.2-T2V-14B-Diffusers \\
        --prompt "A cat" --num-inference-steps 4 \\
        --stream-every-n-steps 1 --decode-frames
"""

import argparse
import os

import torch

from sglang.multimodal_gen.runtime.entrypoints.diffusion_generator import DiffGenerator


def save_preview(frames: torch.Tensor, step: int, output_dir: str) -> None:
    """
    Save the first batch item as a preview image or video.

    Args:
        frames: Decoded pixel tensor, shape [C, T, H, W] (video) or [C, H, W] (image).
                Values are in [0, 1] float32.
        step: Denoising step index, used in the output filename.
        output_dir: Directory to save preview files.
    """
    try:
        from PIL import Image
    except ImportError:
        print("  [preview] PIL not available — skipping save (pip install Pillow)")
        return

    os.makedirs(output_dir, exist_ok=True)

    if frames.ndim == 3:
        # Image: [C, H, W]
        img_np = (
            (frames.permute(1, 2, 0).cpu().float().numpy() * 255)
            .clip(0, 255)
            .astype("uint8")
        )
        path = os.path.join(output_dir, f"preview_step_{step:04d}.png")
        Image.fromarray(img_np).save(path)
        print(f"  [preview] Saved image preview → {path}")
    elif frames.ndim == 4:
        # Video: [C, T, H, W] — save first frame as image
        first_frame = frames[:, 0]  # [C, H, W]
        img_np = (
            (first_frame.permute(1, 2, 0).cpu().float().numpy() * 255)
            .clip(0, 255)
            .astype("uint8")
        )
        path = os.path.join(output_dir, f"preview_step_{step:04d}_frame0.png")
        Image.fromarray(img_np).save(path)
        print(f"  [preview] Saved video preview (first frame) → {path}")
    else:
        print(
            f"  [preview] Unexpected frames shape {list(frames.shape)}, skipping save"
        )


def main():
    parser = argparse.ArgumentParser(description="Real-time diffusion streaming demo")
    parser.add_argument("--model", required=True, help="Model path or HF repo ID")
    parser.add_argument(
        "--prompt", default="A cat playing in the garden", help="Generation prompt"
    )
    parser.add_argument(
        "--stream-every-n-steps",
        type=int,
        default=5,
        help="Emit a partial result every N denoising steps (default: 5)",
    )
    parser.add_argument(
        "--decode-frames",
        action="store_true",
        help=(
            "Decode and save pixel-space preview frames at each streaming step. "
            "Adds VAE decode overhead per step (~100ms for images, ~2s for video). "
            "Practical for image models or few-step video models (e.g. TurboWan with 4 steps)."
        ),
    )
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=None,
        help="Number of denoising steps (overrides model default if set)",
    )
    parser.add_argument(
        "--preview-dir",
        default="streaming_previews",
        help="Directory to save preview images when --decode-frames is set (default: streaming_previews)",
    )
    args = parser.parse_args()

    gen = DiffGenerator.from_pretrained(model_path=args.model)

    decode_info = " (with pixel-space decode)" if args.decode_frames else ""
    print(f"Starting streaming generation{decode_info} for: {args.prompt!r}")

    sampling_kwargs = {"prompt": args.prompt}
    if args.num_inference_steps is not None:
        sampling_kwargs["num_inference_steps"] = args.num_inference_steps

    for partial in gen.generate_streaming(
        sampling_params_kwargs=sampling_kwargs,
        stream_every_n_steps=args.stream_every_n_steps,
        stream_decode_latents=args.decode_frames,
    ):
        latent_info = (
            f"  latent_shape={list(partial.latents.shape)}"
            if partial.latents is not None
            else ""
        )
        frames_info = (
            f"  frames_shape={list(partial.frames.shape)}"
            if partial.frames is not None
            else ""
        )
        final_tag = "  [FINAL]" if partial.is_final else ""
        print(
            f"  [step {partial.step_index:3d}/{partial.total_steps}]"
            f"  t={partial.timestep}"
            f"{latent_info}{frames_info}{final_tag}"
        )

        if partial.frames is not None:
            # frames shape: [B, C, T, H, W] (video) or [B, C, H, W] (image)
            # Save the first batch item as a preview
            save_preview(
                partial.frames[0], step=partial.step_index, output_dir=args.preview_dir
            )

    print("Done.")


if __name__ == "__main__":
    main()
