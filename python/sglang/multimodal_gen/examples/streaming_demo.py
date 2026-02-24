"""
Minimal demo: real-time diffusion streaming.

Intermediate latents are published via ZMQ PUB/SUB after every N denoising steps,
so clients receive progress updates *during* generation rather than waiting for the
final output.

Usage:
    python streaming_demo.py --model Wan-AI/Wan2.2-T2V-14B-Diffusers \\
        --prompt "A cat playing in the garden" \\
        --stream-every-n-steps 5
"""

import argparse

from sglang.multimodal_gen.runtime.entrypoints.diffusion_generator import DiffGenerator


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
    args = parser.parse_args()

    gen = DiffGenerator.from_pretrained(model_path=args.model)

    print(f"Starting streaming generation for: {args.prompt!r}")
    for partial in gen.generate_streaming(
        sampling_params_kwargs={"prompt": args.prompt},
        stream_every_n_steps=args.stream_every_n_steps,
    ):
        latent_info = (
            f"  latent_shape={list(partial.latents.shape)}"
            if partial.latents is not None
            else ""
        )
        final_tag = "  [FINAL]" if partial.is_final else ""
        print(
            f"  [step {partial.step_index:3d}/{partial.total_steps}]"
            f"  t={partial.timestep}"
            f"{latent_info}{final_tag}"
        )

    print("Done.")


if __name__ == "__main__":
    main()
