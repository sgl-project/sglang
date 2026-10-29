# SPDX-License-Identifier: Apache-2.0
"""Generate one Physis H3 video using Ulysses, timestep or block placement."""

import argparse

from sglang.multimodal_gen import DiffGenerator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--mode", choices=("off", "timestep", "block"), default="off")
    parser.add_argument("--degree", type=int, default=4)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--width", type=int, default=1344)
    parser.add_argument("--num-frames", type=int, default=158)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap-chunks", type=int, default=5)
    parser.add_argument("--output-path", default="outputs/physis_h3")
    parser.add_argument(
        "--prompt",
        default="A red fox walks slowly through a snowy pine forest. "
        "Soft morning sunlight, gentle falling snow, smooth cinematic tracking shot.",
    )
    args = parser.parse_args()
    parallel = args.mode != "off"
    generator = DiffGenerator.from_pretrained(
        model_path=args.model_path,
        backend="sglang",
        pipeline_config={"physis_bootstrap_chunks": args.bootstrap_chunks},
        num_gpus=args.degree + int(parallel),
        tp_size=1,
        sp_degree=1 if parallel else args.degree,
        ulysses_degree=1 if parallel else args.degree,
        ring_degree=1,
        causal_pipeline_mode=args.mode,
        causal_pipeline_parallel_size=args.degree if parallel else 1,
        performance_mode="manual",
        warmup_mode="off",
        attention_backend="torch_sdpa",
        encoder_parallel="replicate",
        text_encoder_cpu_offload=not parallel,
        dit_cpu_offload=False,
        vae_cpu_offload=False,
        output_path=args.output_path,
    )
    try:
        generator.generate(
            sampling_params_kwargs=dict(
                prompt=args.prompt,
                seed=args.seed,
                height=args.height,
                width=args.width,
                num_frames=args.num_frames,
                num_inference_steps=4,
                guidance_scale=1.0,
                fps=24,
                save_output=True,
                output_path=args.output_path,
            )
        )
    finally:
        generator.shutdown()


if __name__ == "__main__":
    main()
