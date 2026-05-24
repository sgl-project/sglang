#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
End-to-end verification script for LongCat-Video T2V generation.

This script loads the LongCat-Video model via DiffGenerator, generates a short
video clip, and validates that the output file exists and is non-empty.

Usage:
    python -m sglang.multimodal_gen.test.e2e.verify_longcat_video_t2v \
        --model-path /path/to/LongCat-Video/weights/LongCat-Video

Requirements:
    - Single GPU (LongCat-Video MVP limitation)
    - num_frames must satisfy (num_frames - 1) % 4 == 0
    - Supported resolutions: (832, 480) or (480, 832)
"""

import argparse
import os
import sys
import time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="End-to-end verification for LongCat-Video T2V"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the LongCat-Video model weights directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save generated video (default: ./longcat_video_e2e_output)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="A serene mountain landscape with flowing clouds and a gentle breeze.",
        help="Text prompt for video generation",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Video height in pixels (default: 480)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=832,
        help="Video width in pixels (default: 832)",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=25,
        help="Number of frames to generate (must satisfy (n-1) %% 4 == 0, default: 25)",
    )
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=10,
        help="Number of denoising steps (default: 10)",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=4.0,
        help="Classifier-free guidance scale (default: 4.0)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """Validate command-line arguments against LongCat-Video constraints."""
    # Check num_frames constraint: (num_frames - 1) % 4 == 0
    if (args.num_frames - 1) % 4 != 0:
        valid_examples = [n for n in range(25, 94) if (n - 1) % 4 == 0]
        print(
            f"[ERROR] num_frames={args.num_frames} does not satisfy "
            f"(num_frames - 1) % 4 == 0."
        )
        print(f"        Valid values include: {valid_examples}")
        sys.exit(1)

    # Check resolution constraint
    supported_resolutions = [(832, 480), (480, 832)]
    if (args.width, args.height) not in supported_resolutions:
        print(
            f"[ERROR] Resolution ({args.width}, {args.height}) is not supported."
        )
        print(f"        Supported resolutions: {supported_resolutions}")
        sys.exit(1)

    # Check model path exists
    if not os.path.isdir(args.model_path):
        print(f"[ERROR] Model path does not exist: {args.model_path}")
        sys.exit(1)


def main() -> None:
    args = parse_args()
    validate_args(args)

    output_dir = args.output_dir or os.path.join(
        os.getcwd(), "longcat_video_e2e_output"
    )
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("LongCat-Video T2V End-to-End Verification")
    print("=" * 60)
    print(f"  Model path:          {args.model_path}")
    print(f"  Output directory:    {output_dir}")
    print(f"  Prompt:              {args.prompt}")
    print(f"  Resolution:          {args.width}x{args.height}")
    print(f"  Num frames:          {args.num_frames}")
    print(f"  Inference steps:     {args.num_inference_steps}")
    print(f"  Guidance scale:      {args.guidance_scale}")
    print(f"  Seed:                {args.seed}")
    print("=" * 60)

    # Import here to avoid slow import at argument-parsing time
    from sglang.multimodal_gen.runtime.entrypoints.diffusion_generator import (
        DiffGenerator,
    )

    gen = None
    try:
        # Initialize the generator (single GPU only for LongCat-Video)
        print("\n[INFO] Initializing DiffGenerator...")
        start_init = time.perf_counter()
        gen = DiffGenerator.from_pretrained(
            model_path=args.model_path,
            local_mode=True,
            num_gpus=1,
        )
        init_time = time.perf_counter() - start_init
        print(f"[INFO] Generator initialized in {init_time:.2f}s")

        # Generate video
        print("\n[INFO] Starting video generation...")
        start_gen = time.perf_counter()
        result = gen.generate(
            sampling_params_kwargs={
                "prompt": args.prompt,
                "height": args.height,
                "width": args.width,
                "num_frames": args.num_frames,
                "num_inference_steps": args.num_inference_steps,
                "guidance_scale": args.guidance_scale,
                "seed": args.seed,
                "output_path": output_dir,
                "output_file_name": "longcat_t2v_test",
                "save_output": True,
            }
        )
        gen_time = time.perf_counter() - start_gen
        print(f"[INFO] Generation completed in {gen_time:.2f}s")

        # Validate result — generate() returns a single GenerationResult for
        # a single prompt, or None if every request failed.
        if result is None:
            print("\n[FAIL] Generation returned None - no output produced.")
            sys.exit(1)

        # Narrow the type: single prompt → single GenerationResult
        from sglang.multimodal_gen.runtime.entrypoints.utils import GenerationResult

        if isinstance(result, list):
            r: GenerationResult = result[0]
        else:
            r = result

        output_path = r.output_file_path
        print(f"\n[INFO] Output file path: {output_path}")
        print(f"[INFO] Generation time:  {r.generation_time:.2f}s")
        print(f"[INFO] Peak memory:      {r.peak_memory_mb:.2f} MB")

        # Verify output file
        if output_path is None:
            print("\n[FAIL] output_file_path is None in GenerationResult.")
            sys.exit(1)

        if not os.path.exists(output_path):
            print(f"\n[FAIL] Output file does not exist: {output_path}")
            sys.exit(1)

        file_size = os.path.getsize(output_path)
        if file_size <= 0:
            print(f"\n[FAIL] Output file is empty (0 bytes): {output_path}")
            sys.exit(1)

        # Report success
        print("\n" + "=" * 60)
        print("[PASS] LongCat-Video T2V verification succeeded!")
        print("=" * 60)
        print(f"  Output file:   {output_path}")
        print(f"  File size:     {file_size / 1024:.1f} KB")
        print(f"  Init time:     {init_time:.2f}s")
        print(f"  Gen time:      {gen_time:.2f}s")
        print(f"  Total time:    {init_time + gen_time:.2f}s")
        if r.metrics:
            print(f"  Metrics:       {r.metrics}")
        print("=" * 60)

    except Exception as e:
        print(f"\n[FAIL] Verification failed with exception: {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    finally:
        if gen is not None:
            print("\n[INFO] Shutting down generator...")
            gen.shutdown()
            print("[INFO] Generator shut down successfully.")


if __name__ == "__main__":
    main()
