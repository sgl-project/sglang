#!/usr/bin/env python3
"""
Ground Truth Generation Script for Consistency Testing.

This script generates ground truth (GT) outputs for consistency tests.
GT files are saved to a staging directory (./gt_staging/) and must be
manually uploaded to the sgl-test-files repository.

IMPORTANT: GT correctness is controlled by Reviewer during PR review.
This script uses the exact same generation code path as the tests to ensure
consistency between GT generation and test execution.

Usage:
    # Generate GT for a specific case
    python generate_consistency_gt.py --case qwen_image_t2i --num-gpus 1

    # Generate GT for all 1-GPU cases
    python generate_consistency_gt.py --suite 1-gpu

    # Generate GT for all 2-GPU cases
    python generate_consistency_gt.py --suite 2-gpu

    # Generate GT for all cases (both 1-GPU and 2-GPU)
    python generate_consistency_gt.py --all

    # List available cases without generating
    python generate_consistency_gt.py --list

After generation, upload the files to:
    https://github.com/sgl-project/sgl-test-files/tree/main/images/consistency_gt/
"""

from __future__ import annotations

import argparse
import datetime
import sys
import time
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from openai import OpenAI

from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.test.server.consistency_utils import (
    DEFAULT_SEED,
    DEFAULT_SSIM_THRESHOLD_IMAGE,
    DEFAULT_SSIM_THRESHOLD_VIDEO,
    GT_METADATA_PATH,
    extract_key_frames_from_video,
    image_bytes_to_numpy,
    load_gt_metadata,
    save_frames_as_gt,
    save_gt_metadata,
)
from sglang.multimodal_gen.test.server.test_server_utils import (
    ServerManager,
    get_generate_fn,
)
from sglang.multimodal_gen.test.server.testcase_configs import (
    ONE_GPU_CASES_A,
    ONE_GPU_CASES_B,
    TWO_GPU_CASES_A,
    TWO_GPU_CASES_B,
    DiffusionTestCase,
)
from sglang.multimodal_gen.test.test_utils import get_dynamic_server_port

logger = init_logger(__name__)

# Staging directory for generated GT files (to be uploaded to sgl-test-files)
GT_STAGING_DIR = Path("./gt_staging")

# All test cases organized by GPU configuration
ALL_CASES = {
    "1-gpu": ONE_GPU_CASES_A + ONE_GPU_CASES_B,
    "2-gpu": TWO_GPU_CASES_A + TWO_GPU_CASES_B,
}

# Flatten for lookup by case ID
CASE_BY_ID = {case.id: case for cases in ALL_CASES.values() for case in cases}


def get_case_by_id(case_id: str) -> DiffusionTestCase | None:
    """Get a test case by its ID."""
    return CASE_BY_ID.get(case_id)


def list_cases():
    """Print all available test cases."""
    print("\n=== Available Test Cases ===\n")
    for suite_name, cases in ALL_CASES.items():
        print(f"Suite: {suite_name}")
        print("-" * 40)
        for case in cases:
            modality = case.server_args.modality
            model = case.server_args.model_path.split("/")[-1]
            lora = " [LoRA]" if case.server_args.lora_path else ""
            print(f"  {case.id:<35} ({modality}, {model}){lora}")
        print()


def generate_gt_for_case(
    case: DiffusionTestCase,
    port: int,
    metadata: dict,
    seed: int = DEFAULT_SEED,
) -> bool:
    """
    Generate ground truth for a single test case.

    Uses the exact same generation code path (get_generate_fn) as the tests
    to ensure GT and test outputs are generated identically.

    Args:
        case: The test case to generate GT for
        port: Port to use for the server
        metadata: GT metadata dict to update
        seed: Fixed seed for reproducibility (default: 1024)

    Returns:
        True if successful, False otherwise
    """
    case_id = case.id
    num_gpus = case.server_args.num_gpus
    is_video = case.server_args.modality == "video"

    logger.info(f"\n{'='*60}")
    logger.info(f"Generating GT for: {case_id} ({num_gpus}-GPU)")
    logger.info(f"{'='*60}")

    # Build extra args for server
    extra_args = f"--num-gpus {num_gpus}"
    if case.server_args.ulysses_degree is not None:
        extra_args += f" --ulysses-degree {case.server_args.ulysses_degree}"
    if case.server_args.ring_degree is not None:
        extra_args += f" --ring-degree {case.server_args.ring_degree}"
    if case.server_args.lora_path:
        extra_args += f" --lora-path {case.server_args.lora_path}"

    # Start server
    manager = ServerManager(
        model=case.server_args.model_path,
        port=port,
        wait_deadline=1200.0,
        extra_args=extra_args,
    )

    try:
        ctx = manager.start()
        client = OpenAI(
            api_key="sglang-anything",
            base_url=f"http://localhost:{ctx.port}/v1",
        )

        # Use the SAME generation function as tests (get_generate_fn)
        # This ensures GT is generated with identical code path
        generate_fn = get_generate_fn(
            model_path=case.server_args.model_path,
            modality=case.server_args.modality,
            sampling_params=case.sampling_params,
            seed=seed,
        )

        # Generate output using the same function as tests
        logger.info(f"Generating output with seed={seed}...")
        start_time = time.time()
        rid, content = generate_fn(case_id, client)
        elapsed = time.time() - start_time
        logger.info(f"Generation completed in {elapsed:.2f}s (rid={rid})")

        # Convert content to frames
        if is_video:
            frames = extract_key_frames_from_video(content)
        else:
            frames = [image_bytes_to_numpy(content)]

        logger.info(f"Extracted {len(frames)} frame(s)")

        # Save frames to staging directory (for upload to sgl-test-files)
        gt_dir = save_frames_as_gt(
            frames, case_id, num_gpus, is_video=is_video, output_dir=GT_STAGING_DIR
        )
        logger.info(f"GT saved to staging: {gt_dir}")

        # Update metadata
        threshold = (
            DEFAULT_SSIM_THRESHOLD_VIDEO if is_video else DEFAULT_SSIM_THRESHOLD_IMAGE
        )
        metadata["cases"][case_id] = {
            "num_gpus": num_gpus,
            "seed": seed,
            "ssim_threshold": threshold,
            "model_path": case.server_args.model_path,
            "modality": case.server_args.modality,
            "prompt": case.sampling_params.prompt,
            "resolution": case.sampling_params.output_size,
            "num_frames": len(frames),
            "key_frame_indices": (
                [0, "mid", "last"] if is_video and len(frames) == 3 else [0]
            ),
            "generated_at": datetime.datetime.now().isoformat(),
        }

        if case.server_args.lora_path:
            metadata["cases"][case_id]["lora_path"] = case.server_args.lora_path

        return True

    except Exception as e:
        logger.error(f"Failed to generate GT for {case_id}: {e}")
        import traceback

        traceback.print_exc()
        return False

    finally:
        try:
            ctx.cleanup()
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser(
        description="Generate ground truth for consistency testing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--case",
        type=str,
        help="Specific case ID to generate GT for",
    )
    parser.add_argument(
        "--suite",
        type=str,
        choices=["1-gpu", "2-gpu"],
        help="Generate GT for all cases in a suite",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Generate GT for all cases",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available cases without generating",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Server port (default: auto-detect)",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=None,
        help="Override number of GPUs (only with --case)",
    )

    args = parser.parse_args()

    # Handle --list
    if args.list:
        list_cases()
        return

    # Validate arguments
    if not any([args.case, args.suite, args.all]):
        parser.error("Please specify --case, --suite, or --all")

    if args.num_gpus is not None and args.case is None:
        parser.error("--num-gpus can only be used with --case")

    # Determine port
    port = args.port or get_dynamic_server_port()

    # Load existing metadata
    metadata = load_gt_metadata()
    metadata.setdefault("version", "1.0")
    metadata.setdefault("default_seed", DEFAULT_SEED)
    metadata.setdefault("default_ssim_threshold_image", DEFAULT_SSIM_THRESHOLD_IMAGE)
    metadata.setdefault("default_ssim_threshold_video", DEFAULT_SSIM_THRESHOLD_VIDEO)
    metadata.setdefault("cases", {})

    # Determine cases to generate
    cases_to_generate: list[DiffusionTestCase] = []

    if args.case:
        case = get_case_by_id(args.case)
        if case is None:
            print(f"Error: Case '{args.case}' not found.")
            print("Use --list to see available cases.")
            sys.exit(1)
        cases_to_generate = [case]
    elif args.suite:
        cases_to_generate = ALL_CASES[args.suite]
    elif args.all:
        for suite_cases in ALL_CASES.values():
            cases_to_generate.extend(suite_cases)

    # Generate GT for each case
    print(f"\n{'='*60}")
    print(f"Generating GT for {len(cases_to_generate)} case(s)")
    print(f"{'='*60}\n")

    success_count = 0
    fail_count = 0

    for i, case in enumerate(cases_to_generate, 1):
        print(f"\n[{i}/{len(cases_to_generate)}] Processing: {case.id}")

        if generate_gt_for_case(case, port, metadata):
            success_count += 1
            # Save metadata after each successful generation
            save_gt_metadata(metadata)
        else:
            fail_count += 1

    # Final summary
    print(f"\n{'='*60}")
    print("GT Generation Summary")
    print(f"{'='*60}")
    print(f"  Total cases:  {len(cases_to_generate)}")
    print(f"  Successful:   {success_count}")
    print(f"  Failed:       {fail_count}")
    print(f"\nMetadata saved to: {GT_METADATA_PATH}")
    print(f"GT files saved to: {GT_STAGING_DIR.absolute()}")
    print(f"{'='*60}")

    # Print upload instructions
    if success_count > 0:
        print(f"\n{'='*60}")
        print("UPLOAD INSTRUCTIONS")
        print(f"{'='*60}")
        print(f"\nGenerated GT files are in: {GT_STAGING_DIR.absolute()}")
        print("\nTo upload to sgl-test-files repository:")
        print("1. Clone sgl-test-files repo (if not already):")
        print("   git clone https://github.com/sgl-project/sgl-test-files.git")
        print("")
        print("2. Copy the generated files:")
        print(
            f"   cp -r {GT_STAGING_DIR.absolute()}/* sgl-test-files/images/consistency_gt/"
        )
        print("")
        print("3. Commit and push:")
        print("   cd sgl-test-files")
        print("   git add images/consistency_gt/")
        print('   git commit -m "Add consistency GT files"')
        print("   git push")
        print("")
        print(
            "4. After the PR is merged, the tests will automatically use the new GT files."
        )
        print(f"{'='*60}\n")

    if fail_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
