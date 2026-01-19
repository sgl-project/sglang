#!/usr/bin/env python3
"""
Generate diffusion CI outputs for consistency testing.

Usage:
    python gen_diffusion_ci_outputs.py --suite 1-gpu --partition-id 0 --total-partitions 2 --out-dir ./output
    python gen_diffusion_ci_outputs.py --suite 1-gpu --case-ids qwen_image_t2i flux_image_t2i --out-dir ./output
"""

import argparse
import base64
import os
import time
from pathlib import Path

import requests
from openai import OpenAI
from PIL import Image

from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.test.server.test_server_utils import (
    ServerManager,
    WarmupRunner,
    download_image_from_url,
)
from sglang.multimodal_gen.test.server.testcase_configs import (
    ONE_GPU_CASES_A,
    ONE_GPU_CASES_B,
    TWO_GPU_CASES_A,
    TWO_GPU_CASES_B,
    DiffusionTestCase,
)
from sglang.multimodal_gen.test.test_utils import (
    extract_key_frames_from_video,
    get_dynamic_server_port,
    is_image_url,
)

logger = init_logger(__name__)


def _openai_client(port: int) -> OpenAI:
    """Create OpenAI client for the server."""
    return OpenAI(api_key="sglang-anything", base_url=f"http://localhost:{port}/v1")


def _build_server_extra_args(case: DiffusionTestCase) -> str:
    """Build extra_args string for server startup."""
    extra_args = "--backend diffusers"
    extra_args += f" --num-gpus {case.server_args.num_gpus}"

    if case.server_args.tp_size is not None:
        extra_args += f" --tp-size {case.server_args.tp_size}"

    if case.server_args.ulysses_degree is not None:
        extra_args += f" --ulysses-degree {case.server_args.ulysses_degree}"

    if case.server_args.dit_layerwise_offload:
        extra_args += " --dit-layerwise-offload true"

    if case.server_args.text_encoder_cpu_offload:
        extra_args += " --text-encoder-cpu-offload"

    if case.server_args.ring_degree is not None:
        extra_args += f" --ring-degree {case.server_args.ring_degree}"

    if case.server_args.lora_path:
        extra_args += f" --lora-path {case.server_args.lora_path}"

    if case.server_args.enable_warmup:
        extra_args += " --enable-warmup"

    return extra_args


def _build_env_vars(case: DiffusionTestCase) -> dict[str, str]:
    """Build environment variables dict for server startup."""
    env_vars = {}
    if case.server_args.enable_cache_dit:
        env_vars["SGLANG_CACHE_DIT_ENABLED"] = "true"
    return env_vars


def _get_cases_for_suite(
    suite: str, case_ids: list[str] | None = None
) -> list[DiffusionTestCase]:
    """Get test cases for the specified suite, optionally filtered by case IDs."""
    if suite == "1-gpu":
        all_cases = ONE_GPU_CASES_A + ONE_GPU_CASES_B
    elif suite == "2-gpu":
        all_cases = TWO_GPU_CASES_A + TWO_GPU_CASES_B
    else:
        raise ValueError(f"Invalid suite: {suite}. Must be '1-gpu' or '2-gpu'")

    # Deduplicate by case.id
    seen: set[str] = set()
    deduplicated: list[DiffusionTestCase] = []
    for c in all_cases:
        if c.id not in seen:
            seen.add(c.id)
            deduplicated.append(c)

    # Filter by case_ids if provided
    if case_ids is not None and len(case_ids) > 0:
        case_id_set = set(case_ids)
        filtered_cases = [c for c in deduplicated if c.id in case_id_set]
        if len(filtered_cases) == 0:
            logger.warning(f"No matching cases found for provided case IDs: {case_ids}")
        missing_ids = case_id_set - {c.id for c in filtered_cases}
        if missing_ids:
            logger.warning(f"Some case IDs not found: {missing_ids}")
        return filtered_cases

    return deduplicated


def _should_skip_case(case: DiffusionTestCase) -> bool:
    """Check if case should be skipped (e.g., HIP with ring_degree > 1)."""
    if (
        current_platform.is_hip()
        and case.server_args.ring_degree is not None
        and case.server_args.ring_degree > 1
    ):
        return True
    return False


def _process_image_case(
    case: DiffusionTestCase, client: OpenAI, out_dir: Path, continue_on_error: bool
) -> None:
    """Process an image generation case."""
    sp = case.sampling_params
    output_size = os.environ.get("SGLANG_TEST_OUTPUT_SIZE", sp.output_size)
    output_format = sp.output_format or "png"
    ext = output_format.lower()
    if ext not in ["png", "jpeg", "jpg", "webp"]:
        ext = "png"

    try:
        img_bytes = None

        # Determine generation type
        if sp.direct_url_test and sp.image_path and sp.prompt:
            # Direct URL test (TI2I with URL)
            image_urls = sp.image_path
            if not isinstance(image_urls, list):
                image_urls = [image_urls]

            response = client.images.with_raw_response.edit(
                model=case.server_args.model_path,
                prompt=sp.prompt,
                image=[],  # Empty for OpenAI verification
                n=1,
                size=output_size,
                response_format="b64_json",
                output_format=output_format,
                extra_body={"url": image_urls, "num_frames": sp.num_frames},
            )
            result = response.parse()
            img_bytes = base64.b64decode(result.data[0].b64_json)

        elif sp.image_path and sp.prompt:
            # Image edit (TI2I)
            image_paths = sp.image_path
            if not isinstance(image_paths, list):
                image_paths = [image_paths]

            new_image_paths = []
            for image_path in image_paths:
                if is_image_url(image_path):
                    new_image_paths.append(download_image_from_url(str(image_path)))
                else:
                    path_obj = Path(image_path)
                    if not path_obj.exists():
                        if continue_on_error:
                            logger.warning(
                                f"{case.id}: image file missing: {image_path}, skipping"
                            )
                            return
                        raise FileNotFoundError(
                            f"{case.id}: image file missing: {image_path}"
                        )
                    new_image_paths.append(path_obj)

            images = [open(img_path, "rb") for img_path in new_image_paths]
            try:
                response = client.images.with_raw_response.edit(
                    model=case.server_args.model_path,
                    image=images,
                    prompt=sp.prompt,
                    n=1,
                    size=output_size,
                    response_format="b64_json",
                    output_format=output_format,
                    extra_body={"num_frames": sp.num_frames},
                )
            finally:
                for img in images:
                    img.close()

            result = response.parse()
            img_bytes = base64.b64decode(result.data[0].b64_json)

        else:
            # Text to Image (T2I)
            if not sp.prompt:
                if continue_on_error:
                    logger.warning(f"{case.id}: no prompt configured, skipping")
                    return
                raise ValueError(f"{case.id}: no prompt configured")

            response = client.images.with_raw_response.generate(
                model=case.server_args.model_path,
                prompt=sp.prompt,
                n=1,
                size=output_size,
                response_format="b64_json",
            )
            result = response.parse()
            img_bytes = base64.b64decode(result.data[0].b64_json)

        # Save image
        output_path = out_dir / f"{case.id}.{ext}"
        output_path.write_bytes(img_bytes)
        logger.info(f"Saved image: {output_path}")

    except Exception as e:
        if continue_on_error:
            logger.error(f"{case.id}: failed to process image case: {e}", exc_info=True)
        else:
            raise


def _process_video_case(
    case: DiffusionTestCase, client: OpenAI, out_dir: Path, continue_on_error: bool
) -> None:
    """Process a video generation case."""
    sp = case.sampling_params
    output_size = os.environ.get("SGLANG_TEST_OUTPUT_SIZE", sp.output_size)

    try:
        # Create video job
        create_kwargs: dict[str, any] = {
            "model": case.server_args.model_path,
            "size": output_size,
        }
        if sp.prompt:
            create_kwargs["prompt"] = sp.prompt
        if sp.seconds:
            create_kwargs["seconds"] = sp.seconds
        if sp.num_frames:
            create_kwargs["num_frames"] = sp.num_frames

        # Handle input_reference (image for I2V/TI2V)
        if sp.image_path:
            if sp.direct_url_test:
                # Direct URL test
                image_urls = sp.image_path
                if not isinstance(image_urls, list):
                    image_urls = [image_urls]
                extra_body = {
                    "reference_url": image_urls,
                }
                if sp.fps:
                    extra_body["fps"] = sp.fps
                if sp.num_frames:
                    extra_body["num_frames"] = sp.num_frames
                create_kwargs["extra_body"] = extra_body
                job = client.videos.create(**create_kwargs)
            else:
                # Download or use local image
                image_path = sp.image_path
                if not isinstance(image_path, list):
                    image_path = [image_path]

                # Handle single image case
                if len(image_path) == 1:
                    img_path = image_path[0]
                    if is_image_url(img_path):
                        img_path = download_image_from_url(str(img_path))
                    else:
                        img_path = Path(img_path)
                        if not img_path.exists():
                            if continue_on_error:
                                logger.warning(
                                    f"{case.id}: image file missing: {img_path}, skipping"
                                )
                                return
                            raise FileNotFoundError(
                                f"{case.id}: image file missing: {img_path}"
                            )

                    with open(img_path, "rb") as fh:
                        create_kwargs["input_reference"] = fh
                        extra_body = {}
                        if sp.fps:
                            extra_body["fps"] = sp.fps
                        if sp.num_frames:
                            extra_body["num_frames"] = sp.num_frames
                        if extra_body:
                            create_kwargs["extra_body"] = extra_body
                        job = client.videos.create(**create_kwargs)
                else:
                    # Multiple images - use direct URL approach
                    image_urls = []
                    for img in image_path:
                        if is_image_url(img):
                            image_urls.append(img)
                        else:
                            if continue_on_error:
                                logger.warning(
                                    f"{case.id}: local multi-image not fully supported, skipping"
                                )
                                return
                            raise ValueError(
                                f"{case.id}: local multi-image not supported"
                            )
                    extra_body = {
                        "reference_url": image_urls,
                    }
                    if sp.fps:
                        extra_body["fps"] = sp.fps
                    if sp.num_frames:
                        extra_body["num_frames"] = sp.num_frames
                    create_kwargs["extra_body"] = extra_body
                    job = client.videos.create(**create_kwargs)
        else:
            job = client.videos.create(**create_kwargs)

        video_id = job.id
        logger.info(f"{case.id}: created video job {video_id}")

        # Poll for completion
        timeout = 2400.0 if current_platform.is_hip() else 1200.0
        deadline = time.time() + timeout
        job_completed = False

        while time.time() < deadline:
            page = client.videos.list()
            item = next((v for v in page.data if v.id == video_id), None)

            if item and getattr(item, "status", None) == "completed":
                job_completed = True
                break

            time.sleep(1)

        if not job_completed:
            if continue_on_error:
                logger.error(
                    f"{case.id}: video job {video_id} timed out after {timeout}s"
                )
                return
            raise TimeoutError(
                f"{case.id}: video job {video_id} did not complete in time"
            )

        # Download video
        resp = client.videos.download_content(video_id=video_id)
        video_bytes = resp.read()
        logger.info(
            f"{case.id}: downloaded video {video_id} ({len(video_bytes)} bytes)"
        )

        # Extract key frames
        frames = extract_key_frames_from_video(video_bytes, num_frames=sp.num_frames)

        if len(frames) != 3:
            if continue_on_error:
                logger.warning(
                    f"{case.id}: expected 3 frames, got {len(frames)}, skipping frame save"
                )
                return
            raise ValueError(f"{case.id}: expected 3 frames, got {len(frames)}")

        # Save frames
        frame_suffixes = ["_frame_0", "_frame_mid", "_frame_last"]
        for frame, suffix in zip(frames, frame_suffixes):
            frame_path = out_dir / f"{case.id}{suffix}.png"
            img = Image.fromarray(frame)
            img.save(frame_path)
            logger.info(f"Saved frame: {frame_path}")

        # Note: We intentionally do NOT save the .mp4 file

    except Exception as e:
        if continue_on_error:
            logger.error(f"{case.id}: failed to process video case: {e}", exc_info=True)
        else:
            raise


def _run_case(case: DiffusionTestCase, out_dir: Path, continue_on_error: bool) -> None:
    """Run a single test case."""
    if _should_skip_case(case):
        logger.info(f"Skipping {case.id}: HIP with ring_degree > 1")
        return

    port = get_dynamic_server_port()
    extra_args = _build_server_extra_args(case)
    env_vars = _build_env_vars(case)

    logger.info(f"Starting server for {case.id} on port {port}")
    manager = ServerManager(
        model=case.server_args.model_path,
        port=port,
        wait_deadline=1200.0,
        extra_args=extra_args,
        env_vars=env_vars,
    )
    ctx = manager.start()

    try:
        client = _openai_client(ctx.port)

        # Handle dynamic LoRA loading
        if case.server_args.dynamic_lora_path:
            base_url = f"http://localhost:{ctx.port}/v1"
            payload = {
                "lora_nickname": "default",
                "lora_path": case.server_args.dynamic_lora_path,
            }
            logger.info(
                f"{case.id}: loading dynamic LoRA: {case.server_args.dynamic_lora_path}"
            )
            resp = requests.post(f"{base_url}/set_lora", json=payload, timeout=30)
            if resp.status_code != 200:
                error_msg = f"set_lora failed: {resp.text}"
                if continue_on_error:
                    logger.warning(f"{case.id}: {error_msg}")
                else:
                    raise RuntimeError(f"{case.id}: {error_msg}")

        # Handle warmup
        if case.server_args.warmup > 0:
            sp = case.sampling_params
            output_size = os.environ.get("SGLANG_TEST_OUTPUT_SIZE", sp.output_size)
            warmup = WarmupRunner(
                port=ctx.port,
                model=case.server_args.model_path,
                prompt=sp.prompt or "A colorful raccoon icon",
                output_size=output_size,
                output_format=sp.output_format,
            )

            if sp.image_path and sp.prompt:
                # Edit warmup
                image_path_list = sp.image_path
                if not isinstance(image_path_list, list):
                    image_path_list = [image_path_list]

                new_image_path_list = []
                for image_path in image_path_list:
                    if is_image_url(image_path):
                        new_image_path_list.append(
                            download_image_from_url(str(image_path))
                        )
                    else:
                        path_obj = Path(image_path)
                        if not path_obj.exists():
                            if continue_on_error:
                                logger.warning(
                                    f"{case.id}: warmup image missing: {image_path}, skipping warmup"
                                )
                                break
                            raise FileNotFoundError(
                                f"{case.id}: warmup image missing: {image_path}"
                            )
                        new_image_path_list.append(path_obj)

                if new_image_path_list:
                    warmup.run_edit_warmups(
                        count=case.server_args.warmup,
                        edit_prompt=sp.prompt,
                        image_path=new_image_path_list,
                    )
            else:
                # Text warmup
                warmup.run_text_warmups(case.server_args.warmup)

        # Generate content
        if case.server_args.modality == "image":
            _process_image_case(case, client, out_dir, continue_on_error)
        elif case.server_args.modality == "video":
            _process_video_case(case, client, out_dir, continue_on_error)
        else:
            raise ValueError(
                f"{case.id}: unknown modality: {case.server_args.modality}"
            )

    finally:
        ctx.cleanup()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Generate diffusion CI outputs")
    parser.add_argument(
        "--suite",
        type=str,
        choices=["1-gpu", "2-gpu"],
        required=True,
        help="Test suite to run (1-gpu or 2-gpu)",
    )
    parser.add_argument(
        "--partition-id",
        type=int,
        required=False,
        help="Partition ID for matrix partitioning (0-based)",
    )
    parser.add_argument(
        "--total-partitions",
        type=int,
        required=False,
        help="Total number of partitions",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        required=True,
        help="Output directory for generated files",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue processing other cases if one fails",
    )
    parser.add_argument(
        "--case-ids",
        type=str,
        nargs="*",
        required=False,
        help="Specific case IDs to run (space-separated). If provided, only these cases will be run.",
    )

    args = parser.parse_args()

    # Validate partition arguments
    if args.partition_id is not None and args.total_partitions is not None:
        if args.partition_id < 0 or args.partition_id >= args.total_partitions:
            parser.error(f"partition-id must be in range [0, {args.total_partitions})")
    elif args.partition_id is not None or args.total_partitions is not None:
        parser.error(
            "Both --partition-id and --total-partitions must be provided together"
        )

    # Create output directory
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Get cases
    all_cases = _get_cases_for_suite(
        args.suite, args.case_ids if args.case_ids else None
    )

    # Apply partition filtering if specified
    if args.partition_id is not None and args.total_partitions is not None:
        my_cases = [
            c
            for i, c in enumerate(all_cases)
            if i % args.total_partitions == args.partition_id
        ]
        logger.info(
            f"Partition {args.partition_id}/{args.total_partitions}: "
            f"running {len(my_cases)} of {len(all_cases)} cases"
        )
    else:
        my_cases = all_cases
        logger.info(f"Running {len(my_cases)} cases")

    # Run cases
    for case in my_cases:
        logger.info(f"Processing case: {case.id}")
        try:
            _run_case(case, out_dir, args.continue_on_error)
        except Exception as e:
            if args.continue_on_error:
                logger.error(f"{case.id}: failed: {e}", exc_info=True)
            else:
                logger.error(f"{case.id}: failed: {e}", exc_info=True)
                raise


if __name__ == "__main__":
    main()
