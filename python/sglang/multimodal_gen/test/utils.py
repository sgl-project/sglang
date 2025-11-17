# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
import json
import os

import numpy as np
import torch
from pytorch_msssim import ms_ssim, ssim
from torchvision.io import read_video

from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


def compute_video_ssim_torchvision(video1_path, video2_path, use_ms_ssim=True):
    """
    Compute SSIM between two videos.

    Args:
        video1_path: Path to the first video.
        video2_path: Path to the second video.
        use_ms_ssim: Whether to use Multi-Scale Structural Similarity(MS-SSIM) instead of SSIM.
    """
    print(f"Computing SSIM between {video1_path} and {video2_path}...")
    if not os.path.exists(video1_path):
        raise FileNotFoundError(f"Video1 not found: {video1_path}")
    if not os.path.exists(video2_path):
        raise FileNotFoundError(f"Video2 not found: {video2_path}")

    frames1, _, _ = read_video(video1_path, pts_unit="sec", output_format="TCHW")
    frames2, _, _ = read_video(video2_path, pts_unit="sec", output_format="TCHW")

    # Ensure same number of frames
    min_frames = min(frames1.shape[0], frames2.shape[0])
    frames1 = frames1[:min_frames]
    frames2 = frames2[:min_frames]

    frames1 = frames1.float() / 255.0
    frames2 = frames2.float() / 255.0

    if torch.cuda.is_available():
        frames1 = frames1.cuda()
        frames2 = frames2.cuda()

    ssim_values = []

    # Process each frame individually
    for i in range(min_frames):
        img1 = frames1[i : i + 1]
        img2 = frames2[i : i + 1]

        with torch.no_grad():
            if use_ms_ssim:
                value = ms_ssim(img1, img2, data_range=1.0)
            else:
                value = ssim(img1, img2, data_range=1.0)

            ssim_values.append(value.item())

    if ssim_values:
        mean_ssim = np.mean(ssim_values)
        min_ssim = np.min(ssim_values)
        max_ssim = np.max(ssim_values)
        min_frame_idx = np.argmin(ssim_values)
        max_frame_idx = np.argmax(ssim_values)

        print(f"Mean SSIM: {mean_ssim:.4f}")
        print(f"Min SSIM: {min_ssim:.4f} (at frame {min_frame_idx})")
        print(f"Max SSIM: {max_ssim:.4f} (at frame {max_frame_idx})")

        return mean_ssim, min_ssim, max_ssim
    else:
        print("No SSIM values calculated")
        return 0, 0, 0


def compare_folders(reference_folder, generated_folder, use_ms_ssim=True):
    """
    Compare videos with the same filename between reference_folder and generated_folder

    Example usage:
        results = compare_folders(reference_folder, generated_folder,
                              args.use_ms_ssim)
        for video_name, ssim_value in results.items():
            if ssim_value is not None:
                print(
                    f"{video_name}: {ssim_value[0]:.4f}, Min SSIM: {ssim_value[1]:.4f}, Max SSIM: {ssim_value[2]:.4f}"
                )
            else:
                print(f"{video_name}: Error during comparison")

        valid_ssims = [v for v in results.values() if v is not None]
        if valid_ssims:
            avg_ssim = np.mean([v[0] for v in valid_ssims])
            print(f"\nAverage SSIM across all videos: {avg_ssim:.4f}")
        else:
            print("\nNo valid SSIM values to average")
    """

    reference_videos = [f for f in os.listdir(reference_folder) if f.endswith(".mp4")]

    results = {}

    for video_name in reference_videos:
        ref_path = os.path.join(reference_folder, video_name)
        gen_path = os.path.join(generated_folder, video_name)

        if os.path.exists(gen_path):
            print(f"\nComparing {video_name}...")
            try:
                ssim_value = compute_video_ssim_torchvision(
                    ref_path, gen_path, use_ms_ssim
                )
                results[video_name] = ssim_value
            except Exception as e:
                print(f"Error comparing {video_name}: {e}")
                results[video_name] = None
        else:
            print(f"\nSkipping {video_name} - no matching file in generated folder")

    return results


def write_ssim_results(
    output_dir, ssim_values, reference_path, generated_path, num_inference_steps, prompt
):
    """
    Write SSIM results to a JSON file in the same directory as the generated videos.
    """
    try:
        logger.info(f"Attempting to write SSIM results to directory: {output_dir}")

        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        mean_ssim, min_ssim, max_ssim = ssim_values

        result = {
            "mean_ssim": mean_ssim,
            "min_ssim": min_ssim,
            "max_ssim": max_ssim,
            "reference_video": reference_path,
            "generated_video": generated_path,
            "parameters": {
                "num_inference_steps": num_inference_steps,
                "prompt": prompt,
            },
        }

        test_name = f"steps{num_inference_steps}_{prompt[:100]}"
        result_file = os.path.join(output_dir, f"{test_name}_ssim.json")
        logger.info(f"Writing JSON results to: {result_file}")
        with open(result_file, "w") as f:
            json.dump(result, f, indent=2)

        logger.info(f"SSIM results written to {result_file}")
        return True
    except Exception as e:
        logger.error(f"ERROR writing SSIM results: {str(e)}")
        return False
