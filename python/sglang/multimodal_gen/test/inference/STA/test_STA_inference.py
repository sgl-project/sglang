import os
import subprocess
import sys
from pathlib import Path

import pytest

NUM_NODES = "1"
NUM_GPUS_PER_NODE = "2"

# Set environment variables
os.environ["SGL_DIFFUSION_ATTENTION_CONFIG"] = "assets/mask_strategy_wan.json"
os.environ["SGL_DIFFUSION_ATTENTION_BACKEND"] = "SLIDING_TILE_ATTN"


def test_inference():
    """Test the inference functionality"""
    # Create command as in wan_14B-STA.sh
    cmd = [
        "sgl-diffusion",
        "generate",
        "--model-path",
        "Wan-AI/Wan2.1-T2V-14B-Diffusers",
        "--sp-size",
        "2",
        "--tp-size",
        "2",
        "--num-gpus",
        "2",
        "--height",
        "768",
        "--width",
        "1280",
        "--num-frames",
        "69",
        "--num-inference-steps",
        "2",
        "--fps",
        "16",
        "--guidance-scale",
        "5.0",
        "--flow-shift",
        "5.0",
        "--prompt",
        "A majestic lion strides across the golden savanna, its powerful frame glistening under the warm afternoon sun. The tall grass ripples gently in the breeze, enhancing the lion's commanding presence. The tone is vibrant, embodying the raw energy of the wild. Low angle, steady tracking shot, cinematic.",
        "--negative-prompt",
        "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards",
        "--seed",
        "1024",
        "--output-path",
        "outputs_video/STA_1024/",
    ]

    # Run the command
    subprocess.run(cmd, check=True)

    # Verify output directory exists
    output_dir = Path("outputs_video/STA_1024/")
    assert output_dir.exists(), f"Output directory {output_dir} does not exist"

    # Verify that video files were generated
    video_files = list(output_dir.glob("*.mp4"))
    assert len(video_files) > 0, "No video files were generated"

    # Verify the video file properties
    for video_file in video_files:
        assert video_file.stat().st_size > 0, f"Video file {video_file} is empty"


if __name__ == "__main__":
    test_inference()
