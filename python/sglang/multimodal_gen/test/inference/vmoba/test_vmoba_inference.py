# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
from pathlib import Path


def test_inference_vmoba():
    """Test sgl-diffusion VMOBA_ATTN inference pipeline"""

    num_gpus = "1"
    model_base = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
    output_dir = Path("outputs_video/vmoba_1.3B/")
    moba_config = "sgl_diffusion/configs/backend/vmoba/wan_1.3B_77_480_832.json"

    os.environ["SGL_DIFFUSION_ATTENTION_BACKEND"] = "VMOBA_ATTN"

    cmd = [
        "sgl-diffusion",
        "generate",
        "--model-path",
        model_base,
        "--sp-size",
        num_gpus,
        "--tp-size",
        "1",
        "--num-gpus",
        num_gpus,
        "--dit-cpu-offload",
        "False",
        "--vae-cpu-offload",
        "False",
        "--text-encoder-cpu-offload",
        "True",
        "--pin-cpu-memory",
        "False",
        "--height",
        "480",
        "--width",
        "832",
        "--num-frames",
        "77",
        "--num-inference-steps",
        "50",
        "--moba-config-path",
        moba_config,
        "--fps",
        "16",
        "--guidance-scale",
        "6.0",
        "--flow-shift",
        "8.0",
        "--prompt",
        "A majestic lion strides across the golden savanna, its powerful frame glistening under the warm afternoon sun. The tall grass ripples gently in the breeze, enhancing the lion's commanding presence. The tone is vibrant, embodying the raw energy of the wild. Low angle, steady tracking shot, cinematic.",
        "--negative-prompt",
        (
            "Bright tones, overexposed, static, blurred details, subtitles, style, "
            "works, paintings, images, static, overall gray, worst quality, low quality, "
            "JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, "
            "poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, "
            "still picture, messy background, three legs, many people in the background, walking backwards"
        ),
        "--seed",
        "1024",
        "--output-path",
        str(output_dir),
    ]

    subprocess.run(cmd, check=True)

    assert output_dir.exists(), f"Output directory {output_dir} does not exist"

    video_files = list(output_dir.glob("*.mp4"))
    assert len(video_files) > 0, "No video files were generated"

    for video_file in video_files:
        assert video_file.stat().st_size > 0, f"Video file {video_file} is empty"


if __name__ == "__main__":
    test_inference_vmoba()
