# SPDX-License-Identifier: Apache-2.0
import json
import os

import pytest
import torch
from diffusers import DiffusionPipeline
from torch.distributed.tensor import DTensor
from torch.testing import assert_close

from sgl_diffusion import DiffGenerator
from sgl_diffusion.runtime.loader.utils import (
    get_param_names_mapping,
    hf_to_custom_state_dict,
)
from sgl_diffusion.runtime.pipelines import build_pipeline
from sgl_diffusion.runtime.server_args import ServerArgs
from sgl_diffusion.runtime.utils.logging_utils import init_logger
from sgl_diffusion.runtime.worker import Scheduler
from sgl_diffusion.test.utils import (
    compute_video_ssim_torchvision,
    write_ssim_results,
)

logger = init_logger(__name__)
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "29500"

# Base parameters for LoRA inference tests
WAN_LORA_PARAMS = {
    "num_gpus": 1,
    "model_path": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    "height": 480,
    "width": 832,
    "num_frames": 45,
    "num_inference_steps": 32,
    "guidance_scale": 5.0,
    "flow_shift": 3.0,
    "seed": 42,
    "fps": 24,
    "neg_prompt": "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
    "text-encoder-precision": ("fp32",),
    "dit_cpu_offload": True,
}

# LoRA configurations for testing
LORA_CONFIGS = [
    {
        "lora_path": "benjamin-paine/steamboat-willie-1.3b",
        "lora_nickname": "steamboat",
        "prompt": "steamboat willie style, golden era animation, close-up of a short fluffy monster kneeling beside a melting red candle. the mood is one of wonder and curiosity, as the monster gazes at the flame with wide eyes and open mouth. Its pose and expression convey a sense of innocence and playfulness, as if it is exploring the world around it for the first time. The use of warm colors and dramatic lighting further enhances the cozy atmosphere of the image.",
        "negative_prompt": "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
        "ssim_threshold": 0.79,
    },
    {
        "lora_path": "motimalu/wan-flat-color-1.3b-v2",
        "lora_nickname": "flat_color",
        "prompt": "flat color, no lineart, blending, negative space, artist:[john kafka|ponsuke kaikai|hara id 21|yoneyama mai|fuzichoco], 1girl, sakura miko, pink hair, cowboy shot, white shirt, floral print, off shoulder, outdoors, cherry blossom, tree shade, wariza, looking up, falling petals, half-closed eyes, white sky, clouds, live2d animation, upper body, high quality cinematic video of a woman sitting under a sakura tree. Dreamy and lonely, the camera close-ups on the face of the woman as she turns towards the viewer. The Camera is steady, This is a cowboy shot. The animation is smooth and fluid.",
        "negative_prompt": "bad quality video,色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
        "ssim_threshold": 0.79,
    },
]

MODEL_TO_PARAMS = {
    "Wan-AI/Wan2.1-T2V-1.3B-Diffusers": WAN_LORA_PARAMS,
}


@pytest.mark.parametrize("model_id", list(MODEL_TO_PARAMS.keys()))
def test_merge_lora_weights(model_id):
    lora_config = LORA_CONFIGS[0]  # test only one
    hf_pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
    hf_pipe.enable_model_cpu_offload()

    lora_nickname = lora_config["lora_nickname"]
    lora_path = lora_config["lora_path"]
    args = ServerArgs.from_kwargs(
        model_path=model_id,
        dit_cpu_offload=True,
        dit_precision="bf16",
    )
    pipe = build_pipeline(args)
    pipe.set_lora_adapter(lora_nickname, lora_path)
    custom_transformer = pipe.modules["transformer"]
    custom_state_dict = custom_transformer.state_dict()

    hf_pipe.load_lora_weights(lora_path, adapter_name=lora_nickname)
    for name, layer in hf_pipe.transformer.named_modules():
        if hasattr(layer, "unmerge"):
            layer.unmerge()
            layer.merge(adapter_names=[lora_nickname])

    hf_transformer = hf_pipe.transformer
    param_names_mapping = get_param_names_mapping(
        custom_transformer.param_names_mapping
    )
    hf_state_dict, _ = hf_to_custom_state_dict(
        hf_transformer.state_dict(), param_names_mapping
    )
    for key in hf_state_dict.keys():
        if "base_layer" not in key:
            continue
        hf_param = hf_state_dict[key]
        custom_param = (
            custom_state_dict[key].to_local()
            if isinstance(custom_state_dict[key], DTensor)
            else custom_state_dict[key]
        )
        assert_close(hf_param, custom_param, atol=7e-4, rtol=7e-4)


@pytest.mark.parametrize("ATTENTION_BACKEND", ["TORCH_SDPA"])
@pytest.mark.parametrize("model_id", list(MODEL_TO_PARAMS.keys()))
def test_lora_inference_similarity(ATTENTION_BACKEND, model_id):
    """
    Test that runs LoRA inference with LoRA switching and compares the output
    to reference videos using SSIM.
    """
    os.environ["SGL_DIFFUSION_ATTENTION_BACKEND"] = ATTENTION_BACKEND

    script_dir = os.path.dirname(os.path.abspath(__file__))

    output_dir = os.path.join(
        script_dir, "generated_videos", model_id.split("/")[-1], ATTENTION_BACKEND
    )

    os.makedirs(output_dir, exist_ok=True)

    BASE_PARAMS = MODEL_TO_PARAMS[model_id]
    num_inference_steps = BASE_PARAMS["num_inference_steps"]

    init_kwargs = {
        "num_gpus": BASE_PARAMS["num_gpus"],
        "flow_shift": BASE_PARAMS["flow_shift"],
        "dit_cpu_offload": BASE_PARAMS["dit_cpu_offload"],
    }
    if "text-encoder-precision" in BASE_PARAMS:
        init_kwargs["text_encoder_precisions"] = BASE_PARAMS["text-encoder-precision"]

    generation_kwargs = {
        "num_inference_steps": num_inference_steps,
        "output_path": output_dir,
        "height": BASE_PARAMS["height"],
        "width": BASE_PARAMS["width"],
        "num_frames": BASE_PARAMS["num_frames"],
        "guidance_scale": BASE_PARAMS["guidance_scale"],
        "seed": BASE_PARAMS["seed"],
        "fps": BASE_PARAMS["fps"],
        "save_output": True,
    }
    generator = DiffGenerator.from_pretrained(
        model_path=BASE_PARAMS["model_path"], **init_kwargs
    )
    for lora_config in LORA_CONFIGS:
        lora_nickname = lora_config["lora_nickname"]
        lora_path = lora_config["lora_path"]
        prompt = lora_config["prompt"]
        generation_kwargs["negative_prompt"] = lora_config["negative_prompt"]

        generator.set_lora_adapter(lora_nickname=lora_nickname, lora_path=lora_path)
        output_file_name = f"{lora_path.split('/')[-1]}_{prompt[:50]}"
        generation_kwargs["output_path"] = output_dir
        generation_kwargs["output_file_name"] = output_file_name

        generator.generate(prompt, **generation_kwargs)

        assert os.path.exists(
            output_dir
        ), f"Output video was not generated at {output_dir}"

        reference_folder = os.path.join(
            script_dir,
            "L40S_reference_videos",
            model_id.split("/")[-1],
            ATTENTION_BACKEND,
        )

        if not os.path.exists(reference_folder):
            logger.error("Reference folder missing")
            raise FileNotFoundError(
                f"Reference video folder does not exist: {reference_folder}"
            )

        # Find the matching reference video for the switched LoRA
        reference_video_name = None

        for filename in os.listdir(reference_folder):
            # Check if the filename starts with the expected output_file_name and ends with .mp4
            if filename.startswith(output_file_name) and filename.endswith(".mp4"):
                reference_video_name = (
                    filename  # Remove .mp4 extension to match the logic below
                )
                break

        if not reference_video_name:
            logger.error(
                f"Reference video not found for adapter: {lora_path} with prompt: {prompt[:50]} and backend: {ATTENTION_BACKEND}"
            )
            raise FileNotFoundError(f"Reference video missing for adapter {lora_path}")

        reference_video_path = os.path.join(reference_folder, reference_video_name)
        generated_video_path = os.path.join(output_dir, output_file_name + ".mp4")

        logger.info(
            f"Computing SSIM between {reference_video_path} and {generated_video_path}"
        )
        ssim_values = compute_video_ssim_torchvision(
            reference_video_path, generated_video_path, use_ms_ssim=True
        )

        mean_ssim = ssim_values[0]
        logger.info(f"SSIM mean value: {mean_ssim}")
        logger.info(f"Writing SSIM results to directory: {output_dir}")

        success = write_ssim_results(
            output_dir,
            ssim_values,
            reference_video_path,
            generated_video_path,
            num_inference_steps,
            prompt,
        )

        if not success:
            logger.error("Failed to write SSIM results to file")

        min_acceptable_ssim = lora_config["ssim_threshold"]
        assert (
            mean_ssim >= min_acceptable_ssim
        ), f"SSIM value {mean_ssim} is below threshold {min_acceptable_ssim} for adapter {lora_config['lora_path']}"
