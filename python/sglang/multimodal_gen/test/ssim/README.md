The reference videos in the `*_reference_videos` directory are used as part of an e2e test to ensure consistency in video generation quality across code changes. `test_inference_similarity.py` compares newly generated videos against these references using Structural Similarity Index (SSIM) metrics to detect any regressions in visual quality across code changes.

`A40_reference_videos` are generated on A40s and so on.

run `bash update_reference_videos.sh` from inside the `sgl-diffusion/tests/ssim/` directory after running `test_inference_similarity.py` to update reference videos. Note: make sure to update the path to the corresponding device.

reference videos were generated on commit `4aeabbc629e0edf91477e80e795e7bb1823c71cb`
causal videos were generated on commit b318063c0a4618f1d5d99ea82ca67a06aad0d19d

## Generation Details

2 x NVIDIA L40S GPUs

## Generation Parameters

FastHunyuan-diffusers: {
"num_gpus": 2,
"model_path": "data/FastHunyuan-diffusers",
"height": 720,
"width": 1280,
"num_frames": 45,
"num_inference_steps": 6,
"guidance_scale": 1,
"embedded_cfg_scale": 6,
"flow_shift": 17,
"seed": 1024,
"sp_size": 2,
"tp_size": 1,
"vae_sp": true,
"fps": 24
}

Wan2.1-T2V-1.3B-Diffusers: {
"num_gpus": 2,
"model_path": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
"height": 480,
"width": 832,
"num_frames": 45,
"num_inference_steps": 20,
"guidance_scale": 3,
"embedded_cfg_scale": 6,
"flow_shift": 7.0,
"seed": 1024,
"sp_size": 2,
"tp_size": 1,
"vae_sp": True,
"fps": 24,
"neg_prompt": "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards",
"text-encoder-precision": "fp32"
}

Wan2.1-I2V-14B-480P-Diffusers: {
"num_gpus": 2,
"model_path": "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers",
"height": 480,
"width": 832,
"num_frames": 45,
"num_inference_steps": 6,
"guidance_scale": 5.0,
"embedded_cfg_scale": 6,
"flow_shift": 7.0,
"seed": 1024,
"sp_size": 2,
"tp_size": 1,
"vae_sp": True,
"fps": 24,
"neg_prompt": "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards",
"text-encoder-precision": "fp32"
}

### Text-to-Video Prompts

1. "Will Smith casually eats noodles, his relaxed demeanor contrasting with the energetic background of a bustling street food market. The scene captures a mix of humor and authenticity. Mid-shot framing, vibrant lighting."

2. "A lone hiker stands atop a towering cliff, silhouetted against the vast horizon. The rugged landscape stretches endlessly beneath, its earthy tones blending into the soft blues of the sky. The scene captures the spirit of exploration and human resilience. High angle, dynamic framing, with soft natural lighting emphasizing the grandeur of nature."

### Image-to-Video Prompts

1. "An astronaut hatching from an egg, on the surface of the moon, the darkness and depth of space realised in the background. High quality, ultrarealistic detail and breath-taking movie-like camera shot."
    Image path: "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/astronaut.jpg"
