"""Shared request setup and output checks for direct ComfyUI pipeline tests."""

import torch


def prepare_passthrough_request(req):
    if req.guidance_scale > 1.0 and req.negative_prompt_embeds is not None:
        req.do_classifier_free_guidance = True
    else:
        req.do_classifier_free_guidance = False
    if req.seed is not None:
        generator_device = req.generator_device
        device_str = "cpu" if generator_device == "cpu" else "cuda"
        req.generator = [
            torch.Generator(device_str).manual_seed(req.seed + i)
            for i in range(req.num_outputs_per_prompt)
        ]
    else:
        req.generator = [
            torch.Generator("cuda") for _ in range(req.num_outputs_per_prompt)
        ]


def check_passthrough_output(generator, req, *, label="", device="cuda"):
    output_batch = generator._send_to_scheduler_and_wait_for_response([req])
    noise_pred = output_batch.noise_pred

    assert noise_pred is not None, "noise_pred should not be None in OutputBatch"
    assert isinstance(noise_pred, torch.Tensor), "noise_pred should be a torch.Tensor"
    assert noise_pred.device.type == device, (
        f"noise_pred should be on {device}, got {noise_pred.device}"
    )
    assert noise_pred.dtype == torch.bfloat16, (
        f"noise_pred should be bfloat16, got {noise_pred.dtype}"
    )

    print(f"\u2713 Successfully retrieved noise_pred from OutputBatch{label}!")
    print(f"  noise_pred shape: {noise_pred.shape}")
    print(f"  noise_pred dtype: {noise_pred.dtype}")
    print(f"  noise_pred device: {noise_pred.device}")

    latents = output_batch.output if output_batch.output is not None else req.latents
    assert latents is not None, "latents should not be None"
    return latents
