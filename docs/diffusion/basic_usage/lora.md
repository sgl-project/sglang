# LoRA Usage

This document covers how to use LoRA (Low-Rank Adaptation) adapters with SGLang Diffusion.

## Loading LoRA

TODO: Examples of loading LoRA adapters via CLI, OpenAI API, and Python SDK.

## Verified LoRA Examples

This section lists example LoRAs that have been explicitly tested and verified with each base model in **SGLang Diffusion** pipeline.

> Important:
> LoRAs that are not listed here are not necessarily incompatible.
> In practice, most standard LoRAs are expected to work, especially those following common Diffusers or SD-style conventions.
> The entries below simply reflect configurations that have been manually validated by SGLang team.

### Verified LoRAs by Base Model

| Base Model       | Supported LoRAs |
|:-----------------|:----------------|
| Wan2.2           | `lightx2v/Wan2.2-Distill-Loras`<br>`Cseti/wan2.2-14B-Arcane_Jinx-lora-v1` |
| Wan2.1           | `lightx2v/Wan2.1-Distill-Loras` |
| Z-Image-Turbo    | `tarn59/pixel_art_style_lora_z_image_turbo`<br>`wcde/Z-Image-Turbo-DeJPEG-Lora` |
| Qwen-Image       | `lightx2v/Qwen-Image-Lightning`<br>`flymy-ai/qwen-image-realism-lora`<br>`prithivMLmods/Qwen-Image-HeadshotX`<br>`starsfriday/Qwen-Image-EVA-LoRA` |
| Qwen-Image-Edit  | `ostris/qwen_image_edit_inpainting`<br>`lightx2v/Qwen-Image-Edit-2511-Lightning` |
| Flux             | `dvyio/flux-lora-simple-illustration`<br>`XLabs-AI/flux-furry-lora`<br>`XLabs-AI/flux-RealismLora` |

## Merging and Switching LoRAs

TODO: Guide on merging multiple LoRAs and dynamically switching between them.
