# How to Support New Diffusion Models

This document explains how to add support for new diffusion models in SGLang Diffusion.

## Architecture Overview

SGLang Diffusion is engineered for both performance and flexibility, built upon a modular pipeline architecture. This design allows developers to easily construct complex, customized pipelines for various diffusion models by combining and reusing different components.

At its core, the architecture revolves around two key concepts, as highlighted in our [blog post](https://lmsys.org/blog/2025-11-07-sglang-diffusion/#architecture):

-   **`ComposedPipeline`**: This class orchestrates a series of `PipelineStage`s to define the complete generation process for a specific model. It acts as the main entry point for a model and manages the data flow between the different stages of the diffusion process.
-   **`PipelineStage`**: Each stage is a modular component that encapsulates a common function within the diffusion process. Examples include prompt encoding, the denoising loop, or VAE decoding. These stages are designed to be self-contained and reusable across different pipelines.

## Key Components for Implementation

To add support for a new diffusion model, you will primarily need to define or configure the following components:

1.  **`PipelineConfig`**: This is a dataclass that holds all the static configurations for your model pipeline. It includes paths to model components (like UNet, VAE, text encoders), precision settings (e.g., `fp16`, `bf16`), and other model-specific architectural parameters. Each model typically has its own subclass of `PipelineConfig`.

2.  **`SamplingParams`**: This dataclass defines the parameters that control the generation process at runtime. These are the user-provided inputs for a generation request, such as the `prompt`, `negative_prompt`, `guidance_scale`, `num_inference_steps`, `seed`, output dimensions (`height`, `width`), etc.

3.  **`ComposedPipeline` (not a config)**: This is the central class where you define the structure of your model's generation pipeline. You will create a new class that inherits from `ComposedPipelineBase` and, within it, instantiate and chain together the necessary `PipelineStage`s in the correct order. See `ComposedPipelineBase` and `PipelineStage` base definitions:
    - [`ComposedPipelineBase`](https://github.com/sgl-project/sglang/blob/main/python/sglang/multimodal_gen/runtime/pipelines/composed_pipeline_base.py)
    - [`PipelineStage`]( https://github.com/sgl-project/sglang/blob/main/python/sglang/multimodal_gen/runtime/pipelines/stages/base.py)
    - [Central registry (models/config mapping)](https://github.com/sgl-project/sglang/blob/main/python/sglang/multimodal_gen/registry.py)

4.  **Modules (components referenced by the pipeline)**: Each pipeline references a set of modules that are loaded from the model repository (e.g., Diffusers `model_index.json`) and assembled via the registry/loader. Common modules include:
    - `text_encoder`: Encodes text prompts into embeddings
    - `tokenizer`: Tokenizes raw text input for the text encoder(s).
    - `processor`: Preprocesses images and extracts features; often used in image-to-image tasks.
    - `image_encoder`: Specialized image feature extractor (may be distinct from or combined with `processor`).
    - `dit/transformer`: The core denoising network (DiT/UNet architecture) operating in latent space.
    - `scheduler`: Controls the timestep schedule and denoising dynamics throughout inference.
    - `vae`: Variational Autoencoder for encoding/decoding between pixel space and latent space.

## Available Pipeline Stages

You can build your custom `ComposedPipeline` by combining the following available stages as your will. Each stage is responsible for a specific part of the generation process.

| Stage Class                      | Description                                                                                             |
| -------------------------------- | ------------------------------------------------------------------------------------------------------- |
| `InputValidationStage`           | Validates the user-provided `SamplingParams` to ensure they are correct before starting the pipeline.     |
| `TextEncodingStage`              | Encodes text prompts into embeddings using one or more text encoders.                                   |
| `ImageEncodingStage`             | Encodes input images into embeddings, often used in image-to-image tasks.                               |
| `ImageVAEEncodingStage`          | Specifically encodes an input image into the latent space using a Variational Autoencoder (VAE).        |
| `ConditioningStage`              | Prepares the conditioning tensors (e.g., from text or image embeddings) for the denoising loop.         |
| `TimestepPreparationStage`       | Prepares the scheduler's timesteps for the diffusion process.                                           |
| `LatentPreparationStage`         | Creates the initial noisy latent tensor that will be denoised.                                          |
| `DenoisingStage`                 | Executes the main denoising loop, iteratively applying the model (e.g., UNet) to refine the latents.    |
| `DecodingStage`                  | Decodes the final latent tensor from the denoising loop back into pixel space (e.g., an image) using the VAE. |
| `DmdDenoisingStage`              | A specialized denoising stage for certain model architectures.                                          |
| `CausalDMDDenoisingStage`        | A specialized causal denoising stage for specific video models.                                         |

## Example: Implementing `Qwen-Image-Edit`

To illustrate the process, let's look at how `Qwen-Image-Edit` is implemented. The typical implementation order is:

1.  **Analyze Required Modules**:
    - Study the target model's components by examining its `model_index.json` or Diffusers implementation to identify required modules:
      - `processor`: Image preprocessing and feature extraction
      - `scheduler`: Diffusion timestep scheduling
      - `text_encoder`: Text-to-embedding conversion
      - `tokenizer`: Text tokenization for the encoder
      - `transformer`: Core DiT denoising network
      - `vae`: Variational autoencoder for latent encoding/decoding

2.  **Create Configs**:
    - **PipelineConfig**: [`QwenImageEditPipelineConfig`](https://github.com/sgl-project/sglang/blob/main/python/sglang/multimodal_gen/configs/pipelines/qwen_image.py) defines model-specific parameters, precision settings, preprocessing functions, and latent shape calculations.
    - **SamplingParams**: [`QwenImageSamplingParams`](https://github.com/sgl-project/sglang/blob/main/python/sglang/multimodal_gen/configs/sample/qwenimage.py) sets runtime defaults like `num_frames=1`, `guidance_scale=4.0`, `num_inference_steps=50`.

3.  **Implement Model Components**:
    - Adapt or implement specific model components in the appropriate directories:
      - **DiT/Transformer**: Implement in [`runtime/models/dits/`](https://github.com/sgl-project/sglang/blob/main/python/sglang/multimodal_gen/runtime/models/dits/) - e.g., [`qwen_image.py`](https://github.com/sgl-project/sglang/blob/main/python/sglang/multimodal_gen/runtime/models/dits/qwen_image.py) for Qwen's DiT architecture
      - **Encoders**: Implement in [`runtime/models/encoders/`](https://github.com/sgl-project/sglang/blob/main/python/sglang/multimodal_gen/runtime/models/encoders/) - e.g., text encoders like [`qwen2_5vl.py`](https://github.com/sgl-project/sglang/blob/main/python/sglang/multimodal_gen/runtime/models/encoders/qwen2_5vl.py)
      - **VAEs**: Implement in [`runtime/models/vaes/`](https://github.com/sgl-project/sglang/blob/main/python/sglang/multimodal_gen/runtime/models/vaes/) - e.g., [`autoencoder_kl_qwenimage.py`](https://github.com/sgl-project/sglang/blob/main/python/sglang/multimodal_gen/runtime/models/vaes/autoencoder_kl_qwenimage.py)
      - **Schedulers**: Implement in [`runtime/models/schedulers/`](https://github.com/sgl-project/sglang/blob/main/python/sglang/multimodal_gen/runtime/models/schedulers/) if needed
    - These components handle the core model logic, attention mechanisms, and data transformations specific to the target diffusion model.

4.  **Define Pipeline Class**:
    - The [`QwenImageEditPipeline`](https://github.com/sgl-project/sglang/blob/main/python/sglang/multimodal_gen/runtime/architectures/basic/qwen_image/qwen_image.py) class inherits from `ComposedPipelineBase` and orchestrates stages sequentially.
    - Declare required modules via `_required_config_modules` and implement the pipeline stages:

    ```python
    class QwenImageEditPipeline(ComposedPipelineBase):
        pipeline_name = "QwenImageEditPipeline"  # Matches Diffusers model_index.json
        _required_config_modules = ["processor", "scheduler", "text_encoder", "tokenizer", "transformer", "vae"]

        def create_pipeline_stages(self, server_args: ServerArgs):
            """Set up pipeline stages sequentially."""
            self.add_stage(stage_name="input_validation_stage", stage=InputValidationStage())
            self.add_stage(stage_name="prompt_encoding_stage_primary", stage=ImageEncodingStage(...))
            self.add_stage(stage_name="image_encoding_stage_primary", stage=ImageVAEEncodingStage(...))
            self.add_stage(stage_name="timestep_preparation_stage", stage=TimestepPreparationStage(...))
            self.add_stage(stage_name="latent_preparation_stage", stage=LatentPreparationStage(...))
            self.add_stage(stage_name="conditioning_stage", stage=ConditioningStage())
            self.add_stage(stage_name="denoising_stage", stage=DenoisingStage(...))
            self.add_stage(stage_name="decoding_stage", stage=DecodingStage(...))
    ```
    The pipeline is constructed by adding stages in order. `Qwen-Image-Edit` uses `ImageEncodingStage` (for prompt and image processing) and `ImageVAEEncodingStage` (for latent extraction) before standard denoising and decoding.

5.  **Register Configs**:
    - Register the configs in the central registry ([`registry.py`](https://github.com/sgl-project/sglang/blob/main/python/sglang/multimodal_gen/registry.py)) via `_register_configs` to enable automatic loading and instantiation for the model. Modules are automatically loaded and injected based on the config and repository structure.

By following this modular pattern of defining configurations and composing pipelines, you can integrate new diffusion models into SGLang with clarity and ease.
