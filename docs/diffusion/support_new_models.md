# How to Support New Diffusion Models

This document explains how to add support for new diffusion models in SGLang Diffusion.

## Architecture Overview

SGLang Diffusion is engineered for both performance and flexibility, built upon a pipeline architecture. This
design allows developers to construct pipelines for various diffusion models while keeping the core generation
loop standardized for optimization.

At its core, the architecture revolves around two key concepts, as highlighted in our [blog post](https://lmsys.org/blog/2025-11-07-sglang-diffusion/#architecture):

-   **`ComposedPipeline`**: This class orchestrates a series of `PipelineStage`s to define the complete generation process for a specific model. It acts as the main entry point for a model and manages the data flow between the different stages of the diffusion process.
-   **`PipelineStage`**: Each stage is a modular component that encapsulates a function within the diffusion process. Examples include prompt encoding, the denoising loop, or VAE decoding.

### Two Pipeline Styles

SGLang Diffusion supports two pipeline composition styles. Both are valid; choose the one that best fits your model.

#### Style A: Hybrid Monolithic Pipeline (Recommended Default)

The recommended default for most new models. Uses a three-stage structure:

```
BeforeDenoisingStage (model-specific)  →  DenoisingStage (standard)  →  DecodingStage (standard)
```

| Stage | Ownership | Responsibility |
|-------|-----------|----------------|
| `{Model}BeforeDenoisingStage` | Model-specific | All pre-processing: input validation, text/image encoding, latent preparation, timestep computation |
| `DenoisingStage` | Framework-standard | The denoising loop (DiT/UNet forward passes), shared across all models |
| `DecodingStage` | Framework-standard | VAE decoding from latent space to pixel space, shared across all models |

**Why recommended?** Modern diffusion models often have highly heterogeneous pre-processing requirements — different text encoders, different latent formats, different conditioning mechanisms. The Hybrid approach keeps pre-processing isolated per model, avoids fragile shared stages with excessive conditional logic, and lets developers port Diffusers reference code quickly.

#### Style B: Modular Composition Style

Uses the framework's fine-grained standard stages (`TextEncodingStage`, `LatentPreparationStage`, `TimestepPreparationStage`, etc.) to build the pipeline by composition. Convenience methods like `add_standard_t2i_stages()` and `add_standard_ti2i_stages()` make this very concise.

This style is appropriate when:
- **The new model's pre-processing can largely reuse existing stages** — e.g., a model that uses standard CLIP/T5 text encoding + standard latent preparation with minimal customization.
- **A model-specific optimization needs to be extracted as a standalone stage** — e.g., a specialized encoding or conditioning step that benefits from being a separate stage for profiling, parallelism control, or reuse across multiple pipeline variants.

#### How to Choose

| Situation | Recommended Style |
|-----------|-------------------|
| Model has unique/complex pre-processing (VLM captioning, AR token generation, custom latent packing, etc.) | **Hybrid** — consolidate into a BeforeDenoisingStage |
| Model fits neatly into standard text-to-image or text+image-to-image pattern | **Modular** — use `add_standard_t2i_stages()` / `add_standard_ti2i_stages()` |
| Porting a Diffusers pipeline with many custom steps | **Hybrid** — copy the `__call__` logic into a single stage |
| Adding a variant of an existing model that shares most logic | **Modular** — reuse existing stages, customize via PipelineConfig callbacks |
| A specific pre-processing step needs special parallelism or profiling isolation | **Modular** — extract that step as a dedicated stage |

## Key Components for Implementation

To add support for a new diffusion model, you will need to define or configure the following components:

1.  **`PipelineConfig`**: A dataclass holding static configurations for your model pipeline — precision settings, model architecture parameters, and callback methods used by the standard `DenoisingStage` and `DecodingStage`. Each model has its own subclass.

2.  **`SamplingParams`**: A dataclass defining runtime generation parameters — `prompt`, `negative_prompt`, `guidance_scale`, `num_inference_steps`, `seed`, `height`, `width`, etc.

3.  **Pre-processing stage(s)**: Either a single model-specific `{Model}BeforeDenoisingStage` (Hybrid style) or a combination of standard stages (Modular style). See [Two Pipeline Styles](#two-pipeline-styles) above.

4.  **`ComposedPipeline`**: A class that wires together your pre-processing stage(s) with the standard `DenoisingStage` and `DecodingStage`. See base definitions:
    - [`ComposedPipelineBase`](https://github.com/sgl-project/sglang/blob/main/python/sglang/multimodal_gen/runtime/pipelines_core/composed_pipeline_base.py)
    - [`PipelineStage`](https://github.com/sgl-project/sglang/blob/main/python/sglang/multimodal_gen/runtime/pipelines_core/stages/base.py)
    - [Central registry](https://github.com/sgl-project/sglang/blob/main/python/sglang/multimodal_gen/registry.py)

5.  **Modules (model components)**: Each pipeline references modules loaded from the model repository (e.g., Diffusers `model_index.json`):
    - `text_encoder`: Encodes text prompts into embeddings.
    - `tokenizer`: Tokenizes raw text input for the text encoder(s).
    - `processor`: Preprocesses images and extracts features; often used in image-to-image tasks.
    - `image_encoder`: Specialized image feature extractor.
    - `dit/transformer`: The core denoising network (DiT/UNet architecture) operating in latent space.
    - `scheduler`: Controls the timestep schedule and denoising dynamics.
    - `vae`: Variational Autoencoder for encoding/decoding between pixel space and latent space.

## Pipeline Stages Reference

### Core Stages (used by all pipelines)

| Stage Class                      | Description                                                                                             |
| -------------------------------- | ------------------------------------------------------------------------------------------------------- |
| `DenoisingStage`                 | Executes the main denoising loop, iteratively applying the model (DiT/UNet) to refine the latents.      |
| `DecodingStage`                  | Decodes the final latent tensor back into pixel space using the VAE.                                    |
| `DmdDenoisingStage`              | A specialized denoising stage for DMD model architectures.                                              |
| `CausalDMDDenoisingStage`        | A specialized causal denoising stage for specific video models.                                         |

### Pre-processing Stages (for Modular Composition Style)

The following fine-grained stages can be composed to build the pre-processing portion of a pipeline. They are best suited for models whose pre-processing largely fits the standard patterns. If your model requires significant customization, consider the Hybrid style with a single `BeforeDenoisingStage` instead.

| Stage Class                      | Description                                                                                             |
| -------------------------------- | ------------------------------------------------------------------------------------------------------- |
| `InputValidationStage`           | Validates user-provided `SamplingParams`.                                                               |
| `TextEncodingStage`              | Encodes text prompts into embeddings using one or more text encoders.                                   |
| `ImageEncodingStage`             | Encodes input images into embeddings, often used in image-to-image tasks.                               |
| `ImageVAEEncodingStage`          | Encodes an input image into latent space using the VAE.                                                 |
| `TimestepPreparationStage`       | Prepares the scheduler's timesteps for the diffusion process.                                           |
| `LatentPreparationStage`         | Creates the initial noisy latent tensor that will be denoised.                                          |

## Implementation Guide

### Step 1: Obtain and Study the Reference Implementation

Before writing any code, obtain the model's original implementation or Diffusers pipeline code:
- The model's Diffusers pipeline source (e.g., the `pipeline_*.py` file from the `diffusers` library or HuggingFace repo)
- Or the model's official reference implementation (e.g., from the model author's GitHub repo)
- Or the HuggingFace model ID to look up `model_index.json` and the associated pipeline class

Once you have the reference code, study it thoroughly:

1. Find the model's `model_index.json` to identify required modules.
2. Read the Diffusers pipeline's `__call__` method to understand:
   - How text prompts are encoded
   - How latents are prepared (shape, dtype, scaling)
   - How timesteps/sigmas are computed
   - What conditioning kwargs the DiT expects
   - How the denoising loop works
   - How VAE decoding is done

### Step 2: Evaluate Reuse of Existing Pipelines and Stages

Before creating any new files, check whether an existing pipeline or stage can be reused or extended. Only create new pipelines/stages when the existing ones would need substantial structural changes or when no architecturally similar implementation exists.

- **Compare against existing pipelines** (Flux, Wan, Qwen-Image, GLM-Image, HunyuanVideo, LTX, etc.). If the new model shares most of its structure with an existing one, prefer adding a new config variant or reusing existing stages.
- **Check existing stages** in `runtime/pipelines_core/stages/` and `stages/model_specific_stages/`.
- **Check existing model components** — many models share VAEs (e.g., `AutoencoderKL`), text encoders (CLIP, T5), and schedulers. Reuse these directly.

### Step 3: Implement Model Components

Adapt the model's core components:

- **DiT/Transformer**: Implement in [`runtime/models/dits/`](https://github.com/sgl-project/sglang/blob/main/python/sglang/multimodal_gen/runtime/models/dits/)
- **Encoders**: Implement in [`runtime/models/encoders/`](https://github.com/sgl-project/sglang/blob/main/python/sglang/multimodal_gen/runtime/models/encoders/)
- **VAEs**: Implement in [`runtime/models/vaes/`](https://github.com/sgl-project/sglang/blob/main/python/sglang/multimodal_gen/runtime/models/vaes/)
- **Schedulers**: Implement in [`runtime/models/schedulers/`](https://github.com/sgl-project/sglang/blob/main/python/sglang/multimodal_gen/runtime/models/schedulers/) if needed

Use SGLang's fused kernels where possible (see `LayerNormScaleShift`, `RMSNormScaleShift`, `apply_qk_norm`, etc.).

**Tensor Parallel (TP) and Sequence Parallel (SP)**: For multi-GPU deployment, it is recommended to add TP/SP support to the DiT model. This can be done incrementally after the single-GPU implementation is verified. Reference implementations:
- **Wan model** (`runtime/models/dits/wanvideo.py`) — Full TP + SP: `ColumnParallelLinear`/`RowParallelLinear` for attention, sequence dimension sharding via `get_sp_world_size()`
- **Qwen-Image model** (`runtime/models/dits/qwen_image.py`) — SP via `USPAttention` (Ulysses + Ring Attention)

### Step 4: Create Configs

- **DiT Config**: `configs/models/dits/{model_name}.py`
- **VAE Config**: `configs/models/vaes/{model_name}.py`
- **SamplingParams**: `configs/sample/{model_name}.py`

### Step 5: Create PipelineConfig

The `PipelineConfig` provides callbacks that the standard `DenoisingStage` and `DecodingStage` use:

```python
# python/sglang/multimodal_gen/configs/pipeline_configs/my_model.py

@dataclass
class MyModelPipelineConfig(ImagePipelineConfig):
    task_type: ModelTaskType = ModelTaskType.T2I
    vae_precision: str = "bf16"
    should_use_guidance: bool = True
    dit_config: DiTConfig = field(default_factory=MyModelDitConfig)
    vae_config: VAEConfig = field(default_factory=MyModelVAEConfig)

    def get_freqs_cis(self, batch, device, rotary_emb, dtype):
        """Prepare rotary position embeddings for the DiT."""
        ...

    def prepare_pos_cond_kwargs(self, batch, latent_model_input, t, **kwargs):
        """Build positive conditioning kwargs for each denoising step."""
        return {
            "hidden_states": latent_model_input,
            "encoder_hidden_states": batch.prompt_embeds[0],
            "timestep": t,
        }

    def prepare_neg_cond_kwargs(self, batch, latent_model_input, t, **kwargs):
        """Build negative conditioning kwargs for CFG."""
        return {
            "hidden_states": latent_model_input,
            "encoder_hidden_states": batch.negative_prompt_embeds[0],
            "timestep": t,
        }

    def get_decode_scale_and_shift(self):
        """Return (scale, shift) for latent denormalization before VAE decode."""
        ...
```

### Step 6: Implement Pre-processing

Choose based on your model's needs (see [How to Choose](#how-to-choose)):

#### Option A: BeforeDenoisingStage (Hybrid Style)

Create a single stage that handles all pre-processing. Best when the model has custom/complex pre-processing logic.

```python
# python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/my_model.py

class MyModelBeforeDenoisingStage(PipelineStage):
    """Monolithic pre-processing stage for MyModel.

    Consolidates: input validation, text/image encoding, latent
    preparation, and timestep computation.
    """

    def __init__(self, vae, text_encoder, tokenizer, transformer, scheduler):
        super().__init__()
        self.vae = vae
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.transformer = transformer
        self.scheduler = scheduler

    @torch.no_grad()
    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        device = get_local_torch_device()

        # 1. Encode prompt (model-specific logic)
        prompt_embeds, negative_prompt_embeds = self._encode_prompt(...)

        # 2. Prepare latents
        latents = self._prepare_latents(...)

        # 3. Prepare timesteps
        timesteps, sigmas = self._prepare_timesteps(...)

        # 4. Populate batch for DenoisingStage
        batch.prompt_embeds = [prompt_embeds]
        batch.negative_prompt_embeds = [negative_prompt_embeds]
        batch.latents = latents
        batch.timesteps = timesteps
        batch.num_inference_steps = len(timesteps)
        batch.sigmas = sigmas.tolist()
        batch.generator = generator
        batch.raw_latent_shape = latents.shape
        return batch
```

#### Option B: Standard Stages (Modular Style)

Skip creating a custom stage entirely — configure via `PipelineConfig` callbacks and use framework helpers. Best when the model fits standard patterns.

(This option has no separate stage file; the pipeline class in Step 7 calls `add_standard_t2i_stages()` directly.)

**Key batch fields that `DenoisingStage` expects** (regardless of which option you choose):

| Field | Type | Description |
|-------|------|-------------|
| `batch.latents` | `torch.Tensor` | Initial noisy latent tensor |
| `batch.timesteps` | `torch.Tensor` | Timestep schedule |
| `batch.num_inference_steps` | `int` | Number of denoising steps |
| `batch.sigmas` | `list[float]` | Sigma schedule (must be a Python list, not numpy) |
| `batch.prompt_embeds` | `list[torch.Tensor]` | Positive prompt embeddings (wrapped in a list) |
| `batch.negative_prompt_embeds` | `list[torch.Tensor]` | Negative prompt embeddings (wrapped in a list) |
| `batch.generator` | `torch.Generator` | RNG generator for reproducibility |
| `batch.raw_latent_shape` | `tuple` | Original latent shape before any packing |

### Step 7: Define the Pipeline Class

#### Hybrid Style

```python
# python/sglang/multimodal_gen/runtime/pipelines/my_model.py

class MyModelPipeline(LoRAPipeline, ComposedPipelineBase):
    pipeline_name = "MyModelPipeline"  # Must match model_index.json _class_name

    _required_config_modules = [
        "text_encoder", "tokenizer", "vae", "transformer", "scheduler",
    ]

    def create_pipeline_stages(self, server_args: ServerArgs):
        # 1. Monolithic pre-processing (model-specific)
        self.add_stage(
            MyModelBeforeDenoisingStage(
                vae=self.get_module("vae"),
                text_encoder=self.get_module("text_encoder"),
                tokenizer=self.get_module("tokenizer"),
                transformer=self.get_module("transformer"),
                scheduler=self.get_module("scheduler"),
            ),
        )

        # 2. Standard denoising loop (framework-provided)
        self.add_stage(
            DenoisingStage(
                transformer=self.get_module("transformer"),
                scheduler=self.get_module("scheduler"),
            ),
        )

        # 3. Standard VAE decoding (framework-provided)
        self.add_standard_decoding_stage()


EntryClass = [MyModelPipeline]
```

#### Modular Style

```python
# python/sglang/multimodal_gen/runtime/pipelines/my_model.py

class MyModelPipeline(LoRAPipeline, ComposedPipelineBase):
    pipeline_name = "MyModelPipeline"

    _required_config_modules = [
        "text_encoder", "tokenizer", "vae", "transformer", "scheduler",
    ]

    def create_pipeline_stages(self, server_args: ServerArgs):
        # All pre-processing + denoising + decoding in one call
        self.add_standard_t2i_stages(
            prepare_extra_timestep_kwargs=[prepare_mu],  # model-specific hooks
        )


EntryClass = [MyModelPipeline]
```

### Step 8: Register the Model

Register your configs in [`registry.py`](https://github.com/sgl-project/sglang/blob/main/python/sglang/multimodal_gen/registry.py):

```python
register_configs(
    model_family="my_model",
    sampling_param_cls=MyModelSamplingParams,
    pipeline_config_cls=MyModelPipelineConfig,
    hf_model_paths=["org/my-model-name"],
)
```

The `EntryClass` in your pipeline file is automatically discovered by the registry — no additional registration needed for the pipeline class itself.

### Step 9: Verify Output Quality

After implementation, verify that the generated output is not noise. A noisy or garbled output is the most common sign of an incorrect implementation. Common causes include:

- Incorrect latent scale/shift factors
- Wrong timestep/sigma schedule (order, dtype, or value range)
- Mismatched conditioning kwargs
- Rotary embedding style mismatch (`is_neox_style`)

Debug by comparing intermediate tensor values against the Diffusers reference pipeline with the same seed.

## Reference Implementations

### Hybrid Style

| Model | Pipeline | BeforeDenoisingStage | PipelineConfig |
|-------|----------|---------------------|----------------|
| GLM-Image | `runtime/pipelines/glm_image.py` | `stages/model_specific_stages/glm_image.py` | `configs/pipeline_configs/glm_image.py` |
| Qwen-Image-Layered | `runtime/pipelines/qwen_image.py` | `stages/model_specific_stages/qwen_image_layered.py` | `configs/pipeline_configs/qwen_image.py` |

### Modular Style

| Model | Pipeline | Notes |
|-------|----------|-------|
| Qwen-Image (T2I) | `runtime/pipelines/qwen_image.py` | Uses `add_standard_t2i_stages()` |
| Qwen-Image-Edit | `runtime/pipelines/qwen_image.py` | Uses `add_standard_ti2i_stages()` |
| Flux | `runtime/pipelines/flux.py` | Uses `add_standard_t2i_stages()` with custom `prepare_mu` |
| Wan | `runtime/pipelines/wan_pipeline.py` | Uses `add_standard_ti2v_stages()` |

## Checklist

Before submitting your implementation, verify:

**Common (both styles):**
- [ ] **Pipeline file** at `runtime/pipelines/{model_name}.py` with `EntryClass`
- [ ] **PipelineConfig** at `configs/pipeline_configs/{model_name}.py`
- [ ] **SamplingParams** at `configs/sample/{model_name}.py`
- [ ] **DiT model** at `runtime/models/dits/{model_name}.py`
- [ ] **Model configs** (DiT, VAE) at `configs/models/dits/` and `configs/models/vaes/`
- [ ] **Registry entry** in `registry.py` via `register_configs()`
- [ ] `pipeline_name` matches Diffusers `model_index.json` `_class_name`
- [ ] `_required_config_modules` lists all modules from `model_index.json`
- [ ] `PipelineConfig` callbacks (`prepare_pos_cond_kwargs`, etc.) match the DiT's `forward()` signature
- [ ] Uses framework-standard `DenoisingStage` and `DecodingStage` (not custom denoising loops)
- [ ] **TP/SP support** considered for DiT model (recommended; reference `wanvideo.py` for TP+SP, `qwen_image.py` for USPAttention)
- [ ] **Output quality verified** — generated images/videos are not noise; compared against Diffusers reference output

**Hybrid style only:**
- [ ] **BeforeDenoisingStage** at `stages/model_specific_stages/{model_name}.py`
- [ ] `BeforeDenoisingStage.forward()` populates all batch fields required by `DenoisingStage`
