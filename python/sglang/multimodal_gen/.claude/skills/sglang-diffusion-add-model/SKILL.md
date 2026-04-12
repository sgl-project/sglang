---
name: sglang-diffusion-add-model
description: Use when adding a new diffusion model or Diffusers pipeline to SGLang.
---

# Add a Diffusion Model to SGLang

Use this skill when adding a new diffusion model or pipeline variant to `sglang.multimodal_gen`.

## Two Pipeline Styles

### Style A: Hybrid Monolithic Pipeline (Recommended)

The recommended default for most new models. Uses a three-stage structure:

```
BeforeDenoisingStage (model-specific)  -->  DenoisingStage (standard)  -->  DecodingStage (standard)
```

- **BeforeDenoisingStage**: A single, model-specific stage that consolidates all pre-processing logic: input validation, text encoding, image encoding, latent preparation, timestep setup. This stage is unique per model.
- **DenoisingStage**: Framework-standard stage for the denoising loop (DiT/UNet forward passes). Shared across models.
- **DecodingStage**: Framework-standard stage for VAE decoding. Shared across models.

**Why recommended?** Modern diffusion models have highly heterogeneous pre-processing requirements (different text encoders, different latent formats, different conditioning mechanisms). The Hybrid approach keeps pre-processing isolated per model, avoids fragile shared stages with excessive conditional logic, and lets developers port Diffusers reference code quickly.

### Style B: Modular Composition Style

Uses the framework's fine-grained standard stages (`TextEncodingStage`, `LatentPreparationStage`, `TimestepPreparationStage`, etc.) to build the pipeline by composition.

This style is appropriate when:
- **The new model's pre-processing can largely reuse existing stages** — e.g., a model that uses standard CLIP/T5 text encoding + standard latent preparation with minimal customization. In this case, `add_standard_t2i_stages()` or `add_standard_ti2i_stages()` may be all you need.
- **A model-specific optimization needs to be extracted as a standalone stage** — e.g., a specialized encoding or conditioning step that benefits from being a separate stage for profiling, parallelism control, or reuse across multiple pipeline variants.

See existing Modular examples: `QwenImagePipeline` (uses `add_standard_t2i_stages`), `FluxPipeline`, `WanPipeline`.

### How to Choose

| Situation | Recommended Style |
|-----------|-------------------|
| Model has unique/complex pre-processing (VLM captioning, AR token generation, custom latent packing, etc.) | **Hybrid** — consolidate into a BeforeDenoisingStage |
| Model fits neatly into standard text-to-image or text+image-to-image pattern | **Modular** — use `add_standard_t2i_stages()` / `add_standard_ti2i_stages()` |
| Porting a Diffusers pipeline with many custom steps | **Hybrid** — copy the `__call__` logic into a single stage |
| Adding a variant of an existing model that shares most logic | **Modular** — reuse existing stages, customize via PipelineConfig callbacks |
| A specific pre-processing step needs special parallelism or profiling isolation | **Modular** — extract that step as a dedicated stage |

**Key principle (both styles)**: The stage(s) before `DenoisingStage` must produce a `Req` batch object with all the standard tensor fields that `DenoisingStage` expects (latents, timesteps, prompt_embeds, etc.). As long as this contract is met, the pipeline remains composable regardless of which style you use.

---

## Key Files and Directories

| Purpose | Path |
|---------|------|
| Pipeline classes | `python/sglang/multimodal_gen/runtime/pipelines/` |
| Model-specific stages | `python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/` |
| PipelineStage base class | `python/sglang/multimodal_gen/runtime/pipelines_core/stages/base.py` |
| Pipeline base class | `python/sglang/multimodal_gen/runtime/pipelines_core/composed_pipeline_base.py` |
| Standard stages (Denoising, Decoding) | `python/sglang/multimodal_gen/runtime/pipelines_core/stages/` |
| Pipeline configs | `python/sglang/multimodal_gen/configs/pipeline_configs/` |
| Sampling params | `python/sglang/multimodal_gen/configs/sample/` |
| DiT model implementations | `python/sglang/multimodal_gen/runtime/models/dits/` |
| VAE implementations | `python/sglang/multimodal_gen/runtime/models/vaes/` |
| Encoder implementations | `python/sglang/multimodal_gen/runtime/models/encoders/` |
| Scheduler implementations | `python/sglang/multimodal_gen/runtime/models/schedulers/` |
| Model/VAE/DiT configs | `python/sglang/multimodal_gen/configs/models/dits/`, `vaes/`, `encoders/` |
| Central registry | `python/sglang/multimodal_gen/registry.py` |

---

## Step-by-Step Implementation

### Step 1: Obtain and Study the Reference Implementation

**Before writing any code, obtain the model's reference implementation or Diffusers pipeline code.** You need the actual source code to work from — do not guess or assume the model's architecture. If the user already gave a HuggingFace model ID or repo, inspect that yourself first. Ask the user only when the reference implementation is private, ambiguous, or otherwise unavailable. Typical sources are:
- The model's Diffusers pipeline source (e.g., the `pipeline_*.py` file from the `diffusers` library or HuggingFace repo)
- Or the model's official reference implementation (e.g., from the model author's GitHub repo)
- Or the HuggingFace model ID so you can look up `model_index.json` and the associated pipeline class

Once you have the reference code, study it thoroughly:

1. Find the model's `model_index.json` to identify required modules (text_encoder, vae, transformer, scheduler, etc.)
2. Read the Diffusers pipeline's `__call__` method end-to-end. Identify:
   - How text prompts are encoded
   - How latents are prepared (shape, dtype, scaling)
   - How timesteps/sigmas are computed
   - What conditioning kwargs the DiT/UNet expects
   - How the denoising loop works (classifier-free guidance, etc.)
   - How VAE decoding is done (scaling factors, tiling, etc.)

### Step 2: Evaluate Reuse of Existing Pipelines and Stages

**Before creating any new files, check whether an existing pipeline or stage can be reused or extended.** Only create new pipelines/stages when the existing ones would require extensive modifications or when no similar implementation exists.

Specifically:
1. **Compare the new model's architecture against existing pipelines** (Flux, Wan, Qwen-Image, GLM-Image, HunyuanVideo, LTX, etc.). If the new model shares most of its structure with an existing one (e.g., same text encoders, similar latent format, compatible denoising loop), prefer:
   - Adding a new config variant to the existing pipeline rather than creating a new pipeline class
   - Reusing the existing `BeforeDenoisingStage` with minor parameter differences
   - Using `add_standard_t2i_stages()` / `add_standard_ti2i_stages()` / `add_standard_ti2v_stages()` if the model fits standard patterns
2. **Check existing stages** in `runtime/pipelines_core/stages/` and `stages/model_specific_stages/`. If an existing stage handles 80%+ of what the new model needs, extend it rather than duplicating it.
3. **Check existing model components** — many models share VAEs (e.g., `AutoencoderKL`), text encoders (CLIP, T5), and schedulers. Reuse these directly instead of re-implementing.

**Rule of thumb**: Only create a new file when the existing implementation would need substantial structural changes to accommodate the new model, or when no architecturally similar implementation exists.

### Step 3: Implement Model Components

Adapt or implement the model's core components in the appropriate directories.

**DiT/Transformer** (`runtime/models/dits/{model_name}.py`):

```python
# python/sglang/multimodal_gen/runtime/models/dits/my_model.py

import torch
import torch.nn as nn

from sglang.multimodal_gen.runtime.layers.layernorm import (
    LayerNormScaleShift,
    RMSNormScaleShift,
)
from sglang.multimodal_gen.runtime.layers.attention.selector import (
    get_attn_backend,
)


class MyModelTransformer2DModel(nn.Module):
    """DiT model for MyModel.

    Adapt from the Diffusers/reference implementation. Key points:
    - Use SGLang's fused LayerNorm/RMSNorm ops (see `existing-fast-paths.md` under the benchmark/profile skill)
    - Use SGLang's attention backend selector
    - Keep the same parameter naming as Diffusers for weight loading compatibility
    """

    def __init__(self, config):
        super().__init__()
        # ... model layers ...

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        # ... model-specific kwargs ...
    ) -> torch.Tensor:
        # ... forward pass ...
        return output
```

**Tensor Parallel (TP) and Sequence Parallel (SP)**: For multi-GPU deployment, it is recommended to add TP/SP support to the DiT model. This can be done incrementally after the single-GPU implementation is verified. Reference existing implementations and adapt to your model's architecture:

- **Wan model** (`runtime/models/dits/wanvideo.py`) — Full TP + SP reference:
  - TP: Uses `ColumnParallelLinear` for Q/K/V projections, `RowParallelLinear` for output projections, attention heads divided by `tp_size`
  - SP: Sequence dimension sharding via `get_sp_world_size()`, padding for alignment, `sequence_model_parallel_all_gather` for aggregation
  - Cross-attention skips SP (`skip_sequence_parallel=is_cross_attention`)
- **Qwen-Image model** (`runtime/models/dits/qwen_image.py`) — SP + USPAttention reference:
  - SP: Uses `USPAttention` (Ulysses + Ring Attention), configured via `--ulysses-degree` / `--ring-degree`
  - TP: Uses `MergedColumnParallelLinear` for QKV (with Nunchaku quantization), `ReplicatedLinear` otherwise

**Important**: These are references only — each model has its own architecture and parallelism requirements. Consider:
- How attention heads can be divided across TP ranks
- Whether the model's sequence dimension is naturally shardable for SP
- Which linear layers benefit from column/row parallel sharding vs. replication
- Whether cross-attention or other special modules need SP exclusion

Key imports for distributed support:
```python
from sglang.multimodal_gen.runtime.distributed import (
    divide,
    get_sp_group,
    get_sp_world_size,
    get_tp_world_size,
    sequence_model_parallel_all_gather,
)
from sglang.multimodal_gen.runtime.layers.linear import (
    ColumnParallelLinear,
    RowParallelLinear,
    ReplicatedLinear,
)
```

**VAE** (`runtime/models/vaes/{model_name}.py`): Implement if the model uses a non-standard VAE. Many models reuse existing VAEs.

**Encoders** (`runtime/models/encoders/{model_name}.py`): Implement if the model uses custom text/image encoders.

**Schedulers** (`runtime/models/schedulers/{scheduler_name}.py`): Implement if the model requires a custom scheduler not available in Diffusers.

### Step 4: Create Model Configs

**DiT Config** (`configs/models/dits/{model_name}.py`):

```python
# python/sglang/multimodal_gen/configs/models/dits/mymodel.py

from dataclasses import dataclass, field

from sglang.multimodal_gen.configs.models.dits.base import DiTConfig


@dataclass
class MyModelDitConfig(DiTConfig):
    arch_config: dict = field(default_factory=lambda: {
        "in_channels": 16,
        "num_layers": 24,
        "patch_size": 2,
        # ... model-specific architecture params ...
    })
```

**VAE Config** (`configs/models/vaes/{model_name}.py`):

```python
from dataclasses import dataclass, field

from sglang.multimodal_gen.configs.models.vaes.base import VAEConfig


@dataclass
class MyModelVAEConfig(VAEConfig):
    vae_scale_factor: int = 8
    # ... VAE-specific params ...
```

**Sampling Params** (`configs/sample/{model_name}.py`):

```python
from dataclasses import dataclass

from sglang.multimodal_gen.configs.sample.base import SamplingParams


@dataclass
class MyModelSamplingParams(SamplingParams):
    num_inference_steps: int = 50
    guidance_scale: float = 7.5
    height: int = 1024
    width: int = 1024
    # ... model-specific defaults ...
```

### Step 5: Create PipelineConfig

The `PipelineConfig` holds static model configuration and defines callback methods used by the standard `DenoisingStage` and `DecodingStage`.

```python
# python/sglang/multimodal_gen/configs/pipeline_configs/my_model.py

from dataclasses import dataclass, field

from sglang.multimodal_gen.configs.pipeline_configs.base import (
    ImagePipelineConfig,      # for image generation
    # SpatialImagePipelineConfig,  # alternative base
    # VideoPipelineConfig,         # for video generation
)
from sglang.multimodal_gen.configs.models.dits.mymodel import MyModelDitConfig
from sglang.multimodal_gen.configs.models.vaes.mymodel import MyModelVAEConfig


@dataclass
class MyModelPipelineConfig(ImagePipelineConfig):
    """Pipeline config for MyModel.

    This config provides callbacks that the standard DenoisingStage and
    DecodingStage use during execution. The BeforeDenoisingStage handles
    all model-specific pre-processing independently.
    """

    task_type: ModelTaskType = ModelTaskType.T2I
    vae_precision: str = "bf16"
    should_use_guidance: bool = True
    vae_tiling: bool = False
    enable_autocast: bool = False

    dit_config: DiTConfig = field(default_factory=MyModelDitConfig)
    vae_config: VAEConfig = field(default_factory=MyModelVAEConfig)

    # --- Callbacks used by DenoisingStage ---

    def get_freqs_cis(self, batch, device, rotary_emb, dtype):
        """Prepare rotary position embeddings for the DiT."""
        # Model-specific RoPE computation
        ...
        return freqs_cis

    def prepare_pos_cond_kwargs(self, batch, latent_model_input, t, **kwargs):
        """Build positive conditioning kwargs for each denoising step."""
        return {
            "hidden_states": latent_model_input,
            "encoder_hidden_states": batch.prompt_embeds[0],
            "timestep": t,
            # ... model-specific kwargs ...
        }

    def prepare_neg_cond_kwargs(self, batch, latent_model_input, t, **kwargs):
        """Build negative conditioning kwargs for CFG."""
        return {
            "hidden_states": latent_model_input,
            "encoder_hidden_states": batch.negative_prompt_embeds[0],
            "timestep": t,
            # ... model-specific kwargs ...
        }

    # --- Callbacks used by DecodingStage ---

    def get_decode_scale_and_shift(self):
        """Return (scale, shift) for latent denormalization before VAE decode."""
        return self.vae_config.latents_std, self.vae_config.latents_mean

    def post_denoising_loop(self, latents, batch):
        """Optional post-processing after the denoising loop finishes."""
        return latents.to(torch.bfloat16)

    def post_decoding(self, frames, server_args):
        """Optional post-processing after VAE decoding."""
        return frames
```

**Important**: The `prepare_pos_cond_kwargs` / `prepare_neg_cond_kwargs` methods define what the DiT receives at each denoising step. These must match the DiT's `forward()` signature.

### Step 6: Implement the BeforeDenoisingStage (Core Step)

This is the heart of the Hybrid pattern. Create a single stage that handles ALL pre-processing.

```python
# python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/my_model.py

import torch
from typing import List, Optional, Union

from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


class MyModelBeforeDenoisingStage(PipelineStage):
    """Monolithic pre-processing stage for MyModel.

    Consolidates all logic before the denoising loop:
    - Input validation
    - Text/image encoding
    - Latent preparation
    - Timestep/sigma computation

    This stage produces a Req batch with all fields required by
    the standard DenoisingStage.
    """

    def __init__(self, vae, text_encoder, tokenizer, transformer, scheduler):
        super().__init__()
        self.vae = vae
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.transformer = transformer
        self.scheduler = scheduler
        # ... other initialization (image processors, scale factors, etc.) ...

    # --- Internal helper methods ---
    # Copy/adapt directly from the Diffusers reference pipeline.
    # These are private to this stage; no need to make them reusable.

    def _encode_prompt(self, prompt, device, dtype):
        """Encode text prompt into embeddings."""
        # ... model-specific text encoding logic ...
        return prompt_embeds, negative_prompt_embeds

    def _prepare_latents(self, batch_size, height, width, dtype, device, generator):
        """Create initial noisy latents."""
        # ... model-specific latent preparation ...
        return latents

    def _prepare_timesteps(self, num_inference_steps, device):
        """Compute the timestep/sigma schedule."""
        # ... model-specific timestep computation ...
        return timesteps, sigmas

    # --- Main forward method ---

    @torch.no_grad()
    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        """Execute all pre-processing and populate batch for DenoisingStage.

        This method mirrors the first half of a Diffusers pipeline __call__,
        up to (but not including) the denoising loop.
        """
        device = get_local_torch_device()
        dtype = torch.bfloat16
        generator = torch.Generator(device=device).manual_seed(batch.seed)

        # 1. Encode prompt
        prompt_embeds, negative_prompt_embeds = self._encode_prompt(
            batch.prompt, device, dtype
        )

        # 2. Prepare latents
        latents = self._prepare_latents(
            batch_size=1,
            height=batch.height,
            width=batch.width,
            dtype=dtype,
            device=device,
            generator=generator,
        )

        # 3. Prepare timesteps
        timesteps, sigmas = self._prepare_timesteps(
            batch.num_inference_steps, device
        )

        # 4. Populate batch with everything DenoisingStage needs
        batch.prompt_embeds = [prompt_embeds]
        batch.negative_prompt_embeds = [negative_prompt_embeds]
        batch.latents = latents
        batch.timesteps = timesteps
        batch.num_inference_steps = len(timesteps)
        batch.sigmas = sigmas
        batch.generator = generator
        batch.raw_latent_shape = latents.shape
        batch.height = batch.height
        batch.width = batch.width

        return batch
```

**Key fields that `DenoisingStage` expects on the batch** (set these in your `forward`):

| Field | Type | Description |
|-------|------|-------------|
| `batch.latents` | `torch.Tensor` | Initial noisy latent tensor |
| `batch.timesteps` | `torch.Tensor` | Timestep schedule |
| `batch.num_inference_steps` | `int` | Number of denoising steps |
| `batch.sigmas` | `list[float]` | Sigma schedule (as a list, not numpy) |
| `batch.prompt_embeds` | `list[torch.Tensor]` | Positive prompt embeddings (wrapped in list) |
| `batch.negative_prompt_embeds` | `list[torch.Tensor]` | Negative prompt embeddings (wrapped in list) |
| `batch.generator` | `torch.Generator` | RNG generator for reproducibility |
| `batch.raw_latent_shape` | `tuple` | Original latent shape before any packing |
| `batch.height` / `batch.width` | `int` | Output dimensions |

### Step 7: Define the Pipeline Class

The pipeline class is minimal -- it just wires the stages together.

```python
# python/sglang/multimodal_gen/runtime/pipelines/my_model.py

from sglang.multimodal_gen.runtime.pipelines_core import LoRAPipeline
from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
    ComposedPipelineBase,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages import DenoisingStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.my_model import (
    MyModelBeforeDenoisingStage,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs


class MyModelPipeline(LoRAPipeline, ComposedPipelineBase):
    pipeline_name = "MyModelPipeline"  # Must match model_index.json _class_name

    _required_config_modules = [
        "text_encoder",
        "tokenizer",
        "vae",
        "transformer",
        "scheduler",
        # ... list all modules from model_index.json ...
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


# REQUIRED: This is how the registry discovers the pipeline
EntryClass = [MyModelPipeline]
```

### Step 8: Register the Model

In `python/sglang/multimodal_gen/registry.py`, register your configs:

```python
register_configs(
    model_family="my_model",
    sampling_param_cls=MyModelSamplingParams,
    pipeline_config_cls=MyModelPipelineConfig,
    hf_model_paths=[
        "org/my-model-name",  # HuggingFace model ID(s)
    ],
)
```

The `EntryClass` in your pipeline file is automatically discovered by the registry's `_discover_and_register_pipelines()` function -- no additional registration needed for the pipeline class itself.

### Step 9: Verify Output Quality

After implementation, **you must verify that the generated output is not noise**. A noisy or garbled output image/video is the most common sign of an incorrect implementation. Common causes include:

- Incorrect latent scale/shift factors (`get_decode_scale_and_shift` returning wrong values)
- Wrong timestep/sigma schedule (order, dtype, or value range)
- Mismatched conditioning kwargs (fields not matching the DiT's `forward()` signature)
- Incorrect VAE decoder configuration (wrong `vae_scale_factor`, missing denormalization)
- Rotary embedding style mismatch (`is_neox_style` set incorrectly)
- Wrong prompt embedding format (missing list wrapping, wrong encoder output selection)

**If the output is noise, the implementation is incorrect — do not ship it.** Debug by:
1. Comparing intermediate tensor values (latents, prompt_embeds, timesteps) against the Diffusers reference pipeline
2. Running the Diffusers pipeline and SGLang pipeline side-by-side with the same seed
3. Checking each stage's output shape and value range independently

## Reference Implementations

### Hybrid Style (recommended for most new models)

| Model | Pipeline | BeforeDenoisingStage | PipelineConfig |
|-------|----------|---------------------|----------------|
| GLM-Image | `runtime/pipelines/glm_image.py` | `stages/model_specific_stages/glm_image.py` | `configs/pipeline_configs/glm_image.py` |
| Qwen-Image-Layered | `runtime/pipelines/qwen_image.py` (`QwenImageLayeredPipeline`) | `stages/model_specific_stages/qwen_image_layered.py` | `configs/pipeline_configs/qwen_image.py` (`QwenImageLayeredPipelineConfig`) |

### Modular Style (when standard stages fit well)

| Model | Pipeline | Notes |
|-------|----------|-------|
| Qwen-Image (T2I) | `runtime/pipelines/qwen_image.py` | Uses `add_standard_t2i_stages()` — standard text encoding + latent prep fits this model |
| Qwen-Image-Edit | `runtime/pipelines/qwen_image.py` | Uses `add_standard_ti2i_stages()` — standard image-to-image flow |
| Flux | `runtime/pipelines/flux.py` | Uses `add_standard_t2i_stages()` with custom `prepare_mu` |
| Wan | `runtime/pipelines/wan_pipeline.py` | Uses `add_standard_ti2v_stages()` |

---

## Checklist

Before submitting, verify:

**Common (both styles):**
- [ ] **Pipeline file** exists at `runtime/pipelines/{model_name}.py` with `EntryClass`
- [ ] **PipelineConfig** at `configs/pipeline_configs/{model_name}.py`
- [ ] **SamplingParams** at `configs/sample/{model_name}.py`
- [ ] **DiT model** at `runtime/models/dits/{model_name}.py`
- [ ] **DiT config** at `configs/models/dits/{model_name}.py`
- [ ] **VAE** — reuse existing (e.g., `AutoencoderKL`) or create new at `runtime/models/vaes/`
- [ ] **VAE config** — reuse existing or create new at `configs/models/vaes/{model_name}.py`
- [ ] **Registry entry** in `registry.py` via `register_configs()`
- [ ] `pipeline_name` matches Diffusers `model_index.json` `_class_name`
- [ ] `_required_config_modules` lists all modules from `model_index.json`
- [ ] `PipelineConfig` callbacks (`prepare_pos_cond_kwargs`, `get_freqs_cis`, etc.) match DiT's `forward()` signature
- [ ] Latent scale/shift factors are correctly configured
- [ ] Use fused kernels where possible (see `existing-fast-paths.md` under the benchmark/profile skill)
- [ ] Weight names match Diffusers for automatic loading
- [ ] **TP/SP support** considered for DiT model (recommended; reference `wanvideo.py` for TP+SP, `qwen_image.py` for USPAttention)
- [ ] **Output quality verified** — generated images/videos are not noise; compared against Diffusers reference output

**Hybrid style only:**
- [ ] **BeforeDenoisingStage** at `stages/model_specific_stages/{model_name}.py`
- [ ] `BeforeDenoisingStage.forward()` populates all fields needed by `DenoisingStage`

## Common Pitfalls

1. **`batch.sigmas` must be a Python list**, not a numpy array. Use `.tolist()` to convert.
2. **`batch.prompt_embeds` is a list of tensors** (one per encoder), not a single tensor. Wrap with `[tensor]`.
3. **Don't forget `batch.raw_latent_shape`** -- `DecodingStage` uses it to unpack latents.
4. **Rotary embedding style matters**: `is_neox_style=True` = split-half rotation, `is_neox_style=False` = interleaved. Check the reference model carefully.
5. **VAE precision**: Many VAEs need fp32 or bf16 for numerical stability. Set `vae_precision` in the PipelineConfig accordingly.
6. **Avoid forcing model-specific logic into shared stages**: If your model's pre-processing doesn't naturally fit the existing standard stages, prefer the Hybrid pattern with a dedicated BeforeDenoisingStage rather than adding conditional branches to shared stages.

## After Implementation: Tests and Performance Data

### Component Accuracy When Adding a New Testcase Config

If you add a new entry to `python/sglang/multimodal_gen/test/server/testcase_configs.py`, you must treat component accuracy as part of the model-adding workflow. Do not assume the new testcase will automatically fit the existing component-accuracy harness.

The component-accuracy harness compares SGLang components against Diffusers/HF reference components. This is stricter than pipeline-level inference. New testcase configs commonly fail here for one of three reasons:

1. **The model family needs explicit hook wiring** in `python/sglang/multimodal_gen/test/server/accuracy_hooks.py`.
   - Add hook logic only when the harness cannot call the raw component correctly without it.
   - Valid examples:
     - required forward arguments are missing from the synthetic input bundle
     - a known runtime execution context must be matched for the component to run at all, such as transformer autocast
     - the reference and SGLang expose the same component contract, but the harness needs family-specific input preparation to reach it
   - Invalid examples:
     - changing the compared output mode just to make shapes or values line up
     - adding a harness-side behavior override that changes the component contract instead of matching it

2. **The component is already covered by another testcase with the same source component and topology**.
   - In that case, do not add redundant component-accuracy coverage.
   - Add a skip entry in `python/sglang/multimodal_gen/test/server/accuracy_config.py` with a concrete reason such as:
     - `Representative VAE accuracy is already covered by ... for the same source component and topology`
   - This is the preferred path for variant-only cases such as LoRA, cache-dit, upscaling, or other testcases that reuse the same underlying component weights and topology.

3. **The HF/Diffusers reference component cannot be loaded or compared faithfully in the harness**.
   - Add a skip entry in `python/sglang/multimodal_gen/test/server/accuracy_config.py` with the exact technical failure.
   - Good reasons include:
     - missing or unsupported HF component layout
     - incomplete or partially initialized HF checkpoint
     - unsupported raw component contract for trustworthy comparison
     - proven divergence after matched weight transfer and matching output shape
   - Keep the skip reason concrete and technical. Do not write vague reasons like "component accuracy flaky" or "needs investigation."

When adding a new testcase config, make this decision explicitly:
- if the model family needs minimal harness wiring, add the smallest possible change in `accuracy_hooks.py`
- if the testcase is only a variant of an already covered source component and topology, add a skip in `accuracy_config.py`
- if the HF/Diffusers reference component cannot be compared faithfully, add a skip in `accuracy_config.py`

Do not add a new testcase config and wait for CI to discover missing component-accuracy wiring. Do not use `accuracy_hooks.py` to change the compared component contract just to make the test pass.

Once the model is working and output quality is verified, **ask the user** whether they would like to:

1. **Add tests** — Create unit tests and/or integration tests for the new model. Tests should cover:
   - Pipeline construction and stage wiring
   - Single-GPU inference producing non-noise output
   - Multi-GPU inference (TP/SP) if supported
   - See the `write-sglang-test` skill for test conventions and placement guidelines

2. **Generate performance data** — Run benchmarks and collect perf metrics:
   - Single-GPU latency and throughput (look for `Pixel data generated successfully in xxxx seconds` in console output; use the `warmup excluded` line for accurate timing)
   - Multi-GPU scaling (TP/SP) throughput comparison
   - Use `python/sglang/multimodal_gen/benchmarks/bench_serving.py` for serving benchmarks

Do not skip this step — always ask the user before proceeding, as test and benchmark requirements vary per model.
