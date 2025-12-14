# Efficient Video Sampling (EVS)

Implementation of [Efficient Video Sampling: Pruning Temporally Redundant Tokens for Faster VLM Inference](https://arxiv.org/abs/2510.14624).

## Overview

> NOTE: The current implementation in sglang is cannot work with VLMs that use positional embeddings [Such as Qwen2.5VL]. Further work is warranted.
> NOTE: Actual retained accuracy post-EVS may depend on how dynamic the input videos are, how high the pruning rate is, whether or not the model was trained with EVS on or not, etc. To learn more, read the paper above. It is incumbent on the user to evaluate as per their use case and benchmarks.

Video frames often contain redundant information, as consecutive frames may be nearly identical. EVS exploits this in the latent space [=embedding space] by computing similarity between adjacent frame token embeddings and pruning tokens that are highly similar to the previous frames. This reduces the token count while preserving informative content.

Key properties:
- The first frame is always fully retained (provides complete initial context)
- Configurable via `video_pruning_rate` in model config.json (0 = disabled, 0.7 = ~70% reduction; ~30% retained.)

## Architecture

### Request Flow

1. Prompt Construction (EVSProcessor)
    * Calculates estimated tokens per frame based on pruning rate, so the emitted input_ids tensor's length will by definition match the final sequence length post pruning. This is necessary for 3.
2. Embedding Generation (EVS)
    * Calls original model `get_video_feature()` for full embeddings
    * Retains top-k dissimilar tokens
    * Returns EVSEmbeddingResult in addition to pruned token counts *per frame*
3. Token Redistribution (mm_utils)
    * Adjusts input_ids so each frame's placeholder tokens matches the pruned count from 2.


## Integration Guide

### Step 1: Model [See `NemotronH_Nano_VL_V2`]

Make your model inherit from `EVS` and implement `create_evs_config`:

```python
from sglang.srt.multimodal.evs import EVSConfig, EVS

class MyEVSVideoModel(EVS):
    @staticmethod
    def create_evs_config(config):
        return EVSConfig(
            video_pruning_rate=config.video_pruning_rate
        )

    def __init__(self, config, ...):
        super().__init__(config)  # EVS wraps get_video_feature
        ...

    def get_video_feature(self, items):
        # Your existing implementation
        # Returns: (total_frames, tokens_per_frame, hidden_dim)
        ...
```

### Step 2: Processor [See `NanoNemotronVLImageProcessor`]

Create an `EVSProcessor` as a member of your VLImageProcessor:

```python
from sglang.srt.multimodal.evs import EVSProcessor

class MyProcessor:
    models = [MyEVSVideoModel, MyNonEVSModel] # You may mix evs and non evs models in a processor

    def __init__(hf_config):
        self.evs = EVSProcessor(hf_config, config_to_evs_model={MyEVSVideoModelConfig: MyEVSVideoModel})

    def process_video(self, ...):
        for video in videos:
            tokens_per_frame = self.tokens_per_frame()
        mm_items = create_data_items(
            image=image_feature,
            image_offsets=img_offsets,
            video=video_feature,
            video_offsets=video_offsets,
        )
```

### Step 3: Config [See `NemotronH_Nano_VL_V2_Config`]

Add `video_pruning_rate` to your model config:

```python
class MyModelConfig(PretrainedConfig):
    def __init__(self, ..., video_pruning_rate=0.0, ...):
        self.video_pruning_rate = video_pruning_rate
```

## Files

- `evs_core.py`: Core algorithms (retention mask computation, token redistribution)
- `evs_module.py`: EVS, configs)
- `evs_processor.py`: EVSProcessor
