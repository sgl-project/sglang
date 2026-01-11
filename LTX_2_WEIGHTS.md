# LTX-2 Model Weights Structure for SGLang

To run the LTX-2 Audio-Video pipeline in SGLang, you need to organize your model weights as follows.

## Directory Structure

```
ltx-2-model/
├── transformer/
│   ├── config.json
│   └── diffusion_pytorch_model.safetensors
├── vae/  # Video VAE
│   ├── config.json
│   └── diffusion_pytorch_model.safetensors
├── text_encoder/
│   ├── config.json
│   └── model.safetensors
├── tokenizer/
│   ├── tokenizer_config.json
│   ├── vocab.json
│   └── merges.txt
├── scheduler/
│   └── scheduler_config.json
├── audio_vae/  # New: Audio Decoder
│   ├── config.json
│   └── diffusion_pytorch_model.safetensors
└── vocoder/    # New: Audio Vocoder
    ├── config.json
    └── diffusion_pytorch_model.safetensors
└── upsampler/  # New: Latent Upsampler
    ├── config.json
    └── diffusion_pytorch_model.safetensors
```

## Configuration Details

### Audio VAE (audio_vae/config.json)
Should contain parameters for `AudioDecoder`:
- `ch`: 128
- `out_ch`: 128
- `ch_mult`: [1, 2, 4, 8]
- `num_res_blocks`: 2
- `attn_resolutions`: []
- `resolution`: 256
- `z_channels`: 8
- `sample_rate`: 16000

### Vocoder (vocoder/config.json)
Should contain parameters for `Vocoder`:
- `resblock_kernel_sizes`: [3, 7, 11]
- `upsample_rates`: [6, 5, 2, 2, 2]
- `upsample_initial_channel`: 1024
- `output_sample_rate`: 24000

### Upsampler (upsampler/config.json)
Should contain parameters for `LatentUpsampler`:
- `in_channels`: 128
- `mid_channels`: 512
- `num_blocks_per_stage`: 4
- `dims`: 3
- `spatial_upsample`: true
- `temporal_upsample`: false
- `spatial_scale`: 2.0
- `rational_resampler`: false

## Notes
- The `transformer` config must support `audio_hidden_states` input.
- Ensure `scheduler` is compatible with Flow Matching (e.g., `FlowMatchEulerDiscreteScheduler`).
