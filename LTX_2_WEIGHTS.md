# LTX-2 Model Weights Structure for SGLang

To run the LTX-2 Audio-Video pipeline in SGLang, you need to organize your model weights as follows.

## Directory Structure

```
ltx-2-model/
├── model_index.json
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
├── audio_vae/  # Audio decoder
│   ├── config.json
│   └── diffusion_pytorch_model.safetensors
├── vocoder/    # Audio vocoder
│   ├── config.json
│   └── diffusion_pytorch_model.safetensors
└── upsampler/  # Latent upsampler (two-stage)
    ├── config.json
    └── diffusion_pytorch_model.safetensors
```

## Configuration Details

### Audio VAE (`audio_vae/config.json`)
Should contain parameters for `LTX2AudioDecoder` (or be nested under `audio_vae.model.params.ddconfig`):

- `ch`: 128
- `out_ch`: 2
- `ch_mult`: [1, 2, 4, 8]
- `num_res_blocks`: 2
- `attn_resolutions`: []
- `resolution`: 256
- `z_channels`: 8
- `sample_rate`: 16000
- `mel_bins`: 64

### Vocoder (`vocoder/config.json`)
Should contain parameters for `LTX2Vocoder` (or be nested under `vocoder`):

- `resblock_kernel_sizes`: [3, 7, 11]
- `upsample_rates`: [6, 5, 2, 2, 2]
- `upsample_initial_channel`: 1024
- `output_sample_rate`: 24000

### Upsampler (`upsampler/config.json`)
Should contain parameters for `LatentUpsampler` (or be nested under `upsampler`).

## Notes

- The `transformer` implementation must support `audio_hidden_states` input.
- Ensure `scheduler` is compatible with Flow Matching (e.g., `FlowMatchEulerDiscreteScheduler`).


- `vocoder/config.json`: Vocoder configuration.
- `vocoder/diffusion_pytorch_model.safetensors`: Vocoder weights.

## Model Index Configuration

To use the native SGLang backend for LTX-2 (One-Stage), your `model_index.json` must specify the correct pipeline class name.

**For One-Stage Pipeline (Text-to-Video with Audio):**

```json
{
  "_class_name": "LTXVideoPipeline",
  "_diffusers_version": "0.29.0",
  "transformer": [
    "ltx_video",
    "LTXVideoTransformer3DModel"
  ],
  "text_encoder": [
    "transformers",
    "T5EncoderModel"
  ],
  "tokenizer": [
    "transformers",
    "T5Tokenizer"
  ],
  "scheduler": [
    "diffusers",
    "FlowMatchEulerDiscreteScheduler"
  ],
  "vae": [
    "diffusers",
    "AutoencoderKL"
  ],
  "audio_vae": [
    "ltx_video",
    "LTXVideoAudioVAE"
  ],
  "vocoder": [
    "ltx_video",
    "LTXVideoVocoder"
  ]
}
```

**For Two-Stage Pipeline (if applicable):**

Set `"_class_name": "LTXVideoTwoStagePipeline"`.

## Implementation Notes

- The pipeline supports both video and audio generation.
- Audio generation is enabled by default but can be disabled by setting `generate_audio=False` in the request.
- The implementation uses a custom `LTX2AVDenoisingStage` to handle joint video-audio denoising.
- Text embeddings are processed to support LTX-2's specific requirements (separate contexts for video and audio streams).