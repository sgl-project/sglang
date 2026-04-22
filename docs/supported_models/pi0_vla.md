# π0 (Pi-Zero) VLA Model

π0 is a Vision-Language-Action (VLA) model from [Physical Intelligence](https://www.physicalintelligence.company/) that uses **flow matching** to generate continuous robot actions. SGLang supports π0 inference for robotics applications.

## Overview

π0 takes multi-camera images, a language instruction, and robot proprioceptive state as input, and outputs a chunk of continuous robot actions via iterative denoising.

### Architecture

- **PaliGemma backbone**: SigLIP vision encoder (So400m/14) + Gemma 2B language model
- **Gemma 300M action expert**: Processes robot state and actions with shared attention
- **Flow matching head**: Iterative denoising from Gaussian noise to actions

### Key Features

- Multi-camera support (up to 3 views: base, left wrist, right wrist)
- Continuous action output (not tokenized)
- Configurable denoising steps (default: 10)
- Action chunk prediction (default: 50 timesteps × 32 action dims)

## Quickstart

### 1. Download the Model Weights

The π0 checkpoint is published on the HuggingFace Hub at
[`lerobot/pi0_base`](https://huggingface.co/lerobot/pi0_base):

```bash
pip install "huggingface_hub[cli]"
hf download lerobot/pi0_base --local-dir /path/to/pi0
```

### 2. Patch `config.json`

`lerobot/pi0_base` ships a config shaped for LeRobot's own loader. SGLang
recognizes π0 via the `model_type: "pi0"` tag and the
`Pi0ForActionPrediction` architecture, so edit the downloaded
`config.json` (merge these keys — keep everything else as-is):

```jsonc
{
  "model_type": "pi0",
  "architectures": ["Pi0ForActionPrediction"],
  "auto_map": {
    "AutoConfig": "sglang.srt.models.pi0--Pi0Config"
  }
}
```

### 3. Install the PaliGemma Tokenizer

π0 uses the PaliGemma-3B tokenizer, which is not bundled with the
`lerobot/pi0_base` checkpoint. Pull just the tokenizer assets from the
upstream PaliGemma repo and drop them into the model directory:

```bash
# google/paligemma-3b-pt-224 is gated — you need a HuggingFace token
# with access to the PaliGemma license accepted.
export HF_TOKEN=hf_your_token_here
pip install huggingface_hub

python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    'google/paligemma-3b-pt-224',
    allow_patterns=['tokenizer*', 'special_tokens_map*'],
    local_dir='/path/to/pi0',
)
"
```

Then patch `tokenizer_config.json` to enable the leading BOS token — the π0
prompt tokenization contract (see [`Pi0NewLineProcessor`][pi0-newline] in
LeRobot) appends a trailing `\n` **and** expects a leading BOS:

```jsonc
{
  "add_bos_token": true
}
```

[pi0-newline]: https://github.com/huggingface/lerobot/blob/main/src/lerobot/policies/pi0/processor_pi0.py

### 4. Launch the Server

```bash
CUDA_VISIBLE_DEVICES=0 python -m sglang.launch_server \
    --model-path /path/to/pi0 \
    --port 8890 \
    --dtype float32
```

`--dtype float32` is the default for the reference π0 checkpoint; bfloat16
parity has not been validated end-to-end yet.

### Send a Request

```python
import requests
import base64

# Encode your camera images as base64
with open("camera_base.jpg", "rb") as f:
    base_img = base64.b64encode(f.read()).decode()
with open("camera_left_wrist.jpg", "rb") as f:
    left_img = base64.b64encode(f.read()).decode()
with open("camera_right_wrist.jpg", "rb") as f:
    right_img = base64.b64encode(f.read()).decode()

response = requests.post(
    "http://localhost:8890/generate",
    json={
        "text": "pick up the red block",
        "image_data": [
            f"data:image/jpeg;base64,{base_img}",
            f"data:image/jpeg;base64,{left_img}",
            f"data:image/jpeg;base64,{right_img}",
        ],
        "sampling_params": {
            "max_new_tokens": 1,  # π0 generates actions, not tokens
        },
        "extra_body": {
            "state": [0.1, 0.2, ...],  # 32-dim robot proprioceptive state
            "num_inference_steps": 10,  # denoising steps (optional, default=10)
        },
    },
)

# Response contains continuous actions
actions = response.json()["actions"]  # shape: (50, 32)
```

### Response Schema

VLA responses are **not** token streams. The `/generate` endpoint returns a
single JSON document with the following shape:

```jsonc
{
  "text": "",                        // always empty for VLA — do not try to parse
  "output_ids": [],                  // always empty for VLA
  "actions": [[...], [...], ...],    // (action_horizon, action_dim) nested list
  "meta_info": {
    "id": "...",
    "finish_reason": {"type": "stop", "matched": "vla_done"},
    "prompt_tokens": N,
    "completion_tokens": 0,          // VLA emits no tokens
    ...
  }
}
```

The stable client-facing contract is: **read `response.json()["actions"]`**.
Don't parse `text`; don't rely on `completion_tokens`. `finish_reason.matched
== "vla_done"` is the explicit signal that this was a VLA response.

### Normalization Contract

π0 consumes **normalized** state and emits **normalized** actions by default.
SGLang supports two deployment modes:

1. **Client-side normalization (default).** If `config.json` does **not** set
   `norm_stats`, SGLang logs an INFO message at load time and passes state /
   actions through unchanged. The client must pre-normalize the state vector
   with the training-dataset statistics and un-normalize the returned actions
   the same way. This is the most common setup with LeRobot datasets.

2. **Server-side normalization.** Add a `norm_stats` block to `config.json`
   and SGLang will apply it transparently (forward on the state input,
   inverse on the action output). Example:

   ```json
   {
     "model_type": "pi0",
     "norm_stats": {
       "state":  {"mode": "mean_std", "mean": [...], "std": [...]},
       "action": {"mode": "mean_std", "mean": [...], "std": [...]}
     }
   }
   ```

   `mode` may be `"mean_std"` or `"min_max"` (matches LeRobot's
   `NormalizerProcessorStep` semantics). Vector length can be less than
   `max_state_dim` / `max_action_dim` — only the leading entries are
   transformed, padded tail passes through unchanged.


### Using the OpenAI-Compatible API

```python
import openai

client = openai.Client(base_url="http://localhost:8890/v1", api_key="none")

response = client.chat.completions.create(
    model="default",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base_img}"},
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{left_img}"},
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{right_img}"},
                },
                {
                    "type": "text",
                    "text": "pick up the red block",
                },
            ],
        },
    ],
    extra_body={
        "state": [0.1, 0.2, 0.3],  # robot state (will be padded to 32 dims)
        "num_inference_steps": 10,
    },
)
```

## Model Variants

| Model | HuggingFace ID | Description |
|-------|---------------|-------------|
| π0 Base | `lerobot/pi0_base` | Flow-matching VLA (continuous actions) |

## Configuration

Key configuration parameters (from `config.json`):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `chunk_size` | 50 | Number of action timesteps per chunk |
| `max_action_dim` | 32 | Action vector dimension |
| `max_state_dim` | 32 | State vector dimension |
| `num_inference_steps` | 10 | Number of flow-matching denoising steps |
| `image_resolution` | (224, 224) | Input image resolution |
| `paligemma_variant` | `gemma_2b` | VLM backbone variant |
| `action_expert_variant` | `gemma_300m` | Action expert variant |

## How It Works

### Inference Pipeline

1. **Image encoding**: Each camera image is resized to 224×224 (with aspect-ratio-preserving padding) and encoded by SigLIP into 256 visual tokens.

2. **Language encoding**: The instruction is tokenized (max 48 tokens) and embedded by Gemma 2B's embedding layer, scaled by √(hidden_dim).

3. **Prefix KV cache**: Image + language tokens are concatenated and forwarded through the PaliGemma backbone to produce KV cache.

4. **Denoising loop**: Starting from Gaussian noise x₁, for each step:
   - Embed state + noisy actions + timestep → suffix tokens
   - Forward suffix through action expert with prefix KV cache
   - Euler step: x_t = x_t + dt × v_t

5. **Output**: The denoised x₀ is the predicted action chunk (50 × 32).

### Attention Pattern

```
                 Images  Language  State  Actions
Images           ✓       ✓         ✗      ✗
Language         ✓       ✓         ✗      ✗
State            ✓       ✓         ✓      ✗
Actions          ✓       ✓         ✓      ✓ (causal)
```

- Prefix tokens (images + language): bidirectional attention
- State token: attends to prefix but prefix cannot attend to it
- Action tokens: causal attention among themselves, attend to everything above

## References

- [π0 Paper](https://arxiv.org/abs/2410.24164): "π₀: A Vision-Language-Action Flow Model for General Robot Control"
- [OpenPi Repository](https://github.com/Physical-Intelligence/openpi): Official implementation
- [LeRobot](https://github.com/huggingface/lerobot): HuggingFace port
- [π0 Weights](https://huggingface.co/lerobot/pi0_base): HuggingFace model hub