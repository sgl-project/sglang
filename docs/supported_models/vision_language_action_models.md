# Vision-Language-Action Models

Vision-Language-Action (VLA) models combine vision encoders with language models to predict robot actions from images and natural language instructions. They enable robots to understand visual scenes and follow human commands for manipulation tasks.

## Example Launch Command

```shell
python3 -m sglang.launch_server \
  --model-path openvla/openvla-7b \
  --trust-remote-code \
  --host 0.0.0.0 \
  --port 30000
```

## Sending Requests

VLA models use a specific prompt format and output action tokens instead of text:

```python
import requests
import base64

def image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

# Prepare request
image_b64 = image_to_base64("robot_scene.png")
instruction = "pick up the red block"

payload = {
    "text": f"In: What action should the robot take to {instruction}?\nOut:",
    "image_data": f"data:image/png;base64,{image_b64}",
    "sampling_params": {
        "max_new_tokens": 7,  # 7 action dimensions
        "temperature": 0,
    },
}

response = requests.post("http://localhost:30000/generate", json=payload)
result = response.json()

# Decode action tokens to continuous values
action_tokens = result["output_ids"][-7:]  # Last 7 tokens
actions = [(tok - 31744) / 255.0 * 2 - 1 for tok in action_tokens]
# actions = [dx, dy, dz, drx, dry, drz, gripper]
```

## Supported Models

| Model | Example HuggingFace Identifier | Description | Notes |
|-------|-------------------------------|-------------|-------|
| **OpenVLA** | `openvla/openvla-7b` | 7B VLA model with DINOv2 + SigLIP vision encoders and Llama-2-7B backbone. Trained on Open X-Embodiment dataset (970k episodes). Outputs 7-DoF actions (6 pose + 1 gripper). | Requires `--trust-remote-code` |

## Action Token Format

VLA models output discrete action tokens that map to continuous robot actions:

- **Token range**: 31744-31999 (last 256 tokens of Llama vocabulary)
- **Dimensions**: 7 tokens per action (dx, dy, dz, drx, dry, drz, gripper)
- **Value mapping**: `action = (token - 31744) / 255.0 * 2 - 1` gives values in [-1, 1]

## Equivalence with HuggingFace Transformers

SGLang's OpenVLA implementation achieves **4/5 (80%) exact token equivalence** with the reference HuggingFace Transformers implementation:

| Test | Result | Notes |
|------|--------|-------|
| Vision features | Exact match | DINOv2 + SigLIP outputs identical |
| Position encoding | Exact match | BOS at 0, vision at 1-256, text at 257+ |
| Token output | 4/5 samples exact | 1 sample differs due to low model confidence |

The single non-matching sample has very low confidence (0.375 logprob gap between top-1 and top-2 predictions), where small numerical differences can flip the output. The mismatch is only 1 bin difference, resulting in functionally equivalent actions.

### Running the Consistency Test

```bash
# Start server
python -m sglang.launch_server --model openvla/openvla-7b --trust-remote-code --port 30000

# Run consistency test
python test/manual/test_openvla_consistency.py
```

## Performance

Benchmarked on NVIDIA L4 (24GB VRAM):

| Metric | Value |
|--------|-------|
| Latency (mean) | ~405 ms |
| Throughput | ~2.5 Hz |
| GPU Memory | ~15 GB |

Achieves real-time control rate (>2 Hz) required for robotic manipulation tasks.
