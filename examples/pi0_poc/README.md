# pi0-style VLA action POC

This example exercises a minimal SGLang path for pi0-style continuous action
prediction. The POC model emits one dummy text token and returns the continuous
action chunk through `customized_info`, which surfaces as `meta_info.actions` on
the native `/generate` API and `sglext.actions` on OpenAI-compatible responses.

Create a tiny dummy model directory:

```bash
mkdir -p /tmp/pi0-poc
cat >/tmp/pi0-poc/config.json <<'JSON'
{
  "architectures": ["Pi0ForActionPrediction"],
  "model_type": "llama",
  "vocab_size": 32000,
  "hidden_size": 16,
  "num_hidden_layers": 1,
  "num_attention_heads": 1,
  "intermediate_size": 64,
  "eos_token_id": 0,
  "pad_token_id": 0,
  "pi0_dummy_token_id": 0,
  "pi0_action_dim": 14,
  "pi0_action_horizon": 50,
  "pi0_num_inference_steps": 10
}
JSON
```

Launch SGLang with dummy weights:

```bash
python -m sglang.launch_server \
  --model-path /tmp/pi0-poc \
  --tokenizer-path hf-internal-testing/llama-tokenizer \
  --load-format dummy \
  --host 127.0.0.1 \
  --port 30000
```

Run the native API client:

```bash
python examples/pi0_poc/client.py
```

The important request shape is:

```json
{
  "input_ids": [1, 2, 3],
  "sampling_params": {
    "max_new_tokens": 1,
    "temperature": 0,
    "custom_params": {
      "state": [0.0, 0.1, 0.2],
      "num_inference_steps": 8,
      "seed": 7
    }
  }
}
```

For a real pi0 port, replace `Pi0ForActionPrediction._sample_actions` with the
OpenPI-style PaliGemma prefix pass, action-expert suffix pass, and flow matching
denoising loop. Keep the serving contract the same: `max_new_tokens=1`,
continuous `actions` in `customized_info`, and robot state in `custom_params`.
