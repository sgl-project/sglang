# Auto-Spec: Adaptive Speculative Decoding

Auto-Spec dynamically adjusts the speculative decoding depth (`num_steps`) at runtime based on acceptance rate feedback. Instead of using a fixed speculation depth for all workloads, it monitors how well the draft model's predictions are accepted and tunes the depth per batch size to maximize throughput.

## Quick Start

```bash
# Basic usage: add --auto-spec to any EAGLE speculative decoding launch
python -m sglang.launch_server \
    --model-path meta-llama/Llama-3.1-8B-Instruct \
    --speculative-algorithm EAGLE \
    --speculative-draft-model-path path/to/eagle-model \
    --speculative-num-steps 3 \
    --auto-spec
```

The `--speculative-num-steps` value serves as the initial default. Auto-Spec will adjust it up or down during inference.

## How It Works

```
                    ┌─────────────────────────┐
                    │   AutoSpecEngine        │
                    │                         │
 accept rate ──────►│  EMA tracker per BS     │
 feedback           │  threshold comparison   │
                    │  step adjustment        │
                    └───────────┬─────────────┘
                                │ best num_steps
                                ▼
                    ┌─────────────────────────┐
                    │  EAGLEWorker /          │
                    │  EAGLEWorkerV2          │
                    │                         │
                    │  switch CUDA graphs     │
                    │  switch attn backends   │
                    └─────────────────────────┘
```

1. **Startup**: For each candidate `num_steps` value, the system captures a set of CUDA graphs (draft, draft-extend, verify) and attention backends.
2. **Runtime**: After each decode batch, the scheduler feeds the acceptance rate to the engine.
3. **Decision**: The engine uses EMA-smoothed acceptance rates per batch size to decide:
   - **Rate ≥ increase threshold** → try a larger `num_steps` (more aggressive speculation)
   - **Rate < decrease threshold** → fall back to a smaller `num_steps` (more conservative)
4. **Switching**: Before each decode forward pass, the worker checks the engine's recommendation and hot-swaps CUDA graphs + attention backends if the optimal `num_steps` has changed.

## Command-Line Arguments

| Argument | Type | Default | Description |
|---|---|---|---|
| `--auto-spec` | flag | `false` | Enable adaptive speculative decoding |
| `--speculative-config-file` | `str` | `None` | Path to JSON config with per-model step ranges |
| `--pos-threshold` | `float...` | built-in | Acceptance rate thresholds for *increasing* num_steps, one per BS bucket |
| `--neg-threshold` | `float...` | built-in | Acceptance rate thresholds for *decreasing* num_steps, one per BS bucket |

### Threshold Arguments

The `--pos-threshold` and `--neg-threshold` accept space-separated floats, one per batch size bucket in ascending order. The default batch size buckets are `[1, 2, 4, 8, 16, 32, 64, 128]`.

```bash
# Example: custom thresholds for 8 batch size buckets
python -m sglang.launch_server \
    --model-path ... \
    --speculative-algorithm EAGLE \
    --speculative-draft-model-path ... \
    --auto-spec \
    --pos-threshold 0.55 0.55 0.60 0.80 0.95 0.91 0.95 0.95 \
    --neg-threshold 0.50 0.50 0.50 0.50 0.60 0.66 0.65 0.65
```

- **pos-threshold**: If the EMA acceptance rate exceeds this value, `num_steps` is increased.
- **neg-threshold**: If the EMA acceptance rate drops below this value, `num_steps` is decreased.
- Higher thresholds make the tuner more conservative (less likely to increase steps).

## Speculative Config File

You can define per-model step ranges in a JSON file:

```json
{
    "meta-llama/Llama-3.1-8B-Instruct": {
        "1": [3, 4, 5, 6],
        "2": [3, 4, 5, 6],
        "4": [3, 4, 5, 6],
        "8": [2, 3, 4],
        "16": [2, 3, 4],
        "32": [2, 3, 4],
        "64": [1, 2, 3],
        "128": [1, 2]
    }
}
```

Each key is a batch size, and the value is the list of `num_steps` values the engine is allowed to choose from for that batch size. Use with:

```bash
--speculative-config-file /path/to/config.json
```

If the file is not provided or the model name is not found, built-in defaults are used.

## Default Step Ranges

| Batch Size | Allowed num_steps | Initial num_steps |
|---|---|---|
| 1 | 3, 4, 5, 6 | 5 |
| 2 | 3, 4, 5, 6 | 5 |
| 4 | 3, 4, 5, 6 | 5 |
| 8 | 2, 3, 4 | 4 |
| 16 | 2, 3, 4 | 3 |
| 32 | 2, 3, 4 | 3 |
| 64 | 1, 2, 3, 4 | 3 |
| 128 | 1, 2, 3, 4 | 1 |

The intuition: small batches benefit from deeper speculation (more tokens drafted per step), while large batches should speculate conservatively (draft overhead grows linearly with batch size).

## Compatibility

- **Speculative algorithms**: EAGLE, EAGLE3
- **topk**: Auto-Spec enforces `topk=1` (required for multi-step graph switching)
- **Spec V1 and V2**: Both `EAGLEWorker` (V1) and `EAGLEWorkerV2` (V2/SpecV2) are supported
- **CUDA graphs**: Required (auto-spec pre-captures graphs for each step count)
- **Tensor parallelism**: Supported
- **Data parallelism**: Supported

## Memory Considerations

Auto-Spec captures CUDA graphs for multiple `num_steps` values, which increases GPU memory usage. The engine automatically limits the number of step variants based on available memory at startup (~2GB per additional step set).

If memory is tight, you can reduce the step range via the config file:

```json
{
    "my-model": {
        "1": [3, 5],
        "8": [2, 3],
        "32": [2, 3],
        "128": [1, 2]
    }
}
```

## Example: Full Deployment

```bash
python -m sglang.launch_server \
    --model-path meta-llama/Llama-3.1-70B-Instruct \
    --speculative-algorithm EAGLE \
    --speculative-draft-model-path path/to/eagle-70b \
    --speculative-num-steps 4 \
    --auto-spec \
    --tp 4 \
    --max-running-requests 128 \
    --speculative-config-file ./spec_config.json
```

## Troubleshooting

**Q: Startup is slow with auto-spec enabled.**
A: This is expected. Auto-spec captures CUDA graphs for each `num_steps` variant at startup. You can reduce the number of variants in the config file to speed up startup.

**Q: How do I know which num_steps is being used?**
A: The decode log line already prints `accept len` and `accept rate`. The engine adjusts based on these metrics. You can also check the server logs for `AutoSpec` messages that show initialization parameters.

**Q: Auto-spec makes throughput worse for my workload.**
A: This can happen if the default thresholds don't suit your model/dataset. Try:
1. Adjusting `--pos-threshold` / `--neg-threshold`
2. Narrowing the step range in the config file
3. Running with a fixed `--speculative-num-steps` that you've benchmarked as optimal
