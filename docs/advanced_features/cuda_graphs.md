# CUDA Graph Capture

This page describes how CUDA Graph capture works in SGLang and how to tune it for latency, throughput, and memory.

## Overview
CUDA Graph capture records a representative forward pass and replays it with near-zero launch overhead. It mainly helps small to medium batch sizes by reducing per-token latency. SGLang builds multiple graphs for a configured set of batch sizes.

## When It Is Active
CUDA Graph is enabled by default (omit `--disable-cuda-graph`).
It targets smaller batch sizes (internal defaults) unless you extend the range. Larger batches fall back to normal execution.
You cannot serve requests in the server during the capture process.

### Selecting Batch Sizes
Control which batch sizes are captured:
- `--cuda-graph-bs <b1,b2,...>`: Explicit list of batch sizes to capture.
- `--cuda-graph-max-bs N`: Automatically extend the internal batch-size list up to `N`.
See full flag descriptions in [Server Arguments](server_arguments.md) (Optimization/debug section) for canonical definitions.

Capture runs largest -> smallest to maximize buffer reuse.

## Hidden State Capture Modes
Some features (returning hidden states, speculative decoding) need intermediate layer outputs. SGLang uses three modes:
- `NULL` (default when nothing needs hidden states): Do not capture hidden states.
- `LAST`: Capture only the last token's final-layer hidden state (saves memory vs full capture, still supports some draft/verification flows).
- `FULL`: Capture all hidden states for all tokens in the captured batch.

The runtime upgrades (recaptures) when a request needs a higher mode. It never downgrades. Example: starting at `NULL` then requesting hidden states triggers a recapture to `FULL`.

Guidance:
- Plain generation / chat: stay at `NULL` (maybe `LAST`).
- Need last token state only: `LAST`.
- Need embeddings / all layer outputs: first such request forces `FULL`.

## Garbage Collection and Memory
Capture allocates temporary tensors. Python GC can add overhead or fragmentation. SGLang optimizes by:
1. Running `gc.collect()` pre-capture.
2. Freezing GC during capture (default).
3. Unfreezing after.

Adjust with:
- `--enable-cudagraph-gc`: keep GC active (slower capture, slightly lower peak memory).

Recommendation: Keep it disabled unless you OOM during capture.

## Recapture Triggers
Recapture triggers:
- Request asks to return hidden states (`--enable-return-hidden-states`).
- Speculative decoding needs LAST or FULL.
- Draft/extend speculative flows.

## Memory Impact
Memory components: model weights + KV cache pool + CUDA graph buffers + activations.
Graph buffer size increases with: (1) number of captured batch sizes, (2) largest captured batch size, (3) hidden mode (`FULL > LAST > NULL`).
If OOM during capture, try (in order):
- Remove rarely used largest batch sizes or lower `--cuda-graph-max-bs`.
- Slightly reduce `--mem-fraction-static`.
- Enable `--enable-cudagraph-gc` (last resort; slows capture).
- Reduce hidden mode demand (avoid requesting hidden states unless needed).
- (If unset) Note: default `mem_fraction_static` is computed heuristically from GPU memory, parallel size, speculative decoding, and DP attention; see Hyperparameter Tuning for the formula.

## Disabling or Limiting CUDA Graph
Flags:
- `--disable-cuda-graph`: force eager path (debug/profiling/compat).
- `--disable-cuda-graph-padding`: skip when padding would be required.
- `--enable-profile-cuda-graph`: log capture timing and shapes.

## Troubleshooting
| Symptom | Possible Cause | Action |
|--------|----------------|--------|
| OOM during capture | Too many / too large captured batch sizes; FULL hidden mode | See above section on memory impact |
| Lower throughput than expected on small batches | CUDA Graph disabled or not capturing those sizes | Add them via `--cuda-graph-bs` or raise `--cuda-graph-max-bs` |
| Profiling needs kernel names | CUDA Graph obscures launch details | Use `--disable-cuda-graph` for profiling run |
| Quantization instability (e.g., int8dq issue) | Known interaction with capture | Temporarily disable CUDA Graph |

## Flags summary
- `--cuda-graph-bs`
- `--cuda-graph-max-bs`
- `--disable-cuda-graph`
- `--disable-cuda-graph-padding`
- `--enable-profile-cuda-graph`
- `--enable-cudagraph-gc`
- `--enable-return-hidden-states`

## Recommended Defaults
Recommended:
- Keep CUDA Graph enabled.
- Leave GC frozen (do not set `--enable-cudagraph-gc`).
- Avoid `FULL` unless you need embeddings or full hidden outputs.

## Example Launch Adjusting CUDA Graph Range
```bash
python -m sglang.launch_server \
  --model-path meta-llama/Meta-Llama-3-8B-Instruct \
  --cuda-graph-max-bs 384 \
  --mem-fraction-static 0.88
```

## FAQ
**Q: Does CUDA Graph help at very large batch sizes?**
Not usually; kernel launch overhead becomes relatively small. You can omit very large batches from capture.

**Q: Will enabling return hidden states always force FULL?**
Yes, the first time a request needs full hidden states the system upgrades and recaptures at FULL.

**Q: Can I revert to a lower mode to reclaim memory?**
Not dynamically. Restart the server with workloads that don't require FULL.

**Q: How do I know which batches are captured?**
Check startup logs or enable profiling; they enumerate captured batch sizes.
