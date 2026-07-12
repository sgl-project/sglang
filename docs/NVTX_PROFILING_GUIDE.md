# NVTX Profiling Guide for SGLang

This guide explains how to use NVIDIA NVTX (NVIDIA Tools Extension) markers
to profile SGLang scheduler performance with **NVIDIA Nsight Systems** (`nsys`).

## Quick Start

### 1. Enable NVTX markers

```bash
# Enable NVTX profiling markers in SGLang
export SGLANG_ENABLE_NVTX=1

# Launch your benchmark or workload as usual
python -m sglang.launch_server --model ... &
```

### 2. Profile with Nsight Systems

```bash
# Capture a profile of the scheduler
nsys profile -o sglang_profile --trace=nvtx,osrt,cuda,cudnn,cublas \
    python -m sglang.launch_server --model ...

# Or profile an already-running process by PID
nsys profile -o sglang_profile --trace=nvtx,osrt,cuda,cudnn,cublas \
    --attach-pid <PID>
```

### 3. View the profile

```bash
# Open the timeline UI
nsys-ui sglang_profile.nsys-rep
```

## What You'll See

The NVTX markers divide scheduler execution into color-coded ranges:

| Color | Marker | Phase | Description |
|-------|--------|-------|-------------|
| 🔵 Blue | `scheduler.recv_requests` | Input | Receiving and batching incoming requests |
| 🟣 Purple | `scheduler.process_input_requests` | Input | Processing and tokenizing input |
| 🟢 Green | `scheduler.get_next_batch_to_run` | Scheduling | Selecting the next batch for execution |
| 🔴 Red | `scheduler.run_batch` | Execution | Running the batch on GPU |
| 🔵 Cyan | `scheduler.process_batch_result` | Output | Processing and sending results |
| 🟤 Dark Blue | `scheduler.event_loop_normal` | Loop | Normal scheduler loop (no overlap) |
| 🟢 Teal | `scheduler.event_loop_overlap` | Loop | Overlapping CPU/GPU scheduler loop |
| 🟠 Orange | `scheduler.update_running_batch` | Scheduling | Updating the running decoding batch |
| 🟣 Magenta | `scheduler.get_new_batch_prefill` | Scheduling | Selecting new prefill batch |

### Event Loop Modes

SGLang supports two scheduler loop modes:

- **`event_loop_normal`** (dark blue): CPU processes one batch, waits for GPU, processes next.
  Simple and predictable, but the CPU is idle during GPU execution.

- **`event_loop_overlap`** (teal): CPU prepares the next batch while GPU is computing
  the current one. Better throughput when requests have variable lengths.

To select overlap mode:
```bash
export SGLANG_ENABLE_OVERLAP=1  # Default is 1 (enabled)
```

## Common Patterns

### Identifying bottlenecks

1. **Long `run_batch` (red)**: GPU kernel is slow — check model size, batch size,
   or tensor parallelism configuration.

2. **Long `process_batch_result` (cyan)**: Output processing bottleneck — large
   numbers of requests generating many tokens per step.

3. **Gap between `run_batch` end and next `recv_requests`**: CPU overhead —
   consider enabling overlap mode or tuning the scheduler.

4. **Short `get_next_batch_to_run`**: Scheduling overhead is minimal —
   the scheduler is not the bottleneck.

### Overlap effectiveness

Compare the idle gap between `run_batch` ranges:
- **Small/no gap**: Overlap is working well
- **Large gap**: Overlap isn't helping — check if requests are too short or
  batch sizes are too small to amortize the overlap benefit

## Troubleshooting

| Issue | Solution |
|-------|----------|
| No NVTX markers appear | Set `SGLANG_ENABLE_NVTX=1` and restart the server |
| `nsys` not found | Install CUDA toolkit which includes Nsight Systems |
| Profile is too large | Use `--duration` to limit capture time, or `--stop-on-exit` |
| Only CUDA events, no NVTX | Ensure NVTX is enabled *before* the Python process starts |
| Slow profiling overhead | Use `--trace=cuda,nvtx` instead of all trace types |

## Advanced Usage

### Selective profiling

Profile only specific request types by filtering NVTX ranges in the nsys UI:
use the search bar to filter by marker name.

### Exporting metrics

```bash
# Export to CSV for analysis
nsys export -f csv sglang_profile.nsys-rep > sglang_timeline.csv

# Summarize NVTX ranges
nsys stats --report nvtx_sum sglang_profile.nsys-rep
```

### Comparing configurations

```bash
# Profile without overlap
nsys profile -o baseline --trace=nvtx,cuda python ... &
# Profile with overlap
nsys profile -o overlap --trace=nvtx,cuda python ... &
# Compare
nsys-ui baseline.nsys-rep overlap.nsys-rep
```

## Integration with CI

To quickly validate NVTX markers don't regress:

```bash
SGLANG_ENABLE_NVTX=1 python -c "
import sglang.srt.managers.scheduler as s
print('NVTX markers loaded successfully')
"
```

## See Also

- `python/sglang/srt/utils/nvtx_utils.py` — NVTX utility implementation
- `docs/TORCH_FSDP2_TUTORIAL.md` — Existing profiling documentation
- [Nsight Systems Documentation](https://docs.nvidia.com/nsight-systems/)
