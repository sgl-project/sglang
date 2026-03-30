# Diffusion Denoising CUDA Graph — Design Spec

## Problem

FP8 quantization on ZImage-Turbo (256×256, seq_len=768) is slower than BF16 because:

- FP8 GEMM requires extra kernels per Linear layer (quantize → GEMM → dequantize), increasing total kernel count.
- Each individual FP8 GPU kernel runs ~2× faster than BF16, but CPU-side kernel launch latency is >2× longer due to higher kernel count and launch gaps.
- Result: GPU bubbles between kernels — GPU is idle waiting for CPU to launch the next kernel.

**Root cause**: CPU-bound kernel launch overhead dominates when GPU compute per kernel is small.

## Solution

Capture one full DiT forward pass (`dit.forward()`) into a CUDA Graph. On replay, all kernels execute from a single `cudaGraphLaunch()` call, eliminating all inter-kernel CPU launch latency.

**Graph boundary** — matches the LLM-side pattern exactly:

```
Graph captures:  dit.forward()  (model inference only)
Graph excludes:  scheduler.step(), timestep expansion, CFG combine, profiling
```

## Scope

First implementation targets the narrowest useful case:

| Constraint | Value | Reason |
|---|---|---|
| Model | ZImage-Turbo | Primary optimization target |
| Resolution | 256×256 (seq_len=768) | Fixed single shape first |
| Batch size | 1 | Server serves one request at a time per worker |
| Attention | FlashAttention | No `current_timestep` dependency in GPU compute |
| CFG | Disabled (`should_use_guidance=False`) | One forward pass per step |
| Cache-DiT | Disabled | Deferred — every step executes the same kernel sequence |
| Scheduler | Outside graph | Avoids dynamic tensor allocation issues |

## Architecture

### Data Flow Per Denoising Step

```
[CPU side]                              [GPU side]

timestep_buffer.copy_(t_device)  ──────► ┌─────────────────────┐
latent_buffer.copy_(latents)     ──────► │                     │
                                         │   CUDA Graph        │
(constant: encoder_hidden_states,        │   ┌───────────────┐ │
 guidance, freqs_cis, patch_size,        │   │ dit.forward() │ │
 f_patch_size — bound at capture)        │   │               │ │
                                         │   │ FP8 quantize  │ │
                                         │   │ FP8 GEMM      │ │
                                         │   │ FP8 dequant   │ │
                                         │   │ attention     │ │
                                         │   │ norm, ffn     │ │
                                         │   │ ...×N layers  │ │
                                         │   └───────┬───────┘ │
                                         │           │         │
                                         └───────────┼─────────┘
                                                     ▼
noise_pred = output_buffer       ◄────── output_buffer

[CPU side — outside graph]
latents = scheduler.step(noise_pred, t, latents)
```

### Denoising Loop Structure

```python
for i, t_host in enumerate(timesteps_cpu):
    t_device = timesteps[i]
    timestep = expand_timestep(t_device)           # CPU side
    latent_model_input = prepare_latent_input()     # CPU side

    if i == 0 and cuda_graph_enabled:
        # === CAPTURE ===
        # 1. Allocate fixed-address buffers
        # 2. Copy inputs into buffers
        # 3. Warmup run (2×) to stabilize memory
        # 4. Capture dit.forward() into CUDA Graph
        # 5. Extract output buffer reference
        noise_pred = capture_and_run(...)
    elif cuda_graph_enabled and graph_captured:
        # === REPLAY ===
        # 1. Copy timestep/latents into pre-allocated buffers
        # 2. graph.replay()
        # 3. Read from output buffer
        noise_pred = replay(timestep, latent_model_input)
    else:
        # === EAGER FALLBACK ===
        noise_pred = eager_forward(...)

    # Always outside graph
    latents = scheduler.step(noise_pred, t_device, latents)
```

### New Class: `DiffusionCudaGraphRunner`

Location: `python/sglang/multimodal_gen/runtime/managers/diffusion_cuda_graph_runner.py`

```python
class DiffusionCudaGraphRunner:
    """CUDA Graph runner for diffusion DiT forward pass.

    Captures a single DiT forward pass and replays it with updated inputs.
    Analogous to srt's CudaGraphRunner but for diffusion models.
    """

    def __init__(self, device: torch.device):
        self.device = device
        self.graph: Optional[torch.cuda.CUDAGraph] = None
        self.pool = None                    # shared memory pool
        self.input_buffers: dict = {}       # fixed-address input tensors
        self.output_buffer: Optional[torch.Tensor] = None
        self._captured = False

    @property
    def captured(self) -> bool:
        return self._captured

    def capture(
        self,
        dit_forward_fn: Callable,
        # Fixed-address input tensors (pre-allocated)
        timestep_buffer: torch.Tensor,
        latent_buffer: torch.Tensor,
        # Static inputs (addresses bound at capture time)
        static_kwargs: dict,
    ) -> torch.Tensor:
        """Capture one DiT forward pass.

        Args:
            dit_forward_fn: The model's forward callable.
            timestep_buffer: Pre-allocated tensor for timestep input.
            latent_buffer: Pre-allocated tensor for latent input.
            static_kwargs: All other model inputs (encoder_hidden_states,
                guidance, freqs_cis, etc.) — their tensor addresses are
                baked into the graph.

        Returns:
            Output tensor (noise_pred). Its address is fixed for replay.
        """
        self.input_buffers = {
            'timestep': timestep_buffer,
            'latent': latent_buffer,
        }

        # Build the callable that graph will capture
        # Note: ZImage expects hidden_states as List[Tensor].
        # We wrap the fixed-address buffer in a list each call.
        # The list itself is transient; the tensor address is stable.
        def run_fn():
            return dit_forward_fn(
                hidden_states=[latent_buffer],
                timestep=timestep_buffer,
                **static_kwargs,
            )

        # Warmup: 2 eager runs to stabilize memory allocator
        for _ in range(2):
            run_fn()
        torch.cuda.synchronize()

        # Capture
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, pool=self.pool):
            output = run_fn()

        self.output_buffer = output
        self._captured = True
        return output

    def replay(
        self,
        timestep: torch.Tensor,
        latents: torch.Tensor,
    ) -> torch.Tensor:
        """Update inputs and replay the captured graph.

        Args:
            timestep: New timestep value for this denoising step.
            latents: New latent input for this denoising step.

        Returns:
            noise_pred from the output buffer (same address, updated data).
        """
        assert self._captured, "Must call capture() first"

        # Copy new data into fixed-address buffers
        self.input_buffers['timestep'].copy_(timestep)
        self.input_buffers['latent'].copy_(latents)

        # Replay — all kernels launch from one cudaGraphLaunch
        self.graph.replay()

        return self.output_buffer

    def reset(self):
        """Release graph and buffers."""
        if self.graph is not None:
            del self.graph
            self.graph = None
        self.input_buffers.clear()
        self.output_buffer = None
        self._captured = False
```

## Inputs: What Changes vs. What Is Fixed

### Per-Step Changing Inputs (copy before replay)

| Input | Type | How Updated |
|---|---|---|
| `timestep` | GPU tensor (scalar or [B]) | `timestep_buffer.copy_(new_timestep)` |
| `hidden_states` (latents) | `List[torch.Tensor]` — ZImage wraps a single [C, H, W] tensor in a list. Buffer is the inner tensor; list wrapper is recreated each replay. | `latent_buffer.copy_(new_latents)` |

### Fixed Across All Steps (bound at capture time)

| Input | Type | Why Fixed |
|---|---|---|
| `encoder_hidden_states` | GPU tensor (list) | Same prompt for all steps |
| `guidance` | GPU tensor (scalar) | Constant (ZImage: 0) |
| `freqs_cis` | GPU tensor tuple | RoPE embeddings, shape-dependent only |
| `patch_size`, `f_patch_size` | Python int | Fixed per resolution |
| Model weights | GPU tensors | Inference only |

### CPU-Side State (not in graph, updated freely)

| State | Where Used | Impact on Graph |
|---|---|---|
| `_forward_context.current_timestep` | Python global | None for FlashAttention |
| `attn_metadata` | Python dataclass | None for FlashAttention (returns None) |
| `batch.is_cfg_negative` | Python attribute | Not applicable (no CFG) |

## Integration with Denoising Stage

Modifications to `denoising.py`'s `forward()` method:

```python
# In _prepare_denoising_loop or at top of forward():
cuda_graph_enabled = server_args.enable_diffusion_cuda_graph  # new flag
graph_runner = None

if cuda_graph_enabled:
    graph_runner = DiffusionCudaGraphRunner(device=get_local_torch_device())
    # Pre-allocate fixed-address buffers matching expected shapes
    timestep_buffer = torch.empty_like(timesteps[0].repeat(bsz))
    latent_buffer = torch.empty_like(latents)

    # Collect static kwargs for dit.forward()
    static_kwargs = {
        'encoder_hidden_states': pos_cond_kwargs['encoder_hidden_states'],
        'guidance': guidance,
        'patch_size': server_args.pipeline_config.dit_config.patch_size,
        'f_patch_size': server_args.pipeline_config.dit_config.f_patch_size,
        'freqs_cis': pos_cond_kwargs.get('freqs_cis'),
    }
```

Then in the loop:

```python
for i, t_host in enumerate(timesteps_cpu):
    t_device = timesteps[i]
    timestep = self.expand_timestep_before_forward(batch, server_args, t_device, ...)
    latent_model_input = latents.to(target_dtype)
    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t_device)

    if cuda_graph_enabled and not graph_runner.captured:
        # Step 0: capture
        timestep_buffer.copy_(timestep)
        latent_buffer.copy_(latent_model_input)
        with set_forward_context(current_timestep=i, attn_metadata=None, forward_batch=batch):
            noise_pred = graph_runner.capture(
                dit_forward_fn=current_model,
                timestep_buffer=timestep_buffer,
                latent_buffer=latent_buffer,
                static_kwargs=static_kwargs,
            )
    elif cuda_graph_enabled and graph_runner.captured:
        # Step 1+: replay
        noise_pred = graph_runner.replay(timestep, latent_model_input)
    else:
        # Fallback eager
        noise_pred = self._predict_noise_with_cfg(...)

    # Outside graph — always eager
    latents = self.scheduler.step(
        model_output=noise_pred, timestep=t_device, sample=latents,
        **extra_step_kwargs, return_dict=False,
    )[0]
```

## Config & Enablement

New flag in `ServerArgs`:

```python
# server_args.py
enable_diffusion_cuda_graph: bool = False  # opt-in, default off
```

CLI usage:

```bash
sglang serve --model-path ZhipuAI/ZImage-Turbo \
    --enable-diffusion-cuda-graph \
    --quantization fp8
```

## Memory Considerations

- CUDA Graph capture allocates a private memory pool. The graph's internal allocations are reused across replays.
- Pre-allocated input/output buffers persist for the request lifetime. For ZImage 256×256 with batch=1, this is small (latent: [1, 16, 32, 32] = 64KB in BF16).
- Memory overhead is bounded: one graph + buffers per active resolution.

## Correctness Guarantees

1. **Step 0 captures and also produces valid output** — the capture run is a real forward pass, its output is used.
2. **`scheduler.step()` stays outside graph** — free to allocate tensors, branch, etc.
3. **`set_forward_context` only matters for sparse attention backends** — FlashAttention ignores it.
4. **Warmup runs (2×) before capture** — ensures PyTorch's caching allocator is in steady state, avoiding allocation pattern changes that would invalidate the graph.

## Expected Performance

| Metric | BF16 Eager | FP8 Eager | FP8 + CUDA Graph |
|---|---|---|---|
| GPU kernel time | Long | ~0.5× of BF16 | ~0.5× of BF16 |
| CPU launch overhead | Low | >2× of BF16 | ~0 (single launch) |
| GPU bubble | Minimal | Large | None |
| Overall latency | Baseline | Worse than BF16 | Should beat BF16 |

## Future Extensions

1. **Multiple resolution buckets**: Cache graphs keyed by `(height, width)` in a dict.
2. **Dynamic capture + caching**: First request per resolution captures; subsequent requests replay.
3. **CFG support**: Capture two forward passes (positive + negative) in one graph.
4. **Cache-DiT compatibility**: Capture different graphs for "full compute" vs "cached" steps.
5. **torch.compile integration**: Use `torch.compile(mode="reduce-overhead")` as an alternative backend for graph capture with automatic kernel fusion.
