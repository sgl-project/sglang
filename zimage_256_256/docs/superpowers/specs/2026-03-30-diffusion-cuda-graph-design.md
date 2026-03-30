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
| Model switching | Disabled | ZImage `boundary_ratio=None`, all steps use same transformer |
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
        # 1. Update forward context (CPU-side state)
        # 2. Copy timestep/latents into pre-allocated buffers
        # 3. graph.replay()
        # 4. Read from output buffer
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

    Preconditions enforced at capture time:
    - Attention backend must be FlashAttention (no timestep-dependent kernel paths)
    - boundary_timestep must be None (no mid-request model switching)
    """

    def __init__(self, device: torch.device):
        self.device = device
        self.graph: Optional[torch.cuda.CUDAGraph] = None
        # Shared memory pool — avoids fragmentation across captures.
        # Mirrors LLM-side CudaGraphRunner behavior.
        self.pool = torch.cuda.graph_pool_handle()
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

        # Build the callable that graph will capture.
        # ZImage expects hidden_states and encoder_hidden_states as
        # List[Tensor]. We wrap the fixed-address buffer in a list each
        # call. The list wrapper is transient Python overhead; the
        # underlying tensor addresses are stable — which is what CUDA
        # Graph cares about.
        def run_fn():
            return dit_forward_fn(
                hidden_states=[latent_buffer],
                timestep=timestep_buffer,
                **static_kwargs,
            )

        # Warmup: 2 eager runs to stabilize PyTorch caching allocator.
        # Run 1 may trigger new CUDA memory allocations.
        # Run 2 confirms steady state (no new allocations).
        # This is critical: if the allocator's state differs between
        # warmup and capture, the graph may record incorrect addresses.
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
| `timestep` | GPU tensor (scalar `[B]`) | `timestep_buffer.copy_(new_timestep)` |
| `hidden_states` (latents) | `List[torch.Tensor]` — ZImage wraps a single `[C, H, W]` tensor in a list. Buffer is the inner tensor; list wrapper is recreated each call (transient Python object, not captured by graph). | `latent_buffer.copy_(new_latents)` |

### Fixed Across All Steps (bound at capture time)

| Input | Type | Why Fixed |
|---|---|---|
| `encoder_hidden_states` | `List[torch.Tensor]` — same list format as `hidden_states`. Comes from `pos_cond_kwargs['encoder_hidden_states']` which is `batch.prompt_embeds` (typed `list[torch.Tensor]`). Inner tensor addresses are baked into graph. | Same prompt for all steps |
| `guidance` | GPU tensor (scalar) | Constant (ZImage: `0`) |
| `freqs_cis` | `Tuple[torch.Tensor, torch.Tensor]` | RoPE embeddings, shape-dependent only |
| `patch_size`, `f_patch_size` | Python int | Fixed per resolution |
| Model weights | GPU tensors | Inference only |

### CPU-Side State (not in graph, updated freely)

| State | Where Used | Impact on Graph |
|---|---|---|
| `_forward_context.current_timestep` | Python global via `get_forward_context()` | None for FlashAttention. Must still be updated for profiling/logging correctness. |
| `attn_metadata` | Python dataclass | `None` for FlashAttention (`_build_attn_metadata` returns `None` when no sparse backend is configured) |
| `batch.is_cfg_negative` | Python attribute | Not applicable (ZImage: no CFG) |

## Integration with Denoising Stage

Modifications to `denoising.py`'s `forward()` method:

```python
# In _prepare_denoising_loop or at top of forward():
cuda_graph_enabled = server_args.enable_diffusion_cuda_graph  # new flag
graph_runner = None

if cuda_graph_enabled:
    # === Runtime safety checks ===
    # These assert the preconditions documented in the Scope table.
    # If any fails, the user gets a clear error instead of silent
    # incorrect results from replaying a mismatched graph.
    assert self.attn_backend.get_enum() == AttentionBackendEnum.FLASH_ATTN, (
        f"Diffusion CUDA Graph requires FlashAttention backend, "
        f"got {self.attn_backend.get_enum()}. Sparse attention backends "
        f"(STA, VSA, SVG2) use current_timestep for kernel path selection "
        f"which is incompatible with graph replay."
    )
    assert boundary_timestep is None, (
        f"Diffusion CUDA Graph does not support mid-request model switching "
        f"(boundary_timestep={boundary_timestep}). All denoising steps must "
        f"use the same transformer."
    )

    graph_runner = DiffusionCudaGraphRunner(device=get_local_torch_device())
    # Pre-allocate fixed-address buffers matching expected shapes
    timestep_buffer = torch.empty_like(timesteps[0].repeat(bsz))
    latent_buffer = torch.empty_like(latents)

    # Collect static kwargs for dit.forward().
    # These are filtered through prepare_extra_func_kwargs() which
    # inspects dit.forward()'s signature — only accepted params pass.
    # For ZImage: encoder_hidden_states, guidance, patch_size,
    # f_patch_size, freqs_cis.
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
        # Maintain forward context for profiling/logging consistency,
        # even though FlashAttention does not read current_timestep.
        with set_forward_context(current_timestep=i, attn_metadata=None, forward_batch=batch):
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

- CUDA Graph capture uses a shared memory pool (`torch.cuda.graph_pool_handle()`), matching the LLM-side `CudaGraphRunner` pattern. This avoids memory fragmentation when multiple graphs coexist (future: multi-resolution).
- Pre-allocated input/output buffers persist for the request lifetime. For ZImage 256×256 with batch=1, this is small (latent: [1, 16, 32, 32] = 64KB in BF16).
- Memory overhead is bounded: one graph + buffers per active resolution.

## Correctness Guarantees

1. **Step 0 captures and also produces valid output** — the capture run is a real forward pass, its output is used.
2. **`scheduler.step()` stays outside graph** — free to allocate tensors, branch, etc.
3. **Runtime asserts before capture** — attention backend must be FlashAttention; `boundary_timestep` must be `None`. These catch misuse at startup rather than producing silent incorrect results.
4. **`set_forward_context` set in both capture and replay** — maintains CPU-side state consistency for profiling, logging, and any code that reads the forward context.
5. **Warmup runs (2×) before capture** — ensures PyTorch's caching allocator is in steady state. Run 1 may trigger new allocations; Run 2 confirms no further allocations occur. If the allocator state differs between warmup and capture, the graph could record incorrect memory addresses.

## Expected Performance

| Metric | BF16 Eager | FP8 Eager | FP8 + CUDA Graph |
|---|---|---|---|
| GPU kernel time | Long | ~0.5× of BF16 | ~0.5× of BF16 |
| CPU launch overhead | Low | >2× of BF16 | ~0 (single launch) |
| GPU bubble | Minimal | Large | None |
| Overall latency | Baseline | Worse than BF16 | Should beat BF16 |

## Future Extensions

1. **Multiple resolution buckets**: Cache graphs keyed by `(height, width)` in a dict, each with its own input/output buffers.
2. **Dynamic capture + caching**: First request per resolution captures; subsequent requests replay from cache.
3. **CFG support**: Capture two forward passes (positive + negative) in one graph, or capture them as two separate graphs.
4. **Cache-DiT compatibility**: Capture different graphs for "full compute" vs "cached" steps; select at replay time based on Cache-DiT's per-step decision.
5. **torch.compile integration**: Use `torch.compile(mode="reduce-overhead")` as an alternative backend for graph capture with automatic kernel fusion.
6. **Sparse attention backends**: Would require either (a) asserting `current_timestep` does not change the kernel path, or (b) capturing separate graphs per distinct kernel path and selecting at replay time.
