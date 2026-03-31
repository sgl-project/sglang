# Diffusion CUDA Graph Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Capture the DiT forward pass into a CUDA Graph to eliminate CPU-side kernel launch latency for FP8 quantized ZImage-Turbo inference.

**Architecture:** New `DiffusionCudaGraphRunner` class handles capture/replay of `dit.forward()`. The denoising loop in `denoising.py` gains a three-way branch: capture (step 0), replay (step 1+), or eager fallback. `scheduler.step()` stays outside the graph. Enabled via `--enable-diffusion-cuda-graph` CLI flag.

**Tech Stack:** PyTorch CUDA Graphs (`torch.cuda.CUDAGraph`), SGLang diffusion runtime

**Spec:** `zimage_256_256/docs/superpowers/specs/2026-03-30-diffusion-cuda-graph-design.md`

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `python/sglang/multimodal_gen/runtime/managers/diffusion_cuda_graph_runner.py` | **Create** | `DiffusionCudaGraphRunner` class — capture, replay, reset |
| `python/sglang/multimodal_gen/runtime/server_args.py` | **Modify** (line ~162) | Add `enable_diffusion_cuda_graph: bool = False` flag |
| `python/sglang/multimodal_gen/runtime/pipelines_core/stages/denoising.py` | **Modify** (lines 60, 992, 1012-1090) | Import runner, add CUDA graph branch to denoising loop |

---

## Task 1: Add `enable_diffusion_cuda_graph` Flag to ServerArgs

**Files:**
- Modify: `python/sglang/multimodal_gen/runtime/server_args.py:162`

- [ ] **Step 1: Add the flag**

In `python/sglang/multimodal_gen/runtime/server_args.py`, after line 161 (`enable_torch_compile: bool = False`), add:

```python
    # CUDA Graph for diffusion denoising (captures dit.forward())
    enable_diffusion_cuda_graph: bool = False
```

The resulting block (lines 160-163) should read:

```python
    # Compilation
    enable_torch_compile: bool = False

    # CUDA Graph for diffusion denoising (captures dit.forward())
    enable_diffusion_cuda_graph: bool = False
```

No argparse changes needed — SGLang's `ServerArgs` is a `@dataclass` and argument registration at line 1094 handles it automatically.

- [ ] **Step 2: Commit**

```bash
git add python/sglang/multimodal_gen/runtime/server_args.py
git commit -m "feat(diffusion): add enable_diffusion_cuda_graph flag to ServerArgs"
```

---

## Task 2: Create `DiffusionCudaGraphRunner` Class

**Files:**
- Create: `python/sglang/multimodal_gen/runtime/managers/diffusion_cuda_graph_runner.py`

- [ ] **Step 1: Create the file**

Create `python/sglang/multimodal_gen/runtime/managers/diffusion_cuda_graph_runner.py` with the following content:

```python
# SPDX-License-Identifier: Apache-2.0
"""
CUDA Graph runner for diffusion DiT forward pass.

Captures a single DiT forward pass and replays it with updated inputs.
Analogous to srt's CudaGraphRunner but scoped to diffusion models.

Graph boundary:
    Captures:  dit.forward()  (model inference only)
    Excludes:  scheduler.step(), timestep expansion, CFG combine, profiling
"""

from typing import Callable, Optional

import torch

from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

_WARMUP_RUNS = 2


class DiffusionCudaGraphRunner:
    """CUDA Graph runner for diffusion DiT forward pass.

    Captures a single DiT forward pass and replays it with updated inputs.
    On replay, all GPU kernels execute from a single cudaGraphLaunch() call,
    eliminating CPU-side kernel launch latency between kernels.

    Preconditions (enforced by caller before capture):
    - Attention backend must NOT use current_timestep for kernel path selection
      (FlashAttention, FA2, TORCH_SDPA are safe; STA, VSA, SVG2 are not)
    - boundary_timestep must be None (no mid-request model switching)

    Usage:
        runner = DiffusionCudaGraphRunner(device)
        # Step 0: capture (within set_forward_context)
        output = runner.capture(model, timestep_buf, latent_buf, static_kwargs)
        # Step 1+: replay
        output = runner.replay(new_timestep, new_latents)
    """

    def __init__(self, device: torch.device):
        self.device = device
        self.graph: Optional[torch.cuda.CUDAGraph] = None
        # Shared memory pool — avoids fragmentation across captures.
        # Mirrors LLM-side CudaGraphRunner behavior.
        self.pool = torch.cuda.graph_pool_handle()
        self.input_buffers: dict[str, torch.Tensor] = {}
        self.output_buffer: Optional[torch.Tensor] = None
        self._captured = False

    @property
    def captured(self) -> bool:
        return self._captured

    def capture(
        self,
        dit_forward_fn: Callable,
        timestep_buffer: torch.Tensor,
        latent_buffer: torch.Tensor,
        static_kwargs: dict,
    ) -> torch.Tensor:
        """Capture one DiT forward pass into a CUDA Graph.

        Must be called within a set_forward_context() block to ensure
        warmup runs and the capture run share the same forward context.
        The context is read by attention backends and profiling code
        during the warmup eager executions.

        Args:
            dit_forward_fn: The model's forward callable (e.g. transformer).
            timestep_buffer: Pre-allocated tensor for timestep input.
                Its address is baked into the graph; data is updated
                via .copy_() before each replay.
            latent_buffer: Pre-allocated tensor for latent input.
                Same address-stability requirement as timestep_buffer.
            static_kwargs: All other model inputs (encoder_hidden_states,
                guidance, freqs_cis, patch_size, f_patch_size, etc.).
                Their tensor addresses are baked into the graph and must
                not change between capture and replay.

        Returns:
            Output tensor (noise_pred, shape [B, C, H, W]).
            Its address is fixed — on replay the same tensor object
            is populated with new data.
        """
        self.input_buffers = {
            "timestep": timestep_buffer,
            "latent": latent_buffer,
        }

        # Build the callable that the graph will capture.
        # ZImage expects hidden_states and encoder_hidden_states as
        # List[Tensor]. We wrap the fixed-address buffer in a list
        # each call. The list wrapper is transient Python overhead;
        # the underlying tensor addresses are stable — which is what
        # CUDA Graph cares about.
        def run_fn():
            return dit_forward_fn(
                hidden_states=[latent_buffer],
                timestep=timestep_buffer,
                **static_kwargs,
            )

        # Warmup: run eager forward passes to stabilize PyTorch's
        # caching allocator. Run 1 may trigger new CUDA memory
        # allocations; Run 2 confirms steady state (no new allocations).
        # If the allocator's state differs between warmup and capture,
        # the graph may record incorrect memory addresses.
        logger.info(
            "CUDA Graph: warming up with %d eager runs before capture",
            _WARMUP_RUNS,
        )
        for _ in range(_WARMUP_RUNS):
            run_fn()
        torch.cuda.synchronize()

        # Capture
        logger.info("CUDA Graph: capturing dit.forward()")
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, pool=self.pool):
            output = run_fn()

        self.output_buffer = output
        self._captured = True
        logger.info("CUDA Graph: capture complete")
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
            noise_pred from the output buffer (same tensor object,
            updated data after replay).
        """
        assert self._captured, "Must call capture() before replay()"

        # Copy new data into fixed-address buffers.
        # These copies are the only GPU work before the graph launch.
        self.input_buffers["timestep"].copy_(timestep)
        self.input_buffers["latent"].copy_(latents)

        # Replay — all kernels launch from one cudaGraphLaunch
        self.graph.replay()

        return self.output_buffer

    def reset(self):
        """Release the captured graph and all buffers."""
        if self.graph is not None:
            del self.graph
            self.graph = None
        self.input_buffers.clear()
        self.output_buffer = None
        self._captured = False
```

- [ ] **Step 2: Commit**

```bash
git add python/sglang/multimodal_gen/runtime/managers/diffusion_cuda_graph_runner.py
git commit -m "feat(diffusion): add DiffusionCudaGraphRunner for DiT CUDA Graph capture/replay"
```

---

## Task 3: Integrate CUDA Graph into Denoising Loop

**Files:**
- Modify: `python/sglang/multimodal_gen/runtime/pipelines_core/stages/denoising.py`
  - Lines 60 (imports)
  - Lines 992-993 (after prepared_vars unpacking)
  - Lines 1054-1077 (noise prediction block, replace with three-way branch)

This is the core integration. We modify the denoising loop to: (1) validate preconditions, (2) set up buffers, (3) capture on step 0, (4) replay on step 1+.

- [ ] **Step 1: Add import**

In `python/sglang/multimodal_gen/runtime/pipelines_core/stages/denoising.py`, after line 60 (`from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context`), add:

```python
from sglang.multimodal_gen.runtime.managers.diffusion_cuda_graph_runner import (
    DiffusionCudaGraphRunner,
)
```

- [ ] **Step 2: Add CUDA Graph setup after prepared_vars unpacking**

After line 992 (`guidance = prepared_vars["guidance"]`), add the following block. This goes before the trajectory initialization (before line 994 `trajectory_timesteps: list[torch.Tensor] = []`):

```python
        # CUDA Graph setup for denoising
        cuda_graph_enabled = server_args.enable_diffusion_cuda_graph
        graph_runner = None
        if cuda_graph_enabled:
            # Runtime safety checks — catch misuse at startup, not silently
            # produce incorrect results from replaying a mismatched graph.
            _TIMESTEP_DEPENDENT_BACKENDS = {
                AttentionBackendEnum.SLIDING_TILE_ATTN,
                AttentionBackendEnum.VIDEO_SPARSE_ATTN,
                AttentionBackendEnum.SPARSE_VIDEO_GEN_2_ATTN,
                AttentionBackendEnum.VMOBA_ATTN,
            }
            assert self.attn_backend.get_enum() not in _TIMESTEP_DEPENDENT_BACKENDS, (
                f"Diffusion CUDA Graph is incompatible with attention backend "
                f"'{self.attn_backend.get_enum().name}'. These backends use "
                f"current_timestep for kernel path selection, which cannot "
                f"change after graph capture. Use FlashAttention instead."
            )
            assert boundary_timestep is None, (
                f"Diffusion CUDA Graph does not support mid-request model "
                f"switching (boundary_timestep={boundary_timestep}). All "
                f"denoising steps must use the same transformer."
            )

            graph_runner = DiffusionCudaGraphRunner(
                device=get_local_torch_device()
            )
            # Pre-allocate fixed-address buffers matching expected shapes.
            # These tensors' addresses are baked into the CUDA Graph.
            bsz = batch.raw_latent_shape[0]
            timestep_buffer = torch.empty(
                bsz,
                dtype=target_dtype,
                device=get_local_torch_device(),
            )
            latent_buffer = torch.empty_like(latents)

            # Collect static kwargs for dit.forward().
            # For ZImage: encoder_hidden_states (List[Tensor]), guidance,
            # patch_size, f_patch_size, freqs_cis.
            # prepare_extra_func_kwargs() filters by dit.forward() signature,
            # so only accepted params are included.
            static_kwargs = self.prepare_extra_func_kwargs(
                getattr(self.transformer, "forward", self.transformer),
                {
                    "encoder_hidden_states": pos_cond_kwargs.get(
                        "encoder_hidden_states"
                    ),
                    "guidance": guidance,
                    "patch_size": server_args.pipeline_config.dit_config.patch_size,
                    "f_patch_size": getattr(
                        server_args.pipeline_config.dit_config, "f_patch_size", 1
                    ),
                    "freqs_cis": pos_cond_kwargs.get("freqs_cis"),
                },
            )
            logger.info(
                "Diffusion CUDA Graph enabled. Static kwargs: %s",
                list(static_kwargs.keys()),
            )
```

- [ ] **Step 3: Replace noise prediction block with three-way branch**

Replace the block from line 1054 to line 1077 (the `attn_metadata` build + `_predict_noise_with_cfg` call):

**Original code (lines 1054-1077):**
```python
                        # Predict noise residual
                        attn_metadata = self._build_attn_metadata(
                            i,
                            batch,
                            server_args,
                            timestep_value=t_int,
                            timesteps=timesteps_cpu,
                        )
                        noise_pred = self._predict_noise_with_cfg(
                            current_model=current_model,
                            latent_model_input=latent_model_input,
                            timestep=timestep,
                            batch=batch,
                            timestep_index=i,
                            attn_metadata=attn_metadata,
                            target_dtype=target_dtype,
                            current_guidance_scale=current_guidance_scale,
                            image_kwargs=image_kwargs,
                            pos_cond_kwargs=pos_cond_kwargs,
                            neg_cond_kwargs=neg_cond_kwargs,
                            server_args=server_args,
                            guidance=guidance,
                            latents=latents,
                        )
```

**Replacement code:**
```python
                        # Predict noise residual
                        if (
                            cuda_graph_enabled
                            and not graph_runner.captured
                        ):
                            # === Step 0: CUDA Graph Capture ===
                            # Copy initial data into fixed-address buffers
                            timestep_buffer.copy_(timestep)
                            latent_buffer.copy_(latent_model_input)
                            with set_forward_context(
                                current_timestep=i,
                                attn_metadata=None,
                                forward_batch=batch,
                            ):
                                noise_pred = graph_runner.capture(
                                    dit_forward_fn=current_model,
                                    timestep_buffer=timestep_buffer,
                                    latent_buffer=latent_buffer,
                                    static_kwargs=static_kwargs,
                                )
                        elif cuda_graph_enabled and graph_runner.captured:
                            # === Step 1+: CUDA Graph Replay ===
                            # Maintain forward context for profiling/logging
                            # consistency, even though FlashAttention does
                            # not read current_timestep.
                            with set_forward_context(
                                current_timestep=i,
                                attn_metadata=None,
                                forward_batch=batch,
                            ):
                                noise_pred = graph_runner.replay(
                                    timestep, latent_model_input
                                )
                        else:
                            # === Eager fallback ===
                            attn_metadata = self._build_attn_metadata(
                                i,
                                batch,
                                server_args,
                                timestep_value=t_int,
                                timesteps=timesteps_cpu,
                            )
                            noise_pred = self._predict_noise_with_cfg(
                                current_model=current_model,
                                latent_model_input=latent_model_input,
                                timestep=timestep,
                                batch=batch,
                                timestep_index=i,
                                attn_metadata=attn_metadata,
                                target_dtype=target_dtype,
                                current_guidance_scale=current_guidance_scale,
                                image_kwargs=image_kwargs,
                                pos_cond_kwargs=pos_cond_kwargs,
                                neg_cond_kwargs=neg_cond_kwargs,
                                server_args=server_args,
                                guidance=guidance,
                                latents=latents,
                            )
```

- [ ] **Step 4: Commit**

```bash
git add python/sglang/multimodal_gen/runtime/pipelines_core/stages/denoising.py
git commit -m "feat(diffusion): integrate CUDA Graph capture/replay into denoising loop

Adds three-way branch in the denoising loop:
- Step 0: capture dit.forward() into CUDA Graph
- Step 1+: replay graph with updated timestep/latents
- Fallback: eager mode when CUDA Graph is disabled

Runtime asserts enforce preconditions:
- Attention backend must not depend on current_timestep
- boundary_timestep must be None (no model switching)"
```

---

## Task 4: Cleanup and Final Verification

**Files:**
- All three modified files

- [ ] **Step 1: Verify no syntax errors with static check**

```bash
cd /data/home/rhyshen/sgl-workspace/sglang
python -c "
import ast, sys
files = [
    'python/sglang/multimodal_gen/runtime/managers/diffusion_cuda_graph_runner.py',
    'python/sglang/multimodal_gen/runtime/server_args.py',
    'python/sglang/multimodal_gen/runtime/pipelines_core/stages/denoising.py',
]
for f in files:
    try:
        with open(f) as fh:
            ast.parse(fh.read())
        print(f'OK: {f}')
    except SyntaxError as e:
        print(f'FAIL: {f}: {e}')
        sys.exit(1)
print('All files parse successfully.')
"
```

Expected output:
```
OK: python/sglang/multimodal_gen/runtime/managers/diffusion_cuda_graph_runner.py
OK: python/sglang/multimodal_gen/runtime/server_args.py
OK: python/sglang/multimodal_gen/runtime/pipelines_core/stages/denoising.py
All files parse successfully.
```

- [ ] **Step 2: Verify import chain works**

```bash
cd /data/home/rhyshen/sgl-workspace/sglang
python -c "
from sglang.multimodal_gen.runtime.managers.diffusion_cuda_graph_runner import DiffusionCudaGraphRunner
print(f'DiffusionCudaGraphRunner imported successfully')
print(f'Methods: {[m for m in dir(DiffusionCudaGraphRunner) if not m.startswith(\"_\")]}')
"
```

Expected output:
```
DiffusionCudaGraphRunner imported successfully
Methods: ['capture', 'captured', 'replay', 'reset']
```

- [ ] **Step 3: Final commit with all changes**

If any fixups were needed during verification:

```bash
git add -u
git commit -m "fix(diffusion): address verification issues in CUDA Graph integration"
```
