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
