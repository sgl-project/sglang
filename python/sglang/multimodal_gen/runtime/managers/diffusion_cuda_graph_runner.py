# SPDX-License-Identifier: Apache-2.0
"""
CUDA Graph runner for diffusion DiT forward pass.

Captures a single DiT forward pass and replays it with updated inputs.
Analogous to srt's CudaGraphRunner but scoped to diffusion models.

Graph boundary:
    Captures:  dit.forward()  (model inference only)
    Excludes:  scheduler.step(), timestep expansion, CFG combine, profiling

Cross-request caching:
    The runner is cached on the DenoisingStage instance keyed by latent
    shape. On the first request, warmup + capture occurs. On subsequent
    requests with the same shape, only buffer copies + replay happen.
    Per-request inputs (encoder_hidden_states, freqs_cis, guidance,
    timestep, latents) are copied into persistent fixed-address buffers
    before each replay.
"""

from typing import Callable, Optional

import torch

from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

_WARMUP_RUNS = 2


def _deep_copy_to_buffer(src, dst):
    """Copy tensor data from src into dst, handling nested structures.

    Supports: Tensor, List[Tensor], Tuple[Tensor, ...], and nested
    combinations. src and dst must have identical structure and shapes.
    """
    if isinstance(src, torch.Tensor):
        dst.copy_(src)
    elif isinstance(src, (list, tuple)):
        for s, d in zip(src, dst):
            _deep_copy_to_buffer(s, d)
    # Non-tensor types (int, float, None) — no copy needed


def _deep_clone_structure(obj):
    """Create persistent buffer copies of a nested tensor structure.

    Returns a structure with the same nesting but each tensor replaced
    by a .clone() (new memory address, stable for CUDA Graph).
    Non-tensor leaves are returned as-is.
    """
    if isinstance(obj, torch.Tensor):
        return obj.clone()
    elif isinstance(obj, list):
        return [_deep_clone_structure(x) for x in obj]
    elif isinstance(obj, tuple):
        return tuple(_deep_clone_structure(x) for x in obj)
    else:
        return obj


class DiffusionCudaGraphRunner:
    """CUDA Graph runner for diffusion DiT forward pass.

    Captures a single DiT forward pass and replays it with updated inputs.
    On replay, all GPU kernels execute from a single cudaGraphLaunch() call,
    eliminating CPU-side kernel launch latency between kernels.

    The runner is designed to be cached across requests. All inputs —
    including per-request ones like encoder_hidden_states — are held in
    persistent fixed-address buffers. New request data is copied into
    these buffers before replay.

    Preconditions (enforced by caller before capture):
    - Attention backend must NOT use current_timestep for kernel path selection
    - boundary_timestep must be None (no mid-request model switching)

    Usage:
        # First request: capture
        runner = DiffusionCudaGraphRunner(device)
        runner.capture(model, timestep, latents, static_kwargs)

        # Same request, subsequent steps:
        runner.replay(new_timestep, new_latents)

        # Next request (different prompt, same resolution):
        runner.update_static_kwargs(new_static_kwargs)
        runner.replay(timestep, latents)  # no re-capture needed
    """

    def __init__(self, device: torch.device):
        self.device = device
        self.graph: Optional[torch.cuda.CUDAGraph] = None
        # Shared memory pool — avoids fragmentation across captures.
        self.pool = torch.cuda.graph_pool_handle()
        self.input_buffers: dict[str, torch.Tensor] = {}
        # Persistent buffers for static kwargs (encoder_hidden_states, etc.)
        self.static_buffers: dict[str, object] = {}
        self.output_buffer: Optional[torch.Tensor] = None
        self._captured = False

    @property
    def captured(self) -> bool:
        return self._captured

    def capture(
        self,
        dit_forward_fn: Callable,
        timestep: torch.Tensor,
        latents: torch.Tensor,
        static_kwargs: dict,
    ) -> torch.Tensor:
        """Capture one DiT forward pass into a CUDA Graph.

        Must be called within a set_forward_context() block.

        Creates persistent buffers for ALL inputs (timestep, latents,
        and every tensor in static_kwargs). The graph records these
        buffer addresses. Subsequent calls to replay() or
        update_static_kwargs() copy new data into these buffers.

        Args:
            dit_forward_fn: The model's forward callable.
            timestep: Initial timestep tensor (will be cloned into buffer).
            latents: Initial latent tensor (will be cloned into buffer).
            static_kwargs: Per-request model inputs (encoder_hidden_states,
                guidance, freqs_cis, etc.). All tensors are cloned into
                persistent buffers.

        Returns:
            Output tensor (noise_pred). Address is fixed for replay.
        """
        # Create persistent buffers for dynamic inputs
        timestep_buffer = timestep.clone()
        latent_buffer = latents.clone()
        self.input_buffers = {
            "timestep": timestep_buffer,
            "latent": latent_buffer,
        }

        # Create persistent buffers for static kwargs.
        # Each tensor in static_kwargs is cloned so we own the memory
        # address. CUDA Graph bakes in these addresses.
        self.static_buffers = {}
        graph_static_kwargs = {}
        for key, val in static_kwargs.items():
            buf = _deep_clone_structure(val)
            self.static_buffers[key] = buf
            graph_static_kwargs[key] = buf

        # Build the callable that the graph will capture.
        # Note: ZImage's eager path passes hidden_states as a plain
        # tensor, not a list, despite the List[Tensor] type hint.
        def run_fn():
            return dit_forward_fn(
                hidden_states=latent_buffer,
                timestep=timestep_buffer,
                **graph_static_kwargs,
            )

        # Warmup: 2 eager runs to stabilize PyTorch's caching allocator.
        # Run 1 may trigger new CUDA memory allocations.
        # Run 2 confirms steady state (no new allocations).
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

    def update_static_kwargs(self, static_kwargs: dict):
        """Copy new per-request data into persistent static buffers.

        Call this once at the start of each new request (before the
        denoising loop) to update encoder_hidden_states, freqs_cis, etc.
        The buffer addresses stay the same — only data changes.

        Args:
            static_kwargs: New per-request static kwargs. Must have the
                same keys and tensor shapes as those used during capture.
        """
        assert self._captured, "Must call capture() before update_static_kwargs()"
        for key, new_val in static_kwargs.items():
            if key in self.static_buffers:
                _deep_copy_to_buffer(new_val, self.static_buffers[key])

    def replay(
        self,
        timestep: torch.Tensor,
        latents: torch.Tensor,
    ) -> torch.Tensor:
        """Update per-step inputs and replay the captured graph.

        Args:
            timestep: New timestep value for this denoising step.
            latents: New latent input for this denoising step.

        Returns:
            noise_pred from the output buffer.
        """
        assert self._captured, "Must call capture() before replay()"

        self.input_buffers["timestep"].copy_(timestep)
        self.input_buffers["latent"].copy_(latents)

        self.graph.replay()

        return self.output_buffer

    def reset(self):
        """Release the captured graph and all buffers."""
        if self.graph is not None:
            del self.graph
            self.graph = None
        self.input_buffers.clear()
        self.static_buffers.clear()
        self.output_buffer = None
        self._captured = False
