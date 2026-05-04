"""
SGLang SRT Hardware Platform Abstraction.

Defines SRTPlatform — the base class for SRT (LLM inference) platform
backends.  SRTPlatform inherits DeviceMixin for shared device operations
and adds SRT-specific subsystem factory methods, capability flags, and
configuration lifecycle hooks.

Out-of-tree platforms register via setuptools entry_points under the
"sglang.platform_plugins" group and should subclass SRTPlatform.
"""

from typing import TYPE_CHECKING

import torch

from sglang.srt.platforms.device_mixin import DeviceMixin, PlatformEnum

if TYPE_CHECKING:
    pass

# Re-export for convenience
__all__ = ["SRTPlatform", "PlatformEnum"]


class SRTPlatform(DeviceMixin):
    """
    Base class for SRT hardware platform backends.

    Inherits device identity queries and operations from DeviceMixin.
    Adds SRT-specific factory methods, capability flags, and lifecycle hooks.

    OOT platforms should subclass SRTPlatform and override the methods
    relevant to their hardware.
    """

    # SRT-specific class-level attribute
    supported_quantization: list[str] = []

    # ------------------------------------------------------------------
    # Configuration lifecycle
    # ------------------------------------------------------------------

    def apply_server_args_defaults(self, server_args) -> None:
        """Apply platform-specific default values to server arguments.

        Called after ServerArgs is parsed.
        """
        pass

    # ------------------------------------------------------------------
    # Subsystem factory methods
    # ------------------------------------------------------------------

    def get_default_attention_backend(self) -> str:
        """Return the default attention backend name for this platform."""
        raise NotImplementedError

    def get_graph_runner_cls(self) -> type:
        """Return the graph runner class for this platform."""
        raise NotImplementedError

    def get_mha_kv_pool_cls(self) -> type:
        """Return the MHA KV pool class for this platform."""
        raise NotImplementedError

    def get_mla_kv_pool_cls(self) -> type:
        """Return the MLA KV pool class for this platform."""
        raise NotImplementedError

    def get_nsa_kv_pool_cls(self) -> type:
        """Return the NSA KV pool class for this platform (DeepSeek V3.2)."""
        raise NotImplementedError

    def get_paged_allocator_cls(self) -> type:
        """Return the paged allocator class for this platform."""
        raise NotImplementedError

    def get_compile_backend(self, mode: str | None = None) -> str:
        """Return the compilation backend identifier.

        ``mode`` is an optional hint for the platform (e.g. "npugraph_ex").
        """
        return "inductor"

    def get_piecewise_backend_cls(self) -> type:
        """Return the piecewise compilation backend class for this platform."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Capability flags (safe conservative defaults)
    # ------------------------------------------------------------------

    def supports_fp8(self) -> bool:
        """Whether this platform supports FP8 quantization."""
        return False

    def is_pin_memory_available(self) -> bool:
        """Whether pinned memory is available on this platform."""
        return True

    def support_cuda_graph(self) -> bool:
        """Whether this platform supports device graph capture and replay.
        Controls CUDA graph (CudaGraphRunner) for the decode path.
        OOT platforms that support graph-style capture should return True.
        """
        return False

    def support_piecewise_cuda_graph(self) -> bool:
        """Whether this platform supports piecewise CUDA graph.

        Controls PiecewiseCudaGraphRunner for the prefill/extend path
        (torch.compile backend).
        """
        return False

    def supports_torch_compile(self) -> bool:
        """Whether this platform supports generic torch.compile usage."""
        return True

    def use_pynccl_by_default(self, backend: str) -> bool:
        """Return whether model-parallel groups should enable PyNccl by default.

        The base implementation preserves the existing built-in platform checks.
        OOT platforms may override this method to adjust that decision.
        """
        return not (self.is_npu() or self.is_xpu() or backend == "mooncake")

    def should_use_fallback_rotary_embedding(self, *, head_size: int) -> bool:
        """Return whether RotaryEmbedding should use the fallback implementation."""
        return (
            (not self.is_cuda() or head_size not in [64, 128, 256, 512])
            and not self.is_cpu()
            and not self.is_xpu()
            and not self.is_npu()
            and not self.is_musa()
            and not self.is_mps()
        )

    def get_group_coordinator_device(
        self, local_rank: int, *, one_visible_device_per_process: bool = False
    ) -> torch.device:
        """Return the device used by GroupCoordinator metadata tensors."""
        if self.is_cuda_alike():
            device_id = 0 if one_visible_device_per_process else local_rank
            return torch.device(f"cuda:{device_id}")
        if self.is_npu():
            return torch.device(f"npu:{local_rank}")
        if self.is_xpu():
            return torch.device(f"xpu:{local_rank}")
        if self.is_musa():
            return torch.device(f"musa:{local_rank}")
        return torch.device("cpu")

    def stream_context(self, device_module, stream):
        """Return a context manager that selects the given stream."""
        stream_context = getattr(device_module, "StreamContext", None)
        if stream_context is not None:
            return stream_context(stream)
        return device_module.stream(stream)

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def init_backend(self) -> None:
        """One-time backend initialization.  Called in each worker."""
        pass

    # ------------------------------------------------------------------
    # MultiPlatformOp integration
    # ------------------------------------------------------------------

    def get_dispatch_key_name(self) -> str:
        """Return the dispatch key name for MultiPlatformOp.

        Determines which ``forward_<key>()`` method is selected.
        E.g. "cuda", "npu", "hip", "xpu", "cpu".
        """
        return "native"
