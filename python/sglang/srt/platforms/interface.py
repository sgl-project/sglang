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
