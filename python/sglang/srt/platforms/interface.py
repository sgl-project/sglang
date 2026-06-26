"""
SGLang SRT Hardware Platform Abstraction.

Defines SRTPlatform — the base class for SRT (LLM inference) platform
backends.  SRTPlatform inherits DeviceMixin for shared device operations
and adds SRT-specific subsystem factory methods, capability flags, and
configuration lifecycle hooks.

Out-of-tree platforms register via setuptools entry_points under the
"sglang.srt.platforms" group and should subclass SRTPlatform.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Type

from sglang.srt.platforms.device_mixin import DeviceMixin, PlatformEnum

if TYPE_CHECKING:
    from sglang.srt.layers.quantization.base_config import QuantizationConfig

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

    def get_graph_runner_cls(self) -> Optional[type]:
        """Return the graph runner class for this platform.

        Return None to let the caller use the generic in-tree default
        (DecodeCudaGraphRunner). In-tree platforms that need a different
        runner (CPU, NPU) override this; OOT platforms must provide one.
        """
        return None

    def get_mha_kv_pool_cls(self) -> Optional[type]:
        """Return the MHA KV pool class, or None for the in-tree default."""
        return None

    def get_mla_kv_pool_cls(self) -> Optional[type]:
        """Return the MLA KV pool class, or None for the in-tree default."""
        return None

    def get_dsa_kv_pool_cls(self) -> Optional[type]:
        """Return the DSA KV pool class (DeepSeek V3.2), or None for the default."""
        return None

    def get_paged_allocator_cls(self) -> Optional[type]:
        """Return the paged allocator class, or None for the in-tree default."""
        return None

    def get_compile_backend(self, mode: str | None = None) -> str:
        """Return the compilation backend identifier.

        ``mode`` is an optional hint for the platform (e.g. "npugraph_ex").
        """
        return "inductor"

    def get_piecewise_backend_cls(self) -> Optional[type]:
        """Return the piecewise backend class, or None for the in-tree default."""
        return None

    def get_quantization_config(
        self, quantization: str
    ) -> Optional[Type[QuantizationConfig]]:
        """Return hardware-specific quantization config for the specific
        quantization scheme, raise an error if not supported or return None
        to use the default config."""
        return None

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
