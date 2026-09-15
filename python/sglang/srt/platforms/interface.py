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

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Type

from sglang.srt.platforms.device_mixin import DeviceMixin, PlatformEnum

if TYPE_CHECKING:
    from sglang.srt.layers.quantization.base_config import QuantizationConfig

# Re-export for convenience
__all__ = ["HostMemoryMapping", "SRTPlatform", "PlatformEnum"]


@dataclass(frozen=True)
class HostMemoryMapping:
    """A device-visible view of a host-memory range.

    ``registered_host_ptr`` is the (possibly page-aligned) address that must be
    passed back to the runtime. ``owned`` prevents callers from unregistering a
    mapping that was created by another component, such as torch_npu's pinned
    allocator.
    """

    device_ptr: int
    registered_host_ptr: int
    owned: bool


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

    def get_dsa_kv_pool_cls(self) -> type:
        """Return the DSA KV pool class for this platform (DeepSeek V3.2)."""
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
    # Host memory visible to device kernels
    # ------------------------------------------------------------------

    def register_host_memory(
        self, host_ptr: int, nbytes: int
    ) -> Optional[HostMemoryMapping]:
        """Return a device-visible mapping for the range at ``host_ptr``.

        Some kernels read a table straight out of host memory. On devices with
        unified addressing (CUDA/ROCm UVA, and the pageable-access devices the
        ``file`` PLE backend targets) the raw host pointer is already usable, so
        the default returns ``None`` and callers keep using ``host_ptr``.

        Platforms whose kernels cannot dereference a host address -- an Ascend
        kernel faults with "The DDR address of the MTE instruction is out of
        range" -- must override this and map the range, then pass the returned
        address to the kernel.

        Raises RuntimeError when this platform needs a mapping but cannot make
        one; callers should surface that instead of falling back to the raw
        pointer.
        """
        return None

    def unregister_host_memory(self, mapping: HostMemoryMapping) -> None:
        """Release an owned mapping returned by :meth:`register_host_memory`."""
        pass

    def get_max_kernel_grid_size(self) -> Optional[int]:
        """Upper bound on a 1-D kernel launch grid, or ``None`` when unbounded.

        Ascend rejects launches whose grid (``coreDim``) exceeds 65535, so
        kernels that size their grid from the input must clamp it. CUDA has no
        comparable limit.
        """
        return None

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def init_backend(self) -> None:
        """One-time backend initialization.  Called in each worker."""
        pass

    # ------------------------------------------------------------------
    # BaseFusedOp integration
    # ------------------------------------------------------------------

    def get_dispatch_key_name(self) -> str:
        """Return the dispatch key name for BaseFusedOp
        (``sglang.kernels.fused_op``).

        Determines which ``forward_<key>()`` method is selected on an
        out-of-tree platform. E.g. "cuda", "npu", "hip", "xpu", "cpu".
        Forwards registered via ``BaseFusedOp.register_oot_forward`` with
        this key take precedence over the method lookup.
        """
        return "native"
