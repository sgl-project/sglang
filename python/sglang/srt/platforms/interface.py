"""
SGLang SRT Hardware Platform Abstraction.

Defines SRTPlatform — the base class for SRT (LLM inference) platform
backends.  SRTPlatform inherits DeviceMixin for shared device operations
and adds SRT-specific subsystem factory methods, a capability declaration,
and configuration lifecycle hooks.

Out-of-tree platforms register via setuptools entry_points under the
"sglang.srt.platforms" group and should subclass SRTPlatform.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Optional, Type

import msgspec

from sglang.srt.platforms.device_mixin import DeviceMixin, PlatformEnum

if TYPE_CHECKING:
    import torch

    from sglang.srt.layers.quantization.base_config import QuantizationConfig

# Re-export for convenience
__all__ = [
    "KVPoolKind",
    "PlatformCapabilities",
    "SRTPlatform",
    "PlatformEnum",
    "require_out_of_tree_impl",
    "reject_out_of_tree_path",
]

KVPoolKind = Literal["mha", "mla", "dsa"]


class PlatformCapabilities(msgspec.Struct, frozen=True, kw_only=True):
    """What the platform can do.

    Core reads these before it constructs anything and picks a code path
    the platform can serve, instead of announcing a decision and asking the
    platform to honor it. Every field defaults to the conservative answer.

    ``supports_triton``: Triton kernels can launch on this device. When
    False, core uses the torch-native allocator and req-to-token writers.
    ``graph_capture``: the decode graph runner (``get_graph_runner_cls``)
    runs. ``piecewise_graph``: the prefill piecewise compilation backend
    runs.
    ``hicache_device_kernels``: the sgl_kernel HiCache transfer / write-back
    kernels are available; otherwise the host pools skip their staging
    buffers.
    """

    supports_triton: bool = False
    graph_capture: bool = False
    piecewise_graph: bool = False
    hicache_device_kernels: bool = False


class SRTPlatform(DeviceMixin):
    """
    Base class for SRT hardware platform backends.

    Inherits device identity queries and operations from DeviceMixin.
    Adds SRT-specific factory methods, a capability declaration, and
    lifecycle hooks.

    OOT platforms subclass SRTPlatform and override the methods relevant to
    their hardware.
    """

    # SRT-specific class-level attributes
    supported_quantization: list[str] = []
    capabilities: PlatformCapabilities = PlatformCapabilities()

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
        """Return the graph runner class, or None for the in-tree default.

        ``None`` means "no platform opinion": the caller falls back to the
        in-tree device-keyed selection.
        """
        return None

    def get_kv_pool_cls(self, *, kind: KVPoolKind) -> Optional[type]:
        """Return this platform's KV pool class for ``kind``, or None.

        The class must subclass the in-tree pool for that kind
        (``MHATokenToKVPool`` / ``MLATokenToKVPool`` / ``DSATokenToKVPool``)
        and is constructed by the configurator with exactly the keyword
        arguments the in-tree class receives, at every site the in-tree
        class is built (standalone, the SWA composite, the hybrid-linear
        full-attention leaf). Override ``_create_buffers`` to change how the
        backing storage is carved; the rest of the pool (addressing, index
        translation, PD registration) stays core's. ``None`` (the default)
        means the in-tree class, which allocates on ``device`` and is
        correct for any torch device.
        """
        return None

    def get_paged_allocator_cls(self) -> Optional[type]:
        """Return the paged allocator class, or None for the in-tree default.

        Honored at every paged-allocator construction site, including
        inside ``SWATokenToKVPoolAllocator``. The in-tree allocator already
        falls back to torch-native kernels when ``capabilities.supports_triton``
        is False, so most platforms need no override.
        """
        return None

    def get_compile_backend(self, mode: str | None = None) -> str:
        """Return the compilation backend identifier.

        ``mode`` is an optional hint for the platform (e.g. "npugraph_ex").
        """
        return "inductor"

    def get_piecewise_backend_cls(self) -> Optional[type]:
        """Return the piecewise compilation backend class, or None for the
        in-tree device-keyed default."""
        return None

    def get_quantization_config(
        self, quantization: str
    ) -> Optional[Type[QuantizationConfig]]:
        """Return hardware-specific quantization config for the specific
        quantization scheme, raise an error if not supported or return None
        to use the default config."""
        return None

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def init_backend(self) -> None:
        """One-time backend initialization.  Called in each worker."""
        pass

    def post_load_model(self, model: torch.nn.Module) -> None:
        """Called once after the model's weights are loaded, in the worker.

        The place to relocate parameters, swap quant methods, or install
        forward hooks on an in-tree model class; mutate ``model`` in place.
        """
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


def require_out_of_tree_impl(
    platform: SRTPlatform, *, hook: str, subsystem: str
) -> None:
    """Reject "no platform opinion" from an out-of-tree platform; no-op in-tree."""
    if not platform.is_out_of_tree():
        return
    raise NotImplementedError(
        f"Out-of-tree platform {type(platform).__name__} (device "
        f"{platform.device_name!r}) provides no {subsystem}. Override "
        f"{hook} on your SRTPlatform subclass: the in-tree fallback is "
        f"device-keyed and would hand this device the CUDA implementation."
    )


def reject_out_of_tree_path(platform: SRTPlatform, *, subsystem: str) -> None:
    """Reject an in-tree-only path that has no platform seam; no-op in-tree."""
    if not platform.is_out_of_tree():
        return
    raise NotImplementedError(
        f"Out-of-tree platform {type(platform).__name__} (device "
        f"{platform.device_name!r}) reached {subsystem}, which has no platform "
        f"seam yet and would otherwise build in-tree classes that assume CUDA."
    )
