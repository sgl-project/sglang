"""SAIL GDN attention backend: PLA behind the GDN kernel slots.

The community ``GDNAttnBackend`` keeps its Triton / CuTe DSL / FlashInfer
implementations; this module only rebinds the Triton slots to the PLA-backed
kernel, which itself falls back to the community kernel for everything the
operator does not cover.
"""

from __future__ import annotations

from typing import Optional

from sglang.srt.hardware_backend.sail.attention.gdn_triton import SailTritonGDNKernel
from sglang.srt.layers.attention.linear.gdn_backend import (
    GDNAttnBackend,
    GDNKernelDispatcher,
)
from sglang.srt.layers.attention.linear.kernels.gdn_triton import TritonGDNKernel
from sglang.srt.layers.attention.linear.utils import LinearAttnKernelBackend


class SailGDNKernelDispatcher(GDNKernelDispatcher):
    """GDN kernel dispatcher whose Triton slots run on PLA."""

    def __init__(
        self,
        decode_backend: LinearAttnKernelBackend,
        prefill_backend: LinearAttnKernelBackend,
        verify_backend: Optional[LinearAttnKernelBackend] = None,
    ):
        super().__init__(decode_backend, prefill_backend, verify_backend)
        sail_triton = SailTritonGDNKernel()
        for slot in (
            "decode_kernel",
            "extend_kernel",
            "verify_kernel",
            "tree_verify_kernel",
        ):
            if isinstance(getattr(self, slot), TritonGDNKernel):
                setattr(self, slot, sail_triton)


class SailGDNAttnBackend(GDNAttnBackend):
    """GDNAttnBackend whose Triton kernel slots run on the PLA operators."""

    def _create_kernel_dispatcher(self, backends):
        return SailGDNKernelDispatcher(
            backends.decode, backends.prefill, backends.verify
        )
