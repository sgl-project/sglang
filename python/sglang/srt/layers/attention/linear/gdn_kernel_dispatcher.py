import torch

from sglang.srt.environ import Envs
from sglang.srt.layers.attention.linear.kernel_backend import LinearAttnKernelCore
from sglang.srt.layers.attention.linear.kernels.triton_gdn import TritonGDNKernel
from sglang.srt.layers.attention.linear.utils import LinearAttnKernelBackend
from sglang.srt.utils import is_cuda
from sglang.srt.utils.common import rank0_log


class GDNKernelDispatcher:
    """Dispatches GDN kernel calls to the appropriate backend per mode.

    Resolves decode/extend/verify kernels based on the configured backend,
    with AUTO resolution using CuTe DSL for decode on CUDA when the env var
    is set, and Triton for everything else.
    """

    def __init__(self, backend: LinearAttnKernelBackend):
        triton_kernel = TritonGDNKernel()

        if backend.is_auto():
            self._resolve_auto(triton_kernel)
        elif backend.is_cutedsl():
            self._resolve_cutedsl(triton_kernel)
        elif backend.is_triton():
            self._decode_kernel = triton_kernel
            self._extend_kernel = triton_kernel
            self._verify_kernel = triton_kernel
        else:
            raise ValueError(f"Unsupported GDN kernel backend: {backend}")

        rank0_log(
            f"GDN kernel dispatcher: decode={self._decode_kernel.__class__.__name__}, "
            f"extend={self._extend_kernel.__class__.__name__}, "
            f"verify={self._verify_kernel.__class__.__name__}"
        )

    def _resolve_auto(self, triton_kernel: LinearAttnKernelCore):
        """AUTO: Use CuTe DSL for decode on CUDA if env var is set, else Triton."""
        use_cutedsl = is_cuda() and Envs.SGLANG_USE_CUTEDSL_GDN_DECODE.get()
        if use_cutedsl:
            from sglang.srt.layers.attention.linear.kernels.cutedsl_gdn import (
                CuteDSLGDNKernel,
            )

            self._decode_kernel = CuteDSLGDNKernel()
        else:
            self._decode_kernel = triton_kernel
        self._extend_kernel = triton_kernel
        self._verify_kernel = triton_kernel

    def _resolve_cutedsl(self, triton_kernel: LinearAttnKernelCore):
        """CUTEDSL: Use CuTe DSL for decode, Triton for extend/verify."""
        if not is_cuda():
            raise ValueError("CuTe DSL backend requires CUDA")
        from sglang.srt.layers.attention.linear.kernels.cutedsl_gdn import (
            CuteDSLGDNKernel,
        )

        self._decode_kernel = CuteDSLGDNKernel()
        self._extend_kernel = triton_kernel
        self._verify_kernel = triton_kernel

    def decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        *,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        return self._decode_kernel.decode(
            q,
            k,
            v,
            a,
            b,
            A_log=A_log,
            dt_bias=dt_bias,
            ssm_states=ssm_states,
            cache_indices=cache_indices,
            query_start_loc=query_start_loc,
            **kwargs,
        )

    def extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        *,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        **kwargs,
    ) -> tuple:
        return self._extend_kernel.extend(
            q,
            k,
            v,
            g,
            beta,
            ssm_states=ssm_states,
            cache_indices=cache_indices,
            query_start_loc=query_start_loc,
            **kwargs,
        )

    def target_verify(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        *,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        return self._verify_kernel.target_verify(
            q,
            k,
            v,
            g,
            beta,
            ssm_states=ssm_states,
            cache_indices=cache_indices,
            query_start_loc=query_start_loc,
            **kwargs,
        )
