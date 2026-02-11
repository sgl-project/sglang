import torch

from sglang.srt.layers.attention.linear.kernel_backend import LinearAttnKernelBase
from sglang.srt.layers.attention.linear.kernels.triton_gdn import TritonGDNKernel
from sglang.srt.layers.attention.linear.utils import LinearAttnKernelBackend
from sglang.srt.utils import is_cuda
from sglang.srt.utils.common import rank0_log


class GDNKernelDispatcher:
    """Dispatches GDN kernel calls to the appropriate backend per mode.

    Resolves decode and prefill/extend/verify kernels independently based on
    the configured backends.
    """

    def __init__(
        self,
        decode_backend: LinearAttnKernelBackend,
        prefill_backend: LinearAttnKernelBackend,
    ):
        triton_kernel = TritonGDNKernel()

        self._decode_kernel = self._resolve_kernel(decode_backend, triton_kernel)
        self._extend_kernel = self._resolve_kernel(prefill_backend, triton_kernel)
        self._verify_kernel = triton_kernel

        rank0_log(
            f"GDN kernel dispatcher: decode={self._decode_kernel.__class__.__name__}, "
            f"extend={self._extend_kernel.__class__.__name__}, "
            f"verify={self._verify_kernel.__class__.__name__}"
        )

    def _resolve_kernel(
        self,
        backend: LinearAttnKernelBackend,
        triton_kernel: LinearAttnKernelBase,
    ) -> LinearAttnKernelBase:
        if backend.is_triton():
            return triton_kernel
        elif backend.is_cutedsl():
            if not is_cuda():
                raise ValueError("CuTe DSL backend requires CUDA")
            from sglang.srt.layers.attention.linear.kernels.cutedsl_gdn import (
                CuteDSLGDNKernel,
            )

            return CuteDSLGDNKernel()
        else:
            raise ValueError(f"Unsupported GDN kernel backend: {backend}")

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
