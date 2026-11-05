"""SAIL DeepEP token dispatcher.

The shared ``deepep.py`` maps one ``DispatcherOutputDtype`` onto a fixed set of
transport flags. SAIL's DeepEP build additionally speaks INT8 and UINT8
(MXFP4-packed) on the wire, quantizes FP8 per channel, pads normal dispatch to
alignment 1, and needs two buffer knobs of its own (MNNVL fabric support and the
dispatch timeout). None of that upstream supports, so it is kept here instead of
growing the shared dispatcher.

``DeepEPDispatcher.__init__`` instantiates these subclasses on SAIL; every other
platform keeps using ``_DeepEPDispatcherImplNormal`` /
``_DeepEPDispatcherImplLowLatency`` unchanged.
"""

import torch

from sglang.srt.environ import envs
from sglang.srt.layers import deep_gemm_wrapper
from sglang.srt.layers.moe.token_dispatcher import deepep
from sglang.srt.layers.moe.token_dispatcher.deepep import (
    DeepEPBuffer,
    _DeepEPDispatcherImplBase,
    _DeepEPDispatcherImplLowLatency,
    _DeepEPDispatcherImplNormal,
)
from sglang.srt.layers.moe.utils import DispatcherOutputDtype

# Vendor builds differ in whether the buffer exposes a timeout knob.
DEEPEP_SUPPORT_TIMEOUT_CONTROL = deepep.use_deepep and hasattr(
    deepep.Buffer, "set_timeout_seconds"
)


class SailDeepEPBuffer(DeepEPBuffer):
    """DeepEP comm buffer with SAIL's fabric flag and dispatch timeout."""

    @classmethod
    def mnnvl_fabric_supported(cls) -> bool:
        # SAIL advertises fabric support itself; the MNNVL probe behind the
        # shared implementation is NVIDIA-only and always fails here.
        return envs.SGLANG_SAIL_MNNVL_FABRIC_SUPPORTED.get()

    @classmethod
    def _configure_buffer(cls, buffer) -> None:
        # DeepEP's own all-to-all timeout is too short for SAIL's slower
        # dispatch, so reset it from the SAIL knob right after construction.
        if DEEPEP_SUPPORT_TIMEOUT_CONTROL:
            buffer.set_timeout_seconds(envs.SGLANG_SAIL_NORMAL_DISPATCH_TIMEOUT.get())


class SailDeepEPDispatcherImplBase(_DeepEPDispatcherImplBase):
    """Adds the SAIL buffer and the INT8/UINT8 dispatch dtypes to the impl."""

    _buffer_cls = SailDeepEPBuffer

    # INT8 must travel as int8 instead of falling back to fp8 like the shared
    # table does, and UINT8 reuses the MXFP4 path.
    dispatch_dtype_flags = {
        **_DeepEPDispatcherImplBase.dispatch_dtype_flags,
        DispatcherOutputDtype.INT8: {
            "use_fp8": False,
            "use_mxfp4": False,
            "use_mxfp8": False,
            "use_nvfp4": False,
            "use_int8": True,
        },
        DispatcherOutputDtype.UINT8: {
            "use_fp8": False,
            "use_mxfp4": True,
            "use_mxfp8": False,
            "use_nvfp4": False,
            "use_int8": False,
        },
    }

    def _validate_and_adjust_dtype(self) -> None:
        # SAIL dispatches int8 and uint8 natively; only nvfp4 is unsupported.
        if self.deepep_output_dtype == DispatcherOutputDtype.NVFP4:
            raise RuntimeError(
                "SAIL does not support nvfp4 deepep_dispatcher_output_dtype."
            )


class SailDeepEPDispatcherImplNormal(
    SailDeepEPDispatcherImplBase, _DeepEPDispatcherImplNormal
):
    """Normal dispatch with SAIL's fp8/int8/mxfp4 quantizers."""

    def _quantize_dispatch_input(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM and self.use_fp8:
            from sglang.kernels.ops.quantization.fp8_kernel import (
                sglang_per_token_group_quant_fp8,
            )

            # FP8 block-wise by default; FP8 channel-wise when requested via
            # the ``is_channel_quant`` flag in the quant config dict.
            is_channel_quant = self.quant_config.get("is_channel_quant", False)
            return sglang_per_token_group_quant_fp8(
                hidden_states,
                hidden_states.shape[-1] if is_channel_quant else 128,
                column_major_scales=deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,
                scale_tma_aligned=deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,
                scale_ue8m0=deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,
            )
        if self.use_int8:
            from sglang.kernels.ops.quantization.int8_kernel import per_token_quant_int8

            return per_token_quant_int8(hidden_states)
        if self.use_mxfp4:
            from triton_kernels.numerics_details.mxfp import downcast_to_mxfp

            # [WA] before deepep support uint16_t column major scale
            return downcast_to_mxfp(hidden_states, torch.uint8, axis=1)
        return hidden_states

    def _expert_alignment(self) -> int:
        # Aligning to 128 receives more tokens than needed, which SAIL's expert
        # kernels do not handle.
        return 1


class SailDeepEPDispatcherImplLowLatency(
    SailDeepEPDispatcherImplBase, _DeepEPDispatcherImplLowLatency
):
    """Low-latency dispatch with SAIL's int8/mxfp4 wire formats."""

    def _dispatch_quant_kwargs(self, hidden_states: torch.Tensor) -> dict:
        # FP8 block-wise by default; FP8 channel-wise when requested via
        # the ``is_channel_quant`` flag in the quant config dict.
        is_channel_quant = bool(
            self.use_int8 or self.quant_config.get("is_channel_quant", False)
        )
        quant_size = hidden_states.shape[-1] if is_channel_quant else 128
        if self.use_mxfp4:
            quant_size = 32
        return dict(
            use_int8=self.use_int8,
            use_mxfp4=self.use_mxfp4,
            quant_size=quant_size,
            mxfp4_scale_row_major=False,
        )
