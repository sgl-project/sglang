from __future__ import annotations

from typing import TYPE_CHECKING, Literal, NamedTuple, Optional, Union

import torch

from sglang.jit_kernel.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)
from sglang.srt.utils import is_hip, is_xpu

from .utils import make_name

_is_xpu = is_xpu()
if _is_xpu:
    from sgl_kernel import compress_norm_rope_store as compress_norm_rope_store_xpu
    from sgl_kernel import (
        flash_compress4_decode,
        flash_compress4_prefill,
        flash_compress128_decode,
        flash_compress128_prefill,
        plan_compress_decode,
        plan_compress_decode_legacy,
        plan_compress_prefill,
        plan_compress_prefill_legacy,
    )

    _XPU_COMPRESS_FNS = {
        4: (flash_compress4_decode, flash_compress4_prefill),
        128: (flash_compress128_decode, flash_compress128_prefill),
    }


if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_compress_norm_rope_module(
    dtype: torch.dtype,
    head_dim: int,
    rope_dim: int,
    page_size: int,
    bf16_store: bool = False,
) -> Module:
    args = make_cpp_args(
        dtype, head_dim, rope_dim, page_size, is_arch_support_pdl(), bf16_store
    )
    cuda_wrappers = [("forward", f"FusedNormRopeKernel<{args}>::forward")]
    if head_dim == 128:
        cuda_wrappers.append(
            ("forward_fp4", f"FusedNormRopeKernel<{args}>::forward_fp4")
        )
    return load_jit(
        make_name(f"fused_norm_rope_v2"),
        *args,
        cuda_files=[f"deepseek_v4/fused_norm_rope_v2.cuh"],
        cuda_wrappers=cuda_wrappers,
    )


@cache_once
def _jit_compress_module(
    head_dim: int,
    dtype_buffer: torch.dtype,
    dtype_in: torch.dtype,
    dtype_out: torch.dtype,
    ratio: Literal[4, 128],
) -> Module:
    args = make_cpp_args(
        head_dim, dtype_buffer, dtype_in, dtype_out, is_arch_support_pdl()
    )
    kernel_class = f"FlashCompress{ratio}Kernel<{args}>"
    return load_jit(
        make_name(f"compress_{ratio}_v2"),
        *args,
        cuda_files=[f"deepseek_v4/c{ratio}_v2.cuh"],
        cuda_wrappers=[
            ("decode", f"{kernel_class}::run_decode"),
            ("prefill", f"{kernel_class}::run_prefill"),
        ],
        extra_cuda_cflags=["-use_fast_math"],
    )


@cache_once
def _jit_compress_128_online_module(
    head_dim: int, dtype_buffer: torch.dtype = torch.float32
) -> Module:
    assert head_dim == 512
    args = make_cpp_args(head_dim, dtype_buffer, is_arch_support_pdl())
    kernel_class = f"FlashCompress128OnlineKernel<{args}>"
    return load_jit(
        make_name(f"compress_128_online_v2"),
        *args,
        cuda_files=["deepseek_v4/c128_online_v2.cuh"],
        cuda_wrappers=[
            ("decode", f"{kernel_class}::run_decode"),
            ("prefill", f"{kernel_class}::run_prefill"),
            ("plan_decode", "plan_compress_128_online_decode"),
            ("plan_prefill", "plan_compress_128_online_prefill"),
        ],
        extra_cuda_cflags=["-use_fast_math"],
    )


@cache_once
def _jit_compress_plan_module() -> Module:
    return load_jit(
        make_name(f"compress_plan"),
        cuda_files=[f"deepseek_v4/c_plan.cuh"],
        cuda_wrappers=[
            ("plan_prefill", "plan_compress_prefill"),
            ("plan_decode", "plan_compress_decode"),
            ("plan_prefill_legacy", "plan_compress_prefill_legacy"),
            ("plan_decode_legacy", "plan_compress_decode_legacy"),
        ],
    )


# ----------------------------------------------------------------------------
# Plan tensor sizes (must match the C++ structs in compress.cuh).
# ----------------------------------------------------------------------------
_PREFILL_PLAN_BYTES = 24


# ----------------------------------------------------------------------------
# Plan dataclasses. The element at index 1 is the consumer for
# `compress_fused_norm_rope_inplace` (which reads ragged_id / seq_len from a
# 16-byte plan tensor --- both DecodePlan and CompressPlan satisfy that layout).
# ----------------------------------------------------------------------------


class CompressorDecodePlan(NamedTuple):
    compress_ratio: int
    plan_d: torch.Tensor  # [batch_size, 16] uint8 --- DecodePlan

    def copy_(self, other) -> None:
        assert isinstance(other, CompressorDecodePlan)
        assert self.compress_ratio == other.compress_ratio
        self.plan_d.copy_(other.plan_d)

    @staticmethod
    def generate(
        compress_ratio: Literal[4, 128],
        req_pool_indices: torch.Tensor,
        req_to_token: torch.Tensor,
        full_to_state: torch.Tensor,
        seq_lens: torch.Tensor,
        swa_page_size: int,
        ring_size: int,
    ) -> CompressorDecodePlan:
        if _is_xpu:
            fn = plan_compress_decode
        else:
            module = _jit_compress_plan_module()
            fn = module.plan_decode

        plan_d = fn(
            req_pool_indices,
            req_to_token,
            full_to_state,
            seq_lens,
            int(compress_ratio),
            int(swa_page_size),
            int(ring_size),
        )
        return CompressorDecodePlan(compress_ratio, torch.from_dlpack(plan_d))

    @staticmethod
    def generate_legacy(
        compress_ratio: Literal[4, 128],
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
    ) -> CompressorDecodePlan:
        if _is_xpu:
            fn = plan_compress_decode_legacy
        else:
            module = _jit_compress_plan_module()
            fn = module.plan_decode_legacy

        plan_d = fn(req_pool_indices, seq_lens, compress_ratio)
        return CompressorDecodePlan(compress_ratio, torch.from_dlpack(plan_d))

    @staticmethod
    def generate_online(
        seq_lens: torch.Tensor,
        req_pool_indices: torch.Tensor,
        req_to_token: torch.Tensor,
        state_slot_offset: int = 0,
    ) -> CompressorDecodePlan:
        batch_size = int(seq_lens.shape[0])
        module = _jit_compress_128_online_module(512)
        plan_d = torch.empty(
            (batch_size, 16),
            dtype=torch.uint8,
            device=req_pool_indices.device,
        )
        module.plan_decode(
            seq_lens,
            req_pool_indices,
            req_to_token,
            plan_d,
            int(state_slot_offset),
        )
        return CompressorDecodePlan(128, plan_d)

    @property
    def is_decode(self) -> bool:
        return True


class CompressorPrefillPlan(NamedTuple):
    compress_ratio: int
    plan_c: torch.Tensor  # [num_q_tokens, 16] uint8 --- CompressPlan
    plan_w: torch.Tensor  # [num_q_tokens,  8] uint8 --- WritePlan
    pin_buffer: Optional[torch.Tensor] = None  # keep alive

    def copy_(self, other) -> None:
        assert isinstance(other, CompressorPrefillPlan)
        assert self.compress_ratio == other.compress_ratio
        self.plan_c.copy_(other.plan_c)
        self.plan_w.copy_(other.plan_w)

    @staticmethod
    def generate(
        compress_ratio: Literal[4, 128],
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        extend_lens: torch.Tensor,
        req_to_token: torch.Tensor,
        full_to_state: torch.Tensor,
        swa_page_size: int,
        ring_size: int,
        num_q_tokens: int,
        use_cuda_graph: bool = False,
    ) -> CompressorPrefillPlan:
        is_gpu_input = seq_lens.device.type in ["cuda", "xpu"]
        pin_buffer = torch.empty(
            0 if is_gpu_input else num_q_tokens * _PREFILL_PLAN_BYTES,
            dtype=torch.uint8,
            pin_memory=not is_gpu_input,
        )
        # DP-safe empty-batch guard: a TBO ubatch (or tail batch) can have 0
        # query tokens (num_q_tokens==0) on THIS rank while other DP ranks are
        # non-empty. The global TBO decision must stay uniform across ranks, so
        # return an empty plan here (downstream compressor then processes 0
        # tokens = no-op) instead of skipping TBO per-rank. Avoids the
        # c_plan.cuh RuntimeCheck(batch_size <= num_q_tokens) failure at B>=1.
        if int(num_q_tokens) == 0 and is_hip():
            _dev = req_to_token.device
            return CompressorPrefillPlan(
                compress_ratio,
                torch.empty((0, 16), dtype=torch.uint8, device=_dev),
                torch.empty((0, 8), dtype=torch.uint8, device=_dev),
                pin_buffer,
            )
        module = _jit_compress_plan_module()
        if _is_xpu:
            fn = plan_compress_prefill
        else:
            module = _jit_compress_plan_module()
            fn = module.plan_prefill

        plan_c, plan_w = fn(
            req_pool_indices,
            req_to_token,
            full_to_state,
            seq_lens,
            extend_lens,
            pin_buffer,
            int(num_q_tokens),
            int(compress_ratio),
            int(swa_page_size),
            int(ring_size),
            bool(use_cuda_graph),
        )
        return CompressorPrefillPlan(
            compress_ratio,
            torch.from_dlpack(plan_c) if not _is_xpu else plan_c,
            torch.from_dlpack(plan_w) if not _is_xpu else plan_w,
            pin_buffer,
        )

    @staticmethod
    def generate_legacy(
        compress_ratio: Literal[4, 128],
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        extend_lens: torch.Tensor,
        num_q_tokens: int,
        device: torch.device,
        use_cuda_graph: bool = False,
    ) -> CompressorPrefillPlan:
        pin_buffer = torch.empty(
            num_q_tokens * _PREFILL_PLAN_BYTES,
            dtype=torch.uint8,
            pin_memory=True,
        )
        if _is_xpu:
            fn = plan_compress_prefill_legacy
        else:
            module = _jit_compress_plan_module()
            fn = module.plan_prefill_legacy

        plan_c, plan_w = fn(
            req_pool_indices,
            seq_lens,
            extend_lens,
            pin_buffer,
            int(num_q_tokens),
            int(compress_ratio),
            bool(use_cuda_graph),
        )
        return CompressorPrefillPlan(
            compress_ratio,
            torch.from_dlpack(plan_c) if not _is_xpu else plan_c,
            torch.from_dlpack(plan_w) if not _is_xpu else plan_w,
            pin_buffer,
        )

    @staticmethod
    def generate_online(
        seq_lens: torch.Tensor,
        extend_lens: torch.Tensor,
        req_pool_indices: torch.Tensor,
        req_to_token: torch.Tensor,
        num_q_tokens: int,
        use_cuda_graph: bool = False,
        state_slot_offset: int = 0,
    ) -> CompressorPrefillPlan:
        seq_lens_cpu = seq_lens.detach().to(torch.int64).cpu()
        extend_lens_cpu = extend_lens.detach().to(torch.int64).cpu()
        rid_i64 = req_pool_indices.to(torch.int64)
        r2t_i32 = req_to_token.to(torch.int32)
        pin_buffer = torch.empty(
            (2, num_q_tokens, 16), dtype=torch.uint8, pin_memory=True
        )
        plan_c_pin, plan_w_pin = pin_buffer[0], pin_buffer[1]
        device = req_pool_indices.device
        plan_c_dev = torch.empty((num_q_tokens, 16), dtype=torch.uint8, device=device)
        plan_w_dev = torch.empty((num_q_tokens, 16), dtype=torch.uint8, device=device)
        module = _jit_compress_128_online_module(512)  # NOTE: only support dim=512
        num_c, num_w = module.plan_prefill(
            seq_lens_cpu,
            extend_lens_cpu,
            rid_i64,
            r2t_i32,
            plan_c_pin,
            plan_w_pin,
            plan_c_dev,
            plan_w_dev,
            int(state_slot_offset),
            bool(use_cuda_graph),
        )
        return CompressorPrefillPlan(
            128,
            plan_c_dev[: int(num_c)],
            plan_w_dev[: int(num_w)],
            pin_buffer,
        )

    @property
    def is_decode(self) -> bool:
        return False


def compress_forward(
    kv_score_buffer: torch.Tensor,
    kv_score_input: torch.Tensor,
    ape: torch.Tensor,
    plan: Union[CompressorDecodePlan, CompressorPrefillPlan],
    *,
    head_dim: int,
    compress_ratio: Literal[4, 128],
    out: Optional[torch.Tensor] = None,
    is_online: bool = False,
) -> torch.Tensor:
    if out is None:
        num_q_tokens = plan[1].shape[0]  # NOTE: decode = bs, prefill = dynamic
        out = kv_score_input.new_empty((num_q_tokens, head_dim))
    assert plan.compress_ratio == compress_ratio
    if is_online:
        assert compress_ratio == 128 and head_dim == 512
        module = _jit_compress_128_online_module(512, kv_score_buffer.dtype)
    else:
        if _is_xpu:
            decode_fn, prefill_fn = _XPU_COMPRESS_FNS[compress_ratio]
        else:
            dtype_in, dtype_out = kv_score_input.dtype, out.dtype
            module = _jit_compress_module(
                head_dim, kv_score_buffer.dtype, dtype_in, dtype_out, compress_ratio
            )

    if _is_xpu:
        fn = decode_fn if plan.is_decode else prefill_fn
    else:
        fn = module.decode if plan.is_decode else module.prefill

    fn(kv_score_buffer, kv_score_input, out, ape, *plan[1:3])
    return out


def compress_norm_rope_store(
    kv: torch.Tensor,
    plan: Union[CompressorDecodePlan, CompressorPrefillPlan],
    *,
    norm_weight: torch.Tensor,
    norm_eps: float,
    freq_cis: torch.Tensor,
    out_loc: torch.Tensor,
    kvcache: torch.Tensor,
    page_size: int,
    use_fp4: bool = False,
    bf16_store: bool = False,
) -> None:
    if use_fp4:
        assert kv.shape[-1] == 128
    freq_cis = torch.view_as_real(freq_cis).flatten(-2)
    if _is_xpu:
        compress_norm_rope_store_xpu(
            kv,
            plan[1],
            norm_weight,
            norm_eps,
            freq_cis,
            out_loc,
            kvcache,
            plan.is_decode,
            plan.compress_ratio,
            page_size,
            use_fp4,
        )
    else:
        module = _jit_compress_norm_rope_module(
            kv.dtype, kv.shape[-1], freq_cis.shape[-1], page_size, bf16_store
        )
        fn = module.forward_fp4 if use_fp4 else module.forward
        fn(
            kv,
            plan[1],
            norm_weight,
            norm_eps,
            freq_cis,
            out_loc,
            kvcache,
            plan.is_decode,
            plan.compress_ratio,
        )
