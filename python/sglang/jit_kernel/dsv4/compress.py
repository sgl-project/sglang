from __future__ import annotations

from typing import Literal, NamedTuple, Optional, Union

import torch
from tvm_ffi.module import Module

from sglang.jit_kernel.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)
from sglang.srt.environ import envs

from .utils import make_name


@cache_once
def _jit_compress_norm_rope_module(
    dtype: torch.dtype,
    head_dim: int,
    rope_dim: int,
) -> Module:
    args = make_cpp_args(dtype, head_dim, rope_dim, is_arch_support_pdl())
    return load_jit(
        make_name(f"fused_norm_rope_v2"),
        *args,
        cuda_files=[f"deepseek_v4/fused_norm_rope_v2.cuh"],
        cuda_wrappers=[("forward", f"FusedNormRopeKernel<{args}>::forward")],
    )


@cache_once
def _jit_compress_module(
    head_dim: int,
    dtype_in: torch.dtype,
    dtype_out: torch.dtype,
    ratio: Literal[4, 128],
) -> Module:
    args = make_cpp_args(head_dim, dtype_in, dtype_out, is_arch_support_pdl())
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
def _jit_compress_plan_module() -> Module:
    return load_jit(
        make_name(f"compress_plan"),
        cuda_files=[f"deepseek_v4/c_plan.cuh"],
        cuda_wrappers=[
            ("plan_prefill", "plan_compress_prefill"),
            ("plan_decode", "plan_compress_decode"),
        ],
    )


@cache_once
def _jit_compress_128_online_plan_module() -> Module:
    """Host-side plan generator for online compress 128 (no template args)."""
    return load_jit(
        make_name("compress_128_online_plan"),
        cuda_files=["deepseek_v4/c128_online.cuh"],
        cuda_wrappers=[
            ("plan_compress_online_prefill", "plan_compress_online_prefill"),
        ],
    )


@cache_once
def _jit_compress_128_online_module(head_dim: int) -> Module:
    """Online compress 128 kernel: ring_size=1, per-index (max, sum, kv) state."""
    args = make_cpp_args(head_dim, is_arch_support_pdl())
    kernel_class = f"FlashCompress128OnlineKernel<{args}>"
    return load_jit(
        make_name("compress_128_online"),
        *args,
        cuda_files=["deepseek_v4/c128_online.cuh"],
        cuda_wrappers=[
            ("decode", f"{kernel_class}::run_decode"),
            ("prefill", f"{kernel_class}::run_prefill"),
        ],
        extra_cuda_cflags=["-use_fast_math"],
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
        full_to_swa: torch.Tensor,
        seq_lens: torch.Tensor,
        swa_page_size: int,
        ring_size: int,
    ) -> CompressorDecodePlan:
        module = _jit_compress_plan_module()
        plan_d = module.plan_decode(
            req_pool_indices,
            req_to_token,
            full_to_swa,
            seq_lens,
            int(compress_ratio),
            int(swa_page_size),
            int(ring_size),
        )
        return CompressorDecodePlan(compress_ratio, torch.from_dlpack(plan_d))

    @property
    def is_decode(self) -> bool:
        return True


class CompressorPrefillPlan(NamedTuple):
    compress_ratio: int
    plan_c: torch.Tensor  # [num_q_tokens, 16] uint8 --- CompressPlan
    plan_w: torch.Tensor  # [num_q_tokens,  8] uint8 --- WritePlan

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
        full_to_swa: torch.Tensor,
        swa_page_size: int,
        ring_size: int,
        num_q_tokens: int,
        device: torch.device,
        use_cuda_graph: bool = False,
    ) -> CompressorPrefillPlan:
        if compress_ratio == 128 and envs.SGLANG_OPT_USE_ONLINE_COMPRESS.get():
            return CompressorPrefillPlan._generate_online(
                num_q_tokens=num_q_tokens,
                seq_lens=seq_lens,
                extend_lens=extend_lens,
                device=device,
                use_cuda_graph=use_cuda_graph,
            )
        pin_buffer = torch.empty(
            num_q_tokens * _PREFILL_PLAN_BYTES,
            dtype=torch.uint8,
            pin_memory=True,
        )
        module = _jit_compress_plan_module()
        plan_c, plan_w = module.plan_prefill(
            req_pool_indices,
            req_to_token,
            full_to_swa,
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
            torch.from_dlpack(plan_c),
            torch.from_dlpack(plan_w),
        )

    @staticmethod
    def _generate_online(
        num_q_tokens: int,
        seq_lens: torch.Tensor,
        extend_lens: torch.Tensor,
        device: torch.device,
        use_cuda_graph: bool,
    ) -> CompressorPrefillPlan:
        # Online plan host-side path: only CPU/cuda-host implemented today
        plan_tensor = torch.empty(
            (2, num_q_tokens, 16),
            dtype=torch.uint8,
            device="cpu",
            pin_memory=True,
        )
        module = _jit_compress_128_online_plan_module()
        plan_lens = module.plan_compress_online_prefill(
            extend_lens,
            seq_lens,
            plan_tensor[0],
            plan_tensor[1],
            use_cuda_graph,
        )
        return CompressorPrefillPlan(
            128,
            plan_tensor[0, : plan_lens[0]].to(device, non_blocking=True),
            plan_tensor[1, : plan_lens[1]].to(device, non_blocking=True),
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
) -> torch.Tensor:
    """Run the c4/c128 compress kernel.

    The plan tensors carry all the state-pool slot information; no separate
    `indices` or `extra_data` arguments are needed.
    """
    assert head_dim % 128 == 0
    num_q_tokens = kv_score_input.shape[0]
    if out is None:
        out = kv_score_input.new_empty((num_q_tokens, head_dim))
    assert plan.compress_ratio == compress_ratio, "Mismatched compress ratio in plan!"
    # Online c128: separate JIT module, fp32 state, no compile-time dtypes.
    if compress_ratio == 128 and envs.SGLANG_OPT_USE_ONLINE_COMPRESS.get():
        online_module = _jit_compress_128_online_module(head_dim=head_dim)
        fn = online_module.decode if plan.is_decode else online_module.prefill
        fn(kv_score_buffer, kv_score_input, out, ape, *plan[1:])
        return out
    module = _jit_compress_module(
        head_dim, kv_score_input.dtype, out.dtype, compress_ratio
    )
    fn = module.decode if plan.is_decode else module.prefill
    fn(kv_score_buffer, kv_score_input, out, ape, *plan[1:])
    return out


def compress_fused_norm_rope_inplace(
    kv: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    freq_cis: torch.Tensor,
    plan: Union[CompressorDecodePlan, CompressorPrefillPlan],
) -> None:
    freq_cis = torch.view_as_real(freq_cis).flatten(-2)
    module = _jit_compress_norm_rope_module(kv.dtype, kv.shape[-1], freq_cis.shape[-1])
    module.forward(
        kv,
        weight,
        plan[1],
        freq_cis,
        int(plan.is_decode),
        eps,
        plan.compress_ratio,
    )
