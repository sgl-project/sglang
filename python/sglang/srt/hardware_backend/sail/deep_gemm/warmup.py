"""Warmup executors for the kernel types only the SAIL DeepGEMM build serves.

``layers/deep_gemm_wrapper/compile_utils.py`` owns the warmup machinery (the M
list, the per-group compile lock, the memory budget) and the executors for the
kernel types the CUDA build serves. The nopad, channel-wise, int8, fp4, int4
and gather-read fused types exist in the SAIL vendor build only, so their
tensor factories and executors live here. ``_BaseWarmupExecutor.create``
imports this module on demand, for a kernel type the shared table does not
cover, so the shared compile path never imports the platform tree at module
scope.

The types themselves are declared in ``kernel_types``: the shared enum names
only what every DeepGEMM build serves, and the shared tables key on ``.name``,
so a member declared here needs no counterpart there.
"""

import torch

from sglang.srt.hardware_backend.sail.deep_gemm.kernel_types import (
    SailDeepGemmKernelType,
)
from sglang.srt.layers.deep_gemm_wrapper.compile_utils import (
    _BaseWarmupExecutor,
    _empty_block_fp8,
    _empty_token_fp8,
)
from sglang.srt.layers.deep_gemm_wrapper.configurer import ENABLE_JIT_DEEPGEMM

if ENABLE_JIT_DEEPGEMM:
    import deep_gemm


def _preprocess_mxfp4_scales(scales):
    """Pack E8M0 scales into the layout the FP4 GEMMs read.

    A host without the DeepGEMM build imports this module only to price a kernel
    type (see test_warmup_memory), so leave the raw scales in place there; the
    vendor module is exactly what such a host cannot import.
    """
    if not ENABLE_JIT_DEEPGEMM:
        return scales
    return deep_gemm.preprocess_mxfp4_scales(scales)


def _empty_token_int8(size):
    *dims, k = size
    return (
        torch.empty(size, device="cuda", dtype=torch.int8),
        torch.empty((*dims, 1), device="cuda", dtype=torch.float32),
    )


def _empty_block_int8(size):
    *dims, n, k = size
    return (
        torch.empty(size, device="cuda", dtype=torch.int8),
        torch.empty(
            (*dims, n, 1),
            device="cuda",
            dtype=torch.float32,
        ),
    )


def _empty_token_fp8_channel(size):
    *dims, k = size
    return (
        torch.empty(size, device="cuda", dtype=torch.float8_e4m3fn),
        torch.empty((*dims, 1), device="cuda", dtype=torch.float32),
    )


def _empty_block_fp8_channel(size):
    *dims, n, k = size
    return (
        torch.empty(size, device="cuda", dtype=torch.float8_e4m3fn),
        torch.empty(
            (*dims, n, 1),
            device="cuda",
            dtype=torch.float32,
        ),
    )


def _empty_token_bf16(size):
    *dims, k = size
    return torch.empty(size, device="cuda", dtype=torch.bfloat16)


def _empty_block_bf16(size):
    *dims, n, k = size
    return torch.empty(size, device="cuda", dtype=torch.bfloat16)


def _empty_token_uint8(size):
    *dims, k = size
    return (
        torch.empty(size, device="cuda", dtype=torch.uint8),
        torch.empty((*dims, k // 16), device="cuda", dtype=torch.uint8),
    )


def _empty_block_uint8(size):
    *dims, n, k = size
    return (
        torch.empty(size, device="cuda", dtype=torch.uint8),
        torch.empty(
            (*dims, n, k // 16),
            device="cuda",
            dtype=torch.uint8,
        ),
    )


def _empty_marlin_int4(size, scale_dtype=torch.bfloat16):
    # Currently deepgemm w4a16 only supports group_size=32
    *dims, n, k = size
    group_size = 32
    return (
        torch.empty(
            (*dims, k // 16, n * 2),
            device="cuda",
            dtype=torch.int32,
        ),
        torch.empty(
            (*dims, k // group_size, n),
            device="cuda",
            dtype=scale_dtype,
        ),
    )


def _empty_mxfp4_w4a16_mma(size):
    *dims, n, k = size
    if n % 64 != 0 or k % 64 != 0:
        raise ValueError(f"W4A16 MMA requires N/K aligned to 64, got N={n}, K={k}.")
    return (
        torch.empty((*dims, n, k // 2), device="cuda", dtype=torch.uint8),
        torch.empty((*dims, n // 64, k * 2), device="cuda", dtype=torch.uint8),
    )


class _GroupedNopadWarmupExecutor(_BaseWarmupExecutor):
    def __init__(self, max_m: int, n: int, k: int, num_groups: int):
        super().__init__(max_m, n, k, num_groups)

    def setup_tensors(self, max_m: int, n: int, k: int, num_groups: int):
        self.lhs_q, self.lhs_s = _empty_token_fp8((max_m, k))
        self.rhs_q, self.rhs_s = _empty_block_fp8((num_groups, n, k))
        self.m_indices = torch.zeros((max_m,), device="cuda", dtype=torch.int32)
        self.out = torch.empty((max_m, n), device="cuda", dtype=torch.bfloat16)

    def execute(self, m):
        deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_nopad(
            (self.lhs_q[:m], self.lhs_s[:m]),
            (self.rhs_q, self.rhs_s),
            self.out[:m],
            m_indices=self.m_indices[:m],
        )


class _NormalWarmupExecutor_channel(_BaseWarmupExecutor):
    def __init__(self, max_m: int, n: int, k: int, num_groups: int):
        super().__init__(max_m, n, k, num_groups)

    def setup_tensors(self, max_m: int, n: int, k: int, num_groups: int):
        self.lhs_q, self.lhs_s = _empty_token_fp8_channel((max_m, k))
        self.rhs_q, self.rhs_s = _empty_block_fp8_channel((n, k))
        self.out = torch.empty((max_m, n), device="cuda", dtype=torch.bfloat16)

    def execute(self, m):
        deep_gemm.fp8_gemm_nt(
            (self.lhs_q[:m], self.lhs_s[:m]),
            (self.rhs_q, self.rhs_s),
            self.out[:m],
        )


class _GroupedContWarmupExecutor_channel(_BaseWarmupExecutor):
    def __init__(self, max_m: int, n: int, k: int, num_groups: int):
        super().__init__(max_m, n, k, num_groups)

    def setup_tensors(self, max_m: int, n: int, k: int, num_groups: int):
        self.lhs_q, self.lhs_s = _empty_token_fp8_channel((max_m, k))
        self.rhs_q, self.rhs_s = _empty_block_fp8_channel((num_groups, n, k))
        self.m_indices = torch.zeros((max_m,), device="cuda", dtype=torch.int32)
        self.out = torch.empty((max_m, n), device="cuda", dtype=torch.bfloat16)

    def execute(self, m):
        deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
            (self.lhs_q[:m], self.lhs_s[:m]),
            (self.rhs_q, self.rhs_s),
            self.out[:m],
            m_indices=self.m_indices[:m],
        )


class _GroupedMaskedWarmupExecutor_channel(_BaseWarmupExecutor):
    def __init__(self, max_m: int, n: int, k: int, num_groups: int):
        super().__init__(max_m, n, k, num_groups)

    def setup_tensors(self, max_m: int, n: int, k: int, num_groups: int):
        self.lhs_q, self.lhs_s = _empty_token_fp8_channel((num_groups, max_m, k))
        self.rhs_q, self.rhs_s = _empty_block_fp8_channel((num_groups, n, k))
        self.masked_m = torch.zeros((num_groups,), device="cuda", dtype=torch.int32)
        self.out = torch.empty(
            (num_groups, max_m, n), device="cuda", dtype=torch.bfloat16
        )

    def execute(self, m):
        deep_gemm.fp8_m_grouped_gemm_nt_masked(
            (self.lhs_q, self.lhs_s),
            (self.rhs_q, self.rhs_s),
            self.out,
            masked_m=self.masked_m,
            # DeepGEMM uses `expected_m` instead of input shape for `get_best_config`
            expected_m=m,
        )


class _GroupedNopadWarmupExecutor_channel(_BaseWarmupExecutor):
    def __init__(self, max_m: int, n: int, k: int, num_groups: int):
        super().__init__(max_m, n, k, num_groups)

    def setup_tensors(self, max_m: int, n: int, k: int, num_groups: int):
        self.lhs_q, self.lhs_s = _empty_token_fp8_channel((max_m, k))
        self.rhs_q, self.rhs_s = _empty_block_fp8_channel((num_groups, n, k))
        self.m_indices = torch.zeros((max_m,), device="cuda", dtype=torch.int32)
        self.out = torch.empty((max_m, n), device="cuda", dtype=torch.bfloat16)

    def execute(self, m):
        deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_nopad(
            (self.lhs_q[:m], self.lhs_s[:m]),
            (self.rhs_q, self.rhs_s),
            self.out[:m],
            m_indices=self.m_indices[:m],
        )


class _NormalWarmupExecutor_int8(_BaseWarmupExecutor):
    def __init__(self, max_m: int, n: int, k: int, num_groups: int):
        super().__init__(max_m, n, k, num_groups)

    def setup_tensors(self, max_m: int, n: int, k: int, num_groups: int):
        self.lhs_q, self.lhs_s = _empty_token_int8((max_m, k))
        self.rhs_q, self.rhs_s = _empty_block_int8((n, k))
        self.out = torch.empty((max_m, n), device="cuda", dtype=torch.bfloat16)

    def execute(self, m):
        deep_gemm.gemm_int8_int8_bf16_nt(
            (self.lhs_q[:m], self.lhs_s[:m]),
            (self.rhs_q, self.rhs_s),
            self.out[:m],
        )


class _GroupedContWarmupExecutor_int8(_BaseWarmupExecutor):
    def __init__(self, max_m: int, n: int, k: int, num_groups: int):
        super().__init__(max_m, n, k, num_groups)

    def setup_tensors(self, max_m: int, n: int, k: int, num_groups: int):
        self.lhs_q, self.lhs_s = _empty_token_int8((max_m, k))
        self.rhs_q, self.rhs_s = _empty_block_int8((num_groups, n, k))
        self.m_indices = torch.zeros((max_m,), device="cuda", dtype=torch.int32)
        self.out = torch.empty((max_m, n), device="cuda", dtype=torch.bfloat16)

    def execute(self, m):
        deep_gemm.m_grouped_gemm_int8_int8_bf16_nt_contiguous(
            (self.lhs_q[:m], self.lhs_s[:m]),
            (self.rhs_q, self.rhs_s),
            self.out[:m],
            m_indices=self.m_indices[:m],
        )


class _GroupedMaskedWarmupExecutor_int8(_BaseWarmupExecutor):
    def __init__(self, max_m: int, n: int, k: int, num_groups: int):
        super().__init__(max_m, n, k, num_groups)

    def setup_tensors(self, max_m: int, n: int, k: int, num_groups: int):
        self.lhs_q, self.lhs_s = _empty_token_int8((num_groups, max_m, k))
        self.rhs_q, self.rhs_s = _empty_block_int8((num_groups, n, k))
        self.masked_m = torch.zeros((num_groups,), device="cuda", dtype=torch.int32)
        self.out = torch.empty(
            (num_groups, max_m, n), device="cuda", dtype=torch.bfloat16
        )

    def execute(self, m):
        deep_gemm.m_grouped_gemm_int8_int8_bf16_nt_masked(
            (self.lhs_q, self.lhs_s),
            (self.rhs_q, self.rhs_s),
            self.out,
            masked_m=self.masked_m,
            # DeepGEMM uses `expected_m` instead of input shape for `get_best_config`
            expected_m=m,
        )


class _GroupedNopadWarmupExecutor_int8(_BaseWarmupExecutor):
    def __init__(self, max_m: int, n: int, k: int, num_groups: int):
        super().__init__(max_m, n, k, num_groups)

    def setup_tensors(self, max_m: int, n: int, k: int, num_groups: int):
        self.lhs_q, self.lhs_s = _empty_token_int8((max_m, k))
        self.rhs_q, self.rhs_s = _empty_block_int8((num_groups, n, k))
        self.m_indices = torch.zeros((max_m,), device="cuda", dtype=torch.int32)
        self.out = torch.empty((max_m, n), device="cuda", dtype=torch.bfloat16)

    def execute(self, m):
        deep_gemm.m_grouped_gemm_int8_int8_bf16_nt_nopad(
            (self.lhs_q[:m], self.lhs_s[:m]),
            (self.rhs_q, self.rhs_s),
            self.out[:m],
            m_indices=self.m_indices[:m],
        )


class _GroupedNopadWarmupExecutor_bf16(_BaseWarmupExecutor):
    def __init__(self, max_m: int, n: int, k: int, num_groups: int):
        super().__init__(max_m, n, k, num_groups)

    def setup_tensors(self, max_m: int, n: int, k: int, num_groups: int):
        self.lhs_q = _empty_token_bf16((max_m, k))
        self.rhs_q = _empty_block_bf16((num_groups, n, k))
        self.m_indices = torch.zeros((max_m,), device="cuda", dtype=torch.int32)
        self.out = torch.empty((max_m, n), device="cuda", dtype=torch.bfloat16)

    def execute(self, m):
        deep_gemm.m_grouped_gemm_bf16_bf16_bf16_nt_nopad(
            self.lhs_q[:m],
            self.rhs_q,
            self.out[:m],
            m_indices=self.m_indices[:m],
        )


class _GroupedMaskedWarmupExecutor_fp4(_BaseWarmupExecutor):
    def __init__(self, max_m: int, n: int, k: int, num_groups: int):
        super().__init__(max_m, n, k, num_groups)

    def setup_tensors(self, max_m: int, n: int, k: int, num_groups: int):
        self.lhs_q, self.lhs_s = _empty_token_uint8((num_groups, max_m, k))
        self.rhs_q, self.rhs_s = _empty_block_uint8((num_groups, n, k))
        self.lhs_s = _preprocess_mxfp4_scales(self.lhs_s)
        self.rhs_s = _preprocess_mxfp4_scales(self.rhs_s)
        self.bias = None
        self.masked_m = torch.zeros((num_groups,), device="cuda", dtype=torch.int32)
        self.out = torch.empty(
            (num_groups, max_m, n), device="cuda", dtype=torch.bfloat16
        )

    def execute(self, m):
        deep_gemm.m_grouped_gemm_fp4_fp4_bf16_nt_masked(
            (self.lhs_q, self.lhs_s),
            (self.rhs_q, self.rhs_s),
            self.bias,
            self.out,
            masked_m=self.masked_m,
            # DeepGEMM uses `expected_m` instead of input shape for `get_best_config`
            expected_m=m,
        )


class _GroupedNopadWarmupExecutor_fp4(_BaseWarmupExecutor):
    def __init__(self, max_m: int, n: int, k: int, num_groups: int):
        super().__init__(max_m, n, k, num_groups)

    def setup_tensors(self, max_m: int, n: int, k: int, num_groups: int):
        self.lhs_q, self.lhs_s = _empty_token_uint8((max_m, k))
        self.rhs_q, self.rhs_s = _empty_block_uint8((num_groups, n, k))
        self.lhs_s = _preprocess_mxfp4_scales(self.lhs_s)
        self.rhs_s = _preprocess_mxfp4_scales(self.rhs_s)
        self.bias = None
        self.m_indices = torch.zeros((max_m,), device="cuda", dtype=torch.int32)
        self.out = torch.empty((max_m, n), device="cuda", dtype=torch.bfloat16)

    def execute(self, m):
        deep_gemm.m_grouped_gemm_fp4_fp4_bf16_nt_nopad(
            (self.lhs_q[:m], self.lhs_s[:m]),
            (self.rhs_q, self.rhs_s),
            self.bias,
            self.out[:m],
            m_indices=self.m_indices[:m],
        )


class _NormalWarmupExecutor_fp4(_BaseWarmupExecutor):
    def __init__(self, max_m: int, n: int, k: int, num_groups: int):
        super().__init__(max_m, n, k, num_groups)

    def setup_tensors(self, max_m: int, n: int, k: int, num_groups: int):
        self.lhs_q, self.lhs_s = _empty_token_uint8((max_m, k))
        self.rhs_q, self.rhs_s = _empty_block_uint8((n, k))
        self.lhs_s = _preprocess_mxfp4_scales(self.lhs_s)
        self.rhs_s = _preprocess_mxfp4_scales(self.rhs_s)
        self.bias = None
        self.out = torch.empty((max_m, n), device="cuda", dtype=torch.bfloat16)

    def execute(self, m):
        deep_gemm.gemm_fp4_fp4_bf16_nt(
            (self.lhs_q[:m], self.lhs_s[:m]),
            (self.rhs_q, self.rhs_s),
            self.bias,
            self.out[:m],
        )


class _GroupedMaskedWarmupExecutor_fp4_bias(_GroupedMaskedWarmupExecutor_fp4):
    def setup_tensors(self, max_m: int, n: int, k: int, num_groups: int):
        super().setup_tensors(max_m, n, k, num_groups)
        self.bias = torch.zeros((num_groups, n), device="cuda", dtype=torch.float32)


class _GroupedNopadWarmupExecutor_fp4_bias(_GroupedNopadWarmupExecutor_fp4):
    def setup_tensors(self, max_m: int, n: int, k: int, num_groups: int):
        super().setup_tensors(max_m, n, k, num_groups)
        self.bias = torch.zeros((num_groups, n), device="cuda", dtype=torch.float32)


class _NormalWarmupExecutor_fp4_bias(_NormalWarmupExecutor_fp4):
    def setup_tensors(self, max_m: int, n: int, k: int, num_groups: int):
        super().setup_tensors(max_m, n, k, num_groups)
        self.bias = torch.zeros((n,), device="cuda", dtype=torch.float32)


class _GroupedMaskedWarmupExecutor_int4(_BaseWarmupExecutor):
    def __init__(self, max_m: int, n: int, k: int, num_groups: int):
        super().__init__(max_m, n, k, num_groups)

    def setup_tensors(self, max_m: int, n: int, k: int, num_groups: int):
        self.lhs = _empty_token_bf16((num_groups, max_m, k))
        self.rhs_q, self.rhs_s = _empty_marlin_int4((num_groups, n, k))
        self.masked_m = torch.zeros((num_groups,), device="cuda", dtype=torch.int32)
        self.out = torch.empty(
            (num_groups, max_m, n), device="cuda", dtype=torch.bfloat16
        )

    def execute(self, m):
        deep_gemm.m_grouped_gemm_w4a16_masked(
            self.lhs,
            (self.rhs_q, self.rhs_s),
            self.out,
            masked_m=self.masked_m,
            # DeepGEMM uses `expected_m` instead of input shape for `get_best_config`
            expected_m=m,
        )


class _GroupedMaskedWarmupExecutor_int4_mma(_GroupedMaskedWarmupExecutor_int4):
    def setup_tensors(self, max_m: int, n: int, k: int, num_groups: int):
        self.lhs = _empty_token_bf16((num_groups, max_m, k))
        self.rhs_q, self.rhs_s = _empty_mxfp4_w4a16_mma((num_groups, n, k))
        self.masked_m = torch.zeros((num_groups,), device="cuda", dtype=torch.int32)
        self.out = torch.empty(
            (num_groups, max_m, n), device="cuda", dtype=torch.bfloat16
        )


class _GroupedFusedWarmupExecutor_int4(_BaseWarmupExecutor):
    scale_dtype = torch.bfloat16

    def setup_tensors(self, max_m: int, n: int, k: int, num_groups: int):
        self.lhs = _empty_token_bf16((max_m, k))
        self.rhs_q, self.rhs_s = _empty_marlin_int4(
            (num_groups, n, k), self.scale_dtype
        )
        self.out = torch.empty((max_m, n), device="cuda", dtype=torch.bfloat16)
        self.topk_ids = torch.randint(
            0, num_groups, (max_m, 1), device="cuda", dtype=torch.int32
        )

    def execute(self, m):
        config, rows, experts, tokens, blocks, _, _ = deep_gemm.moe_align_block_size(
            self.lhs[:m], self.rhs_q, self.topk_ids[:m], perchannel_quant=False
        )
        deep_gemm.m_grouped_gemm_w4a16_fused(
            self.lhs[:m],
            (self.rhs_q, self.rhs_s),
            self.out[:m],
            rows,
            experts,
            tokens,
            blocks,
            config,
        )


class _GroupedFusedWarmupExecutor_int4_mma(_GroupedFusedWarmupExecutor_int4):
    def setup_tensors(self, max_m: int, n: int, k: int, num_groups: int):
        self.lhs = _empty_token_bf16((max_m, k))
        self.rhs_q, self.rhs_s = _empty_mxfp4_w4a16_mma((num_groups, n, k))
        self.out = torch.empty((max_m, n), device="cuda", dtype=torch.bfloat16)
        self.topk_ids = torch.randint(
            0, num_groups, (max_m, 1), device="cuda", dtype=torch.int32
        )


class _GroupedFusedWarmupExecutor_mxfp4_valu(_GroupedFusedWarmupExecutor_int4):
    scale_dtype = torch.uint8


class _GroupedFusedWarmupExecutor(_BaseWarmupExecutor):
    """Warm up the routing metadata and the gather-read GEMM without recursing into
    the execution hook."""

    dtype = "fp8"
    perchannel = False
    gemm_name = "m_grouped_gemm_fp8_fp8_bf16_nt_fused"

    def setup_tensors(self, max_m: int, n: int, k: int, num_groups: int):
        if self.dtype == "bf16":
            self.lhs_q = _empty_token_bf16((max_m, k))
            self.rhs_q = _empty_block_bf16((num_groups, n, k))
        else:
            if self.dtype == "int8":
                token_factory, block_factory = _empty_token_int8, _empty_block_int8
            elif self.perchannel:
                token_factory, block_factory = (
                    _empty_token_fp8_channel,
                    _empty_block_fp8_channel,
                )
            else:
                token_factory, block_factory = _empty_token_fp8, _empty_block_fp8
            self.lhs_q, self.lhs_s = token_factory((max_m, k))
            self.rhs_q, self.rhs_s = block_factory((num_groups, n, k))
        self.out = torch.empty((max_m, n), device="cuda", dtype=torch.bfloat16)
        self.topk_ids = torch.randint(
            0, num_groups, (max_m, 1), device="cuda", dtype=torch.int32
        )

    def execute(self, m):
        config, rows, experts, tokens, blocks, _, _ = deep_gemm.moe_align_block_size(
            self.lhs_q[:m],
            self.rhs_q,
            self.topk_ids[:m],
            perchannel_quant=self.perchannel,
        )
        lhs = self.lhs_q[:m]
        rhs = self.rhs_q
        if self.dtype != "bf16":
            lhs = (lhs, self.lhs_s[:m])
            rhs = (rhs, self.rhs_s)
        getattr(deep_gemm, self.gemm_name)(
            lhs, rhs, self.out[:m], rows, experts, tokens, blocks, config
        )


class _GroupedFusedWarmupExecutor_fp8_channel(_GroupedFusedWarmupExecutor):
    perchannel = True


class _GroupedFusedWarmupExecutor_int8(_GroupedFusedWarmupExecutor):
    dtype = "int8"
    perchannel = True
    gemm_name = "m_grouped_gemm_int8_int8_bf16_nt_fused"


class _GroupedFusedWarmupExecutor_bf16(_GroupedFusedWarmupExecutor):
    dtype = "bf16"
    gemm_name = "m_grouped_gemm_bf16_bf16_bf16_nt_fused"


class _GroupedFusedWarmupExecutor_fp4(_GroupedFusedWarmupExecutor):
    dtype = "fp4"
    gemm_name = "m_grouped_gemm_fp4_fp4_bf16_nt_fused"

    def setup_tensors(self, max_m: int, n: int, k: int, num_groups: int):
        self.lhs_q, self.lhs_s = _empty_token_uint8((max_m, k))
        self.rhs_q, self.rhs_s = _empty_token_uint8((num_groups, n, k))
        # gather-read consumes K-major scales directly; only the RHS goes through
        # the vendor preprocessing.
        self.lhs_s = self.lhs_s.view(torch.uint16)
        self.rhs_s = _preprocess_mxfp4_scales(self.rhs_s)
        self.out = torch.empty((max_m, n), device="cuda", dtype=torch.bfloat16)
        self.topk_ids = torch.randint(
            0, num_groups, (max_m, 1), device="cuda", dtype=torch.int32
        )


class _GroupedNopadWarmupExecutor_int4(_BaseWarmupExecutor):
    def __init__(self, max_m: int, n: int, k: int, num_groups: int):
        super().__init__(max_m, n, k, num_groups)

    def setup_tensors(self, max_m: int, n: int, k: int, num_groups: int):
        self.lhs = _empty_token_bf16((max_m, k))
        self.rhs_q, self.rhs_s = _empty_marlin_int4((num_groups, n, k))
        self.m_indices = torch.zeros((max_m,), device="cuda", dtype=torch.int32)
        self.out = torch.empty((max_m, n), device="cuda", dtype=torch.bfloat16)

    def execute(self, m):
        deep_gemm.m_grouped_gemm_w4a16_nopad(
            self.lhs[:m],
            (self.rhs_q, self.rhs_s),
            self.out[:m],
            m_indices=self.m_indices[:m],
        )


class _GroupedNopadWarmupExecutor_int4_mma(_GroupedNopadWarmupExecutor_int4):
    def setup_tensors(self, max_m: int, n: int, k: int, num_groups: int):
        self.lhs = _empty_token_bf16((max_m, k))
        self.rhs_q, self.rhs_s = _empty_mxfp4_w4a16_mma((num_groups, n, k))
        self.m_indices = torch.zeros((max_m,), device="cuda", dtype=torch.int32)
        self.out = torch.empty((max_m, n), device="cuda", dtype=torch.bfloat16)


_SAIL_WARMUP_EXECUTORS = {
    SailDeepGemmKernelType.GROUPED_GEMM_NT_F8F8BF16_NOPAD: _GroupedNopadWarmupExecutor,
    SailDeepGemmKernelType.GEMM_NT_F8F8BF16_CHANNEL: _NormalWarmupExecutor_channel,
    SailDeepGemmKernelType.GROUPED_GEMM_NT_F8F8BF16_CONTIG_CHANNEL: _GroupedContWarmupExecutor_channel,
    SailDeepGemmKernelType.GROUPED_GEMM_NT_F8F8BF16_MASKED_CHANNEL: _GroupedMaskedWarmupExecutor_channel,
    SailDeepGemmKernelType.GROUPED_GEMM_NT_F8F8BF16_NOPAD_CHANNEL: _GroupedNopadWarmupExecutor_channel,
    SailDeepGemmKernelType.GROUPED_GEMM_NT_I8I8BF16_MASKED: _GroupedMaskedWarmupExecutor_int8,
    SailDeepGemmKernelType.GROUPED_GEMM_NT_I8I8BF16_CONTIG: _GroupedContWarmupExecutor_int8,
    SailDeepGemmKernelType.GROUPED_GEMM_NT_I8I8BF16_NOPAD: _GroupedNopadWarmupExecutor_int8,
    SailDeepGemmKernelType.GEMM_NT_I8I8BF16: _NormalWarmupExecutor_int8,
    SailDeepGemmKernelType.GROUPED_GEMM_NT_BF16_NOPAD: _GroupedNopadWarmupExecutor_bf16,
    SailDeepGemmKernelType.GROUPED_GEMM_NT_F4F4BF16_MASKED: _GroupedMaskedWarmupExecutor_fp4,
    SailDeepGemmKernelType.GROUPED_GEMM_NT_F4F4BF16_NOPAD: _GroupedNopadWarmupExecutor_fp4,
    SailDeepGemmKernelType.GEMM_NT_F4F4BF16: _NormalWarmupExecutor_fp4,
    SailDeepGemmKernelType.GROUPED_GEMM_NT_F4F4BF16_MASKED_BIAS: _GroupedMaskedWarmupExecutor_fp4_bias,
    SailDeepGemmKernelType.GROUPED_GEMM_NT_F4F4BF16_NOPAD_BIAS: _GroupedNopadWarmupExecutor_fp4_bias,
    SailDeepGemmKernelType.GEMM_NT_F4F4BF16_BIAS: _NormalWarmupExecutor_fp4_bias,
    SailDeepGemmKernelType.GROUPED_GEMM_NT_BF16I4BF16_MASKED: _GroupedMaskedWarmupExecutor_int4,
    SailDeepGemmKernelType.GROUPED_GEMM_NT_BF16I4BF16_NOPAD: _GroupedNopadWarmupExecutor_int4,
    SailDeepGemmKernelType.GROUPED_GEMM_NT_F8F8BF16_FUSED: _GroupedFusedWarmupExecutor,
    SailDeepGemmKernelType.GROUPED_GEMM_NT_F8F8BF16_FUSED_CHANNEL: _GroupedFusedWarmupExecutor_fp8_channel,
    SailDeepGemmKernelType.GROUPED_GEMM_NT_I8I8BF16_FUSED: _GroupedFusedWarmupExecutor_int8,
    SailDeepGemmKernelType.GROUPED_GEMM_NT_BF16_FUSED: _GroupedFusedWarmupExecutor_bf16,
    SailDeepGemmKernelType.GROUPED_GEMM_NT_F4F4BF16_FUSED: _GroupedFusedWarmupExecutor_fp4,
    SailDeepGemmKernelType.GROUPED_GEMM_NT_BF16I4BF16_MMA_MASKED: _GroupedMaskedWarmupExecutor_int4_mma,
    SailDeepGemmKernelType.GROUPED_GEMM_NT_BF16I4BF16_MMA_NOPAD: _GroupedNopadWarmupExecutor_int4_mma,
    SailDeepGemmKernelType.GROUPED_GEMM_NT_BF16I4BF16_MMA_FUSED: _GroupedFusedWarmupExecutor_int4_mma,
    SailDeepGemmKernelType.GROUPED_GEMM_NT_BF16I4BF16_FUSED: _GroupedFusedWarmupExecutor_int4,
    SailDeepGemmKernelType.GROUPED_GEMM_NT_BF16I4BF16_VALU_FUSED: _GroupedFusedWarmupExecutor_mxfp4_valu,
}


# Grouped layouts this build stores contiguously along M. Only the M values the
# build's M alignment admits compile, so the shared warmup trims its M list to
# them before compiling.
_CONTIG_LAYOUTS = frozenset(
    {
        SailDeepGemmKernelType.GROUPED_GEMM_NT_F8F8BF16_CONTIG_CHANNEL.name,
        SailDeepGemmKernelType.GROUPED_GEMM_NT_I8I8BF16_CONTIG.name,
    }
)


def is_contig_layout(name: str) -> bool:
    """Whether this build stores the grouped GEMM contiguously along M."""
    return name in _CONTIG_LAYOUTS


def create_sail_warmup_executor(kernel_type: SailDeepGemmKernelType, **kwargs):
    """Build the warmup executor for a kernel type the shared table omits."""
    return _SAIL_WARMUP_EXECUTORS[kernel_type](**kwargs)
