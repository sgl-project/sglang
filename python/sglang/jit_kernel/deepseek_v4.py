from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, NamedTuple, Optional, Tuple, Union

import torch
import triton
import triton.language as tl

from sglang.jit_kernel.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)
from sglang.srt.debug_utils.deepseek_v4_debug_utils import (
    deepseek_v4_moe_code_path_checker,
)
from sglang.srt.environ import envs
from sglang.srt.utils import is_cuda, is_xpu

if TYPE_CHECKING:
    from tvm_ffi.module import Module


# JIT-compiled CUDA kernels in this module require tvm_ffi and a working CUDA
# toolchain. On non-CUDA backends (e.g. XPU) those entrypoints fall back to
# triton/torch implementations.
_is_cuda = is_cuda()
_is_xpu = is_xpu()

def make_name(name: str) -> str:
    return f"dpsk_v4_{name}"


@cache_once
def _jit_common_module() -> Module:
    return load_jit(
        make_name(f"common"),
        cuda_files=[f"deepseek_v4/common.cuh"],
        cuda_wrappers=[("plan_compress_prefill", "plan_compress_prefill")],
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


@cache_once
def _jit_topk_module() -> Module:
    args = make_cpp_args(is_arch_support_pdl())
    return load_jit(
        make_name("topk"),
        *args,
        cuda_files=["deepseek_v4/topk.cuh"],
        cuda_wrappers=[("topk_transform", f"TopK512Kernel<{args}>::transform")],
    )


@cache_once
def _jit_topk1024_module() -> Module:
    args = make_cpp_args(is_arch_support_pdl())
    return load_jit(
        make_name("topk1024"),
        *args,
        cuda_files=["deepseek_v4/topk_1024.cuh"],
        cuda_wrappers=[("topk_transform", f"TopK1024Kernel<{args}>::transform")],
    )


@cache_once
def _jit_topk_v2_module(topk: int) -> Module:
    return load_jit(
        make_name("topk_v2"),
        str(topk),
        cuda_files=["deepseek_v4/topk_v2.cuh"],
        cuda_wrappers=[
            ("topk_transform", "CombinedTopKKernel::transform"),
            ("topk_plan", "CombinedTopKKernel::plan"),
        ],
        extra_cuda_cflags=[f"-DSGL_TOPK={topk}"],
    )


@cache_once
def _jit_mask_topk_module() -> Module:
    return load_jit(
        make_name("mask_topk"),
        cuda_files=["deepseek_v4/hash_topk.cuh"],
        cuda_wrappers=[("run", "MaskKernel::run")],
    )


@cache_once
def _jit_hash_topk_module() -> Module:
    args = make_cpp_args("act_sqrt_softplus", is_arch_support_pdl())
    return load_jit(
        make_name("hash_topk"),
        *args,
        cuda_files=["deepseek_v4/hash_topk.cuh"],
        cuda_wrappers=[("hash_topk", f"HashTopKKernel<{args}>::run")],
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
        make_name(f"compress_{ratio}"),
        *args,
        cuda_files=[f"deepseek_v4/c{ratio}.cuh"],
        cuda_wrappers=[
            ("decode", f"{kernel_class}::run_decode"),
            ("prefill", f"{kernel_class}::run_prefill"),
        ],
        extra_cuda_cflags=["-use_fast_math"],
    )


@cache_once
def _jit_compress_module_v2_defensive(
    head_dim: int,
    dtype_in: torch.dtype,
    dtype_out: torch.dtype,
) -> Module:
    args = make_cpp_args(head_dim, dtype_in, dtype_out, is_arch_support_pdl())
    kernel_class = f"FlashCompress128Kernel<{args}>"
    return load_jit(
        make_name("compress_128_v2_defensive"),
        *args,
        cuda_files=["deepseek_v4/c128_v2.cuh"],
        cuda_wrappers=[
            ("prefill", f"{kernel_class}::run_prefill"),
        ],
        extra_cuda_cflags=["-use_fast_math"],
    )


@cache_once
def _jit_rmsnorm_head_module(head_dim: int, dtype: torch.dtype):
    args = make_cpp_args(head_dim, dtype, is_arch_support_pdl())
    kernel_class = f"RMSNormKernel<{args}>"
    return load_jit(
        make_name("rmsnorm_head"),
        *args,
        cuda_files=["deepseek_v4/rmsnorm.cuh"],
        cuda_wrappers=[("run_self", f"{kernel_class}::run_self")],
    )


@cache_once
def _jit_fused_rope_module() -> Module:
    args = make_cpp_args(is_arch_support_pdl())
    return load_jit(
        make_name("fused_rope"),
        *args,
        cuda_files=["deepseek_v4/rope.cuh"],
        cuda_wrappers=[("forward", f"FusedQKRopeKernel<{args}>::forward")],
    )


@cache_once
def _jit_norm_rope_module(
    dtype: torch.dtype,
    head_dim: int,
    rope_dim: int,
) -> Module:
    args = make_cpp_args(dtype, head_dim, rope_dim, is_arch_support_pdl())
    return load_jit(
        make_name(f"fused_norm_rope"),
        *args,
        cuda_files=[f"deepseek_v4/fused_norm_rope.cuh"],
        cuda_wrappers=[
            ("forward", f"FusedNormRopeKernel<{args}>::forward"),
        ],
    )


@cache_once
def _jit_fused_store_module(
    name: Literal["flashmla", "indexer"],
    input_dtype: torch.dtype,
    index_dtype: torch.dtype,
    page_size: int,
) -> Module:
    args = make_cpp_args(input_dtype, index_dtype, page_size, is_arch_support_pdl())
    cname = "FlashMLA" if name == "flashmla" else "Indexer"
    kernel_class = f"FusedStoreCache{cname}Kernel<{args}>"
    return load_jit(
        make_name("store_" + name),
        *args,
        cuda_files=["deepseek_v4/store.cuh"],
        cuda_wrappers=[("run", f"{kernel_class}::run")],
    )


@cache_once
def _jit_metadata_module():
    return load_jit(
        make_name("metadata"),
        cuda_files=["deepseek_v4/paged_mqa_metadata.cuh"],
        cuda_wrappers=[("run", "IndexerMetadataKernel::run")],
    )


@cache_once
def _jit_silu_mul_quant_varlen_module(
    quant_group_size: int,
    scale_ue8m0: bool,
    swizzle: bool,
    apply_swiglu_limit: bool,
) -> Module:
    args = make_cpp_args(
        quant_group_size,
        scale_ue8m0,
        swizzle,
        is_arch_support_pdl(),
        apply_swiglu_limit,
    )
    return load_jit(
        make_name("silu_mul_quant_varlen"),
        *args,
        cuda_files=["deepseek_v4/silu_and_mul_masked_post_quant.cuh"],
        cuda_wrappers=[("run", f"SiluAndMulMaskedPostQuantKernel<{args}>::run")],
        extra_cuda_cflags=["-use_fast_math"],
    )


@cache_once
def _jit_silu_mul_quant_contig_module(
    quant_group_size: int,
    scale_ue8m0: bool,
    swizzle: bool,
    apply_swiglu_limit: bool,
) -> Module:
    args = make_cpp_args(
        quant_group_size,
        scale_ue8m0,
        swizzle,
        is_arch_support_pdl(),
        apply_swiglu_limit,
    )
    return load_jit(
        make_name("silu_mul_quant_contig"),
        *args,
        cuda_files=["deepseek_v4/silu_and_mul_masked_post_quant.cuh"],
        cuda_wrappers=[("run", f"SiluAndMulContigPostQuantKernel<{args}>::run")],
        extra_cuda_cflags=["-use_fast_math"],
    )


@cache_once
def _jit_silu_and_mul_clamp_module(dtype: torch.dtype) -> Module:
    args = make_cpp_args(dtype, is_arch_support_pdl())
    return load_jit(
        make_name("silu_and_mul_clamp"),
        *args,
        cuda_files=["deepseek_v4/silu_and_mul_masked_post_quant.cuh"],
        cuda_wrappers=[("run", f"SiluAndMulClampKernel<{args}>::run")],
        extra_cuda_cflags=["-use_fast_math"],
    )


# ---------------------------------------------------------------------------
# Byte-equal fallbacks: when SGLANG_OPT_FIX_MEGA_MOE_MEMORY is off, route
# silu_and_mul_masked_post_quant / silu_and_mul_clamp through these _tmp
# modules, which load a copy of the optimize-branch kernel (different
# precision behavior ??? bf16 silu roundtrip, expf, fp32 clamp).
# ---------------------------------------------------------------------------


@cache_once
def _jit_silu_mul_quant_tmp_module(
    quant_group_size: int, scale_ue8m0: bool, apply_swiglu_limit: bool
) -> Module:
    args = make_cpp_args(
        quant_group_size, scale_ue8m0, is_arch_support_pdl(), apply_swiglu_limit
    )
    return load_jit(
        make_name("silu_mul_quant_tmp"),
        *args,
        cuda_files=["deepseek_v4/silu_and_mul_masked_post_quant_tmp.cuh"],
        cuda_wrappers=[("run", f"SiluAndMulMaskedPostQuantKernel<{args}>::run")],
    )


@cache_once
def _jit_silu_and_mul_clamp_tmp_module(dtype: torch.dtype) -> Module:
    args = make_cpp_args(dtype, is_arch_support_pdl())
    return load_jit(
        make_name("silu_and_mul_clamp_tmp"),
        *args,
        cuda_files=["deepseek_v4/silu_and_mul_masked_post_quant_tmp.cuh"],
        cuda_wrappers=[("run", f"SiluAndMulClampKernel<{args}>::run")],
    )


@cache_once
def _jit_mega_moe_pre_dispatch_module(quant_group_size: int) -> Module:
    args = make_cpp_args(quant_group_size, is_arch_support_pdl())
    return load_jit(
        make_name("mega_moe_pre_dispatch"),
        *args,
        cuda_files=["deepseek_v4/mega_moe_pre_dispatch.cuh"],
        cuda_wrappers=[("run", f"MegaMoEPreDispatchKernel<{args}>::run")],
    )


@cache_once
def _jit_hisparse_transfer_module() -> Module:
    return load_jit(
        make_name("hisparse_transfer"),
        cuda_files=["deepseek_v4/hisparse_transfer.cuh"],
        cuda_wrappers=[("hisparse_transfer", "hisparse_transfer")],
    )


def hisparse_offload_to_host(
    gpu_ptrs: torch.Tensor,
    cpu_ptrs: torch.Tensor,
    gpu_indices: torch.Tensor,
    cpu_indices: torch.Tensor,
) -> None:
    module = _jit_hisparse_transfer_module()
    module.hisparse_transfer(gpu_ptrs, cpu_ptrs, gpu_indices, cpu_indices)


def topk_transform_512(
    scores: torch.Tensor,
    seq_lens: torch.Tensor,
    page_tables: torch.Tensor,
    out_page_indices: torch.Tensor,
    page_size: int,
    out_raw_indices: Optional[torch.Tensor] = None,
) -> None:
    if out_page_indices.shape[1] == 512:
        module = _jit_topk_module()
    else:
        module = _jit_topk1024_module()
    module.topk_transform(
        scores, seq_lens, page_tables, out_page_indices, page_size, out_raw_indices
    )


_WORKSPACE_INTS_PER_BATCH = 2 + 1024 * 2
_PLAN_METADATA_INTS_PER_BATCH = 4


def plan_topk_v2(seq_lens: torch.Tensor, static_threshold: int = 0) -> torch.Tensor:
    module = _jit_topk_v2_module(512)  # does not matter
    bs = seq_lens.shape[0]
    metadata = seq_lens.new_empty(bs + 1, _PLAN_METADATA_INTS_PER_BATCH)
    module.topk_plan(seq_lens, metadata, static_threshold)
    return metadata


def topk_transform_512_v2(
    scores: torch.Tensor,
    seq_lens: torch.Tensor,
    page_tables: torch.Tensor,
    out_page_indices: torch.Tensor,
    page_size: int,
    metadata: torch.Tensor,
) -> None:
    module = _jit_topk_v2_module(out_page_indices.shape[1])
    bs = scores.shape[0]
    workspace = seq_lens.new_empty(bs, _WORKSPACE_INTS_PER_BATCH)
    module.topk_transform(
        scores,
        seq_lens,
        page_tables,
        out_page_indices,
        page_size,
        workspace,
        metadata,
    )


def hash_topk(
    router_logits: torch.Tensor,
    input_ids: torch.Tensor,
    tid2eid: torch.Tensor,
    num_fused_shared_experts: int = 0,
    routed_scaling_factor: float = 1.0,
    scoring_func: str = "sqrtsoftplus",
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert scoring_func == "sqrtsoftplus"
    num_tokens = router_logits.size(0)
    topk_routed = tid2eid.size(1)
    topk_fused = topk_routed + num_fused_shared_experts
    topk_ids = torch.empty(
        (num_tokens, topk_fused), dtype=torch.int32, device=router_logits.device
    )
    topk_weights = torch.empty(
        (num_tokens, topk_fused), dtype=torch.float32, device=router_logits.device
    )
    module = _jit_hash_topk_module()
    module.hash_topk(
        router_logits,
        input_ids,
        tid2eid,
        topk_weights,
        topk_ids,
        routed_scaling_factor,
    )
    return topk_weights, topk_ids


def mask_topk_ids(topk_ids: torch.Tensor, num_token_non_padded: torch.Tensor):
    return _jit_mask_topk_module().run(topk_ids, num_token_non_padded)


class CompressorPrefillPlan(NamedTuple):
    compress_ratio: int
    compress_plan: torch.Tensor
    write_plan: torch.Tensor

    def copy_(self, other: CompressorPrefillPlan) -> None:
        assert self.compress_ratio == other.compress_ratio
        self.compress_plan.copy_(other.compress_plan)
        self.write_plan.copy_(other.write_plan)

    @staticmethod
    def generate(
        compress_ratio: Literal[4, 128],
        num_q_tokens: int,
        seq_lens: torch.Tensor,
        extend_lens: torch.Tensor,
        device: torch.device,
        use_cuda_graph: bool = False,
    ) -> CompressorPrefillPlan:
        from sglang.srt.environ import envs

        # Online c128 keeps the same NamedTuple shape (compress_plan, write_plan)
        # so call sites that splat `*plan[1:]` continue to work, but the C++
        # plan struct semantics differ (last-token coords + window_len).
        if compress_ratio == 128 and envs.SGLANG_OPT_USE_ONLINE_COMPRESS.get():
            return CompressorPrefillPlan._generate_online(
                num_q_tokens=num_q_tokens,
                seq_lens=seq_lens,
                extend_lens=extend_lens,
                device=device,
                use_cuda_graph=use_cuda_graph,
            )
        assert seq_lens.device == extend_lens.device
        seq_lens = seq_lens.to(torch.int64)
        extend_lens = extend_lens.to(torch.int64)
        plan_tensor = torch.empty(
            (2, num_q_tokens, 16),
            dtype=torch.uint8,
            device=seq_lens.device,
            pin_memory=seq_lens.is_cpu,
        )
        is_overlap = compress_ratio == 4
        if _is_cuda:
            module = _jit_common_module()
            plan_lens = module.plan_compress_prefill(
                extend_lens,
                seq_lens,
                plan_tensor[0],
                plan_tensor[1],
                compress_ratio,
                is_overlap,
                use_cuda_graph,
            )
        else:
            # Pure-torch fallback on non-CUDA backends (e.g. XPU). Mirrors
            # plan_prefill_host in jit_kernel/csrc/deepseek_v4/common.cuh.
            plan_lens = _torch_plan_compress_prefill(
                extend_lens,
                seq_lens,
                plan_tensor[0],
                plan_tensor[1],
                compress_ratio,
                is_overlap,
                use_cuda_graph,
            )
        return CompressorPrefillPlan(
            compress_ratio,
            plan_tensor[0, : plan_lens[0]].to(device, non_blocking=True),
            plan_tensor[1, : plan_lens[1]].to(device, non_blocking=True),
        )

    @staticmethod
    def _generate_online(
        num_q_tokens: int,
        seq_lens: torch.Tensor,
        extend_lens: torch.Tensor,
        device: torch.device,
        use_cuda_graph: bool,
    ) -> CompressorPrefillPlan:
        # Online plan host-side path: only CPU/cuda-host implemented today.
        # Move inputs to CPU pinned memory then bounce the result to device.
        seq_lens_cpu = seq_lens.detach().to(torch.int64).cpu()
        extend_lens_cpu = extend_lens.detach().to(torch.int64).cpu()
        plan_tensor = torch.empty(
            (2, num_q_tokens, 16),
            dtype=torch.uint8,
            device="cpu",
            pin_memory=True,
        )
        module = _jit_compress_128_online_plan_module()
        plan_lens = module.plan_compress_online_prefill(
            extend_lens_cpu,
            seq_lens_cpu,
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


def _torch_plan_compress_prefill(
    extend_lens: torch.Tensor,
    seq_lens: torch.Tensor,
    compress_plan: torch.Tensor,
    write_plan: torch.Tensor,
    compress_ratio: int,
    is_overlap: bool,
    use_cuda_graph: bool,
) -> Tuple[int, int]:
    """Pure-torch fallback for ``plan_compress_prefill``.

    Mirrors ``host::compress::plan_prefill_host`` in
    ``jit_kernel/csrc/deepseek_v4/common.cuh``. Each plan slot is
    ``PrefillPlan{ragged_id, batch_id, position, window_len}`` packed as
    4 little-endian uint32 (16 bytes).
    """
    import struct

    assert compress_plan.dtype == torch.uint8
    assert write_plan.dtype == torch.uint8
    num_tokens = compress_plan.shape[0]
    assert write_plan.shape[0] == num_tokens
    assert compress_plan.shape[1] == 16 and write_plan.shape[1] == 16

    extend_lens_cpu = extend_lens.detach().to("cpu", dtype=torch.int64).tolist()
    seq_lens_cpu = seq_lens.detach().to("cpu", dtype=torch.int64).tolist()
    batch_size = len(extend_lens_cpu)
    assert len(seq_lens_cpu) == batch_size

    ratio = compress_ratio * (2 if is_overlap else 1)
    counter = 0
    compress_entries: list = []
    write_entries: list = []

    for i in range(batch_size):
        seq_len = int(seq_lens_cpu[i])
        extend_len = int(extend_lens_cpu[i])
        assert 0 < extend_len <= seq_len
        prefix_len = seq_len - extend_len
        pos = (seq_len // compress_ratio) * compress_ratio
        if is_overlap:
            start_write_pos = pos - compress_ratio if pos >= compress_ratio else 0
        else:
            start_write_pos = pos
        for j in range(extend_len):
            position = prefix_len + j
            window_len = ratio - min(j + 1, ratio)
            plan = (counter + j, i, position, window_len)
            if (position + 1) % compress_ratio == 0:
                compress_entries.append(plan)
            if position >= start_write_pos:
                write_entries.append(plan)
        counter += extend_len
    assert (
        counter == num_tokens
    ), f"input size {counter} != num_q_tokens {num_tokens}"

    kInvalid = 0xFFFFFFFF
    invalid_row = struct.pack("<IIII", kInvalid, kInvalid, kInvalid, kInvalid)

    def _fill(buf: torch.Tensor, entries: list) -> int:
        n_entries = len(entries)
        n_rows = num_tokens if use_cuda_graph else n_entries
        if n_rows == 0:
            return num_tokens if use_cuda_graph else 0
        valid_bytes = b"".join(struct.pack("<IIII", *e) for e in entries)
        if use_cuda_graph and n_entries < num_tokens:
            payload = valid_bytes + invalid_row * (num_tokens - n_entries)
        else:
            payload = valid_bytes
        cpu_view = torch.frombuffer(bytearray(payload), dtype=torch.uint8).view(
            n_rows, 16
        )
        buf[:n_rows].copy_(cpu_view)
        return num_tokens if use_cuda_graph else n_entries

    compress_count = _fill(compress_plan, compress_entries)
    write_count = _fill(write_plan, write_entries)
    return compress_count, write_count


def _decode_prefill_plan(plan_bytes: torch.Tensor) -> torch.Tensor:
    """Decode packed PrefillPlan tensor ([N, 16] uint8) into [N, 4] int64.

    Each plan slot is 4 little-endian uint32:
    (ragged_id, batch_id, position, window_len). Invalid entries use
    ``0xFFFFFFFF`` for every field.

    On-device bitcast: avoids a D->H sync that drains the L0 queue every
    layer on XPU. uint8 [N, 16] is reinterpreted as int32 [N, 4] (LE on
    XPU/x86), promoted to int64, then masked to keep the unsigned value
    so that the kInvalid sentinel 0xFFFFFFFF (which would bitcast to -1
    in int32) compares correctly.
    """
    t = plan_bytes.detach().contiguous()
    as_i32 = t.view(torch.int32).reshape(-1, 4)
    return as_i32.to(torch.int64) & 0xFFFFFFFF


# ---------------------------------------------------------------------------
# Triton fallbacks for non-CUDA backends (XPU/AMD).
#
# These kernels mirror the CUDA c{4,128} kernels exactly. Each program
# fuses gather + softmax + weighted-sum for one (work_item, head_dim_split)
# tile, avoiding the intermediate [B, 128, HD] / [N, 128, HD] allocations
# a torch implementation would need.
#
# Layout reminders (matching the C++ kernels):
#   c4 buffer:  [N_idx, page_size, 4*HD]  | kv_overlap | kv | score_ovr | score |
#               page_size = 4 (paged) or 8 (ring)
#   c4 input:   [N_q,                4*HD] same chunk layout
#   c128 buffer:[N_idx, 128,         2*HD] | kv | score |
#   c128 input: [N_q,                2*HD] same chunk layout
#   ape:        [SEQ, HD]   SEQ = 8 for c4, 128 for c128
#
# Plan tensor on entry is [N_plan, 16] uint8; we view it as int32 [N_plan, 4]:
#   (ragged_id, batch_id, position, window_len), with 0xFFFFFFFF (== -1 as
#   int32) as the invalid sentinel.
# ---------------------------------------------------------------------------


def _pick_block_hd(head_dim: int) -> int:
    # Triton tiles need power-of-two; HD is always a multiple of 128 here.
    return head_dim if head_dim <= 128 else 128


# --- c128 decode -----------------------------------------------------------


@triton.jit
def _triton_c128_decode_kernel(
    buf_ptr,            # [N_idx, 128, 2*HD]
    inp_ptr,            # [B, 2*HD]
    out_ptr,            # [B, HD]
    ape_ptr,            # [128, HD]
    indices_ptr,        # [B] int32
    seq_lens_ptr,       # [B] int32
    HD: tl.constexpr,
    BLOCK_HD: tl.constexpr,
):
    SEQ: tl.constexpr = 128
    pid_b = tl.program_id(0)
    pid_s = tl.program_id(1)

    index = tl.load(indices_ptr + pid_b).to(tl.int64)
    seq_len = tl.load(seq_lens_ptr + pid_b).to(tl.int64)

    offs_hd = pid_s * BLOCK_HD + tl.arange(0, BLOCK_HD)
    hd_mask = offs_hd < HD

    # Load the current step's kv + score (input chunks 0 and 1).
    inp_base = inp_ptr + pid_b * (2 * HD)
    inp_kv = tl.load(inp_base + offs_hd, mask=hd_mask)
    inp_score = tl.load(inp_base + HD + offs_hd, mask=hd_mask)

    do_fwd = (seq_len % SEQ) == 0

    if do_fwd:
        # When seq_len % 128 == 0, write_pos == 127. Read positions 0..126
        # from the buffer; supply position 127 from `inp_kv`/`inp_score`
        # directly to avoid depending on store-then-load coherence inside
        # the same program.
        offs_s = tl.arange(0, SEQ)
        is_last = (offs_s == (SEQ - 1))[:, None]

        slot_off_2d = (
            index * (SEQ * 2 * HD)
            + offs_s[:, None].to(tl.int64) * (2 * HD)
            + offs_hd[None, :].to(tl.int64)
        )
        load_mask = (~is_last) & hd_mask[None, :]
        buf_kv = tl.load(buf_ptr + slot_off_2d, mask=load_mask, other=0.0).to(tl.float32)
        buf_score = tl.load(buf_ptr + slot_off_2d + HD, mask=load_mask, other=0.0).to(tl.float32)

        kv_t = tl.where(is_last, inp_kv[None, :].to(tl.float32), buf_kv)
        score_t = tl.where(is_last, inp_score[None, :].to(tl.float32), buf_score)

        bias = tl.load(
            ape_ptr + offs_s[:, None] * HD + offs_hd[None, :],
            mask=hd_mask[None, :],
            other=0.0,
        ).to(tl.float32)

        s = score_t + bias
        m = tl.max(s, axis=0)
        w = tl.exp(s - m[None, :])
        num = tl.sum(kv_t * w, axis=0)
        den = tl.sum(w, axis=0)
        result = num / den

        tl.store(out_ptr + pid_b * HD + offs_hd, result, mask=hd_mask)

    # Always commit the current step into the buffer at write_pos. Done last
    # so the conditional forward above never reads our own pending store.
    write_pos = (seq_len + (SEQ - 1)) % SEQ
    write_base = buf_ptr + index * (SEQ * 2 * HD) + write_pos * (2 * HD)
    tl.store(write_base + offs_hd, inp_kv, mask=hd_mask)
    tl.store(write_base + HD + offs_hd, inp_score, mask=hd_mask)


# --- c128 prefill ----------------------------------------------------------


@triton.jit
def _triton_c128_prefill_compress_kernel(
    buf_ptr,
    inp_ptr,
    out_ptr,
    ape_ptr,
    plan_ptr,           # [N_plan, 4] int32 (ragged_id, batch_id, position, window_len)
    load_indices_ptr,   # [B] int32  (== `indices` when no extra is provided)
    HD: tl.constexpr,
    BLOCK_HD: tl.constexpr,
):
    SEQ: tl.constexpr = 128
    pid_p = tl.program_id(0)
    pid_s = tl.program_id(1)

    plan_base = plan_ptr + pid_p * 4
    ragged_id = tl.load(plan_base + 0)
    batch_id = tl.load(plan_base + 1)
    window_len = tl.load(plan_base + 3)

    if ragged_id == -1:
        return

    index = tl.load(load_indices_ptr + batch_id.to(tl.int64)).to(tl.int64)

    offs_hd = pid_s * BLOCK_HD + tl.arange(0, BLOCK_HD)
    hd_mask = offs_hd < HD

    offs_s = tl.arange(0, SEQ)
    use_buf = (offs_s < window_len)[:, None]

    # Buffer source (j < window_len).
    buf_off_2d = (
        index * (SEQ * 2 * HD)
        + offs_s[:, None].to(tl.int64) * (2 * HD)
        + offs_hd[None, :].to(tl.int64)
    )
    buf_load_mask = use_buf & hd_mask[None, :]
    buf_kv = tl.load(buf_ptr + buf_off_2d, mask=buf_load_mask, other=0.0).to(tl.float32)
    buf_score = tl.load(buf_ptr + buf_off_2d + HD, mask=buf_load_mask, other=0.0).to(tl.float32)

    # Ragged tail source: rag_off = max(ragged_id + j - 127, 0).
    rag_off = tl.maximum(
        ragged_id.to(tl.int64) + offs_s.to(tl.int64) - (SEQ - 1), 0
    )
    rag_off_2d = rag_off[:, None] * (2 * HD) + offs_hd[None, :].to(tl.int64)
    rag_load_mask = (~use_buf) & hd_mask[None, :]
    rag_kv = tl.load(inp_ptr + rag_off_2d, mask=rag_load_mask, other=0.0).to(tl.float32)
    rag_score = tl.load(inp_ptr + rag_off_2d + HD, mask=rag_load_mask, other=0.0).to(tl.float32)

    kv_t = tl.where(use_buf, buf_kv, rag_kv)
    score_t = tl.where(use_buf, buf_score, rag_score)

    bias = tl.load(
        ape_ptr + offs_s[:, None] * HD + offs_hd[None, :],
        mask=hd_mask[None, :],
        other=0.0,
    ).to(tl.float32)

    s = score_t + bias
    m = tl.max(s, axis=0)
    w = tl.exp(s - m[None, :])
    num = tl.sum(kv_t * w, axis=0)
    den = tl.sum(w, axis=0)
    result = num / den

    tl.store(out_ptr + ragged_id.to(tl.int64) * HD + offs_hd, result, mask=hd_mask)


@triton.jit
def _triton_c128_prefill_write_kernel(
    buf_ptr,
    inp_ptr,
    plan_ptr,
    indices_ptr,
    HD: tl.constexpr,
    BLOCK_HD: tl.constexpr,
):
    SEQ: tl.constexpr = 128
    pid_p = tl.program_id(0)
    pid_s = tl.program_id(1)

    plan_base = plan_ptr + pid_p * 4
    ragged_id = tl.load(plan_base + 0)
    batch_id = tl.load(plan_base + 1)
    position = tl.load(plan_base + 2)

    if ragged_id == -1:
        return

    index = tl.load(indices_ptr + batch_id.to(tl.int64)).to(tl.int64)

    offs_hd = pid_s * BLOCK_HD + tl.arange(0, BLOCK_HD)
    hd_mask = offs_hd < HD

    pos = (position % SEQ).to(tl.int64)
    src_base = inp_ptr + ragged_id.to(tl.int64) * (2 * HD)
    tgt_base = buf_ptr + index * (SEQ * 2 * HD) + pos * (2 * HD)

    src_kv = tl.load(src_base + offs_hd, mask=hd_mask)
    src_score = tl.load(src_base + HD + offs_hd, mask=hd_mask)
    tl.store(tgt_base + offs_hd, src_kv, mask=hd_mask)
    tl.store(tgt_base + HD + offs_hd, src_score, mask=hd_mask)


# --- c4 decode -------------------------------------------------------------


@triton.jit
def _triton_c4_decode_kernel(
    buf_ptr,
    inp_ptr,
    out_ptr,
    ape_ptr,
    indices_ptr,
    seq_lens_ptr,
    extra_ptr,            # [B] int32 (index_prev) for paged; ignored for ring
    HD: tl.constexpr,
    BLOCK_HD: tl.constexpr,
    PAGE_SIZE: tl.constexpr,    # 4 (paged) or 8 (ring)
):
    SEQ: tl.constexpr = 8
    NUM_CHUNKS: tl.constexpr = 4
    PAGED: tl.constexpr = (PAGE_SIZE == 4)
    STRIDE_IDX: tl.constexpr = PAGE_SIZE * NUM_CHUNKS * HD
    STRIDE_POS: tl.constexpr = NUM_CHUNKS * HD

    pid_b = tl.program_id(0)
    pid_s = tl.program_id(1)

    index = tl.load(indices_ptr + pid_b).to(tl.int64)
    seq_len = tl.load(seq_lens_ptr + pid_b).to(tl.int64)

    offs_hd = pid_s * BLOCK_HD + tl.arange(0, BLOCK_HD)
    hd_mask = offs_hd < HD

    # Input chunks (0=kv_ovr, 1=kv, 2=score_ovr, 3=score).
    inp_base = inp_ptr + pid_b * (NUM_CHUNKS * HD)
    inp_kv_ovr = tl.load(inp_base + 0 * HD + offs_hd, mask=hd_mask)
    inp_kv = tl.load(inp_base + 1 * HD + offs_hd, mask=hd_mask)
    inp_score_ovr = tl.load(inp_base + 2 * HD + offs_hd, mask=hd_mask)
    inp_score = tl.load(inp_base + 3 * HD + offs_hd, mask=hd_mask)

    do_fwd = (seq_len % 4) == 0

    if do_fwd:
        offs_s = tl.arange(0, SEQ)
        chunk_kv_off = tl.where(offs_s < 4, 0, HD).to(tl.int64)
        chunk_score_off = chunk_kv_off + 2 * HD

        if PAGED:
            index_prev = tl.load(extra_ptr + pid_b).to(tl.int64)
            page_idx = tl.where(offs_s < 4, index_prev, index)
            slot_pos = (offs_s % 4).to(tl.int64)
            slot_base_2d = page_idx[:, None] * STRIDE_IDX + slot_pos[:, None] * STRIDE_POS
        else:
            slot_pos = (seq_len + offs_s.to(tl.int64)) % 8
            slot_base_2d = index * STRIDE_IDX + slot_pos[:, None] * STRIDE_POS

        kv_off_2d = slot_base_2d + chunk_kv_off[:, None] + offs_hd[None, :].to(tl.int64)
        score_off_2d = slot_base_2d + chunk_score_off[:, None] + offs_hd[None, :].to(tl.int64)

        # i == 7 (newest entry) is supplied by the input directly to avoid
        # depending on store-then-load coherence (we may not have written it
        # yet; even if we had, the buffer write happens at the end).
        is_i7 = (offs_s == 7)[:, None]
        load_mask = (~is_i7) & hd_mask[None, :]
        buf_kv = tl.load(buf_ptr + kv_off_2d, mask=load_mask, other=0.0).to(tl.float32)
        buf_score = tl.load(buf_ptr + score_off_2d, mask=load_mask, other=0.0).to(tl.float32)

        kv_t = tl.where(is_i7, inp_kv[None, :].to(tl.float32), buf_kv)
        score_t = tl.where(is_i7, inp_score[None, :].to(tl.float32), buf_score)

        # seq_len == 4 special case: zero overlap kv, -inf overlap score.
        sl4_active = ((seq_len == 4) & (offs_s < 4))[:, None]
        kv_t = tl.where(sl4_active, tl.zeros([SEQ, BLOCK_HD], tl.float32), kv_t)
        score_t = tl.where(
            sl4_active, tl.full([SEQ, BLOCK_HD], -1e9, tl.float32), score_t
        )

        bias = tl.load(
            ape_ptr + offs_s[:, None] * HD + offs_hd[None, :],
            mask=hd_mask[None, :],
            other=0.0,
        ).to(tl.float32)

        s = score_t + bias
        m = tl.max(s, axis=0)
        w = tl.exp(s - m[None, :])
        num = tl.sum(kv_t * w, axis=0)
        den = tl.sum(w, axis=0)
        result = num / den

        tl.store(out_ptr + pid_b * HD + offs_hd, result, mask=hd_mask)

    # Always commit the current step (4 chunks) at write_pos.
    write_pos = (seq_len + PAGE_SIZE - 1) % PAGE_SIZE
    write_base = buf_ptr + index * STRIDE_IDX + write_pos * STRIDE_POS
    tl.store(write_base + 0 * HD + offs_hd, inp_kv_ovr, mask=hd_mask)
    tl.store(write_base + 1 * HD + offs_hd, inp_kv, mask=hd_mask)
    tl.store(write_base + 2 * HD + offs_hd, inp_score_ovr, mask=hd_mask)
    tl.store(write_base + 3 * HD + offs_hd, inp_score, mask=hd_mask)


# --- c4 prefill ------------------------------------------------------------


@triton.jit
def _triton_c4_prefill_compress_kernel(
    buf_ptr,
    inp_ptr,
    out_ptr,
    ape_ptr,
    plan_ptr,
    indices_ptr,
    extra_ptr,            # [B, 4] int32 for paged; ignored for ring
    HD: tl.constexpr,
    BLOCK_HD: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
):
    SEQ: tl.constexpr = 8
    NUM_CHUNKS: tl.constexpr = 4
    PAGED: tl.constexpr = (PAGE_SIZE == 4)
    STRIDE_IDX: tl.constexpr = PAGE_SIZE * NUM_CHUNKS * HD
    STRIDE_POS: tl.constexpr = NUM_CHUNKS * HD

    pid_p = tl.program_id(0)
    pid_s = tl.program_id(1)

    plan_base = plan_ptr + pid_p * 4
    ragged_id = tl.load(plan_base + 0)
    batch_id = tl.load(plan_base + 1)
    position = tl.load(plan_base + 2)
    window_len = tl.load(plan_base + 3)

    if ragged_id == -1:
        return

    seq_len = position + 1

    offs_hd = pid_s * BLOCK_HD + tl.arange(0, BLOCK_HD)
    hd_mask = offs_hd < HD

    offs_s = tl.arange(0, SEQ)
    use_buf = (offs_s < window_len)[:, None]
    chunk_kv_off = tl.where(offs_s < 4, 0, HD).to(tl.int64)
    chunk_score_off = chunk_kv_off + 2 * HD

    # ---- Buffer offsets ----
    if PAGED:
        ex_base = extra_ptr + batch_id.to(tl.int64) * 4
        load_first_page = tl.load(ex_base + 0).to(tl.int64)
        load_second_page = tl.load(ex_base + 1).to(tl.int64)
        # When window_len <= 4 the overlap half is unused, so reuse load_second
        # to keep loads in-bounds (matches the CUDA kernel's selection).
        page_idx_overlap = tl.where(window_len <= 4, load_second_page, load_first_page)
        page_idx = tl.where(offs_s < 4, page_idx_overlap, load_second_page)
        slot_pos = (offs_s % 4).to(tl.int64)
        slot_base_2d = page_idx[:, None] * STRIDE_IDX + slot_pos[:, None] * STRIDE_POS
    else:
        page_idx_scalar = tl.load(indices_ptr + batch_id.to(tl.int64)).to(tl.int64)
        slot_pos = (seq_len.to(tl.int64) + offs_s.to(tl.int64)) % 8
        slot_base_2d = page_idx_scalar * STRIDE_IDX + slot_pos[:, None] * STRIDE_POS

    kv_off_2d = slot_base_2d + chunk_kv_off[:, None] + offs_hd[None, :].to(tl.int64)
    score_off_2d = slot_base_2d + chunk_score_off[:, None] + offs_hd[None, :].to(tl.int64)
    buf_load_mask = use_buf & hd_mask[None, :]
    buf_kv = tl.load(buf_ptr + kv_off_2d, mask=buf_load_mask, other=0.0).to(tl.float32)
    buf_score = tl.load(buf_ptr + score_off_2d, mask=buf_load_mask, other=0.0).to(tl.float32)

    # ---- Ragged tail offsets: rag_off = max(ragged_id + i - 7, 0) ----
    rag_off = tl.maximum(ragged_id.to(tl.int64) + offs_s.to(tl.int64) - 7, 0)
    rag_base_2d = rag_off[:, None] * (NUM_CHUNKS * HD)
    rag_kv_off_2d = rag_base_2d + chunk_kv_off[:, None] + offs_hd[None, :].to(tl.int64)
    rag_score_off_2d = rag_base_2d + chunk_score_off[:, None] + offs_hd[None, :].to(tl.int64)
    rag_load_mask = (~use_buf) & hd_mask[None, :]
    rag_kv = tl.load(inp_ptr + rag_kv_off_2d, mask=rag_load_mask, other=0.0).to(tl.float32)
    rag_score = tl.load(inp_ptr + rag_score_off_2d, mask=rag_load_mask, other=0.0).to(tl.float32)

    kv_t = tl.where(use_buf, buf_kv, rag_kv)
    score_t = tl.where(use_buf, buf_score, rag_score)

    sl4_active = ((seq_len == 4) & (offs_s < 4))[:, None]
    kv_t = tl.where(sl4_active, tl.zeros([SEQ, BLOCK_HD], tl.float32), kv_t)
    score_t = tl.where(
        sl4_active, tl.full([SEQ, BLOCK_HD], -1e9, tl.float32), score_t
    )

    bias = tl.load(
        ape_ptr + offs_s[:, None] * HD + offs_hd[None, :],
        mask=hd_mask[None, :],
        other=0.0,
    ).to(tl.float32)

    s = score_t + bias
    m = tl.max(s, axis=0)
    w = tl.exp(s - m[None, :])
    num = tl.sum(kv_t * w, axis=0)
    den = tl.sum(w, axis=0)
    result = num / den

    tl.store(out_ptr + ragged_id.to(tl.int64) * HD + offs_hd, result, mask=hd_mask)


@triton.jit
def _triton_c4_prefill_write_kernel(
    buf_ptr,
    inp_ptr,
    plan_ptr,
    indices_ptr,
    extra_ptr,
    HD: tl.constexpr,
    BLOCK_HD: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
):
    NUM_CHUNKS: tl.constexpr = 4
    PAGED: tl.constexpr = (PAGE_SIZE == 4)
    STRIDE_IDX: tl.constexpr = PAGE_SIZE * NUM_CHUNKS * HD
    STRIDE_POS: tl.constexpr = NUM_CHUNKS * HD

    pid_p = tl.program_id(0)
    pid_s = tl.program_id(1)

    plan_base = plan_ptr + pid_p * 4
    ragged_id = tl.load(plan_base + 0)
    batch_id = tl.load(plan_base + 1)
    position = tl.load(plan_base + 2)

    if ragged_id == -1:
        return

    offs_hd = pid_s * BLOCK_HD + tl.arange(0, BLOCK_HD)
    hd_mask = offs_hd < HD

    if PAGED:
        ex_base = extra_ptr + batch_id.to(tl.int64) * 4
        write_first_page = tl.load(ex_base + 2).to(tl.int64)
        last_pos = tl.load(ex_base + 3)
        write_second_page = tl.load(indices_ptr + batch_id.to(tl.int64)).to(tl.int64)
        tgt_index = tl.where(position < last_pos, write_first_page, write_second_page)
        tgt_pos = (position % 4).to(tl.int64)
    else:
        tgt_index = tl.load(indices_ptr + batch_id.to(tl.int64)).to(tl.int64)
        tgt_pos = (position % 8).to(tl.int64)

    src_base = inp_ptr + ragged_id.to(tl.int64) * (NUM_CHUNKS * HD)
    tgt_base = buf_ptr + tgt_index * STRIDE_IDX + tgt_pos * STRIDE_POS

    for c in tl.static_range(NUM_CHUNKS):
        v = tl.load(src_base + c * HD + offs_hd, mask=hd_mask)
        tl.store(tgt_base + c * HD + offs_hd, v, mask=hd_mask)


# --- Triton wrapper functions ----------------------------------------------


def _triton_c128_decode(
    kv_score_buffer: torch.Tensor,
    kv_score_input: torch.Tensor,
    out: torch.Tensor,
    ape: torch.Tensor,
    indices: torch.Tensor,
    seq_lens: torch.Tensor,
    *,
    head_dim: int,
) -> None:
    B = indices.shape[0]
    if B == 0:
        return
    BLOCK_HD = _pick_block_hd(head_dim)
    grid = (B, triton.cdiv(head_dim, BLOCK_HD))
    _triton_c128_decode_kernel[grid](
        kv_score_buffer,
        kv_score_input,
        out,
        ape,
        indices,
        seq_lens,
        HD=head_dim,
        BLOCK_HD=BLOCK_HD,
    )


def _triton_c128_prefill(
    kv_score_buffer: torch.Tensor,
    kv_score_input: torch.Tensor,
    out: torch.Tensor,
    ape: torch.Tensor,
    indices: torch.Tensor,
    compress_plan_bytes: torch.Tensor,
    write_plan_bytes: torch.Tensor,
    extra: Optional[torch.Tensor],
    *,
    head_dim: int,
) -> None:
    cplan = compress_plan_bytes.view(torch.int32).reshape(-1, 4)
    wplan = write_plan_bytes.view(torch.int32).reshape(-1, 4)
    load_indices = extra if extra is not None else indices
    BLOCK_HD = _pick_block_hd(head_dim)
    if cplan.shape[0] > 0:
        grid = (cplan.shape[0], triton.cdiv(head_dim, BLOCK_HD))
        _triton_c128_prefill_compress_kernel[grid](
            kv_score_buffer,
            kv_score_input,
            out,
            ape,
            cplan,
            load_indices,
            HD=head_dim,
            BLOCK_HD=BLOCK_HD,
        )
    if wplan.shape[0] > 0:
        grid = (wplan.shape[0], triton.cdiv(head_dim, BLOCK_HD))
        _triton_c128_prefill_write_kernel[grid](
            kv_score_buffer,
            kv_score_input,
            wplan,
            indices,
            HD=head_dim,
            BLOCK_HD=BLOCK_HD,
        )


def _triton_c4_decode(
    kv_score_buffer: torch.Tensor,
    kv_score_input: torch.Tensor,
    out: torch.Tensor,
    ape: torch.Tensor,
    indices: torch.Tensor,
    seq_lens: torch.Tensor,
    extra: Optional[torch.Tensor],
    *,
    head_dim: int,
) -> None:
    B = indices.shape[0]
    if B == 0:
        return
    page_size = kv_score_buffer.shape[1]
    assert page_size in (4, 8)
    BLOCK_HD = _pick_block_hd(head_dim)
    grid = (B, triton.cdiv(head_dim, BLOCK_HD))
    extra_arg = extra if extra is not None else indices  # dummy ptr for ring
    _triton_c4_decode_kernel[grid](
        kv_score_buffer,
        kv_score_input,
        out,
        ape,
        indices,
        seq_lens,
        extra_arg,
        HD=head_dim,
        BLOCK_HD=BLOCK_HD,
        PAGE_SIZE=page_size,
    )


def _triton_c4_prefill(
    kv_score_buffer: torch.Tensor,
    kv_score_input: torch.Tensor,
    out: torch.Tensor,
    ape: torch.Tensor,
    indices: torch.Tensor,
    compress_plan_bytes: torch.Tensor,
    write_plan_bytes: torch.Tensor,
    extra: Optional[torch.Tensor],
    *,
    head_dim: int,
) -> None:
    page_size = kv_score_buffer.shape[1]
    assert page_size in (4, 8)
    BLOCK_HD = _pick_block_hd(head_dim)
    cplan = compress_plan_bytes.view(torch.int32).reshape(-1, 4)
    wplan = write_plan_bytes.view(torch.int32).reshape(-1, 4)
    extra_arg = extra if extra is not None else indices  # dummy ptr for ring
    if cplan.shape[0] > 0:
        grid = (cplan.shape[0], triton.cdiv(head_dim, BLOCK_HD))
        _triton_c4_prefill_compress_kernel[grid](
            kv_score_buffer,
            kv_score_input,
            out,
            ape,
            cplan,
            indices,
            extra_arg,
            HD=head_dim,
            BLOCK_HD=BLOCK_HD,
            PAGE_SIZE=page_size,
        )
    if wplan.shape[0] > 0:
        grid = (wplan.shape[0], triton.cdiv(head_dim, BLOCK_HD))
        _triton_c4_prefill_write_kernel[grid](
            kv_score_buffer,
            kv_score_input,
            wplan,
            indices,
            extra_arg,
            HD=head_dim,
            BLOCK_HD=BLOCK_HD,
            PAGE_SIZE=page_size,
        )


def _triton_compress_forward(
    kv_score_buffer: torch.Tensor,
    kv_score_input: torch.Tensor,
    out: torch.Tensor,
    ape: torch.Tensor,
    indices: torch.Tensor,
    plan: Union[CompressorDecodePlan, CompressorPrefillPlan],
    extra_data: Optional[torch.Tensor],
    *,
    head_dim: int,
    compress_ratio: Literal[4, 128],
) -> None:
    if compress_ratio == 4:
        if plan.is_decode:
            _triton_c4_decode(
                kv_score_buffer,
                kv_score_input,
                out,
                ape,
                indices,
                plan.seq_lens,
                extra_data,
                head_dim=head_dim,
            )
        else:
            _triton_c4_prefill(
                kv_score_buffer,
                kv_score_input,
                out,
                ape,
                indices,
                plan.compress_plan,
                plan.write_plan,
                extra_data,
                head_dim=head_dim,
            )
    else:
        assert compress_ratio == 128
        if plan.is_decode:
            _triton_c128_decode(
                kv_score_buffer,
                kv_score_input,
                out,
                ape,
                indices,
                plan.seq_lens,
                head_dim=head_dim,
            )
        else:
            _triton_c128_prefill(
                kv_score_buffer,
                kv_score_input,
                out,
                ape,
                indices,
                plan.compress_plan,
                plan.write_plan,
                extra_data,
                head_dim=head_dim,
            )


class CompressorDecodePlan(NamedTuple):
    compress_ratio: int
    seq_lens: torch.Tensor

    def copy_(self, other: CompressorDecodePlan) -> None:
        assert self.compress_ratio == other.compress_ratio
        self.seq_lens.copy_(other.seq_lens)

    @property
    def is_decode(self) -> bool:
        return True


def compress_plan(
    compress_ratio: Literal[4, 128],
    num_q_tokens: int,
    seq_lens: torch.Tensor,
    extend_lens: Optional[torch.Tensor],
    device: torch.device,
) -> Union[CompressorDecodePlan, CompressorPrefillPlan]:
    if extend_lens is not None:
        return CompressorPrefillPlan.generate(
            compress_ratio,
            num_q_tokens,
            seq_lens,
            extend_lens,
            device,
        )
    else:
        assert num_q_tokens == len(seq_lens)
        seq_lens = seq_lens.to(device, non_blocking=True)
        return CompressorDecodePlan(compress_ratio, seq_lens)


def compress_forward(
    kv_score_buffer: torch.Tensor,
    kv_score_input: torch.Tensor,
    ape: torch.Tensor,
    indices: torch.Tensor,
    plan: Union[CompressorDecodePlan, CompressorPrefillPlan, None] = None,
    extra_data: Optional[torch.Tensor] = None,
    *,
    head_dim: int,
    compress_ratio: Literal[4, 128],
    out: Optional[torch.Tensor] = None,
    seq_lens: Optional[torch.Tensor] = None,
    extend_lens: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    assert head_dim % 128 == 0
    num_q_tokens = kv_score_input.shape[0]
    if out is None:
        out = kv_score_input.new_empty((num_q_tokens, head_dim))
    if plan is None:
        assert seq_lens is not None
        plan = compress_plan(
            compress_ratio,
            num_q_tokens,
            seq_lens,
            extend_lens,
            kv_score_input.device,
        )
    assert plan.compress_ratio == compress_ratio, "Mismatched compress ratio in plan!"
    # Online c128: separate JIT module, fp32 state, no compile-time dtypes.
    if compress_ratio == 128 and envs.SGLANG_OPT_USE_ONLINE_COMPRESS.get():
        online_module = _jit_compress_128_online_module(head_dim=head_dim)
        F = online_module.decode if plan.is_decode else online_module.prefill
        F(kv_score_buffer, kv_score_input, out, ape, indices, *plan[1:], extra_data)
        return out
    if _is_xpu:
        # Triton fallback for non-CUDA backends. Mirrors
        # FlashCompress{4,128}Kernel in jit_kernel/csrc/deepseek_v4/c{4,128}.cuh.
        _triton_compress_forward(
            kv_score_buffer,
            kv_score_input,
            out,
            ape,
            indices,
            plan,
            extra_data,
            head_dim=head_dim,
            compress_ratio=compress_ratio,
        )
        return out
    module = _jit_compress_module(
        head_dim,
        kv_score_input.dtype,
        out.dtype,
        compress_ratio,
    )
    if plan.is_decode:
        F = module.decode
    elif compress_ratio == 128 and _should_use_c128_prefill_defensive():
        F = _jit_compress_module_v2_defensive(
            head_dim,
            kv_score_input.dtype,
            out.dtype,
        ).prefill
    else:
        F = module.prefill
    F(kv_score_buffer, kv_score_input, out, ape, indices, *plan[1:], extra_data)
    return out


def _should_use_c128_prefill_defensive() -> bool:
    from sglang.srt.environ import envs

    return envs.SGLANG_HANDLE_C128_PREFILL_KERNEL.get()


def compress_fused_norm_rope_inplace(
    kv: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    freq_cis: torch.Tensor,
    plan: Union[CompressorDecodePlan, CompressorPrefillPlan],
) -> None:
    freq_cis = torch.view_as_real(freq_cis).flatten(-2)
    mode = 1 if plan.is_decode else 0
    if _is_xpu:
        _torch_fused_norm_rope(
            kv,
            weight,
            plan[1],
            freq_cis,
            mode,
            eps,
            plan.compress_ratio,
        )
        return
    module = _jit_norm_rope_module(kv.dtype, kv.shape[-1], freq_cis.shape[-1])
    module.forward(
        kv,
        weight,
        plan[1],
        freq_cis,
        mode,
        eps,
        plan.compress_ratio,
    )


def fused_norm_rope_inplace(
    kv: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    freq_cis: torch.Tensor,
    positions: torch.Tensor,
) -> None:
    freq_cis = torch.view_as_real(freq_cis).flatten(-2)
    if _is_xpu:
        _torch_fused_norm_rope(
            kv,
            weight,
            positions,
            freq_cis,
            2,
            eps,
            0,
        )
        return
    module = _jit_norm_rope_module(kv.dtype, kv.shape[-1], freq_cis.shape[-1])
    module.forward(
        kv,
        weight,
        positions,
        freq_cis,
        2,
        eps,
        0,
    )


def _torch_fused_norm_rope(
    input_tensor: torch.Tensor,
    weight: torch.Tensor,
    handle: torch.Tensor,
    freqs_cis: torch.Tensor,
    mode: int,
    eps: float,
    compress_ratio: int,
) -> None:
    """Pure-torch fallback for ``FusedNormRopeKernel::forward``.

    Mirrors ``fused_norm_rope`` in
    ``jit_kernel/csrc/deepseek_v4/fused_norm_rope.cuh``: per-row RMSNorm
    in-place, then RoPE on the trailing ``rope_dim`` of each selected row.

    ``mode``:
      0 = CompressExtend  (handle = packed PrefillPlan, [N, 16] uint8)
      1 = CompressDecode  (handle = seq_lens, [N])
      2 = DefaultForward  (handle = positions, [N])
    """
    head_dim = input_tensor.shape[-1]
    rope_dim = freqs_cis.shape[-1]
    device = input_tensor.device
    in_dtype = input_tensor.dtype

    if mode == 0:
        # Prefill plan is already pre-sliced to its exact valid length on
        # the host (no cuda-graph padding on the prefill path), so all rows
        # are valid. Avoid the per-layer ``bool((plan != INVALID).any())``
        # / boolean-mask gather host-syncs that drain the L0 queue.
        plan = _decode_prefill_plan(handle)
        if plan.shape[0] == 0:
            return
        rows = plan[:, 0]
        positions = plan[:, 2] + 1 - compress_ratio
        row_mask = None  # writeback unconditional via index
    elif mode == 1:
        # Decode: avoid host syncs (``bool(valid.any())`` /
        # ``valid.nonzero()``) that drain the L0 queue every layer.
        # Compute over all rows and mask the writeback with torch.where.
        seq_lens = handle.to(torch.int64)
        valid = (seq_lens % compress_ratio) == 0  # [B] bool
        num_works = seq_lens.shape[0]
        rows = torch.arange(num_works, device=device, dtype=torch.int64)
        # For invalid rows, ``positions`` would be negative; clamp to 0
        # so freqs_cis[positions] is always in-bounds. Result is masked
        # out below before writeback.
        positions = (seq_lens - compress_ratio).clamp_min(0)
        row_mask = valid
    elif mode == 2:
        num_works = handle.shape[0]
        positions = handle.to(torch.int64)
        rows = torch.arange(num_works, device=device, dtype=torch.int64)
        row_mask = None
    else:
        raise ValueError(f"unsupported fused_norm_rope mode: {mode}")

    # RMSNorm in-place on selected rows.
    x = input_tensor[rows].float()  # [N, head_dim]
    var = (x * x).mean(dim=-1, keepdim=True)
    x = x * torch.rsqrt(var + eps) * weight.float()

    # RoPE on the trailing rope_dim, viewed as (real, imag) interleaved pairs.
    n = x.shape[0]
    rope = x[..., -rope_dim:].view(n, rope_dim // 2, 2)
    freq = freqs_cis.float()[positions].view(n, rope_dim // 2, 2)
    xr, xi = rope[..., 0], rope[..., 1]
    fr, fi = freq[..., 0], freq[..., 1]
    out_r = xr * fr - xi * fi
    out_i = xr * fi + xi * fr
    rope_out = torch.stack([out_r, out_i], dim=-1).reshape(n, rope_dim)
    x[..., -rope_dim:] = rope_out
    if row_mask is None:
        input_tensor[rows] = x.to(in_dtype)
    else:
        # Only update rows where ``valid`` is true; preserve others.
        mask = row_mask.view(-1, *([1] * (x.ndim - 1)))
        input_tensor[rows] = torch.where(
            mask, x.to(in_dtype), input_tensor[rows]
        )


def fused_rope(
    q: torch.Tensor,
    k: Optional[torch.Tensor],
    freqs_cis: torch.Tensor,
    positions: torch.Tensor,
    inverse: bool = False,
) -> None:
    if _is_cuda:
        freqs_real = torch.view_as_real(freqs_cis).flatten(-2).contiguous()
        module = _jit_fused_rope_module()
        module.forward(q, k, freqs_real, positions, inverse)
    else:
        # Triton fallback for non-CUDA backends (e.g. XPU). Mirrors
        # FusedQKRopeKernel: apply rotary embedding in-place to q and
        # (when provided) k.
        from sglang.srt.layers.deepseek_v4_rope import apply_rotary_emb_triton

        apply_rotary_emb_triton(q, freqs_cis, positions=positions, inverse=inverse)
        if k is not None:
            apply_rotary_emb_triton(
                k, freqs_cis, positions=positions, inverse=inverse
            )


@cache_once
def _tilelang_make_swa_indices_kernel(swa_window_size: int, threads: int = 128) -> Any:
    import tilelang
    import tilelang.language as T

    batch_size = T.dynamic("batch_size")
    batch_size_plus_1 = T.dynamic("batch_size_plus_1")
    num_q_tokens = T.dynamic("num_q_tokens")
    num_warps = threads // 32
    assert swa_window_size % 32 == 0

    @tilelang.jit
    def make_swa_prefill_indices(
        seq_lens_k: T.Tensor[(batch_size,), T.int32],
        seq_lens_q: T.Tensor[(batch_size,), T.int32],
        cu_seqlens_q: T.Tensor[(batch_size_plus_1,), T.int32],
        swa_indices: T.Tensor[(num_q_tokens, swa_window_size), T.int32],
    ):
        _ = batch_size_plus_1
        with T.Kernel(T.ceildiv(num_q_tokens, num_warps), threads=threads) as bx:
            tx = T.get_thread_binding()
            warp_id = tx // 32
            lane_id = tx % 32
            s_batch_id = T.alloc_shared((num_warps,), dtype=T.int32)

            token_id = warp_id + bx * num_warps
            if token_id >= num_q_tokens:
                return
            for i in T.serial(0, batch_size, step=32):
                j = i + lane_id
                if cu_seqlens_q[j] <= token_id < cu_seqlens_q[j + 1]:
                    s_batch_id[warp_id] = j
            T.sync_warp()

            seq_idx = s_batch_id[warp_id]
            kv_len = seq_lens_k[seq_idx]
            qo_len = seq_lens_q[seq_idx]
            cum_qo_len = cu_seqlens_q[seq_idx]
            prefix_len = kv_len - qo_len
            curr_seq_qo_idx = token_id - cum_qo_len
            end_abs_pos = prefix_len + curr_seq_qo_idx + 1
            start_abs_pos = T.max(end_abs_pos - swa_window_size, 0)
            old_kv_start = seq_idx * swa_window_size
            new_kv_start = batch_size * swa_window_size + cum_qo_len

            for i in T.unroll(0, swa_window_size, step=32):
                j = i + lane_id
                abs_pos = start_abs_pos + j
                swa_indices[token_id, j] = T.if_then_else(
                    abs_pos < end_abs_pos,
                    T.if_then_else(
                        abs_pos < prefix_len,
                        old_kv_start + abs_pos % swa_window_size,
                        new_kv_start + (abs_pos - prefix_len),
                    ),
                    -1,
                )

    return make_swa_prefill_indices


def tilelang_make_swa_prefill_indices(
    seq_lens_k: torch.Tensor,
    seq_lens_q: torch.Tensor,
    swa_indices: torch.Tensor,
    cu_seqlens_q: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if cu_seqlens_q is None:
        cu_seqlens_q = torch.cumsum(seq_lens_q, dim=0, dtype=torch.int32)
        cu_seqlens_q = torch.nn.functional.pad(cu_seqlens_q, (1, 0), value=0)
    swa_window_size = swa_indices.shape[1]
    kernel = _tilelang_make_swa_indices_kernel(swa_window_size)
    kernel(seq_lens_k, seq_lens_q, cu_seqlens_q, swa_indices)
    return swa_indices


@triton.jit
def create_paged_compress_data_kernel(
    req_pool_indices_ptr,
    seq_lens_ptr,
    extend_seq_lens_ptr,
    req_to_token_ptr,
    full_to_swa_index_mapping_ptr,
    out_0_ptr,
    out_1_ptr,
    batch_size,
    stride_req_to_token_0,
    stride_req_to_token_1: tl.constexpr,
    stride_out_1_0,
    stride_out_1_1: tl.constexpr,
    compress_ratio: tl.constexpr,
    is_overlap: tl.constexpr,
    swa_page_size: tl.constexpr,
    ring_size: tl.constexpr,
    BLOCK: tl.constexpr,
) -> None:
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < batch_size

    rid = tl.load(req_pool_indices_ptr + offs, mask=mask, other=0).to(tl.int32)
    seq_len = tl.load(seq_lens_ptr + offs, mask=mask, other=0).to(tl.int32)
    extend_len = tl.load(extend_seq_lens_ptr + offs, mask=mask, other=0).to(tl.int32)
    prefix_len = seq_len - extend_len

    cr = compress_ratio
    write_pos = ((seq_len - 1) // cr) * cr
    load_pos = ((prefix_len - 1) // cr) * cr
    write_overlap_pos = write_pos - cr
    load_overlap_pos = load_pos - cr
    v0 = tl.zeros([BLOCK], tl.int32)
    v1 = tl.zeros([BLOCK], tl.int32)
    v2 = tl.zeros([BLOCK], tl.int32)
    v3 = tl.zeros([BLOCK], tl.int32)

    for i in tl.static_range(4):
        if i == 0:
            pos = load_pos
        elif i == 1:
            pos = write_pos
        elif i == 2:
            pos = load_overlap_pos
        else:
            pos = write_overlap_pos
        pos = tl.maximum(pos, 0)
        loc = tl.load(
            req_to_token_ptr
            + rid.to(tl.int64) * stride_req_to_token_0
            + pos.to(tl.int64) * stride_req_to_token_1,
            mask=mask,
            other=0,
        ).to(tl.int32)
        swa_loc = tl.load(full_to_swa_index_mapping_ptr + loc, mask=mask, other=0).to(
            tl.int32
        )
        swa_page = swa_loc // swa_page_size
        state_loc = swa_page * ring_size + (swa_loc % ring_size)
        state_loc = state_loc // cr
        if i == 0:
            v0 = state_loc
        elif i == 1:
            v1 = state_loc
        elif i == 2:
            v2 = state_loc
        else:
            v3 = state_loc

    tl.store(out_0_ptr + offs, v1, mask=mask)

    if is_overlap:
        base = out_1_ptr + offs * stride_out_1_0
        tl.store(base + 0 * stride_out_1_1, v2, mask=mask)
        tl.store(base + 1 * stride_out_1_1, v0, mask=mask)
        tl.store(base + 2 * stride_out_1_1, v3, mask=mask)
        tl.store(base + 3 * stride_out_1_1, write_pos.to(tl.int32), mask=mask)
    else:
        base = out_1_ptr + offs * stride_out_1_0
        tl.store(base + 0 * stride_out_1_1, v0, mask=mask)


_mmap_dumper = None


def _get_mmap_dumper():
    global _mmap_dumper
    if _mmap_dumper is None:
        from sglang.srt.debug_utils.mmap_dumper import MmapDumper
        from sglang.srt.environ import envs

        dump_dir = envs.SGLANG_HACK_DEBUG_DUMP_CREATE_PAGED_COMPRESS_DATA.get()
        _mmap_dumper = MmapDumper(dump_dir or None)
    return _mmap_dumper


_dumped_static_meta_once = False


def _maybe_dump_create_paged_compress_data_inputs(
    *,
    compress_ratio: int,
    is_overlap: bool,
    swa_page_size: int,
    ring_size: int,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    extend_seq_lens: torch.Tensor,
    req_to_token: torch.Tensor,
    full_to_swa_index_mapping: torch.Tensor,
    block: int,
) -> None:
    d = _get_mmap_dumper()
    if not d.is_active():
        return

    # Print static config (constant after server init) once per process.
    global _dumped_static_meta_once
    if not _dumped_static_meta_once:
        print(
            f"[c128_dump_static] swa_page_size={swa_page_size} ring_size={ring_size} "
            f"block={block} req_to_token_shape={tuple(req_to_token.shape)} "
            f"full_to_swa_shape={tuple(full_to_swa_index_mapping.shape)}",
            flush=True,
        )
        _dumped_static_meta_once = True

    # Per-ratio dump (small): req_pool_indices / seq_lens / extend_seq_lens.
    # These are forward_batch fields, identical across c4 and c128 within the
    # same forward — but small (KB-level) so dumping twice is cheap.
    p = f"c{compress_ratio}_plan"
    d.dump(
        {
            f"{p}_compress_ratio": compress_ratio,
            f"{p}_is_overlap": is_overlap,
            f"{p}_req_pool_indices": req_pool_indices,
            f"{p}_seq_lens": seq_lens,
            f"{p}_extend_seq_lens": extend_seq_lens,
        }
    )

    # Global tensors shared between c4 and c128 (multi-MB-GB). Only dump on
    # the first call per forward to avoid 2x GPU->CPU copy (~184 MB + ~33 MB).
    # Backends call create_paged_compressor_data with c4 first, then c128
    # (deepseek_v4_backend_radix.py:457-458), so dump on c4 only.
    if compress_ratio == 4:
        cols = min(10000, req_to_token.shape[1])
        req_to_token_partial = req_to_token[:, :cols].contiguous()
        d.dump(
            {
                "global_req_to_token_dumped_cols": cols,
                "global_req_to_token_partial": req_to_token_partial,
                "global_full_to_swa_index_mapping": full_to_swa_index_mapping,
            }
        )


def _maybe_dump_create_paged_compress_data_outputs(
    *,
    compress_ratio: int,
    out_0: torch.Tensor,
    out_1: torch.Tensor,
) -> None:
    d = _get_mmap_dumper()
    if not d.is_active():
        return
    p = f"c{compress_ratio}_plan"
    d.dump(
        {
            f"{p}_out_0": out_0,
            f"{p}_out_1": out_1,
            f"{p}_out_0_shape": list(out_0.shape),
            f"{p}_out_1_shape": list(out_1.shape),
        }
    )


_printed_buffer_shape_once: dict = {}


def maybe_dump_compress_metadata_extras(
    *,
    compress_ratio: int,
    kv_score_buffer_shape: Tuple[int, ...],
    kv_score_buffer_dtype: torch.dtype,
    plan_compress_plan: torch.Tensor,
    plan_write_plan: torch.Tensor,
) -> None:
    """Public helper to be called from compressor.py at metadata-prepare time
    (once per forward per ratio, not per layer). Dumps the prefill kernel's
    real bound (kv_score_buffer.shape) plus the actual plan tensors that get
    fed to flash_c{ratio}_prefill.
    """
    d = _get_mmap_dumper()
    if not d.is_active():
        return

    # Print kv_score_buffer.shape once per ratio (constant after init).
    if compress_ratio not in _printed_buffer_shape_once:
        print(
            f"[c128_dump_static] c{compress_ratio} "
            f"kv_score_buffer_shape={tuple(kv_score_buffer_shape)} "
            f"dtype={kv_score_buffer_dtype}",
            flush=True,
        )
        _printed_buffer_shape_once[compress_ratio] = True

    p = f"c{compress_ratio}_meta"
    d.dump(
        {
            f"{p}_plan_compress_plan": plan_compress_plan,
            f"{p}_plan_write_plan": plan_write_plan,
            f"{p}_plan_compress_count": int(plan_compress_plan.shape[0]),
            f"{p}_plan_write_count": int(plan_write_plan.shape[0]),
        }
    )


def triton_create_paged_compress_data(
    *,
    compress_ratio: int,
    is_overlap: bool,
    swa_page_size: int,
    ring_size: int,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    extend_seq_lens: torch.Tensor,
    req_to_token: torch.Tensor,
    full_to_swa_index_mapping: torch.Tensor,
    block: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor]:
    _should_dump = bool(envs.SGLANG_HACK_DEBUG_DUMP_CREATE_PAGED_COMPRESS_DATA.get())
    if _should_dump:
        torch.cuda.synchronize()
        _maybe_dump_create_paged_compress_data_inputs(
            compress_ratio=compress_ratio,
            is_overlap=is_overlap,
            swa_page_size=swa_page_size,
            ring_size=ring_size,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            extend_seq_lens=extend_seq_lens,
            req_to_token=req_to_token,
            full_to_swa_index_mapping=full_to_swa_index_mapping,
            block=block,
        )

    batch_size = req_pool_indices.shape[0]
    out_dim = 4 if is_overlap else 1
    device_args: dict = dict(device=req_pool_indices.device, dtype=torch.int32)
    out_0 = torch.empty((batch_size,), **device_args)
    out_1 = torch.empty((batch_size, out_dim), **device_args)
    grid = (triton.cdiv(batch_size, block),)
    create_paged_compress_data_kernel[grid](
        req_pool_indices,
        seq_lens,
        extend_seq_lens,
        req_to_token,
        full_to_swa_index_mapping,
        out_0,
        out_1,
        batch_size=batch_size,
        stride_req_to_token_0=req_to_token.stride(0),
        stride_req_to_token_1=req_to_token.stride(1),
        stride_out_1_0=out_1.stride(0),
        stride_out_1_1=out_1.stride(1),
        compress_ratio=compress_ratio,
        is_overlap=1 if is_overlap else 0,
        swa_page_size=swa_page_size,
        ring_size=ring_size,
        BLOCK=block,
    )

    if _should_dump:
        torch.cuda.synchronize()
        _maybe_dump_create_paged_compress_data_outputs(
            compress_ratio=compress_ratio, out_0=out_0, out_1=out_1
        )

    if not is_overlap:
        out_1.squeeze_(1)
    return out_0, out_1


def fused_store_cache(
    input: torch.Tensor,
    cache: torch.Tensor,
    indices: torch.Tensor,
    *,
    page_size: int,
    type: Literal["flashmla", "indexer"],
) -> None:
    module = _jit_fused_store_module(
        name=type,
        input_dtype=input.dtype,
        index_dtype=indices.dtype,
        page_size=page_size,
    )
    module.run(input, cache, indices)


def silu_and_mul_clamp(
    input: torch.Tensor,
    output: torch.Tensor,
    swiglu_limit: float,
) -> None:
    # Fallback path is hacky on purpose: when the mega-moe-memory flag is off
    # we must be bitwise-identical to the optimize branch, which used the
    # pre-refactor kernel.
    from sglang.srt.environ import envs

    deepseek_v4_moe_code_path_checker.observed += 1
    if envs.SGLANG_OPT_FIX_MEGA_MOE_MEMORY.get():
        module = _jit_silu_and_mul_clamp_module(input.dtype)
    else:
        module = _jit_silu_and_mul_clamp_tmp_module(input.dtype)
    module.run(input, output, float(swiglu_limit))


def silu_and_mul_masked_post_quant(
    input: torch.Tensor,
    output: torch.Tensor,
    output_scale: torch.Tensor,
    quant_group_size: int,
    masked_m: torch.Tensor,
    scale_ue8m0: bool = False,
    topk: int = 8,
    transposed: bool = False,
    swiglu_limit: Optional[float] = None,
    swizzle: bool = False,
) -> None:
    apply_swiglu_limit = swiglu_limit is not None
    if apply_swiglu_limit:
        deepseek_v4_moe_code_path_checker.observed += 1
    if swizzle:
        module = _jit_silu_mul_quant_varlen_module(
            quant_group_size, scale_ue8m0, swizzle, apply_swiglu_limit
        )
    else:
        module = _jit_silu_mul_quant_tmp_module(
            quant_group_size, scale_ue8m0, apply_swiglu_limit
        )
    module.run(
        input,
        output,
        output_scale,
        masked_m,
        topk,
        transposed,
        float(swiglu_limit) if apply_swiglu_limit else 0.0,
    )


def silu_and_mul_contig_post_quant(
    input: torch.Tensor,
    output: torch.Tensor,
    output_scale: torch.Tensor,
    quant_group_size: int,
    scale_ue8m0: bool = False,
    transposed: bool = False,
    swiglu_limit: Optional[float] = None,
    swizzle: bool = False,
) -> None:
    apply_swiglu_limit = swiglu_limit is not None
    if apply_swiglu_limit:
        deepseek_v4_moe_code_path_checker.observed += 1
    module = _jit_silu_mul_quant_contig_module(
        quant_group_size, scale_ue8m0, swizzle, apply_swiglu_limit
    )
    module.run(
        input,
        output,
        output_scale,
        transposed,
        float(swiglu_limit) if apply_swiglu_limit else 0.0,
    )


def mega_moe_pre_dispatch(
    x: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    buf_x: torch.Tensor,
    buf_x_sf: torch.Tensor,
    buf_topk_idx: torch.Tensor,
    buf_topk_weights: torch.Tensor,
    quant_group_size: int = 32,
) -> None:
    module = _jit_mega_moe_pre_dispatch_module(quant_group_size)
    module.run(
        x,
        topk_idx,
        topk_weights,
        buf_x,
        buf_x_sf,
        buf_topk_idx,
        buf_topk_weights,
    )


def get_paged_mqa_logits_metadata(seq_lens: torch.Tensor, page_size: int, num_sm: int):
    assert page_size == 64
    seq_lens = seq_lens.view(-1).to(torch.int32)
    metadata = seq_lens.new_empty(num_sm + 1, 2)
    module = _jit_metadata_module()
    module.run(seq_lens, metadata)
    return metadata


def rmsnorm_self(q: torch.Tensor, eps: float) -> torch.Tensor:
    module = _jit_rmsnorm_head_module(q.shape[-1], q.dtype)
    out = q.new_empty(q.shape)
    module.run_self(q, out, eps)
    return out



def linear_bf16_fp32(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    from sglang.srt.environ import envs

    algo = envs.SGLANG_OPT_BF16_FP32_GEMM_ALGO.get()

    if algo == "auto":
        from sglang.srt.layers.linear_bf16_fp32.selector import pick_backend

        algo = pick_backend(m=x.size(0), n=y.size(0), k=x.size(1))

    return _dispatch_bf16_fp32_backend(x, y, algo=algo)


def _dispatch_bf16_fp32_backend(
    x: torch.Tensor, y: torch.Tensor, *, algo: str
) -> torch.Tensor:
    if algo == "cublas":
        # cuBLAS BF16xBF16 -> FP32 GEMM via PyTorch native API (torch >= 2.9).
        # Bit-exact and matches the previous JIT cublasGemmEx kernel.
        return torch.mm(x, y.t(), out_dtype=torch.float32)
    elif algo == "deep_gemm":
        import deep_gemm

        z = x.new_empty(x.size(0), y.size(0), dtype=torch.float32)
        deep_gemm.bf16_gemm_nt(x, y, z)
        return z
    else:
        return torch.nn.functional.linear(x.float(), y.float())


def _compile_one(*input_tuple) -> None:
    name, job_fn, *args = input_tuple
    print(f"Compiling {name}...", flush=True)
    job_fn(*args)
    print(f"Finished compiling {name}.", flush=True)


def compile_aot():
    c_dtype = torch.float32
    jobs = [
        ("common", _jit_common_module),
        ("mask_topk", _jit_mask_topk_module),
        ("topk", _jit_topk_module),
        ("topk_v2", _jit_topk_v2_module),
        ("hash_topk", _jit_hash_topk_module),
        ("rope", _jit_fused_rope_module),
        ("metadata", _jit_metadata_module),
        (
            "compress_128_4",
            _jit_compress_module,
            128,
            c_dtype,
            c_dtype,
            4,
        ),
        (
            "compress_512_4",
            _jit_compress_module,
            512,
            c_dtype,
            c_dtype,
            4,
        ),
        (
            "compress_512_128",
            _jit_compress_module,
            512,
            c_dtype,
            c_dtype,
            128,
        ),
        (
            "norm_rope_128_64",
            _jit_norm_rope_module,
            c_dtype,
            128,
            64,
        ),
        (
            "norm_rope_512_64",
            _jit_norm_rope_module,
            c_dtype,
            512,
            64,
        ),
        (
            "store_flashmla_bf16_swa_256",
            _jit_fused_store_module,
            "flashmla",
            torch.bfloat16,
            torch.int32,
            256,
        ),
        (
            "store_flashmla_fp32_c4_64",
            _jit_fused_store_module,
            "flashmla",
            torch.float32,
            torch.int32,
            64,
        ),
        (
            "store_flashmla_fp32_c128_2",
            _jit_fused_store_module,
            "flashmla",
            torch.float32,
            torch.int32,
            2,
        ),
        (
            "store_indexer_fp32_c4_64",
            _jit_fused_store_module,
            "indexer",
            torch.float32,
            torch.int32,
            64,
        ),
        (
            "rmsnorm_head_512_bf16",
            _jit_rmsnorm_head_module,
            512,
            torch.bfloat16,
        ),
    ]
    import multiprocessing

    max_parallel_jobs = min(len(jobs), multiprocessing.cpu_count())
    with multiprocessing.Pool(processes=max_parallel_jobs) as pool:
        pool.starmap(_compile_one, jobs)


if __name__ == "__main__":
    compile_aot()
