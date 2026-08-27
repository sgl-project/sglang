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
from sglang.srt.environ import envs

if TYPE_CHECKING:
    from tvm_ffi.module import Module


def make_name(name: str) -> str:
    return f"dpsk_v4_{name}"


@cache_once
def _jit_common_module() -> Module:
    return load_jit(
        make_name("common"),
        cuda_files=["deepseek_v4/common.cuh"],
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
        make_name("fused_norm_rope"),
        *args,
        cuda_files=["deepseek_v4/fused_norm_rope.cuh"],
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
        module = _jit_common_module()
        is_overlap = compress_ratio == 4
        plan_lens = module.plan_compress_prefill(
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
    module = _jit_compress_module(
        head_dim,
        kv_score_input.dtype,
        out.dtype,
        compress_ratio,
    )
    F = module.decode if plan.is_decode else module.prefill
    F(kv_score_buffer, kv_score_input, out, ape, indices, *plan[1:], extra_data)
    return out


def compress_fused_norm_rope_inplace(
    kv: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    freq_cis: torch.Tensor,
    plan: Union[CompressorDecodePlan, CompressorPrefillPlan],
) -> None:
    freq_cis = torch.view_as_real(freq_cis).flatten(-2)
    module = _jit_norm_rope_module(kv.dtype, kv.shape[-1], freq_cis.shape[-1])
    module.forward(
        kv,
        weight,
        plan[1],
        freq_cis,
        int(plan.is_decode),
        eps,
        plan.compress_ratio,
    )


def fused_rope(
    q: torch.Tensor,
    k: Optional[torch.Tensor],
    freqs_cis: torch.Tensor,
    positions: torch.Tensor,
    inverse: bool = False,
) -> None:
    freqs_real = torch.view_as_real(freqs_cis).flatten(-2).contiguous()
    module = _jit_fused_rope_module()
    module.forward(q, k, freqs_real, positions, inverse)


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
    module = _jit_silu_and_mul_clamp_module(input.dtype)
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
    module = _jit_silu_mul_quant_varlen_module(
        quant_group_size, scale_ue8m0, swizzle, apply_swiglu_limit
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


@cache_once
def _jit_torch_cublas_bf16_fp32() -> Any:
    import torch.utils.cpp_extension

    source = """
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cublas_v2.h>

torch::Tensor linear_bf16_fp32(
    torch::Tensor X,
    torch::Tensor W)
{
    int batch = X.size(0);
    int in_features = X.size(1);
    int out_features = W.size(0);

    auto Y = torch::empty(
        {batch, out_features},
        torch::dtype(torch::kFloat32).device(X.device()));

    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();

    float alpha = 1.0f;
    float beta = 0.0f;

    cublasGemmEx(
        handle,
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        out_features,
        batch,
        in_features,
        &alpha,
        W.data_ptr(), CUDA_R_16BF, in_features,
        X.data_ptr(), CUDA_R_16BF, in_features,
        &beta,
        Y.data_ptr(), CUDA_R_32F, out_features,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
    );

    return Y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("linear_bf16_fp32", &linear_bf16_fp32, "BF16xBF16 -> FP32 linear (no bias)");
}
"""
    module = torch.utils.cpp_extension.load_inline(
        name="linear_bf16_fp32",
        cpp_sources="",
        cuda_sources=source,
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3"],
        verbose=False,
    )
    return module


def linear_bf16_fp32(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    from sglang.srt.environ import envs

    algo = envs.SGLANG_OPT_BF16_FP32_GEMM_ALGO.get()
    return _dispatch_bf16_fp32_backend(x, y, algo=algo)


def _dispatch_bf16_fp32_backend(
    x: torch.Tensor, y: torch.Tensor, *, algo: str
) -> torch.Tensor:
    if algo == "cublas":
        module = _jit_torch_cublas_bf16_fp32()
        return module.linear_bf16_fp32(x, y)
    elif algo == "deep_gemm":
        import deep_gemm

        z = x.new_empty(x.size(0), y.size(0), dtype=torch.float32)
        deep_gemm.bf16_gemm_nt(x, y, z)
        return z
    else:
        return torch.nn.functional.linear(x.float(), y.float())
