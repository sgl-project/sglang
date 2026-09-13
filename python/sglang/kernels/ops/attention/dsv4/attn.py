from typing import Literal, Optional, Tuple, Union

import torch
import triton
import triton.language as tl

from sglang.kernels.jit.utils import (
    cache_once,
    is_arch_support_pdl,
    is_hip_runtime,
    load_jit,
    make_cpp_args,
)

from .kv_layout import KVLayout
from .utils import make_name


@cache_once
def _jit_metadata_module():
    return load_jit(
        make_name("metadata"),
        cuda_files=["deepseek_v4/paged_mqa_metadata.cuh"],
        cuda_wrappers=[("run", "IndexerMetadataKernel::run")],
    )


@cache_once
def _jit_fused_store_module(
    name: Literal["flashmla", "indexer"],
    input_dtype: torch.dtype,
    index_dtype: torch.dtype,
    page_size: int,
    layout: KVLayout = KVLayout.V4,
):
    args = make_cpp_args(input_dtype, index_dtype, page_size, is_arch_support_pdl())
    cname = "FlashMLA" if name == "flashmla" else "Indexer"
    wrappers = ["run"]
    if layout is not KVLayout.V4:
        assert name == "flashmla", "only the FlashMLA cache has V4.1 layouts"
        # The layout is a trailing template argument; V4 keeps the default so
        # that its build key does not change.
        args = make_cpp_args(*args, layout.cpp_name)
        wrappers.append("run_rope")
    kernel_class = f"FusedStoreCache{cname}Kernel<{args}>"
    return load_jit(
        make_name("store_" + name),
        *args,
        cuda_files=["deepseek_v4/store.cuh"],
        cuda_wrappers=[(w, f"{kernel_class}::{w}") for w in wrappers],
    )


def get_paged_mqa_logits_metadata(seq_lens: torch.Tensor, page_size: int, num_sm: int):
    # The schedule only depends on the sequence lengths (256-token splits), not
    # on the page size; DeepGEMM takes 32 / 64 / 128 on SM100 and we page at 64
    # or 128 slots.
    assert page_size in (64, 128), page_size
    seq_lens = seq_lens.view(-1).to(torch.int32)
    bs = int(seq_lens.shape[0])
    metadata = seq_lens.new_empty(num_sm + 1, 2)
    # Workspace for the multi-block path; kMBTileSize must match the .cuh.
    if bs > 2048:
        kMBTileSize = 4096
        workspace = seq_lens.new_empty(
            bs + (bs + kMBTileSize - 1) // kMBTileSize, dtype=torch.int32
        )
    else:
        workspace = seq_lens.new_empty(0, dtype=torch.int32)
    module = _jit_metadata_module()
    module.run(seq_lens, metadata, workspace)
    return metadata


def fused_store_cache(
    input: torch.Tensor,
    cache: torch.Tensor,
    indices: torch.Tensor,
    *,
    page_size: int,
    type: Literal["flashmla", "indexer"],
    layout: Union[KVLayout, str] = KVLayout.V4,
    freqs_cis: Optional[torch.Tensor] = None,
) -> None:
    """Quantize ``input`` ``[num_tokens, 512]`` (bf16, normed and rotated) into the
    paged cache at ``indices``.

    :param layout: the cache's :class:`KVLayout`. ``V4`` is the 584-byte layout
        (fp8 nope, bf16 rope); ``V41`` (528 B) and ``V41_FP4`` (288 B) are the
        V4.1 formats, fp8 with per-32 ue8m0 scales and e2m1 with per-16 e4m3
        scales over all 512 dims.
    :param freqs_cis: V4.1 layouts only. ``[num_tokens, 32]`` complex or
        ``[num_tokens, 64]`` fp32 (real / imag interleaved): rotate the 64-dim
        RoPE tail in-kernel first, so that the caller passes the un-rotated,
        un-quantized latent and the fp4 / fp8 rounding happens exactly once.
    """
    layout = KVLayout.parse(layout)
    if is_hip_runtime():
        assert layout is KVLayout.V4 and freqs_cis is None, (
            "the V4.1 KV layouts are CUDA (sm100) only"
        )
        from sglang.kernels.ops.kvcache.triton_store_cache import (
            triton_fused_store_cache,
        )

        triton_fused_store_cache(input, cache, indices, page_size=page_size, type=type)
    else:
        module = _jit_fused_store_module(
            name=type,
            input_dtype=input.dtype,
            index_dtype=indices.dtype,
            page_size=page_size,
            layout=layout,
        )
        if freqs_cis is None:
            module.run(input, cache, indices)
        else:
            assert layout is not KVLayout.V4, "the V4 layout has no in-kernel RoPE"
            if freqs_cis.is_complex():
                freqs_cis = torch.view_as_real(freqs_cis).flatten(-2)
            module.run_rope(input, cache, indices, freqs_cis.contiguous())


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
        if compress_ratio == 128:
            state_loc = rid * ring_size + (pos % ring_size)
        else:
            loc = tl.load(
                req_to_token_ptr
                + rid.to(tl.int64) * stride_req_to_token_0
                + pos.to(tl.int64) * stride_req_to_token_1,
                mask=mask,
                other=0,
            ).to(tl.int32)
            swa_loc = tl.load(
                full_to_swa_index_mapping_ptr + loc, mask=mask, other=0
            ).to(tl.int32)
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
        stride_req_to_token_1=req_to_token.stride(1),  # type: ignore
        stride_out_1_0=out_1.stride(0),
        stride_out_1_1=out_1.stride(1),  # type: ignore
        compress_ratio=compress_ratio,  # type: ignore
        is_overlap=1 if is_overlap else 0,  # type: ignore
        swa_page_size=swa_page_size,  # type: ignore
        ring_size=ring_size,  # type: ignore
        BLOCK=block,  # type: ignore
    )

    if not is_overlap:
        out_1.squeeze_(1)
    return out_0, out_1
