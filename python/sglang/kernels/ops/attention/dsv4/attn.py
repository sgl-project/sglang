from typing import Literal, Optional, Union

import torch

from sglang.kernels.jit.utils import (
    cache_once,
    is_arch_support_pdl,
    is_hip_runtime,
    load_jit,
    make_cpp_args,
)
from sglang.srt.utils import is_gfx95_supported

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
    layout: KVLayout,
):
    if name == "flashmla":
        args = make_cpp_args(
            input_dtype, index_dtype, page_size, layout.cpp_name, is_arch_support_pdl()
        )
        # The V4 layout keeps its RoPE dims in bf16 and has no in-kernel RoPE.
        cname = "FlashMLA"
        wrappers = ["run"] if layout is KVLayout.V4 else ["run", "run_rope"]
    else:
        assert layout is KVLayout.V4, "only the FlashMLA cache has V4.1 layouts"
        args = make_cpp_args(input_dtype, index_dtype, page_size, is_arch_support_pdl())
        cname, wrappers = "Indexer", ["run"]
    kernel_class = f"FusedStoreCache{cname}Kernel<{args}>"
    return load_jit(
        make_name("store_" + name),
        *args,
        cuda_files=["deepseek_v4/store.cuh"],
        cuda_wrappers=[(w, f"{kernel_class}::{w}") for w in wrappers],
    )


def get_paged_mqa_logits_metadata(seq_lens: torch.Tensor, page_size: int, num_sm: int):
    # The schedule only depends on the sequence lengths (256-token splits), not
    # on the page size.
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
        ``[num_tokens, 64]`` fp32 (real / imag interleaved); rotates the 64-dim
        RoPE tail in-kernel, so ``input`` must then be the un-rotated latent.
    """
    layout = KVLayout.parse(layout)
    if is_hip_runtime() and layout is KVLayout.V4:
        assert freqs_cis is None, "the V4 store receives already-rotated inputs"
        from sglang.kernels.ops.kvcache.triton_store_cache import (
            triton_fused_store_cache,
        )

        triton_fused_store_cache(input, cache, indices, page_size=page_size, type=type)
    else:
        assert not is_hip_runtime() or is_gfx95_supported(), (
            "V4.1 KV stores on HIP require gfx950"
        )
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
