from __future__ import annotations

from typing import TYPE_CHECKING, List, Literal, Optional, TypeAlias, Union, cast

import torch

from sglang.jit_kernel.dsv4 import (
    CompressorDecodePlan,
    CompressorPrefillPlan,
    compress_forward,
    compress_norm_rope_store,
)
from sglang.srt.environ import envs

if TYPE_CHECKING:
    from sglang.srt.layers.attention.compressed.metadata import DeepseekV4Metadata
    from sglang.srt.mem_cache.deepseekv4_memory_pool import DeepSeekV4TokenToKVPool
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.models.deepseek_v4 import Compressor, DeepseekRefRMSNorm


CompressMetadata: TypeAlias = Union[CompressorDecodePlan, CompressorPrefillPlan]


class CompressorBackend:
    def __init__(self):
        super().__init__()
        self.forward_metadata: DeepseekV4Metadata
        assert (
            not envs.SGLANG_OPT_USE_ONLINE_COMPRESS.get()
        ), "online compress 128 not supported yet"

    # NOTE: Will be overridden
    def _maybe_upgrade_forward_metadata(self): ...

    def _get_paged_compress_metadata(self, compress_ratio: int) -> CompressMetadata:
        attr_name = f"c{compress_ratio}_compress_metadata"
        metadata = getattr(self.forward_metadata, attr_name)
        assert isinstance(metadata, (CompressorDecodePlan, CompressorPrefillPlan))
        return metadata

    def _get_out_loc(self, compress_ratio: int) -> torch.Tensor:
        core_metadata = self.forward_metadata.core_metadata
        return (
            core_metadata.c4_out_loc
            if compress_ratio == 4
            else core_metadata.c128_out_loc
        )

    def _forward_compress_all_in_one(
        self,
        *,
        kv_score_buffer: torch.Tensor,
        kv_score_input: torch.Tensor,
        ape: torch.Tensor,
        head_dim: int,
        norm: DeepseekRefRMSNorm,
        freqs_cis_cache: torch.Tensor,
        kv_cache: torch.Tensor,
        is_indexer: bool,
        rotate: bool,
        compress_ratio: int,
        page_size: int,
    ) -> None:
        assert compress_ratio == 4 or compress_ratio == 128
        assert rotate == is_indexer == (head_dim == 128)

        plan = self._get_paged_compress_metadata(compress_ratio)
        coff = 2 if is_overlap_compress(compress_ratio) else 1
        last_dim = 2 * head_dim * coff
        assert kv_score_buffer.shape[-1] == last_dim
        kv_score_buffer = kv_score_buffer.view(-1, compress_ratio, last_dim)
        kv_compressed = compress_forward(
            kv_score_buffer=kv_score_buffer,
            kv_score_input=kv_score_input,
            ape=ape.view(-1, head_dim),
            plan=plan,
            compress_ratio=compress_ratio,
            head_dim=head_dim,
        )
        # NOTE: we use some hack here...
        compress_norm_rope_store(
            kv_compressed,
            plan,
            norm_weight=norm.weight,
            norm_eps=norm.eps,
            freq_cis=freqs_cis_cache,
            out_loc=self._get_out_loc(compress_ratio),
            kvcache=kv_cache,
            page_size=page_size,
        )

    def forward_unified(
        self,
        x: torch.Tensor,
        forward_batch: ForwardBatch,
        layer_id: int,
        compressor: Compressor,
    ) -> None:
        if forward_batch.forward_mode.is_idle():
            return

        self._maybe_upgrade_forward_metadata()
        token_to_kv_pool = forward_batch.token_to_kv_pool
        token_to_kv_pool = cast("DeepSeekV4TokenToKVPool", token_to_kv_pool)
        kv_score_input = compressor.compute_kv_score(x, forward_batch)
        state_pool = compressor.get_state_pool(forward_batch)
        if compressor.is_in_indexer:
            kv_cache = token_to_kv_pool.get_index_k_with_scale_buffer(layer_id)
            page_size = token_to_kv_pool.get_index_k_page_size()
        else:
            kv_cache = token_to_kv_pool.get_extra_key_buffer(layer_id)
            page_size = token_to_kv_pool.get_extra_key_page_size(layer_id)
        self._forward_compress_all_in_one(
            kv_score_buffer=state_pool.kv_score_buffer.kv_score,
            kv_score_input=kv_score_input,
            ape=compressor.ape,
            head_dim=compressor.head_dim,
            norm=compressor.norm,
            freqs_cis_cache=compressor.freqs_cis,
            kv_cache=kv_cache.view(dtype=torch.uint8),
            is_indexer=compressor.is_in_indexer,
            rotate=compressor.rotate,
            compress_ratio=compressor.ratio,
            page_size=page_size,
        )

    # NOTE: alias for backward compatibility
    forward_indexer_compressor = forward_unified
    forward_core_compressor = forward_unified


def is_overlap_compress(compress_ratio: int) -> bool:
    return compress_ratio == 4


def create_paged_compressor_data(
    compress_ratio: Literal[4, 128],
    *,
    is_prefill: bool,
    token_to_kv_pool: DeepSeekV4TokenToKVPool,
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    extend_lens: Optional[torch.Tensor] = None,
    seq_lens_cpu: Optional[List[int]] = None,
    extend_lens_cpu: Optional[List[int]] = None,
    use_prefill_cuda_graph: bool = False,
    num_q_tokens: Optional[int] = None,
) -> CompressMetadata:
    """Build the paged compress metadata (= the plan).

    State-pool slot translation is done inside the C++ planner; the
    Python side just hands the relevant tensors over.
    """
    swa_page_size = token_to_kv_pool.swa_page_size
    ring_size = token_to_kv_pool.get_ring_size(compress_ratio=compress_ratio)
    # NOTE: This is actually a proxy, which encounter some bug with tvm-ffi.
    # As a workaround, we use `.detach()` to get the real tensor.
    full_to_swa = token_to_kv_pool.full_to_swa_index_mapping.detach()

    # The planner wants int64 across the board for the device tables (so they
    # can be indexed by int64 multiplication inside the GPU finalize kernel).
    req_pool_indices_i64 = req_pool_indices.to(torch.int64)

    if is_prefill:
        assert extend_lens is not None
        if seq_lens_cpu is not None:
            assert extend_lens_cpu is not None
            seq_lens_planner = torch.tensor(seq_lens_cpu, dtype=torch.int64)
            extend_lens_planner = torch.tensor(extend_lens_cpu, dtype=torch.int64)
            num_q_tokens = sum(extend_lens_cpu)
        else:
            assert num_q_tokens is not None
            seq_lens_planner = seq_lens.to(torch.int64)
            extend_lens_planner = extend_lens.to(torch.int64)

        return CompressorPrefillPlan.generate(
            compress_ratio=compress_ratio,
            req_pool_indices=req_pool_indices_i64,
            seq_lens=seq_lens_planner,
            extend_lens=extend_lens_planner,
            req_to_token=req_to_token,
            full_to_swa=full_to_swa,
            swa_page_size=swa_page_size,
            ring_size=ring_size,
            num_q_tokens=num_q_tokens,
            use_cuda_graph=use_prefill_cuda_graph,
        )
    else:
        # Decode plan: seq_lens lives on the same device as the input tables
        # (the c_plan.cuh decode kernel reads everything on GPU).
        return CompressorDecodePlan.generate(
            compress_ratio=compress_ratio,
            req_pool_indices=req_pool_indices_i64,
            req_to_token=req_to_token,
            full_to_swa=full_to_swa,
            seq_lens=seq_lens.to(torch.int64),
            swa_page_size=int(swa_page_size),
            ring_size=int(ring_size),
        )
