from __future__ import annotations

from typing import TYPE_CHECKING, List, Literal, Optional, TypeAlias, Union

import torch

from sglang.jit_kernel.dsv4 import (
    CompressorDecodePlan,
    CompressorPrefillPlan,
    compress_forward,
    compress_fused_norm_rope_inplace,
)
from sglang.srt.environ import envs
from sglang.srt.layers.attention.nsa.quant_k_cache_v4 import (
    quant_to_nope_fp8_rope_bf16_pack_triton,
)
from sglang.srt.layers.attention.nsa.triton_kernel import act_quant
from sglang.srt.layers.attention.nsa.utils import (
    assert_tensor_identical_across_cp_ranks,
)

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

    def get_paged_compress_metadata(self, compress_ratio: int) -> CompressMetadata:
        attr_name = f"c{compress_ratio}_compress_metadata"
        metadata = getattr(self.forward_metadata, attr_name)
        assert isinstance(metadata, (CompressorDecodePlan, CompressorPrefillPlan))
        return metadata

    def forward_compress(
        self,
        *,
        kv_score_buffer: torch.Tensor,
        kv_score_input: torch.Tensor,
        ape: torch.Tensor,
        head_dim: int,
        norm: DeepseekRefRMSNorm,
        freqs_cis_cache: torch.Tensor,
        rotate: bool,
        forward_batch: ForwardBatch,
        compress_ratio: int,
        is_paged: bool = False,
    ) -> torch.Tensor:
        from sglang.srt.layers.attention.nsa.nsa_indexer import rotate_activation

        assert compress_ratio == 4 or compress_ratio == 128
        assert is_paged, "Non-paged compress path is no longer supported."

        plan = self.get_paged_compress_metadata(compress_ratio)
        coff = 2 if is_overlap_compress(compress_ratio) else 1
        if compress_ratio == 128 and envs.SGLANG_OPT_USE_ONLINE_COMPRESS.get():
            kv_score_buffer = kv_score_buffer.view(-1, 1, head_dim * 3)
        else:
            last_dim = 2 * head_dim * coff
            assert kv_score_buffer.shape[-1] == last_dim
            kv_score_buffer = kv_score_buffer.view(-1, compress_ratio, last_dim)

        kv_compressed = compress_forward(
            kv_score_buffer=kv_score_buffer,
            kv_score_input=kv_score_input,
            ape=ape,
            plan=plan,
            compress_ratio=compress_ratio,
            head_dim=head_dim,
        )
        compress_fused_norm_rope_inplace(
            kv_compressed,
            norm.weight,
            norm.eps,
            freqs_cis_cache,
            plan,
        )
        return rotate_activation(kv_compressed) if rotate else kv_compressed

    def forward_core_compressor(
        self,
        x: torch.Tensor,
        forward_batch: ForwardBatch,
        layer_id: int,
        compressor: Compressor,
    ) -> None:
        if forward_batch.forward_mode.is_idle():
            return
        # PREP_IN_CG lazy upgrade: the concrete backend (DeepseekV4BackendRadix)
        # owns this helper. MQALayer._forward_prepare calls us before
        # attn_backend.forward(), so Raw -> Radix must happen here too
        # (e.g. 1.6T layer 0 has compress_ratio=128 and needs cX_compress_metadata).
        self._maybe_upgrade_forward_metadata()
        token_to_kv_pool = forward_batch.token_to_kv_pool
        if TYPE_CHECKING:
            assert isinstance(token_to_kv_pool, DeepSeekV4TokenToKVPool)

        new_compressed_kv = compressor(x, forward_batch)
        if envs.SGLANG_DEBUG_HACK_CP_CHECK_RANK_CONSISTENCY.get():
            assert_tensor_identical_across_cp_ranks(
                new_compressed_kv,
                tag=f"compressor(ratio={compressor.ratio}) layer_id={layer_id}",
                forward_batch=forward_batch,
            )
        core_metadata = self.forward_metadata.core_metadata
        out_loc = (
            core_metadata.c4_out_loc
            if compressor.ratio == 4
            else core_metadata.c128_out_loc
        )
        if envs.SGLANG_OPT_USE_FUSED_STORE_CACHE.get():
            token_to_kv_pool.set_extra_key_buffer_fused(
                layer_id=layer_id,
                loc=out_loc,
                cache_k=new_compressed_kv,
            )
        else:
            pack = quant_to_nope_fp8_rope_bf16_pack_triton(new_compressed_kv.bfloat16())
            token_to_kv_pool.set_extra_key_buffer(layer_id, out_loc, pack)

    def forward_indexer_compressor(
        self,
        x: torch.Tensor,
        forward_batch: ForwardBatch,
        layer_id: int,
        compressor: Compressor,
    ) -> None:
        assert is_overlap_compress(compressor.ratio)
        # PREP_IN_CG lazy upgrade (see forward_core_compressor for rationale).
        self._maybe_upgrade_forward_metadata()
        token_to_kv_pool = forward_batch.token_to_kv_pool
        if TYPE_CHECKING:
            assert isinstance(token_to_kv_pool, DeepSeekV4TokenToKVPool)

        new_compressed_kv = compressor(x, forward_batch)
        if envs.SGLANG_DEBUG_HACK_CP_CHECK_RANK_CONSISTENCY.get():
            assert_tensor_identical_across_cp_ranks(
                new_compressed_kv,
                tag=f"indexer_compressor(ratio={compressor.ratio}) layer_id={layer_id}",
                forward_batch=forward_batch,
            )
        if envs.SGLANG_OPT_USE_FUSED_STORE_CACHE.get():
            token_to_kv_pool.set_index_k_fused(
                layer_id=layer_id,
                loc=self.forward_metadata.core_metadata.c4_out_loc,
                cache_k=new_compressed_kv,
            )
        else:
            new_compressed_kv_fp8, new_compressed_kv_scale = act_quant(
                new_compressed_kv
            )
            token_to_kv_pool.set_index_k_scale_buffer(
                layer_id=layer_id,
                loc=self.forward_metadata.core_metadata.c4_out_loc,
                index_k=new_compressed_kv_fp8,
                index_k_scale=new_compressed_kv_scale,
            )


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
        # The CPU planner loop reads seq_lens / extend_lens; both must be
        # CPU-resident (per c_plan.cuh's current implementation).
        if seq_lens_cpu is not None:
            assert extend_lens_cpu is not None
            seq_lens_planner = torch.tensor(seq_lens_cpu, dtype=torch.int64)
            extend_lens_planner = torch.tensor(extend_lens_cpu, dtype=torch.int64)
            num_q_tokens = sum(extend_lens_cpu)
        else:
            assert num_q_tokens is not None
            assert False, "Not supported yet"

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
            device=seq_lens.device,
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
