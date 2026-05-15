from __future__ import annotations

from typing import TYPE_CHECKING, List, Literal, Optional, TypeAlias, Union, cast

import torch

from sglang.jit_kernel.dsv4 import (
    CompressorDecodePlan,
    CompressorPrefillPlan,
    compress_forward,
    compress_norm_rope_store,
)
from sglang.jit_kernel.deepseek_v4 import (
    compress_forward as legacy_compress_forward,
    compress_fused_norm_rope_inplace,
)
from sglang.srt.environ import envs
from sglang.srt.layers.attention.dsv4.compressor import (
    create_paged_compressor_data as create_legacy_paged_compressor_data,
)
from sglang.srt.layers.attention.dsv4.quant_k_cache import (
    quant_to_nope_fp8_rope_bf16_pack_triton,
)
from sglang.srt.layers.attention.nsa.triton_kernel import act_quant

if TYPE_CHECKING:
    from sglang.srt.layers.attention.deepseek_v4_backend import DSV4Metadata
    from sglang.srt.layers.attention.dsv4.compressor import Compressor
    from sglang.srt.layers.layernorm import RMSNorm
    from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


CompressMetadata: TypeAlias = Union[CompressorDecodePlan, CompressorPrefillPlan]
# NOTE: alias for backward compatibility
FusedCompressMetadata: TypeAlias = CompressMetadata


def _use_online_compress(compress_ratio: int) -> bool:
    """Online state-pool path is c128-only."""
    return compress_ratio == 128 and envs.SGLANG_OPT_USE_ONLINE_COMPRESS.get()


class CompressorBackendMixin:
    def __init__(self):
        super().__init__()
        self.forward_metadata: DSV4Metadata

    # NOTE: Will be overridden
    def _maybe_upgrade_forward_metadata(self): ...

    def _get_paged_compress_metadata(self, compress_ratio: int) -> CompressMetadata:
        attr_name = f"c{compress_ratio}_compress_metadata"
        return getattr(self.forward_metadata, attr_name)

    def _get_out_loc(self, compress_ratio: int) -> torch.Tensor:
        attr_name = f"c{compress_ratio}_out_loc"
        return getattr(self.forward_metadata.core_metadata, attr_name)

    def _forward_compress_all_in_one(
        self,
        *,
        kv_score_buffer: torch.Tensor,
        kv_score_input: torch.Tensor,
        ape: torch.Tensor,
        head_dim: int,
        norm: RMSNorm,
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
        core_metadata = self.forward_metadata.core_metadata
        is_online = _use_online_compress(compress_ratio)
        if is_online:
            kv_score_buffer = kv_score_buffer.view(-1, 1, head_dim * 3)
        else:
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
            is_online=is_online,
        )
        if kv_compressed.shape[0] == 0:
            if (
                getattr(core_metadata, "no_prefix_ragged_prefill", False)
                and not plan.is_decode
            ):
                if is_indexer:
                    self._current_ragged_indexer_kv_fp8 = kv_compressed
                    self._current_ragged_indexer_kv_scale = kv_compressed.new_empty(
                        (0, 1)
                    )
                else:
                    compact_kv = kv_compressed
                    if compact_kv.ndim == 2:
                        compact_kv = compact_kv.unsqueeze(1)
                    self._current_ragged_extra_kv[compress_ratio] = (
                        compact_kv.contiguous()
                    )
                if envs.SGLANG_DSV4_DEBUG_NO_PREFIX_RAGGED.get():
                    import logging

                    logging.getLogger(__name__).warning(
                        "DSV4 no-prefix ragged compact kv v2: skip empty "
                        "compress output, ratio=%s, is_indexer=%s",
                        compress_ratio,
                        is_indexer,
                    )
            return
        if (
            getattr(core_metadata, "no_prefix_ragged_prefill", False)
            and not plan.is_decode
        ):
            compact_kv = kv_compressed.clone()
            compress_fused_norm_rope_inplace(
                compact_kv,
                norm.weight,
                norm.variance_epsilon,
                freqs_cis_cache,
                plan,
            )
            if is_indexer:
                from sglang.srt.layers.attention.nsa.nsa_indexer import (
                    rotate_activation,
                )

                compact_kv = rotate_activation(compact_kv).contiguous()
                if compact_kv.numel() == 0:
                    self._current_ragged_indexer_kv_fp8 = compact_kv
                    self._current_ragged_indexer_kv_scale = compact_kv.new_empty((0, 1))
                else:
                    (
                        self._current_ragged_indexer_kv_fp8,
                        self._current_ragged_indexer_kv_scale,
                    ) = act_quant(compact_kv)
                if envs.SGLANG_DSV4_DEBUG_NO_PREFIX_RAGGED.get():
                    import logging

                    logging.getLogger(__name__).warning(
                        "DSV4 no-prefix ragged compact indexer kv v2: "
                        "ratio=%s, compact_kv=%s",
                        compress_ratio,
                        tuple(compact_kv.shape),
                    )
            else:
                if compact_kv.dtype != torch.bfloat16:
                    compact_kv = compact_kv.to(torch.bfloat16)
                if compact_kv.ndim == 2:
                    compact_kv = compact_kv.unsqueeze(1)
                self._current_ragged_extra_kv[compress_ratio] = compact_kv.contiguous()
                if envs.SGLANG_DSV4_DEBUG_NO_PREFIX_RAGGED.get():
                    import logging

                    logging.getLogger(__name__).warning(
                        "DSV4 no-prefix ragged compact core kv v2: "
                        "ratio=%s, compact_kv=%s",
                        compress_ratio,
                        tuple(compact_kv.shape),
                    )
        # NOTE: we use some hack here...
        compress_norm_rope_store(
            kv_compressed,
            plan,
            norm_weight=norm.weight,
            norm_eps=norm.variance_epsilon,
            freq_cis=freqs_cis_cache,
            out_loc=self._get_out_loc(compress_ratio),
            kvcache=kv_cache,
            page_size=page_size,
        )
    def _forward_no_prefix_legacy_compress(
        self,
        *,
        kv_score_buffer: torch.Tensor,
        kv_score_input: torch.Tensor,
        ape: torch.Tensor,
        head_dim: int,
        norm: RMSNorm,
        freqs_cis_cache: torch.Tensor,
        forward_batch: ForwardBatch,
        layer_id: int,
        is_indexer: bool,
        rotate: bool,
        compress_ratio: int,
    ) -> None:
        from sglang.srt.layers.attention.nsa.nsa_indexer import rotate_activation

        token_to_kv_pool = cast("DeepSeekV4TokenToKVPool", forward_batch.token_to_kv_pool)
        core_metadata = self.forward_metadata.core_metadata
        metadata = create_legacy_paged_compressor_data(
            compress_ratio=cast(Literal[4, 128], compress_ratio),
            is_prefill=True,
            token_to_kv_pool=token_to_kv_pool,
            req_to_token=forward_batch.req_to_token_pool.req_to_token,
            req_pool_indices=forward_batch.req_pool_indices,
            seq_lens=forward_batch.seq_lens.to(torch.int32),
            extend_lens=forward_batch.extend_seq_lens,
            seq_lens_cpu=(
                forward_batch.seq_lens_cpu.tolist()
                if forward_batch.seq_lens_cpu is not None
                else None
            ),
            extend_lens_cpu=forward_batch.extend_seq_lens_cpu,
            use_prefill_cuda_graph=False,
        )
        coff = 2 if is_overlap_compress(compress_ratio) else 1
        last_dim = 2 * head_dim * coff
        assert kv_score_buffer.shape[-1] == last_dim
        kv_score_buffer = kv_score_buffer.view(-1, compress_ratio, last_dim)
        kv_compressed = legacy_compress_forward(
            kv_score_buffer=kv_score_buffer,
            kv_score_input=kv_score_input,
            ape=ape.view(-1, head_dim),
            indices=metadata.write_loc,
            plan=metadata.plan,
            compress_ratio=compress_ratio,
            head_dim=head_dim,
            extra_data=metadata.extra_data,
        )
        compress_fused_norm_rope_inplace(
            kv_compressed,
            norm.weight,
            norm.variance_epsilon,
            freqs_cis_cache,
            metadata.plan,
        )
        if rotate:
            kv_compressed = rotate_activation(kv_compressed)

        emit_mask = (core_metadata.seq_lens_casual % compress_ratio) == 0
        compact_kv = kv_compressed[emit_mask].contiguous()
        if is_indexer:
            if compact_kv.numel() == 0:
                self._current_ragged_indexer_kv_fp8 = compact_kv
                self._current_ragged_indexer_kv_scale = compact_kv.new_empty((0, 1))
            else:
                (
                    self._current_ragged_indexer_kv_fp8,
                    self._current_ragged_indexer_kv_scale,
                ) = act_quant(compact_kv)
            if envs.SGLANG_OPT_USE_FUSED_STORE_CACHE.get():
                token_to_kv_pool.set_index_k_fused(
                    layer_id=layer_id,
                    loc=core_metadata.c4_out_loc,
                    cache_k=kv_compressed,
                )
            else:
                index_k, index_k_scale = act_quant(kv_compressed)
                token_to_kv_pool.set_index_k_scale_buffer(
                    layer_id=layer_id,
                    loc=core_metadata.c4_out_loc,
                    index_k=index_k,
                    index_k_scale=index_k_scale,
                )
        else:
            compact_extra = compact_kv
            if compact_extra.dtype != torch.bfloat16:
                compact_extra = compact_extra.to(torch.bfloat16)
            if compact_extra.ndim == 2:
                compact_extra = compact_extra.unsqueeze(1)
            self._current_ragged_extra_kv[compress_ratio] = compact_extra.contiguous()
            out_loc = (
                core_metadata.c4_out_loc
                if compress_ratio == 4
                else core_metadata.c128_out_loc
            )
            if envs.SGLANG_OPT_USE_FUSED_STORE_CACHE.get():
                token_to_kv_pool.set_extra_key_buffer_fused(
                    layer_id=layer_id,
                    loc=out_loc,
                    cache_k=kv_compressed,
                )
            else:
                pack = quant_to_nope_fp8_rope_bf16_pack_triton(kv_compressed.bfloat16())
                token_to_kv_pool.set_extra_key_buffer(layer_id, out_loc, pack)

        if envs.SGLANG_DSV4_DEBUG_NO_PREFIX_RAGGED.get():
            import logging

            logging.getLogger(__name__).warning(
                "DSV4 no-prefix ragged legacy compressor under v2: ratio=%s, "
                "is_indexer=%s, kv=%s, compact=%s",
                compress_ratio,
                is_indexer,
                tuple(kv_compressed.shape),
                tuple(compact_kv.shape),
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
        core_metadata = self.forward_metadata.core_metadata
        if (
            forward_batch.forward_mode.is_extend_without_speculative()
            and getattr(core_metadata, "no_prefix_ragged_prefill", False)
            and not _use_online_compress(compressor.ratio)
        ):
            state_pool = compressor.get_state_pool(forward_batch)
            self._forward_no_prefix_legacy_compress(
                kv_score_buffer=state_pool.kv_score_buffer.kv_score,
                kv_score_input=kv_score_input,
                ape=compressor.ape,
                head_dim=compressor.head_dim,
                norm=compressor.norm,
                freqs_cis_cache=compressor.freqs_cis,
                forward_batch=forward_batch,
                layer_id=layer_id,
                is_indexer=compressor.is_in_indexer,
                rotate=compressor.rotate,
                compress_ratio=compressor.ratio,
            )
            return
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
    if _use_online_compress(compress_ratio):
        return _create_online_paged_compressor_data(
            is_prefill=is_prefill,
            token_to_kv_pool=token_to_kv_pool,
            req_to_token=req_to_token,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            extend_lens=extend_lens,
            seq_lens_cpu=seq_lens_cpu,
            extend_lens_cpu=extend_lens_cpu,
            use_prefill_cuda_graph=use_prefill_cuda_graph,
            num_q_tokens=num_q_tokens,
        )

    swa_page_size = token_to_kv_pool.swa_page_size
    ring_size = token_to_kv_pool.get_ring_size(compress_ratio=compress_ratio)
    # NOTE: This is actually a proxy, which encounter some bug with tvm-ffi.
    # As a workaround, we use `.detach()` to get the real tensor.
    full_to_swa = token_to_kv_pool.full_to_swa_index_mapping.detach()
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
        return CompressorDecodePlan.generate(
            compress_ratio=compress_ratio,
            req_pool_indices=req_pool_indices_i64,
            req_to_token=req_to_token,
            full_to_swa=full_to_swa,
            seq_lens=seq_lens.to(torch.int64),
            swa_page_size=swa_page_size,
            ring_size=ring_size,
        )


def _create_online_paged_compressor_data(
    *,
    is_prefill: bool,
    token_to_kv_pool: DeepSeekV4TokenToKVPool,
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    extend_lens: Optional[torch.Tensor],
    seq_lens_cpu: Optional[List[int]],
    extend_lens_cpu: Optional[List[int]],
    use_prefill_cuda_graph: bool,
    num_q_tokens: Optional[int],
) -> CompressMetadata:
    assert not use_prefill_cuda_graph, "online c128 doesn't support cuda graph"

    swa_page_size = int(token_to_kv_pool.swa_page_size)
    full_to_swa = token_to_kv_pool.full_to_swa_index_mapping.detach()
    req_pool_indices = req_pool_indices.to(torch.int64)

    if is_prefill:
        # Sync-on-entry: catch IMA from a prior layer / kernel BEFORE we touch
        # anything in this builder, so blame doesn't land on us spuriously.
        assert extend_lens is not None
        if seq_lens_cpu is not None:
            assert extend_lens_cpu is not None
            seq_lens_planner = torch.tensor(seq_lens_cpu, dtype=torch.int64)
            extend_lens_planner = torch.tensor(extend_lens_cpu, dtype=torch.int64)
            num_q_tokens_planner = sum(extend_lens_cpu)
        else:
            assert num_q_tokens is not None
            seq_lens_planner = seq_lens.to(torch.int64)
            extend_lens_planner = extend_lens.to(torch.int64)
            num_q_tokens_planner = num_q_tokens

        return CompressorPrefillPlan.generate_online(
            seq_lens=seq_lens_planner,
            extend_lens=extend_lens_planner,
            req_pool_indices=req_pool_indices,
            req_to_token=req_to_token,
            full_to_swa=full_to_swa,
            num_q_tokens=int(num_q_tokens_planner),
            swa_page_size=swa_page_size,
        )
    else:
        return CompressorDecodePlan.generate_online(
            seq_lens=seq_lens.to(torch.int64),
            req_pool_indices=req_pool_indices,
            req_to_token=req_to_token,
            full_to_swa=full_to_swa,
            swa_page_size=swa_page_size,
        )
