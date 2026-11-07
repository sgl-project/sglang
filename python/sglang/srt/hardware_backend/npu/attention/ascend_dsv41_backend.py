from __future__ import annotations

import functools
import logging
import os
from types import SimpleNamespace
from typing import TYPE_CHECKING, Optional

import torch
import torch.nn.functional as F
import torch_npu

from sglang.srt.hardware_backend.npu.attention.ascend_dsv4_backend import (
    DeepseekV4AscendAttnBackend,
    DeepseekV4AscendMultiStepDraftBackend,
)
from sglang.srt.hardware_backend.npu.utils import is_npu_arch35
from sglang.srt.runtime_context import get_parallel

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)


@functools.cache
def _ds41_ops():
    """The `cann_ops_transformer.ops.ds41` module with the V4.1 Python operators.

    Only `compressor`, `indexer_prologue_k`, `quant_lightning_indexer` and
    `quant_sparse_lightning_indexer` are exposed as Python ops (they need
    Python-side argument marshalling); the remaining V4.1 operators go through
    `torch.ops.*` and are called inline.

    `ds41` is NOT in `cann_ops_transformer.ops`'s static registration table, so
    importing this module is also what registers the
    `torch.ops.cann_ops_transformer.ds41.*` namespace (SMLA, mhc_post, the
    `_metadata` variants). Call this once during backend init, before any
    `torch.ops.cann_ops_transformer.ds41.*` use.
    """
    from cann_ops_transformer.ops import ds41

    return ds41


@functools.cache
def _dsv41_a3_ops():
    """Use the paired BF16 A3 operators bundled by sgl-kernel-npu."""
    from sgl_kernel_npu.operators.attention import (
        sparse_flash_mla,
        sparse_flash_mla_metadata,
    )
    from sgl_kernel_npu.operators.indexer import (
        lightning_indexer_v2,
        sparse_lightning_indexer,
    )

    return SimpleNamespace(
        attention=sparse_flash_mla,
        metadata=sparse_flash_mla_metadata,
        indexer=lightning_indexer_v2,
        # The public API selects the four-output overload when candidate
        # arguments are supplied; ordinary top-k still returns two outputs.
        candidate=lightning_indexer_v2,
        sparse_indexer=sparse_lightning_indexer,
    )


@functools.cache
def _dsv41_a3_compressor_kernel():
    from sgl_kernel_npu.operators.attention import compressor_v2

    return compressor_v2


@torch.library.custom_op("sglang::dsv41_a3_compressor", mutates_args=("state_cache",))
def _dsv41_a3_compressor(
    x: torch.Tensor,
    wkv: torch.Tensor,
    wgate: torch.Tensor,
    state_cache: torch.Tensor,
    state_block_table: torch.Tensor,
    cu_seqlens: torch.Tensor,
    seqused: torch.Tensor,
    start_pos: torch.Tensor,
    cmp_ratio: int,
) -> torch.Tensor:
    # Keep the cycle-bank mutation visible to graph tracing while using the
    # public sgl-kernel-npu entry point (including its input validation).
    return _dsv41_a3_compressor_kernel()(
        x,
        wkv,
        wgate,
        state_cache,
        cmp_ratio,
        state_block_table=state_block_table,
        cu_seqlens=cu_seqlens,
        seqused=seqused,
        start_pos=start_pos,
    )


@_dsv41_a3_compressor.register_fake
def _dsv41_a3_compressor_fake(
    x,
    wkv,
    wgate,
    state_cache,
    state_block_table,
    cu_seqlens,
    seqused,
    start_pos,
    cmp_ratio,
):
    rows = min(x.shape[0], x.shape[0] // cmp_ratio + cu_seqlens.shape[0] - 1)
    return x.new_empty((rows, wkv.shape[0]))


class LowRatioCompressAscendBackendMixin:
    """NPU implementation of the DeepSeek-V4.1 low-ratio (1 / 2) source path.

    Mirrors ``DeepseekV4AttnBackend.forward_low_ratio_sources``
    (layers/attention/deepseek_v4_backend.py L3203-L3281) but keeps its own
    orchestration rather than sharing the CUDA backend's, for the same reason
    :class:`C4IndexerAscendBackendMixin` and
    :class:`CompressorAscendBackendMixin` do: ``deepseek_v4_backend`` imports
    Triton kernels at module scope, and the two generations differ in the
    kernels and the cache row formats, not in the skeleton.

    Deliberately dropped relative to the CUDA source (unreachable on NPU):
      * the context-parallel branch (``_forward_low_ratio_sources_cp``);
      * the prefill-graph branch — :attr:`low_ratio_prefill_graph` stays False.

    Intentional deviation from the CUDA source: seam A hands its pre-RoPE
    ``latent`` to seam B, because ``indexer_prologue_k`` projects the index K
    from that latent (``wk: [index_head_dim, head_dim]``).  CUDA needs no such
    handoff: it writes the index K *inside* its compress step, where the same
    latent already exists (``_low_ratio_write_group`` ->
    ``indexer.forward_wk(latent)``, deepseek_v4_backend.py L3613-L3643),
    whereas on NPU the fused ``indexer_prologue_k`` is a separate op call.

    Each method notes its CUDA anchor; keep them in sync when upstream
    semantics change (see the plan's drift checklist).
    """

    # CUDA-only prefill graph (BCG). The model reads this via
    # getattr(attn_backend, "low_ratio_prefill_graph", False)
    # (deepseek_v4.py L2391) to choose the two-call side-stream contract over
    # the single call, so declaring False is what selects L2397-2403 on NPU.
    low_ratio_prefill_graph = False
    # Written by the model / replay-tail path exactly as on CUDA
    # (deepseek_v4_backend.py L2012 / L2067, deepseek_v4_replay_tail.py L637).
    tail_source_kv_layer_id: Optional[int] = None

    def forward_low_ratio_sources(
        self,
        *,
        layer,
        x,
        q_lora,
        positions,
        forward_batch,
        run_compressor: bool = True,
        run_indexer: bool = True,
    ) -> None:
        """Runs on every ratio 1/2 layer before its attention.

        Anchor: DeepseekV4AttnBackend.forward_low_ratio_sources
        (deepseek_v4_backend.py L3203-L3281).
        """
        if forward_batch.forward_mode.is_idle():
            return
        if forward_batch.encoder_swa_replay:
            run_compressor = False
        if layer.layer_id == self.tail_source_kv_layer_id:
            run_compressor = False

        pos = positions
        latent = None
        if run_compressor and layer.compressor is not None:
            latent = self._low_ratio_compress(layer, x, pos, forward_batch)
        if run_indexer and layer.indexer is not None:
            # Seam B needs the latent only to produce and store the indexer K
            # (a layer that does not own K, e.g. a tail layer whose compressor
            # was skipped, only scores against the stored one), so the check
            # lives there rather than here.
            self._low_ratio_index_topk(layer, latent, x, q_lora, pos, forward_batch)

    def _low_ratio_compress(self, layer, x, pos, forward_batch):
        """Seam A: project, normalize and RoPE one ratio-1/2 compressed latent,
        then write it into the ratio's compressed pool.

        Anchor: DeepseekV4AttnBackend._low_ratio_compress_torch
        (deepseek_v4_backend.py L3531). Returns the pre-RoPE latent for seam B.
        """
        ratio = layer.compress_ratio
        compressor = layer.compressor
        fm = self.forward_metadata
        if ratio == 1:
            # No gate and no pairing for ratio 1; the recipes branch is a plain
            # linear as well (compressor.py L127).
            kv, _ = compressor.project(x)
        else:
            # `DeepseekV41Compressor` only builds the fused `wkv_gate` where the
            # CUDA-only fused-compress path is available; on NPU it must expose
            # the two separate projections this call reads. Fail here rather than
            # with an AttributeError on `compressor.wkv` deep inside the forward.
            assert not compressor.use_fused_gate, (
                "DeepSeek-V4.1 ratio-2 compression on NPU needs the unfused "
                "compressor (separate wkv/wgate), but the module has wkv_gate."
            )
            pool = self.token_to_kv_pool
            # Mirrors forward_compress (L453-L517) for the metadata side: A5
            # cache_mode=2 is CYCLE, so the compressor gets one request bank per
            # row and derives the in-bank offset from start_pos.
            # This is the layer's attention compressor, never the nested c4-indexer
            # one, so it always reads the attention state pool. (DeepseekV41Compressor
            # has no `is_in_indexer`; only V4's Compressor does.)
            state_pool = pool._get_state_pool(layer.layer_id, False)
            compress = (
                _ds41_ops().compressor if is_npu_arch35() else _dsv41_a3_compressor
            )
            kv = compress(
                x=x.view(-1, x.shape[-1]),
                wkv=compressor.wkv.weight,
                wgate=compressor.wgate.weight,
                seqused=fm.seqused,
                start_pos=fm.start_pos,
                cmp_ratio=ratio,
                cu_seqlens=fm.actual_seq_lengths_q_pa,
                state_cache=state_pool.state_cache_3d,
                state_block_table=fm.dsv4_cycle_state_block_table,
            )
        out_dim = kv.shape[-1]
        kv = torch_npu.npu_rms_norm(kv, compressor.norm.weight, compressor.norm.eps)[0]
        latent = kv.clone()

        from sglang.srt.hardware_backend.npu.dsv4.dsv4_rope import Dsv4NpuRoPE

        # DeepseekV41Compressor carries no freqs_cis of its own — unlike the V4
        # Compressor, which is handed the layer's — so the layer's table is the
        # rope for its compressed latents (and for the indexer K/Q below). A
        # compress_ratio != 0 layer already builds that table with
        # compress_rope_theta + YaRN (deepseek_v4.py L1053-L1074), which is the
        # reference model's compress_rotary_emb.
        freqs_cis = layer.freqs_cis
        table = Dsv4NpuRoPE.for_freqs(freqs_cis, getattr(layer, "rotary_emb", None))
        rope_dim = freqs_cis.shape[-1] * 2
        cos, sin = table.get_cos_sin(
            getattr(fm, f"positions_cmp_padding_c{ratio}"),
            kv.dtype,
            view_4d=True,
            # The fp32 table is built at model init (deepseek_v4.py L1287), so a
            # captured decode never runs repeat_interleave inside the graph.
            allow_build=False,
            cache_dtype=torch.float32,
        )
        Dsv4NpuRoPE.apply_rotary_mul_inplace(
            kv.view(-1, 1, out_dim),
            None,
            cos,
            sin,
            qk_nope_dim=out_dim - rope_dim,
        )

        slot = getattr(fm, f"c{ratio}_loc", None)
        if slot is None:
            raise RuntimeError(
                f"DeepSeek-V4.1 ratio {ratio} has no c-pool slot metadata: "
                "the compress-loc builder has not been run for this mode yet."
            )
        self.token_to_kv_pool.set_compress_buffer(layer.layer_id, slot, kv, None, False)
        return latent

    def _low_ratio_index_topk(
        self,
        layer,
        latent,
        x,
        q_lora,
        pos,
        forward_batch,
        *,
        rows_per_request=None,  # NOTE: only used under CP (unsupported here)
    ) -> None:
        """Seam B: store the indexer K from ``latent`` and score it with QLI.

        Anchor: DeepseekV4AttnBackend._low_ratio_index_topk
        (deepseek_v4_backend.py L3657-L3705) for the role dispatch, and
        cann-recipes-infer models/deepseek_v4_1/models/modules/indexer.py
        L270-L400 for the NPU operator ABI.
        """
        indexer = layer.indexer
        ratio = layer.compress_ratio
        fm = self.forward_metadata
        if indexer.owns_k:
            self._dsv41_index_k_store(layer, latent)
        if indexer.is_candidate_source and not os.environ.get(
            "SGLANG_DSV41_DISABLE_QSLI"
        ):
            # Level one: the plain top-k this layer attends with, plus the block
            # candidates every later indexer layer rescoring inside.
            topk, blocks = self._dsv41_index_full_topk(
                layer, x, q_lora, pos, forward_batch, candidate=True
            )
            fm.dsv41_candidate_blocks = blocks
        elif indexer.uses_candidates and not os.environ.get(
            "SGLANG_DSV41_DISABLE_QSLI"
        ):
            topk = self._dsv41_index_qsli_topk(layer, x, q_lora, pos, forward_batch)
        else:
            topk, _ = self._dsv41_index_full_topk(layer, x, q_lora, pos, forward_batch)
        setattr(fm, f"c{ratio}_topk_indices", topk)
        # No indexer-topk capture here: get_num_indexer_layers() sizes that
        # capturer from the ratio-4 layers (model_config.py L482-L484), which
        # DeepSeek-V4.1 has none of, so the capturer is disabled for it -- and
        # compress_layer_id is a per-ratio counter that collides across ratios
        # anyway (both V4.1 sources are index 0).

    def _dsv41_index_k_store(self, layer, latent) -> None:
        """Project, norm, RoPE and quantize the index K straight into the split
        index pool (the cache QLI reads).

        Recipes: indexer.py L270-L279 (``write_prologue_k`` with storage_mode=0).
        """
        if latent is None:
            raise RuntimeError(
                "DeepSeek-V4.1 index K needs the compressor latent, but the "
                f"compressor did not run for layer {layer.layer_id}."
            )
        import torch_npu

        indexer = layer.indexer
        ratio = layer.compress_ratio
        fm = self.forward_metadata
        pool = self.token_to_kv_pool
        # -1 rows are the compressor's padding rows; the operator skips them, so
        # the padded (garbage) latent rows are dropped for free.
        cache_index = getattr(fm, f"c{ratio}_loc", None)
        if cache_index is None:
            raise RuntimeError(
                f"DeepSeek-V4.1 index K for layer {layer.layer_id} has no "
                f"c{ratio}_loc slot metadata to write into."
            )
        positions = getattr(fm, f"positions_cmp_padding_c{ratio}", None)
        if positions is None:
            raise RuntimeError(
                f"DeepSeek-V4.1 index K for layer {layer.layer_id} has no "
                f"positions_cmp_padding_c{ratio} metadata."
            )
        from sglang.srt.hardware_backend.npu.dsv4.dsv4_rope import Dsv4NpuRoPE

        # indexer_prologue_k wants flat FP32 [T, Dr] RoPE tensors, unlike
        # inplace_partial_rotary_mul's [T, 1, 1, Dr] bf16.
        cos, sin = Dsv4NpuRoPE.for_freqs(
            layer.freqs_cis, getattr(layer, "rotary_emb", None)
        ).get_cos_sin(
            positions,
            torch.float32,
            view_4d=False,
            allow_build=False,
            cache_dtype=torch.float32,
        )
        if not is_npu_arch35():
            # This checkpoint does not require FP4 rounding. Store ordinary
            # BF16 index keys, using the same pre-RoPE compressed latent.
            index_k = indexer.k_norm(indexer.wk(latent)).to(torch.bfloat16)
            Dsv4NpuRoPE.apply_rotary_mul_inplace(
                index_k.view(-1, 1, indexer.index_head_dim),
                None,
                cos.view(-1, 1, 1, cos.shape[-1]),
                sin.view(-1, 1, 1, sin.shape[-1]),
                qk_nope_dim=indexer.index_head_dim - indexer.rope_head_dim,
            )
            pool.set_compress_buffer(layer.layer_id, cache_index, index_k, None, True)
            return
        ds41 = _ds41_ops()
        # The operator reads wk in FRACTAL_NZ, and the cast is deliberately
        # *inside* the traced region: the NZ tensor is then either an
        # intermediate node consumed straight by the op, or a constant cast
        # at trace time -- in neither case a *graph input*, so aclgraph's
        # input clone cannot degrade it back to ND.  The README's "ACLGraph
        # must disable input clone" note is about a pre-cast NZ tensor that
        # arrives *as an input*: that is the recipes layout, where
        # modeling_deepseek.py L2854-2863 casts attn.indexer.wk in place to
        # FRACTAL_NZ at load time and passes the parameter straight through.
        # Casting here yields the same format without depending on that
        # framework flag, which sglang never sets.  Do NOT hoist this cast
        # into a cached attribute: a long-lived NZ tensor can be lifted as a
        # graph input, which is exactly the case the README warns about.
        # Both storage modes share every argument but the destination.
        common = dict(
            latent=latent.reshape(-1, latent.shape[-1]),
            wk=torch_npu.npu_format_cast(
                indexer.wk.weight, torch_npu.Format.FRACTAL_NZ
            ),
            norm_weight=indexer.k_norm.weight.to(torch.float32),
            rope_sin=sin,
            rope_cos=cos,
            cache_index=cache_index.to(torch.int64),
            norm_eps=indexer.k_norm.eps,
        )
        ds41.indexer_prologue_k(
            **common,
            k_cache=pool.get_index_k(layer.layer_id),
            k_scale_cache=pool.get_index_scale(layer.layer_id),
            storage_mode=0,
        )
        if indexer.is_candidate_source and not os.environ.get(
            "SGLANG_DSV41_DISABLE_QSLI_CACHE"
        ):
            # Level-two K, read by QSLI: the same rows again, eight to a record,
            # fused with their scales (indexer_prologue_k_dsl README L49-L52).
            ds41.indexer_prologue_k(
                **common,
                k_cache=self._dsv41_qsli_cache(layer),
                k_scale_cache=None,
                storage_mode=1,
                combined_block_size=indexer.candidate_block_size,
            )

    def _dsv41_index_q(self, layer, x, q_lora, pos):
        """Packed MXFP4 Q, its E8M0 scales and the head weights, for both indexers.

        Recipes: indexer.py L341-L357 and L410-L415 — QLI and QSLI take the same
        Q side, so it is built once here for both.
        """
        import torch_npu

        indexer = layer.indexer
        t = int(q_lora.shape[0])
        packed_dim = indexer.index_head_dim // 2

        q, _ = indexer.wq_b(q_lora)
        q = q.view(t, indexer.n_heads, indexer.index_head_dim)
        from sglang.srt.hardware_backend.npu.dsv4.dsv4_rope import Dsv4NpuRoPE

        # Like the compressor, DeepseekV41Indexer is built without its own
        # freqs_cis/rotary_emb (deepseek_v4.py L1366-1372), so the layer's own
        # table is the rope for its Q (and for the index K in
        # _dsv41_index_k_store). A compress_ratio != 0 layer builds that table
        # with compress_rope_theta + YaRN (L1053-L1074), which is exactly the
        # reference model's compress_rotary_emb (recipes indexer.py L234 uses
        # cos_sin["comp"] here).
        table = Dsv4NpuRoPE.for_freqs(
            layer.freqs_cis, getattr(layer, "rotary_emb", None)
        )
        cos, sin = table.get_cos_sin(
            pos,
            torch.bfloat16,
            view_4d=True,
            allow_build=False,
            cache_dtype=torch.bfloat16,
        )
        Dsv4NpuRoPE.apply_rotary_mul_inplace(
            q,
            None,
            cos,
            sin,
            qk_nope_dim=indexer.index_head_dim - indexer.rope_head_dim,
        )
        if not is_npu_arch35():
            return q, None, indexer.head_weights(x).view(-1, indexer.n_heads).float()
        q_packed, q_scale = torch_npu.npu_dynamic_mx_quant(
            q, dst_type=torch_npu.float4_e2m1fn_x2
        )
        q_packed = q_packed.view(torch.uint8).view(t, indexer.n_heads, packed_dim)
        # QLI wants the E8M0 scales observed as [..., 2, 2]; the pool stores the
        # same 4 bytes per row, so both are pure reshapes.
        descale_q = q_scale.view(torch.uint8).view(t, indexer.n_heads, 2, 2)
        w = indexer.head_weights(x).view(-1, indexer.n_heads).float()
        return q_packed, descale_q, w

    def _dsv41_index_full_topk(
        self, layer, x, q_lora, pos, forward_batch, *, candidate: bool = False
    ):
        """Score the compressed history with ``quant_lightning_indexer``.

        With ``candidate`` the same operator call also emits the level-one block
        candidates, and its third and fourth returns stop being empty tensors.
        Recipes: indexer.py L338-L400.
        """
        indexer = layer.indexer
        ratio = layer.compress_ratio
        fm = self.forward_metadata
        pool = self.token_to_kv_pool
        t = int(q_lora.shape[0])
        packed_dim = indexer.index_head_dim // 2
        q_packed, descale_q, w = self._dsv41_index_q(layer, x, q_lora, pos)

        k = pool.get_index_k(layer.layer_id)
        if not is_npu_arch35():
            kwargs = self._dsv41_a3_index_kwargs(forward_batch, ratio)
            if candidate:
                indices, _, blocks, lengths = _dsv41_a3_ops().candidate(
                    q_packed,
                    k,
                    w,
                    self._dsv4_index_topk,
                    candidate_topk_blocks=indexer.candidate_topk_blocks,
                    candidate_block_size=indexer.candidate_block_size,
                    **kwargs,
                )
                # A3 V3 reserves candidate lengths (empty tensor); consumers
                # use -1 padded block IDs, not the A5 length array.
                return indices.view(t, -1), (blocks, lengths)
            indices, _ = _dsv41_a3_ops().indexer(
                q_packed, k, w, self._dsv4_index_topk, **kwargs
            )
            return indices.view(t, -1), None
        k_scale = pool.get_index_scale(layer.layer_id)
        # PA_ND rows are [pages, page_rows, 1, dim]; QLI reads the packed fp4
        # payload and its E8M0 scales as [..., 2, 2], both pure reshapes of the
        # buffers the prologue wrote.
        pages, page_rows = k.shape[0], k.shape[1]
        k = k.view(pages, page_rows, 1, packed_dim)
        descale_k = k_scale.view(torch.uint8).view(pages, page_rows, 1, 2, 2)

        page_table = getattr(fm, f"c{ratio}_index_page_table", None)
        if page_table is None:
            raise RuntimeError(
                f"DeepSeek-V4.1 QLI for layer {layer.layer_id} is missing the "
                "compressed index page table."
            )
        # QLI's seqused_k is the compressed history length and cmp_residual_k is
        # its unfinished-group remainder; recipes computes them as
        # ``actual_seq_k // ratio`` and ``actual_seq_k % ratio`` (indexer.py
        # L359-L366 feeding attention_data.py L907-L918). Derive them from this
        # batch's seq_lens rather than from a metadata tensor: seq_lens is a
        # graph input (refreshed every replay), while a captured metadata buffer
        # would keep the capture-time lengths for every later step.
        seq_lens = forward_batch.seq_lens
        # Same generation-agnostic `index_topk` the QLI metadata op and the graph
        # buffer are sized with (L1336 / L1761); the op both selects and reshapes
        # with it.
        topk = self._dsv4_index_topk
        key = (
            f"c{ratio}a_qli_candidate_metadata"
            if candidate
            else f"c{ratio}a_qli_metadata"
        )
        metadata = fm.kernel_metadata.get(key)
        if metadata is None:
            raise RuntimeError(
                f"DeepSeek-V4.1 QLI for layer {layer.layer_id} has no {key}; the "
                "kernel-metadata builder must run before the source layers."
            )
        result = _ds41_ops().quant_lightning_indexer(
            q=q_packed,
            k=k,
            w=w,
            descale_q=descale_q,
            descale_k=descale_k,
            topk=topk,
            quant_mode=1,
            cu_seqlens_q=fm.actual_seq_lengths_q_pa,
            seqused_q=None,
            seqused_k=(seq_lens // ratio).to(torch.int32),
            cmp_residual_k=(None if ratio == 1 else (seq_lens % ratio).to(torch.int32)),
            block_table=page_table.to(torch.int32),
            max_seqlen_q=-1,
            mask_mode=3,
            cmp_ratio=ratio,
            layout_q="TND",
            layout_k="PA_BBND",
            return_value=False,
            metadata=metadata,
            # -1 leaves the operator's third and fourth returns empty.
            candidate_topk_blocks=indexer.candidate_topk_blocks if candidate else -1,
            candidate_block_size=indexer.candidate_block_size if candidate else -1,
        )
        indices, _, candidate_indices, candidate_lengths = result
        scored = indices.view(t, topk).to(torch.int32)
        if not candidate:
            return scored, None
        return scored, (candidate_indices, candidate_lengths)

    def _dsv41_a3_index_kwargs(self, forward_batch, ratio):
        return dict(
            cu_seqlens_q=self.forward_metadata.actual_seq_lengths_q_pa,
            seqused_k=(forward_batch.seq_lens // ratio).to(torch.int32),
            cmp_residual_k=(forward_batch.seq_lens % ratio).to(torch.int32)
            if ratio > 1
            else None,
            block_table=getattr(self.forward_metadata, f"c{ratio}_index_page_table").to(
                torch.int32
            ),
            layout_q="TND",
            layout_k="PA_BBND",
            mask_mode=3,
            cmp_ratio=ratio,
        )

    def _dsv41_qsli_cache(self, layer) -> torch.Tensor:
        """The combined index-K record QSLI reads: ``[pages, page_size/8, 1, 544]``.

        ``indexer_prologue_k`` in ``storage_mode=1`` packs the same K rows eight
        to a record -- K[8, 64] as 512B, then their e8m0 scales as 32B -- which
        is exactly the PA_BBND K layout QSLI takes
        (quant_sparse_lightning_indexer_dsl/README.md L12-L29). Only the
        candidate source layer writes it, so it is allocated per (ratio, source
        slot) on first use instead of for every layer of the ratio; the consumer
        layers reach the source's buffer through the same ``layer_mapping`` slot.
        """
        pool = self.token_to_kv_pool
        item = pool.layer_mapping[layer.layer_id]
        key = (item.compress_ratio, item.compress_layer_id)
        cache = self._dsv41_qsli_caches.get(key)
        if cache is None:
            index_k = pool.get_index_k(layer.layer_id)
            block = self._dsv41_candidate_block_size
            pages, page_rows = int(index_k.shape[0]), int(index_k.shape[1])
            if block <= 0 or page_rows % block:
                raise RuntimeError(
                    "DeepSeek-V4.1 QSLI needs candidate_block_size to divide the "
                    f"compressed page rows, got block={block} rows={page_rows}."
                )
            head_dim = int(index_k.shape[-1]) * 2
            # 32 = the prologue's e4m3 scale group (indexer_prologue_k_dsl
            # README L40-L52: 64 payload bytes + 4 scale bytes per 128-dim row).
            record = block * (head_dim // 2 + head_dim // 32)
            cache = torch.zeros(
                pages,
                page_rows // block,
                1,
                record,
                dtype=torch.uint8,
                device=index_k.device,
            )
            self._dsv41_qsli_caches[key] = cache
        return cache

    def _dsv41_index_qsli_topk(self, layer, x, q_lora, pos, forward_batch):
        """Level two: rescore the published candidate blocks with this layer's Q.

        The candidates are logical ``candidate_block_size`` blocks of compressed
        tokens; the operator expands them itself, so the block ids pass through
        unchanged. Recipes: indexer.py L402-L453.
        """
        indexer = layer.indexer
        ratio = layer.compress_ratio
        fm = self.forward_metadata
        t = int(q_lora.shape[0])
        blocks = getattr(fm, "dsv41_candidate_blocks", None)
        if blocks is None:
            raise RuntimeError(
                f"DeepSeek-V4.1 QSLI for layer {layer.layer_id} has no published "
                "candidate blocks; the candidate source layer must run before "
                "its consumers."
            )
        candidate_indices, candidate_lengths = blocks
        if candidate_indices.shape[0] != t:
            raise RuntimeError(
                "DeepSeek-V4.1 QSLI candidate blocks are stale: "
                f"{candidate_indices.shape[0]} rows published, {t} needed."
            )
        q_packed, descale_q, w = self._dsv41_index_q(layer, x, q_lora, pos)
        if not is_npu_arch35():
            indices, _ = _dsv41_a3_ops().sparse_indexer(
                q_packed,
                self.token_to_kv_pool.get_index_k(layer.layer_id),
                w,
                self._dsv4_index_topk,
                candidate_indices,
                candidate_block_size=indexer.candidate_block_size,
                **self._dsv41_a3_index_kwargs(forward_batch, ratio),
            )
            return indices.view(t, -1)
        page_table = getattr(fm, f"c{ratio}_index_page_table", None)
        if page_table is None:
            raise RuntimeError(
                f"DeepSeek-V4.1 QSLI for layer {layer.layer_id} is missing the "
                "compressed index page table."
            )
        metadata = fm.kernel_metadata.get(f"c{ratio}a_qsli_metadata")
        if metadata is None:
            raise RuntimeError(
                f"DeepSeek-V4.1 QSLI for layer {layer.layer_id} has no "
                f"c{ratio}a_qsli_metadata; the kernel-metadata builder must run "
                "before the index layers."
            )
        topk = self._dsv4_index_topk
        qsli_k = self._dsv41_qsli_cache(layer)
        indices, _ = _ds41_ops().quant_sparse_lightning_indexer(
            q_packed,
            # QSLI takes the record rows without the singleton head axis.
            qsli_k.squeeze(-2),
            w,
            descale_q,
            candidate_indices,
            candidate_lengths,
            topk,
            1,
            indexer.candidate_block_size,
            descale_k=None,
            cu_seqlens_q=fm.actual_seq_lengths_q_pa,
            seqused_q=None,
            seqused_k=(forward_batch.seq_lens // ratio).to(torch.int32),
            # Only mask_mode=3 with cmp_ratio != 1 needs it, and the consumers
            # are ratio-1 layers (README "公共可选参数").
            cmp_residual_k=None,
            block_table=page_table.to(torch.int32),
            output_idx_offset=None,
            metadata=metadata,
            max_seqlen_q=-1,
            mask_mode=3,
            cmp_ratio=ratio,
            layout_q="TND",
            layout_k="PA_BBND",
            return_value=False,
        )
        scored = indices.view(t, topk).to(torch.int32)
        if os.environ.get("SGLANG_DSV41_DEBUG_QSLI"):
            # What MQSMLA will be told: exactly the helper's buffer (position
            # derived, dead rows zeroed).  Any -1 of `scored` inside that prefix
            # is the wild-address violation -- the CPU golden raises on it
            # (mqsmla_golden.py L215) and the kernel does not skip it -- so print
            # every occurrence, not just the first call per layer.
            lens = (
                self._dsv41_cmp_topk_length(forward_batch, ratio)
                .reshape(-1)
                .to(torch.long)
            )
            in_prefix = torch.arange(topk, device=scored.device) < lens.unsqueeze(1)
            neg_in = int(((scored < 0) & in_prefix).sum())
            cov = (
                candidate_lengths.reshape(-1).to(torch.long)[:t]
                * int(indexer.candidate_block_size)
            ).clamp_max(int(topk))
            short = bool((cov < lens).any())
            key = (layer.layer_id, "qsli")
            if neg_in or short or key not in self._dsv41_debugged:
                self._dsv41_debugged.add(key)
                print(
                    f"[dsv41][{self._dsv41_rank_tag()}] qsli layer={layer.layer_id} "
                    f"ratio={ratio} rows={t} "
                    f"neg_in_prefix={neg_in} len_max={int(lens.max())} "
                    f"vis_max={int((forward_batch.seq_lens // ratio).max())} "
                    f"cov_max={int(cov.max())} "
                    f"cand={tuple(candidate_indices.shape)} "
                    f"block_len={int(candidate_lengths.min())}/"
                    f"{int(candidate_lengths.max())} "
                    f"scored_min={int(scored.min())} "
                    f"scored_max={int(scored.max())} "
                    f"k={tuple(qsli_k.shape)} "
                    f"pt={tuple(page_table.shape)}/{int(page_table.max())}",
                    flush=True,
                )
        return scored

    def _dsv41_index_kernel_metadata(self, forward_batch: ForwardBatch) -> dict:
        """QLI dispatch tables, one per low ratio (the ``[1024]`` int32 tensor the
        compute entry point requires; QLI README "Metadata 参数与返回值").

        Recipes: indexer.py L368-L394 reads
        ``attn_metadata["kernel_metadata"][f"c{ratio}a_qli_metadata"]``. Built from
        ``forward_batch.seq_lens`` directly so it does not depend on the
        compress-metadata pass running first.
        """
        if not is_npu_arch35():
            # arch22 LI V2/V3 and SLI do not consume a metadata tensor.
            return {}
        fm = self.forward_metadata
        seq_lens = forward_batch.seq_lens.to(torch.int64)
        cu_seqlens_q = fm.actual_seq_lengths_q_pa
        if cu_seqlens_q is None:
            return {}
        ds41 = _ds41_ops()
        out: dict = {}

        def keep(key: str, metadata: torch.Tensor) -> None:
            # aclgraph binds the address at capture: refresh in place.
            buf = fm.kernel_metadata.get(key)
            if buf is not None and buf.shape == metadata.shape:
                buf.copy_(metadata)
                metadata = buf
            out[key] = metadata

        batch_size = int(forward_batch.batch_size)
        for ratio in self._dsv41_low_ratios():
            cmp_lens = (seq_lens // ratio).to(torch.int32)
            keep(
                f"c{ratio}a_qli_metadata",
                ds41.quant_lightning_indexer_metadata(
                    cu_seqlens_q=cu_seqlens_q,
                    seqused_q=None,
                    seqused_k=cmp_lens,
                    # mask_mode=3 + cmp_ratio != 1 requires the residual (README).
                    cmp_residual_k=(
                        None if ratio == 1 else (seq_lens % ratio).to(torch.int32)
                    ),
                    batch_size=batch_size,
                    num_heads_q=self._dsv4_index_n_heads,
                    num_heads_k=1,
                    head_dim=self._dsv4_index_head_dim,
                    topk=self._dsv4_index_topk,
                    mask_mode=3,
                    cmp_ratio=ratio,
                    layout_q="TND",
                    layout_k="PA_BBND",
                ),
            )
        candidate_ratio = self._dsv41_candidate_ratio()
        if candidate_ratio is not None:
            block = self._dsv41_candidate_block_size
            if block <= 0 or self._dsv41_candidate_topk_blocks <= 0:
                raise RuntimeError(
                    "DeepSeek-V4.1 has a candidate source layer but no "
                    f"candidate_topk_blocks/candidate_block_size "
                    f"({self._dsv41_candidate_topk_blocks}/{block})."
                )
            cmp_lens = (seq_lens // candidate_ratio).to(torch.int32)
            # Level one publishes blocks, so its dispatch table is a separate
            # instance from the plain per-ratio one (recipes indexer.py L388-L393).
            keep(
                f"c{candidate_ratio}a_qli_candidate_metadata",
                ds41.quant_lightning_indexer_metadata(
                    cu_seqlens_q=cu_seqlens_q,
                    seqused_q=None,
                    seqused_k=cmp_lens,
                    cmp_residual_k=(
                        None
                        if candidate_ratio == 1
                        else (seq_lens % candidate_ratio).to(torch.int32)
                    ),
                    batch_size=batch_size,
                    num_heads_q=self._dsv4_index_n_heads,
                    num_heads_k=1,
                    head_dim=self._dsv4_index_head_dim,
                    topk=self._dsv4_index_topk,
                    mask_mode=3,
                    cmp_ratio=candidate_ratio,
                    layout_q="TND",
                    layout_k="PA_BBND",
                    candidate_topk_blocks=self._dsv41_candidate_topk_blocks,
                    candidate_block_size=block,
                ),
            )
        if candidate_ratio is not None and self._dsv41_has_candidate_consumers():
            # QSLI only needs a scheduling bound for the candidate prefix, so the
            # length is the visible block count per row, computed the way the
            # reference does it (modeling_deepseek.py L3241-L3245).
            visible_blocks = (
                torch.div(
                    forward_batch.positions.reshape(-1) + candidate_ratio * block,
                    candidate_ratio * block,
                    rounding_mode="floor",
                )
                .clamp_min(1)
                .clamp_max(self._dsv41_candidate_topk_blocks)
            )
            keep(
                f"c{candidate_ratio}a_qsli_metadata",
                ds41.quant_sparse_lightning_indexer_metadata(
                    visible_blocks.to(torch.int32).view(-1, 1),
                    cu_seqlens_q=cu_seqlens_q,
                    seqused_q=None,
                    seqused_k=(seq_lens // candidate_ratio).to(torch.int32),
                    cmp_residual_k=None,
                    batch_size=batch_size,
                    num_heads_q=self._dsv4_index_n_heads,
                    num_heads_k=1,
                    head_dim=self._dsv4_index_head_dim,
                    topk=self._dsv4_index_topk,
                    quant_mode=1,
                    candidate_block_size=block,
                    mask_mode=3,
                    cmp_ratio=candidate_ratio,
                    layout_q="TND",
                    layout_k="PA_BBND",
                ),
            )
        return out

    # ---- Seam D: mixed-quant sparse MLA ---------------------------------------
    #
    # DeepSeek-V4.1 runs a *different* attention operator from V4's
    # npu_kv_quant_sparse_attn_sharedkv: ``ds41.mixed_quant_sparse_flash_mla``,
    # which takes the window/compressed sides purely as explicit sparse indices +
    # topk lengths -- it has no mask, window or compress-ratio parameters at all
    # (mixed_quant_sparse_flash_mla_dsl/README.md L208).  Pool row widths and
    # group sizes line up with the pools built in dsv4_memory_pool.py:
    # ori 544B = FP8E4M3 group32 + BF16 scales, cmp 320B = FP4E2M1 group16 +
    # BF16 scales (README L185-L192).

    def _dsv41_window_indices(self, forward_batch: ForwardBatch):
        """Explicit causal sliding-window indices + lengths for the window side.

        Recipes: attention_data.py L333-L352 (``generate_win_topk_ids``) and
        L614-L618.  Both are pure device arithmetic, so this is aclgraph-safe;
        the results are written into stable buffers because the captured graph
        holds their addresses.
        """
        fm = self.forward_metadata
        positions = forward_batch.positions.reshape(-1)
        t = int(positions.numel())
        topk = int(self._dsv4_sliding_window_size)
        indices = getattr(fm, "dsv41_win_sparse_indices", None)
        lengths = getattr(fm, "dsv41_win_topk_length", None)
        if indices is None or indices.shape[0] != t:
            indices = torch.empty(
                (t, 1, topk), dtype=torch.int32, device=positions.device
            )
            lengths = torch.empty((t, 1), dtype=torch.int32, device=positions.device)
            setattr(fm, "dsv41_win_sparse_indices", indices)
            setattr(fm, "dsv41_win_topk_length", lengths)
        offsets = torch.arange(topk, dtype=positions.dtype, device=positions.device)
        valid_len = torch.clamp(positions + 1, max=topk)
        # Dead rows (batch padding / idle slots: seq_lens == 0, positions == 0)
        # have no KV at all, but (positions + 1) claims one window token.  The
        # indexer scores nothing for them and fills the row with -1, so the
        # claimed prefix would feed MQSMLA an in-prefix -1 (contract violation,
        # mqsmla_golden.py L215).  Zero the length instead: README L211 makes a
        # zero length skip the side and emit 0.
        seq_lens = forward_batch.seq_lens
        if seq_lens is not None:
            seq_lens = seq_lens.reshape(-1)
            if seq_lens.numel() == t:
                valid_len = torch.where(
                    seq_lens > 0, valid_len, torch.zeros_like(valid_len)
                )
        window_start = positions - valid_len + 1
        ids = window_start.unsqueeze(1) + offsets.unsqueeze(0)
        valid = offsets.unsqueeze(0) < valid_len.unsqueeze(1)
        ids = torch.where(valid, ids, ids.new_full((), -1))
        indices.copy_(ids.unsqueeze(1).to(torch.int32))
        lengths.copy_(valid_len.to(torch.int32).unsqueeze(1))
        return indices, lengths

    def _dsv41_cmp_topk_length(self, forward_batch: ForwardBatch, ratio: int):
        """Per-query valid compressed top-k length, ``min((p+1)//ratio, index_topk)``.

        Recipes: attention_data.py L891-L904.
        """
        fm = self.forward_metadata
        positions = forward_batch.positions.reshape(-1)
        t = int(positions.numel())
        buf = getattr(fm, f"dsv41_cmp_topk_length_c{ratio}", None)
        if buf is None or buf.shape[0] != t:
            buf = torch.empty((t, 1), dtype=torch.int32, device=positions.device)
            setattr(fm, f"dsv41_cmp_topk_length_c{ratio}", buf)
        lens = ((positions + 1) // ratio).clamp_max(int(self._dsv4_index_topk))
        # Same dead-row rule as _dsv41_window_indices: a seq_lens == 0 row has
        # no compressed tokens, so its length must be 0, not (0 + 1) // ratio.
        seq_lens = forward_batch.seq_lens
        if seq_lens is not None:
            seq_lens = seq_lens.reshape(-1)
            if seq_lens.numel() == t:
                lens = torch.where(seq_lens > 0, lens, torch.zeros_like(lens))
        buf.copy_(lens.to(torch.int32).view(-1, 1))
        return buf

    def _dsv41_attention_kernel_metadata(self, forward_batch: ForwardBatch) -> dict:
        """MQSMLA dispatch tables: ``win`` (ratio 0) plus ``c{r}a`` (ratio 1/2).

        Recipes: modeling_deepseek.py L3154-L3176.
        """
        fm = self.forward_metadata
        if not is_npu_arch35():
            out = {}
            for ratio in (0, *self._dsv41_low_ratios()):
                key = "win_metadata" if ratio == 0 else f"c{ratio}a_metadata"
                metadata = _dsv41_a3_ops().metadata(
                    self._dsv4_q_head_num,
                    1,
                    self._dsv4_head_dim,
                    batch_size=forward_batch.batch_size,
                    has_ori_kv=True,
                    has_cmp_kv=ratio > 0,
                    cmp_topk=self._dsv4_index_topk if ratio else 0,
                    **self._dsv41_a3_attention_kwargs(forward_batch, ratio),
                )
                buf = fm.kernel_metadata.get(key)
                if buf is not None:
                    buf.copy_(metadata)
                    metadata = buf
                out[key] = metadata
            return out
        _, win_lengths = self._dsv41_window_indices(forward_batch)
        ds41 = _ds41_ops()
        cu_seqlens_q = fm.actual_seq_lengths_q_pa
        head_num = int(self._dsv4_q_head_num)
        # Take the width from the backend's own config, like the graph path
        # (``_init_dsv41_graph_metadata``) and the V4 eager path
        # (``_compute_kernel_metadata``).  The DSV4 pool exposes
        # ``qk_nope_head_dim``/``qk_rope_head_dim`` but has no ``head_dim``
        # attribute, so reading it off the pool is an AttributeError.
        head_dim = int(self._dsv4_head_dim)
        out = {}
        for ratio in (0, *self._dsv41_low_ratios()):
            attention_type = "win" if ratio == 0 else f"c{ratio}a"
            if ratio == 0:
                cmp_lengths = torch.zeros_like(win_lengths)
            else:
                cmp_lengths = self._dsv41_cmp_topk_length(forward_batch, ratio)
            key = f"{attention_type}_metadata"
            metadata = ds41.mixed_quant_sparse_flash_mla_metadata(
                win_lengths,
                cmp_lengths,
                cu_seqlens_q=cu_seqlens_q,
                num_heads_q=head_num,
                num_heads_kv=1,
                head_dim=head_dim,
                quant_mode=1,
                layout_q="TND",
                layout_kv="PA_BBND",
                has_ori_kv=True,
                has_cmp_kv=ratio > 0,
            )
            # aclgraph binds the tensor address at capture, so refresh the
            # graph-allocated buffer in place instead of rebinding.
            buf = fm.kernel_metadata.get(key)
            if buf is not None and buf.shape == metadata.shape:
                buf.copy_(metadata)
                metadata = buf
            out[key] = metadata
        return out

    def _dsv41_a3_attention_kwargs(self, forward_batch, ratio, *, metadata=None):
        # Unlike MQSMLA, the BF16 A3 kernel derives causal windows and the
        # compressed prefix from lengths, masks and the unfinished C2 pair.
        seq_lens = forward_batch.seq_lens.to(torch.int32)
        fm = self.forward_metadata if metadata is None else metadata
        return dict(
            cu_seqlens_q=fm.actual_seq_lengths_q_pa,
            seqused_ori_kv=seq_lens,
            seqused_cmp_kv=seq_lens // ratio if ratio else None,
            cmp_residual_kv=seq_lens % ratio if ratio > 1 else None,
            cmp_ratio=ratio or 1,
            ori_mask_mode=4,
            cmp_mask_mode=3 if ratio else 0,
            ori_win_left=self._dsv4_sliding_window_size - 1,
            ori_win_right=0,
            layout_q="TND",
            layout_kv="PA_BBND",
        )

    def _dsv41_attention_sinks(self, attn_sink, device, head_num: int):
        """MQSMLA always needs a float32 ``(N1,)`` sink (README L110/L219)."""
        if attn_sink is None:
            cached = getattr(self, "_dsv41_zero_sinks", None)
            if cached is None or cached.numel() != head_num:
                cached = torch.zeros(head_num, dtype=torch.float32, device=device)
                setattr(self, "_dsv41_zero_sinks", cached)
            return cached
        return attn_sink.reshape(-1).to(torch.float32)

    def _dsv41_rank_tag(self) -> str:
        """Per-rank tag for every [dsv41] debug line.

        Multi-rank logs interleave, so each line carries its origin; without it
        the same step number from two DP ranks looks like time travel.
        """
        cached = getattr(self, "_dsv41_rank_tag_cache", None)
        if cached is None:
            try:
                p = get_parallel()
                cached = (
                    f"dp{getattr(p, 'attn_dp_rank', '?')}"
                    f".tp{getattr(p, 'tp_rank', '?')}"
                )
            except Exception:
                cached = "dp?.tp?"
            self._dsv41_rank_tag_cache = cached
        return cached

    def _dsv41_regenerate_smla_metadata(self, kw: dict):
        """Re-run the metadata op on the exact tensors this SMLA call receives.

        The two-stage contract says the dispatch table must be derived from this
        step's lengths; a table that differs from a fresh regeneration is a
        stale buffer on the sglang side, while an equal-but-broken table points
        at the operator.  Debug tooling only: never mask the underlying fault.
        """
        try:
            ori_len = kw["ori_topk_length"]
            has_cmp = "cmp_topk_length" in kw
            cmp_len = kw["cmp_topk_length"] if has_cmp else torch.zeros_like(ori_len)
            return _ds41_ops().mixed_quant_sparse_flash_mla_metadata(
                ori_len,
                cmp_len,
                cu_seqlens_q=kw.get("cu_seqlens_q"),
                num_heads_q=int(self._dsv4_q_head_num),
                num_heads_kv=1,
                head_dim=int(self._dsv4_head_dim),
                quant_mode=1,
                layout_q="TND",
                layout_kv="PA_BBND",
                has_ori_kv=True,
                has_cmp_kv=has_cmp,
            )
        except Exception as err:
            print(
                f"[dsv41][{self._dsv41_rank_tag()}] metadata regeneration "
                f"failed: {err}",
                flush=True,
            )
            return None

    def _dsv41_audit_smla(self, layer, q, forward_batch, kw, ratio: int) -> None:
        """Row-accurate MQSMLA contract audit.

        The device contract (mixed_quant_sparse_flash_mla_dsl/README.md L209-L214,
        enforced by the CPU golden mqsmla_golden.py L211-L216) is: for every query
        row, every sparse index inside the ``topk_length`` prefix must be a valid
        logical token index of that row's batch, and must map through
        ``block_table[b, idx // block_size]`` to a column inside the table and a
        page inside the pool.  The kernel does no bounds checking, so any
        violation is exactly the wild GM address of the aicore fault.  Every
        check is per-row and names the offending rows, so the last VIOLATION
        line before the async fault pinpoints the layer / side / row that fed
        the bad address.  A run that crashes with no VIOLATION line (but with
        the cfg/snapshot lines present) moves the suspicion off the values and
        onto the operator/dispatch internals.
        """
        phase = "decode" if forward_batch.forward_mode.is_decode() else "extend"
        step = getattr(self, "_dsv41_audit_step", 0) + 1
        self._dsv41_audit_step = step
        if ("cfg",) not in self._dsv41_debugged:
            self._dsv41_debugged.add(("cfg",))
            print(
                f"[dsv41][{self._dsv41_rank_tag()}] cfg "
                f"index_topk={self._dsv4_index_topk} "
                f"sliding_window={self._dsv4_sliding_window_size} "
                f"candidate_topk_blocks={self._dsv41_candidate_topk_blocks} "
                f"candidate_block_size={self._dsv41_candidate_block_size} "
                f"low_ratios={self._dsv41_low_ratios()}",
                flush=True,
            )
        rows = int(q.shape[0])
        positions = forward_batch.positions.reshape(-1).to(torch.long)
        seq_lens = forward_batch.seq_lens.reshape(-1).to(torch.long)
        device = positions.device
        cu = kw.get("cu_seqlens_q")
        problems = []
        if rows != int(positions.numel()):
            problems.append(f"q rows {rows} != positions {int(positions.numel())}")
        if cu is not None and int(cu[-1]) != rows:
            problems.append(f"cu span {int(cu[-1])} != q rows {rows}")
        # row -> batch map (decode: one token per request; extend: cu bucketing)
        if cu is not None and rows == int(cu[-1]) and rows != cu.numel() - 1:
            row_b = torch.repeat_interleave(
                torch.arange(cu.numel() - 1, device=device),
                (cu[1:] - cu[:-1]).to(torch.long),
            )
        else:
            row_b = torch.arange(rows, device=device)
        # The dispatch table's padded tail is NOT a valid check target: healthy
        # single-curl runs show the same garbage there, so the wheel pads the
        # (2*blocks,) table to a fixed 1024 and leaves the padding
        # uninitialized while the kernel only consumes the head region
        # (blockDim=32 in the fault dumps).  Compare only the head against a
        # fresh regeneration, and dump the layout once so the real format is
        # on record.
        md = kw.get("metadata")
        if md is not None and md.numel() % 2 == 0:
            half = md.shape[0] // 2
            fresh = self._dsv41_regenerate_smla_metadata(kw)
            if fresh is not None and fresh.shape == md.shape:
                if not bool(torch.equal(md[:64], fresh[:64])):
                    problems.append(
                        "metadata head differs from a fresh regeneration: "
                        f"buf={md[:8].tolist()} fresh={fresh[:8].tolist()}"
                    )
                if ("md_layout",) not in self._dsv41_debugged:
                    self._dsv41_debugged.add(("md_layout",))
                    diff = (md != fresh).nonzero().flatten()
                    rt = self._dsv41_rank_tag()
                    print(
                        f"[dsv41][{rt}] md_layout rows={rows} blocks={half} "
                        f"buf[:8]={md[:8].tolist()} "
                        f"fresh[:8]={fresh[:8].tolist()}\n"
                        f"[dsv41][{rt}]   buf_half_head={md[half : half + 8].tolist()} "
                        f"fresh_half_head={fresh[half : half + 8].tolist()} "
                        f"buf_tail_sum={int(md[half:].sum())} "
                        f"fresh_tail_sum={int(fresh[half:].sum())} "
                        f"first_diff={int(diff[0]) if diff.numel() else -1} "
                        f"n_diff={int(diff.numel())}",
                        flush=True,
                    )
        for prefix in ("ori", "cmp"):
            kv = kw.get(f"{prefix}_kv")
            bt = kw.get(f"{prefix}_block_table")
            idx_t = kw.get(f"{prefix}_sparse_indices")
            lens_t = kw.get(f"{prefix}_topk_length")
            if kv is None or bt is None or idx_t is None or lens_t is None:
                continue
            # Per-row visible length: extend rows outnumber batches, so the
            # batch-level lengths must be gathered through the row->batch map.
            vis_batch = seq_lens if prefix == "ori" else seq_lens // max(1, ratio)
            vis = vis_batch[row_b]
            idx = idx_t.reshape(rows, -1).to(torch.long)
            lens = lens_t.reshape(-1).to(torch.long)
            k = idx.shape[-1]
            block = max(1, int(kv.shape[1]))
            width = int(bt.shape[1])
            pages = int(kv.shape[0])
            in_prefix = torch.arange(k, device=device) < lens.unsqueeze(1)
            col = torch.div(idx, block, rounding_mode="floor")
            page = bt.to(torch.long)[row_b.unsqueeze(1), col.clamp(0, width - 1)]
            checks = (
                # golden rule: the prefix must be exact -- a -1 inside it is
                # consumed as a logical token index and becomes a wild address
                ("neg_in_prefix", (idx < 0) & in_prefix),
                # in-prefix index at/after the row's visible logical length
                ("idx_ge_visible", (idx >= vis.clamp_min(1).unsqueeze(1)) & in_prefix),
                # in-prefix index maps to a column outside the block table
                ("bt_col_oob", (col >= width) & in_prefix),
                # the page at that column is outside the pool (or negative)
                ("page_oob", ((page < 0) | (page >= pages)) & in_prefix),
                # outside the prefix only the -1 filler is allowed
                ("garbage_tail", idx < -1),
            )
            for name, mask in checks:
                n = int(mask.sum())
                if not n:
                    continue
                r0 = int(mask.any(dim=1).nonzero()[0, 0])
                j = mask[r0].nonzero()[:4, 0].tolist()
                problems.append(
                    f"{prefix}.{name} n={n} first_row={r0} "
                    f"(b={int(row_b[r0])} pos={int(positions[r0])} "
                    f"seq={int(seq_lens[row_b[r0]])} len={int(lens[r0])} "
                    f"bt={width}x pages={pages} block={block}) "
                    + " ".join(
                        f"[j={a} idx={int(idx[r0, a])} col={int(idx[r0, a]) // block} "
                        f"page={int(page[r0, a])}]"
                        for a in j
                    )
                )
            # window-side staleness: the prefix must end at the row position.
            # Dead rows (length 0) are skipped; their gathered tail is filler.
            if prefix == "ori" and rows == int(positions.numel()):
                last = idx.gather(1, (lens - 1).clamp_min(0).unsqueeze(1)).squeeze(1)
                stale = (lens > 0) & (last != positions)
                if bool(stale.any()):
                    bad = stale.nonzero()[:8, 0].tolist()
                    problems.append(
                        f"ori.window_stale rows={bad} last={last[bad].tolist()} "
                        f"pos={positions[bad].tolist()}"
                    )
            # direct test of the QSLI prefix hypothesis: the length passed to
            # MQSMLA must not exceed the candidate tokens QSLI could fill
            if prefix == "cmp":
                cand = getattr(self.forward_metadata, "dsv41_candidate_blocks", None)
                if cand is not None and self._dsv41_candidate_block_size > 0:
                    cov = (
                        cand[1].reshape(-1).to(torch.long)[:rows]
                        * self._dsv41_candidate_block_size
                    ).clamp_max(self._dsv4_index_topk)
                    over = lens > cov
                    if bool(over.any()):
                        r = over.nonzero()[:8, 0].tolist()
                        problems.append(
                            f"cmp.len_gt_candidate rows={r} "
                            f"len={lens[r].tolist()} cov={cov[r].tolist()} "
                            f"(QSLI pads -1 inside this prefix)"
                        )
        if problems:
            print(
                f"[dsv41][{self._dsv41_rank_tag()}] VIOLATION step={step} "
                f"layer={layer.layer_id} ratio={ratio} "
                f"phase={phase} bs={forward_batch.batch_size} rows={rows}: "
                + " | ".join(problems),
                flush=True,
            )
        key = (layer.layer_id, ratio, phase)
        if key not in self._dsv41_debugged:
            self._dsv41_debugged.add(key)
            shapes = {n: tuple(v.shape) for n, v in kw.items() if torch.is_tensor(v)}
            rt = self._dsv41_rank_tag()
            print(
                f"[dsv41][{rt}] AUDIT snapshot step={step} layer={layer.layer_id} "
                f"ratio={ratio} phase={phase} bs={forward_batch.batch_size} "
                f"q={tuple(q.shape)}\n"
                f"[dsv41][{rt}]   {shapes}\n"
                f"[dsv41][{rt}]   pos=[{int(positions.min())},{int(positions.max())}] "
                f"seq=[{int(seq_lens.min())},{int(seq_lens.max())}] "
                f"dead={int((seq_lens <= 0).sum())}\n"
                f"[dsv41][{rt}]   ori_len_max={int(kw['ori_topk_length'].max())} "
                f"ori_idx_max={int(kw['ori_sparse_indices'].max())} "
                f"ori_bt={tuple(kw['ori_block_table'].shape)} "
                f"ori_bt_max={int(kw['ori_block_table'].max())} "
                f"ori_pages={kw['ori_kv'].shape[0]}"
                + (
                    f"\n[dsv41][{rt}]   cmp_len_max={int(kw['cmp_topk_length'].max())} "
                    f"cmp_idx_max={int(kw['cmp_sparse_indices'].max())} "
                    f"cmp_bt={tuple(kw['cmp_block_table'].shape)} "
                    f"cmp_bt_max={int(kw['cmp_block_table'].max())} "
                    f"cmp_pages={kw['cmp_kv'].shape[0]}"
                    if "cmp_kv" in kw
                    else ""
                ),
                flush=True,
            )

    def _forward_low_ratio_attention(
        self,
        q: torch.Tensor,
        layer,
        forward_batch: ForwardBatch,
        attn_sink: Optional[torch.Tensor],
        compress_ratio: int,
    ) -> torch.Tensor:
        """The V4.1 attention body: window side always, compressed side when
        ``compress_ratio > 0``.

        Recipes: modeling_deepseek.py L1189-L1238 (``prepare_fa_kwargs``).
        """
        fm = self.forward_metadata
        pool = self.token_to_kv_pool
        metadata = fm.kernel_metadata.get(
            "win_metadata" if compress_ratio == 0 else f"c{compress_ratio}a_metadata"
        )
        if metadata is None:
            raise RuntimeError(
                "DeepSeek-V4.1 attention metadata missing for "
                f"layer_id={layer.layer_id} compress_ratio={compress_ratio}; "
                f"available: {list(fm.kernel_metadata.keys())}."
            )
        if not is_npu_arch35():
            kwargs = self._dsv41_a3_attention_kwargs(forward_batch, compress_ratio)
            if compress_ratio:
                topk = getattr(fm, f"c{compress_ratio}_topk_indices")
                kwargs.update(
                    cmp_kv=pool.get_compress_buffer(layer.layer_id, False),
                    cmp_sparse_indices=topk.view(-1, 1, self._dsv4_index_topk),
                    cmp_block_table=getattr(fm, f"c{compress_ratio}_page_table").to(
                        torch.int32
                    ),
                )
            attn_out, _ = _dsv41_a3_ops().attention(
                q,
                ori_kv=pool.get_swa_buffer(layer.layer_id),
                ori_block_table=fm.swa_page_table.to(torch.int32),
                sinks=None if attn_sink is None else attn_sink.reshape(-1).float(),
                metadata=metadata,
                softmax_scale=layer.scaling,
                **kwargs,
            )
            # Padded graph rows have no real cache slot, even if an operator
            # leaves the result of a zero-length request unspecified.
            live = forward_batch.out_cache_loc.reshape(-1) >= self.page_size
            return attn_out.masked_fill(~live[:, None, None], 0)
        win_indices, win_lengths = self._dsv41_window_indices(forward_batch)
        attn_kwargs = dict(
            ori_kv=pool.get_swa_buffer(layer.layer_id).view(torch.uint8),
            ori_sparse_indices=win_indices,
            ori_topk_length=win_lengths,
            ori_block_table=fm.swa_page_table.to(torch.int32),
            cu_seqlens_q=fm.actual_seq_lengths_q_pa,
            seqused_q=None,
            # Reserved by the operator (README L106): accepted, not consumed.
            seqused_ori_kv=fm.actual_seq_lengths_kv,
            sinks=self._dsv41_attention_sinks(
                attn_sink, q.device, int(self._dsv4_q_head_num)
            ),
            metadata=metadata,
            quant_mode=1,
            # ``layer`` here is the ``RadixAttention`` the model passes to
            # ``forward`` (``self.attn_mqa``), *not* the ``MQALayer`` that
            # ``forward_low_ratio_sources`` receives.  ``scaling`` is the
            # RadixAttention attribute holding ``MQALayer.softmax_scale``
            # (= head_dim**-0.5, deepseek_v4.py L1374-1377); the MQALayer spells
            # the same number ``softmax_scale``.  Do not "fix" this to
            # ``layer.softmax_scale``: that would be the AttributeError.
            softmax_scale=layer.scaling,
            layout_q="TND",
            layout_kv="PA_BBND",
            return_softmax_lse=False,
        )
        if compress_ratio > 0:
            topk = getattr(fm, f"c{compress_ratio}_topk_indices", None)
            if topk is None:
                raise RuntimeError(
                    f"DeepSeek-V4.1 ratio-{compress_ratio} attention has no "
                    f"c{compress_ratio}_topk_indices; the indexer source layer "
                    "must run before this layer."
                )
            # Reserved by the operator (README L107); derived like the rest so
            # the value is live rather than capture-time.
            compressed_lens = (forward_batch.seq_lens // compress_ratio).to(torch.int32)
            cmp_kv = pool.get_compress_buffer(layer.layer_id, False).view(torch.uint8)
            cmp_bt = getattr(fm, f"c{compress_ratio}_page_table").to(torch.int32)
            if (
                os.environ.get("SGLANG_DSV41_DEBUG_QSLI")
                and ("smla", layer.layer_id) not in self._dsv41_debugged
            ):
                self._dsv41_debugged.add(("smla", layer.layer_id))
                rows_per_page = int(cmp_kv.shape[1]) if cmp_kv.dim() > 1 else 1
                need_cols = int((compressed_lens.max() - 1) // rows_per_page) + 1
                # Per-row page maxima: a row above the pool size is the row the
                # kernel turns into a wild address, and a -1 is the sentinel
                # leaking out of the write path.
                row_max = cmp_bt.max(dim=1).values[:16].tolist()
                rt = self._dsv41_rank_tag()
                print(
                    f"[dsv41][{rt}] smla layer={layer.layer_id} ratio={compress_ratio}\n"
                    f"[dsv41][{rt}]   cmp_kv={tuple(cmp_kv.shape)} rows_per_page={rows_per_page}\n"
                    f"[dsv41][{rt}]   bt={tuple(cmp_bt.shape)} bt_max={int(cmp_bt.max())} "
                    f"need_cols={need_cols} bt_cols_ok="
                    f"{cmp_bt.shape[1] >= need_cols}\n"
                    f"[dsv41][{rt}]   page_max_ok={int(cmp_bt.max()) < cmp_kv.shape[0]} "
                    f"negs={int((cmp_bt < 0).sum())}\n"
                    f"[dsv41][{rt}]   row_max={row_max}\n"
                    f"[dsv41][{rt}]   visible={compressed_lens[:16].tolist()}\n"
                    f"[dsv41][{rt}]   cmp_topk={tuple(topk.shape)} "
                    f"topk_max={int(topk.max())} topk_neg={int((topk < 0).sum())}",
                    flush=True,
                )
            attn_kwargs.update(
                cmp_kv=cmp_kv,
                cmp_sparse_indices=topk.view(-1, 1, topk.shape[-1]).to(torch.int32),
                cmp_block_table=cmp_bt,
                # seqused_cmp_kv is reserved by the operator (README L107).
                seqused_cmp_kv=compressed_lens,
                # One length per query row, from the position, capped by
                # index_topk (recipes attention_data.py L892-L905). The reference
                # uses this same upper bound for the QSLI layers too, so the -1
                # suffix inside it is expected to be skipped.
                cmp_topk_length=self._dsv41_cmp_topk_length(
                    forward_batch, compress_ratio
                ),
            )
        # Audit every ratio (0 included): a win-only layer can fault through the
        # ori side too, and the audit must name the last call before the async
        # aicore fault.
        if os.environ.get("SGLANG_DSV41_DEBUG_SMLA"):
            self._dsv41_audit_smla(layer, q, forward_batch, attn_kwargs, compress_ratio)
        attn_out, _lse = _ds41_ops().mixed_quant_sparse_flash_mla(q, **attn_kwargs)
        return attn_out

    # ---- V4.1 metadata assembly ------------------------------------------------
    #
    # ``ds41.compressor`` emits ``min(T, T // ratio + B)`` rows laid out as
    # <batch0>compressed + <batch1>compressed + ... + pad
    # (ops-transformer experimental/attention/compressor_v2/docs/
    # torchapi_compressor.md L96).  Both the RoPE positions and the slot list
    # handed to seam A therefore carry exactly that many rows: live rows first,
    # filler after.  Slots use -1 as filler because that is
    # ``kv_compress_epilog_v2``'s skip sentinel (cann-recipes-infer ops/ascendc/
    # docs/custom-npu_kv_compress_epilog_v2.md L89); positions use 0, which is
    # never stored because its row is skipped.
    #
    # V4 reads its slots from the allocator bundle.  V4.1 has no bundle
    # (``out_cache_loc_dsv4`` stays None), so a slot is derived as
    # ``full_loc // ratio``: that is exactly how the compressed pools are sized
    # and addressed (size = full_size // ratio, pages line up with the full
    # pool), so nothing beyond ``out_cache_loc`` is needed.

    _DSV41_SLOT_SKIP = -1

    def _dsv41_low_ratios(self) -> list:
        return [r for r in self._dsv4_unique_compress_ratios if r in (1, 2)]

    def _dsv41_candidate_ratio(self) -> Optional[int]:
        """The compression ratio of the layer that publishes candidate blocks."""
        layer_id = self._dsv41_candidate_source_layer
        if layer_id < 0 or layer_id >= len(self._dsv4_compress_ratios):
            return None
        return self._dsv4_compress_ratios[layer_id]

    def _dsv41_has_candidate_consumers(self) -> bool:
        """Whether any later index source layer rescoring inside the candidates."""
        source = self._dsv41_candidate_source_layer
        return self._dsv41_candidate_ratio() is not None and any(
            layer > source for layer in self._dsv41_index_source_layers
        )

    @staticmethod
    def _dsv41_compact_1d(
        dst: torch.Tensor,
        values: torch.Tensor,
        keep: torch.Tensor,
        fill: int,
    ) -> None:
        """``_stable_compact_1d`` with a caller-chosen tail filler.

        Same device-only ordinal scatter (no NonZero, no device-to-host size
        read), so it stays aclgraph-safe; only the untouched tail differs.

        The scatter *adds* into ``dst``, so the destination must start at zero:
        filling it with ``fill`` first would make every written slot
        ``value + fill`` (with the ``-1`` skip sentinel that is an off-by-one
        row). The tail is masked in afterwards instead, which keeps both the
        assignment semantics for written slots and a device-only implementation.
        """
        values = values.reshape(-1)
        keep = keep.reshape(-1)
        if values.numel() != keep.numel():
            raise ValueError(
                "dsv41 stable compact requires value/mask size equality, got "
                f"{values.numel()} and {keep.numel()}"
            )
        dst.zero_()
        if dst.numel() == 0 or values.numel() == 0:
            dst.fill_(fill)
            return
        ranks = torch.cumsum(keep.to(torch.int64), dim=0) - 1
        in_bounds = keep & (ranks < dst.numel())
        safe_ranks = ranks.clamp(min=0, max=dst.numel() - 1)
        compact = torch.where(
            in_bounds,
            values.to(dst.dtype),
            torch.zeros_like(values, dtype=dst.dtype),
        )
        dst.scatter_add_(0, safe_ranks, compact)
        if fill != 0:
            # Written slots are exactly ``[0, kept)``: ranks are dense over the
            # kept elements. Summing the mask keeps the count on device.
            kept = in_bounds.to(torch.int64).sum()
            tail = torch.arange(dst.numel(), device=dst.device) >= kept
            dst.masked_fill_(tail, fill)

    def _build_low_ratio_compress_metadata(self, forward_batch: ForwardBatch) -> None:
        """Eager half of the V4.1 metadata contract (V4 ``_compute_compress_locs``
        counterpart).  aclgraph allocates these buffers at capture
        (:meth:`_init_dsv41_graph_metadata`) and refreshes them in place on
        replay (:meth:`_refresh_graph_dsv41_1d`), so this runs eager-only.
        """
        ratios = self._dsv41_low_ratios()
        if not ratios:
            return
        forward_mode = forward_batch.forward_mode
        if forward_mode.is_decode() or forward_mode.is_extend_without_speculative():
            for ratio in ratios:
                self._build_dsv41_ratio_metadata(forward_batch, ratio)

    def _build_dsv41_ratio_metadata(
        self, forward_batch: ForwardBatch, ratio: int
    ) -> None:
        fm = self.forward_metadata
        device = forward_batch.positions.device
        positions = forward_batch.positions.reshape(-1).to(torch.int64)
        out_loc = forward_batch.out_cache_loc.reshape(-1).to(torch.int64)
        n = min(int(positions.numel()), int(out_loc.numel()))
        positions, out_loc = positions[:n], out_loc[:n]
        bs = int(forward_batch.batch_size)
        # One compressed row per token that closes a `ratio`-token group, i.e.
        # the token whose global position is `ratio - 1` (mod ratio).
        # Slots below one page are the reserved dummy page, so filtering on
        # `out_loc >= page_size` drops padding rows -- the same criterion recipes
        # uses (`slots >= block_size`, attention_data.py L878-L879).
        completes = ((positions % ratio) == (ratio - 1)) & (
            out_loc >= int(self.page_size)
        )
        if ratio == 1:
            # Ratio 1 has no fused compressor: the latent is the plain `wkv`
            # projection of every token row, so its row i is token i and the
            # slot row must keep that order -- mask in place (excluded rows get
            # -1 and the operator skips them) instead of compacting, exactly like
            # the graph refresh.  `pad == n` here (n // 1 + bs >= n).
            pos_row = positions.clone()
            pos_row.masked_fill_(~completes, 0)
            slot_row = out_loc.to(torch.int32)
            slot_row.masked_fill_(~completes, self._DSV41_SLOT_SKIP)
        else:
            # The fused compressor emits exactly ``min(n, n // ratio + bs)``
            # rows, dense in (request, position) order with the pad rows last,
            # so the slots are the completed groups compacted into that order.
            pad = min(n, n // ratio + bs)
            pos_row = torch.empty(pad, dtype=torch.int64, device=device)
            slot_row = torch.empty(pad, dtype=torch.int32, device=device)
            self._dsv41_compact_1d(
                pos_row, (positions // ratio) * ratio, completes, fill=0
            )
            self._dsv41_compact_1d(
                slot_row, out_loc // ratio, completes, fill=self._DSV41_SLOT_SKIP
            )
        setattr(fm, f"positions_cmp_padding_c{ratio}", pos_row)
        setattr(fm, f"c{ratio}_loc", slot_row)
        # Compressed pages carry the full pool's page ids (compressed page j is
        # full page j), so the full page table doubles as the compressed page
        # table read with page_size // ratio.
        block_tables = getattr(fm, "block_tables", None)
        setattr(fm, f"c{ratio}_page_table", block_tables)
        # The indexer-K pool pages at DSV41_INDEX_PAGE_SIZE slots, so its block
        # table is the FULL page table expanded to that granularity: full page p
        # owns the consecutive index pages [k*p, k*p + k), where
        # k = (page_size // ratio) // DSV41_INDEX_PAGE_SIZE (1 for ratio 2, 2 for
        # ratio 1). This is the mapping the CUDA backends build with the shared
        # ``expand_index_page_table`` helper, and the index-K write path
        # (``cache_index = full_loc // ratio``) depends on it.
        index_page_table = None
        if block_tables is not None:
            from sglang.srt.layers.attention.dsv4.metadata import (
                expand_index_page_table,
            )
            from sglang.srt.mem_cache.deepseek_v4_memory_pool import (
                DSV41_INDEX_PAGE_SIZE,
            )

            index_page_table = expand_index_page_table(
                block_tables,
                full_page_size=int(self.page_size),
                compress_ratio=ratio,
                index_page_size=DSV41_INDEX_PAGE_SIZE,
            )
        setattr(fm, f"c{ratio}_index_page_table", index_page_table)

    def _init_dsv41_graph_metadata(
        self, metadata, bs: int, tokens_per_req: int, device, *, forward_batch=None
    ) -> None:
        """Allocate the aclgraph-persistent low-ratio buffers for one bs bucket."""
        t = bs * tokens_per_req
        for ratio in self._dsv41_low_ratios():
            pad = min(t, t // ratio + bs)
            setattr(
                metadata,
                f"positions_cmp_padding_c{ratio}",
                torch.zeros(pad, dtype=torch.int64, device=device),
            )
            setattr(
                metadata,
                f"c{ratio}_loc",
                torch.full(
                    (pad,),
                    self._DSV41_SLOT_SKIP,
                    dtype=torch.int32,
                    device=device,
                ),
            )
            setattr(
                metadata,
                f"c{ratio}_page_table",
                getattr(metadata, "block_tables", None),
            )
            setattr(
                metadata,
                f"c{ratio}_index_page_table",
                self._dsv41_graph_index_page_table(metadata, ratio),
            )
            setattr(
                metadata,
                f"dsv41_cmp_topk_length_c{ratio}",
                torch.zeros((t, 1), dtype=torch.int32, device=device),
            )
        # The graph binds these tensor addresses at capture, so they must live in
        # fixed buffers that the replay refresh writes into.
        setattr(
            metadata,
            "dsv41_win_sparse_indices",
            torch.empty(
                (t, 1, int(self._dsv4_sliding_window_size)),
                dtype=torch.int32,
                device=device,
            ),
        )
        setattr(
            metadata,
            "dsv41_win_topk_length",
            torch.zeros((t, 1), dtype=torch.int32, device=device),
        )
        if not hasattr(metadata, "kernel_metadata") or metadata.kernel_metadata is None:
            metadata.kernel_metadata = {}
        if not is_npu_arch35():
            # BF16 SMLA and its AICPU metadata have a fixed, shared ABI. Do not
            # probe MQSMLA/QLI or allocate packed-index metadata on A3.
            probe_batch = (
                forward_batch
                if forward_batch is not None
                else SimpleNamespace(seq_lens=metadata.actual_seq_lengths_kv)
            )
            for ratio in (0, *self._dsv41_low_ratios()):
                attention_type = "win" if ratio == 0 else f"c{ratio}a"
                key = f"kernel_metadata_{attention_type}"
                buf = self.graph_metadata.get(key)
                if buf is None or buf.numel() != 2048:
                    buf = torch.zeros(2048, dtype=torch.int32, device=device)
                    self.graph_metadata[key] = buf
                # Warmup/capture uses this table before the first replay
                # refresh; an all-zero allocation is not a valid schedule.
                buf.copy_(
                    _dsv41_a3_ops().metadata(
                        self._dsv4_q_head_num,
                        1,
                        self._dsv4_head_dim,
                        batch_size=bs,
                        has_ori_kv=True,
                        has_cmp_kv=ratio > 0,
                        cmp_topk=self._dsv4_index_topk if ratio else 0,
                        **self._dsv41_a3_attention_kwargs(
                            probe_batch, ratio, metadata=metadata
                        ),
                    )
                )
                metadata.kernel_metadata[f"{attention_type}_metadata"] = buf
            return
        # The dispatch tables are variable-size, so ask the metadata ops for one
        # probe tensor per attention type and keep a same-shaped buffer.
        ds41 = _ds41_ops()
        cu_seqlens_q = (
            torch.arange(bs + 1, dtype=torch.int32, device=device) * tokens_per_req
        )
        probe_cmp_lengths = torch.zeros((t, 1), dtype=torch.int32, device=device)
        for ratio in (0, *self._dsv41_low_ratios()):
            attention_type = "win" if ratio == 0 else f"c{ratio}a"
            probe = ds41.mixed_quant_sparse_flash_mla_metadata(
                metadata.dsv41_win_topk_length,
                probe_cmp_lengths,
                cu_seqlens_q=cu_seqlens_q,
                num_heads_q=int(self._dsv4_q_head_num),
                num_heads_kv=1,
                head_dim=int(self._dsv4_head_dim),
                quant_mode=1,
                layout_q="TND",
                layout_kv="PA_BBND",
                has_ori_kv=True,
                has_cmp_kv=ratio > 0,
            )
            key = f"kernel_metadata_{attention_type}"
            buf = self.graph_metadata.get(key)
            if buf is None or buf.shape != probe.shape:
                buf = torch.zeros_like(probe)
                self.graph_metadata[key] = buf
            metadata.kernel_metadata[f"{attention_type}_metadata"] = buf
        for ratio in self._dsv41_low_ratios():
            probe = ds41.quant_lightning_indexer_metadata(
                cu_seqlens_q=cu_seqlens_q,
                seqused_q=None,
                seqused_k=torch.zeros(bs, dtype=torch.int32, device=device),
                cmp_residual_k=(
                    None
                    if ratio == 1
                    else torch.zeros(bs, dtype=torch.int32, device=device)
                ),
                batch_size=bs,
                num_heads_q=int(self._dsv4_index_n_heads),
                num_heads_k=1,
                head_dim=int(self._dsv4_index_head_dim),
                topk=int(self._dsv4_index_topk),
                mask_mode=3,
                cmp_ratio=ratio,
                layout_q="TND",
                layout_k="PA_BBND",
            )
            key = f"kernel_metadata_c{ratio}a_qli"
            buf = self.graph_metadata.get(key)
            if buf is None or buf.shape != probe.shape:
                buf = torch.zeros_like(probe)
                self.graph_metadata[key] = buf
            metadata.kernel_metadata[f"c{ratio}a_qli_metadata"] = buf

    def _dsv41_graph_index_page_table(self, metadata, ratio: int):
        """Captured buffer for one ratio's indexer-K block table.

        The indexer-K pool pages at ``DSV41_INDEX_PAGE_SIZE`` slots, so its block
        table is the FULL page table expanded to that granularity: full page p
        owns the consecutive index pages ``[k*p, k*p + k)`` with
        ``k = (page_size // ratio) // DSV41_INDEX_PAGE_SIZE`` (1 for ratio 2,
        2 for ratio 1).  That widens the row when ``k > 1``, so it always gets a
        dedicated buffer rather than aliasing ``block_tables``; aclgraph binds
        the address at capture and :meth:`_refresh_graph_dsv41_1d` rewrites it in
        place on every replay.
        """
        block_tables = getattr(metadata, "block_tables", None)
        if block_tables is None:
            return None
        from sglang.srt.mem_cache.deepseek_v4_memory_pool import (
            DSV41_INDEX_PAGE_SIZE,
        )

        slots_per_page = int(self.page_size) // ratio
        assert slots_per_page % DSV41_INDEX_PAGE_SIZE == 0, (
            f"ratio-{ratio} index pages of {DSV41_INDEX_PAGE_SIZE} slots do "
            f"not tile a FULL page of {slots_per_page} slots"
        )
        blocks_per_page = slots_per_page // DSV41_INDEX_PAGE_SIZE
        shape = (block_tables.shape[0], block_tables.shape[1] * blocks_per_page)
        key = f"dsv41_index_page_table_c{ratio}"
        buf = self.graph_metadata.get(key)
        if buf is None or tuple(buf.shape) != shape:
            buf = torch.zeros(shape, dtype=torch.int32, device=block_tables.device)
            self.graph_metadata[key] = buf
        return buf

    def _refresh_graph_dsv41_1d(self, ctx) -> None:
        """Decode replay refresh: derive the rows from ``out_cache_loc``."""
        fm = ctx.fm
        n = int(ctx.bs)
        live = ctx.live_seq_lens[:n].to(torch.int64)
        out_loc = ctx.forward_batch.out_cache_loc.reshape(-1).to(torch.int64)[:n]
        if not is_npu_arch35() and out_loc.numel() < n:
            # Replay views pad seq_lens to the bucket, but retain the original
            # output slots. Dummy rows must not broadcast a live request slot.
            out_loc = F.pad(out_loc, (0, n - out_loc.numel()))
        positions = torch.clamp(live - 1, min=0)
        completes = (live > 0) & (out_loc >= int(self.page_size))
        for ratio in self._dsv41_low_ratios():
            keep = completes & ((positions % ratio) == (ratio - 1))
            pos_dst = getattr(fm, f"positions_cmp_padding_c{ratio}")
            slot_dst = getattr(fm, f"c{ratio}_loc")
            if ratio == 1:
                # Per-token rows (native projection): identity, not compacted.
                pos_dst.copy_(torch.where(keep, positions, torch.zeros_like(positions)))
                slot_dst.copy_(
                    torch.where(
                        keep,
                        out_loc,
                        torch.full_like(out_loc, self._DSV41_SLOT_SKIP),
                    ).to(slot_dst.dtype)
                )
                continue
            self._dsv41_compact_1d(
                pos_dst,
                (positions // ratio) * ratio,
                keep,
                fill=0,
            )
            self._dsv41_compact_1d(
                slot_dst,
                out_loc // ratio,
                keep,
                fill=self._DSV41_SLOT_SKIP,
            )

        # The indexer-K block table is the FULL page table expanded to the index
        # pool's page granularity (full page p -> index pages [k*p, k*p + k)).
        # The shared ``block_tables`` view has already been refreshed by the
        # parent, so re-derive the expansion into the captured buffers here.
        block_tables = getattr(fm, "block_tables", None)
        if block_tables is None:
            return
        from sglang.srt.layers.attention.dsv4.metadata import (
            expand_index_page_table,
        )
        from sglang.srt.mem_cache.deepseek_v4_memory_pool import (
            DSV41_INDEX_PAGE_SIZE,
        )

        for ratio in self._dsv41_low_ratios():
            dst = getattr(fm, f"c{ratio}_index_page_table", None)
            if dst is None:
                continue
            dst.copy_(
                expand_index_page_table(
                    block_tables,
                    full_page_size=int(self.page_size),
                    compress_ratio=ratio,
                    index_page_size=DSV41_INDEX_PAGE_SIZE,
                )
            )


class DeepseekV41AscendAttnBackend(
    DeepseekV4AscendAttnBackend, LowRatioCompressAscendBackendMixin
):
    """DeepSeek-V4.1 low-ratio attention on Ascend A3 and A5."""

    def __init__(self, model_runner: ModelRunner, speculative_step_id: int = 0):
        super().__init__(model_runner, speculative_step_id=speculative_step_id)
        hf = getattr(model_runner.model_config, "hf_config", model_runner.model_config)
        # Two-level indexer: the layer that publishes candidate blocks and the
        # consumer layers that rescore inside them. -1 / 0 on a plain V4 model.
        self._dsv41_candidate_source_layer = int(
            getattr(hf, "candidate_source_layer_id", -1)
        )
        self._dsv41_candidate_topk_blocks = int(getattr(hf, "candidate_topk_blocks", 0))
        self._dsv41_candidate_block_size = int(getattr(hf, "candidate_block_size", 0))
        # A layer that publishes candidates is only useful when a later index
        # source layer consumes them; read through _dsv41_has_candidate_consumers
        # because a draft worker empties the ratio list below.
        self._dsv41_index_source_layers = tuple(
            getattr(hf, "index_source_layer_ids", ())
        )
        # QSLI's 544B/8-token K records, allocated on first write; see
        # _dsv41_qsli_cache.
        self._dsv41_qsli_caches: dict = {}
        # Layers whose QSLI geometry has already been dumped (SGLANG_DSV41_DEBUG_QSLI).
        self._dsv41_debugged: set = set()
        if self._dsv4_has_low_ratio and is_npu_arch35():
            # MQSMLA fixes N1 = 64 and does not shard the query heads over TP
            # (mixed_quant_sparse_flash_mla_dsl/README.md L399 "N1=64 ... 固定").
            # A narrower per-rank width would feed the op the wrong head count,
            # so reject it loudly here instead of at the first attention call.
            assert self._dsv4_q_head_num == 64, (
                "DeepSeek-V4.1 on NPU needs all 64 query heads on every rank "
                f"(got {self._dsv4_q_head_num}): check num_attention_heads and "
                "the attention TP size."
            )
        if self._dsv4_has_low_ratio and not is_npu_arch35():
            _dsv41_a3_ops()
            # Resolve the public API here; graph warmup loads the lazy binding.
            _dsv41_a3_compressor_kernel()

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        *,
        compress_ratio: int = 0,
        attn_sink: torch.Tensor | None = None,
        save_kv_cache: bool = True,
    ) -> torch.Tensor:
        # Generation-exclusive: V4.1 only ever declares compress_ratios in
        # (0, 1, 2) (recipes configuration_deepseek.py L218), V4 only in
        # (0, 4, 128). Mixing the two would silently route a ratio to an
        # operator that has no metadata for it.
        if compress_ratio not in (
            (0, 1, 2) if self._dsv4_has_low_ratio else (0, 4, 128)
        ):
            raise ValueError(
                "V4 attention expects compress_ratio in (0, 4, 128); "
                f"V4.1 in (0, 1, 2); got {compress_ratio}"
            )
        # idle ranks only feed the MoE collectives; skip attn + store_cache and return zeros
        if forward_batch.forward_mode.is_idle():
            return torch.zeros_like(q)
        # MQALayer prepass already stores K and passes save_kv_cache=False; True callers still get the write
        if save_kv_cache:
            self.store_cache(
                layer_id=layer.layer_id, swa_k=k, forward_batch=forward_batch
            )
        if self._dsv4_has_low_ratio:
            # V4.1 uses ds41.mixed_quant_sparse_flash_mla for every ratio: it
            # expresses both the window and the compressed side as explicit
            # sparse indices, so ratio 0 is not the old "swa" op either.
            return self._forward_low_ratio_attention(
                q, layer, forward_batch, attn_sink, compress_ratio
            )
        if compress_ratio == 0:
            return self._forward_swa(q, layer, forward_batch, attn_sink)
        return self._forward_compressed(
            q, layer, forward_batch, attn_sink, compress_ratio
        )


class DeepseekV41AscendMultiStepDraftBackend(DeepseekV4AscendMultiStepDraftBackend):
    backend_cls = DeepseekV41AscendAttnBackend
