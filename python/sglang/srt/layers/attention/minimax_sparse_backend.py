from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.configs.model_config import (
    get_minimax_sparse_attention_config,
    get_minimax_sparse_disable_value_layer_ids,
    get_minimax_sparse_layer_ids,
    get_minimax_sparse_score_type,
)
from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.attention.minimax_sparse_ops.minimax_sparse import (
    minimax_sparse_decode,
    minimax_sparse_prefill,
)
from sglang.kernels.ops.attention.minimax_sparse.verify import (
    minimax_sparse_verify_prefill,
)
from sglang.srt.mem_cache.memory_pool import MiniMaxSparseKVPool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.server_args import m3_fp8_attn_gemm_enabled

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)


def _quant_q_fp8(q: torch.Tensor, q_scale: Optional[float]) -> torch.Tensor:
    # Same convention as the KV pools: the fp8 tensor stores value/scale and
    # the attention kernels multiply the logits back by the scale (None = unit).
    if q_scale is not None:
        q = q / q_scale
    return q.to(torch.float8_e4m3fn)


class MiniMaxSparseAttnBackend(AttentionBackend):
    def __init__(self, runner: ModelRunner):
        assert isinstance(runner.token_to_kv_pool, MiniMaxSparseKVPool)
        self.kv_pool = runner.token_to_kv_pool
        self.req_to_token = runner.req_to_token_pool.req_to_token
        self.max_context_len = int(runner.model_config.context_len)
        self.fp8_attn_gemm = m3_fp8_attn_gemm_enabled(runner.server_args)
        if self.fp8_attn_gemm:
            assert self.kv_pool.main_pool.dtype == torch.float8_e4m3fn, (
                "fp8 attn-GEMM mode requires an fp8_e4m3fn main KV pool, got "
                f"{self.kv_pool.main_pool.dtype}"
            )

        hf_config = runner.model_config.hf_config
        sparse_cfg = get_minimax_sparse_attention_config(hf_config)
        self.idx_head_dim = sparse_cfg["sparse_index_dim"]
        self.dense_layer_ids, self.sparse_layer_ids = get_minimax_sparse_layer_ids(
            sparse_cfg
        )
        self.disable_value_layer_ids: set[int] = set(
            get_minimax_sparse_disable_value_layer_ids(sparse_cfg)
        )
        self.score_type: str = get_minimax_sparse_score_type(sparse_cfg)

        # Plain Python int so it is safe inside CUDA graphs (no .item() at graph time).
        self._max_seqlen_q: int = 1
        self._max_seqlen_k: int = 1

        self.block_size_q = 1
        self.block_size_k = sparse_cfg["sparse_block_size"]
        if "sparse_init_block" in sparse_cfg:
            self.init_blocks = sparse_cfg["sparse_init_block"]
        else:
            init_tokens = sparse_cfg["sparse_init_tokens"]
            self.init_blocks = (
                init_tokens + self.block_size_k - 1
            ) // self.block_size_k
        if "sparse_local_block" in sparse_cfg:
            self.local_blocks = sparse_cfg["sparse_local_block"]
        else:
            local_tokens = sparse_cfg["sparse_local_tokens"]
            self.local_blocks = (
                local_tokens + self.block_size_k - 1
            ) // self.block_size_k + 1
        self.topk_blocks = sparse_cfg["sparse_topk_blocks"]

        # Fixed upper bound on the number of KV blocks, used by the EAGLE3
        # TARGET_VERIFY kernel to size its score buffer's K-block dimension with
        # a *constant* shape. Capture and replay must allocate the same tensor
        # shape; sizing it off the live seq_len (which varies per request/batch)
        # would lock a small shape at capture and overflow it at replay. We bound
        # it by the full context length plus the largest possible draft, so any
        # legal verify batch fits. Used ONLY by the verify path; normal
        # prefill/decode keep their dynamic sizing.
        max_draft = int(
            getattr(runner.server_args, "speculative_num_draft_tokens", 0) or 0
        )
        if max_draft <= 0:
            # Generous fallback when spec args are absent (e.g. draft worker side).
            max_draft = 8
        _k_bound = self.max_context_len + max_draft
        self.max_seqblock_k_upper = (
            _k_bound + self.block_size_k - 1
        ) // self.block_size_k

        # MSA (fmha_sm100) is SM100-only; fall back to the Triton sparse path when
        # the kernel is unavailable or its constraints don't hold.
        from sglang.srt.environ import envs
        from sglang.srt.layers.attention.minimax_sparse_ops.msa import (
            msa_available,
        )

        # MSA (fmha_sm100) runs bf16, or uniform fp8_e4m3 under fp8 attn-GEMM mode
        # (which also casts q to fp8). An fp8 main KV cache WITHOUT the flag
        # would pair a bf16 q with fp8 K/V — unsupported by fmha_sm100's
        # uniform-dtype kernels — so it stays on the Triton sparse path (which
        # dequants fp8 on load). e5m2 is never allowed into MSA (fmha_sm100's
        # variant lookup would silently dispatch the e4m3 kernel).
        _main_kv_is_fp8 = self.kv_pool.main_pool.dtype in (
            torch.float8_e4m3fn,
            torch.float8_e5m2,
        )
        _msa_fp8_ok = (
            self.fp8_attn_gemm and self.kv_pool.main_pool.dtype == torch.float8_e4m3fn
        )
        self.use_msa = (
            not envs.SGLANG_DISABLE_MSA.get()
            and msa_available()
            and self.block_size_k == 128
            and self.kv_pool.page_size == self.block_size_k
            and self.topk_blocks in (4, 8, 16, 32)
            and (not _main_kv_is_fp8 or _msa_fp8_ok)
        )
        if (
            not self.use_msa
            and not envs.SGLANG_DISABLE_MSA.get()
            and msa_available()
            and self.block_size_k == 128
            and self.kv_pool.page_size != self.block_size_k
        ):
            logger.warning(
                "MiniMax-M3 MSA decode disabled: page_size=%d != sparse block size "
                "%d. Pass --page-size 128 (with an attention backend that allows it, "
                "e.g. fa4 or trtllm_mha) to enable the faster MSA kernel; falling "
                "back to the Triton sparse path.",
                self.kv_pool.page_size,
                self.block_size_k,
            )
        self._msa_dec_meta = None
        if self.use_msa:
            from sglang.srt.runtime_context import get_parallel

            self.num_q_heads = (
                runner.model_config.num_attention_heads // get_parallel().attn_tp_size
            )
            self.num_kv_heads = self.kv_pool.main_pool.head_num
            self._msa_nb_max = (
                self.max_context_len + self.block_size_k - 1
            ) // self.block_size_k
            self._msa_cg: dict[int, tuple] = {}

        self.page_size = self.kv_pool.page_size
        self.use_dense_sparse_decode = (
            envs.SGLANG_OPT_USE_MINIMAX_DENSE_SPARSE_DECODE.get()
            and self.block_size_k % self.page_size == 0
            # _dense_sparse_main_decode calls trtllm decode with a bf16 q and
            # unit bmm scales — no fp8 handling yet (follow-up).
            and not self.fp8_attn_gemm
        )
        from sglang.srt.model_executor.cuda_graph_config import (
            Backend,
            Phase,
            check_cuda_graph_backend,
        )

        _sa = getattr(runner, "server_args", None)
        _decode_cuda_graph = not check_cuda_graph_backend(
            Phase.DECODE, Backend.DISABLED
        )
        self._use_msa_decode = self.use_msa and (
            not _decode_cuda_graph or envs.SGLANG_OPT_USE_MSA_DECODE_UNDER_GRAPH.get()
        )

        # EAGLE3 speculative decoding: TARGET_VERIFY batches are routed to a
        # dedicated graph-safe verify kernel (see _forward_verify) that sizes its
        # score buffer off the constant max_seqblock_k_upper instead of the live
        # seq_len, and pre-allocates cu_seqlens/seq_lens/extend_seq_lens as
        # address-stable graph buffers. This avoids the capture/replay shape and
        # address drift that previously made MSA + spec + cuda graph crash, so the
        # combination is now supported. The decode-step path (MSA or Triton) is
        # independent of the captured TARGET_VERIFY graph.
        if (
            self.use_msa
            and _decode_cuda_graph
            and getattr(_sa, "speculative_algorithm", None) is not None
        ):
            logger.info(
                "[MiniMaxSparse] speculative decoding under CUDA graph is enabled via "
                "the EAGLE3 verify kernel (graph-safe score buffer + pre-allocated "
                "cu_seqlens/seq_lens buffers)."
            )
        self._msa_owns_decode = self._use_msa_decode and not (
            self.use_dense_sparse_decode and self.kv_pool.main_pool.head_num == 1
        )
        self.dense_backend: Optional[AttentionBackend] = None

        logger.info(
            f"[MiniMaxSparse] Backend initialized "
            f"(score_type={self.score_type!r}, "
            f"main_attn={'MSA' if self.use_msa else 'triton'}, "
            f"msa_decode={self._use_msa_decode}, "
            f"msa_owns_decode={self._msa_owns_decode}, "
            f"decode_cuda_graph={_decode_cuda_graph}, "
            f"fp8_attn_gemm={self.fp8_attn_gemm}, "
            f"disable_value_layers={sorted(self.disable_value_layer_ids)})"
        )
        if self.fp8_attn_gemm and self.use_msa:
            logger.info(
                "[MiniMaxSparse] fp8 MSA active: the first forward may "
                "JIT-compile fmha_sm100 fp8 kernel variants (cold cache can "
                "take minutes; compiles serialize across TP ranks)."
            )

    def init_forward_metadata_out_graph(
        self, forward_batch: ForwardBatch, in_capture: bool = False
    ):
        # cuda-graph replay views are a SimpleNamespace without extend_seq_lens_cpu,
        # and TARGET_VERIFY sets it to None despite is_extend() — getattr covers both.
        self._msa_dec_meta = None

        # EAGLE3 TARGET_VERIFY: fill the pre-allocated graph buffers (allocated in
        # init_cuda_graph_state) so their data_ptr is identical at capture and
        # replay. The verify kernel reads cu_seqlens/extend_seq_lens/seq_lens from
        # these address-stable buffers. Both capture and replay run this hook, so
        # the content is re-written with the same (constant) values; only the
        # seq_lens (prefix+D) entry differs (capture sees the dummy fill value,
        # replay sees the real per-request prefix), which is exactly what we want.
        if forward_batch.forward_mode.is_target_verify():
            self._fill_verify_graph_buffers(forward_batch)
        else:
            extend_lens = getattr(forward_batch, "extend_seq_lens_cpu", None)
            if extend_lens is not None:
                self._max_seqlen_q = int(max(extend_lens))
            else:
                self._max_seqlen_q = 1
            if in_capture and forward_batch.forward_mode.is_decode_or_idle():
                self._max_seqlen_k = self.max_context_len
            else:
                self._max_seqlen_k = int(forward_batch.seq_lens_cpu.max().item())

            # Build plan + page table eager (outside capture) so captured
            # forward_decode runs only device-side ops; host-side code can't be
            # captured.
            if self._msa_owns_decode and forward_batch.forward_mode.is_decode_or_idle():
                self._prepare_msa_decode_meta(forward_batch)

    def _fill_verify_graph_buffers(self, forward_batch: ForwardBatch):
        """Populate the EAGLE3 TARGET_VERIFY graph buffers.

        Called from init_forward_metadata_out_graph for every TARGET_VERIFY
        forward (both graph capture and replay, plus any eager verify that ran
        after init_cuda_graph_state). Invariant: every request in a verify batch
        proposes the same number of draft tokens D, so:

            extend_seq_lens = [D, D, ..., D]          (constant per captured graph)
            cu_seqlens      = [0, D, 2D, ..., bs*D]   (constant per captured graph)
            seq_lens        = real_prefix + D         (prefix from the graph
                                                       seq_lens buffer)

        All three depend only on (bs, D) plus the graph seq_lens buffer — none on
        a freshly-allocated temporary — so their addresses are stable across
        capture/replay and the cuda graph replays correctly. The kernel sizes its
        score buffer off the constant ``max_seqblock_k_upper`` (set in __init__),
        not the live seq_len, so the K-block dimension is also graph-safe.
        """
        bs = forward_batch.seq_lens.shape[0]
        spec_info = getattr(forward_batch, "spec_info", None)
        draft_token_num = getattr(spec_info, "draft_token_num", None)
        if draft_token_num is None:
            # Fallback: total tokens / num requests. input_ids holds bs*D tokens.
            draft_token_num = forward_batch.input_ids.shape[0] // max(bs, 1)
        D = int(draft_token_num)

        self._max_seqlen_q = D
        # The verify kernel sizes the score buffer's K-block dim off
        # max_seqblock_k_upper (constant); the live _max_seqlen_k is only a
        # fallback, so pin it to the same constant upper bound to avoid any
        # dynamic-shape leak into the captured graph.
        self._max_seqlen_k = self.max_seqblock_k_upper * self.block_size_k

        # Eager verify (cuda graph disabled, or bs beyond the captured range):
        # init_cuda_graph_state was never called, so there are no pre-allocated
        # buffers. _forward_verify builds temporaries instead — nothing to fill.
        if not hasattr(self, "_verify_cu_seqlens_buf"):
            return

        device = self._verify_cu_seqlens_buf.device
        self._verify_extend_seq_lens_buf[:bs].fill_(D)
        self._verify_cu_seqlens_buf[: bs + 1] = torch.arange(
            0, (bs + 1) * D, step=D, dtype=torch.int32, device=device
        )
        # seq_lens = prefix + D. forward_batch.seq_lens is the (address-stable)
        # graph seq_lens buffer at replay (real prefix) or the dummy fill value at
        # capture; either way writing prefix+D into our buffer is safe — capture
        # just needs a non-crashing dummy, replay needs the real prefix+D.
        self._verify_seq_lens_buf[:bs] = (
            forward_batch.seq_lens[:bs].to(torch.int32) + D
        )

    def _prepare_msa_decode_meta(self, forward_batch: ForwardBatch):
        from sglang.srt.layers.attention.minimax_sparse_ops.msa import (
            build_msa_decode_cg_plan,
            update_msa_decode_cg_meta,
        )

        bs = forward_batch.seq_lens.shape[0]
        if bs == 0:
            return
        entry = self._msa_cg.get(bs)
        if entry is None:
            device = forward_batch.seq_lens.device
            plan = build_msa_decode_cg_plan(
                self.num_q_heads,
                self.num_kv_heads,
                self.block_size_k,
                self.topk_blocks,
                bs,
                device=device,
                is_fp8=self.fp8_attn_gemm,
            )
            kv_indices_buf = torch.zeros(
                bs * self._msa_nb_max, dtype=torch.int32, device=device
            )
            entry = (plan, kv_indices_buf)
            self._msa_cg[bs] = entry
        plan, kv_indices_buf = entry
        update_msa_decode_cg_meta(
            plan,
            kv_indices_buf,
            self.req_to_token,
            forward_batch.req_pool_indices,
            forward_batch.seq_lens,
            self.block_size_k,
            self.topk_blocks,
            self.num_q_heads,
            self.num_kv_heads,
        )
        self._msa_dec_meta = (kv_indices_buf, plan)

    def init_forward_metadata_in_graph(self, forward_batch: ForwardBatch):
        pass

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        # Pre-allocate graph-stable buffers for EAGLE3 TARGET_VERIFY.
        #
        # Building cu_seqlens / seq_lens / extend_seq_lens as fresh temporaries
        # inside the forward (torch.cat / torch.full / `a + b`) makes their
        # data_ptr differ between capture and replay: the graph records the
        # capture-time pointer, replay reads a different address -> garbage /
        # VMFault. Pre-allocating them here (one-shot, address fixed for the
        # backend's lifetime) and *writing into* them in
        # init_forward_metadata_out_graph keeps the address identical at capture
        # and replay. This mirrors the triton backend's qo_indptr pattern.
        #
        # verify invariant: every request proposes the same draft_token_num D, so
        # extend_seq_lens=[D]*bs and cu_seqlens=[0,D,2D,..,bs*D] depend only on
        # (bs, D), both fixed for a given captured graph -> graph-safe.
        device = self.req_to_token.device
        self._verify_max_bs = int(max_bs)
        # [max_bs] of D (filled per forward with the constant D).
        self._verify_extend_seq_lens_buf = torch.zeros(
            (max_bs,), dtype=torch.int32, device=device
        )
        # [max_bs + 1] of arange(0, (bs+1)*D, D).
        self._verify_cu_seqlens_buf = torch.zeros(
            (max_bs + 1,), dtype=torch.int32, device=device
        )
        # [max_bs] of (prefix + D). At replay forward_batch.seq_lens is the real
        # per-request prefix (address-stable graph buffer filled by replay_prep).
        self._verify_seq_lens_buf = torch.zeros(
            (max_bs,), dtype=torch.int32, device=device
        )

    def get_cuda_graph_seq_len_fill_value(self):
        return 1

    @staticmethod
    def _is_sparse_kv_cached_by_fusion(
        forward_batch: ForwardBatch, layer_id: int
    ) -> bool:
        layer_ids = forward_batch.minimax_m3_precached_sparse_layers
        return layer_ids is not None and layer_id in layer_ids

    def forward(
        self,
        q,
        k,
        v,
        layer,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        **kwargs,
    ):
        if forward_batch.forward_mode.is_idle():
            idx_q = kwargs.get("idx_q")
            num_idx_heads = idx_q.shape[1]
            disable_value = layer.layer_id in self.disable_value_layer_ids
            idx_out: Optional[torch.Tensor] = (
                None
                if disable_value
                else q.new_zeros(q.shape[0], num_idx_heads * self.idx_head_dim)
            )
            out = q.new_zeros(q.shape[0], layer.tp_q_head_num * layer.v_head_dim)
            return idx_out, out
        else:
            return super().forward(
                q, k, v, layer, forward_batch, save_kv_cache, **kwargs
            )

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
        *,
        idx_q: torch.Tensor,
        idx_k: torch.Tensor,
        idx_v: Optional[torch.Tensor],
    ):
        # EAGLE3 TARGET_VERIFY: route to the graph-safe verify kernel instead of
        # the normal prefill path. The verify kernel sizes its score buffer off
        # the constant max_seqblock_k_upper (not the live seq_len) and reads
        # cu_seqlens/seq_lens/extend_seq_lens from pre-allocated, address-stable
        # graph buffers, so it replays correctly under cuda graph. The normal
        # extend path below builds those tensors as fresh temporaries, which is
        # fine for eager prefill but breaks capture/replay (data_ptr drift).
        if forward_batch.forward_mode.is_target_verify():
            return self._forward_verify(
                q,
                k,
                v,
                layer,
                forward_batch,
                save_kv_cache,
                layer.layer_id in self.disable_value_layer_ids,
                idx_q=idx_q,
                idx_k=idx_k,
                idx_v=idx_v,
            )

        disable_value = layer.layer_id in self.disable_value_layer_ids
        kv_cached_by_fusion = self._is_sparse_kv_cached_by_fusion(
            forward_batch, layer.layer_id
        )
        if not kv_cached_by_fusion:
            self.kv_pool.set_fused_kv_index_buffer(
                layer,
                forward_batch.out_cache_loc,
                k,
                v,
                idx_k,
                None if disable_value else idx_v,
                layer.k_scale_float,
                layer.v_scale_float,
                layer.idx_k_scale_float,
                layer.idx_v_scale_float,
            )
        k_cache, v_cache = self.kv_pool.get_kv_buffer(layer.layer_id)
        if disable_value:
            idx_k_cache = self.kv_pool.get_index_k_buffer(layer.layer_id)
            idx_v_cache = None
        else:
            idx_k_cache, idx_v_cache = self.kv_pool.get_index_kv_buffer(layer.layer_id)

        cu_seqlens = torch.cat(
            [
                torch.zeros(
                    1, dtype=torch.int32, device=forward_batch.extend_seq_lens.device
                ),
                forward_batch.extend_seq_lens.to(torch.int32).cumsum(0).to(torch.int32),
            ]
        )
        seq_lens = forward_batch.seq_lens.to(torch.int32)
        if forward_batch.extend_prefix_lens is not None:
            prefix_lens = forward_batch.extend_prefix_lens.to(torch.int32)
        else:
            prefix_lens = torch.zeros_like(seq_lens)

        # DP attention pads q beyond the real token count for collective alignment;
        # trim to actual tokens so the sparse kernel sees consistent shapes.
        if forward_batch.extend_seq_lens_cpu is not None:
            actual_num_tokens = int(sum(forward_batch.extend_seq_lens_cpu))
        else:
            actual_num_tokens = int(cu_seqlens[-1].item())
        original_num_tokens = q.shape[0]
        if actual_num_tokens < original_num_tokens:
            q = q[:actual_num_tokens]
            idx_q = idx_q[:actual_num_tokens]

        # fp8 attention GEMMs: quantize q/idx_q AFTER the KV store (which reads
        # the bf16 k/v) and the DP trim.
        if self.fp8_attn_gemm:
            q = _quant_q_fp8(q, layer.q_scale_float)
            idx_q = _quant_q_fp8(idx_q, layer.idx_q_scale_float)

        idx_o, o = minimax_sparse_prefill(
            q,
            k_cache,
            v_cache,
            None,
            idx_q,
            idx_k_cache,
            idx_v_cache,
            None,
            self.req_to_token,
            forward_batch.req_pool_indices,
            cu_seqlens,
            seq_lens,
            prefix_lens,
            self._max_seqlen_q,
            self._max_seqlen_k,
            self.block_size_q,
            self.block_size_k,
            self.topk_blocks,
            self.init_blocks,
            self.local_blocks,
            score_type=self.score_type,
            disable_index_value=disable_value,
            use_msa=self.use_msa,
            seqlens_cpu=forward_batch.extend_seq_lens_cpu,
            q_scale=layer.q_scale_float,
            k_scale=layer.k_scale_float,
            v_scale=layer.v_scale_float,
            idx_q_scale=layer.idx_q_scale_float,
            idx_k_scale=layer.idx_k_scale_float,
            idx_v_scale=layer.idx_v_scale_float,
        )

        if actual_num_tokens < original_num_tokens:
            pad_len = original_num_tokens - actual_num_tokens
            o = torch.cat([o, o.new_zeros(pad_len, *o.shape[1:])], dim=0)
            if idx_o is not None:
                idx_o = torch.cat(
                    [idx_o, idx_o.new_zeros(pad_len, *idx_o.shape[1:])], dim=0
                )

        return (
            (
                None
                if idx_o is None
                else idx_o.reshape(original_num_tokens, -1).contiguous()
            ),
            o.reshape(original_num_tokens, -1).contiguous(),
        )

    def _forward_verify(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer,
        forward_batch: ForwardBatch,
        save_kv_cache: bool,
        disable_value: bool,
        *,
        idx_q: torch.Tensor,
        idx_k: torch.Tensor,
        idx_v: Optional[torch.Tensor],
    ):
        """EAGLE3 TARGET_VERIFY via the graph-safe verify kernel.

        Mirror of forward_extend for the verify batch shape (each request has
        ``draft_token_num`` query tokens), but with two graph-safety changes:

        1. cu_seqlens / extend_seq_lens / seq_lens come from the pre-allocated,
           address-stable buffers filled in init_forward_metadata_out_graph
           (when cuda graph is on), so their data_ptr is identical at capture and
           replay. In pure-eager mode (no graph buffers) we build temporaries
           here — graph-safety is irrelevant without a graph.
        2. The verify kernel sizes its score buffer's K-block dim off the
           constant ``max_seqblock_k_upper`` (passed below), not the live
           seq_len, so that dimension is also constant across capture/replay.

        The kernel still reads the real seq_lens at runtime (from the seq_lens
        buffer, = prefix + D) for causal masking and KV indexing.
        """
        # Write k/v + index k/v into the paged caches. Upstream fuses the main
        # and index KV store into one call (set_fused_kv_index_buffer), unlike
        # the older separate set_kv_buffer / set_index_*_buffer API.
        self.kv_pool.set_fused_kv_index_buffer(
            layer,
            forward_batch.out_cache_loc,
            k,
            v,
            idx_k,
            None if disable_value else idx_v,
            layer.k_scale_float,
            layer.v_scale_float,
            layer.idx_k_scale_float,
            layer.idx_v_scale_float,
        )
        k_cache, v_cache = self.kv_pool.get_kv_buffer(layer.layer_id)
        if disable_value:
            idx_k_cache = self.kv_pool.get_index_k_buffer(layer.layer_id)
            idx_v_cache = None
        else:
            idx_k_cache, idx_v_cache = self.kv_pool.get_index_kv_buffer(
                layer.layer_id
            )

        bs = forward_batch.seq_lens.shape[0]

        # prefix_lens = the verify batch's seq_lens, which is the PREFIX length
        # (the scheduler adds accept_lens after verify, not before). This is the
        # graph seq_lens buffer at replay (address-stable, real prefix).
        prefix_lens = forward_batch.seq_lens.to(torch.int32)

        has_graph_buf = hasattr(self, "_verify_cu_seqlens_buf")
        if has_graph_buf:
            # Graph path: pre-allocated, address-stable buffers. Their content
            # was filled in init_forward_metadata_out_graph (cu_seqlens /
            # extend_seq_lens are the constant [0,D,2D,..] / [D]*bs; seq_lens is
            # prefix+D). Under replay everything is already correct. Under a
            # rare eager-with-buffers run (bs within range but not via graph)
            # recompute seq_lens from the live prefix to be safe.
            cu_seqlens = self._verify_cu_seqlens_buf[: bs + 1]
            extend_seq_lens = self._verify_extend_seq_lens_buf[:bs]
            seq_lens = self._verify_seq_lens_buf[:bs]
            if not torch.cuda.is_current_stream_capturing():
                seq_lens.copy_(prefix_lens + extend_seq_lens)
        else:
            # Pure eager (cuda graph disabled, or bs beyond the captured range):
            # init_cuda_graph_state was never called, so build temporaries.
            spec_info = getattr(forward_batch, "spec_info", None)
            D = int(
                getattr(spec_info, "draft_token_num", self._max_seqlen_q)
            )
            extend_seq_lens = torch.full(
                (bs,), D, dtype=torch.int32, device=prefix_lens.device
            )
            cu_seqlens = torch.cat(
                [
                    torch.zeros(1, dtype=torch.int32, device=prefix_lens.device),
                    extend_seq_lens.cumsum(0).to(torch.int32),
                ]
            )
            seq_lens = prefix_lens + extend_seq_lens

        original_num_tokens = q.shape[0]

        # seqlens_cpu: host-side per-request Q lengths for get_cu_seqblocks, so it
        # sums query blocks on the host instead of calling .item() on a device
        # tensor (which would host-sync and is forbidden inside cuda-graph
        # capture). Every verify request proposes the same D tokens, so this is a
        # constant [D]*bs list — cheap to build, no device read. _max_seqlen_q was
        # set to D (a Python int) in _fill_verify_graph_buffers / the eager branch.
        D_host = int(self._max_seqlen_q)
        seqlens_cpu = [D_host] * bs

        idx_o, o = minimax_sparse_verify_prefill(
            q,
            k_cache,
            v_cache,
            None,
            idx_q,
            idx_k_cache,
            idx_v_cache,
            None,
            self.req_to_token,
            forward_batch.req_pool_indices,
            cu_seqlens,
            seq_lens,
            prefix_lens,
            self._max_seqlen_q,
            self._max_seqlen_k,
            self.max_seqblock_k_upper,
            self.block_size_q,
            self.block_size_k,
            self.topk_blocks,
            self.init_blocks,
            self.local_blocks,
            score_type=self.score_type,
            disable_index_value=disable_value,
            seqlens_cpu=seqlens_cpu,
        )

        return (
            (
                None
                if idx_o is None
                else idx_o.reshape(original_num_tokens, -1).contiguous()
            ),
            o.reshape(original_num_tokens, -1).contiguous(),
        )

    def _dense_sparse_main_decode(
        self,
        q: torch.Tensor,
        page_table: torch.Tensor,
        real_seq_lens: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        layer,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        from sglang.srt.layers.attention.trtllm_mha_backend import TRTLLMHAAttnBackend

        if isinstance(self.dense_backend, TRTLLMHAAttnBackend):
            import flashinfer

            ps = self.page_size
            nkv = 1
            head_dim = q.size(-1)
            # [max_slots, nkv, D] -> [num_pages, page_size, nkv, D]
            #                     -> [num_pages, nkv, page_size, D] (HND, trtllm default)
            kc = k_cache.view(-1, ps, nkv, head_dim).permute(0, 2, 1, 3)
            vc = v_cache.view(-1, ps, nkv, head_dim).permute(0, 2, 1, 3)
            return flashinfer.decode.trtllm_batch_decode_with_kv_cache(  # type: ignore
                query=q.contiguous(),
                kv_cache=(kc, vc),
                workspace_buffer=self.dense_backend.workspace_buffer,
                block_tables=page_table,
                seq_lens=real_seq_lens,
                max_seq_len=self.topk_blocks * self.block_size_k,
                bmm1_scale=layer.scaling,
                bmm2_scale=1.0,
            )
        raise NotImplementedError(
            "dense sparse decode currently supports trtllm_mha only (fa3 is TODO)"
        )

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        *,
        idx_q: torch.Tensor,
        idx_k: torch.Tensor,
        idx_v: Optional[torch.Tensor],
        **kwargs,
    ):
        assert len(kwargs) == 0
        disable_value = layer.layer_id in self.disable_value_layer_ids
        self.kv_pool.set_fused_kv_index_buffer(
            layer,
            forward_batch.out_cache_loc,
            k,
            v,
            idx_k,
            None if disable_value else idx_v,
            layer.k_scale_float,
            layer.v_scale_float,
            layer.idx_k_scale_float,
            layer.idx_v_scale_float,
        )
        k_cache, v_cache = self.kv_pool.get_kv_buffer(layer.layer_id)
        if disable_value:
            idx_k_cache = self.kv_pool.get_index_k_buffer(layer.layer_id)
            idx_v_cache = None
        else:
            idx_k_cache, idx_v_cache = self.kv_pool.get_index_kv_buffer(layer.layer_id)

        attn_fn = None
        if self.use_dense_sparse_decode and k_cache.shape[1] == 1:

            def attn_fn(main_q, page_table, real_seq_lens):
                return self._dense_sparse_main_decode(
                    main_q,
                    page_table,
                    real_seq_lens,
                    k_cache,
                    v_cache,
                    layer,
                    forward_batch,
                )

        msa_kv_indices = msa_plan = None
        if self._use_msa_decode and attn_fn is None:
            if self._msa_dec_meta is not None:
                msa_kv_indices, msa_plan = self._msa_dec_meta
            elif q.shape[0] > 0:
                # Rebuilding the plan inline would run host-side code inside
                # CUDA-graph capture; fail loudly instead.
                raise RuntimeError(
                    "MSA decode metadata missing: init_forward_metadata_out_graph "
                    "did not prepare the plan for this forward (gate mismatch)."
                )

        # fp8 attention GEMMs: quantize q/idx_q AFTER the KV store (which reads
        # the bf16 k/v).
        if self.fp8_attn_gemm:
            q = _quant_q_fp8(q, layer.q_scale_float)
            idx_q = _quant_q_fp8(idx_q, layer.idx_q_scale_float)

        idx_o, o = minimax_sparse_decode(
            q,
            None,
            k_cache,
            v_cache,
            idx_q,
            None,
            idx_k_cache,
            idx_v_cache,
            self.req_to_token,
            forward_batch.req_pool_indices,
            forward_batch.seq_lens,
            self._max_seqlen_k,
            1,
            self.block_size_k,
            self.topk_blocks,
            self.init_blocks,
            self.local_blocks,
            score_type=self.score_type,
            disable_index_value=disable_value,
            dense_main_attn_fn=attn_fn,
            page_size=self.page_size,
            use_msa=self._use_msa_decode,
            msa_kv_indices=msa_kv_indices,
            msa_plan=msa_plan,
            q_scale=layer.q_scale_float,
            k_scale=layer.k_scale_float,
            v_scale=layer.v_scale_float,
            idx_q_scale=layer.idx_q_scale_float,
            idx_k_scale=layer.idx_k_scale_float,
            idx_v_scale=layer.idx_v_scale_float,
        )
        return (
            None if idx_o is None else idx_o.reshape(q.shape[0], -1).contiguous(),
            o.reshape(q.shape[0], -1).contiguous(),
        )


class MiniMaxHybridAttnBackend(AttentionBackend):
    def __init__(
        self,
        dense_backend: AttentionBackend,
        sparse_backend: MiniMaxSparseAttnBackend,
        sparse_layer_ids: list[int],
    ):
        self.dense = dense_backend
        self.sparse = sparse_backend
        self.sparse_layer_ids = sparse_layer_ids
        self.sparse.dense_backend = dense_backend
        self.extend_dummy_seqs_capped_by_req_pool = getattr(
            dense_backend, "extend_dummy_seqs_capped_by_req_pool", False
        ) or getattr(sparse_backend, "extend_dummy_seqs_capped_by_req_pool", False)

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        self.sparse.init_forward_metadata(forward_batch)
        self.dense.init_forward_metadata(forward_batch)

    def init_forward_metadata_out_graph(
        self, forward_batch: ForwardBatch, in_capture: bool = False
    ):
        self.sparse.init_forward_metadata_out_graph(forward_batch, in_capture)
        self.dense.init_forward_metadata_out_graph(forward_batch, in_capture)

    def init_forward_metadata_in_graph(self, forward_batch: ForwardBatch):
        self.sparse.init_forward_metadata_in_graph(forward_batch)
        self.dense.init_forward_metadata_in_graph(forward_batch)

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        self.dense.init_cuda_graph_state(max_bs, max_num_tokens)
        self.sparse.init_cuda_graph_state(max_bs, max_num_tokens)

    def get_cuda_graph_seq_len_fill_value(self):
        return self.sparse.get_cuda_graph_seq_len_fill_value()

    def forward(
        self,
        q,
        k,
        v,
        layer,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        **kwargs,
    ):
        if layer.layer_id in self.sparse_layer_ids:
            return self.sparse.forward(
                q, k, v, layer, forward_batch, save_kv_cache, **kwargs
            )

        # DP attention pads q to an even length but flashinfer builds qo_indptr from
        # extend_seq_lens, so padded q.shape[0] != qo_indptr[-1] and paged-prefill
        # raises. Trim q and re-pad output; k/v stay untrimmed so KV-cache writes
        # align with out_cache_loc.
        mode = forward_batch.forward_mode
        if mode.is_extend() and forward_batch.extend_seq_lens_cpu is not None:
            actual_num_tokens = int(sum(forward_batch.extend_seq_lens_cpu))
            original_num_tokens = q.shape[0]
            if actual_num_tokens < original_num_tokens:
                o = self.dense.forward(
                    q[:actual_num_tokens],
                    k,
                    v,
                    layer,
                    forward_batch,
                    save_kv_cache,
                    **kwargs,
                )
                pad_len = original_num_tokens - actual_num_tokens
                return torch.cat([o, o.new_zeros(pad_len, *o.shape[1:])], dim=0)

        return self.dense.forward(
            q, k, v, layer, forward_batch, save_kv_cache, **kwargs
        )

    def forward_extend(
        self,
        q,
        k,
        v,
        layer,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        **kwargs,
    ):
        if layer.layer_id in self.sparse_layer_ids:
            return self.sparse.forward_extend(
                q, k, v, layer, forward_batch, save_kv_cache, **kwargs
            )
        else:
            return self.dense.forward_extend(
                q, k, v, layer, forward_batch, save_kv_cache, **kwargs
            )

    def forward_decode(
        self,
        q,
        k,
        v,
        layer,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        **kwargs,
    ):
        if layer.layer_id in self.sparse_layer_ids:
            return self.sparse.forward_decode(
                q, k, v, layer, forward_batch, save_kv_cache, **kwargs
            )
        else:
            return self.dense.forward_decode(
                q, k, v, layer, forward_batch, save_kv_cache, **kwargs
            )
