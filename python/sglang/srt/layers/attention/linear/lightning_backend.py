import logging
import math

import torch

from sglang.srt.layers.attention.hybrid_linear_attn_backend import MambaAttnBackendBase
from sglang.srt.layers.attention.linear.lightning_attn import (
    BailingLinearKernel,
    linear_decode_forward_triton,
)
from sglang.srt.layers.attention.linear.linear_metadata import (
    BailingLinearMetadata,
)
from sglang.srt.layers.attention.linear.seg_la import SegLaMeta, seg_la_fwd
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)


class LightningAttentionBackend(MambaAttnBackendBase):
    """
    Note about the init:
    - If no spec decoding
        - FlashAttentionBackend will be init once when the server starts.
    - If spec decoding
        - FlashAttentionBackend will be init once for the target worker
        - FlashAttentionMultiStepBackend will be once for the draft worker
            - It will spawn num_steps FlashAttentionBackend for the draft worker

    Note about CUDA Graph:
    - We only support CUDA Graph for Decode (Normal Decode and Draft Decode) and Target Verify.
    - We don't support CUDA Graph for Extend and Draft Extend.
    - When server init, init_cuda_graph_state will be called first and then init_cuda_graph_capture will be called.
    - For each forward batch, init_replay_cuda_graph will be called first and then replay the graph.
    """

    def __init__(self, model_runner: ModelRunner):
        super().__init__(model_runner)
        self._server_args = model_runner.server_args
        # seg_la processes draft tokens as a chain -- it has no parent-indices
        # plumbing for tree-shaped drafts, so spec v2 tree verify (topk > 1) would
        # commit wrong mamba states silently. Fail fast instead of mis-decoding.
        if self.topk > 1:
            raise NotImplementedError(
                "Lightning (seg_la) linear-attention backend does not support "
                f"speculative decoding with topk > 1 (got topk={self.topk}); "
                "seg_la verifies a draft tree as a chain. Use "
                "--speculative-eagle-topk 1."
            )
        # lightning attn does not need conv cache, but to keep the interface for mamba cache
        self.conv_states_shape = (
            model_runner.req_to_token_pool.mamba_pool.mamba_cache.conv[0].shape
        )

        assert not (
            model_runner.sliding_window_size is not None
            and model_runner.model_config.is_encoder_decoder
        ), "Sliding window and cross attention are not supported together"

        # extra metadata for handling speculative decoding topk > 1, extended draft decode and verify
        self.max_context_len = model_runner.model_config.context_len
        self.device = model_runner.device
        self.decode_cuda_graph_metadata = {}
        self.kv_cache_dtype = model_runner.kv_cache_dtype
        self.kv_cache_dtype_str = model_runner.server_args.kv_cache_dtype
        self.BLOCK = (
            model_runner.model_config.block
            if hasattr(model_runner.model_config, "block")
            else 256
        )
        total_num_heads = model_runner.model_config.hf_config.num_attention_heads
        num_hidden_layers = model_runner.model_config.hf_config.num_hidden_layers
        self.tp_slope = LightningAttentionBackend._build_slope_tensor(
            total_num_heads, num_hidden_layers, self.device
        )
        self.linear_backend = getattr(
            model_runner.model_config.hf_config, "linear_backend", "seg_la"
        )
        # seg_la_vk: same kernel family with [v,k] state layout for all kernels.
        # The state pool is square (H,D,D) — inner k/v ordering is kernel-internal.
        # The MTP scatter is a layout-agnostic whole-slot copy, so as long as every
        # seg_la kernel (including mtp's intermediate_ssm write) uses [v,k], the
        # path is end-to-end self-consistent with no memory_pool / scatter changes.
        self.seg_la_state_layout = "kv"
        if self.linear_backend == "seg_la_vk":
            self.seg_la_state_layout = "vk"
            self.linear_backend = (
                "seg_la"  # reuse all existing seg_la branches unchanged
            )
        elif self.linear_backend == "cula":
            # cuLA's state pool is V-major [v,k]; prefill reuses the seg_la
            # kernel family with the [v,k] layout so the state it writes is
            # directly consumable by cuLA's decode/verify/commit kernels.
            self.seg_la_state_layout = "vk"
        logger.info(
            f"linear_backend for linear attention in hybrid_linear_backend: {self.linear_backend}"
        )

        if self.linear_backend == "cula":
            self.head_dim = (
                model_runner.model_config.hf_config.head_dim
                if hasattr(model_runner.model_config.hf_config, "head_dim")
                else (
                    model_runner.model_config.hf_config.hidden_size
                    // model_runner.model_config.hf_config.num_attention_heads
                )
            )
            self.num_heads = total_num_heads // (
                model_runner.tp_size if hasattr(model_runner, "tp_size") else 1
            )

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        super().init_cuda_graph_state(max_bs, max_num_tokens)
        if self.linear_backend == "cula":
            self._warmup_cula_kernels(max_bs, max_num_tokens)

    def _warmup_cula_kernels(self, max_bs: int, max_num_tokens: int):
        from sglang.srt.layers.attention.linear.cula_entry import (
            cula_commit_fused,
            cula_decode,
            cula_verify,
        )

        draft_token_num = max_num_tokens // max_bs
        decay = self.tp_slope[0]
        scale = self.head_dim**-0.5
        temporal = self.req_to_token_pool.mamba_pool.mamba_cache.temporal[0]
        # Fused commit operates on the full per-layer state pool. Set up the
        # layer-fused warmup tensors once (bs-independent): real temporal_full
        # (3.8GB, not copyable) + small dummy draft buffers + stacked decay.
        temporal_full = self.req_to_token_pool.mamba_pool.mamba_cache.temporal
        num_layers = temporal_full.shape[0]
        pool_size = temporal_full.shape[1]
        mamba_layer_ids = list(self.req_to_token_pool.mamba_map.keys())
        decay_all = torch.stack(
            [self.tp_slope[lid].view(-1).contiguous() for lid in mamba_layer_ids]
        )
        dummy_dk_pool = torch.zeros(
            (num_layers, pool_size, draft_token_num, self.num_heads, self.head_dim),
            device=self.device,
            dtype=torch.float32,
        )
        dummy_dv_pool = torch.zeros_like(dummy_dk_pool)
        # decode and verify both take fp32 q/k/v natively (kernels compute in
        # fp32/TF32). cute.compile bakes the dtype seen at first call, so warmup
        # dtype MUST match runtime dtype (fp32 for both).
        decode_dtype = torch.float32
        verify_dtype = torch.float32
        out_dtype = torch.bfloat16

        # decode is CUDA-graphed -> sparse warmup suffices. verify + commit use
        # sym_int() for B (one compile covers all B) -> sparse warmup is enough.
        decode_warmup = sorted(
            bs for bs in {1, 2, 4, 8, 16, 32, 33, 64, max_bs} if bs <= max_bs
        )
        eager_warmup = [1, 32]  # trigger the single compile for verify + commit

        for bs in decode_warmup:
            dummy_q = torch.zeros(
                (bs, self.num_heads, self.head_dim),
                device=self.device,
                dtype=decode_dtype,
            )
            dummy_out = torch.zeros(
                (bs, self.num_heads, self.head_dim), device=self.device, dtype=out_dtype
            )
            dummy_idx = torch.zeros(bs, dtype=torch.int32, device=self.device)
            cula_decode(
                dummy_q, dummy_q, dummy_q, temporal, dummy_idx, decay, scale, dummy_out
            )
        if draft_token_num > 1:
            for bs in eager_warmup:
                total = bs * draft_token_num
                dummy_qv = torch.zeros(
                    (total, self.num_heads, self.head_dim),
                    device=self.device,
                    dtype=verify_dtype,
                )
                dummy_outv = torch.zeros(
                    (total, self.num_heads, self.head_dim),
                    device=self.device,
                    dtype=torch.float32,
                )
                dummy_idx = torch.zeros(bs, dtype=torch.int32, device=self.device)
                # Warm the write_kv=True variant (runtime uses fused draft writes).
                # k_buf/v_buf are per-layer pool-indexed [pool, T, H, K] fp32 slices.
                cula_verify(
                    dummy_qv,
                    dummy_qv,
                    dummy_qv,
                    temporal,
                    dummy_idx,
                    decay,
                    scale,
                    draft_token_num,
                    dummy_outv,
                    k_buf=dummy_dk_pool[0],
                    v_buf=dummy_dv_pool[0],
                )
                # Fused commit: one launch for ALL layers. draft_k/v dummies are
                # pool-indexed (bs-independent, allocated once); only h0_indices/
                # accepted_len vary with bs. Writes garbage into temporal_full slot
                # 0, zeroed back after the loop.
                dummy_acc = torch.full(
                    (bs,), draft_token_num, dtype=torch.int32, device=self.device
                )
                cula_commit_fused(
                    dummy_dk_pool,
                    dummy_dv_pool,
                    temporal_full,
                    dummy_idx,
                    dummy_acc,
                    decay_all,
                    draft_token_num,
                )
        # Restore the state pool to its initial zeros (fused-commit warmup wrote
        # garbage into the active slots).
        temporal_full.zero_()
        torch.cuda.synchronize()
        logger.info(
            f"cuLA kernel warmup complete (decode={len(decode_warmup)} bs, "
            f"eager verify+commit={len(eager_warmup)} bs)"
        )

    def init_forward_metadata_out_graph(
        self,
        forward_batch: ForwardBatch,
        in_capture: bool = False,
    ):
        # seq_lens_cpu is unused by the underlying _replay_metadata for
        # non-target-verify modes; pass it through for compatibility.
        bs = forward_batch.batch_size
        metadata = self._replay_metadata(
            bs,
            forward_batch.req_pool_indices,
            forward_batch.forward_mode,
            forward_batch.spec_info,
            forward_batch.seq_lens_cpu if not in_capture else None,
        )
        self.forward_metadata = BailingLinearMetadata.prepare_decode(
            metadata.query_start_loc,
            metadata.mamba_cache_indices,
            bs,
            forward_batch.seq_lens,
        )

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        metadata = self._forward_metadata(forward_batch)
        self.forward_metadata = BailingLinearMetadata.prepare_mixed(
            metadata.query_start_loc,
            metadata.mamba_cache_indices,
            forward_batch,
        )

    @staticmethod
    def _build_slope_tensor(
        n_attention_heads: int, num_hidden_layers: int, device="cuda"
    ):
        def get_slopes(n):
            def get_slopes_power_of_2(n):
                start = 2 ** (-(2 ** -(math.log2(n) - 3)))
                ratio = start
                return [start * ratio**i for i in range(n)]

            if math.log2(n).is_integer():
                return get_slopes_power_of_2(n)
            else:
                closest_power_of_2 = 2 ** math.floor(math.log2(n))
                return (
                    get_slopes_power_of_2(closest_power_of_2)
                    + get_slopes(2 * closest_power_of_2)[0::2][: n - closest_power_of_2]
                )

        slopes = torch.tensor(
            get_slopes(n_attention_heads), dtype=torch.float32
        ).reshape(n_attention_heads, 1, 1)
        from sglang.srt.layers.dp_attention import (
            get_attention_tp_rank,
            get_attention_tp_size,
        )

        tp_heads = n_attention_heads // get_attention_tp_size()
        tp_rank = get_attention_tp_rank()
        if num_hidden_layers <= 1:
            slope_rate_list = [slopes * (1 + 1e-5)]
        else:
            slope_rate_list = [
                slopes * (1 - layer_id / (num_hidden_layers - 1) + 1e-5)
                for layer_id in range(num_hidden_layers)
            ]

        tp_slope = [
            slope_rate_list[layer_id][tp_rank * tp_heads : (tp_rank + 1) * tp_heads]
            .contiguous()
            .to(device)
            for layer_id in range(num_hidden_layers)
        ]

        return tp_slope

    def _prefill_and_mix_infer(
        self,
        q,
        k,
        v,
        kv_cache,
        state_indices_tensor,
        forward_batch,
        layer,
        metadata,
    ):
        hidden = []
        for _prefill_idx in range(metadata.num_prefills):
            if _prefill_idx >= forward_batch.extend_start_loc.shape[0]:
                break
            if _prefill_idx >= state_indices_tensor.shape[0]:
                break

            _start = forward_batch.extend_start_loc[_prefill_idx]

            if _prefill_idx + 1 < forward_batch.extend_start_loc.shape[0]:
                _end = forward_batch.extend_start_loc[_prefill_idx + 1]
            else:
                if (
                    forward_batch.extend_seq_lens is not None
                    and _prefill_idx < forward_batch.extend_seq_lens.shape[0]
                    and metadata.num_decodes > 0
                ):
                    seq_len = forward_batch.extend_seq_lens[_prefill_idx]
                    _end = _start + seq_len
                else:
                    _end = q.shape[0]

            slot_id = state_indices_tensor[_prefill_idx]
            qs = q[_start:_end].transpose(0, 1).contiguous()
            ks = k[_start:_end].transpose(0, 1).contiguous()
            vs = v[_start:_end].transpose(0, 1).contiguous()
            slice_layer_cache = kv_cache[slot_id, ...]
            out_slice = BailingLinearKernel.jit_linear_forward_prefix(
                qs,
                ks,
                vs,
                slice_layer_cache,
                self.tp_slope[layer.layer_id],
                self.BLOCK,
                layer_idx=layer.layer_id,
            )
            hidden.append(out_slice.contiguous())
        if metadata.num_decodes > 0:
            hidden.append(
                self._decode_infer(
                    q, k, v, kv_cache, state_indices_tensor, metadata, layer
                )
            )

        if not hidden:
            return torch.empty((0, q.size(-1)), device=q.device, dtype=q.dtype)

        hidden = torch.concat(hidden, dim=0).contiguous()
        return hidden

    def _decode_infer(self, q, k, v, kv_cache, state_indices_tensor, metadata, layer):
        num_prefill_tokens = metadata.num_prefill_tokens
        num_prefills = metadata.num_prefills
        q = q[num_prefill_tokens:].unsqueeze(2).contiguous()
        k = k[num_prefill_tokens:].unsqueeze(2).contiguous()
        v = v[num_prefill_tokens:].unsqueeze(2).contiguous()
        slot_id = state_indices_tensor[num_prefills:]

        assert slot_id.shape[0] == q.shape[0], (
            f"slot_id length {slot_id.shape[0]} does not match decode batch size {q.shape[0]}. "
            "This indicates a bug in the upstream logic that should be investigated."
        )
        hidden = linear_decode_forward_triton(
            q, k, v, kv_cache, self.tp_slope[layer.layer_id], slot_id, 32
        )
        return hidden

    def _linear_attention_entry(
        self,
        q,
        k,
        v,
        kv_cache,
        state_indices_tensor,
        metadata,
        layer,
        mask=None,
        temp_cache=None,
        intermediate_state_indices=None,
    ):
        q_offsets = metadata.query_start_loc

        seg_meta = SegLaMeta(
            batch_size=metadata.batch_size,
            q_offsets=metadata.query_start_loc,
            s_offsets=state_indices_tensor,
            q_lengths=q_offsets.diff(),
            s_scales=metadata.has_initial_states,
            max_q_length=None,
            mask=mask,
        )
        hidden = seg_la_fwd(
            q=q,
            k=k,
            v=v,
            s=kv_cache,
            decay_scales=self.tp_slope[layer.layer_id],
            meta=seg_meta,
            caches=temp_cache,
            cache_indices=intermediate_state_indices,
            decouple=True,
            state_layout=self.seg_la_state_layout,
        )
        return hidden

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
        **kwargs,
    ):
        layer_id = layer.layer_id if layer else kwargs["layer_id"]

        metadata = self.forward_metadata

        if self.kv_cache_dtype_str != "auto" and layer.k_scale is not None:
            q = q.to(self.kv_cache_dtype)

        cache_indices = self.forward_metadata.mamba_cache_indices
        mamba_cache_params = self.req_to_token_pool.mamba2_layer_cache(layer_id)
        ssm_states = mamba_cache_params.temporal
        if self.linear_backend == "minimax":
            o = self._prefill_and_mix_infer(
                q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim),
                k,
                v,
                ssm_states,
                cache_indices,
                forward_batch,
                layer,
                metadata,
            )
        elif self.linear_backend == "seg_la":
            intermediate_state_indices = (
                torch.arange(
                    cache_indices.shape[0],
                    dtype=torch.int32,
                    device=cache_indices.device,
                )
                if forward_batch.forward_mode.is_target_verify()
                else None
            )
            o = self._linear_attention_entry(
                q,
                k,
                v,
                ssm_states,
                cache_indices,
                metadata,
                layer,
                temp_cache=(
                    mamba_cache_params.intermediate_ssm
                    if forward_batch.forward_mode.is_target_verify()
                    else None
                ),
                intermediate_state_indices=intermediate_state_indices,
            )
        elif self.linear_backend == "cula":
            from sglang.srt.layers.attention.linear.cula_entry import cula_verify

            decay = self.tp_slope[layer_id]
            scale = layer.head_dim**-0.5
            if forward_batch.forward_mode.is_target_verify():
                T = forward_batch.spec_info.draft_token_num
                # cula_verify accepts fp32 q/k/v natively and fuses the draft_k/v
                # writes (write_kv=True) -- no separate scatter kernels. draft_k/v
                # are per-layer [pool, T, H, K] / [pool, T, HV, V] fp32 buffers.
                out = torch.empty_like(v, dtype=torch.float32)
                o = cula_verify(
                    q,
                    k,
                    v,
                    ssm_states,
                    cache_indices,
                    decay,
                    scale,
                    T,
                    out,
                    k_buf=mamba_cache_params.draft_k,
                    v_buf=mamba_cache_params.draft_v,
                )
            else:
                # Prefill via the seg_la kernel family with [v,k] state layout
                # (self.seg_la_state_layout == "vk" for cula), so the state
                # written here matches cuLA's V-major state pool.
                o = self._linear_attention_entry(
                    q,
                    k,
                    v,
                    ssm_states,
                    cache_indices,
                    metadata,
                    layer,
                )
        else:
            raise ValueError(
                f"linear backend: {self.linear_backend} is not support for now"
            )

        if (
            not forward_batch.forward_mode.is_target_verify()
            and forward_batch.mamba_track_mask is not None
        ):
            # save mamba cache for extra buffer
            mamba_track_mask = forward_batch.mamba_track_mask
            mamba_track_indices = forward_batch.mamba_track_indices
            dst_masked = mamba_track_indices[mamba_track_mask]
            src_masked = metadata.mamba_cache_indices[mamba_track_mask]
            ssm_states[dst_masked] = ssm_states[src_masked]

        return o.view(-1, layer.tp_q_head_num * layer.v_head_dim)

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
        **kwargs,
    ) -> torch.Tensor:
        layer_id = layer.layer_id if layer else kwargs["layer_id"]

        # Use precomputed metadata across all layers
        metadata = self.forward_metadata

        if self.kv_cache_dtype_str != "auto":
            q = q.to(self.kv_cache_dtype)

        # Do linear attention
        cache_indices = self.forward_metadata.mamba_cache_indices
        mamba_cache_params = self.req_to_token_pool.mamba2_layer_cache(layer_id)
        ssm_states = mamba_cache_params.temporal
        if self.linear_backend == "minimax":
            o = self._decode_infer(q, k, v, ssm_states, cache_indices, metadata, layer)
        elif self.linear_backend == "seg_la":
            o = self._linear_attention_entry(
                q, k, v, ssm_states, cache_indices, metadata, layer
            )
        elif self.linear_backend == "cula":
            from sglang.srt.layers.attention.linear.cula_entry import cula_decode

            decay = self.tp_slope[layer_id]
            scale = layer.head_dim**-0.5
            # Pass q/k/v through in native dtype (fp32); cula_decode computes in
            # fp32 internally. Pre-allocated bf16 out so no graph re-alloc.
            out = torch.empty_like(v, dtype=torch.bfloat16)
            o = cula_decode(q, k, v, ssm_states, cache_indices, decay, scale, out)
        else:
            raise ValueError(
                f"linear backend: {self.linear_backend} is not support for now"
            )
        return o.view(-1, layer.tp_q_head_num * layer.v_head_dim)
