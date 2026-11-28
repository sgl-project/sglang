# Copyright 2026 Baidu ERNIE Team and The SGLang Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import annotations

import torch

from sglang.kernels.ops.attention.dsa.append_window_to_topk import (
    append_window_to_topk,
)
from sglang.kernels.ops.attention.dsa.indexer_q import (
    fused_q_indexer_rope_first_quant,
)
from sglang.kernels.ops.attention.ernie55.hca_qkv_prep import hca_qkv_prep
from sglang.kernels.ops.attention.ernie55.swapab_attention import (
    swapab_combine_local,
    swapab_partial,
    swapab_partial_mla,
)
from sglang.kernels.ops.communication.ernie55_combine_push import swapab_combine_push
from sglang.kernels.ops.communication.nvlink_comm import all_gather_push
from sglang.kernels.ops.quantization.act_quant_triton import act_quant
from sglang.srt.distributed.device_communicators.custom_all_reduce_v2 import (
    CustomAllReduceV2,
)
from sglang.srt.distributed.parallel_state import get_tp_group
from sglang.srt.layers.attention.dsa.dsa_indexer import (
    _broadcast_indexer_topk_from_rank0,
)
from sglang.srt.layers.dp_attention import attn_tp_all_gather_into_tensor
from sglang.srt.layers.layer_boundary import AttentionInputs, get_attn_tp_context
from sglang.srt.layers.quantization.ernie55_low_latency_gemm import (
    Ernie5_5LowLatencyGemm,
)
from sglang.srt.layers.quantization.halops_mxfp8 import (
    HALOPS_MODE,
    swizzle_scale_to_sfatom,
)
from sglang.srt.model_executor.forward_context import (
    get_attn_backend,
    get_token_to_kv_pool,
)
from sglang.srt.model_executor.runner import get_is_capture_mode
from sglang.srt.models.deepseek_v2 import DeepseekV2AttentionMLA
from sglang.srt.models.deepseek_v4 import MQALayer
from sglang.srt.models.deepseek_v4_mhc import AttnOutput
from sglang.srt.models.ernie55 import Ernie5_5HcaAttention, Ernie5_5MlaAttention
from sglang.srt.runtime_context import get_parallel
from sglang.srt.state_capturer.indexer_topk import maybe_capture_indexer_topk
from sglang.srt.utils import print_info_once


class Ernie5_5LowLatencyAttentionMixin:
    def _init_low_latency(
        self, last_linear: torch.nn.Module, supported: bool = True
    ) -> None:
        group = get_tp_group()
        comm = group.ca_comm
        self._enabled = (
            supported
            and self._gate_stream is not None
            and get_parallel().attn_tp_size == group.world_size
            and isinstance(comm, CustomAllReduceV2)
            and not comm.disabled
        )
        if not self._enabled:
            print_info_once(
                f"ERNIE5.5 low-latency {type(self).__name__}: stock path "
                "(needs multi-stream overlap and custom all-reduce v2 over attn TP; "
                "MLA also needs an fp8 KV cache and DSA indexer fusion)"
            )
            return
        qm = last_linear.quant_method
        process = qm.process_weights_after_loading

        def wrapped(layer, *a, **k):
            process(layer, *a, **k)
            self._convert_weights()

        qm.process_weights_after_loading = wrapped

    def _use_low_latency(self, x, forward_batch) -> bool:
        mode = forward_batch.forward_mode
        return (
            get_is_capture_mode()
            and (mode.is_decode() or mode.is_target_verify())
            and 1 <= x.shape[0] <= Ernie5_5LowLatencyGemm.MAX_TOKENS
            and self._enabled
        )

    def _convert_weights(self) -> None:
        attn_tp_size = get_parallel().attn_tp_size
        u_local = self.postmix.vha_postmix_U.data
        v_local = self.postmix.vha_postmix_V.data
        u_all = u_local.new_empty((attn_tp_size, *u_local.shape))
        v_all = torch.empty_like(u_all)
        attn_tp_all_gather_into_tensor(u_all, u_local.contiguous())
        attn_tp_all_gather_into_tensor(v_all, v_local.contiguous())
        num_heads = attn_tp_size * u_local.shape[0]
        u_full = u_all.reshape(num_heads, -1).float()
        v_full = v_all.reshape(num_heads, -1).float()
        eye = torch.eye(num_heads, dtype=torch.float32, device=u_full.device)
        self._full_mix = (
            torch.addmm(eye, v_full, u_full.t()).to(torch.bfloat16).contiguous()
        )


class Ernie5_5LowLatencyHcaAttention(
    Ernie5_5LowLatencyAttentionMixin, Ernie5_5HcaAttention
):
    _late_sync: dict = {}

    def __init__(self, config, layer_id, quant_config, prefix, alt_streams=None):
        super().__init__(config, layer_id, quant_config, prefix, alt_streams)
        self._gate_stream = alt_streams[-1] if alt_streams else None
        self._init_low_latency(self.wo_a)

    def forward(
        self, x, positions, forward_batch, x_quant=None, defer_all_reduce=False
    ):
        if self._use_low_latency(x, forward_batch):
            return self._low_latency_forward(
                x, positions, forward_batch, x_quant, defer_all_reduce
            )
        return super().forward(x, positions, forward_batch, x_quant, defer_all_reduce)

    def _convert_weights(self) -> None:
        super()._convert_weights()
        wo_a = self.wo_a
        device = wo_a.weight.device
        sms = torch.cuda.get_device_properties(device).multi_processor_count
        wo_a_sf = Ernie5_5LowLatencyGemm.block_scale_group32(
            wo_a.weight_scale_inv.data,
            self.o_lora_rank,
            wo_a.weight.shape[1],
            wo_a.quant_method.weight_block_size,
        )
        shift = self.attn_tp_rank * self.n_local_heads
        self._prepared = (
            *Ernie5_5LowLatencyGemm.split_rows(self.wq_b, self.q_up_local_size),
            sms * 15 // 19,
            sms * 8 // 19,
            torch.view_as_real(self.freqs_cis).flatten(-2).contiguous().float(),
            torch.roll(self._full_mix, -shift, dims=0).contiguous(),
            swizzle_scale_to_sfatom(wo_a_sf.flatten(0, -2)).view(wo_a_sf.shape),
        )
        if device not in self._late_sync:
            self._late_sync[device] = torch.zeros(
                8 + Ernie5_5LowLatencyGemm.MAX_TOKENS, dtype=torch.int32, device=device
            )

    def _low_latency_forward(
        self, x, positions, forward_batch, x_quant, defer_all_reduce
    ):
        attn_backend = get_attn_backend()
        pool = get_token_to_kv_pool()
        num_tokens = int(x.shape[0])
        q_plan, gate_plan, q_up_sms, gate_sms, freqs, rank_mix, wo_a_sfb = (
            self._prepared
        )
        current_stream = torch.cuda.current_stream()
        stream_compressor = self.alt_streams[1]
        gate_stream = self._gate_stream

        stream_compressor.wait_stream(current_stream)
        qkv_a, _ = self.wqkv_a(x_quant if x_quant is not None else x)
        prequant = hca_qkv_prep(
            qkv_a,
            self.q_norm.weight,
            self.q_norm.variance_epsilon,
            self.kv_norm.weight.data,
            self.eps,
            freqs,
            positions,
            attn_backend.get_swa_out_cache_loc(forward_batch),
            pool.get_swa_raw_buffer(self.layer_id),
            pool.swa_kv_pool.page_size,
            self.q_lora_rank,
        )
        with torch.cuda.stream(stream_compressor):
            attn_backend.forward_core_compressor(
                x, forward_batch, self.layer_id, self.compressor
            )
        q16 = q_plan.run(prequant, q_up_sms).view(
            num_tokens, self.n_local_heads, self.head_dim
        )

        core = attn_backend.forward_metadata.core_attn_metadata
        swa_cache = pool.get_swa_key_buffer_radix(self.layer_id)
        swa_page = pool.get_swa_key_page_size()
        swa_bytes = pool.get_swa_key_bytes_per_token()
        extra_cache = pool.get_extra_key_buffer(self.layer_id)
        extra_page = pool.get_extra_key_page_size(self.layer_id)
        extra_bytes = pool.get_extra_key_bytes_per_token(self.layer_id)
        extra_indices = core.sparse_page_indices(self.compress_ratio)
        extra_lengths = core.sparse_topk_lengths(self.compress_ratio)
        tokens_per_req = num_tokens // forward_batch.batch_size
        partial_args = dict(
            extra_kv=extra_cache[:, : extra_page * extra_bytes].view(
                extra_cache.shape[0], extra_page, extra_bytes
            ),
            extra_indices=extra_indices,
            extra_lengths=extra_lengths,
            kv_layout=pool.get_swa_key_layout(),
            freqs=freqs,
            positions=positions,
            q_eps=self.eps,
            tokens_per_req=tokens_per_req,
            compress_ratio=self.compress_ratio,
        )
        swa_args = (
            q16,
            swa_cache[:, : swa_page * swa_bytes].view(
                swa_cache.shape[0], swa_page, swa_bytes
            ),
            core.swa_page_indices,
            core.swa_topk_lengths,
            attn_backend.softmax_scale,
        )
        q_ready = current_stream.record_event()
        partial, maximum, sums, nt = swapab_partial(
            *swa_args, tile_mode=1, **partial_args
        )
        partial_done = current_stream.record_event()

        gate_stream.wait_event(partial_done)
        for t in prequant:
            t.record_stream(gate_stream)
        with torch.cuda.stream(gate_stream):
            gate = gate_plan.run(prequant, gate_sms)

        sync = self._late_sync[x.device]
        stream_compressor.wait_event(q_ready)
        with torch.cuda.stream(stream_compressor):
            swapab_partial(
                *swa_args,
                tile_mode=2,
                out=(partial, maximum, sums),
                late_sync=sync,
                **partial_args,
            )
        for t in (q16, partial, maximum, sums):
            t.record_stream(stream_compressor)
        late_sync = (
            sync,
            extra_lengths,
            positions,
            tokens_per_req,
            self.compress_ratio,
        )

        o_global = swapab_combine_push(
            get_tp_group().ca_comm,
            partial,
            maximum,
            sums,
            self._local_attn_sink(self.n_local_heads),
            nt,
            late_sync,
        )

        from halops import batch_gemm, vha_postmix_quant

        o_fp8, o_sf = vha_postmix_quant(
            o_global, rank_mix, freqs, positions, groups=self.n_groups
        )
        o = torch.empty(
            (num_tokens, self.n_local_groups * self.o_lora_rank),
            dtype=torch.bfloat16,
            device=x.device,
        )
        batch_gemm(
            o_fp8[: self.n_local_groups],
            self.wo_a.weight.view(self.n_local_groups, self.o_lora_rank, -1),
            o_sf,
            wo_a_sfb,
            o,
            mode=HALOPS_MODE,
        )

        current_stream.wait_stream(stream_compressor)
        current_stream.wait_stream(gate_stream)
        gate.record_stream(current_stream)
        from halops import sigmoid_gate_mul_group_quant

        wo_b_input = sigmoid_gate_mul_group_quant(o, gate)
        return MQALayer._project_wo_b(self, wo_b_input, defer_all_reduce)


class Ernie5_5LowLatencyMlaAttention(
    Ernie5_5LowLatencyAttentionMixin, Ernie5_5MlaAttention
):
    _step_cache: dict = {}

    def __init__(self, config, layer_id, quant_config, prefix, alt_streams=None):
        super().__init__(config, layer_id, quant_config, prefix)
        self._q_stream = self._k_stream = self._gate_stream = None
        if alt_streams:
            self._q_stream, self._k_stream = alt_streams[1], alt_streams[2]
            self._gate_stream = alt_streams[-1]
        self._init_low_latency(
            self.q_b_proj,
            supported=self.kv_cache_dtype == "fp8_e4m3"
            and self.indexer.use_dsa_indexer_fusion,
        )

    def forward(
        self,
        positions,
        hidden_states,
        forward_batch,
        zero_allocator,
        x_quant=None,
        defer_all_reduce: bool = False,
    ):
        if not defer_all_reduce:
            return super().forward(
                positions, hidden_states, forward_batch, zero_allocator
            )
        if x_quant is not None and self._use_low_latency(hidden_states, forward_batch):
            with get_attn_backend().mla_layer_context():
                return AttnOutput(
                    self._low_latency_forward(
                        positions, hidden_states, forward_batch, x_quant
                    )
                )
        get_attn_tp_context().set_attn_inputs(
            AttentionInputs(hidden_states, forward_batch, self.prepare_qkv_latent)
        )
        try:
            with get_attn_backend().mla_layer_context():
                out = DeepseekV2AttentionMLA.forward(
                    self, positions, hidden_states, forward_batch, zero_allocator
                )
        finally:
            get_attn_tp_context().clear_attn_inputs()
        return AttnOutput(out[0])

    def _step_metadata(self, backend, forward_batch, num_tokens: int):
        layer_id, meta = self.layer_id, backend.forward_metadata
        entry = self._step_cache.get(backend)
        fresh = (
            entry is None
            or entry[0] is not forward_batch
            or entry[1] is not meta
            or layer_id <= entry[2]
        )
        if fresh:
            index_meta = backend.get_indexer_metadata(layer_id, forward_batch)
            seq_lens = meta.dsa_seqlens_expanded
            window_rows = append_window_to_topk(
                seq_lens.new_empty((num_tokens, 1), dtype=torch.int32)[:, :0],
                seq_lens,
                meta.real_page_table,
                window=backend.window,
                page_size=meta.page_size,
                alignment=1,
                page_rows=meta.token_to_batch_idx,
                ragged_offsets=meta.topk_indices_offset,
            )
        else:
            index_meta, window_rows = entry[3], entry[4]
        self._step_cache[backend] = (
            forward_batch,
            meta,
            layer_id,
            index_meta,
            window_rows,
        )
        return index_meta, window_rows, fresh

    def _low_latency_forward(self, positions, x, forward_batch, x_quant):
        num_tokens = int(x.shape[0])
        backend = get_attn_backend()
        indexer = self.indexer
        current_stream = torch.cuda.current_stream()
        q_stream, k_stream = self._q_stream, self._k_stream
        k_stream.wait_stream(current_stream)
        with torch.cuda.stream(k_stream):
            kw = indexer.wk_weights_proj(x)[0]
            key, head_gate = kw.split([indexer.head_dim, indexer.n_heads], dim=-1)
            indexer._fused_k_prepare_and_store(
                key, positions, forward_batch, self.layer_id, act_quant
            )
            index_meta, window_rows, step_meta_fresh = self._step_metadata(
                backend, forward_batch, num_tokens
            )
        qkv = self.fused_qkv_a_proj_with_mqa(x_quant)[0]
        q, latent = qkv.split(
            [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim], dim=-1
        )
        norm = self.q_a_layernorm
        from halops import rms_norm_quant

        q_quant = rms_norm_quant(
            q, norm.weight, norm.variance_epsilon, use_swizzle=False
        )
        index_q = indexer.wq_b(q_quant)[0]
        index_q = index_q.view(-1, indexer.n_heads, indexer.head_dim)
        q_nope = x.new_empty(
            (num_tokens, self.num_local_heads, self.kv_lora_rank), dtype=self.w_kc.dtype
        )
        q_stream.wait_stream(current_stream)
        with torch.cuda.stream(q_stream):
            k_nope = self.kv_a_layernorm(latent[..., : self.kv_lora_rank]).unsqueeze(1)
            q_up = self.q_b_proj(q_quant)[0].view(
                -1, self.num_local_heads, self.qk_head_dim
            )
            q_up_nope, q_pe, k_pe = self._split_q_nope_pe(q_up, latent)
            torch.bmm(q_up_nope.transpose(0, 1), self.w_kc, out=q_nope.transpose(0, 1))
            q_pe, k_pe = self.rotary_emb(positions, q_pe, k_pe)
            backend.kv_layers.layer(self.attn_mqa.layer_id).store_mla(
                self.attn_mqa, forward_batch.out_cache_loc, k_nope, k_pe
            )
        current_stream.wait_stream(k_stream)
        head_gate.record_stream(current_stream)
        if step_meta_fresh:
            index_meta.attn_metadata.dsa_seqlens_expanded.record_stream(current_stream)
            for t in window_rows:
                t.record_stream(current_stream)
        index_q_fp8, index_weights = fused_q_indexer_rope_first_quant(
            index_q.contiguous(),
            head_gate,
            indexer.softmax_scale * indexer.n_heads**-0.5,
            indexer._indexer_cos_sin_cache,
            positions,
            hadamard=indexer.fused_indexer_hadamard,
            neox_style=indexer.is_neox_style,
        )
        topk_indices = indexer._get_topk_paged(
            forward_batch, self.layer_id, index_q_fp8, index_weights, index_meta
        )
        topk_indices = maybe_capture_indexer_topk(
            self.layer_id, _broadcast_indexer_topk_from_rank0(topk_indices)
        )
        current_stream.wait_stream(q_stream)
        q_pe.record_stream(current_stream)
        gate_stream = self._gate_stream
        gate_stream.wait_stream(current_stream)
        with torch.cuda.stream(gate_stream):
            gate = self.gate_proj(q_quant)[0]
        window_slots, window_lengths = window_rows
        partial, maximum, sums, nt = swapab_partial_mla(
            q_nope,
            q_pe,
            backend.kv_layers.read(self.attn_mqa.layer_id).key(),
            topk_indices.to(torch.int32),
            index_meta.attn_metadata.dsa_seqlens_expanded,
            window_slots,
            window_lengths,
            self.attn_mqa.scaling,
        )
        attn = swapab_combine_local(
            partial, maximum, sums, self.learnable_sink_param.detach(), nt
        )
        tp_rank = get_parallel().attn_tp_rank
        n_local, num_heads = self.num_local_heads, self.num_heads
        mix_local = self._full_mix[tp_rank * n_local : (tp_rank + 1) * n_local]
        rows, head_dim = attn.shape[0], self.v_head_dim
        # Head-major so the rank-major gather lands as [num_heads, rows, head_dim].
        o_local = attn.new_empty((n_local, rows, head_dim))
        torch.bmm(attn.transpose(0, 1), self.w_vc, out=o_local)
        o_full = o_local.new_empty((num_heads, rows, head_dim))
        all_gather_push(
            get_tp_group().ca_comm.obj,
            o_local.view(-1, head_dim),
            o_full.view(-1, head_dim),
        )
        o = torch.matmul(mix_local, o_full.transpose(0, 1)).flatten(1)
        current_stream.wait_stream(gate_stream)
        gate.record_stream(current_stream)
        from halops import sigmoid_gate_mul_group_quant

        o = sigmoid_gate_mul_group_quant(o, gate)
        return self.o_proj(o)[0]
