# Copyright 2026 SGLang Team
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
"""Attention backend for the UltraQuant 4-bit KV cache.

Everything about metadata, CUDA/HIP graph capture, and scheduling is inherited
from the Triton backend. Only the two kernel calls change: K and V are read as
packed FP4 codes plus UE8M0 group scales rather than as plain bf16, and
queries are Hadamard-rotated first so they sit in the same basis as the keys,
which were rotated when they were stored.

Extend reads through the unified index list either way, so prefill sees the
same quantized keys decode will read later rather than exact values for the
current chunk. Large chunks dequantize that run once and hand a dense tile to
flash attention; small ones stay on the Triton kernel.
"""

from typing import Optional

import torch

from sglang.kernels.ops.kvcache.kv_indices import (
    create_flashinfer_kv_indices_triton,
    kv_indices_num_token_blocks,
)
from sglang.srt.layers.attention.triton_backend import (
    TritonAttnBackend,
    logit_capping_mod,
)
from sglang.srt.layers.quantization.fp4_kv_cache_quant_method import (
    UltraQuantKVCacheMethod,
)
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.mem_cache.memory_pool import KVWriteLoc
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.utils import is_gfx95_supported

try:
    from aiter import flash_attn_varlen_fp8_pertensor_func, flash_attn_varlen_func
except ImportError:
    flash_attn_varlen_func = flash_attn_varlen_fp8_pertensor_func = None

# Below this chunk size the dequantized run is too small to pay back the extra
# pass over the KV, so the Triton kernel stays ahead.
_DENSE_PREFILL_MIN_CHUNK = 128

# CK's batch-prefill kernels assert on wider heads.
_CK_MAX_HEAD_DIM = 256


class UltraQuantAttnBackend(TritonAttnBackend):
    """Triton attention that consumes the UltraQuant KV pool natively."""

    def __init__(
        self,
        model_runner,
        skip_prefill: bool = False,
        kv_indptr_buf: Optional[torch.Tensor] = None,
    ):
        super().__init__(
            model_runner, skip_prefill=skip_prefill, kv_indptr_buf=kv_indptr_buf
        )
        # The pool stores packed value codes, so size the split-KV workspace
        # from the model's logical value head dim.
        self.v_head_dim = model_runner.model_config.v_head_dim

        from sglang.kernels.ops.attention.flydsl.ultraquant_decode import (
            flydsl_ultraquant_decode,
            flydsl_ultraquant_decode_fits,
            is_flydsl_ultraquant_decode_supported,
            ultraquant_decode_launch_config,
            ultraquant_decode_max_kv_splits,
            ultraquant_decode_reduce,
        )
        from sglang.kernels.ops.attention.ultraquant_decode_attention import (
            decode_attention_fwd_ultraquant,
        )
        from sglang.kernels.ops.attention.ultraquant_extend_attention import (
            extend_attention_fwd_ultraquant,
        )
        from sglang.kernels.ops.kvcache.ultraquant import (
            ultraquant_gather_dequant,
            ultraquant_rotate,
        )

        self.ultraquant_rotate = torch.compiler.disable(ultraquant_rotate)
        self.ultraquant_gather_dequant = torch.compiler.disable(
            ultraquant_gather_dequant
        )
        self._dense_prefill_kv: Optional[torch.Tensor] = None
        self.flydsl_ultraquant_decode = torch.compiler.disable(flydsl_ultraquant_decode)
        self.ultraquant_decode_reduce = torch.compiler.disable(ultraquant_decode_reduce)
        self._flydsl_fits = flydsl_ultraquant_decode_fits
        self.decode_attention_fwd_ultraquant = torch.compiler.disable(
            decode_attention_fwd_ultraquant
        )
        self.extend_attention_fwd_ultraquant = torch.compiler.disable(
            extend_attention_fwd_ultraquant
        )

        # FlyDSL picks its split count per launch, so the split buffers only
        # need to be wide enough for the smallest batch at the max context.
        self.ultraquant_decode_launch_config = ultraquant_decode_launch_config
        self.min_kv_splits = self.max_kv_splits
        self._flydsl_decode = is_flydsl_ultraquant_decode_supported(
            model_runner.model_config.head_dim,
            self.num_head // self.num_kv_head,
            model_runner.model_config.dtype,
        )
        if self._flydsl_decode:
            self.max_kv_splits = ultraquant_decode_max_kv_splits(
                self.max_kv_splits, self.max_context_len
            )
        if self.enable_deterministic:
            # A batch-dependent split count would break batch invariance.
            self.min_kv_splits = self.max_kv_splits

        self._verify_pool_recipe()
        # These layers hold plain K/V, so the stock Triton path serves them.
        self.full_precision_layers = (
            self._ultraquant_pool().quant_method.full_precision_layers
        )

        # E2M1 times a power of two dequantizes exactly into E4M3, so where
        # aiter's FP8 prefill kernel covers the shape the dense run is FP8.
        self._fp8_dense_prefill = False
        if flash_attn_varlen_fp8_pertensor_func is not None and is_gfx95_supported():
            from sglang.srt.layers.attention.aiter_backend import (
                _aiter_fp8_asm_supports_gqa,
            )

            self._fp8_dense_prefill = (
                model_runner.model_config.head_dim == 256
                and self.v_head_dim == 256
                and _aiter_fp8_asm_supports_gqa(self.num_head, self.num_kv_head)
            )
            self._unit_scale = torch.ones(1, dtype=torch.float32, device=self.device)
        self._unified_kv = None
        self._dense_run = None

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        super().init_forward_metadata(forward_batch)
        self._unified_kv = None
        self._dense_run = None

    def _fill_kv_indptr_and_indices(
        self,
        bs: int,
        seq_lens: torch.Tensor,
        req_pool_indices: torch.Tensor,
        kv_indices: torch.Tensor,
    ) -> torch.Tensor:
        # One program per request copies a long context serially and runs every
        # decode step, so spread each request's copy over token blocks.
        translator = self.kv_index_translator
        if translator.reads_are_translated:
            return super()._fill_kv_indptr_and_indices(
                bs, seq_lens, req_pool_indices, kv_indices
            )
        seq_lens = seq_lens[:bs]
        kv_indptr = self.kv_indptr[: bs + 1]
        kv_indptr[1:] = torch.cumsum(seq_lens, dim=0)
        req_to_token = translator.req_to_token
        num_blocks = kv_indices_num_token_blocks(req_to_token.shape[1], bs)
        create_flashinfer_kv_indices_triton[(bs, num_blocks)](
            req_to_token,
            req_pool_indices,
            seq_lens,
            kv_indptr,
            None,
            kv_indices,
            req_to_token.stride(0),
            TOKEN_BLOCK_PARALLEL=num_blocks > 1,
        )
        return kv_indptr

    def _verify_pool_recipe(self) -> None:
        """Fail fast: any other pool layout would silently give wrong numbers."""
        pool = self._ultraquant_pool()
        quant_method = getattr(pool, "quant_method", None)
        if not isinstance(quant_method, UltraQuantKVCacheMethod):
            raise RuntimeError(
                "The ultraquant attention backend requires an UltraQuant KV pool, "
                f"but got {type(pool).__name__} with quant method "
                f"{type(quant_method).__name__}. Sliding-window hybrid pools "
                "are not supported."
            )

    def _ultraquant_pool(self):
        """Return the pool that owns the packed buffers.

        Hybrid models wrap the full-attention pool inside a linear-attention
        pool, so unwrap one level when that indirection is present.
        """
        pool = self.token_to_kv_pool
        return getattr(pool, "full_kv_pool", pool)

    def _kv_buffers(self, layer_id: int):
        # Goes through the outer pool so hybrid models get their global layer
        # id translated to the dense full-attention index.
        return self.token_to_kv_pool.get_raw_kv_buffer(layer_id)

    def _use_flydsl_decode(
        self, q: torch.Tensor, layer: RadixAttention, sinks, logit_cap
    ) -> bool:
        """Whether the gfx950 FlyDSL decode covers this layer (no sinks or capping)."""
        if not self._flydsl_decode or sinks is not None or logit_cap:
            return False
        if layer.qk_head_dim != layer.v_head_dim:
            return False
        return self._flydsl_fits(q.shape[0], layer.tp_q_head_num, self.max_kv_splits)

    def _use_dense_prefill(
        self, layer: RadixAttention, sinks, logit_cap, sliding_window_size: int
    ) -> bool:
        """Whether this extend should dequantize the run and call flash attention.

        Unpacking inside a flash kernel repeats per query block, so long chunks
        are cheaper dequantized once. Sinks, logit capping and sliding windows
        stay on the Triton kernel.
        """
        if flash_attn_varlen_func is None:
            return False
        # The dense path reads the KV length on the host.
        if torch.cuda.is_current_stream_capturing():
            return False
        max_extend_len = self.forward_metadata.max_extend_len
        if max_extend_len is None or max_extend_len <= _DENSE_PREFILL_MIN_CHUNK:
            return False
        if sinks is not None or logit_cap:
            return False
        if sliding_window_size > 0:
            return False
        if self.forward_metadata.custom_mask is not None:
            return False
        if layer.is_cross_attention:
            return False
        if layer.qk_head_dim != layer.v_head_dim:
            return False
        return layer.qk_head_dim <= _CK_MAX_HEAD_DIM

    def _dense_prefill_workspace(
        self, num_tokens: int, head_num: int, head_dim: int, dtype: torch.dtype
    ):
        """Scratch for one dequantized run, shared by all layers and grown on demand."""
        buf = self._dense_prefill_kv
        if (
            buf is None
            or buf.shape[1] < num_tokens
            or buf.shape[2] != head_num
            or buf.shape[3] != head_dim
            or buf.dtype != dtype
        ):
            buf = torch.empty(
                2, num_tokens, head_num, head_dim, dtype=dtype, device=self.device
            )
            self._dense_prefill_kv = buf
        return buf[0, :num_tokens], buf[1, :num_tokens]

    def _dense_run_layout(
        self, forward_batch: ForwardBatch, bs: int, unified_kv_indptr: torch.Tensor
    ):
        """Total and max KV length of the unified run plus int32 offsets, once per forward."""
        if self._dense_run is None:
            kv_lens = [
                p + e
                for p, e in zip(
                    forward_batch.extend_prefix_lens_cpu[:bs],
                    forward_batch.extend_seq_lens_cpu[:bs],
                )
            ]
            self._dense_run = (
                sum(kv_lens),
                max(kv_lens),
                self.forward_metadata.qo_indptr[: bs + 1].to(torch.int32),
                unified_kv_indptr[: bs + 1].to(torch.int32),
            )
        return self._dense_run

    def _forward_extend_dense(
        self,
        q: torch.Tensor,
        o: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        bs: int,
        kv_buffers,
        unified_kv_indptr: torch.Tensor,
        unified_kv_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Dequantize the unified KV run, then run flash attention over it."""
        k_codes, v_codes, k_scales, v_scales = kv_buffers
        head_num, head_dim = layer.tp_q_head_num, layer.qk_head_dim
        total_kv, max_kv_len, qo_indptr, kv_indptr = self._dense_run_layout(
            forward_batch, bs, unified_kv_indptr
        )
        dtype = torch.float8_e4m3fn if self._fp8_dense_prefill else q.dtype

        k_deq, v_deq = self._dense_prefill_workspace(
            total_kv, k_codes.shape[1], head_dim, dtype
        )
        self.ultraquant_gather_dequant(
            k_codes,
            k_scales,
            v_codes,
            v_scales,
            unified_kv_indices[:total_kv],
            k_deq,
            v_deq,
        )

        # Keys stay rotated, so queries are rotated to match. causal=True with
        # unequal q/kv lengths aligns lower-right, as continuation chunks need.
        q = q.view(-1, head_num, head_dim)
        max_extend_len = self.forward_metadata.max_extend_len
        if self._fp8_dense_prefill:
            q_fp8 = torch.empty(q.shape, dtype=dtype, device=q.device)
            out = flash_attn_varlen_fp8_pertensor_func(
                self.ultraquant_rotate(q, q_fp8),
                k_deq,
                v_deq,
                self._unit_scale,
                self._unit_scale,
                self._unit_scale,
                qo_indptr,
                kv_indptr,
                max_extend_len,
                max_kv_len,
                softmax_scale=layer.scaling,
                causal=True,
            )
            return out.to(o.dtype).view(o.shape)

        flash_attn_varlen_func(
            self.ultraquant_rotate(q),
            k_deq,
            v_deq,
            qo_indptr,
            kv_indptr,
            max_extend_len,
            max_kv_len,
            softmax_scale=layer.scaling,
            causal=True,
            out=o.view(-1, head_num, layer.v_head_dim),
        )
        return o

    @staticmethod
    def _reject_unsupported(
        layer: RadixAttention, score_mod, aux_tensors, name: str
    ) -> None:
        if score_mod is not None or aux_tensors is not None:
            raise NotImplementedError(
                f"The ultraquant backend does not support score_mod in {name}."
            )
        if layer.xai_temperature_len > 0:
            raise NotImplementedError(
                f"The ultraquant backend does not support xai temperature in {name}."
            )

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        sinks: Optional[torch.Tensor] = None,
        score_mod=None,
        aux_tensors=None,
    ):
        self._reject_unsupported(layer, score_mod, aux_tensors, "decode")
        if layer.layer_id in self.full_precision_layers:
            return super().forward_decode(
                q, k, v, layer, forward_batch, save_kv_cache, sinks
            )

        q = q.reshape(-1, layer.tp_q_head_num * layer.qk_head_dim)
        o = torch.empty_like(q)

        if save_kv_cache:
            self._set_kv_buffer(
                forward_batch,
                layer,
                KVWriteLoc(
                    forward_batch.out_cache_loc,
                    self.forward_metadata.swa_out_cache_loc,
                    full_loc=self.forward_metadata.out_cache_loc_full_physical,
                ),
                k,
                v,
                layer.k_scale,
                layer.v_scale,
            )

        if layer.sliding_window_size is not None and layer.sliding_window_size > -1:
            kv_indptr = self.forward_metadata.window_kv_indptr
            kv_indices = self.forward_metadata.window_kv_indices
        else:
            kv_indptr = self.forward_metadata.kv_indptr
            kv_indices = self.forward_metadata.kv_indices

        k_codes, v_codes, k_scales, v_scales = self._kv_buffers(layer.layer_id)
        o_heads = o.view(-1, layer.tp_q_head_num, layer.v_head_dim)
        logit_cap = logit_capping_mod(layer.logit_capping_method, layer.logit_cap)

        if self._use_flydsl_decode(q, layer, sinks, logit_cap):
            attn_logits = self.forward_metadata.attn_logits
            attn_lse = self.forward_metadata.attn_lse
            q_heads = q.view(-1, layer.tp_q_head_num, layer.qk_head_dim)
            num_splits, block_kv, work_budget = self.ultraquant_decode_launch_config(
                q_heads.shape[0],
                self.num_kv_head,
                self.min_kv_splits,
                self.max_kv_splits,
                self.device_core_count,
                attn_logits.shape[0] * attn_logits.shape[2],
            )
            self.flydsl_ultraquant_decode(
                q_heads,
                k_codes,
                k_scales,
                v_codes,
                v_scales,
                attn_logits,
                attn_lse,
                kv_indptr,
                kv_indices,
                layer.scaling,
                num_splits=num_splits,
                block_kv=block_kv,
                work_budget=work_budget,
            )
            self.ultraquant_decode_reduce(
                attn_logits,
                attn_lse,
                kv_indptr,
                o_heads,
                num_splits,
                block_kv,
                work_budget,
            )
            return o

        self.decode_attention_fwd_ultraquant(
            q.view(-1, layer.tp_q_head_num, layer.qk_head_dim),
            k_codes,
            k_scales,
            v_codes,
            v_scales,
            o_heads,
            kv_indptr,
            kv_indices,
            self.forward_metadata.attn_logits,
            self.forward_metadata.attn_lse,
            self.forward_metadata.num_kv_splits,
            self.max_kv_splits,
            layer.scaling,
            logit_cap=logit_cap,
            sinks=sinks,
        )

        return o

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        sinks: Optional[torch.Tensor] = None,
        score_mod=None,
        aux_tensors=None,
    ):
        self._reject_unsupported(layer, score_mod, aux_tensors, "extend")
        if layer.layer_id in self.full_precision_layers:
            return super().forward_extend(
                q, k, v, layer, forward_batch, save_kv_cache, sinks
            )

        q = q.reshape(-1, layer.tp_q_head_num * layer.qk_head_dim)
        o = torch.empty_like(q)

        # The unified kernel reads the current chunk back out of the cache, so
        # the write has to happen first.
        if save_kv_cache:
            self._set_kv_buffer(
                forward_batch,
                layer,
                KVWriteLoc(
                    forward_batch.out_cache_loc,
                    self.forward_metadata.swa_out_cache_loc,
                    full_loc=self.forward_metadata.out_cache_loc_full_physical,
                ),
                k,
                v,
                layer.k_scale,
                layer.v_scale,
            )
        else:
            raise NotImplementedError(
                "The ultraquant backend requires save_kv_cache during extend: the "
                "unified kernel reads the current chunk from the KV cache."
            )

        bs = forward_batch.batch_size
        (
            prefix_kv_indptr,
            prefix_kv_indices,
            sliding_window_size,
        ) = self._extend_window_metadata(layer)

        # Every full-attention layer of one forward reads the same unified run.
        if sliding_window_size > 0 or self._unified_kv is None:
            extend_kv_indices = (
                self.forward_metadata.out_cache_loc_full_physical
                if self.forward_metadata.out_cache_loc_full_physical is not None
                else forward_batch.out_cache_loc
            )
            unified_kv = self.build_unified_kv_indices(
                prefix_kv_indptr,
                prefix_kv_indices,
                forward_batch.extend_start_loc,
                forward_batch.extend_seq_lens,
                extend_kv_indices,
                bs,
            )
            if sliding_window_size <= 0:
                self._unified_kv = unified_kv
        else:
            unified_kv = self._unified_kv
        unified_kv_indptr, unified_kv_indices, prefix_lens = unified_kv

        kv_buffers = self._kv_buffers(layer.layer_id)
        k_codes, v_codes, k_scales, v_scales = kv_buffers
        logit_cap = logit_capping_mod(layer.logit_capping_method, layer.logit_cap)

        if self._use_dense_prefill(layer, sinks, logit_cap, sliding_window_size):
            return self._forward_extend_dense(
                q,
                o,
                layer,
                forward_batch,
                bs,
                kv_buffers,
                unified_kv_indptr,
                unified_kv_indices,
            )

        self.extend_attention_fwd_ultraquant(
            q.view(-1, layer.tp_q_head_num, layer.qk_head_dim),
            o.view(-1, layer.tp_q_head_num, layer.v_head_dim),
            k_codes,
            k_scales,
            v_codes,
            v_scales,
            self.forward_metadata.qo_indptr,
            unified_kv_indptr,
            unified_kv_indices,
            prefix_lens.to(torch.int32),
            self.forward_metadata.max_extend_len,
            layer.scaling,
            is_causal=not layer.is_cross_attention,
            custom_mask=self.forward_metadata.custom_mask,
            mask_indptr=self.forward_metadata.mask_indptr,
            sinks=sinks,
            sliding_window_size=sliding_window_size,
            logit_cap=logit_cap,
        )
        return o

    def _extend_window_metadata(self, layer):
        """Resolve the prefix index arrays and sliding-window geometry."""
        if layer.sliding_window_size is not None and layer.sliding_window_size > -1:
            return (
                self.forward_metadata.window_kv_indptr,
                self.forward_metadata.window_kv_indices,
                layer.sliding_window_size,
            )
        return (
            self.forward_metadata.kv_indptr,
            self.forward_metadata.kv_indices,
            -1,
        )
