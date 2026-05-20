"""
TurboQuant-aware FlashInfer attention backend with CUDA graph support.

Overrides forward_decode to perform selective decode:
decode only active tokens into a compact bf16 buffer via a fused Triton kernel,
then run FlashInfer paged attention on the compact buffer.

CUDA graph support: fixed-size grid + N_active read from device tensor.
"""

from __future__ import annotations

import logging
import math
from functools import partial
from typing import TYPE_CHECKING

import torch

from sglang.srt.layers.attention.flashinfer_backend import (
    DecodeMetadata,
    FlashInferAttnBackend,
)
from sglang.srt.layers.attention.utils import create_flashinfer_kv_indices_triton
from sglang.srt.layers.dp_attention import get_attention_tp_size
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.utils import is_flashinfer_available

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)

if is_flashinfer_available():
    from flashinfer import BatchDecodeWithPagedKVCacheWrapper, fast_decode_plan


class FlashInferTQBackend(FlashInferAttnBackend):
    """TurboQuant selective-decode attention backend."""

    def __init__(self, model_runner: ModelRunner, **kwargs):
        super().__init__(model_runner, **kwargs)

        self._num_kv_heads = model_runner.model_config.get_num_kv_heads(
            get_attention_tp_size()
        )
        self._num_qo_heads = (
            model_runner.model_config.num_attention_heads // get_attention_tp_size()
        )
        self._head_dim = model_runner.model_config.head_dim
        self._q_data_type = model_runner.dtype
        self._device = model_runner.device

        # Compact decode buffers — sized to actual token pool, not theoretical max
        pool_size = model_runner.max_total_num_tokens
        self._max_compact = pool_size

        self.compact_k = torch.zeros(
            self._max_compact,
            self._num_kv_heads,
            self._head_dim,
            dtype=torch.bfloat16,
            device=self._device,
        )
        self.compact_v = torch.zeros_like(self.compact_k)

        # Active pool indices buffer (fixed address for CUDA graph)
        self._active_pool_indices = torch.zeros(
            self._max_compact, dtype=torch.int32, device=self._device
        )

        # Sequential kv_indices for compact wrapper
        self._compact_kv_indices = torch.arange(
            self._max_compact, dtype=torch.int32, device=self._device
        )

        # N_active as device tensor (updated before graph replay, read by Triton)
        self._n_active_tensor = torch.zeros(1, dtype=torch.int32, device=self._device)
        self._n_active = 0  # Python mirror for non-graph path

        # Triton scratch buffer — sized to pool so CUDA graph replay
        # covers any N_active (idle programs just early-exit)
        self._max_triton_active = self._max_compact
        scratch_elems = self._max_triton_active * self._num_kv_heads * self._head_dim
        self._fwht_scratch = torch.empty(
            scratch_elems, dtype=torch.float32, device=self._device
        )
        self._use_triton = True

        # Non-graph compact wrapper
        self._compact_decode_wrapper = BatchDecodeWithPagedKVCacheWrapper(
            self.workspace_buffer,
            "NHD",
            backend=self.decode_backend,
            use_tensor_cores=self.decode_use_tensor_cores,
        )

        # CUDA graph compact wrappers (populated during capture)
        self._compact_cuda_graph_metadata = {}
        # Grid size for CUDA graph Triton kernels (per bs)
        self._cuda_graph_grid_n = {}

        compact_mb = (
            self._max_compact * self._num_kv_heads * self._head_dim * 2 / 1024**2
        )
        scratch_mb = scratch_elems * 4 / 1024**2
        logger.info(
            "TQ Backend: max_compact=%d, compact_k=%.1f MB, "
            "triton_scratch=%.1f MB (max_triton=%d)",
            self._max_compact,
            compact_mb,
            scratch_mb,
            self._max_triton_active,
        )

    # ------------------------------------------------------------------
    # Non-graph forward metadata
    # ------------------------------------------------------------------

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        super().init_forward_metadata(forward_batch)
        if forward_batch.forward_mode.is_decode_or_idle():
            self._setup_compact_decode(forward_batch)

    def _setup_compact_decode(self, forward_batch: ForwardBatch):
        bs = forward_batch.batch_size
        seq_lens = forward_batch.seq_lens
        n_active = forward_batch.seq_lens_sum

        if n_active > self._max_compact:
            self._n_active = 0
            self._n_active_tensor.fill_(0)
            return

        self._n_active = n_active
        self._n_active_tensor.fill_(n_active)

        kv_indptr = self.kv_indptr[0]
        req_to_token = forward_batch.req_to_token_pool.req_to_token
        create_flashinfer_kv_indices_triton[(bs,)](
            req_to_token,
            forward_batch.req_pool_indices,
            seq_lens,
            kv_indptr[: bs + 1],
            None,
            self._active_pool_indices[:n_active],
            req_to_token.shape[1],
        )

        self._compact_decode_wrapper.begin_forward(
            kv_indptr[: bs + 1],
            self._compact_kv_indices[:n_active],
            self.kv_last_page_len[:bs],
            self._num_qo_heads,
            self._num_kv_heads,
            self._head_dim,
            1,
            data_type=torch.bfloat16,
            q_data_type=self._q_data_type,
            non_blocking=True,
        )

    # ------------------------------------------------------------------
    # CUDA graph capture / replay
    # ------------------------------------------------------------------

    def init_forward_metadata_capture_cuda_graph(
        self,
        bs,
        num_tokens,
        req_pool_indices,
        seq_lens,
        encoder_lens,
        forward_mode,
        spec_info,
    ):
        # Let parent capture its decode wrappers (we won't use them for decode)
        super().init_forward_metadata_capture_cuda_graph(
            bs,
            num_tokens,
            req_pool_indices,
            seq_lens,
            encoder_lens,
            forward_mode,
            spec_info,
        )

        if not forward_mode.is_decode_or_idle():
            return

        # Compute N_active for capture and set the grid size
        seq_lens_sum = seq_lens.sum().item()
        n_active = seq_lens_sum
        self._n_active = n_active
        self._n_active_tensor.fill_(n_active)

        # Grid for Triton kernel: use max_triton_active so the graph
        # covers any future N_active up to this size
        grid_n = min(self._max_triton_active, self._max_compact)
        self._cuda_graph_grid_n[bs] = grid_n

        # Gather active pool indices
        kv_indptr = self.kv_indptr[0]
        kv_indptr[1 : bs + 1] = torch.cumsum(seq_lens, dim=0)
        req_to_token = self.indices_updater_decode.req_to_token
        create_flashinfer_kv_indices_triton[(bs,)](
            req_to_token,
            req_pool_indices,
            seq_lens,
            kv_indptr[: bs + 1],
            None,
            self._active_pool_indices[:n_active],
            req_to_token.shape[1],
        )

        # Create CUDA-graph-enabled compact wrapper
        compact_wrapper = BatchDecodeWithPagedKVCacheWrapper(
            self.workspace_buffer,
            "NHD",
            backend=self.decode_backend,
            use_cuda_graph=True,
            use_tensor_cores=self.decode_use_tensor_cores,
            paged_kv_indptr_buffer=kv_indptr[: num_tokens + 1],
            paged_kv_indices_buffer=self._compact_kv_indices[:grid_n],
            paged_kv_last_page_len_buffer=self.kv_last_page_len[:num_tokens],
        )

        compact_wrapper.begin_forward(
            kv_indptr[: bs + 1],
            self._compact_kv_indices[:n_active],
            self.kv_last_page_len[:bs],
            self._num_qo_heads,
            self._num_kv_heads,
            self._head_dim,
            1,
            data_type=torch.bfloat16,
            q_data_type=self._q_data_type,
            non_blocking=True,
        )
        # Replace begin_forward with fast path for replay
        compact_wrapper.begin_forward = partial(fast_decode_plan, compact_wrapper)

        self._compact_cuda_graph_metadata[bs] = compact_wrapper
        self.forward_metadata = DecodeMetadata(
            self.decode_cuda_graph_metadata.get(bs, self.decode_wrappers)
        )

    def init_forward_metadata_replay_cuda_graph(
        self,
        bs,
        req_pool_indices,
        seq_lens,
        seq_lens_sum,
        encoder_lens,
        forward_mode,
        spec_info,
        seq_lens_cpu,
    ):
        # Parent updates its wrappers
        super().init_forward_metadata_replay_cuda_graph(
            bs,
            req_pool_indices,
            seq_lens,
            seq_lens_sum,
            encoder_lens,
            forward_mode,
            spec_info,
            seq_lens_cpu,
        )

        if not forward_mode.is_decode_or_idle():
            return

        n_active = seq_lens_sum
        self._n_active = n_active
        self._n_active_tensor.fill_(n_active)

        # Update active pool indices (in-place, same buffer)
        kv_indptr = self.kv_indptr[0]
        req_to_token = self.indices_updater_decode.req_to_token
        create_flashinfer_kv_indices_triton[(bs,)](
            req_to_token,
            req_pool_indices[:bs],
            seq_lens[:bs],
            kv_indptr[: bs + 1],
            None,
            self._active_pool_indices[:n_active],
            req_to_token.shape[1],
        )

        # Update compact wrapper planning
        compact_wrapper = self._compact_cuda_graph_metadata.get(bs)
        if compact_wrapper is not None:
            compact_wrapper.begin_forward(
                kv_indptr[: bs + 1],
                self._compact_kv_indices[:n_active],
                self.kv_last_page_len[:bs],
                self._num_qo_heads,
                self._num_kv_heads,
                self._head_dim,
                1,
                data_type=torch.bfloat16,
                q_data_type=self._q_data_type,
                non_blocking=True,
            )

    # ------------------------------------------------------------------
    # Selective decode
    # ------------------------------------------------------------------

    def _get_compact_wrapper(self, forward_batch):
        """Get the right compact wrapper (CUDA graph or regular)."""
        bs = forward_batch.batch_size if forward_batch is not None else 0
        return self._compact_cuda_graph_metadata.get(bs, self._compact_decode_wrapper)

    def _selective_decode_layer(self, layer_idx, pool, is_key):
        # Boundary layer protection: gather bf16 directly (no TQ decode)
        if hasattr(pool, "_protected_set") and layer_idx in pool._protected_set:
            from sglang.srt.model_executor.cuda_graph_runner import (
                get_is_capture_mode,
            )

            bf16_buf = (
                pool.k_bf16_buffer[layer_idx]
                if is_key
                else pool.v_bf16_buffer[layer_idx]
            )
            compact = self.compact_k if is_key else self.compact_v
            if get_is_capture_mode():
                # Fixed-shape gather for CUDA graph capture/replay.
                # _active_pool_indices is updated in-place before replay;
                # stale tail entries (beyond n_active) produce harmless data
                # that FlashInfer never reads (bounded by kv_indptr).
                compact[:] = bf16_buf[
                    self._active_pool_indices[: self._max_compact].long()
                ]
            else:
                n = self._n_active
                compact[:n] = bf16_buf[self._active_pool_indices[:n].long()]
            return

        cfg = pool.tq_config
        if cfg.mixed_precision:
            self._selective_decode_layer_mixed(layer_idx, pool, is_key)
            return

        n = self._n_active
        state = pool.tq_state

        if is_key:
            mse_buf = pool.k_mse_buffer[layer_idx]
            norm_buf = pool.k_norm_buffer[layer_idx]
            bits = cfg.key_mse_bits
            centroids = state.key_centroids
            signs = state.rotation_signs[layer_idx, 0]
            compact = self.compact_k
        else:
            mse_buf = pool.v_mse_buffer[layer_idx]
            norm_buf = pool.v_norm_buffer[layer_idx]
            bits = cfg.value_bits
            centroids = state.value_centroids
            signs = state.rotation_signs[layer_idx, 1]
            compact = self.compact_v

        use_triton = (
            self._use_triton
            and n <= self._max_triton_active
            and bits >= 1
            and not (is_key and cfg.enable_qjl)
        )

        if use_triton:
            try:
                from sglang.srt.layers.quantization.triton_tq_decode import (
                    triton_tq_selective_decode,
                )
                from sglang.srt.model_executor.cuda_graph_runner import (
                    get_is_capture_mode,
                )

                # During CUDA graph capture: fixed grid so replay works with any N_active.
                # During normal forward: grid = actual N_active (no wasted programs).
                grid_n = self._max_triton_active if get_is_capture_mode() else n

                triton_tq_selective_decode(
                    mse_buffer=mse_buf,
                    norm_buffer=norm_buf,
                    pool_indices=self._active_pool_indices,
                    centroids=centroids,
                    signs=signs,
                    compact_out=compact,
                    scratch=self._fwht_scratch,
                    n_active_tensor=self._n_active_tensor,
                    grid_n=grid_n,
                    bits=bits,
                )
                return
            except Exception as e:
                from sglang.srt.model_executor.cuda_graph_runner import (
                    get_is_capture_mode,
                )

                if get_is_capture_mode():
                    # During CUDA graph capture, the PyTorch fallback uses
                    # dynamic slicing ([:n]) which is NOT graph-safe. Fail hard
                    # instead of silently producing a broken graph.
                    raise RuntimeError(
                        f"Triton TQ decode failed during CUDA graph capture: {e}. "
                        "Cannot fall back to PyTorch path (not graph-safe)."
                    ) from e
                if not hasattr(self, "_triton_warned"):
                    logger.warning(
                        "Triton TQ decode failed, falling back to PyTorch: %s", e
                    )
                    self._triton_warned = True
                self._use_triton = False

        self._selective_decode_layer_pytorch(layer_idx, pool, is_key)

    def _selective_decode_layer_pytorch(self, layer_idx, pool, is_key):
        from sglang.srt.layers.quantization.turboquant import _unpack, rht_inverse

        cfg = pool.tq_config
        state = pool.tq_state
        n = self._n_active
        indices = self._active_pool_indices[:n].long()

        if is_key:
            mse_buf, norm_buf = (
                pool.k_mse_buffer[layer_idx],
                pool.k_norm_buffer[layer_idx],
            )
            bits, centroids = cfg.key_mse_bits, state.key_centroids
            signs, compact = state.rotation_signs[layer_idx, 0], self.compact_k
        else:
            mse_buf, norm_buf = (
                pool.v_mse_buffer[layer_idx],
                pool.v_norm_buffer[layer_idx],
            )
            bits, centroids = cfg.value_bits, state.value_centroids
            signs, compact = state.rotation_signs[layer_idx, 1], self.compact_v

        packed, norms = mse_buf[indices], norm_buf[indices]
        mse_indices = _unpack(packed, bits, cfg.head_dim)
        y_hat = centroids[mse_indices.long()]
        # Norm correction: renormalize in rotated domain before inverse RHT
        y_hat = y_hat / y_hat.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        result = rht_inverse(y_hat, signs) * norms

        if is_key and cfg.enable_qjl and pool.k_qjl_buffer is not None:
            from sglang.srt.layers.quantization.turboquant import unpack_1bit

            qjl_packed = pool.k_qjl_buffer[layer_idx][indices]
            r_norms = pool.k_rnorm_buffer[layer_idx][indices]
            qjl_float = unpack_1bit(qjl_packed, cfg.head_dim).float() * 2 - 1
            qjl_recon = torch.matmul(qjl_float, state.qjl_matrix)
            result = (
                result
                + (math.sqrt(math.pi / 2) / cfg.head_dim * cfg.qjl_score_weight)
                * r_norms
                * qjl_recon
            )

        compact[:n] = result.to(torch.bfloat16)

    def _selective_decode_layer_mixed(self, layer_idx, pool, is_key):
        from sglang.srt.layers.quantization.turboquant import (
            decode_keys_mixed,
            decode_values_mixed,
        )

        n = self._n_active
        indices = self._active_pool_indices[:n].long()
        compact = self.compact_k if is_key else self.compact_v

        if is_key:
            compact[:n] = decode_keys_mixed(
                outlier_packed=pool.k_mse_outlier_buffer[layer_idx][indices],
                regular_packed=pool.k_mse_regular_buffer[layer_idx][indices],
                outlier_norms=pool.k_norm_outlier_buffer[layer_idx][indices],
                regular_norms=pool.k_norm_regular_buffer[layer_idx][indices],
                qjl_packed=(
                    pool.k_qjl_buffer[layer_idx][indices] if pool.k_qjl_buffer else None
                ),
                r_norms=(
                    pool.k_rnorm_buffer[layer_idx][indices]
                    if pool.k_rnorm_buffer
                    else None
                ),
                layer_idx=layer_idx,
                state=pool.tq_state,
                output_dtype=torch.bfloat16,
            )
        else:
            compact[:n] = decode_values_mixed(
                outlier_packed=pool.v_mse_outlier_buffer[layer_idx][indices],
                regular_packed=pool.v_mse_regular_buffer[layer_idx][indices],
                outlier_norms=pool.v_norm_outlier_buffer[layer_idx][indices],
                regular_norms=pool.v_norm_regular_buffer[layer_idx][indices],
                layer_idx=layer_idx,
                state=pool.tq_state,
                output_dtype=torch.bfloat16,
            )

    # ------------------------------------------------------------------
    # Decode forward
    # ------------------------------------------------------------------

    def forward_decode(self, q, k, v, layer, forward_batch, save_kv_cache=True):
        if self._n_active == 0:
            return super().forward_decode(q, k, v, layer, forward_batch, save_kv_cache)

        cache_loc = (
            forward_batch.out_cache_loc
            if not layer.is_cross_attention
            else forward_batch.encoder_out_cache_loc
        )

        if k is not None:
            assert v is not None
            if save_kv_cache:
                forward_batch.token_to_kv_pool.set_kv_buffer(
                    layer, cache_loc, k, v, layer.k_scale, layer.v_scale
                )

        pool = forward_batch.token_to_kv_pool
        layer_idx = layer.layer_id - pool.start_layer

        self._selective_decode_layer(layer_idx, pool, is_key=True)
        self._selective_decode_layer(layer_idx, pool, is_key=False)

        # Always pass FULL compact buffers (fixed address for CUDA graph)
        compact_wrapper = self._get_compact_wrapper(forward_batch)
        o = compact_wrapper.forward(
            q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim),
            (self.compact_k, self.compact_v),
            sm_scale=layer.scaling,
            logits_soft_cap=layer.logit_cap,
            k_scale=layer.k_scale_float,
            v_scale=layer.v_scale_float,
        )

        return o.view(-1, layer.tp_q_head_num * layer.head_dim)
