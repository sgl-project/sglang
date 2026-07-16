from typing import Optional

import torch

from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.attention.dsa.dsa_indexer import BaseIndexerMetadata
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.model_executor.model_runner import ModelRunner


class HybridAttnBackend(AttentionBackend):
    """Support different backends for prefill and decode."""

    def __init__(
        self,
        model_runner: ModelRunner,
        prefill_backend: AttentionBackend,
        decode_backend: AttentionBackend,
    ):
        self.model_runner = model_runner
        self.prefill_backend = prefill_backend
        self.decode_backend = decode_backend
        self.data_type = model_runner.kv_cache_dtype
        self.token_to_kv_pool = model_runner.token_to_kv_pool
        self.req_to_token_pool = model_runner.req_to_token_pool
        self.spec_attn_is_decode = (
            model_runner.server_args.speculative_attention_mode == "decode"
        )
        self.spec_attn_is_prefill = (
            model_runner.server_args.speculative_attention_mode == "prefill"
        )
        # decide_needs_cpu_seq_lens ORs this flag across attn backends; without
        # the delegation the getattr default (True) forces the per-step
        # seq_lens D2H + host sync in FutureMap.resolve_seq_lens_cpu even when
        # both sub-backends opted out (e.g. trtllm_mla prefill + cutedsl_mla
        # decode), serializing the CPU behind every verify step.
        self.needs_cpu_seq_lens = bool(
            getattr(prefill_backend, "needs_cpu_seq_lens", True)
            or getattr(decode_backend, "needs_cpu_seq_lens", True)
        )

    def _select_backend(self, forward_mode: ForwardMode) -> AttentionBackend:
        """
        Select the appropriate attention backend based on the forward mode.

        Args:
            forward_mode: The current forward mode indicating the operation type

        Returns:
            The selected attention backend (prefill or decode)

        Note:
            - decode_or_idle: Always uses decode backend
            - target_verify: Uses decode backend if speculative_attention_mode is "decode", otherwise prefill backend
            - prefill: Always uses prefill backend
        """
        if forward_mode.is_decode_or_idle():
            return self.decode_backend
        elif forward_mode.is_target_verify():
            return (
                self.decode_backend
                if self.spec_attn_is_decode
                else self.prefill_backend
            )
        else:
            return self.prefill_backend

    def init_forward_metadata_out_graph(
        self,
        forward_batch: ForwardBatch,
        in_capture: bool = False,
    ):
        backend = self._select_backend(forward_batch.forward_mode)
        backend.init_forward_metadata_out_graph(forward_batch, in_capture=in_capture)

    def init_forward_metadata_in_graph(self, forward_batch: ForwardBatch):
        backend = self._select_backend(forward_batch.forward_mode)
        backend.init_forward_metadata_in_graph(forward_batch)

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        backend = self._select_backend(forward_batch.forward_mode)
        backend.init_forward_metadata(forward_batch)

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        self.decode_backend.init_cuda_graph_state(max_bs, max_num_tokens)
        if (
            self.model_runner.server_args.speculative_algorithm is not None
            and self.spec_attn_is_prefill
        ):
            # When speculative decoding is enabled, we need to initialize the backend
            # that will be used for target_verify.
            self.prefill_backend.init_cuda_graph_state(max_bs, max_num_tokens)

    def get_cuda_graph_seq_len_fill_value(self):
        return self.decode_backend.get_cuda_graph_seq_len_fill_value()

    def forward(
        self,
        q: Optional[torch.Tensor] = None,  # For full attention
        k: Optional[torch.Tensor] = None,  # For full attention
        v: Optional[torch.Tensor] = None,  # For full attention
        layer: Optional[RadixAttention] = None,
        forward_batch: Optional[ForwardBatch] = None,
        save_kv_cache: bool = True,
        *,
        mixed_qkv: Optional[torch.Tensor] = None,  # For linear attention
        a: Optional[torch.Tensor] = None,  # For linear attention
        b: Optional[torch.Tensor] = None,  # For linear attention
        **kwargs,
    ):
        """Forward method that supports both regular attention (q, k, v) and linear attention (mixed_qkv, a, b)."""
        backend = self._select_backend(forward_batch.forward_mode)
        if mixed_qkv is not None:
            return backend.forward(
                layer=layer,
                forward_batch=forward_batch,
                save_kv_cache=save_kv_cache,
                mixed_qkv=mixed_qkv,
                a=a,
                b=b,
                **kwargs,
            )
        return backend.forward(q, k, v, layer, forward_batch, save_kv_cache, **kwargs)

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        **kwargs,
    ):
        return self.decode_backend.forward_decode(
            q, k, v, layer, forward_batch, save_kv_cache, **kwargs
        )

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        **kwargs,
    ):
        backend = self._select_backend(forward_batch.forward_mode)
        return backend.forward_extend(
            q, k, v, layer, forward_batch, save_kv_cache, **kwargs
        )

    def get_indexer_metadata(
        self, layer_id: int, forward_batch: ForwardBatch
    ) -> Optional[BaseIndexerMetadata]:
        backend = self._select_backend(forward_batch.forward_mode)
        return backend.get_indexer_metadata(layer_id, forward_batch)

    def update_mamba_state_after_mtp_verify(self, *args, **kwargs):
        if self.spec_attn_is_decode:
            backend = self.decode_backend
        else:
            backend = self.prefill_backend
        return backend.update_mamba_state_after_mtp_verify(*args, **kwargs)

    def forward(
        self,
        q: torch.Tensor = None,
        k: torch.Tensor = None,
        v: torch.Tensor = None,
        layer: RadixAttention = None,
        forward_batch: ForwardBatch = None,
        save_kv_cache: bool = True,
        **kwargs,
    ):
        """Delegate forward to the appropriate backend based on forward mode."""
        backend = self._select_backend(forward_batch.forward_mode)
        return backend.forward(
            q=q,
            k=k,
            v=v,
            layer=layer,
            forward_batch=forward_batch,
            save_kv_cache=save_kv_cache,
            **kwargs,
        )
