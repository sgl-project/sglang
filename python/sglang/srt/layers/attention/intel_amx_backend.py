from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

import torch

from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.mem_cache.memory_pool import KVWriteLoc
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.speculative.spec_info import SpecInput


class IntelAMXAttnBackend(AttentionBackend):
    def __init__(
        self,
        model_runner: ModelRunner,
        skip_prefill: bool = False,
    ):
        import sgl_kernel  # noqa: F401

        super().__init__()
        self.forward_metadata = None
        self.skip_prefill = skip_prefill
        self.device = model_runner.device
        # Pool refs — captured at construction so they survive deletion of the
        # corresponding ForwardBatch fields.
        self.req_to_token_pool = model_runner.req_to_token_pool
        self.token_to_kv_pool = model_runner.token_to_kv_pool
        self.max_context_len = model_runner.model_config.context_len

        # full->SWA translated out_cache_loc, computed once per forward (the only
        # set_kv_buffer is in eager forward_extend; decode writes KV in-kernel).
        self.use_sliding_window_kv_pool = (
            isinstance(self.token_to_kv_pool, SWAKVPool)
            and self.token_to_kv_pool.swa_layer_nums > 0
        )
        self.swa_out_cache_loc = None

        self.num_head = (
            model_runner.model_config.num_attention_heads // model_runner.tp_size
        )

        # [NB]: `layer_id` set to 0 for qwen3-next models, as not all attn layers require kv pool
        # using "full_attention_layer_id_mapping" to map which layer needs kv pool
        layer_id = 0
        if hasattr(model_runner.token_to_kv_pool, "full_attention_layer_id_mapping"):
            layer_id = [*model_runner.token_to_kv_pool.full_attention_layer_id_mapping][
                0
            ]
        self.v_head_dim = model_runner.token_to_kv_pool.get_value_buffer(
            layer_id
        ).shape[-1]
        self.decode_attention_fwd = torch.ops.sgl_kernel.decode_attention_cpu
        self.extend_attention_fwd = torch.ops.sgl_kernel.extend_attention_cpu
        
        # speculative decoding params
        self.num_draft_tokens = model_runner.server_args.speculative_num_draft_tokens
        self.speculative_num_steps = model_runner.server_args.speculative_num_steps

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        """Init the metadata for a forward pass."""

        bs = forward_batch.batch_size
        spec_info = forward_batch.spec_info
        attn_logits = torch.zeros(
            (
                bs,
                self.num_head,
                8,  # self.num_kv_splits,
                self.v_head_dim + 1,
            ),
            dtype=torch.float32,
            device=self.device,
        )
        if forward_batch.forward_mode.is_decode_or_idle():
            max_extend_len = None
        elif forward_batch.forward_mode.is_target_verify():
            max_extend_len = self.num_draft_tokens
        elif forward_batch.forward_mode.is_draft_extend():
            max_extend_len = torch.max(spec_info.accept_length).item()
        else:
            max_extend_len = torch.max(forward_batch.extend_seq_lens).item()
        self.forward_metadata = (attn_logits, max_extend_len)

        if self.use_sliding_window_kv_pool and forward_batch.out_cache_loc is not None:
            self.swa_out_cache_loc = (
                self.token_to_kv_pool.translate_loc_from_full_to_swa(
                    forward_batch.out_cache_loc
                )
            )
        else:
            self.swa_out_cache_loc = None

    def get_cpu_graph_seq_len_fill_value(self):
        return 1

    def init_forward_metadata_capture_cpu_graph(
        self,
        bs: int,
        num_tokens: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        encoder_lens,
        forward_mode,
        spec_info,
    ):
        attn_logits = torch.zeros(
            (
                bs,
                self.num_head,
                8,  # self.num_kv_splits,
                self.v_head_dim + 1,
            ),
            dtype=torch.float32,
            device=self.device,
        )
        if forward_mode.is_target_verify():
            max_extend_len = self.num_draft_tokens
        elif forward_mode.is_draft_extend(include_v2=True):
            max_extend_len = self.speculative_num_steps + 1
        else:
            max_extend_len = None
        self.forward_metadata = (attn_logits, max_extend_len)

    def init_cpu_graph_state(self, max_bs: int, max_num_tokens: int):
        pass

    def forward_extend(
        self,
        q,
        k,
        v,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
        sinks=None,
    ):
        if layer.qk_head_dim != layer.v_head_dim:
            o = q.new_empty((q.shape[0], layer.tp_q_head_num * layer.v_head_dim))
        else:
            o = torch.empty_like(q)
        cache_loc = (
            forward_batch.out_cache_loc
            if not layer.is_cross_attention
            else forward_batch.encoder_out_cache_loc
        )
        if save_kv_cache and k is not None and v is not None:
            # Cross-attention never writes to the SWA pool, so only thread the
            # full->SWA location for non-cross-attention layers.
            swa_loc = None if layer.is_cross_attention else self.swa_out_cache_loc
            self.token_to_kv_pool.set_kv_buffer(
                layer, KVWriteLoc(cache_loc, swa_loc), k, v
            )

        # Handle cases where extend_seq_lens or extend_start_loc might not be set
        # In speculative decoding, we can infer these from spec_info or compute them
        bs = forward_batch.batch_size
        if forward_batch.extend_seq_lens is None:
            # TARGET_VERIFY mode: infer extend_seq_lens from spec_info
            if forward_batch.spec_info is not None and hasattr(
                forward_batch.spec_info, "draft_token_num"
            ):
                draft_token_num = forward_batch.spec_info.draft_token_num
                extend_seq_lens = torch.full(
                    (bs,), draft_token_num, dtype=torch.int32, device=self.device
                )
            else:
                raise RuntimeError(
                    "extend_seq_lens is None but cannot infer from spec_info. "
                    "This should not happen in TARGET_VERIFY mode."
                )
        else:
            extend_seq_lens = forward_batch.extend_seq_lens

        _, max_extend_len = self.forward_metadata
        self.extend_attention_fwd(
            q.view(-1, layer.tp_q_head_num, layer.qk_head_dim),
            k,
            v,
            o.view(-1, layer.tp_q_head_num, layer.v_head_dim),
            self.token_to_kv_pool.get_key_buffer(layer.layer_id),
            self.token_to_kv_pool.get_value_buffer(layer.layer_id),
            self.req_to_token_pool.req_to_token,
            forward_batch.req_pool_indices,
            forward_batch.seq_lens,
            # forward_batch.extend_seq_lens,
            extend_seq_lens,
            forward_batch.extend_start_loc,
            max_extend_len,
            layer.scaling,
            layer.logit_cap,
            layer.is_cross_attention,
            layer.sliding_window_size + 1,
            forward_batch.encoder_lens,
            sinks,
        )
        return o

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
        sinks=None,
    ):
        attn_logits, _ = self.forward_metadata

        q = q.reshape(-1, layer.tp_q_head_num * layer.qk_head_dim)

        if layer.qk_head_dim != layer.v_head_dim:
            o = q.new_empty((q.shape[0], layer.tp_q_head_num * layer.v_head_dim))
        else:
            o = torch.empty_like(q)
        cache_loc = (
            forward_batch.out_cache_loc
            if not layer.is_cross_attention
            else forward_batch.encoder_out_cache_loc
        )
        self.decode_attention_fwd(
            q.view(-1, layer.tp_q_head_num, layer.qk_head_dim),
            self.token_to_kv_pool.get_key_buffer(layer.layer_id),
            self.token_to_kv_pool.get_value_buffer(layer.layer_id),
            o.view(-1, layer.tp_q_head_num, layer.v_head_dim),
            k,
            v,
            cache_loc,
            attn_logits,
            self.req_to_token_pool.req_to_token,
            forward_batch.req_pool_indices,
            forward_batch.seq_lens,
            layer.scaling,
            layer.logit_cap,
            layer.is_cross_attention,
            layer.sliding_window_size + 1,
            forward_batch.encoder_lens,
            sinks,
        )
        return o

    def support_triton(self):
        return False


class IntelAMXMultiStepDraftBackend:
    """
    Wrap multiple intel amx attention backends as one for multiple consecutive
    draft decoding steps.
    """

    def __init__(
        self,
        model_runner: ModelRunner,
        topk: int,
        speculative_num_steps: int,
    ):
        self.topk = topk
        self.speculative_num_steps = speculative_num_steps
        max_bs = model_runner.req_to_token_pool.size * self.topk
        self.attn_backends: List[IntelAMXAttnBackend] = []
        for i in range(self.speculative_num_steps - 1):
            self.attn_backends.append(
                IntelAMXAttnBackend(
                    model_runner,
                    skip_prefill=True,
                )
            )
        self.max_context_len = self.attn_backends[0].max_context_len
        self.device = model_runner.device
        # Cached variables for generate_draft_decode_kv_indices
        self.pool_len = model_runner.req_to_token_pool.req_to_token.shape[1]
        self.page_size = model_runner.server_args.page_size

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        for i in range(self.speculative_num_steps - 1):
            self.attn_backends[i].init_forward_metadata(forward_batch)

    def init_cpu_graph_state(self, max_bs: int, max_num_tokens: int):
        pass

    def init_forward_metadata_capture_cpu_graph(self, forward_batch: ForwardBatch):
        for i in range(self.speculative_num_steps - 1):
            self.attn_backends[i].init_forward_metadata_capture_cpu_graph(
                forward_batch.batch_size,
                forward_batch.batch_size * self.topk,
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
                encoder_lens=None,
                forward_mode=ForwardMode.DECODE,
                spec_info=forward_batch.spec_info,
            )


