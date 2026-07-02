from __future__ import annotations

from typing import TYPE_CHECKING, List

import torch

from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.mem_cache.memory_pool import KVWriteLoc
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner


class IntelAMXAttnBackend(AttentionBackend):
    def __init__(self, model_runner: ModelRunner):
        import sgl_kernel  # noqa: F401

        super().__init__()
        self.forward_metadata = None
        self.extend_metadata = None
        self.draft_decode_metadata = None
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

        # Number of KV splits used by decode_attention_cpu; attn_logits is
        # sized [bs, num_head, num_kv_splits, v_head_dim + 1] to match.
        self.num_kv_splits = 8

        # speculative decoding params
        self.num_draft_tokens = model_runner.server_args.speculative_num_draft_tokens

    def _build_extend_metadata(self, forward_batch: ForwardBatch):
        """Resolve (seq_lens, extend_seq_lens, extend_start_loc, tree_mask) for
        forward_extend, once per forward pass.

        In TARGET_VERIFY mode the batch carries no extend_* fields, so they are
        derived from spec_info (mirrors the CUDA unified path in
        triton_backend.py); each request extends by exactly draft_token_num
        tokens. Outside spec decoding the fields are passed through, computing
        extend_start_loc only if the batch did not provide it.
        """
        bs = forward_batch.batch_size
        seq_lens = forward_batch.seq_lens
        tree_mask = None

        if forward_batch.extend_seq_lens is None:
            spec_info = forward_batch.spec_info
            if spec_info is None:
                raise RuntimeError(
                    "extend_seq_lens is None but cannot infer from spec_info. "
                    "This should not happen in TARGET_VERIFY mode."
                )
            draft_token_num = spec_info.draft_token_num
            extend_seq_lens = torch.full(
                (bs,), draft_token_num, dtype=torch.int32, device=self.device
            )
            # Uniform extend lengths: start locations form a plain range.
            extend_start_loc = torch.arange(
                0,
                bs * draft_token_num,
                draft_token_num,
                dtype=torch.int32,
                device=self.device,
            )
            seq_lens = forward_batch.seq_lens + draft_token_num
            # Speculative verify with a token tree: each draft token may only
            # attend to its ancestors among the draft tokens (the committed
            # prefix stays fully visible).
            #
            # tree_topk == 1 means a simple chain, which equals the kernel's
            # built-in causal masking, so no explicit mask is needed. EAGLE
            # has tree_topk == topk (>1 for trees); NGRAM has tree_topk == -1
            # (irregular tree).
            if spec_info.tree_topk != 1:
                custom_mask = spec_info.custom_mask
                if custom_mask is not None and custom_mask.numel() > 0:
                    tree_mask = custom_mask
        else:
            extend_seq_lens = forward_batch.extend_seq_lens
            if forward_batch.extend_start_loc is None:
                extend_start_loc = torch.zeros(
                    bs, dtype=torch.int32, device=self.device
                )
                if bs > 1:
                    extend_start_loc[1:] = torch.cumsum(extend_seq_lens[:-1], dim=0)
            else:
                extend_start_loc = forward_batch.extend_start_loc

        return seq_lens, extend_seq_lens, extend_start_loc, tree_mask

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        """Init the metadata for a forward pass."""

        bs = forward_batch.batch_size
        attn_logits = torch.zeros(
            (
                bs,
                self.num_head,
                self.num_kv_splits,
                self.v_head_dim + 1,
            ),
            dtype=torch.float32,
            device=self.device,
        )
        if forward_batch.forward_mode.is_decode_or_idle():
            max_extend_len = None
            self.extend_metadata = None
        elif forward_batch.forward_mode.is_target_verify():
            max_extend_len = self.num_draft_tokens
            self.extend_metadata = self._build_extend_metadata(forward_batch)
        else:
            max_extend_len = torch.max(forward_batch.extend_seq_lens).item()
            self.extend_metadata = self._build_extend_metadata(forward_batch)
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
                self.num_kv_splits,
                self.v_head_dim + 1,
            ),
            dtype=torch.float32,
            device=self.device,
        )
        max_extend_len = None
        self.forward_metadata = (attn_logits, max_extend_len)
        self.extend_metadata = None

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

        # Precomputed once per forward pass in init_forward_metadata (spec
        # verify batches carry no extend_* fields; see _build_extend_metadata).
        seq_lens, extend_seq_lens, extend_start_loc, tree_mask = self.extend_metadata

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
            seq_lens,
            extend_seq_lens,
            extend_start_loc,
            max_extend_len,
            layer.scaling,
            layer.logit_cap,
            layer.is_cross_attention,
            layer.sliding_window_size + 1,
            forward_batch.encoder_lens,
            sinks,
            tree_mask,
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

        if self.draft_decode_metadata is not None:
            draft_meta = self.draft_decode_metadata
            attn_logits = draft_meta["attn_logits"]
            req_to_token = draft_meta["req_to_token"]
            req_pool_indices = draft_meta["req_pool_indices"]
            seq_lens = draft_meta["seq_lens"]
        else:
            req_to_token = self.req_to_token_pool.req_to_token
            req_pool_indices = forward_batch.req_pool_indices
            seq_lens = forward_batch.seq_lens

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
            req_to_token,
            req_pool_indices,
            seq_lens,
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
        self.attn_backends: List[IntelAMXAttnBackend] = []
        for i in range(self.speculative_num_steps - 1):
            self.attn_backends.append(IntelAMXAttnBackend(model_runner))
        self.device = model_runner.device
        self.pool_len = model_runner.req_to_token_pool.req_to_token.shape[1]

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        from sgl_kernel import build_draft_decode_metadata_cpu

        num_seqs = forward_batch.batch_size
        topk = self.topk
        bs = num_seqs * topk
        num_steps = self.speculative_num_steps
        req_to_token = self.attn_backends[0].req_to_token_pool.req_to_token
        seq_lens = forward_batch.seq_lens
        pool_len = self.pool_len
        num_head = self.attn_backends[0].num_head
        v_head_dim = self.attn_backends[0].v_head_dim
        device = self.device

        # Build expanded req_to_token via C++ kernel
        req_to_token_draft = build_draft_decode_metadata_cpu(
            req_to_token,
            forward_batch.req_pool_indices,
            seq_lens,
            topk,
            num_steps,
            pool_len,
        )

        req_pool_indices_expanded = torch.arange(bs, dtype=torch.int64, device=device)

        num_kv_splits = self.attn_backends[0].num_kv_splits
        for step in range(num_steps - 1):
            # Each candidate sees prefix + (step + 1) draft tokens.
            seq_lens_expanded = seq_lens.repeat_interleave(topk) + step + 1
            attn_logits = torch.zeros(
                (bs, num_head, num_kv_splits, v_head_dim + 1),
                dtype=torch.float32,
                device=device,
            )
            self.attn_backends[step].forward_metadata = (attn_logits, None)
            self.attn_backends[step].draft_decode_metadata = {
                "attn_logits": attn_logits,
                "req_to_token": req_to_token_draft,
                "seq_lens": seq_lens_expanded,
                "req_pool_indices": req_pool_indices_expanded,
            }
