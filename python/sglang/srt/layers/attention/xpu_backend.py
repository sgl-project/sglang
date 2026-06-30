from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.configs.model_config import AttentionArch
from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.attention.flashattention_backend import (
    FlashAttentionMetadata,
    make_local_attention_virtual_batches,
    merge_state_v2_wrapper,
    prepare_swa_spec_page_table_triton,
)
from sglang.srt.managers.schedule_batch import get_global_server_args
from sglang.srt.mem_cache.memory_pool import KVWriteLoc
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner

from sgl_kernel import flash_mla_decode, flash_mla_get_workspace_size, merge_state_v2
from sgl_kernel.flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache


class XPUAttentionBackend(AttentionBackend):
    """XPU FlashAttention backend, currently based on FlashAttentionBackend, will be refactored later.

    TODO:
    - Prefill and Decode disaggregation, currently only chunked prefill is supported
    - Speculative Decoding support
    - XPU Graph support, see https://github.com/pytorch/pytorch/issues/162143
    - MLA Prefill support
    """

    def __init__(
        self,
        model_runner: ModelRunner,
        skip_prefill: bool = False,
        speculative_step_id=0,
        topk=0,
        speculative_num_steps=0,
    ):
        super().__init__()

        assert not (
            model_runner.sliding_window_size is not None
            and model_runner.model_config.is_encoder_decoder
        ), "Sliding window and cross attention are not supported together"

        self.forward_metadata: FlashAttentionMetadata = None
        # extra metadata for handling speculative decoding topk > 1, extended draft decode and verify
        self.forward_metadata_spec_decode_expand: FlashAttentionMetadata = None
        self.max_context_len = model_runner.model_config.context_len
        self.num_attention_heads = (
            model_runner.model_config.hf_text_config.num_attention_heads
        )
        self.tp_size = model_runner.tp_size
        assert self.num_attention_heads % self.tp_size == 0
        self.num_local_heads = self.num_attention_heads // self.tp_size
        self.device = model_runner.device
        self.decode_cuda_graph_metadata = {}
        self.target_verify_metadata = {}
        # Pool refs — captured at construction so they survive deletion of the
        # corresponding ForwardBatch fields.
        self.req_to_token_pool = model_runner.req_to_token_pool
        self.token_to_kv_pool = model_runner.token_to_kv_pool
        self.req_to_token = model_runner.req_to_token_pool.req_to_token
        self.kv_cache_dtype = model_runner.kv_cache_dtype
        self.kv_cache_dtype_str = model_runner.server_args.kv_cache_dtype
        self.page_size = model_runner.page_size
        self.use_mla = model_runner.model_config.attention_arch == AttentionArch.MLA
        self.skip_prefill = skip_prefill
        self.is_hybrid_swa = model_runner.is_hybrid_swa
        self.use_sliding_window_kv_pool = (
            isinstance(model_runner.token_to_kv_pool, SWAKVPool)
            and model_runner.token_to_kv_pool.swa_layer_nums > 0
        )
        if self.use_sliding_window_kv_pool:
            self.token_to_kv_pool = model_runner.token_to_kv_pool
        if self.is_hybrid_swa:
            self.full_to_swa_index_mapping = (
                model_runner.token_to_kv_pool.full_to_swa_index_mapping
            )
        self.topk = model_runner.server_args.speculative_eagle_topk or 0
        self.speculative_num_steps = speculative_num_steps
        self.speculative_num_draft_tokens = (
            model_runner.server_args.speculative_num_draft_tokens
        )
        self.speculative_step_id = speculative_step_id

        # Local attention settings
        self.attention_chunk_size = (
            model_runner.attention_chunk_size
            if hasattr(model_runner, "attention_chunk_size")
            else None
        )

        # For each layer, the sliding_window_size can be different. This is only used for preparing SWA metadata.
        # We use `layer.sliding_window_size` to decide whether to use SWA for each layer.
        self.sliding_window_size = model_runner.sliding_window_size
        self.has_swa = (
            self.sliding_window_size is not None and self.sliding_window_size > -1
        )
        self.is_encoder_decoder = model_runner.model_config.is_encoder_decoder

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        """Initialize forward metadata hence all layers in the forward pass can reuse it."""
        metadata = FlashAttentionMetadata()
        seqlens_in_batch = forward_batch.seq_lens
        batch_size = forward_batch.batch_size
        device = seqlens_in_batch.device

        if forward_batch.forward_mode.is_decode_or_idle():
            # Draft Decode
            if forward_batch.spec_info is not None:
                assert (
                    False
                ), "XPUAttentionBackend doesn't support speculative decoding yet, please use --attention-backend triton instead."
                if self.topk <= 1:
                    metadata.cache_seqlens_int32 = (
                        seqlens_in_batch + (self.speculative_step_id + 1)
                    ).to(torch.int32)
                    metadata.max_seq_len_k = forward_batch.seq_lens_cpu.max().item() + (
                        self.speculative_step_id + 1
                    )
                    metadata.cu_seqlens_q = torch.arange(
                        0, batch_size + 1, dtype=torch.int32, device=device
                    )
                    metadata.cu_seqlens_k = torch.nn.functional.pad(
                        torch.cumsum(
                            metadata.cache_seqlens_int32, dim=0, dtype=torch.int32
                        ),
                        (1, 0),
                    )
                    metadata.page_table = self.req_to_token_pool.req_to_token[
                        forward_batch.req_pool_indices, : metadata.max_seq_len_k
                    ]
                else:
                    metadata.cache_seqlens_int32 = (seqlens_in_batch).to(torch.int32)
                    metadata.max_seq_len_q = self.topk
                    metadata.max_seq_len_k = forward_batch.seq_lens_cpu.max().item()
                    metadata.cu_seqlens_q = torch.arange(
                        0,
                        batch_size * self.topk + 1,
                        step=self.topk,
                        dtype=torch.int32,
                        device=device,
                    )
                    metadata.cu_seqlens_k = torch.nn.functional.pad(
                        torch.cumsum(
                            metadata.cache_seqlens_int32, dim=0, dtype=torch.int32
                        ),
                        (1, 0),
                    )
                    metadata.page_table = self.req_to_token_pool.req_to_token[
                        forward_batch.req_pool_indices, : metadata.max_seq_len_k
                    ]

                    metadata_expand = FlashAttentionMetadata()
                    decode_length = self.speculative_step_id + 1
                    metadata_expand.cache_seqlens_int32 = torch.full(
                        (seqlens_in_batch.numel() * self.topk,),
                        decode_length,
                        device=device,
                        dtype=torch.int32,
                    )
                    metadata_expand.max_seq_len_q = 1
                    metadata_expand.cu_seqlens_q = torch.arange(
                        0,
                        metadata_expand.cache_seqlens_int32.numel() + 1,
                        dtype=torch.int32,
                        device=device,
                    )
                    metadata_expand.cu_seqlens_k = torch.arange(
                        0,
                        metadata_expand.cache_seqlens_int32.numel() * decode_length + 1,
                        step=decode_length,
                        dtype=torch.int32,
                        device=device,
                    )
                    # shape: [bs, num_steps, topk] -> [bs x topk, num_steps]
                    cache_loc = forward_batch.out_cache_loc.view(
                        -1, self.speculative_num_steps
                    )
                    metadata_expand.page_table = (
                        cache_loc[:, :decode_length].contiguous().to(torch.int32)
                    )
                    self.forward_metadata_spec_decode_expand = metadata_expand
            else:
                # Normal Decode
                metadata.cache_seqlens_int32 = seqlens_in_batch.to(torch.int32)
                metadata.max_seq_len_k = forward_batch.seq_lens_cpu.max().item()
                metadata.cu_seqlens_q = torch.arange(
                    0, batch_size + 1, dtype=torch.int32, device=device
                )
                metadata.cu_seqlens_k = torch.nn.functional.pad(
                    torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0)
                )
                metadata.page_table = self.req_to_token_pool.req_to_token[
                    forward_batch.req_pool_indices, : metadata.max_seq_len_k
                ]

            # TODO: we need to test this part for llama 4 eagle case
            self._init_local_attn_metadata(forward_batch, metadata, device)
        elif forward_batch.forward_mode.is_target_verify():
            if self.topk <= 1:
                metadata.cache_seqlens_int32 = (
                    forward_batch.seq_lens + self.speculative_num_draft_tokens
                ).to(torch.int32)
                metadata.max_seq_len_q = self.speculative_num_draft_tokens
                metadata.max_seq_len_k = (
                    forward_batch.seq_lens_cpu.max().item()
                    + self.speculative_num_draft_tokens
                )
                metadata.cu_seqlens_q = torch.arange(
                    0,
                    batch_size * self.speculative_num_draft_tokens + 1,
                    self.speculative_num_draft_tokens,
                    dtype=torch.int32,
                    device=device,
                )
                metadata.cu_seqlens_k = torch.nn.functional.pad(
                    torch.cumsum(
                        metadata.cache_seqlens_int32, dim=0, dtype=torch.int32
                    ),
                    (1, 0),
                )
                metadata.page_table = self.req_to_token_pool.req_to_token[
                    forward_batch.req_pool_indices, : metadata.max_seq_len_k
                ]

                self._init_local_attn_metadata(forward_batch, metadata, device)
            else:
                metadata.cache_seqlens_int32 = forward_batch.seq_lens.to(torch.int32)
                metadata.max_seq_len_q = self.speculative_num_draft_tokens
                metadata.max_seq_len_k = forward_batch.seq_lens_cpu.max().item()
                metadata.cu_seqlens_q = torch.arange(
                    0,
                    batch_size * self.speculative_num_draft_tokens + 1,
                    step=self.speculative_num_draft_tokens,
                    dtype=torch.int32,
                    device=device,
                )
                metadata.cu_seqlens_k = torch.nn.functional.pad(
                    torch.cumsum(
                        metadata.cache_seqlens_int32, dim=0, dtype=torch.int32
                    ),
                    (1, 0),
                )
                metadata.page_table = self.req_to_token_pool.req_to_token[
                    forward_batch.req_pool_indices, : metadata.max_seq_len_k
                ]

                metadata_expand = FlashAttentionMetadata()

                metadata_expand.max_seq_len_q = 1
                metadata_expand.cu_seqlens_q = torch.arange(
                    0,
                    forward_batch.seq_lens.numel() * self.speculative_num_draft_tokens
                    + 1,
                    dtype=torch.int32,
                    device=device,
                )

                # create expand page table
                offsets = torch.arange(
                    self.speculative_num_draft_tokens, device=device
                ).unsqueeze(
                    0
                )  # shape: (1, self.speculative_num_draft_tokens)
                cols = offsets.expand(
                    forward_batch.seq_lens.numel(), -1
                ) + forward_batch.seq_lens.unsqueeze(1)
                cum_len = torch.nn.functional.pad(
                    torch.cumsum(
                        (
                            forward_batch.seq_lens + self.speculative_num_draft_tokens
                        ).repeat_interleave(self.speculative_num_draft_tokens),
                        dim=0,
                    ),
                    (1, 0),
                )[:-1]
                mask_extraction_indices = (
                    cols.repeat_interleave(self.speculative_num_draft_tokens, dim=0)
                    + cum_len[:, None]
                ).view(1, -1)
                mask = forward_batch.spec_info.custom_mask[
                    mask_extraction_indices
                ].view(
                    -1, self.speculative_num_draft_tokens
                )  # (bsz * draft_num, draft_num)

                # shift table indices to avoid padding
                # non_masked_page_table [[8, 9, 10],   mask (display with int format) [[1, 0, 0],
                #                        [8, 9, 10],                                   [1, 1, 0],
                #                        [8, 9, 10]]                                   [1, 0, 1]]
                # if masked with padding [[8, 0, 0],   our mask without padding       [[8, 9, 10],
                #                        [8, 9, 0],                                    [8, 9, 10],
                #                        [8, 0, 10]]                                   [8, 10, 9]]
                # note here cache_seqlens_int32 is [1, 2, 2] so extra page indices will be ignored in each row
                col_indices = offsets.expand(
                    mask.shape[0], self.speculative_num_draft_tokens
                )
                # Build keys: if an entry is valid (mask==True), keep its original index;
                # if not, add self.speculative_num_draft_tokens so that it sorts after all valid entries.
                keys = torch.where(
                    mask, col_indices, col_indices + self.speculative_num_draft_tokens
                )
                _, sort_order = torch.sort(keys, dim=1)
                non_masked_page_table = (
                    self.req_to_token_pool.req_to_token[
                        forward_batch.req_pool_indices, :
                    ]
                    .gather(1, cols)
                    .repeat_interleave(self.speculative_num_draft_tokens, dim=0)
                )  # (bsz, draft_num)
                metadata_expand.page_table = non_masked_page_table.gather(1, sort_order)
                metadata_expand.cache_seqlens_int32 = mask.sum(dim=1).to(torch.int32)
                metadata_expand.cu_seqlens_k = torch.nn.functional.pad(
                    torch.cumsum(
                        metadata_expand.cache_seqlens_int32, dim=0, dtype=torch.int32
                    ),
                    (1, 0),
                )
                self.forward_metadata_spec_decode_expand = metadata_expand

                if self.has_swa:
                    self._init_sliding_window_attn_spec_metadata(
                        metadata, metadata_expand
                    )

        elif forward_batch.forward_mode.is_extend_or_draft_extend_or_mixed():
            metadata.cache_seqlens_int32 = seqlens_in_batch.to(torch.int32)
            metadata.max_seq_len_k = forward_batch.seq_lens_cpu.max().item()
            metadata.cu_seqlens_k = torch.nn.functional.pad(
                torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0)
            )
            metadata.page_table = self.req_to_token_pool.req_to_token[
                forward_batch.req_pool_indices, : metadata.max_seq_len_k
            ]

            if any(forward_batch.extend_prefix_lens_cpu):
                extend_seq_lens = forward_batch.extend_seq_lens
                metadata.max_seq_len_q = max(forward_batch.extend_seq_lens_cpu)
                metadata.cu_seqlens_q = torch.nn.functional.pad(
                    torch.cumsum(extend_seq_lens, dim=0, dtype=torch.int32), (1, 0)
                )
            else:
                metadata.max_seq_len_q = metadata.max_seq_len_k
                metadata.cu_seqlens_q = metadata.cu_seqlens_k

            # Setup local attention if enabled
            if forward_batch.forward_mode == ForwardMode.EXTEND:
                self._init_local_attn_metadata(forward_batch, metadata, device)

        # Encoder metadata for cross attention
        if forward_batch.encoder_lens is not None:
            assert (
                forward_batch.encoder_lens.numel() == 1
            ), "Only encoder size 1 is supported for now"

            metadata.encoder_lens_int32 = forward_batch.encoder_lens.to(torch.int32)
            metadata.encoder_cu_seqlens_k = torch.nn.functional.pad(
                torch.cumsum(metadata.encoder_lens_int32, dim=0, dtype=torch.int32),
                (1, 0),
            )
            metadata.encoder_max_seq_len_k = metadata.encoder_lens_int32.max().item()
            metadata.encoder_page_table = self.req_to_token_pool.req_to_token[
                forward_batch.req_pool_indices, : metadata.encoder_max_seq_len_k
            ]

            # Currently only support forward_batch.encoder_lens.numel() == 1
            metadata.page_table = self.req_to_token_pool.req_to_token[
                forward_batch.req_pool_indices,
                metadata.encoder_max_seq_len_k : (
                    metadata.encoder_max_seq_len_k + metadata.max_seq_len_k
                ),
            ]

        # Translate full-pool indices to SWA-pool indices for hybrid models
        if self.use_sliding_window_kv_pool:
            # flash_attn_with_kvcache requires int32 page tables; the SWA index
            # mapping is int64, so cast (matches flashattention_backend.py).
            metadata.swa_page_table = (
                self.token_to_kv_pool.translate_loc_from_full_to_swa(
                    metadata.page_table
                ).to(torch.int32)
            )
            if forward_batch.out_cache_loc is not None:
                metadata.swa_out_cache_loc = (
                    self.token_to_kv_pool.translate_loc_from_full_to_swa(
                        forward_batch.out_cache_loc
                    )
                )

        if self.use_mla:
            workspace_size = flash_mla_get_workspace_size(
                max_seq_len=self.max_context_len,
                num_batches=batch_size,
                num_heads=self.num_local_heads,
                page_size=self.page_size,
                num_kv_splits=-1,
            )
            if (
                not hasattr(self, "workspace")
                or self.workspace.numel() < workspace_size
            ):
                self.workspace = torch.empty(
                    workspace_size, device=self.device, dtype=torch.uint8
                )

        # Convert the page table to a strided format which is needed by FA3 API
        if self.page_size > 1:
            self.strided_indices = torch.arange(
                0, metadata.page_table.shape[1], self.page_size, device=self.device
            )

            if self.use_sliding_window_kv_pool and metadata.swa_page_table is not None:
                metadata.swa_page_table = (
                    metadata.swa_page_table[:, self.strided_indices] // self.page_size
                )

            metadata.page_table = (
                metadata.page_table[:, self.strided_indices] // self.page_size
            )

        self.forward_metadata = metadata

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
        # For multi-head latent attention
        q_rope: Optional[torch.Tensor] = None,
        k_rope: Optional[torch.Tensor] = None,
        sinks: Optional[torch.Tensor] = None,
    ):
        if k is None and v is None:
            # Cross-layer KV sharing (Gemma 4): the layer reuses another
            # layer's KV cache. The paged kernel reads K/V directly via
            # page_table, and pool.get_kv_buffer(layer.layer_id) routes
            # to the correct sub-pool because RadixAttention is initialized
            # with layer_id=kv_shared_layer_index for shared layers. No
            # materialization needed; just skip the write path.
            pass
        elif k is None or v is None:
            raise ValueError("Both k and v should be None or not None")
        else:
            if save_kv_cache:
                cache_loc = (
                    forward_batch.out_cache_loc
                    if not layer.is_cross_attention
                    else forward_batch.encoder_out_cache_loc
                )
                if not self.use_mla:
                    self.token_to_kv_pool.set_kv_buffer(
                        layer,
                        KVWriteLoc(
                            cache_loc,
                            self.forward_metadata.swa_out_cache_loc,
                        ),
                        k,
                        v,
                        layer.k_scale,
                        layer.v_scale,
                    )
                else:
                    self.token_to_kv_pool.set_mla_kv_buffer(
                        layer,
                        cache_loc,
                        k,
                        k_rope,
                    )

        # Use precomputed metadata across all layers
        metadata = self.forward_metadata

        # Calculate window size (can be moved to metadata if layer properties don't change)
        # we don't do layer.sliding_window_size - 1 since in model.get_attention_sliding_window_size() we already - 1
        # here is two side inclusive
        is_hybrid_swa = (
            layer.sliding_window_size is not None and layer.sliding_window_size > -1
        )
        window_size = (layer.sliding_window_size, 0) if is_hybrid_swa else (-1, -1)

        # currently no FP8 KV cache supported
        k_descale, v_descale = None, None
        # # only use kv scaling if: 1) fp8 kv is explicitly enabled, 2) RadixAttention
        # # has corresponding quantization method so that layer.k_scale is not None,
        # # 3) layer.head_dim <= 256 since fa3 kernel require fp16 and bf16 data type in this case.
        # if self.kv_cache_dtype_str != "auto" and layer.head_dim <= 256:
        #     if layer.k_scale is not None:
        #         descale_shape = (forward_batch.batch_size, layer.tp_k_head_num)
        #         k_descale = layer.k_scale.expand(descale_shape)
        #         v_descale = layer.v_scale.expand(descale_shape)
        #     q = q.to(self.kv_cache_dtype)
        #     q_rope = q_rope.to(self.kv_cache_dtype) if q_rope is not None else None
        #     k_rope = k_rope.to(self.kv_cache_dtype) if k_rope is not None else None
        causal = not layer.is_cross_attention

        # Check if we should use local attention
        use_local_attn = (
            self.attention_chunk_size is not None
            and metadata.local_attn_metadata is not None
            and (hasattr(layer, "use_irope") and layer.use_irope)
        )

        # We do cascade attention for Target Verify with topk > 1
        # We don't use cascade attention for Sliding Window Attention:
        # - Different window sizes should be passed in for each q in the first stage of cascade attention, but FA3 interface doesn't support pass in a list of window sizes.
        # - The overhead of duplicated computation of the common prefix part is small for sliding window layers (seq_len <= window_size), so we can just expand it.
        use_cascade_attn = (
            forward_batch.forward_mode.is_target_verify()
            and self.topk > 1
            and not is_hybrid_swa
        )

        # For fa3 interface version compatibility, we put new fields into conditional keyword args
        kwargs = {}
        if sinks is not None:
            kwargs["sinks"] = sinks

        # Get the appropriate page table based on whether we're using local attention
        if use_local_attn:
            local_metadata = metadata.local_attn_metadata
            page_table = local_metadata.local_block_table
            cu_seqlens_q = local_metadata.local_query_start_loc
            cache_seqlens = local_metadata.local_seqused_k
            max_seqlen_q = local_metadata.local_max_query_len
        elif is_hybrid_swa and metadata.swa_spec_metadata is not None:
            swa_spec_metadata = metadata.swa_spec_metadata
            page_table = swa_spec_metadata.page_table
            cu_seqlens_q = swa_spec_metadata.cu_seqlens_q
            cache_seqlens = swa_spec_metadata.cache_seqlens_int32
            max_seqlen_q = swa_spec_metadata.max_seq_len_q
            cu_seqlens_k = swa_spec_metadata.cu_seqlens_k
        else:
            page_table = metadata.page_table
            if is_hybrid_swa and self.use_sliding_window_kv_pool:
                if metadata.swa_page_table is not None:
                    page_table = metadata.swa_page_table
                else:
                    page_table = self.token_to_kv_pool.translate_loc_from_full_to_swa(
                        metadata.page_table
                    ).to(torch.int32)
            cu_seqlens_q = metadata.cu_seqlens_q
            cache_seqlens = metadata.cache_seqlens_int32
            max_seqlen_q = metadata.max_seq_len_q
            cu_seqlens_k = metadata.cu_seqlens_k

        # Use Flash Attention for prefill
        if not self.use_mla:
            # Do multi-head attention
            key_cache, value_cache = self.token_to_kv_pool.get_kv_buffer(layer.layer_id)
            key_cache = key_cache.view(
                -1, self.page_size, layer.tp_k_head_num, layer.head_dim
            )
            value_cache = value_cache.view(
                -1, self.page_size, layer.tp_v_head_num, layer.head_dim
            )
            if layer.is_cross_attention:
                page_table = metadata.encoder_page_table
                cache_seqlens = metadata.encoder_lens_int32
                cu_seqlens_k = metadata.encoder_cu_seqlens_k
                window_size = (-1, -1)

            result = flash_attn_with_kvcache(
                q=q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim),
                k_cache=key_cache,
                v_cache=value_cache,
                page_table=page_table,
                cache_seqlens=cache_seqlens,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k_new=None,
                max_seqlen_q=max_seqlen_q,
                softmax_scale=layer.scaling,
                causal=False if use_cascade_attn else causal,
                window_size=window_size,
                softcap=layer.logit_cap,
                k_descale=k_descale,
                v_descale=v_descale,
                return_softmax_lse=use_cascade_attn,
                out=(
                    forward_batch._attn_output.view(
                        -1, layer.tp_q_head_num, layer.v_head_dim
                    )
                    if not use_cascade_attn
                    and getattr(forward_batch, "_attn_output", None) is not None
                    else None
                ),
                **kwargs,
            )

            if use_cascade_attn:
                o, softmax_lse, *rest = result
                o_expand, softmax_lse_expand, *rest_expand = flash_attn_with_kvcache(
                    q=q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim),
                    k_cache=key_cache,
                    v_cache=value_cache,
                    page_table=self.forward_metadata_spec_decode_expand.page_table,
                    cache_seqlens=self.forward_metadata_spec_decode_expand.cache_seqlens_int32,
                    cu_seqlens_q=self.forward_metadata_spec_decode_expand.cu_seqlens_q,
                    cu_seqlens_k_new=None,
                    max_seqlen_q=self.forward_metadata_spec_decode_expand.max_seq_len_q,
                    softmax_scale=layer.scaling,
                    causal=False,
                    window_size=window_size,
                    softcap=layer.logit_cap,
                    k_descale=k_descale,
                    v_descale=v_descale,
                    return_softmax_lse=True,
                    **kwargs,
                )
                o, _ = merge_state_v2_wrapper(
                    o,
                    softmax_lse.T.contiguous(),
                    o_expand,
                    softmax_lse_expand.T.contiguous(),
                )
            else:
                o = result
        else:
            if (
                forward_batch.attn_attend_prefix_cache is not None
                and not forward_batch.forward_mode.is_target_verify()
            ):
                # Do multi-head attention with chunked prefix cache
                if forward_batch.attn_attend_prefix_cache:
                    assert not get_global_server_args().disable_chunked_prefix_cache
                    # MHA for chunked prefix kv cache when running model with MLA
                    assert forward_batch.prefix_chunk_idx is not None
                    assert forward_batch.prefix_chunk_cu_seq_lens is not None
                    assert forward_batch.prefix_chunk_max_seq_lens is not None

                    chunk_idx = forward_batch.prefix_chunk_idx
                    assert chunk_idx >= 0

                    assert forward_batch.mha_return_lse
                    output = flash_attn_varlen_func(
                        q=q.view(-1, layer.tp_q_head_num, layer.head_dim),
                        k=k.view(-1, layer.tp_k_head_num, layer.head_dim).to(q.dtype),
                        v=v.view(-1, layer.tp_k_head_num, layer.v_head_dim).to(q.dtype),
                        cu_seqlens_q=metadata.cu_seqlens_q,
                        cu_seqlens_k=forward_batch.prefix_chunk_cu_seq_lens[chunk_idx],
                        max_seqlen_q=metadata.max_seq_len_q,
                        max_seqlen_k=forward_batch.prefix_chunk_max_seq_lens[chunk_idx],
                        softmax_scale=layer.scaling,
                        causal=False,
                        return_softmax_lse=True,
                    )
                else:
                    # MHA for extend part of sequence without attending prefix kv cache
                    output = flash_attn_varlen_func(
                        q=q.view(-1, layer.tp_q_head_num, layer.head_dim),
                        k=k.view(-1, layer.tp_k_head_num, layer.head_dim).to(q.dtype),
                        v=v.view(-1, layer.tp_k_head_num, layer.v_head_dim).to(q.dtype),
                        cu_seqlens_q=metadata.cu_seqlens_q,
                        cu_seqlens_k=metadata.cu_seqlens_q,
                        max_seqlen_q=metadata.max_seq_len_q,
                        max_seqlen_k=metadata.max_seq_len_q,
                        softmax_scale=layer.scaling,
                        causal=True,
                        return_softmax_lse=forward_batch.mha_return_lse,
                    )
                if forward_batch.mha_return_lse:
                    output, lse, *rest = output
                    lse = torch.transpose(lse, 0, 1).contiguous()
                    return output, lse
                return output
            else:
                # Do absorbed multi-latent attention
                kv_cache = self.token_to_kv_pool.get_key_buffer(layer.layer_id).to(
                    q.dtype
                )
                k_rope = kv_cache[:, :, layer.v_head_dim :]
                c_kv = kv_cache[:, :, : layer.v_head_dim]
                k_rope_cache = k_rope.view(
                    -1,
                    self.page_size,
                    layer.tp_k_head_num,
                    layer.head_dim - layer.v_head_dim,
                )
                c_kv_cache = c_kv.view(
                    -1, self.page_size, layer.tp_v_head_num, layer.v_head_dim
                )
                if q_rope is not None:
                    q_nope = q.view(-1, layer.tp_q_head_num, layer.v_head_dim)
                    q_rope = q_rope.view(
                        -1, layer.tp_q_head_num, layer.head_dim - layer.v_head_dim
                    )
                else:
                    q_all = q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim)
                    q_nope = q_all[:, :, : layer.v_head_dim]
                    q_rope = q_all[:, :, layer.v_head_dim :]

                result = flash_attn_with_kvcache(
                    q=q_rope,
                    k_cache=k_rope_cache,
                    v_cache=c_kv_cache,
                    qv=q_nope,
                    page_table=page_table,
                    cache_seqlens=cache_seqlens,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_k_new=None,
                    max_seqlen_q=max_seqlen_q,
                    softmax_scale=layer.scaling,
                    causal=False if use_cascade_attn else causal,
                    softcap=layer.logit_cap,
                    k_descale=k_descale,
                    v_descale=v_descale,
                    return_softmax_lse=use_cascade_attn,
                )
                if use_cascade_attn:
                    o, softmax_lse, *rest = result
                    o_expand, softmax_lse_expand, *rest_expand = (
                        flash_attn_with_kvcache(
                            q=q_rope,
                            k_cache=k_rope_cache,
                            v_cache=c_kv_cache,
                            qv=q_nope,
                            page_table=self.forward_metadata_spec_decode_expand.page_table,
                            cache_seqlens=self.forward_metadata_spec_decode_expand.cache_seqlens_int32,
                            cu_seqlens_q=self.forward_metadata_spec_decode_expand.cu_seqlens_q,
                            cu_seqlens_k_new=None,
                            max_seqlen_q=self.forward_metadata_spec_decode_expand.max_seq_len_q,
                            softmax_scale=layer.scaling,
                            causal=False,
                            window_size=window_size,
                            softcap=layer.logit_cap,
                            k_descale=k_descale,
                            v_descale=v_descale,
                            return_softmax_lse=True,
                        )
                    )
                    o, _ = merge_state_v2_wrapper(
                        o,
                        softmax_lse.T.contiguous(),
                        o_expand,
                        softmax_lse_expand.T.contiguous(),
                    )
                else:
                    o = result

        out = o.view(-1, layer.tp_q_head_num * layer.v_head_dim)
        return out

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
        # For multi-head latent attention
        q_rope: Optional[torch.Tensor] = None,
        k_rope: Optional[torch.Tensor] = None,
        sinks: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if k is None and v is None:
            # Cross-layer KV sharing (Gemma 4): see forward_extend for details.
            pass
        elif k is None or v is None:
            raise ValueError("Both k and v should be None or not None")
        else:
            if save_kv_cache:
                cache_loc = (
                    forward_batch.out_cache_loc
                    if not layer.is_cross_attention
                    else forward_batch.encoder_out_cache_loc
                )
                if not self.use_mla:
                    self.token_to_kv_pool.set_kv_buffer(
                        layer,
                        KVWriteLoc(
                            cache_loc,
                            self.forward_metadata.swa_out_cache_loc,
                        ),
                        k,
                        v,
                        layer.k_scale,
                        layer.v_scale,
                    )
                else:
                    k_rope_val = (
                        k_rope if k_rope is not None else k[:, :, layer.v_head_dim :]
                    )
                    self.token_to_kv_pool.set_mla_kv_buffer(
                        layer,
                        cache_loc,
                        k,
                        k_rope_val,
                    )

        # Use precomputed metadata across all layers
        metadata = self.forward_metadata
        local_attn_metadata = getattr(metadata, "local_attn_metadata", None)
        use_local_attn = (
            self.attention_chunk_size is not None
            and local_attn_metadata is not None
            and (hasattr(layer, "use_irope") and layer.use_irope)
        )

        # When Spec Decode enabled, forward_decode would be called with two mode:
        # 1. DRAFT_DECODE: we enable cascade attention when top_k > 1
        # 2. IDLE: we don’t need cascade attention, spec_info will be none in this case
        use_cascade_attn = forward_batch.spec_info is not None and self.topk > 1

        # Calculate window size (can be moved to metadata if layer properties don't change)
        # we don't do layer.sliding_window_size - 1 since in model.get_attention_sliding_window_size() we already - 1
        # here is two side inclusive
        window_size = (
            (layer.sliding_window_size, 0)
            if layer.sliding_window_size is not None and layer.sliding_window_size > -1
            else (-1, -1)
        )
        causal = not layer.is_cross_attention

        # For fa3 interface version compatibility, we put new fields into conditional keyword args
        kwargs = {}
        if sinks is not None:
            kwargs["sinks"] = sinks

        k_descale, v_descale = None, None
        # only use kv scaling if: 1) fp8 kv is explicitly enabled, 2) RadixAttention
        # has corresponding quantization method so that layer.k_scale is not None,
        # 3) layer.head_dim <= 256 since fa3 kernel require fp16 and bf16 data type in this case.
        if self.kv_cache_dtype_str != "auto" and layer.head_dim <= 256:
            if layer.k_scale is not None:
                descale_shape = (forward_batch.batch_size, layer.tp_k_head_num)
                k_descale = layer.k_scale.expand(descale_shape)
                v_descale = layer.v_scale.expand(descale_shape)
            q = q.to(self.kv_cache_dtype)
            q_rope = q_rope.to(self.kv_cache_dtype) if q_rope is not None else None
            k_rope = k_rope.to(self.kv_cache_dtype) if k_rope is not None else None
        if not self.use_mla:
            # Do multi-head attention

            key_cache, value_cache = self.token_to_kv_pool.get_kv_buffer(layer.layer_id)
            key_cache = key_cache.view(
                -1, self.page_size, layer.tp_k_head_num, layer.head_dim
            )
            value_cache = value_cache.view(
                -1, self.page_size, layer.tp_v_head_num, layer.head_dim
            )

            if layer.is_cross_attention:
                # Always use non-chunked logic for cross-attention
                o = flash_attn_with_kvcache(
                    q=q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim),
                    k_cache=key_cache,
                    v_cache=value_cache,
                    page_table=metadata.encoder_page_table,
                    cache_seqlens=metadata.encoder_lens_int32,
                    cu_seqlens_q=metadata.cu_seqlens_q,
                    cu_seqlens_k_new=None,
                    max_seqlen_q=1,
                    softmax_scale=layer.scaling,
                    causal=False,
                    window_size=(-1, -1),
                    softcap=layer.logit_cap,
                    k_descale=k_descale,
                    v_descale=v_descale,
                    **kwargs,
                )
            elif use_local_attn:
                # Use chunked (local) attention batching for self-attention
                o = flash_attn_with_kvcache(
                    q=q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim),
                    k_cache=key_cache,
                    v_cache=value_cache,
                    page_table=local_attn_metadata.local_block_table,
                    cache_seqlens=local_attn_metadata.local_seqused_k,
                    cu_seqlens_q=local_attn_metadata.local_query_start_loc,
                    cu_seqlens_k_new=None,
                    max_seqlen_q=local_attn_metadata.local_max_query_len,
                    softmax_scale=layer.scaling,
                    causal=True,
                    window_size=(-1, -1),
                    softcap=layer.logit_cap,
                    k_descale=k_descale,
                    v_descale=v_descale,
                    **kwargs,
                )
            else:
                is_swa_layer = (
                    layer.sliding_window_size is not None
                    and layer.sliding_window_size > -1
                )

                page_table = metadata.page_table
                # For SWA layers on hybrid models, use the translated
                # SWA-pool page table so KV reads hit the correct pool.
                if is_swa_layer and self.use_sliding_window_kv_pool:
                    if metadata.swa_page_table is not None:
                        page_table = metadata.swa_page_table
                    else:
                        page_table = (
                            self.token_to_kv_pool.translate_loc_from_full_to_swa(
                                metadata.page_table
                            ).to(torch.int32)
                        )

                cache_seqlens = metadata.cache_seqlens_int32
                cu_seqlens_k = metadata.cu_seqlens_k
                max_seqlen_q = metadata.max_seq_len_q
                q_reshaped = q.contiguous().view(
                    -1, layer.tp_q_head_num, layer.head_dim
                )

                # Default: single-token self-attention
                result = flash_attn_with_kvcache(
                    q=q_reshaped,
                    k_cache=key_cache,
                    v_cache=value_cache,
                    page_table=page_table,
                    cache_seqlens=cache_seqlens,
                    cu_seqlens_q=metadata.cu_seqlens_q,
                    cu_seqlens_k_new=None,
                    max_seqlen_q=max_seqlen_q,
                    softmax_scale=layer.scaling,
                    causal=False if use_cascade_attn else causal,
                    window_size=window_size,
                    softcap=layer.logit_cap,
                    k_descale=k_descale,
                    v_descale=v_descale,
                    return_softmax_lse=use_cascade_attn,
                    **kwargs,
                )
                if use_cascade_attn:
                    o, softmax_lse, *rest = result
                    o_expand, softmax_lse_expand, *rest_expand = (
                        flash_attn_with_kvcache(
                            q=q_reshaped,
                            k_cache=key_cache,
                            v_cache=value_cache,
                            page_table=self.forward_metadata_spec_decode_expand.page_table,
                            cache_seqlens=self.forward_metadata_spec_decode_expand.cache_seqlens_int32,
                            cu_seqlens_q=self.forward_metadata_spec_decode_expand.cu_seqlens_q,
                            cu_seqlens_k_new=None,
                            max_seqlen_q=self.forward_metadata_spec_decode_expand.max_seq_len_q,
                            softmax_scale=layer.scaling,
                            causal=False,
                            window_size=window_size,
                            softcap=layer.logit_cap,
                            k_descale=k_descale,
                            v_descale=v_descale,
                            return_softmax_lse=True,
                            **kwargs,
                        )
                    )
                    o, _ = merge_state_v2(
                        o,
                        softmax_lse.T.contiguous(),
                        o_expand,
                        softmax_lse_expand.T.contiguous(),
                    )
                else:
                    o = result
        else:
            # Do absorbed multi-latent attention
            kv_cache = self.token_to_kv_pool.get_key_buffer(layer.layer_id).to(q.dtype)
            assert not use_cascade_attn, "Cascade attention is not supported with MLA"

            if q_rope is not None:
                q_nope = q.view(-1, layer.tp_q_head_num, layer.v_head_dim)
                q_rope = q_rope.view(
                    -1, layer.tp_q_head_num, layer.head_dim - layer.v_head_dim
                )
            else:
                q_all = q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim)
                q_nope = q_all[:, :, : layer.v_head_dim]
                q_rope = q_all[:, :, layer.v_head_dim :]

            o = flash_mla_decode(
                q_nope,
                q_rope,
                kv_cache.view(-1, self.page_size, layer.head_dim),
                metadata.cache_seqlens_int32,
                metadata.page_table,
                self.workspace,
                layer.scaling,
            )

        out = o.view(-1, layer.tp_q_head_num * layer.v_head_dim)
        return out

    def get_cuda_graph_seq_len_fill_value(self):
        """Get the fill value for sequence length in CUDA graph."""
        return 1

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        """Pre-allocate fixed-size tensors reused across XPU graph captures."""
        max_num_pages = (self.max_context_len + self.page_size - 1) // self.page_size
        self.decode_cuda_graph_metadata = {
            "cache_seqlens": torch.zeros(max_bs, dtype=torch.int32, device=self.device),
            "cu_seqlens_q": torch.arange(
                0, max_bs + 1, dtype=torch.int32, device=self.device
            ),
            "cu_seqlens_k": torch.zeros(
                max_bs + 1, dtype=torch.int32, device=self.device
            ),
            "page_table": torch.zeros(
                max_bs, max_num_pages, dtype=torch.int32, device=self.device
            ),
            "strided_indices": torch.arange(
                0, self.max_context_len, self.page_size, device=self.device
            ),
        }
        if self.use_sliding_window_kv_pool:
            self.decode_cuda_graph_metadata["swa_page_table"] = torch.zeros(
                max_bs, max_num_pages, dtype=torch.int32, device=self.device
            )
            self.decode_cuda_graph_metadata["swa_out_cache_loc"] = torch.zeros(
                max_num_tokens, dtype=torch.int64, device=self.device
            )
        if self.is_encoder_decoder:
            self.encoder_metadata = {
                "encoder_page_table": torch.zeros(
                    max_bs, self.max_context_len, dtype=torch.int32, device=self.device
                ),
                "encoder_lens_int32": torch.zeros(
                    max_bs, dtype=torch.int32, device=self.device
                ),
                "encoder_cu_seqlens_k": torch.zeros(
                    max_bs + 1, dtype=torch.int32, device=self.device
                ),
            }
        else:
            self.encoder_metadata = {}

    def init_forward_metadata_out_graph(
        self,
        forward_batch: ForwardBatch,
        in_capture: bool = False,
    ):
        """New unified graph capture/replay entry point (replaces the legacy
        init_forward_metadata_capture_cuda_graph /
        init_forward_metadata_replay_cuda_graph pair).

        Called by DecodeCudaGraphRunner:
          - capture: in_capture=True  → bind static metadata buffer slices, then fill
          - replay:  in_capture=False → update pre-allocated buffers in-place
          - eager:   via init_forward_metadata() default wrapper
        """
        bs = forward_batch.batch_size
        req_pool_indices = forward_batch.req_pool_indices
        seq_lens = forward_batch.seq_lens
        seq_lens_cpu = getattr(forward_batch, "seq_lens_cpu", None)
        forward_mode = forward_batch.forward_mode
        spec_info = forward_batch.spec_info

        assert (
            spec_info is None
        ), "XPUAttentionBackend does not support speculative decoding in XPU graph"
        assert (
            forward_mode.is_decode_or_idle()
        ), "XPUAttentionBackend XPU graph only supports decode mode"

        if in_capture:
            # Bind static-shape slices of the pre-allocated buffers so the
            # captured graph always reads from the same storage addresses.
            metadata = FlashAttentionMetadata()
            metadata.cache_seqlens_int32 = self.decode_cuda_graph_metadata[
                "cache_seqlens"
            ][:bs]
            metadata.cu_seqlens_q = self.decode_cuda_graph_metadata["cu_seqlens_q"][
                : bs + 1
            ]
            metadata.cu_seqlens_k = self.decode_cuda_graph_metadata["cu_seqlens_k"][
                : bs + 1
            ]
            metadata.page_table = self.decode_cuda_graph_metadata["page_table"][:bs, :]
            if self.use_sliding_window_kv_pool:
                # Bind SWA page table slice so the graph captures the right tensor.
                metadata.swa_page_table = self.decode_cuda_graph_metadata[
                    "swa_page_table"
                ][:bs, :]
            if self.is_encoder_decoder and forward_batch.encoder_lens is not None:
                encoder_bs = forward_batch.encoder_lens.numel()
                metadata.encoder_lens_int32 = self.encoder_metadata[
                    "encoder_lens_int32"
                ][:encoder_bs]
                metadata.encoder_cu_seqlens_k = self.encoder_metadata[
                    "encoder_cu_seqlens_k"
                ][: encoder_bs + 1]
                metadata.encoder_page_table = self.encoder_metadata[
                    "encoder_page_table"
                ][:bs, :]
            self.decode_cuda_graph_metadata[bs] = metadata

        # Both capture and replay: fill data into the pre-allocated buffers.
        seq_lens = seq_lens[:bs]
        seq_lens_cpu = seq_lens_cpu[:bs] if seq_lens_cpu is not None else None
        req_pool_indices = req_pool_indices[:bs]

        metadata = self.decode_cuda_graph_metadata[bs]
        max_len = (
            seq_lens_cpu.max().item()
            if seq_lens_cpu is not None
            else seq_lens.max().item()
        )
        metadata.max_seq_len_k = max_len

        metadata.cache_seqlens_int32.copy_(seq_lens.to(torch.int32))

        metadata.cu_seqlens_k[0] = 0
        metadata.cu_seqlens_k[1 : bs + 1].copy_(
            torch.cumsum(seq_lens.to(torch.int32), dim=0)
        )

        if self.is_encoder_decoder and forward_batch.encoder_lens is not None:
            encoder_lens = forward_batch.encoder_lens[:bs].to(torch.int32)
            metadata.encoder_max_seq_len_k = int(encoder_lens.max().item())
            metadata.encoder_lens_int32.copy_(encoder_lens)
            metadata.encoder_cu_seqlens_k[0] = 0
            metadata.encoder_cu_seqlens_k[1 : bs + 1].copy_(
                torch.cumsum(encoder_lens, dim=0, dtype=torch.int32)
            )
            metadata.encoder_page_table[:bs, : metadata.encoder_max_seq_len_k].copy_(
                self.req_to_token[
                    req_pool_indices, : metadata.encoder_max_seq_len_k
                ].to(torch.int32)
            )
            # Self-attention (text) page_table: decoder tokens start after encoder tokens.
            text_max = metadata.max_seq_len_k
            arange_text = torch.arange(text_max, device=req_pool_indices.device)
            text_col = encoder_lens[:bs].long().unsqueeze(1) + arange_text.unsqueeze(0)
            text_row = req_pool_indices.unsqueeze(1).expand(-1, text_max)
            metadata.page_table[:bs, :text_max].copy_(
                self.req_to_token[text_row, text_col].to(torch.int32)
            )
            metadata.page_table[:bs, text_max:].zero_()
        else:
            raw_page = self.req_to_token[
                req_pool_indices[:, None],
                self.decode_cuda_graph_metadata["strided_indices"][
                    : ((metadata.max_seq_len_k + self.page_size - 1) // self.page_size)
                ][None, :],
            ]
            if self.page_size > 1:
                raw_page = raw_page // self.page_size
            metadata.page_table[:bs, : raw_page.shape[1]].copy_(
                raw_page.to(torch.int32)
            )
            metadata.page_table[:bs, raw_page.shape[1] :].zero_()

        if self.use_sliding_window_kv_pool:
            if forward_batch.out_cache_loc is None:
                raise ValueError(
                    f"out_cache_loc is None for hybrid SWA model in graph "
                    f"{'capture' if in_capture else 'replay'} "
                    f"(forward_mode={forward_batch.forward_mode}). This should not happen."
                )
            swa_out_cache_loc = self.decode_cuda_graph_metadata["swa_out_cache_loc"]
            n = forward_batch.out_cache_loc.shape[0]
            swa_out_cache_loc[n:].zero_()
            swa_out_cache_loc[:n].copy_(
                self.token_to_kv_pool.translate_loc_from_full_to_swa(
                    forward_batch.out_cache_loc
                )
            )
            metadata.swa_out_cache_loc = swa_out_cache_loc[:n]

            if not (self.is_encoder_decoder and forward_batch.encoder_lens is not None):
                max_seq_pages = (
                    metadata.max_seq_len_k + self.page_size - 1
                ) // self.page_size
                swa_page_table = self.decode_cuda_graph_metadata["swa_page_table"]
                swa_page_table[:bs, max_seq_pages:].zero_()
                swa_page_table[:bs, :max_seq_pages].copy_(
                    (
                        self.token_to_kv_pool.translate_loc_from_full_to_swa(raw_page)
                        if self.page_size == 1
                        else self.token_to_kv_pool.translate_loc_from_full_to_swa(
                            self.req_to_token[
                                req_pool_indices[:, None],
                                self.decode_cuda_graph_metadata["strided_indices"][
                                    :max_seq_pages
                                ][None, :],
                            ]
                        )
                        // self.page_size
                    ).to(torch.int32)
                )
                metadata.swa_page_table = swa_page_table[:bs, :]

        self.forward_metadata = metadata

    def init_forward_metadata_in_graph(self, forward_batch: ForwardBatch):
        """Graph-recordable ops for XPU graph (no-op: all metadata setup is
        host-side and lives in init_forward_metadata_out_graph)."""

    def _init_local_attn_metadata(
        self,
        forwardbatch: ForwardBatch,
        metadata: FlashAttentionMetadata,
        device,
    ):
        """Centralized utility to initialize local_attn_metadata if chunked attention is enabled."""
        if self.attention_chunk_size is None:
            metadata.local_attn_metadata = None
            return

        cu_seqlens_q = metadata.cu_seqlens_q
        cache_seqlens_int32 = metadata.cache_seqlens_int32
        if self.is_hybrid_swa:
            page_table = self.full_to_swa_index_mapping[metadata.page_table].to(
                torch.int32
            )
        else:
            page_table = metadata.page_table
        if cu_seqlens_q is None or cache_seqlens_int32 is None or page_table is None:
            metadata.local_attn_metadata = None
            return

        # make_local_attention_virtual_batches expects a page-granularity block table:
        # column p is the logical page number, and the value stored at that column is the
        # physical page index. The raw req_to_token table is token-granularity (column i =
        # the KV slot for token i), so when page_size > 1 we must stride and divide first
        # so that block_starts = k_seqstarts_absolute // page_size correctly indexes the table.
        if self.page_size > 1:
            strided_indices = torch.arange(
                0, page_table.shape[1], self.page_size, device=page_table.device
            )
            page_table = page_table[:, strided_indices] // self.page_size

        cu_seqlens_q_np = cu_seqlens_q.cpu().numpy()
        seq_lens_np = cache_seqlens_int32.cpu().numpy()
        (
            seqlens_q_local_np,
            cu_seqlens_q_local_np,
            seqlens_k_local_np,
            block_table_local,
        ) = make_local_attention_virtual_batches(
            self.attention_chunk_size,
            cu_seqlens_q_np,
            seq_lens_np,
            page_table,
            self.page_size,
        )

        local_metadata = FlashAttentionMetadata.LocalAttentionMetadata(
            local_query_start_loc=torch.from_numpy(cu_seqlens_q_local_np).to(device),
            local_seqused_k=torch.from_numpy(seqlens_k_local_np).to(device),
            local_block_table=block_table_local.to(device),
            local_max_query_len=int(seqlens_q_local_np.max()),
            local_max_seq_len=int(seqlens_k_local_np.max()),
        )
        metadata.local_attn_metadata = local_metadata

    def _init_sliding_window_attn_spec_metadata(
        self,
        metadata: FlashAttentionMetadata,
        metadata_expand: FlashAttentionMetadata,
        metadata_swa: Optional[FlashAttentionMetadata] = None,
    ):
        # TODO: support page_size > 1 for swa spec
        assert (
            self.page_size == 1
        ), "FlashAttention backend doesn't support topk > 1 speculative decoding with page size > 1 sliding window attention"

        cache_seqlens_int32 = (
            metadata.cache_seqlens_int32.repeat_interleave(
                self.speculative_num_draft_tokens
            )
            + metadata_expand.cache_seqlens_int32
        )
        cu_seqlens_k = torch.nn.functional.pad(
            torch.cumsum(cache_seqlens_int32, dim=0, dtype=torch.int32), (1, 0)
        )
        bs = cache_seqlens_int32.shape[0]
        page_table = (
            metadata.page_table.new_zeros(
                (bs, metadata.max_seq_len_k + metadata_expand.page_table.shape[1])
            )
            if metadata_swa is None
            else metadata_swa.page_table
        )

        prepare_swa_spec_page_table_triton(
            page_table,
            metadata.page_table,
            metadata_expand.page_table,
            metadata.cache_seqlens_int32,
            metadata_expand.cache_seqlens_int32,
            self.speculative_num_draft_tokens,
        )

        if metadata_swa is None:
            metadata_swa = FlashAttentionMetadata()
            metadata_swa.max_seq_len_q = 1
            metadata_swa.cu_seqlens_q = metadata_expand.cu_seqlens_q
            metadata_swa.cache_seqlens_int32 = cache_seqlens_int32
            metadata_swa.cu_seqlens_k = cu_seqlens_k
            metadata_swa.page_table = page_table
        else:
            metadata_swa.cache_seqlens_int32.copy_(cache_seqlens_int32)
            metadata_swa.cu_seqlens_k.copy_(cu_seqlens_k)

        metadata.swa_spec_metadata = metadata_swa
